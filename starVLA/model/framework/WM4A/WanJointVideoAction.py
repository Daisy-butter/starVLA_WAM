# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
WanJointVideoAction — joint Wan video + action MoE (``WanVideoActionMoE``) inside WM4A.

WM4A **video–action** framework: composes the pretrained Wan **video stack**
(``world_model.wan_video_stack_*``) with the **joint MoE action pathway**
(``action_model.wan_video_action_moe_model``) and flow-matching training utilities.

  - ``forward(examples) -> {"action_loss": Tensor, ...}``
  - ``predict_action(examples) -> {"normalized_actions": ndarray}``

Training follows the same flow-matching objective as the reference stack: **noised
latents** and **noised actions** (``FlowMatchScheduler``), with optional **video flow** loss.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.modules.action_model.wan_flow_match_scheduler import FlowMatchScheduler
from starVLA.model.modules.action_model.wan_video_action_moe_model import WanVideoActionMoE
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.wan_video_action_moe import build_wan_video_action_moe
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

logger = initialize_overwatch(__name__)


def _action_model_timestep(action_scheduler: FlowMatchScheduler, action_timestep: torch.Tensor) -> torch.Tensor:
    """Match ``WanVideoActionTrainer._action_model_timestep`` for non-absolute sigma schedules."""
    if action_scheduler._use_absolute_sigmas():
        return action_timestep.to(dtype=torch.float32)
    timestep_ids = action_scheduler._nearest_timestep_index(action_timestep.detach().cpu())
    return timestep_ids.to(device=action_timestep.device, dtype=torch.float32)


@dataclass
class WanJointVideoActionDefaultConfig:
    name: str = "WanJointVideoAction"

    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        }
    )

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 7,
            "action_horizon": 16,
        }
    )

    # Optional overrides passed to ``WanVideoActionConfig`` (``action_model.wan_video_action_config``).
    joint_wan_video_action: dict = field(default_factory=dict)

    # Observation preprocessing (matches Wan2 defaults unless overridden).
    obs_height: int = 480
    obs_width: int = 832

    # Flow-Match schedulers (training).
    video_flow_shift: float = 5.0
    action_flow_shift: float = 5.0
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0

    # Action-only denoising steps at inference (video latents fixed to encoded obs).
    num_inference_timesteps: int = 10


@FRAMEWORK_REGISTRY.register("WanJointVideoAction")
class Wan_Joint_Video_Action(baseframework):
    """
    WM4A **video–action** framework: one checkpoint, joint optimisation.

    **Code layout (StarVLA conventions)**:
      - **Video stack** (VAE + DiT ``WanVideoModel``): ``world_model.wan_video_stack_*``
      - **Joint MoE action pathway** + ``WanVideoActionMoE``: ``action_model.wan_video_action_moe_model``
      - **Flow-match schedules** for training: ``action_model.wan_flow_match_scheduler``

    This class is the **orchestrator** (loss, ``predict_action``, YAML); it does **not**
    use ``get_world_model`` + ``get_action_model`` like ``WanGR00T`` / ``WanPI``, because
    actions attend inside Wan blocks rather than on a detached hidden-state tensor.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(WanJointVideoActionDefaultConfig, config)

        self.joint_model: WanVideoActionMoE = build_wan_video_action_moe(self.config)

        self.video_scheduler = FlowMatchScheduler(
            num_train_timesteps=1000,
            shift=float(self.config.framework.get("video_flow_shift", 5.0)),
        )
        self.action_scheduler = FlowMatchScheduler(
            num_train_timesteps=1000,
            shift=float(self.config.framework.get("action_flow_shift", 5.0)),
        )

        self.action_horizon = int(
            OmegaConf.select(self.config, "framework.action_model.action_horizon", default=None)
            or OmegaConf.select(self.config, "framework.action_model.future_action_window_size", default=16)
        )
        self.obs_height = int(self.config.framework.get("obs_height", 480))
        self.obs_width = int(self.config.framework.get("obs_width", 832))
        self.video_loss_weight = float(self.config.framework.get("video_loss_weight", 1.0))
        self.action_loss_weight = float(self.config.framework.get("action_loss_weight", 1.0))
        self.num_inference_timesteps = int(self.config.framework.get("num_inference_timesteps", 10))

    def _ensure_video_processor(self) -> None:
        self.joint_model.video_model._ensure_inference_modules(load_tokenizer=False, load_text_encoder=False)

    @torch.no_grad()
    def _images_to_video_tensor(self, batch_images: List) -> torch.Tensor:
        """List[List[PIL]] or list of sequences -> ``[B, C, T, H, W]`` in VAE dtype/device."""
        self._ensure_video_processor()
        vm = self.joint_model.video_model
        device = vm.vae.device
        dtype = vm.vae.dtype
        processor = vm.video_processor

        preprocessed = []
        frame_counts = []
        for sample_imgs in batch_images:
            if not isinstance(sample_imgs, (list, tuple)):
                sample_imgs = [sample_imgs]
            video_tensor = processor.preprocess_video(sample_imgs, height=self.obs_height, width=self.obs_width)
            video_tensor = video_tensor.to(device=device, dtype=dtype)
            preprocessed.append(video_tensor)
            frame_counts.append(int(video_tensor.shape[2]))

        target_frames = max(frame_counts)
        batch_videos = []
        for video_tensor in preprocessed:
            n = int(video_tensor.shape[2])
            if n > target_frames:
                video_tensor = video_tensor[:, :, :target_frames]
            elif n < target_frames:
                last_frame = video_tensor[:, :, -1:]
                padding = last_frame.repeat(1, 1, target_frames - n, 1, 1)
                video_tensor = torch.cat([video_tensor, padding], dim=2)
            batch_videos.append(video_tensor.squeeze(0))
        return torch.stack(batch_videos, dim=0)

    @torch.no_grad()
    def _encode_text(self, instructions: List[str]) -> torch.Tensor:
        vm = self.joint_model.video_model
        vm._ensure_inference_modules(load_tokenizer=True, load_text_encoder=True)
        prompt_embeds, _ = vm.encode_prompt(prompt=instructions, guidance_scale=1.0)
        return prompt_embeds

    @torch.no_grad()
    def _encode_obs_latents(self, batch_images: List) -> torch.Tensor:
        video = self._images_to_video_tensor(batch_images)
        return self.joint_model.video_model.encode_video(video)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        with torch.no_grad():
            clean_latents = self._encode_obs_latents(batch_images)
            text_emb = self._encode_text(instructions)

        device = clean_latents.device
        bsz = int(clean_latents.shape[0])
        model_dtype = next(self.joint_model.video_model.transformer.parameters()).dtype

        video_timestep, video_weight = self.video_scheduler.sample(
            batch_size=bsz,
            device=device,
            return_timesteps=True,
        )
        action_timestep, action_weight = self.action_scheduler.sample(
            batch_size=bsz,
            device=device,
            return_timesteps=True,
        )
        action_model_timestep = _action_model_timestep(self.action_scheduler, action_timestep)

        video_noise = torch.randn_like(clean_latents)
        future_noisy_latents = self.video_scheduler.add_noise(clean_latents, video_noise, video_timestep)
        video_target = FlowMatchScheduler.training_target(clean_latents, video_noise, video_timestep)

        actions_tensor = torch.tensor(np.array(actions), device=device, dtype=torch.float32)
        actions_chunk = actions_tensor[:, -self.action_horizon :, :]
        if int(actions_chunk.shape[-1]) != int(self.joint_model.config.action_dim):
            raise ValueError(
                f"action_dim mismatch: examples have D={int(actions_chunk.shape[-1])} but model expects "
                f"{int(self.joint_model.config.action_dim)}. Update ``framework.action_model.action_dim`` "
                "and ``framework.joint_wan_video_action`` so WanVideoActionConfig matches the dataset."
            )

        action_noise = torch.randn_like(actions_chunk)
        future_noisy_actions = self.action_scheduler.add_noise(actions_chunk, action_noise, action_timestep)
        action_target = FlowMatchScheduler.training_target(actions_chunk, action_noise, action_timestep)

        with torch.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            outputs = self.joint_model(
                future_noisy_latents=future_noisy_latents.to(dtype=model_dtype),
                future_timestep=video_timestep.to(device=device, dtype=torch.float32),
                future_noisy_actions=future_noisy_actions.to(device=device, dtype=model_dtype),
                action_timestep=action_model_timestep.to(device=device, dtype=torch.float32),
                text_emb=text_emb.to(dtype=model_dtype),
                future_latent_view_indices=None,
                robot_names=None,
                return_dict=True,
            )

        video_sqerr = (outputs.pred_video_flow.float() - video_target.float()).square()
        denom_v = float(video_sqerr.numel())
        video_loss = video_sqerr.sum() / max(denom_v, 1.0)
        video_loss = video_loss * video_weight.float().mean()

        action_sqerr = (outputs.pred_action_flow.float() - action_target.float()).square()
        denom_a = float(action_sqerr.numel())
        action_loss = action_sqerr.sum() / max(denom_a, 1.0)
        action_loss = action_loss * action_weight.float().mean()

        total = self.video_loss_weight * video_loss + self.action_loss_weight * action_loss
        return {
            "action_loss": total,
            "joint_video_loss": video_loss.detach(),
            "joint_action_flow_loss": action_loss.detach(),
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        with torch.no_grad():
            clean_latents = self._encode_obs_latents(batch_images)
            text_emb = self._encode_text(instructions)

        device = clean_latents.device
        bsz = int(clean_latents.shape[0])
        model_dtype = next(self.joint_model.video_model.transformer.parameters()).dtype

        action_infer = self.action_scheduler.build_inference_scheduler(
            device=device,
            num_steps=max(1, int(self.num_inference_timesteps)),
        )
        action_latents = torch.randn(
            (bsz, self.action_horizon, int(self.joint_model.config.action_dim)),
            device=device,
            dtype=torch.float32,
        )
        action_latents = action_latents * float(action_infer.sigmas[0].item())

        video_timestep_eval = torch.zeros((bsz,), device=device, dtype=torch.float32)

        action_dtype = self.joint_model.action_out_proj.weight.dtype
        for _, action_ts in enumerate(action_infer.timesteps):
            action_ts_b = action_ts.to(device=device, dtype=torch.float32).expand(bsz)
            action_model_ts = _action_model_timestep(self.action_scheduler, action_ts_b)

            outputs = self.joint_model(
                future_noisy_latents=clean_latents.to(dtype=model_dtype),
                future_timestep=video_timestep_eval.to(dtype=torch.float32),
                future_noisy_actions=action_latents.to(dtype=action_dtype),
                action_timestep=action_model_ts.to(dtype=torch.float32),
                text_emb=text_emb.to(dtype=model_dtype),
                future_latent_view_indices=None,
                robot_names=None,
                return_dict=True,
            )
            action_latents = action_infer.step(
                outputs.pred_action_flow.to(dtype=torch.float32),
                action_ts,
                action_latents,
                return_dict=False,
            )[0]

        normalized_actions = action_latents.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf
    from PIL import Image

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
    )
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.name = "WanJointVideoAction"
    cfg.framework.world_model = {
        "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    }
    cfg.framework.action_model = {"action_dim": 7, "action_horizon": 8}
    cfg.framework.video_loss_weight = 0.0

    model = Wan_Joint_Video_Action(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float32),
        "image": [image, image],
        "lang": "pick up the cube",
    }
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    out = model([sample, sample])
    print({k: float(v) if torch.is_tensor(v) else v for k, v in out.items()})

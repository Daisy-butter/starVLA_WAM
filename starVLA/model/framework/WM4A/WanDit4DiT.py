# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
WanDit4DiT Framework — DiT4DiT-style 2(b) Hidden-State WAM on Wan2.2-TI2V.

Architecture (following DiT4DiT, arXiv:2603.10448):
  - Video DiT (Wan2.2): joint flow-matching on future observation latents (L_video)
  - Hidden states extracted at fixed τ_f feed the Action DiT via cross-attention
  - Action DiT: flow-matching head (GR00T-style) with L_action
  - Training: L = L_action + λ * L_video
  - Inference: single τ_f forward on current obs (no future video generation)
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class WanDit4DiTDefaultConfig:
    """WanDit4DiT default parameters."""

    name: str = "WanDit4DiT"

    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "extract_layers": [-1],
            "min_pixel_frames": 5,
            "feature_extract_tau": 0.5,
            "num_timestep_buckets": 1000,
            "freeze_vae": True,
            "freeze_text_encoder": True,
        }
    )

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "vl_hidden_dim": 3072,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "future_action_window_size": 7,
            "action_horizon": 8,
            "past_action_window_size": 0,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
                "cross_attention_dim": 512,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    video_loss_weight: float = 0.1


@FRAMEWORK_REGISTRY.register("WanDit4DiT")
class Wan_Dit4DiT(baseframework):
    """
    DiT4DiT-style Video-Action Model with Wan2.2-TI2V backbone.

    Requires training samples with both `image` (current) and `next_image` (future).
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(WanDit4DiTDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)

        wm_hidden = self.backbone.model.config.hidden_size
        cross_attn_dim = self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim
        self.wm_projector = torch.nn.Linear(wm_hidden, cross_attn_dim)

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        wm_cfg = self.config.framework.get("world_model", {})
        self.video_loss_weight = float(
            self.config.framework.get(
                "video_loss_weight",
                wm_cfg.get("video_loss_weight", WanDit4DiTDefaultConfig.video_loss_weight),
            )
        )

    def _project_hidden(self, wm_outputs) -> torch.Tensor:
        last_hidden = wm_outputs.hidden_states[-1]
        return self.wm_projector(last_hidden)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        if "next_image" not in examples[0]:
            raise KeyError(
                "WanDit4DiT requires `next_image` in training samples. "
                "Enable `libero_franka_dit4dit` data config with future_video modality."
            )

        batch_future_images = [example["next_image"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone.forward_joint(
                cond_images=batch_images,
                future_images=batch_future_images,
                instructions=instructions,
            )
            last_hidden = self._project_hidden(wm_outputs)
            video_loss = wm_outputs.video_loss

        with torch.autocast("cuda", dtype=torch.float32):
            actions_tensor = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            actions_target = actions_tensor[:, -self.action_horizon :, :]

            repeated_diffusion_steps = int(
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state_tensor.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated)

        return {
            "action_loss": action_loss,
            "video_loss": video_loss,
            "total_loss": action_loss + self.video_loss_weight * video_loss,
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone.forward_features_for_action(
                cond_images=batch_images,
                instructions=instructions,
            )
            last_hidden = self._project_hidden(wm_outputs)

        state_tensor = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state_tensor)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

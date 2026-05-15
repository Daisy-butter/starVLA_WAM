# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Factory helpers for the Wan **joint** video + action MoE (``WanVideoActionMoE``).

- **Video stack** (VAE + DiT): ``starVLA.model.modules.world_model.wan_video_stack_*``
- **Joint action pathway** (MoE + config): ``wan_video_action_moe_model`` / ``wan_video_action_config``

WM4A frameworks (e.g. ``WanJointVideoAction``) import ``build_wan_video_action_moe`` here;
YAML uses ``framework.world_model`` / ``framework.action_model`` / ``framework.joint_wan_video_action``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.modules.action_model.wan_flow_match_scheduler import FlowMatchScheduler
from starVLA.model.modules.action_model.wan_video_action_config import WanVideoActionConfig
from starVLA.model.modules.action_model.wan_video_action_moe_model import WanVideoActionMoE

logger = logging.getLogger(__name__)

# Keys in HF ``config.json`` that are not ``WanVideoActionConfig`` constructor args.
_HF_CONFIG_SKIP_KEYS = frozenset(
    {
        "architectures",
        "auto_map",
        "dtype",
        "model_type",
        "transformers_version",
        "action_layout",
        "batched_grouped_cross_view_attention",
        "wan_model_variant",
        "wan_transformer_num_layers_override",
        "wan_vae_adaptation_mode",
        "wan_vae_root",
        "wan_vae_variant",
        "conditioning_video_num_frames",
    }
)

DEFAULT_HF_CKPT_DIR = "/SSD_DISK/users/wuruihan/sii_starvla_ckpt/ckpt_debug_libero"
DEFAULT_WAN_ROOT = "/SSD_DISK/users/wuruihan/sii_starvla_ckpt/hugg_model/Wan2.2-TI2V-5B-Diffusers"


def _select(cfg: Any, key: str, default: Any = None) -> Any:
    return OmegaConf.select(cfg, key, default=default)


def build_wan_video_action_config(cfg: Any) -> WanVideoActionConfig:
    """
    Build a ``WanVideoActionConfig`` from a starVLA OmegaConf ``cfg``.

    Reads:
      - ``framework.world_model.base_wm`` (fallback: ``framework.qwenvl.base_vlm``)
      - ``framework.action_model.{action_dim, action_horizon, ...}``
      - ``framework.joint_wan_video_action.*`` optional overrides for MoE / video semantics
    """
    base_wm = _select(cfg, "framework.world_model.base_wm") or _select(cfg, "framework.qwenvl.base_vlm")
    if not base_wm:
        raise ValueError(
            "Wan joint model needs ``framework.world_model.base_wm`` or ``framework.qwenvl.base_vlm`` "
            "pointing to a Wan diffusers checkpoint root."
        )

    action_dim = int(_select(cfg, "framework.action_model.action_dim", default=7) or 7)
    action_horizon = int(
        _select(cfg, "framework.action_model.action_horizon", default=None)
        or _select(cfg, "framework.action_model.future_action_window_size", default=16)
        or 16
    )

    j = _select(cfg, "framework.joint_wan_video_action", default={}) or {}

    kwargs: dict[str, Any] = {
        "wan_model_name_or_path": str(base_wm),
        "load_wan_pretrained": bool(_select(j, "load_wan_pretrained", default=True)),
        "freeze_vae": bool(_select(j, "freeze_vae", default=True)),
        "sample_posterior": bool(_select(j, "sample_posterior", default=True)),
        "compile_vae_encode": bool(_select(j, "compile_vae_encode", default=False)),
        "transformer_gradient_checkpointing": bool(_select(j, "transformer_gradient_checkpointing", default=False)),
        "video_num_frames": int(_select(j, "video_num_frames", default=21)),
        "sparse_history_video_num_frames": int(_select(j, "sparse_history_video_num_frames", default=0)),
        "future_video_num_frames": int(_select(j, "future_video_num_frames", default=20)),
        "video_fps": float(_select(j, "video_fps", default=10.0)),
        "use_view_embedding": bool(_select(j, "use_view_embedding", default=True)),
        "concat_view_embedding": bool(_select(j, "concat_view_embedding", default=False)),
        "view_embedding_dim": int(_select(j, "view_embedding_dim", default=16)),
        "max_view_embeddings": int(_select(j, "max_view_embeddings", default=8)),
        "use_cross_view_attention": bool(_select(j, "use_cross_view_attention", default=False)),
        "text_embed_dim": int(_select(j, "text_embed_dim", default=4096)),
        "sigma_min": float(_select(j, "sigma_min", default=0.0)),
        "sigma_max": float(_select(j, "sigma_max", default=1.0)),
        "video_snr_shift": float(_select(j, "video_snr_shift", default=5.0)),
        "action_dim": action_dim,
        "future_action_steps": action_horizon,
        "history_action_steps": int(_select(j, "history_action_steps", default=0)),
        "action_hidden_size": int(_select(j, "action_hidden_size", default=0) or 0),
        "action_num_heads": int(_select(j, "action_num_heads", default=16)),
        "action_num_layers": int(_select(j, "action_num_layers", default=0) or 0),
        "action_mlp_ratio": float(_select(j, "action_mlp_ratio", default=4.0)),
        "action_modulation_rank": int(_select(j, "action_modulation_rank", default=0)),
        "robot_embed_enabled": bool(_select(j, "robot_embed_enabled", default=False)),
        "action_pos_emb_max_t": int(_select(j, "action_pos_emb_max_t", default=max(128, action_horizon))),
        "action_noise_multiplier": float(_select(j, "action_noise_multiplier", default=1.0)),
        "action_snr_shift": float(_select(j, "action_snr_shift", default=5.0)),
        "action_sigma_min": float(_select(j, "action_sigma_min", default=0.0)),
        "action_sigma_max": float(_select(j, "action_sigma_max", default=1.0)),
        "action_train_sigma_min": float(_select(j, "action_train_sigma_min", default=0.0)),
        "action_train_sigma_max": float(_select(j, "action_train_sigma_max", default=1.0)),
        "action_sampling_distribution": str(_select(j, "action_sampling_distribution", default="uniform_index")),
        "action_hybrid_uniform_ratio": float(_select(j, "action_hybrid_uniform_ratio", default=0.3)),
        "action_hybrid_uniform_lower": float(_select(j, "action_hybrid_uniform_lower", default=1.0)),
        "action_hybrid_uniform_upper": float(_select(j, "action_hybrid_uniform_upper", default=85.0)),
        "action_lognormal_mean": float(_select(j, "action_lognormal_mean", default=1.39)),
        "action_lognormal_std": float(_select(j, "action_lognormal_std", default=1.2)),
        "true_shared_base_sigma": bool(_select(j, "true_shared_base_sigma", default=False)),
        "video_noise_multiplier": float(_select(j, "video_noise_multiplier", default=1.0)),
        "video_high_sigma_ratio": float(_select(j, "video_high_sigma_ratio", default=0.0)),
        "sparse_history_fps": float(_select(j, "sparse_history_fps", default=1.0)),
        "sparse_history_simulate_short_history_prob": float(
            _select(j, "sparse_history_simulate_short_history_prob", default=0.05)
        ),
    }
    return WanVideoActionConfig(**kwargs)


def build_wan_video_action_moe(cfg: Any) -> WanVideoActionMoE:
    """Instantiate ``WanVideoActionMoE`` from starVLA config (loads Wan weights from disk)."""
    return WanVideoActionMoE(build_wan_video_action_config(cfg))


def _action_model_timestep(action_scheduler: FlowMatchScheduler, action_timestep: torch.Tensor) -> torch.Tensor:
    if action_scheduler._use_absolute_sigmas():
        return action_timestep.to(dtype=torch.float32)
    timestep_ids = action_scheduler._nearest_timestep_index(action_timestep.detach().cpu())
    return timestep_ids.to(device=action_timestep.device, dtype=torch.float32)


def resolve_wan_pretrained_root(
    wan_model_name_or_path: str | None,
    *,
    hf_dir: Path,
    wan_root_override: str | None = None,
) -> str:
    """Resolve Wan diffusers root from explicit override, absolute path, or ``hf_dir`` parent."""
    if wan_root_override:
        root = Path(wan_root_override).expanduser().resolve()
        if not (root / "transformer").is_dir() or not (root / "vae").is_dir():
            raise FileNotFoundError(
                f"wan_root_override is not a Wan diffusers root (missing transformer/vae): {root}"
            )
        return str(root)

    if not wan_model_name_or_path:
        raise ValueError("wan_model_name_or_path is missing from HF config and wan_root was not provided.")

    raw = Path(str(wan_model_name_or_path)).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((hf_dir / raw).resolve())
        candidates.append((hf_dir.parent / raw).resolve())

    for candidate in candidates:
        if (candidate / "transformer").is_dir() and (candidate / "vae").is_dir():
            return str(candidate)

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Cannot resolve Wan diffusers root from {wan_model_name_or_path!r}. Tried: {tried}. "
        f"Pass wan_root={DEFAULT_WAN_ROOT!r} explicitly."
    )


def wan_video_action_config_from_hf_json(
    hf_dir: str | Path,
    *,
    wan_root: str | None = None,
) -> WanVideoActionConfig:
    """Build ``WanVideoActionConfig`` from an HF-style checkpoint directory."""
    hf_dir = Path(hf_dir).expanduser().resolve()
    config_path = hf_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json under HF checkpoint dir: {hf_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    resolved_wan = resolve_wan_pretrained_root(
        raw.get("wan_model_name_or_path"),
        hf_dir=hf_dir,
        wan_root_override=wan_root,
    )
    raw["wan_model_name_or_path"] = resolved_wan

    # Eval does not need activation checkpointing inside the video transformer.
    raw["transformer_gradient_checkpointing"] = False

    kwargs = {k: v for k, v in raw.items() if k not in _HF_CONFIG_SKIP_KEYS}
    return WanVideoActionConfig(**kwargs)


def load_wan_video_action_moe_from_hf_dir(
    hf_dir: str | Path,
    *,
    wan_root: str | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> WanVideoActionMoE:
    """
    Load a fine-tuned ``WanVideoActionMoE`` from an HF checkpoint directory.

    Expects:
      - ``config.json``
      - ``model.safetensors.index.json`` + ``model-*.safetensors`` shards

    The Wan **base** weights path is taken from ``config.json``'s
    ``wan_model_name_or_path`` (resolved relative to the checkpoint parent) unless
    ``wan_root`` is set explicitly.
    """
    hf_dir = Path(hf_dir).expanduser().resolve()
    config = wan_video_action_config_from_hf_json(hf_dir, wan_root=wan_root)

    logger.info("Building WanVideoActionMoE with Wan root: %s", config.wan_model_name_or_path)
    model = WanVideoActionMoE(config)

    index_path = hf_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing model.safetensors.index.json under {hf_dir}")

    from safetensors.torch import load_file

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    shard_names = sorted(set(weight_map.values()))
    if not shard_names:
        raise ValueError(f"Empty weight_map in {index_path}")

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard_path = hf_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing weight shard: {shard_path}")
        state_dict.update(load_file(str(shard_path)))

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"State dict mismatch when loading {hf_dir}. "
            f"missing={len(missing)} unexpected={len(unexpected)}. "
            f"First missing: {missing[:5]}. First unexpected: {unexpected[:5]}."
        )
    if missing:
        logger.warning("load_wan_video_action_moe_from_hf_dir: missing keys (%d): %s", len(missing), missing[:8])
    if unexpected:
        logger.warning(
            "load_wan_video_action_moe_from_hf_dir: unexpected keys (%d): %s",
            len(unexpected),
            unexpected[:8],
        )

    target = torch.device(device)
    model = model.to(target).eval()
    logger.info(
        "Loaded WanVideoActionMoE from %s (%d tensors, action_horizon=%d, action_dim=%d)",
        hf_dir,
        len(state_dict),
        int(config.future_action_steps),
        int(config.action_dim),
    )
    return model


class WanJointMoEPolicy:
    """Lightweight inference wrapper around ``WanVideoActionMoE`` (no starVLA YAML)."""

    def __init__(
        self,
        joint_model: WanVideoActionMoE,
        *,
        obs_height: int = 480,
        obs_width: int = 832,
        num_inference_timesteps: int = 10,
        action_flow_shift: float | None = None,
    ) -> None:
        self.joint_model = joint_model
        self.obs_height = int(obs_height)
        self.obs_width = int(obs_width)
        self.num_inference_timesteps = max(1, int(num_inference_timesteps))
        shift = float(action_flow_shift if action_flow_shift is not None else joint_model.config.action_snr_shift)
        self.action_scheduler = FlowMatchScheduler(num_train_timesteps=1000, shift=shift)
        self.action_horizon = int(joint_model.config.future_action_steps)
        self.action_dim = int(joint_model.config.action_dim)

    @classmethod
    def from_hf_dir(
        cls,
        hf_dir: str | Path,
        *,
        wan_root: str | None = None,
        device: str | torch.device = "cuda",
        obs_height: int = 480,
        obs_width: int = 832,
        num_inference_timesteps: int = 10,
        strict: bool = True,
    ) -> "WanJointMoEPolicy":
        model = load_wan_video_action_moe_from_hf_dir(
            hf_dir,
            wan_root=wan_root,
            device=device,
            strict=strict,
        )
        return cls(
            model,
            obs_height=obs_height,
            obs_width=obs_width,
            num_inference_timesteps=num_inference_timesteps,
        )

    def _ensure_video_processor(self) -> None:
        self.joint_model.video_model._ensure_inference_modules(load_tokenizer=False, load_text_encoder=False)

    @torch.no_grad()
    def _images_to_video_tensor(self, batch_images: List) -> torch.Tensor:
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

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict[str, np.ndarray]:
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        clean_latents = self._encode_obs_latents(batch_images)
        text_emb = self._encode_text(instructions)

        device = clean_latents.device
        bsz = int(clean_latents.shape[0])
        model_dtype = next(self.joint_model.video_model.transformer.parameters()).dtype

        action_infer = self.action_scheduler.build_inference_scheduler(
            device=device,
            num_steps=self.num_inference_timesteps,
        )
        action_latents = torch.randn(
            (bsz, self.action_horizon, self.action_dim),
            device=device,
            dtype=torch.float32,
        )
        action_latents = action_latents * float(action_infer.sigmas[0].item())
        video_timestep_eval = torch.zeros((bsz,), device=device, dtype=torch.float32)
        action_dtype = self.joint_model.action_out_proj.weight.dtype

        with torch.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            for action_ts in action_infer.timesteps:
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

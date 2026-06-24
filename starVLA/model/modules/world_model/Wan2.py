# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Wan2.2-TI2V World Model Interface.

Wraps Wan-AI/Wan2.2-TI2V-5B-Diffusers (diffusion-based Text+Image-to-Video model)
as a world-model backend for starVLA action prediction frameworks.

Architecture (diffusers format):
  - UMT5EncoderModel: text instruction → text embeddings [B, L_text, 4096]
  - AutoencoderKLWan (VAE): observation image → video latents [B, 48, T, H/16, W/16]
  - WanTransformer3DModel: 30-layer DiT, hidden_dim=3072 (24 heads × 128 dim)
    Takes noised latents + text embeddings → denoised latents
    We extract intermediate hidden states for action-conditioning.

Note: The diffusers version of Wan2.2-TI2V-5B uses WanPipeline (text-only
conditioning) with expand_timesteps=True for TI2V mode. There is NO CLIP
image_encoder in this model variant — image conditioning is achieved through
per-token timestep expansion where the first frame's latent is conditioned
via timestep=0 (clean).

Key differences from CosmoPredict2:
  - Text encoder: UMT5 (dim=4096) vs T5 (dim=1024)
  - VAE latent channels: 48 vs 16
  - DiT hidden dim: 3072 (24×128) vs 2048 (16×128)
  - Scheduler: UniPCMultistepScheduler vs FlowMatchEulerDiscreteScheduler
  - No condition_mask / padding_mask (those are Cosmos-specific)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@dataclass
class WanJointOutput:
    """Output bundle for DiT4DiT-style joint video + feature extraction."""

    hidden_states: tuple
    video_loss: Optional[torch.Tensor] = None


class _Wan2_Interface(nn.Module):
    """
    World model wrapper for Wan2.2-TI2V-5B-Diffusers.

    The key methods are:
      - forward(**kwargs) → model outputs with hidden_states
      - build_inputs(images, instructions) → dict of tensors
      - generate(**kwargs) → video generation (optional)

    Representation extraction strategy:
      We run a single DiT forward pass at noise level σ≈0 and register
      forward hooks to capture intermediate block outputs. These are
      collected into a [B, N_tokens, hidden_dim] tensor that the action
      head can consume — analogous to VLM hidden_states.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get(
            "base_wm",
            config.framework.get("qwenvl", {}).get("base_vlm", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
        )
        self.config = config

        from diffusers import (
            AutoencoderKLWan,
            UniPCMultistepScheduler,
            WanTransformer3DModel,
        )
        from transformers import T5TokenizerFast, UMT5EncoderModel

        logger.info(f"Loading Wan2.2-TI2V from {model_name}")

        # --- Text encoder: UMT5-XXL ---
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )

        # --- DiT transformer ---
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=torch.bfloat16
        )

        # --- VAE (image → latents for DiT input, z_dim=48) ---
        self.vae = AutoencoderKLWan.from_pretrained(
            model_name, subfolder="vae", torch_dtype=torch.bfloat16
        )

        # --- Scheduler ---
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        # Use diffusers' VideoProcessor for image/video preprocessing (resize, normalize, etc.)
        from diffusers.video_processor import VideoProcessor
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # # Freeze VAE and text encoder by default
        # self.vae.requires_grad_(False)
        # self.text_encoder.requires_grad_(False)

        # DiT: 24 heads × 128 dim = 3072
        self._hidden_size = (
            self.transformer.config.num_attention_heads
            * self.transformer.config.attention_head_dim
        )

        # Config-like shim for framework to read hidden_size
        class _FakeConfig:
            pass

        self._model_config = _FakeConfig()
        self._model_config.hidden_size = self._hidden_size

        # Hook storage for intermediate features
        self._intermediate_features = []
        self._hooks = []

        extract_layers = wm_cfg.get("extract_layers", [-1])
        self._extract_layers = extract_layers
        self._register_hooks()

        # DiT4DiT-style joint supervision settings
        self.min_pixel_frames = int(wm_cfg.get("min_pixel_frames", 5))
        self.num_timestep_buckets = int(wm_cfg.get("num_timestep_buckets", 1000))
        self.feature_extract_tau = float(wm_cfg.get("feature_extract_tau", 0.5))

        if wm_cfg.get("freeze_vae", True):
            self.vae.requires_grad_(False)
        if wm_cfg.get("freeze_text_encoder", True):
            self.text_encoder.requires_grad_(False)
        if wm_cfg.get("enable_gradient_checkpointing", True):
            if hasattr(self.transformer, "enable_gradient_checkpointing"):
                self.transformer.enable_gradient_checkpointing()
                logger.info("Wan transformer gradient checkpointing enabled")

    @property
    def model(self):
        """Compatibility shim: framework code accesses self.backbone.model.config.hidden_size"""

        class _ModelShim:
            pass

        shim = _ModelShim()
        shim.config = self._model_config
        return shim

    def _register_hooks(self):
        """Register forward hooks on selected transformer blocks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        num_blocks = len(self.transformer.blocks)
        for layer_idx in self._extract_layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_blocks + layer_idx
            if 0 <= actual_idx < num_blocks:
                block = self.transformer.blocks[actual_idx]
                hook = block.register_forward_hook(self._capture_hook)
                self._hooks.append(hook)

    def _capture_hook(self, module, input, output):
        """Capture intermediate transformer block output."""
        if isinstance(output, tuple):
            self._intermediate_features.append(output[0])
        else:
            self._intermediate_features.append(output)

    def _encode_text(self, instructions, max_length=512):
        """Encode text instructions using UMT5."""
        device = next(self.text_encoder.parameters()).device

        text_inputs = self.tokenizer(
            instructions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_embeds = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            ).last_hidden_state  # [B, L, 4096]

        return text_embeds.to(dtype=torch.bfloat16)  # [B, max_length, 4096]

    def _encode_images_vae(self, images, num_frames=None):
        """Encode observation images through VAE to get latent tokens.

        Two-pass approach (same as CosmoPredict2):
          Pass 1: preprocess each sample, record real frame counts.
          Determine target_frames = num_frames if given, else batch max.
          Pass 2: truncate or pad each sample to target_frames.

        VAE config: z_dim=48, scale_factor_spatial=16, scale_factor_temporal=4
        T_latent = (target_frames - 1) // 4 + 1

        Args:
            images: List of List of PIL Images [B, [imgs...]]
            num_frames: If given, pad/truncate to this exact count.
                If None (default), pad to the max frame count in the batch.

        Returns:
            latents: [B, 48, T_latent, H/16, W/16] video latent tensor
        """
        device = next(self.vae.parameters()).device
        dtype = self.vae.dtype
        height, width = 480, 832

        # Pass 1: preprocess each sample, record real frame counts
        preprocessed = []
        frame_counts = []
        for sample_imgs in images:
            if not isinstance(sample_imgs, (list, tuple)):
                sample_imgs = [sample_imgs]

            video_tensor = self.video_processor.preprocess_video(sample_imgs, height=height, width=width)
            video_tensor = video_tensor.to(device=device, dtype=dtype)  # [1, C, n_imgs, H, W]
            preprocessed.append(video_tensor)
            frame_counts.append(video_tensor.shape[2])

        # Determine target frame count: use num_frames if specified, otherwise batch max
        target_frames = num_frames if num_frames is not None else max(frame_counts)

        # Pass 2: truncate or pad each sample to target_frames
        batch_videos = []
        for video_tensor in preprocessed:
            n = video_tensor.shape[2]
            if n > target_frames:
                video_tensor = video_tensor[:, :, :target_frames]
            elif n < target_frames:
                # Pad with last-frame repetition (matches official Wan pipeline)
                last_frame = video_tensor[:, :, -1:]
                padding = last_frame.repeat(1, 1, target_frames - n, 1, 1)
                video_tensor = torch.cat([video_tensor, padding], dim=2)
            batch_videos.append(video_tensor.squeeze(0))  # [C, target_frames, H, W]

        # Stack to [B, C, target_frames, H, W]
        video = torch.stack(batch_videos, dim=0)

        with torch.no_grad():
            latents = self.vae.encode(video).latent_dist.sample()  # [B, 48, T_latent, H/16, W/16]

        # Normalize latents (matches official Wan pipeline: (latent - mean) * (1/std))
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = (latents - latents_mean) * latents_std

        return latents

    def build_inputs(self, images, instructions, **kwargs):
        """Build inputs for the Wan DiT world model.

        Encoding pipeline:
        1. Text → UMT5 → text embeddings [B, L, 4096]
        2. Image → VAE → latents [B, 48, T, H', W'] (DiT input)

        Note: No CLIP image conditioning — this diffusers variant uses
        expand_timesteps mode (per-token timesteps) instead.

        Returns:
            dict with keys matching forward() expectations
        """
        assert len(images) == len(instructions)

        device = next(self.transformer.parameters()).device

        text_embeds = self._encode_text(instructions)
        latents = self._encode_images_vae(images)

        batch_size = latents.shape[0]
        device = latents.device

        # Wan2.2 TI2V uses expand_timesteps: timestep is per-token
        # Shape: [B, seq_len] where seq_len = T_lat * (H_lat//p_h) * (W_lat//p_w)
        # For feature extraction at σ≈0, use zeros (clean input)
        p_t, p_h, p_w = self.transformer.config.patch_size
        _, _, T, H, W = latents.shape
        seq_len = (T // p_t) * (H // p_h) * (W // p_w)
        assert seq_len <= 1024, (
            f"seq_len={seq_len} exceeds WanTransformer3D rope_max_seq_len=1024. "
            f"Reduce num_frames or image resolution. "
            f"(T_lat={T}, H_lat={H}, W_lat={W}, patch={p_t},{p_h},{p_w})"
        )
        timestep = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
            "_is_wm_input": True,
        }

    def forward(self, **kwargs):
        """Forward pass through the Wan DiT transformer.

        Runs a single-step forward to extract rich spatiotemporal features.
        Returns an output object with .hidden_states for compatibility.
        """
        kwargs.pop("_is_wm_input", False) # pop internal routing flags from kwargs to avoid passing them downstream
        kwargs.pop("output_hidden_states", False)
        kwargs.pop("return_dict", True)
        kwargs.pop("output_attentions", None)

        self._intermediate_features.clear()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            dit_output = self.transformer(
                hidden_states=kwargs["hidden_states"],
                timestep=kwargs["timestep"],
                encoder_hidden_states=kwargs["encoder_hidden_states"],
            )

        # Collect features from hooks
        # WanTransformer3DModel blocks output [B, seq_len, hidden_dim] (already flattened)
        extracted = []
        for feat in self._intermediate_features:
            if feat.dim() == 5:
                # [B, C, T, H, W] -> [B, T*H*W, C]
                B, C, T, H, W = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(feat)

        # Fallback: use transformer output directly
        if not extracted:
            out = dit_output.sample if hasattr(dit_output, "sample") else dit_output
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 5:
                B, C, T, H, W = out.shape
                out = out.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(out)

        class _WMOutput:
            def __init__(self, hidden_states_tuple, loss=None):
                self.hidden_states = hidden_states_tuple
                self.loss = loss # TODO if you want to add loss for image reconstruction or other auxiliary objectives, you can include it here and return it in the forward pass

        return _WMOutput(hidden_states_tuple=tuple(extracted))

    def _encode_video_batch(
        self,
        images: List[List],
        num_frames: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Encode a batch of frame lists to VAE latents.

        Returns:
            latents: [B, C, T_lat, H_lat, W_lat]
            cond_pixel_frame_counts: number of real (non-pad) pixel frames per sample
        """
        device = next(self.vae.parameters()).device
        dtype = self.vae.dtype
        height, width = 480, 832

        preprocessed = []
        frame_counts = []
        for sample_imgs in images:
            if not isinstance(sample_imgs, (list, tuple)):
                sample_imgs = [sample_imgs]
            video_tensor = self.video_processor.preprocess_video(sample_imgs, height=height, width=width)
            video_tensor = video_tensor.to(device=device, dtype=dtype)
            preprocessed.append(video_tensor)
            frame_counts.append(int(video_tensor.shape[2]))

        target_frames = num_frames if num_frames is not None else max(frame_counts)

        batch_videos = []
        for video_tensor in preprocessed:
            n = int(video_tensor.shape[2])
            if n > target_frames:
                video_tensor = video_tensor[:, :, :target_frames]
                n = target_frames
            elif n < target_frames:
                last_frame = video_tensor[:, :, -1:]
                padding = last_frame.repeat(1, 1, target_frames - n, 1, 1)
                video_tensor = torch.cat([video_tensor, padding], dim=2)
            batch_videos.append(video_tensor.squeeze(0))

        video = torch.stack(batch_videos, dim=0)

        with torch.no_grad():
            latents = self.vae.encode(video).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = (latents - latents_mean) * latents_std
        return latents, frame_counts

    def _encode_cond_future_latents(
        self,
        cond_images: List[List],
        future_images: List[List],
    ) -> Tuple[torch.Tensor, List[int]]:
        """Concatenate cond + future multi-view frames, pad, and VAE-encode."""
        assert len(cond_images) == len(future_images)
        combined = []
        cond_counts = []
        for cond, fut in zip(cond_images, future_images):
            frames = list(cond) + list(fut)
            cond_counts.append(len(cond))
            while len(frames) < self.min_pixel_frames:
                frames.append(frames[-1])
            combined.append(frames)
        latents, _ = self._encode_video_batch(combined, num_frames=self.min_pixel_frames)
        return latents, cond_counts

    def _pixel_frames_to_latent_frames(self, pixel_frame_count: int) -> int:
        return max(1, (int(pixel_frame_count) - 1) // self.vae_scale_factor_temporal + 1)

    def _future_latent_mask(self, latents: torch.Tensor, cond_pixel_counts: List[int]) -> torch.Tensor:
        """Bool mask over latent time axis [B, T_lat]. True = future (supervised) frames."""
        _, _, t_lat, _, _ = latents.shape
        masks = []
        for cond_px in cond_pixel_counts:
            n_cond_lat = min(self._pixel_frames_to_latent_frames(cond_px), t_lat)
            mask = torch.zeros(t_lat, dtype=torch.bool, device=latents.device)
            if n_cond_lat < t_lat:
                mask[n_cond_lat:] = True
            masks.append(mask)
        return torch.stack(masks, dim=0)

    def _tau_to_bucket(self, tau: torch.Tensor) -> torch.Tensor:
        return (tau.float() * self.num_timestep_buckets).long().clamp(0, self.num_timestep_buckets - 1)

    def _build_expand_timesteps(
        self,
        latents: torch.Tensor,
        cond_pixel_counts: List[int],
        tau_buckets: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token timesteps for Wan TI2V expand_timesteps mode."""
        batch_size, _, t_lat, h_lat, w_lat = latents.shape
        p_t, p_h, p_w = self.transformer.config.patch_size
        post_t = t_lat // p_t
        post_h = h_lat // p_h
        post_w = w_lat // p_w
        seq_len = post_t * post_h * post_w
        assert seq_len <= 1024, f"seq_len={seq_len} exceeds Wan rope_max_seq_len=1024"

        timesteps = torch.zeros(batch_size, seq_len, device=latents.device, dtype=torch.long)
        tokens_per_t = post_h * post_w
        for batch_idx in range(batch_size):
            n_cond_lat = min(self._pixel_frames_to_latent_frames(cond_pixel_counts[batch_idx]), post_t)
            bucket = int(tau_buckets[batch_idx].item())
            for t_idx in range(post_t):
                token_tau = 0 if t_idx < n_cond_lat else bucket
                start = t_idx * tokens_per_t
                end = start + tokens_per_t
                timesteps[batch_idx, start:end] = token_tau
        return timesteps

    def _expand_latent_mask(self, latent_mask_bt: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Expand [B, T_lat] mask to [B, C, T_lat, H_lat, W_lat]."""
        return latent_mask_bt[:, None, :, None, None].expand_as(latents)

    def _flow_match_noisy_latents(
        self,
        clean_latents: torch.Tensor,
        tau: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply linear flow-matching corruption on future latent frames only."""
        noise = torch.randn_like(clean_latents)
        tau_view = tau.view(-1, 1, 1, 1, 1).to(dtype=clean_latents.dtype)
        noisy = (1.0 - tau_view) * clean_latents + tau_view * noise
        target = noise - clean_latents

        keep_clean = ~future_mask
        noisy = torch.where(keep_clean, clean_latents, noisy)
        return noisy, target

    def _dit_forward_with_features(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, tuple]:
        self._intermediate_features.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dit_output = self.transformer(
                hidden_states=latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeds,
            )
        pred = dit_output.sample if hasattr(dit_output, "sample") else dit_output[0]

        extracted = []
        for feat in self._intermediate_features:
            if feat.dim() == 5:
                bsz, ch, t, h, w = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(bsz, t * h * w, ch)
            extracted.append(feat)
        if not extracted:
            if pred.dim() == 5:
                bsz, ch, t, h, w = pred.shape
                pred_tokens = pred.permute(0, 2, 3, 4, 1).reshape(bsz, t * h * w, ch)
            else:
                pred_tokens = pred
            extracted = [pred_tokens]
        return pred, tuple(extracted)

    def _compute_video_flow_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not future_mask.any():
            return pred.new_zeros(())
        sqerr = (pred.float() - target.float()).square() * future_mask.float()
        denom = future_mask.float().sum().clamp_min(1.0)
        denom = denom * pred.shape[1] * pred.shape[3] * pred.shape[4]
        return sqerr.sum() / denom

    def forward_joint(
        self,
        cond_images: List[List],
        future_images: List[List],
        instructions: List[str],
        feature_extract_tau: Optional[float] = None,
    ) -> WanJointOutput:
        """DiT4DiT-style joint forward: video flow loss + hidden features for action."""
        assert len(cond_images) == len(future_images) == len(instructions)

        text_embeds = self._encode_text(instructions)
        clean_latents, cond_counts = self._encode_cond_future_latents(cond_images, future_images)
        batch_size = clean_latents.shape[0]
        device = clean_latents.device

        future_latent_mask = self._future_latent_mask(clean_latents, cond_counts)
        future_mask = self._expand_latent_mask(future_latent_mask, clean_latents)

        # Video branch: random τ_v ~ U[0, 1]
        tau_v = torch.rand(batch_size, device=device)
        tau_v_buckets = self._tau_to_bucket(tau_v)
        noisy_v, target_v = self._flow_match_noisy_latents(clean_latents, tau_v, future_mask)
        timesteps_v = self._build_expand_timesteps(clean_latents, cond_counts, tau_v_buckets)
        pred_v, _ = self._dit_forward_with_features(noisy_v, timesteps_v, text_embeds)
        video_loss = self._compute_video_flow_loss(pred_v, target_v, future_mask)

        # Feature branch: fixed τ_f for stable action conditioning
        tau_f_value = self.feature_extract_tau if feature_extract_tau is None else float(feature_extract_tau)
        tau_f = torch.full((batch_size,), tau_f_value, device=device)
        tau_f_buckets = self._tau_to_bucket(tau_f)
        noisy_f, _ = self._flow_match_noisy_latents(clean_latents, tau_f, future_mask)
        timesteps_f = self._build_expand_timesteps(clean_latents, cond_counts, tau_f_buckets)
        _, hidden_states = self._dit_forward_with_features(noisy_f, timesteps_f, text_embeds)

        return WanJointOutput(hidden_states=hidden_states, video_loss=video_loss)

    def forward_features_for_action(
        self,
        cond_images: List[List],
        instructions: List[str],
        feature_extract_tau: Optional[float] = None,
    ) -> WanJointOutput:
        """Inference-time feature extraction at fixed τ_f (no future observations required)."""
        text_embeds = self._encode_text(instructions)
        latents, cond_counts = self._encode_video_batch(cond_images, num_frames=self.min_pixel_frames)
        batch_size = latents.shape[0]
        device = latents.device

        future_latent_mask = self._future_latent_mask(latents, cond_counts)
        future_mask = self._expand_latent_mask(future_latent_mask, latents)

        tau_f_value = self.feature_extract_tau if feature_extract_tau is None else float(feature_extract_tau)
        tau_f = torch.full((batch_size,), tau_f_value, device=device)
        tau_f_buckets = self._tau_to_bucket(tau_f)

        # Future slots use sampled noise; cond slots stay clean (DiT4DiT inference).
        noise = torch.randn_like(latents)
        mixed_clean = torch.where(future_mask, noise, latents)
        noisy_f, _ = self._flow_match_noisy_latents(mixed_clean, tau_f, future_mask)

        timesteps_f = self._build_expand_timesteps(latents, cond_counts, tau_f_buckets)
        _, hidden_states = self._dit_forward_with_features(noisy_f, timesteps_f, text_embeds)
        return WanJointOutput(hidden_states=hidden_states, video_loss=None)

    def generate(self, **kwargs):
        """Video generation using the WanPipeline.

        Not used during standard VLA training, but useful for visualization
        and planning-based approaches.
        """
        from diffusers import WanPipeline

        pipe = WanPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.scheduler,
        )
        return pipe(**kwargs)

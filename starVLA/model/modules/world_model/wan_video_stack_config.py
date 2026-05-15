from __future__ import annotations

"""HuggingFace ``PretrainedConfig`` for the full Wan video stack (VAE + DiT + schedulers).

Used by the joint video–action MoE (``action_model.wan_video_action_moe_model``), which
instantiates ``WanVideoModel`` from ``wan_video_stack_model``. The lightweight WM4A
``Wan2`` wrapper (``Wan2.py``) is a separate code path for feature extraction only.
"""

from transformers.configuration_utils import PretrainedConfig


class WanVideoConfig(PretrainedConfig):
    model_type = "wan_video"

    def __init__(
        self,
        wan_model_name_or_path: str | None = None,
        load_wan_pretrained: bool = True,
        freeze_vae: bool = True,
        sample_posterior: bool = True,
        compile_vae_encode: bool = False,
        transformer_gradient_checkpointing: bool = False,
        use_view_embedding: bool = True,
        concat_view_embedding: bool = False,
        view_embedding_dim: int = 16,
        max_view_embeddings: int = 128,
        use_cross_view_attention: bool = False,
        text_embed_dim: int = 4096,
        video_num_frames: int = 21,
        vae_context_prefix_frames_per_view: int = 0,
        video_fps: float = 10.0,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        video_snr_shift: float = 5.0,
        **kwargs,
    ):
        self.wan_model_name_or_path = wan_model_name_or_path
        self.load_wan_pretrained = bool(load_wan_pretrained)
        self.freeze_vae = bool(freeze_vae)
        self.sample_posterior = bool(sample_posterior)
        self.compile_vae_encode = bool(compile_vae_encode)
        self.transformer_gradient_checkpointing = bool(transformer_gradient_checkpointing)
        self.use_view_embedding = bool(use_view_embedding)
        self.concat_view_embedding = bool(concat_view_embedding)
        self.view_embedding_dim = int(view_embedding_dim)
        self.max_view_embeddings = int(max_view_embeddings)
        self.use_cross_view_attention = bool(use_cross_view_attention)
        self.text_embed_dim = int(text_embed_dim)
        self.video_num_frames = int(video_num_frames)
        self.vae_context_prefix_frames_per_view = int(vae_context_prefix_frames_per_view)
        self.video_fps = float(video_fps)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.video_snr_shift = float(video_snr_shift)
        super().__init__(**kwargs)


WanVideoConfig.register_for_auto_class()

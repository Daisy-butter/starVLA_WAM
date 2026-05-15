from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig

from starVLA.model.modules.world_model.wan_video_action_semantics import canonicalize_video_action_semantics

class WanVideoActionConfig(PretrainedConfig):
    model_type = "wan_video_action"

    def __init__(
        self,
        vla_model_variant: str = "moe_parallel",
        wan_model_name_or_path: str | None = None,
        load_wan_pretrained: bool = True,
        freeze_vae: bool = True,
        sample_posterior: bool = True,
        compile_vae_encode: bool = False,
        transformer_gradient_checkpointing: bool = False,
        video_num_frames: int = 21,
        sparse_history_video_num_frames: int = 0,
        future_video_num_frames: int = 20,
        video_fps: float = 10.0,
        use_view_embedding: bool = True,
        concat_view_embedding: bool = False,
        view_embedding_dim: int = 16,
        max_view_embeddings: int = 8,
        use_cross_view_attention: bool = False,
        text_embed_dim: int = 4096,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        video_snr_shift: float = 5.0,
        action_dim: int = 16,
        future_action_steps: int = 16,
        history_action_steps: int = 0,
        action_hidden_size: int = 1024,
        action_num_heads: int = 16,
        action_num_layers: int = 6,
        action_mlp_ratio: float = 4.0,
        action_modulation_rank: int = 0,
        robot_embed_enabled: bool = False,
        action_pos_emb_max_t: int = 128,
        action_noise_multiplier: float = 1.0,
        action_snr_shift: float = 5.0,
        action_sigma_min: float = 0.0,
        action_sigma_max: float = 1.0,
        action_train_sigma_min: float = 0.0,
        action_train_sigma_max: float = 1.0,
        action_sampling_distribution: str = "uniform_index",
        action_hybrid_uniform_ratio: float = 0.3,
        action_hybrid_uniform_lower: float = 1.0,
        action_hybrid_uniform_upper: float = 85.0,
        action_lognormal_mean: float = 1.39,
        action_lognormal_std: float = 1.2,
        true_shared_base_sigma: bool = False,
        video_noise_multiplier: float = 1.0,
        video_high_sigma_ratio: float = 0.0,
        sparse_history_fps: float = 1.0,
        sparse_history_simulate_short_history_prob: float = 0.05,
        fastwam_anchor_frames: int = 1,
        fastwam_use_anchor_cache: bool = True,
        fastwam_bidirectional_action: bool = True,
        **kwargs,
    ):
        self.vla_model_variant = str(vla_model_variant)
        self.wan_model_name_or_path = wan_model_name_or_path
        self.load_wan_pretrained = bool(load_wan_pretrained)
        self.freeze_vae = bool(freeze_vae)
        self.sample_posterior = bool(sample_posterior)
        self.compile_vae_encode = bool(compile_vae_encode)
        self.transformer_gradient_checkpointing = bool(transformer_gradient_checkpointing)

        self.video_num_frames = int(video_num_frames)
        raw_current_video_num_frames = kwargs.pop("current_video_num_frames", None)
        raw_conditioning_video_num_frames = kwargs.pop("conditioning_video_num_frames", None)
        semantics = canonicalize_video_action_semantics(
            {
                "current_video_num_frames": raw_current_video_num_frames,
                "sparse_history_video_num_frames": int(sparse_history_video_num_frames),
                "vla_model_variant": self.vla_model_variant,
                "fastwam_anchor_frames": int(fastwam_anchor_frames),
            }
        )
        self.current_video_num_frames = int(semantics["current_video_num_frames"])
        self.sparse_history_video_num_frames = int(semantics["sparse_history_video_num_frames"])
        self.conditioning_video_num_frames = int(semantics["conditioning_video_num_frames"])
        if (
            raw_conditioning_video_num_frames is not None
            and int(raw_conditioning_video_num_frames) != int(self.conditioning_video_num_frames)
        ):
            raise ValueError(
                "conditioning_video_num_frames does not match sparse-history semantics. "
                f"Got {raw_conditioning_video_num_frames}, expected {self.conditioning_video_num_frames}."
            )
        self.future_video_num_frames = int(future_video_num_frames)
        self.video_fps = float(video_fps)
        self.use_view_embedding = bool(use_view_embedding)
        self.concat_view_embedding = bool(concat_view_embedding)
        self.view_embedding_dim = int(view_embedding_dim)
        self.max_view_embeddings = int(max_view_embeddings)
        self.use_cross_view_attention = bool(use_cross_view_attention)
        self.text_embed_dim = int(text_embed_dim)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.video_snr_shift = float(video_snr_shift)

        self.action_dim = int(action_dim)
        self.future_action_steps = int(future_action_steps)
        self.history_action_steps = int(history_action_steps)
        self.action_hidden_size = int(action_hidden_size)
        self.action_num_heads = int(action_num_heads)
        self.action_num_layers = int(action_num_layers)
        self.action_mlp_ratio = float(action_mlp_ratio)
        self.action_modulation_rank = int(action_modulation_rank)
        self.robot_embed_enabled = bool(robot_embed_enabled)
        self.action_pos_emb_max_t = int(action_pos_emb_max_t)
        self.action_noise_multiplier = float(action_noise_multiplier)
        self.action_snr_shift = float(action_snr_shift)
        self.action_sigma_min = float(action_sigma_min)
        self.action_sigma_max = float(action_sigma_max)
        self.action_train_sigma_min = float(action_train_sigma_min)
        self.action_train_sigma_max = float(action_train_sigma_max)
        self.action_sampling_distribution = str(action_sampling_distribution)
        self.action_hybrid_uniform_ratio = float(action_hybrid_uniform_ratio)
        self.action_hybrid_uniform_lower = float(action_hybrid_uniform_lower)
        self.action_hybrid_uniform_upper = float(action_hybrid_uniform_upper)
        self.action_lognormal_mean = float(action_lognormal_mean)
        self.action_lognormal_std = float(action_lognormal_std)
        self.true_shared_base_sigma = bool(true_shared_base_sigma)
        self.video_noise_multiplier = float(video_noise_multiplier)
        self.video_high_sigma_ratio = float(video_high_sigma_ratio)
        self.sparse_history_fps = float(sparse_history_fps)
        self.sparse_history_simulate_short_history_prob = float(sparse_history_simulate_short_history_prob)
        self.fastwam_anchor_frames = int(fastwam_anchor_frames)
        self.fastwam_use_anchor_cache = bool(fastwam_use_anchor_cache)
        self.fastwam_bidirectional_action = bool(fastwam_bidirectional_action)

        super().__init__(**kwargs)


WanVideoActionConfig.register_for_auto_class()

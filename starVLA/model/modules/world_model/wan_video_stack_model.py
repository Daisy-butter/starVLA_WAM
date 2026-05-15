from __future__ import annotations

"""Full Wan **video** stack as a ``PreTrainedModel`` (diffusers VAE + ``WanTransformer3D``).

This is the perception / dynamics backbone for **joint** Wan video–action training
(see ``action_model.wan_video_action_moe_model``). It is **not** the same module as
``Wan2._Wan2_Interface`` (hooks-only world model for π-style heads).
"""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from collections import OrderedDict

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_wan import FP32LayerNorm, WanAttention, WanAttnProcessor
from diffusers.video_processor import VideoProcessor
from diffusers.utils.torch_utils import randn_tensor
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoTokenizer, UMT5EncoderModel

from .wan_video_stack_config import WanVideoConfig


def _resolve_wan_root(model_path: str | None) -> Path:
    if not model_path:
        raise ValueError("wan_model_name_or_path must be set.")
    path = Path(model_path).expanduser().resolve()
    if path.is_dir() and (path / "transformer").is_dir() and (path / "vae").is_dir():
        return path
    if path.name in {"transformer", "vae", "scheduler", "tokenizer", "text_encoder"}:
        candidate = path.parent
        if (candidate / "transformer").is_dir() and (candidate / "vae").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot resolve Wan diffusers root from: {path}")


def _load_diffusers_config(model_cls, root: Path, subfolder: str):
    return model_cls.load_config(str(root), subfolder=subfolder)


@dataclass
class WanVideoOutput(ModelOutput):
    pred_flow: torch.FloatTensor | None = None
    latents: torch.FloatTensor | None = None
    noisy_latents: torch.FloatTensor | None = None


@dataclass
class WanVideoGenerationOutput(ModelOutput):
    frames: torch.Tensor | None = None
    latents: torch.Tensor | None = None
    prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None


class WanCrossViewAttentionBlock(nn.Module):
    """Residual cross-view attention aligned by time step across views."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        *,
        max_view_embeddings: int = 8,
        cross_view_attn_map: dict[int, list[int]] | None = None,
    ):
        super().__init__()
        self.layer_norm = FP32LayerNorm(dim, eps, elementwise_affine=True)
        self.attn = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.cross_view_attn_map = cross_view_attn_map or {
            int(view_id): [int(other_id) for other_id in range(int(max_view_embeddings)) if int(other_id) != int(view_id)]
            for view_id in range(int(max_view_embeddings))
        }
        output_proj = self.attn.to_out[0]
        if isinstance(output_proj, nn.Linear):
            nn.init.zeros_(output_proj.weight)
            if output_proj.bias is not None:
                nn.init.zeros_(output_proj.bias)
        self._enable_batched_grouped_attention = False
        self._layout_plan_cache: OrderedDict[tuple[int, ...], dict[str, object]] = OrderedDict()
        self._layout_plan_cache_max_entries = 16

    def set_batched_grouped_attention_enabled(self, enabled: bool) -> None:
        self._enable_batched_grouped_attention = bool(enabled)

    @staticmethod
    def _group_specs_key(neighbor_indices: list[int]) -> tuple[int, bool]:
        return int(len(neighbor_indices) if neighbor_indices else 1), bool(neighbor_indices)

    @staticmethod
    def _reshape_grouped_queries(
        view_states: torch.Tensor,
        view_indices: torch.Tensor,
        frames_per_view: int,
        spatial_tokens: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        gathered = view_states.index_select(0, view_indices)
        return gathered.reshape(int(view_indices.shape[0]) * frames_per_view, spatial_tokens, hidden_dim)

    @staticmethod
    def _reshape_grouped_neighbor_context(
        view_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        frames_per_view: int,
        spatial_tokens: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        num_groups = int(neighbor_indices.shape[0])
        neighbor_count = int(neighbor_indices.shape[1])
        gathered = view_states.index_select(0, neighbor_indices.reshape(-1))
        gathered = gathered.reshape(num_groups, neighbor_count, frames_per_view, spatial_tokens, hidden_dim)
        gathered = gathered.permute(0, 2, 1, 3, 4)
        return gathered.reshape(num_groups * frames_per_view, neighbor_count * spatial_tokens, hidden_dim)

    def _get_sample_layout_plan(
        self,
        sample_frame_view_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> dict[str, object]:
        cache_key = tuple(int(item) for item in sample_frame_view_ids.detach().cpu().tolist())
        cached = self._layout_plan_cache.get(cache_key)
        if cached is None:
            blocks = self._split_view_blocks(sample_frame_view_ids)
            view_ids = [view_id for view_id, _, _ in blocks]
            block_lengths = [end - start for _, start, end in blocks]
            can_use_grouped_cross_view = (
                len(blocks) > 1
                and len(set(view_ids)) == len(view_ids)
                and bool(block_lengths)
                and min(block_lengths) > 0
                and len(set(block_lengths)) == 1
            )
            group_specs_cpu: list[dict[str, object]] = []
            if can_use_grouped_cross_view:
                view_id_to_index = {int(view_id): idx for idx, view_id in enumerate(view_ids)}
                grouped_specs: dict[tuple[int, bool], list[tuple[int, list[int]]]] = defaultdict(list)
                for view_idx, view_id in enumerate(view_ids):
                    neighbor_indices = [
                        view_id_to_index[int(neighbor_view_id)]
                        for neighbor_view_id in self.cross_view_attn_map.get(int(view_id), [])
                        if int(neighbor_view_id) in view_id_to_index and int(neighbor_view_id) != int(view_id)
                    ]
                    grouped_specs[self._group_specs_key(neighbor_indices)].append((view_idx, neighbor_indices))
                for (_, has_neighbors), group_specs in grouped_specs.items():
                    group_specs_cpu.append(
                        {
                            "has_neighbors": bool(has_neighbors),
                            "view_indices": tuple(int(view_idx) for view_idx, _ in group_specs),
                            "neighbor_indices": tuple(tuple(int(idx) for idx in neighbor_indices) for _, neighbor_indices in group_specs),
                        }
                    )
            cached = {
                "blocks": blocks,
                "can_use_grouped_cross_view": bool(can_use_grouped_cross_view),
                "frames_per_view": int(block_lengths[0]) if can_use_grouped_cross_view else 0,
                "group_specs_cpu": group_specs_cpu,
                "device_group_specs": {},
            }
            self._layout_plan_cache[cache_key] = cached
            self._layout_plan_cache.move_to_end(cache_key)
            while len(self._layout_plan_cache) > int(self._layout_plan_cache_max_entries):
                self._layout_plan_cache.popitem(last=False)
        else:
            self._layout_plan_cache.move_to_end(cache_key)

        device_key = str(device)
        device_group_specs = cached["device_group_specs"]  # type: ignore[index]
        if device_key not in device_group_specs:
            resolved_specs: list[dict[str, object]] = []
            for item in cached["group_specs_cpu"]:  # type: ignore[index]
                view_index_tensor = torch.tensor(item["view_indices"], device=device, dtype=torch.long)
                neighbor_indices = item["neighbor_indices"]
                neighbor_index_tensor = None
                if item["has_neighbors"] and neighbor_indices:
                    neighbor_index_tensor = torch.tensor(neighbor_indices, device=device, dtype=torch.long)
                resolved_specs.append(
                    {
                        "has_neighbors": bool(item["has_neighbors"]),
                        "view_index_tensor": view_index_tensor,
                        "neighbor_index_tensor": neighbor_index_tensor,
                    }
                )
            device_group_specs[device_key] = resolved_specs
        return {
            "blocks": cached["blocks"],
            "can_use_grouped_cross_view": cached["can_use_grouped_cross_view"],
            "frames_per_view": cached["frames_per_view"],
            "group_specs": device_group_specs[device_key],
        }

    @staticmethod
    def _frame_view_ids(token_view_ids: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
        if tokens_per_frame <= 0:
            raise ValueError(f"tokens_per_frame must be > 0, got {tokens_per_frame}")
        if int(token_view_ids.shape[1]) % int(tokens_per_frame) != 0:
            raise ValueError(
                "token_view_ids length must be divisible by tokens_per_frame. "
                f"Got {tuple(token_view_ids.shape)} and tokens_per_frame={tokens_per_frame}"
            )
        frame_view_ids = token_view_ids[:, :: int(tokens_per_frame)]
        expanded = frame_view_ids.repeat_interleave(int(tokens_per_frame), dim=1)
        if not torch.equal(expanded, token_view_ids):
            raise ValueError("Each post-patch frame must contain tokens from a single camera/view.")
        return frame_view_ids

    @staticmethod
    def _split_view_blocks(frame_view_ids: torch.Tensor) -> list[tuple[int, int, int]]:
        ids = frame_view_ids.detach().cpu().tolist()
        if not ids:
            return []
        blocks: list[tuple[int, int, int]] = []
        start = 0
        for idx in range(1, len(ids)):
            if ids[idx] != ids[idx - 1]:
                blocks.append((int(ids[start]), start, idx))
                start = idx
        blocks.append((int(ids[start]), start, len(ids)))
        return blocks

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_view_ids: torch.Tensor | None,
        *,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if token_view_ids is None:
            return hidden_states
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be [B, L, D], got {tuple(hidden_states.shape)}")
        if token_view_ids.ndim != 2:
            raise ValueError(f"token_view_ids must be [B, L], got {tuple(token_view_ids.shape)}")
        if tuple(hidden_states.shape[:2]) != tuple(token_view_ids.shape):
            raise ValueError(
                "cross-view token ids must match hidden token layout. "
                f"Got hidden_states={tuple(hidden_states.shape)}, token_view_ids={tuple(token_view_ids.shape)}"
            )
        if int(hidden_states.shape[1]) != int(num_frames) * int(height) * int(width):
            raise ValueError(
                "hidden token length must match num_frames * height * width. "
                f"Got {tuple(hidden_states.shape)} vs num_frames={num_frames}, height={height}, width={width}"
            )

        spatial_tokens = int(height) * int(width)
        frame_view_ids = self._frame_view_ids(token_view_ids, spatial_tokens)
        normalized_states = self.layer_norm(hidden_states)
        attn_residual = torch.zeros_like(hidden_states)
        for batch_idx in range(int(hidden_states.shape[0])):
            sample_states = normalized_states[batch_idx].view(int(num_frames), spatial_tokens, int(hidden_states.shape[-1]))
            sample_frame_view_ids = frame_view_ids[batch_idx]
            sample_layout = self._get_sample_layout_plan(sample_frame_view_ids, device=hidden_states.device)
            blocks = sample_layout["blocks"]  # type: ignore[assignment]
            can_use_grouped_cross_view = bool(sample_layout["can_use_grouped_cross_view"])

            if can_use_grouped_cross_view:
                view_ids = [view_id for view_id, _, _ in blocks]
                frames_per_view = int(sample_layout["frames_per_view"])
                view_chunks = [sample_states[start:end] for _, start, end in blocks]
                view_states = torch.stack(view_chunks, dim=0)
                view_outputs = torch.zeros_like(view_states)

                if self._enable_batched_grouped_attention:
                    hidden_dim = int(hidden_states.shape[-1])
                    for group_spec in sample_layout["group_specs"]:  # type: ignore[index]
                        view_index_tensor = group_spec["view_index_tensor"]  # type: ignore[index]
                        batched_query = self._reshape_grouped_queries(
                            view_states,
                            view_index_tensor,
                            frames_per_view,
                            spatial_tokens,
                            hidden_dim,
                        )
                        if bool(group_spec["has_neighbors"]):  # type: ignore[index]
                            neighbor_index_tensor = group_spec["neighbor_index_tensor"]  # type: ignore[index]
                            batched_context = self._reshape_grouped_neighbor_context(
                                view_states,
                                neighbor_index_tensor,
                                frames_per_view,
                                spatial_tokens,
                                hidden_dim,
                            )
                        else:
                            batched_context = batched_query
                        batched_output = self.attn(
                            batched_query,
                            encoder_hidden_states=batched_context,
                        ).to(dtype=view_outputs.dtype)
                        batched_output = batched_output.reshape(
                            int(view_index_tensor.shape[0]),
                            frames_per_view,
                            spatial_tokens,
                            hidden_dim,
                        )
                        view_outputs.index_copy_(0, view_index_tensor, batched_output)
                else:
                    view_id_to_index = {int(view_id): idx for idx, view_id in enumerate(view_ids)}
                    for view_idx, view_id in enumerate(view_ids):
                        neighbor_indices = [
                            view_id_to_index[int(neighbor_view_id)]
                            for neighbor_view_id in self.cross_view_attn_map.get(int(view_id), [])
                            if int(neighbor_view_id) in view_id_to_index and int(neighbor_view_id) != int(view_id)
                        ]
                        if neighbor_indices:
                            context = view_states[neighbor_indices].permute(1, 0, 2, 3).reshape(
                                frames_per_view,
                                len(neighbor_indices) * spatial_tokens,
                                int(hidden_states.shape[-1]),
                            )
                        else:
                            # Single-view or unmapped-view fallback: keep the same parameter
                            # path active by attending to the current view itself.
                            context = view_states[view_idx]

                        query = view_states[view_idx]
                        view_outputs[view_idx] = self.attn(
                            query,
                            encoder_hidden_states=context,
                        ).to(dtype=view_outputs.dtype)
                sample_residual = view_outputs.reshape(
                    len(view_ids) * frames_per_view,
                    spatial_tokens,
                    int(hidden_states.shape[-1]),
                )
            else:
                fallback_outputs: list[torch.Tensor] = []
                for _, start, end in blocks:
                    if end <= start:
                        continue
                    query = sample_states[start:end]
                    fallback_outputs.append(
                        self.attn(
                            query,
                            encoder_hidden_states=query,
                        ).to(dtype=query.dtype)
                    )
                if not fallback_outputs:
                    continue
                sample_residual = torch.cat(fallback_outputs, dim=0)
            attn_residual[batch_idx] = sample_residual.reshape(int(num_frames) * spatial_tokens, int(hidden_states.shape[-1]))

        return hidden_states + attn_residual


class WanVideoModel(PreTrainedModel):
    config_class = WanVideoConfig
    main_input_name = "latents"

    def __init__(self, config: WanVideoConfig):
        super().__init__(config)
        self.model_root = _resolve_wan_root(config.wan_model_name_or_path)
        self.transformer = self._build_transformer(config)
        self.vae = self._build_vae(config)
        self._compiled_vae_encode = None
        self.latent_channels = int(self.transformer.config.in_channels)
        transformer_text_dim = int(self.transformer.config.text_dim)
        self.concat_view_embedding = bool(getattr(config, "concat_view_embedding", False) and bool(config.use_view_embedding))
        self.tokenizer = None
        self.text_encoder = None
        self.scheduler = None
        self.video_processor = None

        self.text_proj = None
        if int(config.text_embed_dim) != transformer_text_dim:
            self.text_proj = nn.Linear(int(config.text_embed_dim), transformer_text_dim, bias=False)

        self.view_embeddings = None
        self.view_embedding_proj = None
        if bool(config.use_view_embedding):
            view_embedding_dim = int(config.view_embedding_dim)
            self.view_embeddings = nn.Embedding(int(config.max_view_embeddings), view_embedding_dim)
            if not self.concat_view_embedding and view_embedding_dim != self.latent_channels:
                self.view_embedding_proj = nn.Linear(view_embedding_dim, self.latent_channels, bias=False)
            if self.concat_view_embedding:
                nn.init.trunc_normal_(self.view_embeddings.weight, std=0.02)
                self._expand_patch_embedding_for_concat_view(int(view_embedding_dim))
            else:
                nn.init.zeros_(self.view_embeddings.weight)

        self.cross_view_blocks = None
        if bool(getattr(config, "use_cross_view_attention", False)):
            num_heads = int(getattr(self.transformer.config, "num_attention_heads", 0) or 0)
            inner_dim = int(num_heads * int(getattr(self.transformer.config, "attention_head_dim", 0) or 0))
            eps = float(getattr(self.transformer.config, "eps", 1e-6) or 1e-6)
            if num_heads <= 0 or inner_dim <= 0:
                raise ValueError("Wan transformer config missing attention dims required for cross-view attention.")
            self.cross_view_blocks = nn.ModuleList(
                [
                    WanCrossViewAttentionBlock(
                        dim=inner_dim,
                        num_heads=num_heads,
                        eps=eps,
                        max_view_embeddings=int(config.max_view_embeddings),
                    )
                    for _ in self.transformer.blocks
                ]
            )

        if bool(config.transformer_gradient_checkpointing) and hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()

        latent_mean, latent_std = self._build_latent_stat_buffers()
        if latent_mean is not None and latent_std is not None:
            self.register_buffer("_latent_mean", latent_mean, persistent=False)
            self.register_buffer("_latent_std", latent_std, persistent=False)
        else:
            self._latent_mean = None
            self._latent_std = None

        if bool(config.freeze_vae):
            self.vae.requires_grad_(False)
            self.vae.eval()
        self._maybe_compile_vae_encode()

        # Do not call `PreTrainedModel.post_init()`: it would recurse through all
        # submodules and reinitialize the already-loaded diffusers Wan weights.
        # The adapter layers created above already have their own module defaults.

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    def _expand_patch_embedding_for_concat_view(self, view_embedding_dim: int) -> None:
        patch_embedding = self.transformer.patch_embedding
        if not isinstance(patch_embedding, nn.Conv3d):
            raise TypeError(
                "concat_view_embedding expects Wan patch_embedding to be nn.Conv3d, "
                f"got {type(patch_embedding)!r}"
            )
        if int(view_embedding_dim) <= 0:
            raise ValueError(f"view_embedding_dim must be positive for concat_view_embedding, got {view_embedding_dim}")

        new_in_channels = int(patch_embedding.in_channels) + int(view_embedding_dim)
        patch_weight = patch_embedding.weight
        patch_device = patch_weight.device
        patch_dtype = patch_weight.dtype
        expanded_patch_embedding = nn.Conv3d(
            new_in_channels,
            int(patch_embedding.out_channels),
            kernel_size=patch_embedding.kernel_size,
            stride=patch_embedding.stride,
            padding=patch_embedding.padding,
            dilation=patch_embedding.dilation,
            groups=patch_embedding.groups,
            bias=patch_embedding.bias is not None,
            device=patch_device,
            dtype=patch_dtype,
        )

        if not bool(getattr(patch_weight, "is_meta", False)):
            with torch.no_grad():
                expanded_patch_embedding.weight.zero_()
                expanded_patch_embedding.weight[:, : int(patch_embedding.in_channels)].copy_(patch_weight)
                if patch_embedding.bias is not None and expanded_patch_embedding.bias is not None:
                    expanded_patch_embedding.bias.copy_(patch_embedding.bias)

        self.transformer.patch_embedding = expanded_patch_embedding
        self.transformer.config.in_channels = int(new_in_channels)

    def _build_latent_stat_buffers(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        mean_values = getattr(self.vae.config, "latents_mean", None)
        std_values = getattr(self.vae.config, "latents_std", None)
        if mean_values is None or std_values is None:
            return None, None
        latent_channels = int(self.vae.config.z_dim)
        mean = torch.tensor(mean_values, dtype=torch.float32).view(1, latent_channels, 1, 1, 1)
        std = torch.tensor(std_values, dtype=torch.float32).view(1, latent_channels, 1, 1, 1)
        return mean, std

    def _latent_mean_std(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        mean = getattr(self, "_latent_mean", None)
        std = getattr(self, "_latent_std", None)
        if mean is None or std is None:
            return None, None
        return mean.to(device=device, dtype=dtype), std.to(device=device, dtype=dtype)

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = self._latent_mean_std(latents.device, latents.dtype)
        if mean is None or std is None:
            return latents
        return (latents - mean) / std

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = self._latent_mean_std(latents.device, latents.dtype)
        if mean is None or std is None:
            return latents
        return latents * std + mean

    def _ensure_inference_modules(
        self,
        *,
        load_tokenizer: bool = True,
        load_text_encoder: bool = True,
    ) -> None:
        if load_tokenizer and self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_root),
                subfolder="tokenizer",
                trust_remote_code=True,
                use_fast=True,
            )
        if load_text_encoder and self.text_encoder is None:
            self.text_encoder = UMT5EncoderModel.from_pretrained(
                str(self.model_root),
                subfolder="text_encoder",
                torch_dtype=next(self.transformer.parameters()).dtype,
            ).to(device=self.device)
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)
        if self.scheduler is None:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(self.model_root), subfolder="scheduler")
        if self.video_processor is None:
            self.video_processor = VideoProcessor(vae_scale_factor=int(self.vae.config.scale_factor_spatial))

    def train(self, mode: bool = True):
        super().train(mode)
        if bool(self.config.freeze_vae):
            self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def _build_transformer(self, config: WanVideoConfig) -> WanTransformer3DModel:
        if bool(config.load_wan_pretrained):
            return WanTransformer3DModel.from_pretrained(str(self.model_root), subfolder="transformer")
        transformer_config = _load_diffusers_config(WanTransformer3DModel, self.model_root, "transformer")
        return WanTransformer3DModel.from_config(transformer_config)

    def _build_vae(self, config: WanVideoConfig) -> AutoencoderKLWan:
        if bool(config.load_wan_pretrained):
            return AutoencoderKLWan.from_pretrained(str(self.model_root), subfolder="vae")
        vae_config = _load_diffusers_config(AutoencoderKLWan, self.model_root, "vae")
        return AutoencoderKLWan.from_config(vae_config)

    def _maybe_compile_vae_encode(self) -> None:
        if not bool(getattr(self.config, "compile_vae_encode", False)):
            return
        if not hasattr(torch, "compile"):
            return

        def _encode_only(video: torch.Tensor) -> torch.Tensor:
            posterior = self.vae.encode(video).latent_dist
            if bool(self.config.sample_posterior):
                return posterior.sample()
            return posterior.mode()

        try:
            self._compiled_vae_encode = torch.compile(_encode_only, mode="reduce-overhead", fullgraph=False)
        except Exception:
            self._compiled_vae_encode = None

    @torch.inference_mode()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae_param = next(self.vae.parameters())
        video = video.to(device=vae_param.device, dtype=vae_param.dtype)
        if self._compiled_vae_encode is not None:
            latents = self._compiled_vae_encode(video)
        else:
            posterior = self.vae.encode(video).latent_dist
            latents = posterior.sample() if bool(self.config.sample_posterior) else posterior.mode()
        return self._normalize_latents(latents)

    @torch.inference_mode()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_param = next(self.vae.parameters())
        latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
        latents = self._denormalize_latents(latents)
        return self.vae.decode(latents).sample

    @staticmethod
    def _split_view_blocks(view_indices: torch.Tensor) -> list[tuple[int, int, int]]:
        idx = view_indices.detach().cpu().tolist()
        if not idx:
            return []
        blocks: list[tuple[int, int, int]] = []
        start = 0
        for i in range(1, len(idx)):
            if idx[i] != idx[i - 1]:
                blocks.append((int(idx[start]), start, i))
                start = i
        blocks.append((int(idx[start]), start, len(idx)))
        return blocks

    def _num_latent_frames_for_frames(self, num_frames: int) -> int:
        temporal_scale = int(self.vae.config.scale_factor_temporal)
        return (int(num_frames) - 1) // max(temporal_scale, 1) + 1

    @staticmethod
    def _normalize_view_indices(view_indices: torch.Tensor | None) -> torch.Tensor | None:
        if view_indices is None:
            return None
        if view_indices.ndim == 1:
            view_indices = view_indices.unsqueeze(0)
        if view_indices.ndim != 2:
            raise ValueError(f"view_indices must be [B, T] or [T], got {tuple(view_indices.shape)}")
        return view_indices

    def _vae_context_prefix_frames_per_view(self) -> int:
        return int(getattr(self.config, "vae_context_prefix_frames_per_view", 0) or 0)

    def _validate_vae_context_boundary(
        self,
        video: torch.Tensor,
        view_indices: torch.Tensor | None,
    ) -> None:
        context_prefix_frames = self._vae_context_prefix_frames_per_view()
        if context_prefix_frames <= 0:
            return
        if view_indices is None:
            candidate_blocks = [
                (batch_idx, 0, int(video.shape[2]))
                for batch_idx in range(int(video.shape[0]))
                if int(video.shape[2]) > context_prefix_frames
            ]
        else:
            candidate_blocks = []
            for batch_idx in range(int(view_indices.shape[0])):
                for view_id, start, end in self._split_view_blocks(view_indices[batch_idx]):
                    block_frames = int(end - start)
                    if block_frames > context_prefix_frames:
                        candidate_blocks.append((batch_idx, int(view_id), block_frames))
        if not candidate_blocks:
            return
        if (context_prefix_frames - 1) % 4 == 0:
            return

        block_desc = ", ".join(
            f"(batch={batch_idx}, view={view_id}, total_frames={block_frames}, future_tail_frames={block_frames - context_prefix_frames})"
            for batch_idx, view_id, block_frames in candidate_blocks[:8]
        )
        if len(candidate_blocks) > 8:
            block_desc = f"{block_desc}, ..."
        raise ValueError(
            "Unsafe Wan VAE context/future boundary detected before multiview VAE encoding. "
            "Wan VAE encodes raw frame 0 alone and the remaining frames in 4-frame chunks, so "
            "context_prefix_frames_per_view must satisfy (prefix - 1) % 4 == 0 whenever a view block also contains future frames. "
            f"Got context_prefix_frames_per_view={context_prefix_frames}. Offending blocks: {block_desc}"
        )

    def build_latent_view_indices(self, view_indices: torch.Tensor | None) -> torch.Tensor | None:
        view_indices = self._normalize_view_indices(view_indices)
        if view_indices is None:
            return None

        rows: list[torch.Tensor] = []
        expected_len: int | None = None
        for batch_idx in range(int(view_indices.shape[0])):
            sample_chunks: list[torch.Tensor] = []
            for view_id, start, end in self._split_view_blocks(view_indices[batch_idx]):
                latent_len = self._num_latent_frames_for_frames(end - start)
                sample_chunks.append(
                    torch.full((latent_len,), int(view_id), dtype=view_indices.dtype, device=view_indices.device)
                )
            sample = torch.cat(sample_chunks, dim=0) if sample_chunks else torch.zeros((0,), dtype=view_indices.dtype, device=view_indices.device)
            if expected_len is None:
                expected_len = int(sample.shape[0])
            elif int(sample.shape[0]) != expected_len:
                raise ValueError(
                    "All samples in the batch must map to the same latent length for multiview time layout. "
                    f"Got {expected_len} and {int(sample.shape[0])}."
                )
            rows.append(sample)
        return torch.stack(rows, dim=0)

    @torch.inference_mode()
    def encode_multiview_video(
        self,
        video: torch.Tensor,
        view_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        view_indices = self._normalize_view_indices(view_indices)
        if video.ndim != 5:
            raise ValueError(f"video must be [B, C, T, H, W], got {tuple(video.shape)}")
        self._validate_vae_context_boundary(video, view_indices)
        if view_indices is None:
            return self.encode_video(video), None
        if int(video.shape[0]) != int(view_indices.shape[0]):
            raise ValueError(
                "video/view_indices batch mismatch: "
                f"{tuple(video.shape)} vs {tuple(view_indices.shape)}"
            )
        if int(video.shape[2]) != int(view_indices.shape[1]):
            raise ValueError(
                "video/view_indices time mismatch: "
                f"{tuple(video.shape)} vs {tuple(view_indices.shape)}"
            )

        batch_size = int(video.shape[0])
        sample_latent_chunks: list[list[torch.Tensor | None]] = []
        sample_view_chunks: list[list[torch.Tensor | None]] = []
        grouped_chunks: dict[int, list[tuple[int, int, int, int]]] = {}
        for batch_idx in range(batch_size):
            blocks = self._split_view_blocks(view_indices[batch_idx])
            sample_latent_chunks.append([None] * len(blocks))
            sample_view_chunks.append([None] * len(blocks))
            if not blocks:
                continue
            for block_order, (view_id, start, end) in enumerate(blocks):
                grouped_chunks.setdefault(int(end - start), []).append((batch_idx, block_order, int(view_id), start))

        for num_frames, block_specs in grouped_chunks.items():
            batched_video = torch.cat(
                [
                    video[batch_idx : batch_idx + 1, :, start : start + num_frames, :, :]
                    for batch_idx, _, _, start in block_specs
                ],
                dim=0,
            )
            # TODO： 这里是一个bug， 8 hist + 1 current + 8 future 是没问题的。 但是hist一旦增加， current以及history 就有和future混合一起的风险
            batched_latents = self.encode_video(batched_video)
            for block_idx, (batch_idx, block_order, view_id, _) in enumerate(block_specs):
                latent_chunk = batched_latents[block_idx : block_idx + 1]
                sample_latent_chunks[batch_idx][block_order] = latent_chunk
                sample_view_chunks[batch_idx][block_order] = (
                    torch.full(
                        (int(latent_chunk.shape[2]),),
                        int(view_id),
                        dtype=view_indices.dtype,
                        device=view_indices.device,
                    )
                )

        rows: list[torch.Tensor] = []
        latent_view_rows: list[torch.Tensor] = []
        expected_shape: tuple[int, int, int, int] | None = None
        for batch_idx in range(batch_size):
            latent_chunks = sample_latent_chunks[batch_idx]
            latent_view_chunks = sample_view_chunks[batch_idx]
            if latent_chunks:
                if any(chunk is None for chunk in latent_chunks) or any(chunk is None for chunk in latent_view_chunks):
                    raise RuntimeError(f"Incomplete multiview latent assembly for batch index {batch_idx}.")
                sample_latents = torch.cat([chunk for chunk in latent_chunks if chunk is not None], dim=2)
                sample_view_ids = torch.cat([chunk for chunk in latent_view_chunks if chunk is not None], dim=0)
            else:
                sample_latents = self.encode_video(video[batch_idx : batch_idx + 1])
                sample_view_ids = self.build_latent_view_indices(view_indices[batch_idx : batch_idx + 1])[0]
            sample_shape = tuple(int(v) for v in sample_latents.shape[1:])
            if expected_shape is None:
                expected_shape = sample_shape
            elif sample_shape != expected_shape:
                raise ValueError(
                    "All samples in the batch must map to the same latent shape for multiview time layout. "
                    f"Got {expected_shape} and {sample_shape}."
                )
            rows.append(sample_latents.squeeze(0))
            latent_view_rows.append(sample_view_ids)

        return torch.stack(rows, dim=0), torch.stack(latent_view_rows, dim=0)

    @torch.inference_mode()
    def decode_multiview_latents(
        self,
        latents: torch.Tensor,
        view_indices: torch.Tensor | None = None,
        *,
        target_num_frames: int | None = None,
    ) -> torch.Tensor:
        view_indices = self._normalize_view_indices(view_indices)
        if view_indices is None:
            decoded = self.decode_latents(latents)
            if target_num_frames is not None:
                decoded = self._match_num_frames(decoded, int(target_num_frames))
            return decoded
        if latents.ndim != 5:
            raise ValueError(f"latents must be [B, C, T, H, W], got {tuple(latents.shape)}")
        if int(view_indices.shape[0]) == 1 and int(latents.shape[0]) > 1:
            view_indices = view_indices.expand(int(latents.shape[0]), -1)
        if int(latents.shape[0]) != int(view_indices.shape[0]):
            raise ValueError(
                "latents/view_indices batch mismatch: "
                f"{tuple(latents.shape)} vs {tuple(view_indices.shape)}"
            )

        latent_view_indices = self.build_latent_view_indices(view_indices)
        if latent_view_indices is None or int(latent_view_indices.shape[1]) != int(latents.shape[2]):
            raise ValueError(
                "latents temporal length does not match block-aware multiview mapping. "
                f"Got latents={tuple(latents.shape)} and latent_view_indices={None if latent_view_indices is None else tuple(latent_view_indices.shape)}"
            )

        decoded_rows: list[torch.Tensor] = []
        expected_shape: tuple[int, int, int, int] | None = None
        for batch_idx in range(int(latents.shape[0])):
            raw_blocks = self._split_view_blocks(view_indices[batch_idx])
            latent_blocks = self._split_view_blocks(latent_view_indices[batch_idx])
            if len(raw_blocks) != len(latent_blocks):
                raise ValueError(f"Raw/latent block mismatch: {raw_blocks} vs {latent_blocks}")

            decoded_chunks: list[torch.Tensor] = []
            for raw_block, latent_block in zip(raw_blocks, latent_blocks, strict=False):
                raw_view_id, raw_start, raw_end = raw_block
                latent_view_id, latent_start, latent_end = latent_block
                if int(raw_view_id) != int(latent_view_id):
                    raise ValueError(f"View id mismatch between raw and latent blocks: {raw_block} vs {latent_block}")
                latent_chunk = latents[batch_idx : batch_idx + 1, :, latent_start:latent_end, :, :]
                decoded_chunk = self.decode_latents(latent_chunk)
                decoded_chunks.append(self._match_num_frames(decoded_chunk, raw_end - raw_start))

            sample = torch.cat(decoded_chunks, dim=2) if decoded_chunks else self.decode_latents(latents[batch_idx : batch_idx + 1])
            sample_shape = tuple(int(v) for v in sample.shape[1:])
            if expected_shape is None:
                expected_shape = sample_shape
            elif sample_shape != expected_shape:
                raise ValueError(
                    "All samples in the batch must decode to the same frame shape for multiview time layout. "
                    f"Got {expected_shape} and {sample_shape}."
                )
            decoded_rows.append(sample.squeeze(0))

        decoded = torch.stack(decoded_rows, dim=0)
        if target_num_frames is not None:
            decoded = self._match_num_frames(decoded, int(target_num_frames))
        return decoded

    @staticmethod
    def _match_num_frames(frames: torch.Tensor, target_frames: int) -> torch.Tensor:
        if frames.ndim != 5:
            return frames
        cur_frames = int(frames.shape[2])
        if cur_frames == int(target_frames):
            return frames
        if cur_frames < int(target_frames):
            pad = frames[:, :, -1:, :, :].expand(-1, -1, int(target_frames) - cur_frames, -1, -1)
            return torch.cat([frames, pad], dim=2)
        return frames[:, :, : int(target_frames)]

    def _project_text(self, text_emb: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if text_emb is None:
            text_emb = torch.zeros((batch_size, 1, int(self.transformer.config.text_dim)), device=device, dtype=dtype)
        elif text_emb.ndim == 2:
            text_emb = text_emb[:, None, :]
        text_emb = text_emb.to(device=device, dtype=dtype)
        if self.text_proj is not None:
            text_emb = self.text_proj(text_emb)
        return text_emb

    def _apply_view_embedding(self, latents: torch.Tensor, latent_view_indices: torch.Tensor | None) -> torch.Tensor:
        if self.view_embeddings is None or latent_view_indices is None:
            return latents
        if latent_view_indices.ndim != 2:
            raise ValueError(f"latent_view_indices must be [B, T], got {tuple(latent_view_indices.shape)}")
        view_ids = latent_view_indices.to(device=latents.device, dtype=torch.long).clamp(
            min=0, max=int(self.view_embeddings.num_embeddings - 1)
        )
        view_embedding = self.view_embeddings(view_ids)
        view_embedding = view_embedding.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        if self.concat_view_embedding:
            view_embedding = view_embedding.expand(
                int(latents.shape[0]),
                int(view_embedding.shape[1]),
                int(latents.shape[2]),
                int(latents.shape[3]),
                int(latents.shape[4]),
            )
            return torch.cat([latents, view_embedding.to(dtype=latents.dtype)], dim=1)

        if self.view_embedding_proj is not None:
            projected = self.view_embedding_proj(view_embedding.squeeze(-1).squeeze(-1).transpose(1, 2))
            view_embedding = projected.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return latents + view_embedding.to(dtype=latents.dtype)

    def _build_token_view_ids(self, latents: torch.Tensor, latent_view_indices: torch.Tensor | None) -> torch.Tensor | None:
        if latent_view_indices is None:
            return None
        if latent_view_indices.ndim == 1:
            latent_view_indices = latent_view_indices.unsqueeze(0)
        if latent_view_indices.ndim != 2:
            raise ValueError(f"latent_view_indices must be [B, T] or [T], got {tuple(latent_view_indices.shape)}")

        patch_size = tuple(getattr(self.transformer.config, "patch_size", (1, 2, 2)))
        if len(patch_size) != 3:
            raise ValueError(f"Unexpected Wan patch_size={patch_size!r}")
        patch_t, patch_h, patch_w = map(int, patch_size)
        post_patch_num_frames = int(latents.shape[2]) // max(patch_t, 1)
        post_patch_height = int(latents.shape[3]) // max(patch_h, 1)
        post_patch_width = int(latents.shape[4]) // max(patch_w, 1)
        if int(latent_view_indices.shape[1]) != post_patch_num_frames:
            latent_view_indices = self._downsample_view_indices(latent_view_indices, post_patch_num_frames)
        tokens_per_frame = post_patch_height * post_patch_width
        return latent_view_indices.repeat_interleave(tokens_per_frame, dim=1)

    def _forward_transformer_with_cross_view(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        token_view_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, _, num_frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = tuple(getattr(self.transformer.config, "patch_size", (1, 2, 2)))
        post_patch_num_frames = num_frames // int(patch_t)
        post_patch_height = height // int(patch_h)
        post_patch_width = width // int(patch_w)

        rotary_emb = self.transformer.rope(hidden_states)
        hidden_states = self.transformer.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        ts_seq_len = None
        if timestep.ndim == 2:
            ts_seq_len = int(timestep.shape[1])
            timestep = timestep.flatten()

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.transformer.condition_embedder(
            timestep,
            encoder_hidden_states,
            None,
            timestep_seq_len=ts_seq_len,
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if token_view_ids is not None and int(token_view_ids.shape[1]) != int(hidden_states.shape[1]):
            raise ValueError(
                "token_view_ids must match post-patch token length. "
                f"Got token_view_ids={tuple(token_view_ids.shape)}, hidden_states={tuple(hidden_states.shape)}"
            )

        for block_idx, block in enumerate(self.transformer.blocks):
            if torch.is_grad_enabled() and self.transformer.gradient_checkpointing:
                hidden_states = self.transformer._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            hidden_states = self.cross_view_blocks[block_idx](
                hidden_states,
                token_view_ids,
                num_frames=post_patch_num_frames,
                height=post_patch_height,
                width=post_patch_width,
            )

        if temb.ndim == 3:
            shift, scale = (self.transformer.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(
                2, dim=2
            )
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.transformer.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.transformer.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.transformer.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            int(patch_t),
            int(patch_h),
            int(patch_w),
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def _downsample_view_indices(self, view_indices: torch.Tensor | None, latent_frames: int) -> torch.Tensor | None:
        if view_indices is None:
            return None
        if view_indices.ndim == 1:
            view_indices = view_indices.unsqueeze(0)
        if view_indices.ndim != 2:
            raise ValueError(f"view_indices must be [B, T] or [T], got {tuple(view_indices.shape)}")
        block_aware = self.build_latent_view_indices(view_indices)
        if block_aware is not None and int(block_aware.shape[1]) == int(latent_frames):
            return block_aware
        if int(view_indices.shape[1]) == latent_frames:
            return view_indices
        source_idx = torch.linspace(
            0,
            max(int(view_indices.shape[1]) - 1, 0),
            steps=latent_frames,
            device=view_indices.device,
        ).round().long()
        return view_indices.index_select(1, source_idx)

    @torch.no_grad()
    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_videos_per_prompt: int,
        max_sequence_length: int,
    ) -> torch.Tensor:
        self._ensure_inference_modules(load_tokenizer=True, load_text_encoder=True)
        prompt = [prompt] if isinstance(prompt, str) else prompt
        encoded = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device=device) for key, value in encoded.items()}
        prompt_embeds = self.text_encoder(**encoded).last_hidden_state
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        if int(num_videos_per_prompt) > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(int(num_videos_per_prompt), dim=0)
        return prompt_embeds

    @torch.no_grad()
    def encode_prompt(
        self,
        *,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        guidance_scale: float = 5.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = device or self.device
        dtype = dtype or next(self.transformer.parameters()).dtype
        do_classifier_free_guidance = float(guidance_scale) > 1.0

        if prompt is None and prompt_embeds is None:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Only one of `prompt` or `prompt_embeds` can be provided.")

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt,
                device=device,
                dtype=dtype,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if int(num_videos_per_prompt) > 1 and int(prompt_embeds.shape[0]) == 1:
                prompt_embeds = prompt_embeds.repeat_interleave(int(num_videos_per_prompt), dim=0)

        if not do_classifier_free_guidance:
            return prompt_embeds, None

        if negative_prompt_embeds is None:
            if negative_prompt is None:
                negative_prompt = ""
            if isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = int(prompt_embeds.shape[0]) // max(int(num_videos_per_prompt), 1)
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                negative_prompt,
                device=device,
                dtype=dtype,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
            if int(num_videos_per_prompt) > 1 and int(negative_prompt_embeds.shape[0]) == 1:
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(int(num_videos_per_prompt), dim=0)

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def sample_video(
        self,
        *,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        height: int = 224,
        width: int = 224,
        num_frames: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pt",
        max_sequence_length: int = 512,
        attention_kwargs: dict | None = None,
        view_indices: torch.Tensor | None = None,
        cond_latents: torch.Tensor | None = None,
        cond_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> WanVideoGenerationOutput | tuple[torch.Tensor]:
        need_text_encoder = False
        if prompt_embeds is None:
            need_text_encoder = True
        elif float(guidance_scale) > 1.0 and negative_prompt_embeds is None:
            need_text_encoder = True
        self._ensure_inference_modules(load_tokenizer=need_text_encoder, load_text_encoder=need_text_encoder)
        self.eval()

        device = self.device
        transformer_dtype = next(self.transformer.parameters()).dtype
        num_frames = int(num_frames or self.config.video_num_frames)

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            guidance_scale=guidance_scale,
            device=device,
            dtype=transformer_dtype,
        )
        batch_size = int(prompt_embeds.shape[0])

        self.scheduler.set_timesteps(int(num_inference_steps), device=device)
        timesteps = self.scheduler.timesteps

        view_indices = self._normalize_view_indices(view_indices)
        if view_indices is not None:
            view_indices = view_indices.to(device=device)
            if int(view_indices.shape[0]) == 1 and batch_size > 1:
                view_indices = view_indices.expand(batch_size, -1)

        cond_latents_tensor = None
        cond_mask_tensor = None
        if cond_latents is not None:
            cond_latents_tensor = cond_latents.to(device=device, dtype=torch.float32)
            if int(cond_latents_tensor.shape[0]) == 1 and batch_size > 1:
                cond_latents_tensor = cond_latents_tensor.expand(batch_size, -1, -1, -1, -1)
        latent_view_indices = self.build_latent_view_indices(view_indices)
        if latent_view_indices is not None and int(latent_view_indices.shape[0]) == 1 and batch_size > 1:
            latent_view_indices = latent_view_indices.expand(batch_size, -1)

        if latent_view_indices is not None:
            num_latent_frames = int(latent_view_indices.shape[1])
        elif cond_latents_tensor is not None:
            num_latent_frames = int(cond_latents_tensor.shape[2])
        elif latents is not None:
            num_latent_frames = int(latents.shape[2])
        else:
            num_latent_frames = self._num_latent_frames_for_frames(num_frames)

        latent_shape = (
            batch_size,
            self.latent_channels,
            num_latent_frames,
            int(height) // int(self.vae.config.scale_factor_spatial),
            int(width) // int(self.vae.config.scale_factor_spatial),
        )
        if latents is None:
            latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=torch.float32)
        else:
            latents = latents.to(device=device, dtype=torch.float32)

        if cond_mask is not None:
            cond_mask_tensor = cond_mask.to(device=device)
            if cond_mask_tensor.ndim == 1:
                cond_mask_tensor = cond_mask_tensor.unsqueeze(0)
            if cond_mask_tensor.ndim == 2:
                cond_mask_tensor = cond_mask_tensor[:, None, :, None, None]
            if int(cond_mask_tensor.shape[0]) == 1 and batch_size > 1:
                cond_mask_tensor = cond_mask_tensor.expand(batch_size, -1, -1, -1, -1)
            cond_mask_tensor = cond_mask_tensor.to(dtype=torch.float32, device=device)
        if cond_latents_tensor is not None and cond_mask_tensor is not None:
            if tuple(cond_latents_tensor.shape) != tuple(latents.shape):
                raise ValueError(
                    "cond_latents shape must match latents. "
                    f"Got {tuple(cond_latents_tensor.shape)} vs {tuple(latents.shape)}"
                )
            if int(cond_mask_tensor.shape[2]) != int(latents.shape[2]):
                raise ValueError(
                    "cond_mask temporal length must match latents. "
                    f"Got {tuple(cond_mask_tensor.shape)} vs {tuple(latents.shape)}"
                )
            latents = cond_latents_tensor * cond_mask_tensor + latents * (1 - cond_mask_tensor)

        do_classifier_free_guidance = float(guidance_scale) > 1.0 and negative_prompt_embeds is not None

        for timestep in timesteps:
            latent_model_input = latents.to(dtype=transformer_dtype)
            timestep_input = timestep.expand(batch_size).to(device=device, dtype=transformer_dtype)

            cond_ctx = self.transformer.cache_context("cond") if hasattr(self.transformer, "cache_context") else nullcontext()
            with cond_ctx:
                noise_pred = self(
                    latents=latent_model_input,
                    timestep=timestep_input,
                    text_emb=prompt_embeds,
                    latent_view_indices=latent_view_indices,
                    return_dict=True,
                ).pred_flow

            if do_classifier_free_guidance:
                uncond_ctx = self.transformer.cache_context("uncond") if hasattr(self.transformer, "cache_context") else nullcontext()
                with uncond_ctx:
                    noise_uncond = self(
                        latents=latent_model_input,
                        timestep=timestep_input,
                        text_emb=negative_prompt_embeds,
                        latent_view_indices=latent_view_indices,
                        return_dict=True,
                    ).pred_flow
                noise_pred = noise_uncond + float(guidance_scale) * (noise_pred - noise_uncond)

            latents = self.scheduler.step(
                noise_pred.to(dtype=torch.float32),
                timestep,
                latents,
                return_dict=False,
            )[0]
            if cond_latents_tensor is not None and cond_mask_tensor is not None:
                latents = cond_latents_tensor * cond_mask_tensor + latents * (1 - cond_mask_tensor)

        if output_type == "latent":
            frames = latents
        else:
            decoded = self.decode_multiview_latents(
                latents,
                view_indices=view_indices,
                target_num_frames=num_frames,
            )
            if output_type == "raw":
                frames = decoded
            elif output_type in {"pt", "np", "pil"}:
                frames = self.video_processor.postprocess_video(decoded, output_type=output_type)
            else:
                raise ValueError(f"Unsupported output_type={output_type!r}. Use 'latent', 'raw', 'pt', 'np' or 'pil'.")

        output = WanVideoGenerationOutput(
            frames=frames,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        return output if return_dict else (frames,)

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_emb: torch.Tensor | None = None,
        latent_view_indices: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> WanVideoOutput:
        if latents.ndim != 5:
            raise ValueError(f"latents must be [B, C, T, H, W], got {tuple(latents.shape)}")
        if int(latents.shape[1]) != self.latent_channels:
            raise ValueError(f"latent channel mismatch: got {int(latents.shape[1])}, expected {self.latent_channels}")

        batch_size = int(latents.shape[0])
        model_dtype = next(self.transformer.parameters()).dtype
        hidden_states = latents.to(dtype=model_dtype)
        hidden_states = self._apply_view_embedding(hidden_states, latent_view_indices)
        token_view_ids = self._build_token_view_ids(hidden_states, latent_view_indices)
        encoder_hidden_states = self._project_text(
            text_emb,
            batch_size=batch_size,
            device=hidden_states.device,
            dtype=model_dtype,
        )
        timestep = timestep.to(device=hidden_states.device, dtype=model_dtype)

        if self.cross_view_blocks is not None and token_view_ids is not None:
            pred_flow = self._forward_transformer_with_cross_view(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                token_view_ids=token_view_ids.to(device=hidden_states.device, dtype=torch.long),
            )
        else:
            pred_flow = self.transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=True,
            ).sample
        output = WanVideoOutput(
            pred_flow=pred_flow.to(dtype=latents.dtype),
            latents=latents,
            noisy_latents=hidden_states.to(dtype=latents.dtype),
        )
        return output if return_dict else output.to_tuple()


WanVideoModel.register_for_auto_class("AutoModel")

from __future__ import annotations

"""Joint Wan **video + action** MoE (``WanVideoActionMoE``).

This is the **action-side architecture** in StarVLA terms: parallel MoE blocks that
read video keys/values and produce ``pred_action_flow``, while the **video stack**
(``world_model.wan_video_stack_model.WanVideoModel``) stays the perception backbone.
The full ``PreTrainedModel`` lives here so weights and config stay together; WM4A
``WanJointVideoAction`` composes training/inference around it.
"""

from dataclasses import dataclass
from collections import defaultdict
import time

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_wan import (
    FP32LayerNorm,
    WanAttention,
    _get_qkv_projections,
    dispatch_attention_fn,
)
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from starVLA.model.modules.world_model.wan_video_action_semantics import conditioning_video_num_frames
from starVLA.model.modules.world_model.wan_video_stack_config import WanVideoConfig
from starVLA.model.modules.world_model.wan_video_stack_model import WanVideoModel

from .wan_robot_identity import canonicalize_identity, format_unknown_robot_error, get_robot_vocab, is_known_robot_name
from .wan_video_action_config import WanVideoActionConfig


class ActionTokenMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

def _apply_rotary_emb(hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
    if rotary_emb is None:
        return hidden_states
    freqs_cos, freqs_sin = rotary_emb
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def _project_self_attention_qkv(
    attn: WanAttention,
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query, key, value = _get_qkv_projections(attn, hidden_states, None)

    query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
    key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    query = _apply_rotary_emb(query, rotary_emb)
    key = _apply_rotary_emb(key, rotary_emb)
    return query, key, value


def _attention_from_preprojected(
    attn: WanAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    processor = getattr(attn, "processor", None)
    backend = getattr(processor, "_attention_backend", None) if processor is not None else None
    hidden_states = dispatch_attention_fn(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=backend,
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.type_as(query)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


def _build_action_modulation(hidden_size: int, rank: int) -> nn.Sequential:
    rank = int(rank)
    if rank > 0 and rank < hidden_size:
        return nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, rank, bias=False),
            nn.Linear(rank, hidden_size * 3, bias=False),
        )
    return nn.Sequential(
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size * 3, bias=True),
    )


def _zero_init_modulation(modulation: nn.Sequential) -> None:
    linear = modulation[-1]
    if isinstance(linear, nn.Linear):
        nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)


def _init_partial_identity(linear: nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        diag = min(int(linear.out_features), int(linear.in_features))
        if diag > 0:
            eye = torch.eye(diag, device=linear.weight.device, dtype=linear.weight.dtype)
            linear.weight[:diag, :diag].copy_(eye)
        if linear.bias is not None:
            linear.bias.zero_()


def _extract_linear_layers(module: nn.Module) -> list[nn.Linear]:
    if isinstance(module, nn.Linear):
        return [module]

    layers: list[nn.Linear] = []
    for child in module.children():
        layers.extend(_extract_linear_layers(child))
    return layers


class MoEWanBlock(nn.Module):
    """Shared-self-attention Wan block with video-only visibility into action."""

    def __init__(
        self,
        hidden_size: int,
        video_hidden_size: int,
        mlp_ratio: float,
        eps: float,
        modulation_rank: int = 0,
    ):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.hidden_size = int(hidden_size)
        self.video_hidden_size = int(video_hidden_size)

        self.self_norm = FP32LayerNorm(hidden_size, eps, elementwise_affine=False)
        self.self_modulation = _build_action_modulation(hidden_size, modulation_rank)
        self.mlp_norm = FP32LayerNorm(hidden_size, eps, elementwise_affine=False)
        self.mlp_modulation = _build_action_modulation(hidden_size, modulation_rank)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.action_attn_in = nn.Identity()
        self.action_attn_out = nn.Identity()
        if self.hidden_size != self.video_hidden_size:
            self.action_attn_in = nn.Linear(self.hidden_size, self.video_hidden_size, bias=False)
            self.action_attn_out = nn.Linear(self.video_hidden_size, self.hidden_size, bias=False)
            _init_partial_identity(self.action_attn_in)
            _init_partial_identity(self.action_attn_out)

        _zero_init_modulation(self.self_modulation)
        _zero_init_modulation(self.mlp_modulation)
        self._profile_action_enabled = False
        self._profile_action_pending: dict[str, list[object]] = defaultdict(list)

    def set_action_profile_enabled(self, enabled: bool) -> None:
        self._profile_action_enabled = bool(enabled)
        self.reset_action_profile()

    def reset_action_profile(self) -> None:
        self._profile_action_pending.clear()

    def take_action_profile(self) -> dict[str, float]:
        if not self._profile_action_pending:
            return {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stats: dict[str, float] = {}
        for key, pending_items in self._profile_action_pending.items():
            total = 0.0
            for item in pending_items:
                if isinstance(item, tuple):
                    start_event, end_event = item
                    total += float(start_event.elapsed_time(end_event)) / 1000.0
                else:
                    total += float(item)
            stats[key] = float(total)
        self.reset_action_profile()
        return stats

    def _profile_action_start(self, device: torch.device) -> object | None:
        if not self._profile_action_enabled:
            return None
        if device.type == "cuda" and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(device=device))
            return start_event
        return time.perf_counter()

    def _profile_action_end(self, key: str, start_token: object | None, device: torch.device) -> None:
        if start_token is None:
            return
        if device.type == "cuda" and torch.cuda.is_available() and not isinstance(start_token, float):
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream(device=device))
            self._profile_action_pending[key].append((start_token, end_event))
            return
        self._profile_action_pending[key].append(float(time.perf_counter() - float(start_token)))

    @staticmethod
    def _split_video_modulation(
        video_block: nn.Module,
        temb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                video_block.scale_shift_table.unsqueeze(0) + temb
            ).chunk(6, dim=2)
            return (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                c_shift_msa.squeeze(2),
                c_scale_msa.squeeze(2),
                c_gate_msa.squeeze(2),
            )
        return (video_block.scale_shift_table + temb).chunk(6, dim=1)

    def _modulate(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        norm: FP32LayerNorm,
        modulation: nn.Sequential,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mod_dtype = next(modulation.parameters()).dtype
        cond_in = cond.to(dtype=mod_dtype)
        shift, scale, gate = modulation(cond).chunk(3, dim=-1)
        hidden_states = norm(x)
        hidden_states = hidden_states * (1 + scale[:, None, :]) + shift[:, None, :]
        return hidden_states.to(dtype=x.dtype), gate[:, None, :].to(dtype=x.dtype)

    @staticmethod
    def _forward_video_only(
        *,
        video_block: nn.Module,
        transformer: nn.Module | None,
        cross_view_block: nn.Module | None,
        video_num_frames: int,
        video_height: int,
        video_width: int,
        video_tokens: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        video_timestep_proj: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        token_view_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift_msa, scale_msa, _, _, _, _ = MoEWanBlock._split_video_modulation(video_block, video_timestep_proj)
        video_self_norm = (video_block.norm1(video_tokens) * (1 + scale_msa) + shift_msa).type_as(video_tokens)
        _, video_k, video_v = _project_self_attention_qkv(video_block.attn1, video_self_norm, rotary_emb)

        if (
            transformer is not None
            and torch.is_grad_enabled()
            and bool(getattr(transformer, "gradient_checkpointing", False))
        ):
            video_tokens = transformer._gradient_checkpointing_func(
                video_block,
                video_tokens,
                encoder_hidden_states,
                video_timestep_proj,
                rotary_emb,
            )
        else:
            video_tokens = video_block(
                video_tokens,
                encoder_hidden_states,
                video_timestep_proj,
                rotary_emb,
            )

        if cross_view_block is not None and token_view_ids is not None:
            video_tokens = cross_view_block(
                video_tokens,
                token_view_ids,
                num_frames=video_num_frames,
                height=video_height,
                width=video_width,
            )
        return video_tokens, video_k, video_v

    def forward(
        self,
        *,
        video_block: nn.Module,
        transformer: nn.Module | None,
        cross_view_block: nn.Module | None,
        video_num_frames: int,
        video_height: int,
        video_width: int,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        video_timestep_proj: torch.Tensor,
        action_cond: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        token_view_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        profile_device = action_tokens.device
        action_start = self._profile_action_start(profile_device)
        shift_msa, scale_msa, _, _, _, _ = self._split_video_modulation(video_block, video_timestep_proj)
        video_self_norm = (video_block.norm1(video_tokens) * (1 + scale_msa) + shift_msa).type_as(video_tokens)
        self._profile_action_end("video_modulation_prep_sec", action_start, profile_device)

        action_start = self._profile_action_start(profile_device)
        _, video_k, video_v = _project_self_attention_qkv(video_block.attn1, video_self_norm, rotary_emb)
        self._profile_action_end("video_kv_preproj_sec", action_start, profile_device)

        if (
            transformer is not None
            and torch.is_grad_enabled()
            and bool(getattr(transformer, "gradient_checkpointing", False))
        ):
            video_tokens = transformer._gradient_checkpointing_func(
                video_block,
                video_tokens,
                encoder_hidden_states,
                video_timestep_proj,
                rotary_emb,
            )
        else:
            video_tokens = video_block(
                video_tokens,
                encoder_hidden_states,
                video_timestep_proj,
                rotary_emb,
            )

        action_start = self._profile_action_start(profile_device)
        action_norm, action_gate = self._modulate(action_tokens, action_cond, self.self_norm, self.self_modulation)
        self._profile_action_end("action_self_modulation_sec", action_start, profile_device)

        action_start = self._profile_action_start(profile_device)
        action_attn_input = self.action_attn_in(action_norm)
        action_q, action_k, action_v = _project_self_attention_qkv(video_block.attn1, action_attn_input, None)
        self._profile_action_end("action_attn_proj_sec", action_start, profile_device)

        action_start = self._profile_action_start(profile_device)
        action_attn = _attention_from_preprojected(
            video_block.attn1,
            action_q,
            torch.cat([video_k, action_k], dim=1),
            torch.cat([video_v, action_v], dim=1),
        )
        self._profile_action_end("action_attn_core_sec", action_start, profile_device)

        action_start = self._profile_action_start(profile_device)
        action_attn = self.action_attn_out(action_attn).to(dtype=action_tokens.dtype)
        action_tokens = action_tokens + action_gate * action_attn
        self._profile_action_end("action_attn_out_residual_sec", action_start, profile_device)

        if cross_view_block is not None and token_view_ids is not None:
            video_tokens = cross_view_block(
                video_tokens,
                token_view_ids,
                num_frames=video_num_frames,
                height=video_height,
                width=video_width,
            )

        action_start = self._profile_action_start(profile_device)
        action_mlp_norm, action_mlp_gate = self._modulate(action_tokens, action_cond, self.mlp_norm, self.mlp_modulation)
        self._profile_action_end("action_mlp_modulation_sec", action_start, profile_device)

        action_start = self._profile_action_start(profile_device)
        action_tokens = action_tokens + action_mlp_gate * self.mlp(action_mlp_norm)
        self._profile_action_end("action_mlp_core_sec", action_start, profile_device)
        return video_tokens, action_tokens


@dataclass
class WanVideoActionOutput(ModelOutput):
    pred_video_flow: torch.Tensor | None = None
    pred_action_flow: torch.Tensor | None = None
    video_tokens: torch.Tensor | None = None
    action_tokens: torch.Tensor | None = None


class WanVideoActionMoE(PreTrainedModel):
    config_class = WanVideoActionConfig
    main_input_name = "future_noisy_latents"

    def __init__(self, config: WanVideoActionConfig):
        super().__init__(config)
        self.config.vla_model_variant = "moe_parallel"

        video_config = WanVideoConfig(
            wan_model_name_or_path=config.wan_model_name_or_path,
            load_wan_pretrained=config.load_wan_pretrained,
            freeze_vae=config.freeze_vae,
            sample_posterior=config.sample_posterior,
            compile_vae_encode=config.compile_vae_encode,
            transformer_gradient_checkpointing=config.transformer_gradient_checkpointing,
            use_view_embedding=config.use_view_embedding,
            concat_view_embedding=config.concat_view_embedding,
            view_embedding_dim=config.view_embedding_dim,
            max_view_embeddings=config.max_view_embeddings,
            use_cross_view_attention=config.use_cross_view_attention,
            text_embed_dim=config.text_embed_dim,
            video_num_frames=config.video_num_frames,
            vae_context_prefix_frames_per_view=conditioning_video_num_frames(config),
            video_fps=config.video_fps,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            video_snr_shift=config.video_snr_shift,
        )
        self.video_model = WanVideoModel(video_config)
        transformer_cfg = self.video_model.transformer.config
        self.video_hidden_size = int(
            int(getattr(transformer_cfg, "num_attention_heads", 0) or 0)
            * int(getattr(transformer_cfg, "attention_head_dim", 0) or 0)
        )
        self.text_hidden_size = int(getattr(transformer_cfg, "text_dim", self.video_hidden_size) or self.video_hidden_size)
        if self.video_hidden_size <= 0:
            raise ValueError("Cannot infer Wan transformer hidden size.")

        action_hidden = int(config.action_hidden_size or self.video_hidden_size)
        if action_hidden <= 0:
            action_hidden = int(self.video_hidden_size)
        max_action_layers = int(len(self.video_model.transformer.blocks))
        requested_action_layers = int(config.action_num_layers or max_action_layers)
        action_num_layers = min(max(requested_action_layers, 1), max_action_layers)
        eps = float(getattr(transformer_cfg, "eps", 1e-6) or 1e-6)
        self.config.action_hidden_size = action_hidden
        self.config.action_num_heads = int(config.action_num_heads)
        self.config.action_num_layers = action_num_layers
        self.config.action_mlp_ratio = float(config.action_mlp_ratio)
        self.config.action_modulation_rank = int(getattr(config, "action_modulation_rank", 0) or 0)
        self.action_hidden_size = action_hidden
        self.action_video_hidden_size = self.video_hidden_size

        self.action_to_token = ActionTokenMLP(
            in_features=int(config.action_dim),
            hidden_features=action_hidden * 4,
            out_features=action_hidden,
        )
        self.action_timestep_norm = FP32LayerNorm(action_hidden, eps, elementwise_affine=True)
        self.action_t_proj = nn.Identity()
        if action_hidden != self.video_hidden_size:
            self.action_t_proj = nn.Linear(self.video_hidden_size, action_hidden, bias=False)
            _init_partial_identity(self.action_t_proj)
        self.action_out_proj = nn.Linear(action_hidden, int(config.action_dim))
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)

        self.action_pos_emb = nn.Parameter(torch.zeros(int(config.action_pos_emb_max_t), action_hidden))
        nn.init.trunc_normal_(self.action_pos_emb, std=0.02)

        self.action_blocks = nn.ModuleList(
            [
                MoEWanBlock(
                    hidden_size=action_hidden,
                    video_hidden_size=self.video_hidden_size,
                    mlp_ratio=float(config.action_mlp_ratio),
                    eps=eps,
                    modulation_rank=int(getattr(config, "action_modulation_rank", 0) or 0),
                )
                for _ in range(int(action_num_layers))
            ]
        )
        self.action_out_norm = FP32LayerNorm(action_hidden, eps, elementwise_affine=True)
        self.robot_embedding = None
        self._robot_name_to_idx: dict[str, int] = {}
        if bool(getattr(config, "robot_embed_enabled", False)):
            self.ensure_robot_embedding_initialized()
        self.init_action_from_video()
        # Do not call post_init(); it would recurse into Wan submodules and
        # reinitialize pretrained video weights.

    @property
    def device(self) -> torch.device:
        return self.video_model.device

    def encode_video(
        self,
        video: torch.Tensor | None,
        view_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if video is None:
            return None, None
        if video.ndim != 5:
            raise ValueError(f"video must be [B, C, T, H, W], got {tuple(video.shape)}")
        if int(video.shape[2]) == 0:
            return None, None
        return self.video_model.encode_multiview_video(video, view_indices)

    def _action_module_dtype(self) -> torch.dtype:
        return next(self.action_to_token.parameters()).dtype

    def ensure_robot_embedding_initialized(self) -> None:
        vocab = get_robot_vocab()
        self.config.robot_embed_enabled = True
        if self.robot_embedding is not None and int(self.robot_embedding.num_embeddings) == len(vocab):
            self._robot_name_to_idx = {name: idx for idx, name in enumerate(vocab)}
            return
        self.robot_embedding = nn.Embedding(len(vocab), int(self.action_hidden_size))
        nn.init.zeros_(self.robot_embedding.weight)
        self._robot_name_to_idx = {name: idx for idx, name in enumerate(vocab)}

    def _embed_robot(
        self,
        robot_names: list[str] | None,
        *,
        bsz: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.robot_embedding is None:
            raise RuntimeError("robot embedding is not initialized")
        if robot_names is None or len(robot_names) != bsz:
            raise ValueError(f"robot_names batch mismatch: expected {bsz}, got {None if robot_names is None else len(robot_names)}")
        indices: list[int] = []
        for name in robot_names:
            canonical_name = canonicalize_identity(raw_robot_name=name)
            if not is_known_robot_name(canonical_name) or canonical_name not in self._robot_name_to_idx:
                raise ValueError(
                    format_unknown_robot_error(
                        raw_robot_name=name,
                        canonical_name=canonical_name,
                    )
                )
            indices.append(self._robot_name_to_idx[canonical_name])
        index_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        return self.robot_embedding(index_tensor).to(dtype=dtype)

    def _build_action_timestep_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim == 0:
            timestep = timestep.reshape(1, 1)
        elif timestep.ndim == 1:
            timestep = timestep.unsqueeze(1)
        elif timestep.ndim != 2:
            raise ValueError(f"timestep must be [B] or [B, T], got {tuple(timestep.shape)}")

        batch_size, num_steps = int(timestep.shape[0]), int(timestep.shape[1])
        condition_embedder = self.video_model.transformer.condition_embedder
        timestep_proj = condition_embedder.timesteps_proj(timestep.reshape(-1))
        time_embedder_dtype = next(iter(condition_embedder.time_embedder.parameters())).dtype
        if timestep_proj.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep_proj = timestep_proj.to(time_embedder_dtype)
        emb = condition_embedder.time_embedder(timestep_proj)
        emb = emb.reshape(batch_size, num_steps, -1)
        emb = emb.to(device=timestep.device, dtype=self._action_module_dtype())
        emb = self.action_t_proj(emb)
        emb = self.action_timestep_norm(emb)
        return emb

    @staticmethod
    def _init_action_modulation_from_video(
        modulation: nn.Sequential,
        table: torch.Tensor,
    ) -> None:
        linear = modulation[-1]
        if not isinstance(linear, nn.Linear):
            return
        hidden = int(table.shape[-1])
        with torch.no_grad():
            linear.weight.zero_()
            if linear.in_features == hidden and linear.out_features == hidden * 3:
                eye = torch.eye(hidden, device=linear.weight.device, dtype=linear.weight.dtype)
                linear.weight[:hidden].copy_(eye)
                linear.weight[hidden : 2 * hidden].copy_(eye)
                linear.weight[2 * hidden :].copy_(eye)
            linear.bias.copy_(table.reshape(-1).to(device=linear.bias.device, dtype=linear.bias.dtype))

    def init_action_from_video(self) -> None:
        init_scale = 0.25
        with torch.no_grad():
            for video_block, action_block in zip(self.video_model.transformer.blocks, self.action_blocks, strict=False):
                video_ffn = video_block.ffn
                action_layers = [layer for layer in action_block.mlp if isinstance(layer, nn.Linear)]
                video_layers = _extract_linear_layers(video_ffn)
                if len(action_layers) != 2 or len(video_layers) < 2:
                    continue

                action_fc1, action_fc2 = action_layers
                video_fc1 = video_layers[0]
                video_fc2 = video_layers[-1]

                action_fc1.weight.zero_()
                copy_out = min(int(action_fc1.out_features), int(video_fc1.out_features))
                copy_in = min(int(action_fc1.in_features), int(video_fc1.in_features))
                action_fc1.weight[:copy_out, :copy_in].copy_(
                    video_fc1.weight[:copy_out, :copy_in].to(dtype=action_fc1.weight.dtype) * init_scale
                )
                if action_fc1.bias is not None and video_fc1.bias is not None:
                    action_fc1.bias.zero_()
                    action_fc1.bias[:copy_out].copy_(
                        video_fc1.bias[:copy_out].to(dtype=action_fc1.bias.dtype) * init_scale
                    )

                action_fc2.weight.zero_()
                copy_out = min(int(action_fc2.out_features), int(video_fc2.out_features))
                copy_in = min(int(action_fc2.in_features), int(video_fc2.in_features))
                action_fc2.weight[:copy_out, :copy_in].copy_(
                    video_fc2.weight[:copy_out, :copy_in].to(dtype=action_fc2.weight.dtype) * init_scale
                )
                if action_fc2.bias is not None and video_fc2.bias is not None:
                    action_fc2.bias.zero_()
                    action_fc2.bias[:copy_out].copy_(
                        video_fc2.bias[:copy_out].to(dtype=action_fc2.bias.dtype) * init_scale
                    )
                if isinstance(action_block.action_attn_in, nn.Linear):
                    _init_partial_identity(action_block.action_attn_in)
                if isinstance(action_block.action_attn_out, nn.Linear):
                    _init_partial_identity(action_block.action_attn_out)

    def _embed_action_tokens(
        self,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        robot_names: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim != 3:
            raise ValueError(f"actions must be [B, T, D], got {tuple(actions.shape)}")
        steps = int(actions.shape[1])
        if steps > int(self.action_pos_emb.shape[0]):
            raise ValueError(
                f"action steps {steps} exceed action_pos_emb_max_t={int(self.action_pos_emb.shape[0])}"
            )

        action_tokens = self.action_to_token(actions.to(dtype=self._action_module_dtype()))
        action_cond = self._build_action_timestep_embedding(timestep)
        if int(action_cond.shape[1]) == 1:
            action_tokens = action_tokens + action_cond
        elif int(action_cond.shape[1]) == steps:
            action_tokens = action_tokens + action_cond
        else:
            raise ValueError(
                f"Action timestep embedding length {int(action_cond.shape[1])} does not match action steps {steps}"
            )
        action_tokens = action_tokens + self.action_pos_emb[:steps][None, :, :].to(dtype=action_tokens.dtype)
        action_cond_out = action_cond[:, 0, :]
        if self.robot_embedding is not None:
            action_cond_out = action_cond_out + self._embed_robot(
                robot_names,
                bsz=int(actions.shape[0]),
                device=action_tokens.device,
                dtype=action_tokens.dtype,
            )
        return action_tokens, action_cond_out

    def _encode_context_video_frames(
        self,
        video: torch.Tensor,
        view_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if video is None or video.ndim != 5 or int(video.shape[2]) == 0:
            return None, None
        return self.video_model.encode_multiview_video(video, view_indices)

    def _run_joint_backbone(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        actions: torch.Tensor,
        action_timestep: torch.Tensor,
        *,
        text_emb: torch.Tensor | None,
        latent_view_indices: torch.Tensor | None,
        robot_names: list[str] | None = None,
        history_cache: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if latents.ndim != 5:
            raise ValueError(f"latents must be [B, C, T, H, W], got {tuple(latents.shape)}")

        transformer = self.video_model.transformer
        batch_size = int(latents.shape[0])
        model_dtype = next(transformer.parameters()).dtype

        hidden_states = latents.to(dtype=model_dtype)
        hidden_states = self.video_model._apply_view_embedding(hidden_states, latent_view_indices)
        token_view_ids = self.video_model._build_token_view_ids(hidden_states, latent_view_indices)
        encoder_hidden_states = self.video_model._project_text(
            text_emb,
            batch_size=batch_size,
            device=hidden_states.device,
            dtype=model_dtype,
        )
        timestep = timestep.to(device=hidden_states.device, dtype=model_dtype)

        patch_t, patch_h, patch_w = tuple(getattr(transformer.config, "patch_size", (1, 2, 2)))
        post_patch_num_frames = int(hidden_states.shape[2]) // int(patch_t)
        post_patch_height = int(hidden_states.shape[3]) // int(patch_h)
        post_patch_width = int(hidden_states.shape[4]) // int(patch_w)

        rotary_emb = transformer.rope(hidden_states)
        hidden_tokens = transformer.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = transformer.condition_embedder(
            timestep,
            encoder_hidden_states,
            None,
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.cat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        del history_cache
        action_tokens, action_cond = self._embed_action_tokens(actions, action_timestep, robot_names=robot_names)
        cross_view_blocks = self.video_model.cross_view_blocks
        token_view_ids = token_view_ids.to(device=hidden_tokens.device, dtype=torch.long) if token_view_ids is not None else None

        num_action_layers = int(len(self.action_blocks))
        for block_idx, video_block in enumerate(transformer.blocks):
            cross_view_block = None if cross_view_blocks is None else cross_view_blocks[block_idx]

            if block_idx < num_action_layers:
                action_block = self.action_blocks[block_idx]

                def _joint_block_forward(video_states: torch.Tensor, action_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    next_video, next_action = action_block(
                        video_block=video_block,
                        transformer=transformer,
                        cross_view_block=cross_view_block,
                        video_num_frames=post_patch_num_frames,
                        video_height=post_patch_height,
                        video_width=post_patch_width,
                        video_tokens=video_states,
                        action_tokens=action_states,
                        encoder_hidden_states=encoder_hidden_states,
                        video_timestep_proj=timestep_proj,
                        action_cond=action_cond,
                        rotary_emb=rotary_emb,
                        token_view_ids=token_view_ids,
                    )
                    return next_video, next_action

                hidden_tokens, action_tokens = _joint_block_forward(hidden_tokens, action_tokens)
                continue

            hidden_tokens, _, _ = MoEWanBlock._forward_video_only(
                video_block=video_block,
                transformer=transformer,
                cross_view_block=cross_view_block,
                video_num_frames=post_patch_num_frames,
                video_height=post_patch_height,
                video_width=post_patch_width,
                video_tokens=hidden_tokens,
                encoder_hidden_states=encoder_hidden_states,
                video_timestep_proj=timestep_proj,
                rotary_emb=rotary_emb,
                token_view_ids=token_view_ids,
            )

        video_tokens = hidden_tokens
        shift, scale = (transformer.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_tokens = (transformer.norm_out(hidden_tokens) * (1 + scale) + shift).type_as(hidden_tokens)
        hidden_tokens = transformer.proj_out(hidden_tokens)
        hidden_tokens = hidden_tokens.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            int(patch_t),
            int(patch_h),
            int(patch_w),
            -1,
        )
        hidden_tokens = hidden_tokens.permute(0, 7, 1, 4, 2, 5, 3, 6)
        pred_video_flow = hidden_tokens.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        action_tokens = self.action_out_norm(action_tokens)
        pred_action_flow = self.action_out_proj(action_tokens).to(dtype=actions.dtype)
        return video_tokens, pred_video_flow, action_tokens, pred_action_flow

    def forward(
        self,
        future_noisy_latents: torch.Tensor,
        future_timestep: torch.Tensor,
        future_noisy_actions: torch.Tensor,
        action_timestep: torch.Tensor,
        *,
        text_emb: torch.Tensor | None = None,
        future_latent_view_indices: torch.Tensor | None = None,
        history_latents: torch.Tensor | None = None,
        history_latent_view_indices: torch.Tensor | None = None,
        history_actions: torch.Tensor | None = None,
        history_cache: object | None = None,
        robot_names: list[str] | None = None,
        return_dict: bool = True,
    ) -> WanVideoActionOutput:
        del history_latents, history_latent_view_indices, history_actions
        video_tokens, pred_video_flow, action_tokens, pred_action_flow = self._run_joint_backbone(
            future_noisy_latents,
            future_timestep,
            future_noisy_actions,
            action_timestep,
            text_emb=text_emb,
            latent_view_indices=future_latent_view_indices,
            robot_names=robot_names,
            history_cache=history_cache,
        )

        output = WanVideoActionOutput(
            pred_video_flow=pred_video_flow.to(dtype=future_noisy_latents.dtype),
            pred_action_flow=pred_action_flow,
            video_tokens=video_tokens,
            action_tokens=action_tokens,
        )
        return output if return_dict else output.to_tuple()


WanVideoActionMoE.register_for_auto_class("AutoModel")

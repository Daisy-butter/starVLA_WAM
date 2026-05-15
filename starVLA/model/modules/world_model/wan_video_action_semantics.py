from __future__ import annotations

"""Shared video / history layout semantics for Wan **video stack** and joint action configs.

Lives under ``world_model`` because it defines how **latent video** is structured for
``WanVideoConfig`` / ``WanVideoActionConfig`` (conditioning frames, sparse history, etc.).
"""

from typing import Any, Mapping


def _read_value(source: Any, name: str, default=None):
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _read_int(source: Any, name: str, default: int | None = None) -> int | None:
    value = _read_value(source, name, default)
    if value is None:
        return None
    return int(value)


def current_video_num_frames(source: Any) -> int:
    raw_current_frames = _read_int(source, "current_video_num_frames", None)
    if raw_current_frames is not None and raw_current_frames != 1:
        raise ValueError(
            "current_video_num_frames is no longer configurable. "
            f"Expected current_video_num_frames=1, got {raw_current_frames}."
        )
    return 1


def sparse_history_video_num_frames(source: Any) -> int:
    raw_sparse_frames = _read_int(source, "sparse_history_video_num_frames", 0)
    if raw_sparse_frames is None:
        raw_sparse_frames = 0
    if raw_sparse_frames is None:
        raw_sparse_frames = 0
    if raw_sparse_frames < 0:
        raise ValueError(f"sparse_history_video_num_frames must be non-negative, got {raw_sparse_frames}")
    return int(raw_sparse_frames)


def conditioning_video_num_frames(source: Any) -> int:
    return int(current_video_num_frames(source)) + int(sparse_history_video_num_frames(source))


def sparse_history_fps(source: Any) -> float:
    value = _read_value(source, "sparse_history_fps", 1.0)
    return float(value or 0.0)


def sparse_history_simulate_short_history_prob(source: Any) -> float:
    value = _read_value(source, "sparse_history_simulate_short_history_prob", 0.0)
    return float(value or 0.0)


def fastwam_anchor_frames(source: Any) -> int:
    return int(_read_value(source, "fastwam_anchor_frames", 1) or 1)


def vae_context_prefix_frames_per_view(source: Any) -> int:
    variant = str(_read_value(source, "vla_model_variant", "moe_parallel") or "moe_parallel").strip().lower()
    if variant == "fastwam_parallel":
        return fastwam_anchor_frames(source)
    return conditioning_video_num_frames(source)


def uses_sparse_history_context(source: Any) -> bool:
    variant = str(_read_value(source, "vla_model_variant", "moe_parallel") or "moe_parallel").strip().lower()
    if variant == "fastwam_parallel":
        return False
    # MoE-style variants always feed `sparse_history + current` into the video
    # backbone. When sparse history is disabled, this prefix still contains the
    # single current frame.
    return int(conditioning_video_num_frames(source)) > 0


def canonicalize_video_action_semantics(source: Any) -> dict[str, int]:
    variant = str(_read_value(source, "vla_model_variant", "moe_parallel") or "moe_parallel").strip().lower()
    current_frames = int(current_video_num_frames(source))
    sparse_frames = int(sparse_history_video_num_frames(source))
    conditioning_frames = int(conditioning_video_num_frames(source))

    raw_history_frames = _read_int(source, "history_video_num_frames", None)
    if raw_history_frames not in (None, 1):
        raise ValueError(
            "Dense history is no longer supported. "
            f"Expected history_video_num_frames=1, got {raw_history_frames}."
        )

    raw_past_frames = _read_int(source, "past_video_num_frames", None)
    if raw_past_frames not in (None, 0):
        raise ValueError(
            "past_video_num_frames is no longer supported. "
            f"Expected past_video_num_frames=0, got {raw_past_frames}."
        )

    if variant == "fastwam_parallel":
        if sparse_frames != 0:
            raise ValueError(
                "fastwam_parallel does not support sparse history. "
                f"Expected sparse_history_video_num_frames=0, got {sparse_frames}."
            )
        if int(fastwam_anchor_frames(source)) != 1:
            raise ValueError(
                "fastwam_parallel only supports fastwam_anchor_frames=1 in the simplified semantics."
            )

    return {
        "current_video_num_frames": current_frames,
        "sparse_history_video_num_frames": sparse_frames,
        "conditioning_video_num_frames": conditioning_frames,
    }

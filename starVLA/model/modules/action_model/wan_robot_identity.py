"""Minimal robot identity helpers for Wan video-action models.

The original training stack imported these from an external ``dataset`` package.
starVLA uses this module for ``robot_embed_enabled`` on the Wan joint model.
"""

from __future__ import annotations

# Single default slot; extend for multi-robot training with robot_embed_enabled=True.
ROBOT_VOCAB: tuple[str, ...] = ("default",)


def get_robot_vocab() -> list[str]:
    return list(ROBOT_VOCAB)


def canonicalize_identity(*, raw_robot_name: str) -> str:
    name = str(raw_robot_name or "").strip().lower()
    return name or "default"


def is_known_robot_name(canonical_name: str) -> bool:
    return canonical_name in set(ROBOT_VOCAB)


def format_unknown_robot_error(*, raw_robot_name: str, canonical_name: str) -> str:
    return (
        f"Unknown robot name {raw_robot_name!r} (canonical={canonical_name!r}). "
        f"Known robots: {list(ROBOT_VOCAB)}. "
        "Either disable robot_embed_enabled or add the robot to "
        "``starVLA.model.modules.action_model.wan_robot_identity.ROBOT_VOCAB``."
    )

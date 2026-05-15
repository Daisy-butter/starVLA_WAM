#!/usr/bin/env python3
"""
Convert extracted CALVIN task_D_D (or compatible layout) to LeRobot v2.1-style
layout expected by StarVLA's gr00t_lerobot loader (image columns in parquet, total_videos=0).

Layout under --calvin-root (e.g. .../task_D_D after unzip):
  training/episode_XXXXXXX.npz, training/lang_annotations/auto_lang_ann.npy
  validation/...

Usage:
  python examples/calvin/scripts/convert_calvin_task_dd_to_lerobot.py \\
    --calvin-root /path/to/task_D_D \\
    --output-dir /path/to/parent/task_D_D_lerobot

Requires: numpy, pandas, pyarrow (already in StarVLA requirements).
"""

from __future__ import annotations

import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tyro
from PIL import Image


def _encode_image_png(arr: np.ndarray) -> dict:
    """HuggingFace-style image dict for parquet + StarVLA image decoder."""
    im = Image.fromarray(np.asarray(arr, dtype=np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return {"bytes": buf.getvalue()}


@dataclass(frozen=True)
class Args:
    calvin_root: Path
    """Path to extracted `task_D_D` (contains `training/` and usually `validation/`)."""

    output_dir: Path
    """Directory to create for the LeRobot dataset (e.g. .../task_D_D_lerobot)."""

    splits: Literal["training", "validation", "both"] = "training"
    """Which splits to export (StarVLA training typically uses `training` only)."""

    action_key: Literal["rel_actions", "actions"] = "rel_actions"
    """Which action field to read from CALVIN `.npz` files."""

    fps: int = 10
    chunk_size: int = 1000
    """Must match `chunks_size` in meta/info.json (episode parquet path layout)."""

    max_episodes: int | None = None
    """If set, stop after this many episodes **per split** (debug)."""


def _load_lang_pack(root: Path, split: str) -> dict:
    p = root / split / "lang_annotations" / "auto_lang_ann.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing language pack: {p}")
    return np.load(p, allow_pickle=True).item()


def _load_step_npz(root: Path, split: str, step_id: int) -> dict[str, np.ndarray]:
    name = root / split / f"episode_{step_id:07d}.npz"
    if not name.exists():
        raise FileNotFoundError(f"Missing step file: {name}")
    with np.load(name, allow_pickle=True) as npz:
        return {k: np.asarray(npz[k]) for k in npz.files}


def _collect_task_vocabulary(calvin_root: Path, splits: list[str]) -> tuple[list[str], dict[str, int]]:
    """Assign task_index in first-seen order across splits."""
    seen_order: list[str] = []
    seen_set: set[str] = set()
    for split in splits:
        pack = _load_lang_pack(calvin_root, split)
        for t in pack["language"]["task"]:
            ts = str(t)
            if ts not in seen_set:
                seen_set.add(ts)
                seen_order.append(ts)
    str_to_idx = {s: i for i, s in enumerate(seen_order)}
    return seen_order, str_to_idx


def main(args: Args):
    calvin_root = args.calvin_root.resolve()
    out = args.output_dir.resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    (out / "data").mkdir(parents=True)

    splits: list[str] = ["training", "validation"] if args.splits == "both" else [args.splits]

    task_names, str_to_idx = _collect_task_vocabulary(calvin_root, splits)
    task_rows = [{"task_index": i, "task": name} for i, name in enumerate(task_names)]

    global_episode_index = 0
    global_frame_index = 0
    episodes_rows: list[dict] = []

    for split in splits:
        lang_pack = _load_lang_pack(calvin_root, split)
        tasks = lang_pack["language"]["task"]
        lang_ann = lang_pack["language"]["ann"]
        ep_bounds = lang_pack["info"]["indx"]

        n_eps = len(ep_bounds)
        if args.max_episodes is not None:
            n_eps = min(n_eps, args.max_episodes)

        for i in range(n_eps):
            start_idx, end_idx = int(ep_bounds[i][0]), int(ep_bounds[i][1])
            task_index = str_to_idx[str(tasks[i])]

            rows = []
            for si, step_id in enumerate(range(start_idx, end_idx + 1)):
                step = _load_step_npz(calvin_root, split, step_id)
                rgb_static = np.asarray(step["rgb_static"], dtype=np.uint8)
                rgb_gripper = np.asarray(step["rgb_gripper"], dtype=np.uint8)
                robot_obs = np.asarray(step["robot_obs"], dtype=np.float32).reshape(-1)
                act = np.asarray(step[args.action_key], dtype=np.float32).reshape(-1)
                rows.append(
                    {
                        "image": _encode_image_png(rgb_static),
                        "wrist_image": _encode_image_png(rgb_gripper),
                        "state": robot_obs.tolist(),
                        "actions": act.tolist(),
                        "task_index": task_index,
                        "timestamp": float(si) / float(args.fps),
                        "frame_index": si,
                        "episode_index": global_episode_index,
                        "index": global_frame_index,
                    }
                )
                global_frame_index += 1

            ep_len = len(rows)
            chunk = global_episode_index // args.chunk_size
            chunk_dir = out / "data" / f"chunk-{chunk:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(rows)
            pq_path = chunk_dir / f"episode_{global_episode_index:06d}.parquet"
            df.to_parquet(pq_path, index=False)

            episodes_rows.append({"episode_index": global_episode_index, "length": ep_len})
            global_episode_index += 1

    with (out / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for row in task_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_tasks = len(task_rows)
    total_chunks = max(1, (global_episode_index + args.chunk_size - 1) // args.chunk_size)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": global_episode_index,
        "total_frames": global_frame_index,
        "total_tasks": total_tasks,
        "total_videos": 0,
        "total_chunks": total_chunks,
        "chunks_size": args.chunk_size,
        "fps": args.fps,
        "splits": {"train": f"0:{global_episode_index}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "image": {
                "dtype": "image",
                "shape": [200, 200, 3],
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": [84, 84, 3],
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": [15], "names": ["state"]},
            "actions": {"dtype": "float32", "shape": [7], "names": ["actions"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with (out / "meta" / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"Wrote LeRobot dataset to {out} ({global_episode_index} episodes, {global_frame_index} frames).")


if __name__ == "__main__":
    main(tyro.cli(Args))

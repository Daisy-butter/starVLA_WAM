"""
In-process LIBERO evaluation for Wan joint video-action MoE (HF checkpoint).

Skips the websocket policy server — loads ``WanJointMoEPolicy.from_hf_dir`` in the
same process as the LIBERO simulator.

Default paths (this cluster):
  - HF ckpt: /SSD_DISK/users/wuruihan/sii_starvla_ckpt/ckpt_debug_libero
  - Wan base: /SSD_DISK/users/wuruihan/sii_starvla_ckpt/hugg_model/Wan2.2-TI2V-5B-Diffusers

Usage (single terminal, LIBERO conda env with starVLA on PYTHONPATH):

    cd /home/wuruihan/starVLA
    export PYTHONPATH=$PWD
    export LIBERO_HOME=/path/to/LIBERO
    export LIBERO_CONFIG_PATH=$LIBERO_HOME/libero
    export PYTHONPATH=$PYTHONPATH:$LIBERO_HOME
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl

    CUDA_VISIBLE_DEVICES=0 python examples/LIBERO/eval_files/eval_libero_wan_joint_local.py \
      --stats-json /path/to/dataset_statistics.json \
      --task-suite libero_goal \
      --num-trials 5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import pathlib
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# LIBERO pickles numpy arrays; PyTorch 2.6+ defaults weights_only=True.
_original_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_libero_wan_joint_local")


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _binarize_gripper_open(open_val) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def load_action_norm_stats(stats_json: str, unnorm_key: str) -> Dict[str, np.ndarray]:
    with open(stats_json, "r", encoding="utf-8") as f:
        all_stats = json.load(f)
    if unnorm_key not in all_stats:
        raise KeyError(
            f"unnorm_key={unnorm_key!r} not in {stats_json}. "
            f"Available keys: {sorted(all_stats.keys())}"
        )
    action_stats = all_stats[unnorm_key]["action"]
    return {
        "min": np.asarray(action_stats["min"], dtype=np.float32),
        "max": np.asarray(action_stats["max"], dtype=np.float32),
        "mask": np.asarray(
            action_stats.get("mask", np.ones_like(action_stats["min"], dtype=bool)),
            dtype=bool,
        ),
    }


def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
    action_high = np.asarray(action_norm_stats["max"])
    action_low = np.asarray(action_norm_stats["min"])
    normalized_actions = np.clip(normalized_actions, -1, 1)
    if normalized_actions.shape[-1] >= 7:
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


def get_max_steps(task_suite: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[task_suite]


@dataclasses.dataclass
class EvalArgs:
    ckpt_dir: str
    wan_root: str
    stats_json: Optional[str]
    task_suite: str = "libero_goal"
    num_trials: int = 5
    num_steps_wait: int = 10
    seed: int = 7
    video_out: Optional[str] = None
    unnorm_key: str = "franka"
    obs_height: int = 480
    obs_width: int = 832
    num_inference_timesteps: int = 10
    max_tasks: int = -1
    load_strict: bool = True


def run(args: EvalArgs) -> None:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from starVLA.model.modules.action_model.wan_video_action_moe import WanJointMoEPolicy

    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks
    n_eval_tasks = num_tasks if args.max_tasks <= 0 else min(args.max_tasks, num_tasks)
    max_steps = get_max_steps(args.task_suite)

    log.info(
        "task_suite=%s tasks=%d/%d max_steps=%d trials/task=%d",
        args.task_suite,
        n_eval_tasks,
        num_tasks,
        max_steps,
        args.num_trials,
    )
    log.info("ckpt_dir=%s", args.ckpt_dir)
    log.info("wan_root=%s", args.wan_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    policy = WanJointMoEPolicy.from_hf_dir(
        args.ckpt_dir,
        wan_root=args.wan_root,
        device=device,
        obs_height=args.obs_height,
        obs_width=args.obs_width,
        num_inference_timesteps=args.num_inference_timesteps,
        strict=args.load_strict,
    )
    log.info(
        "policy loaded in %.1fs on %s (action_horizon=%d action_dim=%d)",
        time.time() - t0,
        device,
        policy.action_horizon,
        policy.action_dim,
    )

    norm_stats: Optional[Dict[str, np.ndarray]] = None
    if args.stats_json:
        norm_stats = load_action_norm_stats(args.stats_json, args.unnorm_key)
        log.info("loaded action stats from %s (key=%s)", args.stats_json, args.unnorm_key)
    else:
        log.warning(
            "No --stats-json provided: actions will stay in normalized space "
            "(LIBERO success rate will not be meaningful)."
        )

    chunk_size = policy.action_horizon
    if args.video_out:
        pathlib.Path(args.video_out).mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_successes = 0
    per_task_results: Dict[str, Dict[str, int]] = {}

    for task_id in range(n_eval_tasks):
        task = task_suite.get_task(task_id)
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)

        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)

        task_episodes = 0
        task_successes = 0
        log.info("[%d/%d] %s", task_id + 1, n_eval_tasks, task_description)

        for ep_idx in range(args.num_trials):
            env.reset()
            obs = env.set_init_state(initial_states[ep_idx])

            t = 0
            step = 0
            done = False
            replay_imgs: List[np.ndarray] = []
            cached_actions: Optional[np.ndarray] = None

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                replay_imgs.append(img)

                if step % chunk_size == 0:
                    example = {
                        "image": [
                            Image.fromarray(img.astype(np.uint8)),
                            Image.fromarray(wrist_img.astype(np.uint8)),
                        ],
                        "lang": str(task_description),
                    }
                    with torch.no_grad():
                        out = policy.predict_action([example])
                    normed = out["normalized_actions"][0]
                    cached_actions = unnormalize_actions(normed, norm_stats) if norm_stats is not None else normed

                cur_action = cached_actions[step % chunk_size]
                wv = cur_action[:3]
                rot = cur_action[3:6]
                grip = _binarize_gripper_open(cur_action[6:7])
                action7 = np.concatenate([wv, rot, grip], axis=0)
                obs, _, done, _ = env.step(action7.tolist())

                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1
            log.info(
                "  ep%d %s task_sr=%d/%d total_sr=%d/%d (%.1f%%)",
                ep_idx,
                "SUCCESS" if done else "fail",
                task_successes,
                task_episodes,
                total_successes,
                total_episodes,
                100.0 * total_successes / max(total_episodes, 1),
            )

            if args.video_out and replay_imgs:
                import imageio

                tag = "success" if done else "failure"
                fname = pathlib.Path(args.video_out) / f"task{task_id}_ep{ep_idx}_{tag}.mp4"
                imageio.mimwrite(str(fname), replay_imgs, fps=10)

        per_task_results[task_description] = {"success": task_successes, "total": task_episodes}
        log.info(
            "[task %d] FINAL sr=%d/%d (%.1f%%)",
            task_id + 1,
            task_successes,
            task_episodes,
            100.0 * task_successes / max(task_episodes, 1),
        )
        env.close()

    log.info("=" * 60)
    log.info(
        "FINAL TOTAL SR: %d/%d (%.1f%%)",
        total_successes,
        total_episodes,
        100.0 * total_successes / max(total_episodes, 1),
    )
    for k, v in per_task_results.items():
        sr = 100.0 * v["success"] / v["total"] if v["total"] else 0.0
        log.info("  %5.1f%%  %3d/%3d  %s", sr, v["success"], v["total"], k)


def main() -> None:
    from starVLA.model.modules.action_model.wan_video_action_moe import DEFAULT_HF_CKPT_DIR, DEFAULT_WAN_ROOT

    parser = argparse.ArgumentParser(description="Local LIBERO eval for Wan joint MoE (HF checkpoint).")
    parser.add_argument("--ckpt-dir", default=DEFAULT_HF_CKPT_DIR, help="HF checkpoint directory")
    parser.add_argument("--wan-root", default=DEFAULT_WAN_ROOT, help="Wan2.2-TI2V diffusers root")
    parser.add_argument(
        "--stats-json",
        default=None,
        help="dataset_statistics.json for action un-normalization (required for real metrics)",
    )
    parser.add_argument(
        "--task-suite",
        default="libero_goal",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-out", default=None)
    parser.add_argument("--unnorm-key", default="franka")
    parser.add_argument("--obs-height", type=int, default=480)
    parser.add_argument("--obs-width", type=int, default=832)
    parser.add_argument("--num-inference-timesteps", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=-1, help="Smoke test: cap number of tasks (-1 = all)")
    parser.add_argument(
        "--load-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strict state_dict load (disable to tolerate minor key drift)",
    )
    cli = parser.parse_args()

    run(
        EvalArgs(
            ckpt_dir=cli.ckpt_dir,
            wan_root=cli.wan_root,
            stats_json=cli.stats_json,
            task_suite=cli.task_suite,
            num_trials=cli.num_trials,
            num_steps_wait=cli.num_steps_wait,
            seed=cli.seed,
            video_out=cli.video_out,
            unnorm_key=cli.unnorm_key,
            obs_height=cli.obs_height,
            obs_width=cli.obs_width,
            num_inference_timesteps=cli.num_inference_timesteps,
            max_tasks=cli.max_tasks,
            load_strict=cli.load_strict,
        )
    )


if __name__ == "__main__":
    if os.getenv("DEBUG", False):
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        debugpy.wait_for_client()
    main()

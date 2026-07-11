#!/usr/bin/env python3
"""
Collect (seed, instruction) manifests for RoboTwin tasks.

For tasks with seeds in eval_seeds.json, the first N success seeds are used as
the fixed seed pool. For tasks without seeds, sequential seeds starting from
--start_seed are generated.

Usage (inside container):
    python scripts/collect_robotwin_manifests.py \
        --tasks adjust_bottle beat_block_hammer ... \
        --n_episodes 96 \
        --output_dir rlinf/envs/robotwin/seeds
"""
import argparse
import json
import os
import sys
from pathlib import Path


TASKS_20 = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot", "move_can_pot",
    "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad", "open_microwave",
    "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left", "place_a2b_right",
]


def get_seeds_for_task(task_name, n_episodes, eval_seeds_data, start_seed, task_index):
    """Return a list of exactly n_episodes seeds for a task."""
    if task_name in eval_seeds_data:
        success_seeds = eval_seeds_data[task_name].get("success_seeds", [])
        if success_seeds:
            seeds = []
            while len(seeds) < n_episodes:
                seeds.extend(success_seeds)
            return seeds[:n_episodes]

    # No seeds available — generate sequential seeds unique to this task
    offset = task_index * n_episodes
    return list(range(start_seed + offset, start_seed + offset + n_episodes))


def collect_task_manifest(task_name, seeds, assets_path, n_envs=8):
    """Reset RoboTwin envs with given seeds and capture (seed, instruction) pairs."""
    from robotwin.envs.vector_env import VectorEnv

    task_config = {
        "task_name": task_name,
        "embodiment": ["aloha-agilex"],
        "save_path": "/tmp/collect_manifests_scratch",
        "step_lim": 200,
        "episode_num": 100,
        "use_seed": False,
        "save_freq": 15,
        "language_num": 100,
        "collect_data": False,
        "eval_video_log": False,
        "clear_cache_freq": 1,
        "data_type": {
            "rgb": True,
            "third_view": False,
            "depth": False,
            "pointcloud": False,
            "observer": False,
            "endpose": False,
            "qpos": True,
            "mesh_segmentation": False,
            "actor_segmentation": False,
        },
        "camera": {
            "head_camera_type": "D435",
            "wrist_camera_type": "D435",
            "collect_head_camera": True,
            "collect_wrist_camera": False,
        },
        "domain_randomization": {
            "random_background": False,
            "cluttered_table": False,
            "clean_background_rate": 1,
            "random_head_camera_dis": 0,
            "random_table_height": 0,
            "random_light": False,
            "crazy_random_light_rate": 0,
        },
    }

    n_total = len(seeds)
    assert n_total % n_envs == 0, f"n_episodes ({n_total}) must be divisible by n_envs ({n_envs})"

    episodes = []
    batches = [seeds[i : i + n_envs] for i in range(0, n_total, n_envs)]

    # VectorEnv constructor only runs setup_task (no setup_demo yet);
    # must call reset() first so _init_task_env_ runs and sets task attributes.
    venv = VectorEnv(
        task_config=task_config,
        n_envs=n_envs,
        env_seeds=batches[0],
        instructions=None,
        instruction_type="seen",
        strict_episode_manifest=False,
    )

    for batch in batches:
        venv.reset(env_idx=None, env_seeds=batch, instructions=None)
        raw_obs = venv.get_obs()
        for seed, obs in zip(batch, raw_obs):
            episodes.append({"seed": int(seed), "instruction": str(obs["instruction"])})

    venv.close(clear_cache=True)
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Collect seed+instruction manifests for RoboTwin tasks")
    parser.add_argument("--tasks", nargs="+", default=TASKS_20,
                        help="Task names to collect (default: all 20)")
    parser.add_argument("--n_episodes", type=int, default=96,
                        help="Episodes per task (must be divisible by 8)")
    parser.add_argument("--start_seed", type=int, default=100001,
                        help="Base seed for tasks without eval_seeds.json entries")
    parser.add_argument("--output_dir", default="rlinf/envs/robotwin/seeds",
                        help="Directory to save manifest JSON files")
    parser.add_argument("--eval_seeds_path",
                        default="rlinf/envs/robotwin/seeds/eval_seeds.json",
                        help="Path to eval_seeds.json for tasks with known good seeds")
    parser.add_argument("--assets_path",
                        default="/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf",
                        help="RoboTwin assets path")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing manifest files")
    args = parser.parse_args()

    assert args.n_episodes % 8 == 0, "--n_episodes must be divisible by 8 (= n_envs)"

    os.environ.setdefault("ASSETS_PATH", args.assets_path)
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    eval_seeds_data = {}
    if os.path.exists(args.eval_seeds_path):
        with open(args.eval_seeds_path) as f:
            eval_seeds_data = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_index, task_name in enumerate(args.tasks):
        out_path = output_dir / f"{task_name}_manifest_{args.n_episodes}ep.json"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {task_name}: {out_path} already exists")
            continue

        seeds = get_seeds_for_task(
            task_name, args.n_episodes, eval_seeds_data, args.start_seed, task_index
        )
        seed_source = "eval_seeds.json" if task_name in eval_seeds_data else f"range({seeds[0]}..{seeds[-1]})"
        print(f"\n[{task_index+1}/{len(args.tasks)}] {task_name}  seeds: {seed_source}")

        episodes = collect_task_manifest(task_name, seeds, args.assets_path, n_envs=8)

        manifest = {task_name: {"episodes": episodes}}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"  -> saved {len(episodes)} episodes to {out_path}")

    print("\nAll tasks done.")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()

"""
Generate train seeds and eval manifest for a RoboTwin task using the expert planner.

Usage:
    python tools/generate_seeds.py --task <task_name> [options]

Train seeds  : expert-verified, range [0, 100000),  written to train_seeds.json
Eval manifest: expert-verified, range [100000, 300000), written to <task>_manifest_96ep.json
"""

import argparse
import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REPO_PATH = Path(os.environ.get("REPO_PATH", "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin"))
ASSETS_PATH = os.environ.get(
    "ASSETS_PATH",
    os.environ.get("ROBOTWIN_PATH", "/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf"),
)

TRAIN_SEEDS_JSON = REPO_PATH / "rlinf/envs/robotwin/seeds/train_seeds.json"
SEEDS_DIR = REPO_PATH / "rlinf/envs/robotwin/seeds"

TRAIN_SEED_START = 0
TRAIN_SEED_END = 100_000
EVAL_SEED_START = 100_000
EVAL_SEED_END = 300_000

TRAIN_TARGET = 1000
EVAL_TARGET = 96

# Single env: no global_lock contention, no batch timeout
N_ENVS = 1
PER_SEED_TIMEOUT = 180  # seconds; curobo should finish well within this


def build_task_config(task_name: str) -> dict:
    return {
        "task_name": task_name,
        "embodiment": ["aloha-agilex"],
        "save_path": "/tmp/seed_check_tmp",
        "step_lim": 400,
        "planner_backend": "curobo",
        "clear_cache_freq": 8,
        "language_num": 100,
        "render_freq": 0,
        "eval_video_log": False,
        "data_type": {
            "rgb": False,
            "depth": False,
            "endpose": False,
            "qpos": False,
            "mesh_segmentation": False,
            "actor_segmentation": False,
            "third_view": False,
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
        "camera": {
            "head_camera_type": "D435",
            "wrist_camera_type": "D435",
            "collect_head_camera": True,
            "collect_wrist_camera": True,
        },
    }


def check_one_seed(subenv, seed: int, timeout: float = PER_SEED_TIMEOUT) -> dict:
    """Run check_seed on a single SubEnv with per-seed timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(subenv.check_seed, seed)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            log.warning(f"Seed {seed} timed out after {timeout}s, skipping.")
            return {"setup_demo_success": False, "play_once_success": False, "cost_time": timeout}
        except Exception as e:
            log.warning(f"Seed {seed} raised: {e}")
            return {"setup_demo_success": False, "play_once_success": False, "cost_time": 0}


def get_instruction(subenv) -> str:
    try:
        return subenv.create_instruction() or ""
    except Exception:
        try:
            return subenv.task.get_instruction() or ""
        except Exception:
            return ""


CHECKPOINT_INTERVAL = 50  # write partial results every N successes


def collect_seeds(
    task_name: str,
    seed_start: int,
    seed_end: int,
    target: int,
    label: str,
    checkpoint_fn=None,
) -> list[dict]:
    """Run expert planner over [seed_start, seed_end) and collect up to `target` successes."""
    os.environ["ASSETS_PATH"] = ASSETS_PATH

    from robotwin.envs.vector_env import VectorEnv

    task_config = build_task_config(task_name)

    log.info(f"[{label}] Initializing VectorEnv for task={task_name} n_envs={N_ENVS}")
    venv = VectorEnv(
        task_config=task_config,
        n_envs=N_ENVS,
        env_seeds=[seed_start],
        instruction_type="seen",
    )
    subenv = venv.envs[0]
    log.info(f"[{label}] VectorEnv ready. Checking seeds {seed_start}..{seed_end}, target={target}")

    successes = []
    checked = 0
    last_checkpoint = 0
    t0 = time.time()

    for seed in range(seed_start, seed_end):
        if len(successes) >= target:
            break

        result = check_one_seed(subenv, seed)
        checked += 1

        if result.get("play_once_success", False):
            instruction = get_instruction(subenv)
            successes.append({"seed": seed, "instruction": instruction})

            # Incremental checkpoint: write partial results periodically
            if checkpoint_fn and len(successes) - last_checkpoint >= CHECKPOINT_INTERVAL:
                checkpoint_fn(successes)
                last_checkpoint = len(successes)
                log.info(f"[{label}] Checkpoint: {len(successes)} seeds saved.")

        if checked % 10 == 0 or result.get("play_once_success", False):
            elapsed = time.time() - t0
            rate = checked / elapsed if elapsed > 0 else 0
            eta = (target - len(successes)) / (len(successes) / elapsed) if len(successes) > 0 else float("inf")
            log.info(
                f"[{label}] checked={checked} success={len(successes)}/{target} "
                f"rate={rate:.2f}/s cost={result.get('cost_time', 0):.1f}s "
                f"eta={eta/60:.1f}min seed={seed}"
            )

    venv.close()
    # Final write of any remaining since last checkpoint
    if checkpoint_fn and len(successes) > last_checkpoint:
        checkpoint_fn(successes)
        log.info(f"[{label}] Final checkpoint: {len(successes)} seeds saved.")
    log.info(f"[{label}] Done. {len(successes)} success seeds from {checked} checked.")
    return successes[:target]


def write_train_seeds(task_name: str, seeds: list[dict]):
    with open(TRAIN_SEEDS_JSON) as f:
        data = json.load(f)
    if task_name in data:
        log.warning(f"Overwriting existing {task_name} entry in train_seeds.json")
    data[task_name] = {
        "task_name": task_name,
        "success_seeds": [s["seed"] for s in seeds],
    }
    with open(TRAIN_SEEDS_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"Written {len(seeds)} train seeds → {TRAIN_SEEDS_JSON}")


def write_eval_manifest(task_name: str, seeds: list[dict]):
    out_path = SEEDS_DIR / f"{task_name}_manifest_{len(seeds)}ep.json"
    data = {task_name: {"episodes": [{"seed": s["seed"], "instruction": s["instruction"]} for s in seeds]}}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"Written {len(seeds)} eval episodes → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate expert-verified seeds for a RoboTwin task")
    parser.add_argument("--task", required=True, help="Task name, e.g. put_bottles_dustbin")
    parser.add_argument("--train-only", action="store_true", help="Only generate train seeds")
    parser.add_argument("--eval-only", action="store_true", help="Only generate eval manifest")
    parser.add_argument("--train-target", type=int, default=TRAIN_TARGET)
    parser.add_argument("--eval-target", type=int, default=EVAL_TARGET)
    parser.add_argument("--train-start", type=int, default=TRAIN_SEED_START)
    parser.add_argument("--train-end", type=int, default=TRAIN_SEED_END)
    parser.add_argument("--eval-start", type=int, default=EVAL_SEED_START)
    parser.add_argument("--eval-end", type=int, default=EVAL_SEED_END)
    args = parser.parse_args()

    do_train = not args.eval_only
    do_eval = not args.train_only

    if do_train:
        log.info(f"=== Generating TRAIN seeds for {args.task} ===")
        train_seeds = collect_seeds(
            args.task, args.train_start, args.train_end, args.train_target, "TRAIN",
            checkpoint_fn=lambda s: write_train_seeds(args.task, s),
        )
        if train_seeds:
            write_train_seeds(args.task, train_seeds)
        else:
            log.error("No train seeds collected!")

    if do_eval:
        log.info(f"=== Generating EVAL manifest for {args.task} ===")
        eval_seeds = collect_seeds(
            args.task, args.eval_start, args.eval_end, args.eval_target, "EVAL",
            checkpoint_fn=None,  # eval manifest only needs final result
        )
        if eval_seeds:
            write_eval_manifest(args.task, eval_seeds)
        else:
            log.error("No eval seeds collected!")

    log.info("=== All done ===")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

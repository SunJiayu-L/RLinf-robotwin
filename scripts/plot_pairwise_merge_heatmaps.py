#!/usr/bin/env python3
"""为 10 对两两融合各生成一张 1×3 热力图：
  [task_A vs SFT-20k] | [task_B vs SFT-20k] | [merged vs SFT-20k]

x 轴 = 去噪时间步 t，y 轴 = Action Expert 层，颜色 = HiddenSim（统一色阶）。

用法：
    python scripts/plot_pairwise_merge_heatmaps.py \
        --output-dir results/pairwise_merge_heatmaps/ \
        --device cuda
"""
from __future__ import annotations

import argparse
import itertools
import logging
import pathlib
import sys

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── 路径常量 ──────────────────────────────────────────────────────────────────
REPO_PATH = "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin"
MOE_SFT20K = (
    "/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/"
    "pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/20000/model.safetensors"
)
MOE_MODEL_PATH = (
    "/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/"
    "pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/20000"
)
CONFIG_NAME = "pi05_aloha_moe_demo_clean"
MERGED_DIR = "/mnt/petrelfs/muyao/jiayusun/merged_weights"

PPO_WEIGHTS: dict[str, str] = {
    "bbh": f"{REPO_PATH}/logs/20260625-11:37:02-moe_sft20k_beat_block_hammer_ppo100"
           f"/moe_sft20k_beat_block_hammer_ppo100/checkpoints"
           f"/global_step_300/actor/model_state_dict/full_weights.pt",
    "pps": f"{REPO_PATH}/logs/20260703-21:07:14-moe_sft20k_place_phone_stand_ppo100"
           f"/moe_sft20k_place_phone_stand_ppo100/checkpoints"
           f"/global_step_300/actor/model_state_dict/full_weights.pt",
    "pal": f"{REPO_PATH}/logs/20260703-21:07:14-moe_sft20k_place_a2b_left_ppo100"
           f"/moe_sft20k_place_a2b_left_ppo100/checkpoints"
           f"/global_step_300/actor/model_state_dict/full_weights.pt",
    "par": f"{REPO_PATH}/logs/20260705-10:45:17-moe_sft20k_place_a2b_right_ppo100"
           f"/moe_sft20k_place_a2b_right_ppo100/checkpoints"
           f"/global_step_50/actor/model_state_dict/full_weights.pt",
    "mcp": f"{REPO_PATH}/logs/20260703-00:38:41-moe_sft20k_move_can_pot_ppo100"
           f"/moe_sft20k_move_can_pot_ppo100/checkpoints"
           f"/global_step_400/actor/model_state_dict/full_weights.pt",
}

MERGED_WEIGHTS: dict[tuple[str, str], str] = {
    ("bbh", "pps"): f"{MERGED_DIR}/pair_beat_block_hammer__place_phone_stand_a07/full_weights.pt",
    ("bbh", "pal"): f"{MERGED_DIR}/pair_beat_block_hammer__place_a2b_left_a07/full_weights.pt",
    ("bbh", "par"): f"{MERGED_DIR}/pair_beat_block_hammer__place_a2b_right_a07/full_weights.pt",
    ("bbh", "mcp"): f"{MERGED_DIR}/pair_beat_block_hammer__move_can_pot_a07/full_weights.pt",
    ("pps", "pal"): f"{MERGED_DIR}/pair_place_phone_stand__place_a2b_left_a07/full_weights.pt",
    ("pps", "par"): f"{MERGED_DIR}/pair_place_phone_stand__place_a2b_right_a07/full_weights.pt",
    ("pps", "mcp"): f"{MERGED_DIR}/pair_place_phone_stand__move_can_pot_a07/full_weights.pt",
    ("pal", "par"): f"{MERGED_DIR}/pair_place_a2b_left__place_a2b_right_a07/full_weights.pt",
    ("pal", "mcp"): f"{MERGED_DIR}/pair_place_a2b_left__move_can_pot_a07/full_weights.pt",
    ("par", "mcp"): f"{MERGED_DIR}/pair_place_a2b_right__move_can_pot_a07/full_weights.pt",
}

EVAL_RESULTS: dict[tuple[str, str], dict[str, float]] = {
    ("bbh", "pps"): {"bbh": 63.4, "pps": 76.9},
    ("bbh", "pal"): {"bbh": 55.7, "pal": 48.2},
    ("bbh", "par"): {"bbh": 76.5, "par": 59.2},
    ("bbh", "mcp"): {"bbh": 39.9, "mcp": 67.7},
    ("pps", "pal"): {"pps": 54.1, "pal": 66.6},
    ("pps", "par"): {"pps": 70.7, "par": 70.5},
    ("pps", "mcp"): {"pps": 72.0, "mcp": 46.4},
    ("pal", "par"): {"pal": 64.5, "par": 56.0},
    ("pal", "mcp"): {"pal": 62.4, "mcp": 40.8},
    ("par", "mcp"): {"par": 58.1, "mcp": 62.6},
}

SINGLE_TASK_BASELINES: dict[str, float] = {
    "bbh": 86.1, "pps": 70.7, "pal": 51.9, "par": 57.1, "mcp": 63.6,
}

TASK_FULL: dict[str, str] = {
    "bbh": "beat_block_hammer",
    "pps": "place_phone_stand",
    "pal": "place_a2b_left",
    "par": "place_a2b_right",
    "mcp": "move_can_pot",
}

TASK_ORDER = ["bbh", "pps", "pal", "par", "mcp"]
PAIRS: list[tuple[str, str]] = list(itertools.combinations(TASK_ORDER, 2))

NUM_LAYERS = 18
ACTION_DIM_INTERNAL = 32
ACTION_HORIZON = 50
# 9 个去噪时间步，与参考图一致
T_STEPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ── 模型 & 推理 ──────────────────────────────────────────────────────────────

def build_model(device: str):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "model_path": MOE_MODEL_PATH,
        "model_type": "openpi",
        "precision": "bfloat16",
        "num_action_chunks": ACTION_HORIZON,
        "action_dim": 14,
        "num_steps": 10,
        "is_lora": False,
        "load_to_device": False,
        "openpi": {
            "config_name": CONFIG_NAME,
            "num_images_in_input": 3,
            "noise_level": 0.3,
            "detach_critic_input": True,
            "add_value_head": True,
            "train_expert_only": True,
        },
    })
    from rlinf.scheduler.worker.worker import Worker as _Worker
    from rlinf.scheduler.hardware import AcceleratorUtil
    _accel_type = AcceleratorUtil.get_accelerator_type()
    _Worker.torch_platform = AcceleratorUtil.get_torch_platform(_accel_type)
    _Worker.torch_device_type = AcceleratorUtil.get_device_type(_accel_type)
    sys.path.insert(0, REPO_PATH)
    from rlinf.models.embodiment.openpi import get_model
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model.eval()
    model.to(device)
    return model


def load_weights(model, weight_path: str) -> None:
    p = pathlib.Path(weight_path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(weight_path)
    else:
        sd = torch.load(weight_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and len(sd) == 1:
            sd = sd["state_dict"]
    result = model.load_state_dict(sd, strict=False)
    missing = [k for k in result.missing_keys if "value_head" not in k]
    if missing:
        logger.warning(f"  missing keys: {missing[:2]}")
    model.to(torch.bfloat16)


def make_synthetic_obs(batch_size: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    H, W = 224, 224
    main_img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    wrist_l  = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    wrist_r  = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    state    = rng.uniform(-1, 1, (14,)).astype(np.float32)
    return {
        "main_images":  torch.from_numpy(np.stack([main_img] * batch_size)),
        "wrist_images": torch.from_numpy(
            np.stack([np.stack([wrist_l, wrist_r])] * batch_size)
        ),
        "states":       torch.from_numpy(np.stack([state] * batch_size)),
        "task_descriptions": ["beat block hammer"] * batch_size,
        "extra_view_images": None,
    }


def build_prefix_cache(model, env_obs: dict, device: str):
    import openpi.models.model as _model_module
    to_process_obs = model.obs_processor(env_obs)
    processed_obs  = model.input_transform(to_process_obs, transpose=False)
    processed_obs  = model.precision_processor(processed_obs)
    observation    = _model_module.Observation.from_dict(processed_obs)
    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
            observation, train=False
        )
        _, prefix_pad_masks, past_key_values = model._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks
        )
    return prefix_pad_masks, past_key_values, state


class HiddenStateCollector:
    def __init__(self):
        self.states: dict[int, torch.Tensor] = {}
        self._hooks = []

    def register(self, model) -> None:
        self.clear()
        for i, layer in enumerate(model.paligemma_with_expert.gemma_expert.model.layers):
            self._hooks.append(layer.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            self.states[idx] = hs[:, -ACTION_HORIZON:, :].detach().float()
        return hook

    def clear(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.states.clear()


@torch.no_grad()
def collect_hidden_states(model, prefix_pad_masks, past_key_values, state,
                          x_t: torch.Tensor, device: str,
                          collector: HiddenStateCollector) -> dict:
    """Returns {t_val: {layer_idx: np.ndarray(B, 50, 1024)}}"""
    result: dict[float, dict[int, np.ndarray]] = {}
    for t_val in T_STEPS:
        collector.states.clear()
        B = state.shape[0]
        dtype = model.action_in_proj.weight.dtype
        t_tensor = torch.full((B,), t_val, dtype=dtype, device=device)
        _ = model.get_suffix_out(state, prefix_pad_masks, past_key_values, x_t, t_tensor)
        result[t_val] = {li: hs.cpu().numpy() for li, hs in collector.states.items()}
    return result


def compute_hiddensim_matrix(states_a: dict, states_b: dict) -> np.ndarray:
    """Compute HiddenSim averaged over batch & tokens.

    Returns: (NUM_LAYERS, len(T_STEPS))  — averaged over batch & action horizon
    """
    mat = np.zeros((NUM_LAYERS, len(T_STEPS)))
    for ti, t_val in enumerate(T_STEPS):
        for li in range(NUM_LAYERS):
            h_a = states_a[t_val].get(li)   # (B, 50, D)
            h_b = states_b[t_val].get(li)
            if h_a is None or h_b is None:
                continue
            # per-token HiddenSim, then mean
            diff   = np.linalg.norm(h_a - h_b, axis=-1)          # (B, 50)
            norm_a = np.linalg.norm(h_a, axis=-1)
            norm_b = np.linalg.norm(h_b, axis=-1)
            hs = diff / (norm_a + norm_b + 1e-8)                  # (B, 50)
            mat[li, ti] = hs.mean()
    return mat


def get_states(model, env_obs, x_t, device, collector, weight_path: str):
    if not pathlib.Path(weight_path).exists():
        logger.warning(f"  NOT FOUND: {weight_path}")
        return None
    load_weights(model, weight_path)
    model.eval()
    pfx_pads, pkv, state_tensor = build_prefix_cache(model, env_obs, device)
    return collect_hidden_states(model, pfx_pads, pkv, state_tensor, x_t, device, collector)


# ── 绘图 ─────────────────────────────────────────────────────────────────────

def plot_trio(
    mat_a: np.ndarray,   # (18, 9)
    mat_b: np.ndarray,
    mat_m: np.ndarray,
    label_a: str,
    label_b: str,
    ta: str, tb: str,
    out_path: pathlib.Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 统一色阶
    vmin = min(mat_a.min(), mat_b.min(), mat_m.min())
    vmax = max(mat_a.max(), mat_b.max(), mat_m.max())

    delta_a = EVAL_RESULTS[(ta, tb)].get(ta, float("nan")) - SINGLE_TASK_BASELINES[ta]
    delta_b = EVAL_RESULTS[(ta, tb)].get(tb, float("nan")) - SINGLE_TASK_BASELINES[tb]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        f"HiddenSim vs SFT-20k Base  [{ta}+{tb}]   unified scale [vmin={vmin:.4f}, vmax={vmax:.4f}]\n"
        f"Single-task PPO:  {ta}={SINGLE_TASK_BASELINES[ta]:.1f}%  {tb}={SINGLE_TASK_BASELINES[tb]:.1f}%   "
        f"|   Merged eval:  {ta}={EVAL_RESULTS[(ta,tb)].get(ta,float('nan')):.1f}% (Δ{delta_a:+.1f}pp)  "
        f"{tb}={EVAL_RESULTS[(ta,tb)].get(tb,float('nan')):.1f}% (Δ{delta_b:+.1f}pp)",
        fontsize=11,
    )

    panels = [
        (mat_a, label_a),
        (mat_b, label_b),
        (mat_m, f"merged ({ta}+{tb}) vs Base"),
    ]

    for ax, (mat, title) in zip(axes, panels):
        im = ax.imshow(
            mat, aspect="auto", cmap="YlOrRd",
            vmin=vmin, vmax=vmax,
            extent=[T_STEPS[0] - 0.05, T_STEPS[-1] + 0.05,
                    NUM_LAYERS - 0.5, -0.5],
        )
        plt.colorbar(im, ax=ax, label="HiddenSim")
        ax.set_xlabel("Timestep t", fontsize=10)
        ax.set_ylabel("Layer Index", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(T_STEPS)
        ax.set_xticklabels([f"{t:.1f}" for t in T_STEPS], fontsize=8)
        ax.set_yticks(range(0, NUM_LAYERS, 2))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  saved: {out_path.name}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/pairwise_merge_heatmaps", dest="output_dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.device)

    torch.manual_seed(42)
    x_t = torch.randn(args.batch_size, ACTION_HORIZON, ACTION_DIM_INTERNAL,
                      dtype=torch.bfloat16, device=args.device)
    env_obs = make_synthetic_obs(args.batch_size)

    collector = HiddenStateCollector()
    collector.register(model)

    # ── 收集 SFT-20k base hidden states ──────────────────────────────────────
    logger.info("=== 加载 SFT-20k base ===")
    states_base = get_states(model, env_obs, x_t, args.device, collector, MOE_SFT20K)
    assert states_base is not None

    # ── 收集 5 个 PPO hidden states ───────────────────────────────────────────
    logger.info("=== 收集 5 个 PPO hidden states ===")
    ppo_states: dict[str, dict | None] = {}
    ppo_mats: dict[str, np.ndarray] = {}   # HiddenSim vs base, shape (18, 9)
    for task in TASK_ORDER:
        logger.info(f"  {task}...")
        st = get_states(model, env_obs, x_t, args.device, collector, PPO_WEIGHTS[task])
        ppo_states[task] = st
        if st is not None:
            ppo_mats[task] = compute_hiddensim_matrix(states_base, st)
            logger.info(f"  {task}: mean HiddenSim vs base = {ppo_mats[task].mean():.4f}")

    # ── 对每对融合权重收集 hidden states 并出图 ───────────────────────────────
    logger.info("=== 处理 10 对融合权重 ===")
    for ta, tb in PAIRS:
        logger.info(f"\n  pair: {ta}+{tb}")
        merged_path = MERGED_WEIGHTS[(ta, tb)]
        st_merged = get_states(model, env_obs, x_t, args.device, collector, merged_path)
        if st_merged is None:
            logger.warning(f"  跳过 {ta}+{tb}（merged 权重缺失）")
            continue

        mat_a = ppo_mats.get(ta)
        mat_b = ppo_mats.get(tb)
        mat_m = compute_hiddensim_matrix(states_base, st_merged)
        logger.info(f"  merged vs base: {mat_m.mean():.4f}")

        if mat_a is None or mat_b is None:
            logger.warning(f"  跳过绘图（PPO states 缺失）")
            continue

        out_path = out_dir / f"heatmap_{ta}_{tb}.png"
        plot_trio(
            mat_a, mat_b, mat_m,
            label_a=f"{ta} ({TASK_FULL[ta]}) vs Base",
            label_b=f"{tb} ({TASK_FULL[tb]}) vs Base",
            ta=ta, tb=tb,
            out_path=out_path,
        )

    collector.clear()
    logger.info(f"\n完成。输出目录：{out_dir.resolve()}")


if __name__ == "__main__":
    main()

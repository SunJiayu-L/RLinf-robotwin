#!/usr/bin/env python3
"""分析 5 任务 PPO 权重两两融合的 HiddenSim 与 task vector 余弦相似度。

三类指标：
  1. 任务向量余弦相似度（CPU）—— cos(τ_A, τ_B)，逐层 + 全局
  2. PPO 间表征 HiddenSim（GPU）—— HiddenSim(PPO_A, PPO_B)，逐层
  3. 融合后表征 HiddenSim（GPU）—— HiddenSim(merged_AB, PPO_A/B)，逐层

结合已知的两两融合 eval 结果，输出相关性散点图，解释融合成功/失败的根因。

用法：
    python scripts/analyze_pairwise_merge_hiddensim.py \\
        --output-dir results/pairwise_merge_analysis/ \\
        --device cuda
"""
from __future__ import annotations

import argparse
import itertools
import logging
import pathlib
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

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

# 两两融合 eval 结果（已知，从 2026-7-6-ppo5task-weight-merge.md）
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

TASK_NAMES: dict[str, str] = {
    "bbh": "beat_block_hammer",
    "pps": "place_phone_stand",
    "pal": "place_a2b_left",
    "par": "place_a2b_right",
    "mcp": "move_can_pot",
}

PI05_PPO_TRAINABLE_PREFIXES = (
    "paligemma_with_expert.gemma_expert.model.layers.",
    "paligemma_with_expert.gemma_expert.model.norm.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
)

TASK_ORDER = ["bbh", "pps", "pal", "par", "mcp"]
PAIRS: list[tuple[str, str]] = list(itertools.combinations(TASK_ORDER, 2))
NUM_LAYERS = 18
ACTION_DIM_INTERNAL = 32
ACTION_HORIZON = 50
T_STEPS = [0.1, 0.3, 0.5, 0.7, 0.9]


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1：任务向量余弦相似度（CPU）
# ═══════════════════════════════════════════════════════════════════════════════

def load_sd_cpu(path: str) -> dict[str, torch.Tensor]:
    p = pathlib.Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and len(sd) == 1:
            sd = sd["state_dict"]
    return {k: v.to(torch.float32) for k, v in sd.items()
            if any(k.startswith(p) for p in PI05_PPO_TRAINABLE_PREFIXES)}


def get_layer_idx(key: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def compute_pairwise_taskvec_cosine(
    sd_base: dict[str, torch.Tensor],
    sd_ppoa: dict[str, torch.Tensor],
    sd_ppob: dict[str, torch.Tensor],
) -> tuple[np.ndarray, float]:
    """计算两个任务向量的逐层余弦相似度。

    Returns:
        layer_cos: (NUM_LAYERS,)
        global_cos: scalar (全部可训练 key 拼成一个向量)
    """
    common_keys = set(sd_base) & set(sd_ppoa) & set(sd_ppob)

    layer_vecs_a: dict[int, list[torch.Tensor]] = {i: [] for i in range(NUM_LAYERS)}
    layer_vecs_b: dict[int, list[torch.Tensor]] = {i: [] for i in range(NUM_LAYERS)}
    global_vecs_a: list[torch.Tensor] = []
    global_vecs_b: list[torch.Tensor] = []

    for k in sorted(common_keys):
        base = sd_base[k]
        ta = (sd_ppoa[k] - base).flatten()
        tb = (sd_ppob[k] - base).flatten()
        global_vecs_a.append(ta)
        global_vecs_b.append(tb)
        li = get_layer_idx(k)
        if li is not None and 0 <= li < NUM_LAYERS:
            layer_vecs_a[li].append(ta)
            layer_vecs_b[li].append(tb)

    layer_cos = np.zeros(NUM_LAYERS)
    for li in range(NUM_LAYERS):
        if not layer_vecs_a[li]:
            continue
        va = torch.cat(layer_vecs_a[li])
        vb = torch.cat(layer_vecs_b[li])
        layer_cos[li] = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()

    va_global = torch.cat(global_vecs_a)
    vb_global = torch.cat(global_vecs_b)
    global_cos = F.cosine_similarity(va_global.unsqueeze(0), vb_global.unsqueeze(0)).item()

    return layer_cos, global_cos


def run_taskvec_analysis(out_dir: pathlib.Path) -> tuple[np.ndarray, dict]:
    """加载所有 SD，计算 10 对任务向量余弦相似度。

    Returns:
        all_layer_cos: (10, NUM_LAYERS)
        global_cos_dict: {pair: float}
    """
    logger.info("=== Part 1：加载所有 state dict（CPU）===")
    logger.info("  加载 SFT-20k base...")
    sd_base = load_sd_cpu(MOE_SFT20K)
    logger.info(f"  SFT base trainable keys: {len(sd_base)}")

    sd_ppow: dict[str, dict[str, torch.Tensor]] = {}
    for task in TASK_ORDER:
        logger.info(f"  加载 PPO {task}...")
        sd_ppow[task] = load_sd_cpu(PPO_WEIGHTS[task])

    logger.info("\n=== 计算逐层余弦相似度 ===")
    all_layer_cos = np.zeros((len(PAIRS), NUM_LAYERS))
    global_cos_dict: dict[tuple[str, str], float] = {}

    for pi, (ta, tb) in enumerate(PAIRS):
        layer_cos, global_cos = compute_pairwise_taskvec_cosine(
            sd_base, sd_ppow[ta], sd_ppow[tb]
        )
        all_layer_cos[pi] = layer_cos
        global_cos_dict[(ta, tb)] = global_cos
        logger.info(f"  {ta}+{tb}: global_cos={global_cos:.4f}, "
                    f"layer_mean={layer_cos.mean():.4f}, "
                    f"layer_min={layer_cos.min():.4f} @layer{layer_cos.argmin()}")

    np.save(out_dir / "taskvec_cosine_layerwise.npy", all_layer_cos)
    logger.info(f"  saved: taskvec_cosine_layerwise.npy  shape={all_layer_cos.shape}")

    # 输出 CSV
    import csv
    with open(out_dir / "taskvec_cosine_global.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_a", "pair_b", "global_cos", "layer_mean_cos", "layer_min_cos", "layer_min_idx"])
        for pi, (ta, tb) in enumerate(PAIRS):
            lc = all_layer_cos[pi]
            w.writerow([ta, tb,
                        f"{global_cos_dict[(ta,tb)]:.4f}",
                        f"{lc.mean():.4f}",
                        f"{lc.min():.4f}",
                        int(lc.argmin())])

    del sd_base, sd_ppow
    return all_layer_cos, global_cos_dict


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 & 3：GPU HiddenSim
# ═══════════════════════════════════════════════════════════════════════════════

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
        logger.warning(f"  missing keys (non-value_head): {missing[:3]}")
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
        prefix_output, prefix_pad_masks, past_key_values = model._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks
        )
    return prefix_pad_masks, past_key_values, state


class HiddenStateCollector:
    def __init__(self):
        self.states: dict[int, torch.Tensor] = {}
        self._hooks = []

    def register(self, model) -> None:
        self.clear()
        layers = model.paligemma_with_expert.gemma_expert.model.layers
        for i, layer in enumerate(layers):
            handle = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            self.states[layer_idx] = hs[:, -ACTION_HORIZON:, :].detach().float()
        return hook

    def clear(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.states.clear()


@torch.no_grad()
def run_denoising_step(model, prefix_pad_masks, past_key_values, state,
                       x_t: torch.Tensor, t_val: float, device: str):
    B = state.shape[0]
    dtype = model.action_in_proj.weight.dtype
    t_tensor = torch.full((B,), t_val, dtype=dtype, device=device)
    _ = model.get_suffix_out(state, prefix_pad_masks, past_key_values, x_t, t_tensor)


def collect_hidden_states(model, prefix_pad_masks, past_key_values, state,
                          x_t: torch.Tensor, device: str,
                          collector: HiddenStateCollector) -> dict:
    """对所有 t 收集 hidden states。

    Returns: {t_val: {layer_idx: np.ndarray(B, 50, 1024)}}
    """
    result: dict[float, dict[int, np.ndarray]] = {}
    for t_val in T_STEPS:
        collector.states.clear()
        run_denoising_step(model, prefix_pad_masks, past_key_values, state,
                           x_t, t_val, device)
        result[t_val] = {li: hs.cpu().numpy() for li, hs in collector.states.items()}
    return result


def compute_hiddensim_array(states_a: dict, states_b: dict,
                             batch_size: int) -> np.ndarray:
    """HiddenSim per (layer, t, batch, token).

    Returns: (NUM_LAYERS, len(T_STEPS), batch_size, ACTION_HORIZON)
    """
    arr = np.zeros((NUM_LAYERS, len(T_STEPS), batch_size, ACTION_HORIZON))
    for ti, t_val in enumerate(T_STEPS):
        for li in range(NUM_LAYERS):
            h_a = states_a[t_val].get(li)
            h_b = states_b[t_val].get(li)
            if h_a is None or h_b is None:
                continue
            for ki in range(ACTION_HORIZON):
                diff   = np.linalg.norm(h_a[:, ki, :] - h_b[:, ki, :], axis=-1)
                norm_a = np.linalg.norm(h_a[:, ki, :], axis=-1)
                norm_b = np.linalg.norm(h_b[:, ki, :], axis=-1)
                arr[li, ti, :, ki] = diff / (norm_a + norm_b + 1e-8)
    return arr


def run_hiddensim_analysis(out_dir: pathlib.Path, device: str, batch_size: int):
    """GPU 部分：收集各权重 hidden states，计算三类 HiddenSim。"""
    logger.info("\n=== Part 2 & 3：GPU HiddenSim 分析 ===")
    model = build_model(device)
    logger.info("  模型架构加载完毕")

    torch.manual_seed(42)
    x_t = torch.randn(batch_size, ACTION_HORIZON, ACTION_DIM_INTERNAL,
                      dtype=torch.bfloat16, device=device)
    env_obs = make_synthetic_obs(batch_size)

    collector = HiddenStateCollector()
    collector.register(model)

    def get_states(weight_path: str) -> dict | None:
        if not pathlib.Path(weight_path).exists():
            logger.warning(f"  文件不存在，跳过: {weight_path}")
            return None
        load_weights(model, weight_path)
        model.eval()
        try:
            pfx_pads, pkv, state_tensor = build_prefix_cache(model, env_obs, device)
            return collect_hidden_states(model, pfx_pads, pkv, state_tensor, x_t, device, collector)
        except Exception as e:
            logger.error(f"  前向推理失败: {e}")
            return None

    # ── Part 2：收集 5 个 PPO 的 hidden states ──────────────────────────────
    logger.info("  收集 5 个 PPO checkpoint hidden states...")
    ppo_states: dict[str, dict | None] = {}
    for task in TASK_ORDER:
        logger.info(f"    {task}: {PPO_WEIGHTS[task]}")
        ppo_states[task] = get_states(PPO_WEIGHTS[task])
        if ppo_states[task] is not None:
            logger.info(f"    {task}: OK")

    # PPO 间 HiddenSim：10 对
    logger.info("  计算 PPO 间 HiddenSim (10 对)...")
    hs_ppo_pairs = np.zeros((len(PAIRS), NUM_LAYERS, len(T_STEPS), batch_size, ACTION_HORIZON))
    for pi, (ta, tb) in enumerate(PAIRS):
        if ppo_states[ta] is None or ppo_states[tb] is None:
            logger.warning(f"  跳过 {ta}+{tb}（missing states）")
            continue
        arr = compute_hiddensim_array(ppo_states[ta], ppo_states[tb], batch_size)
        hs_ppo_pairs[pi] = arr
        logger.info(f"  {ta}+{tb}: PPO-PPO HiddenSim mean={arr.mean():.4f}")

    np.save(out_dir / "hiddensim_ppo_pairs.npy", hs_ppo_pairs)
    logger.info("  saved: hiddensim_ppo_pairs.npy")

    # ── Part 3：对每个融合权重计算 HiddenSim vs 两个组成 PPO ─────────────────
    logger.info("  计算融合权重 HiddenSim (10 对 × 2)...")
    hs_merged_vs_a = np.zeros((len(PAIRS), NUM_LAYERS, len(T_STEPS), batch_size, ACTION_HORIZON))
    hs_merged_vs_b = np.zeros_like(hs_merged_vs_a)

    for pi, (ta, tb) in enumerate(PAIRS):
        merged_path = MERGED_WEIGHTS[(ta, tb)]
        logger.info(f"  {ta}+{tb}: 加载融合权重")
        merged_states = get_states(merged_path)
        if merged_states is None:
            continue
        if ppo_states[ta] is not None:
            hs_merged_vs_a[pi] = compute_hiddensim_array(merged_states, ppo_states[ta], batch_size)
        if ppo_states[tb] is not None:
            hs_merged_vs_b[pi] = compute_hiddensim_array(merged_states, ppo_states[tb], batch_size)
        logger.info(f"  {ta}+{tb}: vs_a={hs_merged_vs_a[pi].mean():.4f}, "
                    f"vs_b={hs_merged_vs_b[pi].mean():.4f}")

    np.save(out_dir / "hiddensim_merged_vs_a.npy", hs_merged_vs_a)
    np.save(out_dir / "hiddensim_merged_vs_b.npy", hs_merged_vs_b)
    logger.info("  saved: hiddensim_merged_vs_a.npy, hiddensim_merged_vs_b.npy")

    collector.clear()
    return hs_ppo_pairs, hs_merged_vs_a, hs_merged_vs_b


# ═══════════════════════════════════════════════════════════════════════════════
# Part 4：分析与可视化
# ═══════════════════════════════════════════════════════════════════════════════

def compute_delta(pair: tuple[str, str], task: str) -> float:
    """计算单个任务的融合掉分：merged_eval - single_ppo_baseline。"""
    merged_eval = EVAL_RESULTS[pair].get(task, float("nan"))
    baseline = SINGLE_TASK_BASELINES[task]
    return merged_eval - baseline


def run_analysis_and_plots(
    all_layer_cos: np.ndarray,
    global_cos_dict: dict,
    hs_ppo_pairs: np.ndarray,
    hs_merged_vs_a: np.ndarray,
    hs_merged_vs_b: np.ndarray,
    out_dir: pathlib.Path,
    batch_size: int,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import csv

    pair_labels = [f"{ta}+{tb}" for ta, tb in PAIRS]

    # ── 汇总表格 ──────────────────────────────────────────────────────────────
    rows = []
    for pi, (ta, tb) in enumerate(PAIRS):
        global_cos   = global_cos_dict[(ta, tb)]
        layer_cos    = all_layer_cos[pi]
        hs_ppo_mean  = hs_ppo_pairs[pi].mean()
        hs_va_mean   = hs_merged_vs_a[pi].mean()
        hs_vb_mean   = hs_merged_vs_b[pi].mean()
        delta_a      = compute_delta((ta, tb), ta)
        delta_b      = compute_delta((ta, tb), tb)
        avg_delta    = (delta_a + delta_b) / 2
        rows.append({
            "pair": f"{ta}+{tb}",
            "ta": ta, "tb": tb,
            "global_cos": global_cos,
            "layer_mean_cos": layer_cos.mean(),
            "layer_min_cos": layer_cos.min(),
            "hs_ppo_mean": hs_ppo_mean,
            "hs_merged_vs_a": hs_va_mean,
            "hs_merged_vs_b": hs_vb_mean,
            "delta_a": delta_a,
            "delta_b": delta_b,
            "avg_delta": avg_delta,
        })

    with open(out_dir / "summary.csv", "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    logger.info("  saved: summary.csv")

    # ── 图1：任务向量余弦相似度热力图（逐层）────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(all_layer_cos, aspect="auto", cmap="RdBu",
                   vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="cos(τ_A, τ_B)")
    ax.set_yticks(range(len(PAIRS)))
    ax.set_yticklabels(pair_labels, fontsize=8)
    ax.set_xticks(range(NUM_LAYERS))
    ax.set_xlabel("Action Expert Layer Index")
    ax.set_title("Pairwise Task Vector Cosine Similarity (per layer)\n"
                 "Red=conflict (cos<0), Blue=aligned (cos>0)")
    plt.tight_layout()
    plt.savefig(out_dir / "taskvec_cosine_heatmap.png", dpi=150)
    plt.close()
    logger.info("  saved: taskvec_cosine_heatmap.png")

    # ── 图2：全局余弦相似度 vs 平均掉分（散点图）────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    x_cos = [r["global_cos"] for r in rows]
    y_delta = [r["avg_delta"] for r in rows]
    scatter_colors = plt.cm.RdYlGn(
        [(d + 50) / 100 for d in y_delta]
    )
    sc = ax.scatter(x_cos, y_delta, c=y_delta, cmap="RdYlGn", vmin=-50, vmax=20,
                    s=120, zorder=3)
    plt.colorbar(sc, ax=ax, label="Avg Δ (pp)")
    for r in rows:
        ax.annotate(r["pair"], (r["global_cos"], r["avg_delta"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Global Task Vector Cosine Similarity cos(τ_A, τ_B)")
    ax.set_ylabel("Avg performance change (pp vs single-task PPO)")
    ax.set_title("Task Vector Conflict vs Merge Quality\n"
                 "(more negative cos → more conflict → worse merge)")
    ax.grid(alpha=0.25)

    # 线性拟合
    x_arr = np.array(x_cos)
    y_arr = np.array(y_delta)
    valid = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    if valid.sum() >= 3:
        coeffs = np.polyfit(x_arr[valid], y_arr[valid], 1)
        x_line = np.linspace(x_arr[valid].min(), x_arr[valid].max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), "k--", linewidth=1.2,
                label=f"fit: y={coeffs[0]:.1f}x+{coeffs[1]:.1f}")
        corr = np.corrcoef(x_arr[valid], y_arr[valid])[0, 1]
        ax.text(0.05, 0.92, f"Pearson r = {corr:.3f}", transform=ax.transAxes, fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "performance_vs_cosine.png", dpi=150)
    plt.close()
    logger.info("  saved: performance_vs_cosine.png")

    # ── 图3：PPO 间 HiddenSim 热力图（逐层）─────────────────────────────────
    hs_ppo_global = hs_ppo_pairs.mean(axis=(2, 3, 4))  # (10, 18)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(hs_ppo_global, aspect="auto", cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax, label="HiddenSim(PPO_A, PPO_B)")
    ax.set_yticks(range(len(PAIRS)))
    ax.set_yticklabels(pair_labels, fontsize=8)
    ax.set_xticks(range(NUM_LAYERS))
    ax.set_xlabel("Action Expert Layer Index")
    ax.set_title("PPO Representation Divergence (per layer)\n"
                 "HiddenSim(PPO_A, PPO_B) — higher = more different representations")
    plt.tight_layout()
    plt.savefig(out_dir / "hiddensim_ppo_pairs_heatmap.png", dpi=150)
    plt.close()
    logger.info("  saved: hiddensim_ppo_pairs_heatmap.png")

    # ── 图4：PPO 间 HiddenSim（全局）vs 平均掉分 ────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    x_hs = [hs_ppo_pairs[pi].mean() for pi in range(len(PAIRS))]
    y_delta = [r["avg_delta"] for r in rows]
    sc = ax.scatter(x_hs, y_delta, c=y_delta, cmap="RdYlGn", vmin=-50, vmax=20,
                    s=120, zorder=3)
    plt.colorbar(sc, ax=ax, label="Avg Δ (pp)")
    for i, r in enumerate(rows):
        ax.annotate(r["pair"], (x_hs[i], y_delta[i]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Global HiddenSim(PPO_A, PPO_B)")
    ax.set_ylabel("Avg performance change (pp)")
    ax.set_title("PPO Representation Divergence vs Merge Quality")
    ax.grid(alpha=0.25)

    x_arr = np.array(x_hs)
    valid = ~np.isnan(x_arr) & ~np.isnan(np.array(y_delta))
    if valid.sum() >= 3:
        coeffs = np.polyfit(x_arr[valid], np.array(y_delta)[valid], 1)
        x_line = np.linspace(x_arr[valid].min(), x_arr[valid].max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), "k--", linewidth=1.2)
        corr = np.corrcoef(x_arr[valid], np.array(y_delta)[valid])[0, 1]
        ax.text(0.05, 0.92, f"Pearson r = {corr:.3f}", transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / "hiddensim_vs_performance.png", dpi=150)
    plt.close()
    logger.info("  saved: hiddensim_vs_performance.png")

    # ── 图5：每对融合权重 vs 两个 PPO 的 HiddenSim（bar chart）──────────────
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    for pi, (ta, tb) in enumerate(PAIRS):
        ax = axes[pi // 5][pi % 5]
        va_layer = hs_merged_vs_a[pi].mean(axis=(1, 2, 3))  # (18,)
        vb_layer = hs_merged_vs_b[pi].mean(axis=(1, 2, 3))
        x = np.arange(NUM_LAYERS)
        ax.bar(x - 0.2, va_layer, 0.35, label=f"vs {ta}", color="#1f77b4", alpha=0.7)
        ax.bar(x + 0.2, vb_layer, 0.35, label=f"vs {tb}", color="#d62728", alpha=0.7)
        delta_a = compute_delta((ta, tb), ta)
        delta_b = compute_delta((ta, tb), tb)
        ax.set_title(f"{ta}+{tb}\n"
                     f"Δ{ta}={delta_a:+.1f}pp, Δ{tb}={delta_b:+.1f}pp",
                     fontsize=7)
        ax.legend(fontsize=6, loc="upper left")
        ax.set_xlabel("Layer", fontsize=6)
        ax.set_xticks(range(0, NUM_LAYERS, 3))
        ax.grid(alpha=0.25)
    fig.suptitle("Merged Weight HiddenSim vs Each Constituent PPO (per layer)\n"
                 "Higher = merged weight drifted more from that task's representation",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "hiddensim_merged_per_pair.png", dpi=150)
    plt.close()
    logger.info("  saved: hiddensim_merged_per_pair.png")

    # ── 图6：任务级别 —— 每个任务作为 A/B 时融合掉分 vs 任务向量余弦对应列 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for task_idx, focus_task in enumerate(TASK_ORDER):
        task_rows = [(pi, pair) for pi, pair in enumerate(PAIRS) if focus_task in pair]
        partner_cos = []
        partner_delta = []
        partner_labels = []
        for pi, pair in task_rows:
            partner = pair[0] if pair[1] == focus_task else pair[1]
            cos_val = global_cos_dict[pair]
            delta   = compute_delta(pair, focus_task)
            partner_cos.append(cos_val)
            partner_delta.append(delta)
            partner_labels.append(partner)

    ax = axes[0]
    for task in TASK_ORDER:
        task_rows = [(pi, pair) for pi, pair in enumerate(PAIRS) if task in pair]
        xs = [global_cos_dict[pair] for _, pair in task_rows]
        ys = [compute_delta(pair, task) for _, pair in task_rows]
        ax.scatter(xs, ys, label=task, s=80, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("cos(τ_focus, τ_partner)")
    ax.set_ylabel("Δ success (pp) for focus task")
    ax.set_title("Per-task: how much does conflict hurt?")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)

    ax = axes[1]
    task_avg_delta = {}
    task_avg_cos = {}
    for task in TASK_ORDER:
        task_rows = [(pi, pair) for pi, pair in enumerate(PAIRS) if task in pair]
        deltas = [compute_delta(pair, task) for _, pair in task_rows]
        coss   = [global_cos_dict[pair] for _, pair in task_rows]
        task_avg_delta[task] = np.nanmean(deltas)
        task_avg_cos[task]   = np.nanmean(coss)

    colors = plt.cm.tab10(np.linspace(0, 1, len(TASK_ORDER)))
    for i, task in enumerate(TASK_ORDER):
        ax.bar(i, task_avg_delta[task], color=colors[i], label=task, alpha=0.8)
        ax.text(i, task_avg_delta[task] + (1 if task_avg_delta[task] >= 0 else -3),
                f"{task_avg_cos[task]:.2f}", ha="center", fontsize=7, color="black")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels(TASK_ORDER)
    ax.set_ylabel("Avg Δ pp across all 4 partners")
    ax.set_title("Per-task average merge loss\n(annotation = avg cos with partners)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2, axis="y")

    plt.suptitle("Task-level Merge Compatibility Analysis", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "per_task_compatibility.png", dpi=150)
    plt.close()
    logger.info("  saved: per_task_compatibility.png")

    # ── 打印汇总 ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("汇总表格（按 global_cos 升序）")
    logger.info(f"{'Pair':<12} {'cos':>7} {'HS_PPO':>8} {'HS_va':>8} {'HS_vb':>8}"
                f" {'Δa':>7} {'Δb':>7} {'AvgΔ':>7}")
    logger.info("-" * 70)
    for r in sorted(rows, key=lambda r: r["global_cos"]):
        logger.info(
            f"{r['pair']:<12} {r['global_cos']:>7.3f} {r['hs_ppo_mean']:>8.4f}"
            f" {r['hs_merged_vs_a']:>8.4f} {r['hs_merged_vs_b']:>8.4f}"
            f" {r['delta_a']:>+7.1f} {r['delta_b']:>+7.1f} {r['avg_delta']:>+7.1f}"
        )
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/pairwise_merge_analysis", dest="output_dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    ap.add_argument("--skip-gpu", action="store_true", dest="skip_gpu",
                    help="只跑 CPU task vector 部分，跳过 GPU HiddenSim")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Part 1：任务向量余弦（CPU）─────────────────────────────────────────
    all_layer_cos, global_cos_dict = run_taskvec_analysis(out_dir)

    if args.skip_gpu:
        # 仅用 CPU 结果做可视化（HiddenSim 全零）
        zero = np.zeros((len(PAIRS), NUM_LAYERS, len(T_STEPS), args.batch_size, ACTION_HORIZON))
        run_analysis_and_plots(all_layer_cos, global_cos_dict, zero, zero, zero, out_dir, args.batch_size)
    else:
        # ── Part 2 & 3：HiddenSim（GPU）────────────────────────────────────
        hs_ppo_pairs, hs_merged_vs_a, hs_merged_vs_b = run_hiddensim_analysis(
            out_dir, args.device, args.batch_size
        )
        run_analysis_and_plots(
            all_layer_cos, global_cos_dict,
            hs_ppo_pairs, hs_merged_vs_a, hs_merged_vs_b,
            out_dir, args.batch_size,
        )

    logger.info(f"\n完成。输出目录：{out_dir.resolve()}")


if __name__ == "__main__":
    main()

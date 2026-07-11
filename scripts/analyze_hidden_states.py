#!/usr/bin/env python3
"""实验三：Action Expert 隐藏状态相似度分析（需要 GPU）。

比较不同检查点在相同输入下的 Action Expert 各层隐藏状态差异。

对比组合：
  A. ca_sft15k     vs rf20sft              (SFT 专化对表征的影响)
  B. ppo_adj_step50 vs merged_a07          (PPO 专化对表征的影响)
  C. ca_sft15k     vs ppo_adj_step50       (两种方法的直接对比)

指标：HiddenSim = ||h_A - h_B||_F / (||h_A||_F + ||h_B||_F)
           对每个 (layer, t, token_k) 维度分别计算

用法：
    python scripts/analyze_hidden_states.py \\
        --rf20sft   <path>/random_first20_pi05_sft/full_weights.pt \\
        --ca-sft15k <path>/global_step_15000/.../full_weights.pt \\
        --merged-a07 <path>/merged_weights/ppo_adj200.../full_weights.pt \\
        --ppo-step50 <path>/global_step_50/.../full_weights.pt \\
        --output-dir results/hidden_states/ \\
        --device cuda \\
        --batch-size 4 \\
        --n-obs 8
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────────
MODEL_PATH = "/mnt/petrelfs/muyao/jiayusun/random_first20_pi05_sft"
CONFIG_NAME = "pi05_aloha_robotwin15"
ACTION_DIM_INTERNAL = 32   # 模型内部 action dim（Pi0Config.action_dim）
ACTION_HORIZON = 50
NUM_LAYERS = 18
T_STEPS = [0.1, 0.3, 0.5, 0.7, 0.9]


# ── 模型加载 ──────────────────────────────────────────────────
def build_model(device: str = "cpu"):
    """用 rf20sft model_path 建立模型架构 + norm_stats，不加载权重。"""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "model_path": MODEL_PATH,
        "model_type": "openpi",
        "precision": "bfloat16",
        "num_action_chunks": ACTION_HORIZON,
        "action_dim": 14,  # 环境 action dim（用于 ckpt 路径和 unnorm）
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
    # 设置 Worker.torch_platform（让 get_model 能正常加载到 GPU）
    from rlinf.scheduler.worker.worker import Worker as _Worker
    from rlinf.scheduler.hardware import AcceleratorUtil
    _accel_type = AcceleratorUtil.get_accelerator_type()
    _Worker.torch_platform = AcceleratorUtil.get_torch_platform(_accel_type)
    _Worker.torch_device_type = AcceleratorUtil.get_device_type(_accel_type)

    sys.path.insert(0, "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin")
    from rlinf.models.embodiment.openpi import get_model
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model.eval()
    model.to(device)
    return model


def load_weights(model, weight_path: str) -> None:
    """覆盖模型权重（不改变 norm_stats 和 transforms）。"""
    sd = torch.load(weight_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and len(sd) == 1:
        sd = sd["state_dict"]
    result = model.load_state_dict(sd, strict=False)
    missing = [k for k in result.missing_keys if "value_head" not in k]
    if missing:
        logger.warning(f"  缺失 key（非 value_head）: {missing[:5]}...")
    # 权重文件可能是 float32，统一转回 bfloat16
    model.to(torch.bfloat16)
    logger.info(f"  已加载权重：{weight_path}")


# ── 合成观测生成 ───────────────────────────────────────────────
def make_synthetic_obs(batch_size: int, seed: int = 42):
    """生成固定随机观测（robotwin env_obs 格式，HWC uint8）。"""
    rng = np.random.default_rng(seed)
    H, W = 224, 224
    # robotwin env 格式：HWC uint8
    main_img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    wrist_l  = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    wrist_r  = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    state    = rng.uniform(-1, 1, (14,)).astype(np.float32)

    return {
        "main_images":  torch.from_numpy(np.stack([main_img] * batch_size)),          # (B, H, W, 3)
        "wrist_images": torch.from_numpy(
            np.stack([np.stack([wrist_l, wrist_r])] * batch_size)
        ),                                                                              # (B, 2, H, W, 3)
        "states":       torch.from_numpy(np.stack([state] * batch_size)),              # (B, 14)
        "task_descriptions": ["click alarm clock"] * batch_size,
        "extra_view_images": None,
    }


# ── 前缀 KV cache 构建 ─────────────────────────────────────────
def build_prefix_cache(model, env_obs: dict, device: str):
    """从 env_obs（robotwin 格式）构建前缀 KV cache。

    流程：obs_processor → input_transform → precision_processor → Observation.from_dict

    Returns:
        prefix_pad_masks, past_key_values, state_tensor
    """
    import openpi.models.model as _model_module

    # 与 predict_action_batch 相同的流程
    to_process_obs = model.obs_processor(env_obs)
    processed_obs = model.input_transform(to_process_obs, transpose=False)
    processed_obs = model.precision_processor(processed_obs)
    observation = _model_module.Observation.from_dict(processed_obs)

    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
            observation, train=False
        )
        prefix_output, prefix_pad_masks, past_key_values = model._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks
        )
    return prefix_pad_masks, past_key_values, state


# ── Hook 注册 ─────────────────────────────────────────────────
class HiddenStateCollector:
    """在 gemma_expert.model.layers 的每层注册 forward hook。"""

    def __init__(self):
        self.states: dict[int, torch.Tensor] = {}  # layer_idx → (B, 50, 1024)
        self._hooks = []

    def register(self, model) -> None:
        self.clear()
        layers = model.paligemma_with_expert.gemma_expert.model.layers
        for i, layer in enumerate(layers):
            handle = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(handle)
        logger.info(f"  已在 {len(layers)} 层注册 hooks")

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output 是 (hidden_states, ...) 或直接是 Tensor
            hs = output[0] if isinstance(output, (tuple, list)) else output
            # 取 action suffix tokens = 最后 ACTION_HORIZON 个 token
            self.states[layer_idx] = hs[:, -ACTION_HORIZON:, :].detach().float()
        return hook

    def clear(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.states.clear()

    def get(self) -> dict[int, torch.Tensor]:
        return {k: v.clone() for k, v in self.states.items()}


# ── 前向推理（固定 x_t 和 t）────────────────────────────────────
@torch.no_grad()
def run_denoising_step(model, prefix_pad_masks, past_key_values, state,
                       x_t: torch.Tensor, t_val: float, device: str):
    """运行一次 get_suffix_out，触发 gemma_expert.model.layers hooks。"""
    B = state.shape[0]
    dtype = model.action_in_proj.weight.dtype
    t_tensor = torch.full((B,), t_val, dtype=dtype, device=device)
    _ = model.get_suffix_out(state, prefix_pad_masks, past_key_values, x_t, t_tensor)


# ── HiddenSim 计算 ─────────────────────────────────────────────
def compute_hiddensim(h_a: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    """HiddenSim = ||h_A - h_B||_F / (||h_A||_F + ||h_B||_F) per token.

    h_a, h_b: (B, hidden_dim)
    返回: (B,) float
    """
    diff = np.linalg.norm(h_a - h_b, axis=-1)
    norm_a = np.linalg.norm(h_a, axis=-1)
    norm_b = np.linalg.norm(h_b, axis=-1)
    return diff / (norm_a + norm_b + 1e-8)


# ── 主分析函数 ────────────────────────────────────────────────
def collect_hidden_states(model, prefix_pad_masks, past_key_values, state,
                          x_t: torch.Tensor, device: str,
                          collector: HiddenStateCollector) -> dict:
    """对所有 t_steps 收集 hidden states.

    Returns: {t_val: {layer_idx: np.ndarray(B, 50, 1024)}}
    """
    result = {}
    for t_val in T_STEPS:
        collector.states.clear()
        run_denoising_step(model, prefix_pad_masks, past_key_values, state, x_t, t_val, device)
        result[t_val] = {
            li: hs.cpu().numpy()
            for li, hs in collector.states.items()
        }
        logger.info(f"    t={t_val:.1f} 完成，共 {len(result[t_val])} 层")
    return result


def run_analysis(args):
    device = args.device
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 建立模型 ──────────────────────────────────────────
    logger.info("=== 加载模型架构（使用 rf20sft norm_stats）===")
    model = build_model(device)

    # ── 2. 固定随机 x_t ──────────────────────────────────────
    torch.manual_seed(42)
    x_t = torch.randn(args.batch_size, ACTION_HORIZON, ACTION_DIM_INTERNAL,
                      dtype=torch.bfloat16, device=device)

    # ── 3. 合成观测 → 前缀 cache ─────────────────────────────
    logger.info("=== 构建前缀 KV cache（固定合成观测）===")
    env_obs = make_synthetic_obs(args.batch_size)
    collector = HiddenStateCollector()
    collector.register(model)

    # ── 4. 逐检查点收集 hidden states ─────────────────────────
    checkpoints = {
        "rf20sft":    args.rf20sft,
        "ca_sft15k":  args.ca_sft15k,
        "adj_sft15k": args.adj_sft15k,
        "merged_a07": args.merged_a07,
        "ppo_step50": args.ppo_step50,
        "ppo_ca":     args.ppo_ca,
        "ppo_adj":    args.ppo_adj,
    }
    # 跳过路径为空的可选 checkpoint
    checkpoints = {k: v for k, v in checkpoints.items() if v}
    all_states: dict[str, dict] = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        logger.info(f"=== [{ckpt_name}] 加载权重并前向 ===")
        load_weights(model, ckpt_path)
        model.eval()

        # 重新建前缀 cache（因为权重变了）
        try:
            prefix_pad_masks, past_key_values, state_tensor = build_prefix_cache(
                model, env_obs, device
            )
        except Exception as e:
            logger.error(f"  [{ckpt_name}] 前缀 cache 构建失败，跳过: {e}")
            continue

        all_states[ckpt_name] = collect_hidden_states(
            model, prefix_pad_masks, past_key_values, state_tensor, x_t, device, collector
        )

    collector.clear()

    # ── 5. 计算 HiddenSim ────────────────────────────────────
    logger.info("=== 计算 HiddenSim ===")
    pairs = [
        ("ca_sft15k",  "rf20sft",    "A: SFT-ft vs Base"),
        ("ppo_step50", "merged_a07", "B: PPO-adj50 vs merged_a07"),
        ("ca_sft15k",  "ppo_step50", "C: SFT-ft vs PPO-adj50"),
        ("ppo_ca",     "rf20sft",    "D: PPO-CA vs Base"),
        ("ppo_adj",    "rf20sft",    "E: PPO-adj vs Base"),
        ("merged_a07", "rf20sft",    "F: merged_a07 vs Base"),
        ("ppo_step50", "rf20sft",    "G: PPO-adj50 vs Base"),
        ("adj_sft15k", "rf20sft",    "H: adj-SFT-ft vs Base"),
        ("ca_sft15k",  "adj_sft15k", "I: CA-SFT vs adj-SFT"),
    ]

    # 结果张量: {pair_label: np.ndarray shape (n_layers, n_t_steps, B, 50)}
    results = {}
    for ckpt_a, ckpt_b, pair_label in pairs:
        if ckpt_a not in all_states or ckpt_b not in all_states:
            logger.warning(f"  跳过 {pair_label}（缺失 checkpoint）")
            continue
        sim_arr = np.zeros((NUM_LAYERS, len(T_STEPS), args.batch_size, ACTION_HORIZON))
        for ti, t_val in enumerate(T_STEPS):
            for li in range(NUM_LAYERS):
                h_a = all_states[ckpt_a][t_val].get(li)
                h_b = all_states[ckpt_b][t_val].get(li)
                if h_a is None or h_b is None:
                    continue
                # h shape: (B, 50, 1024); compute per token
                for ki in range(ACTION_HORIZON):
                    sim_arr[li, ti, :, ki] = compute_hiddensim(
                        h_a[:, ki, :], h_b[:, ki, :]
                    )
        results[pair_label] = sim_arr
        npy_path = out_dir / f"hiddensim_{ckpt_a}_vs_{ckpt_b}.npy"
        np.save(npy_path, sim_arr)
        logger.info(f"  已保存：{npy_path}，shape={sim_arr.shape}")

    # ── 6. 可视化 ─────────────────────────────────────────────
    _plot_results(results, out_dir)

    logger.info(f"\n=== 完成 ===\n输出目录：{out_dir.resolve()}")
    _print_summary(results)


def _plot_results(results: dict, out_dir: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "A: SFT-ft vs Base":          "#d62728",   # red
        "B: PPO-adj50 vs merged_a07": "#ff7f0e",   # orange
        "C: SFT-ft vs PPO-adj50":     "#9467bd",   # purple
        "D: PPO-CA vs Base":          "#2ca02c",   # green
        "E: PPO-adj vs Base":         "#17becf",   # cyan
        "F: merged_a07 vs Base":      "#1f77b4",   # blue
        "G: PPO-adj50 vs Base":       "#8c564b",   # brown
        "H: adj-SFT-ft vs Base":      "#e377c2",   # pink
        "I: CA-SFT vs adj-SFT":       "#7f7f7f",   # gray
    }
    # ── 图1：逐层曲线（平均 t 和 token）────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, sim_arr in results.items():
        mean_per_layer = sim_arr.mean(axis=(1, 2, 3))  # (n_layers,)
        ax.plot(range(NUM_LAYERS), mean_per_layer, marker="o", markersize=4,
                label=label, color=colors.get(label))
    ax.set_xlabel("Action Expert Layer Index")
    ax.set_ylabel("HiddenSim (normalized L2)")
    ax.set_title("Hidden State Distance by Layer\n(averaged over t, batch, action tokens)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "layer_curves.png", dpi=150)
    plt.close()

    # ── 图2：时间步曲线（平均层和 token）───────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, sim_arr in results.items():
        mean_per_t = sim_arr.mean(axis=(0, 2, 3))  # (n_t_steps,)
        ax.plot(T_STEPS, mean_per_t, marker="s", markersize=6,
                label=label, color=colors.get(label))
    ax.set_xlabel("Denoising Timestep t")
    ax.set_ylabel("HiddenSim")
    ax.set_title("Hidden State Distance by Timestep")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "t_curves.png", dpi=150)
    plt.close()

    # ── 图3：Action token 位置曲线（平均层和 t）─────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, sim_arr in results.items():
        mean_per_token = sim_arr.mean(axis=(0, 1, 2))  # (action_horizon,)
        ax.plot(range(ACTION_HORIZON), mean_per_token,
                label=label, color=colors.get(label))
    ax.set_xlabel("Action Token Position k (0..49)")
    ax.set_ylabel("HiddenSim")
    ax.set_title("Hidden State Distance by Action Horizon Position")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "horizon_curves.png", dpi=150)
    plt.close()

    # ── 图4：热力图：层 × 时间步（每组一张）─────────────────────
    for label, sim_arr in results.items():
        heatmap = sim_arr.mean(axis=(2, 3))  # (n_layers, n_t_steps)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd",
                       extent=[T_STEPS[0], T_STEPS[-1], NUM_LAYERS - 1, 0])
        plt.colorbar(im, ax=ax, label="HiddenSim")
        ax.set_xlabel("Timestep t")
        ax.set_ylabel("Layer Index")
        ax.set_title(f"Hidden State Distance Heatmap\n{label}")
        plt.tight_layout()
        safe_label = label.replace(" ", "_").replace(":", "").replace("/", "_")
        plt.savefig(out_dir / f"heatmap_{safe_label}.png", dpi=150)
        plt.close()

    logger.info(f"  已保存 {3 + len(results)} 张图表到 {out_dir}")


def _print_summary(results: dict) -> None:
    logger.info("\n===== HiddenSim 汇总 =====")
    for label, sim_arr in results.items():
        mean = sim_arr.mean()
        layer_means = sim_arr.mean(axis=(1, 2, 3))
        peak_layer = int(layer_means.argmax())
        logger.info(f"  {label}")
        logger.info(f"    全局均值: {mean:.4f}")
        logger.info(f"    峰值层: {peak_layer}（HiddenSim={layer_means[peak_layer]:.4f}）")
        logger.info(f"    浅层(0-5): {layer_means[:6].mean():.4f}, "
                    f"中层(6-11): {layer_means[6:12].mean():.4f}, "
                    f"深层(12-17): {layer_means[12:].mean():.4f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--rf20sft", required=True)
    ap.add_argument("--ca-sft15k", required=True, dest="ca_sft15k")
    ap.add_argument("--merged-a07", required=True, dest="merged_a07")
    ap.add_argument("--ppo-step50", required=True, dest="ppo_step50")
    ap.add_argument("--ppo-ca", default="", dest="ppo_ca",
                    help="PPO click_alarmclock checkpoint（可选）")
    ap.add_argument("--ppo-adj", default="", dest="ppo_adj",
                    help="PPO adjust_bottle checkpoint（可选）")
    ap.add_argument("--adj-sft15k", default="", dest="adj_sft15k",
                    help="adjust_bottle SFT 15k checkpoint（可选）")
    ap.add_argument("--output-dir", default="results/hidden_states", dest="output_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    ap.add_argument("--n-obs", type=int, default=4, dest="n_obs",
                    help="独立观测批次数（目前用 1 批）")
    args = ap.parse_args()

    logger.info(f"设备: {args.device}, 批大小: {args.batch_size}")
    run_analysis(args)


if __name__ == "__main__":
    main()

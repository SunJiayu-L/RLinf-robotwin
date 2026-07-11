#!/usr/bin/env python3
"""moe-demo-clean PPO 训练过程中的 Action Expert 隐藏状态漂移分析。

对每个任务（beat_block_hammer / place_object_stand），计算：
  HiddenSim(PPO_stepN, SFT_20k)  N ∈ {50, 100, 150, 200, 250, 300}

输出热力图：x = PPO 训练步数，y = Action Expert 层，颜色 = HiddenSim。
另出 t 曲线（x=去噪时间步）和 horizon 曲线（x=action token 位置）。

用法：
    python scripts/analyze_hidden_states_moe.py \\
        --moe-sft20k  <path>/robotwin_moe_demo_clean_full_pytorch/20000/model.safetensors \\
        --bbh-dir     logs/20260625-11:37:02-.../checkpoints/ \\
        --pos-dir     logs/20260625-11:47:20-.../checkpoints/ \\
        --output-dir  results/hidden_states_moe/ \\
        --device cuda --batch-size 4
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
MOE_MODEL_PATH = (
    "/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/"
    "pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/20000"
)
CONFIG_NAME = "pi05_aloha_moe_demo_clean"
ACTION_DIM_INTERNAL = 32
ACTION_HORIZON = 50
NUM_LAYERS = 18
T_STEPS = [0.1, 0.3, 0.5, 0.7, 0.9]
PPO_STEPS = [50, 100, 150, 200, 250, 300]


# ── 模型加载 ──────────────────────────────────────────────────
def build_model(device: str = "cuda"):
    """用 moe SFT 20k model_path 建立模型架构 + norm_stats，不加载权重。"""
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

    sys.path.insert(0, "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin")
    from rlinf.models.embodiment.openpi import get_model
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model.eval()
    model.to(device)
    return model


def load_weights(model, weight_path: str) -> None:
    """加载权重，支持 .pt 和 .safetensors 格式。"""
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
        logger.warning(f"  缺失 key（非 value_head）: {missing[:5]}...")
    model.to(torch.bfloat16)
    logger.info(f"  已加载权重：{weight_path}")


# ── 合成观测生成 ───────────────────────────────────────────────
def make_synthetic_obs(batch_size: int, seed: int = 42):
    """生成固定随机观测（robotwin env_obs 格式，HWC uint8）。"""
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


# ── 前缀 KV cache 构建 ─────────────────────────────────────────
def build_prefix_cache(model, env_obs: dict, device: str):
    import openpi.models.model as _model_module

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
    def __init__(self):
        self.states: dict[int, torch.Tensor] = {}
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
            hs = output[0] if isinstance(output, (tuple, list)) else output
            self.states[layer_idx] = hs[:, -ACTION_HORIZON:, :].detach().float()
        return hook

    def clear(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.states.clear()

    def get(self) -> dict[int, torch.Tensor]:
        return {k: v.clone() for k, v in self.states.items()}


# ── 前向推理 ──────────────────────────────────────────────────
@torch.no_grad()
def run_denoising_step(model, prefix_pad_masks, past_key_values, state,
                       x_t: torch.Tensor, t_val: float, device: str):
    B = state.shape[0]
    dtype = model.action_in_proj.weight.dtype
    t_tensor = torch.full((B,), t_val, dtype=dtype, device=device)
    _ = model.get_suffix_out(state, prefix_pad_masks, past_key_values, x_t, t_tensor)


# ── HiddenSim 计算 ─────────────────────────────────────────────
def compute_hiddensim(h_a: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    """HiddenSim = ||h_A - h_B||_2 / (||h_A||_2 + ||h_B||_2 + ε) per token.

    h_a, h_b: (B, hidden_dim)  → 返回: (B,)
    """
    diff   = np.linalg.norm(h_a - h_b, axis=-1)
    norm_a = np.linalg.norm(h_a, axis=-1)
    norm_b = np.linalg.norm(h_b, axis=-1)
    return diff / (norm_a + norm_b + 1e-8)


def collect_hidden_states(model, prefix_pad_masks, past_key_values, state,
                          x_t: torch.Tensor, device: str,
                          collector: HiddenStateCollector) -> dict:
    """对所有 t_steps 收集 hidden states。

    Returns: {t_val: {layer_idx: np.ndarray(B, 50, 1024)}}
    """
    result = {}
    for t_val in T_STEPS:
        collector.states.clear()
        run_denoising_step(model, prefix_pad_masks, past_key_values, state,
                           x_t, t_val, device)
        result[t_val] = {
            li: hs.cpu().numpy()
            for li, hs in collector.states.items()
        }
    return result


def compute_hiddensim_array(states_base: dict, states_ppo: dict,
                             batch_size: int) -> np.ndarray:
    """计算两个 checkpoint 间的 HiddenSim 数组。

    Returns: np.ndarray shape (NUM_LAYERS, len(T_STEPS), batch_size, ACTION_HORIZON)
    """
    arr = np.zeros((NUM_LAYERS, len(T_STEPS), batch_size, ACTION_HORIZON))
    for ti, t_val in enumerate(T_STEPS):
        for li in range(NUM_LAYERS):
            h_a = states_base[t_val].get(li)
            h_b = states_ppo[t_val].get(li)
            if h_a is None or h_b is None:
                continue
            for ki in range(ACTION_HORIZON):
                arr[li, ti, :, ki] = compute_hiddensim(h_a[:, ki, :], h_b[:, ki, :])
    return arr


# ── 可视化 ───────────────────────────────────────────────────
def plot_task_results(
    task_name: str,
    step_arrs: dict[int, np.ndarray],   # {step: (18, 5, B, 50)}
    out_dir: pathlib.Path,
) -> None:
    """为单个任务绘制所有图表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    steps = sorted(step_arrs.keys())

    # ── 图1：热力图 (layer × PPO step)，对 t/batch/token 平均 ──────────────
    heatmap = np.zeros((NUM_LAYERS, len(steps)))
    for si, step in enumerate(steps):
        heatmap[:, si] = step_arrs[step].mean(axis=(1, 2, 3))  # (18,)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        heatmap, aspect="auto", cmap="YlOrRd",
        extent=[steps[0] - 25, steps[-1] + 25, NUM_LAYERS - 0.5, -0.5],
    )
    plt.colorbar(im, ax=ax, label="HiddenSim")
    ax.set_xlabel("PPO Training Step")
    ax.set_ylabel("Action Expert Layer Index")
    ax.set_title(
        f"Representation Drift: {task_name}\n"
        f"HiddenSim(PPO_stepN, SFT_20k), averaged over t / batch / tokens"
    )
    ax.set_xticks(steps)
    ax.set_yticks(range(NUM_LAYERS))
    plt.tight_layout()
    plt.savefig(out_dir / f"heatmap_step_{task_name}.png", dpi=150)
    plt.close()
    logger.info(f"  saved: heatmap_step_{task_name}.png")

    # ── 图2：热力图 (layer × t)，对 step/batch/token 平均 ──────────────────
    all_arrs = np.stack(list(step_arrs.values()))  # (n_steps, 18, 5, B, 50)
    heatmap_t = all_arrs.mean(axis=(0, 3, 4))      # (18, 5)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        heatmap_t, aspect="auto", cmap="YlOrRd",
        extent=[T_STEPS[0] - 0.05, T_STEPS[-1] + 0.05, NUM_LAYERS - 0.5, -0.5],
    )
    plt.colorbar(im, ax=ax, label="HiddenSim")
    ax.set_xlabel("Denoising Timestep t")
    ax.set_ylabel("Action Expert Layer Index")
    ax.set_title(
        f"Representation Drift by Layer & Timestep: {task_name}\n"
        f"averaged over PPO steps / batch / tokens"
    )
    ax.set_xticks(T_STEPS)
    ax.set_yticks(range(NUM_LAYERS))
    plt.tight_layout()
    plt.savefig(out_dir / f"heatmap_t_{task_name}.png", dpi=150)
    plt.close()
    logger.info(f"  saved: heatmap_t_{task_name}.png")

    # ── 图3：逐步全局 HiddenSim 折线（三条：整体 / t=0.1 / t=0.9）─────────
    global_mean = [step_arrs[s].mean() for s in steps]
    t01_mean    = [step_arrs[s][:, 0, :, :].mean() for s in steps]   # t=0.1
    t09_mean    = [step_arrs[s][:, 4, :, :].mean() for s in steps]   # t=0.9

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, global_mean, "o-", label="All t", color="#1f77b4", linewidth=2)
    ax.plot(steps, t01_mean,   "s--", label="t=0.1 (fine exec)", color="#d62728", linewidth=1.5)
    ax.plot(steps, t09_mean,   "^--", label="t=0.9 (noise cond)", color="#2ca02c", linewidth=1.5)
    ax.set_xlabel("PPO Training Step")
    ax.set_ylabel("HiddenSim (vs SFT 20k)")
    ax.set_title(f"Representation Drift through PPO: {task_name}")
    ax.set_xticks(steps)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"drift_step_{task_name}.png", dpi=150)
    plt.close()
    logger.info(f"  saved: drift_step_{task_name}.png")

    # ── 图4：逐层折线，按 PPO step 着色 ──────────────────────────────────
    cmap = cm.viridis
    fig, ax = plt.subplots(figsize=(12, 5))
    for si, step in enumerate(steps):
        color = cmap(si / max(len(steps) - 1, 1))
        mean_per_layer = step_arrs[step].mean(axis=(1, 2, 3))
        ax.plot(range(NUM_LAYERS), mean_per_layer, marker="o", markersize=4,
                label=f"step {step}", color=color, linewidth=1.5)
    ax.set_xlabel("Action Expert Layer Index")
    ax.set_ylabel("HiddenSim (vs SFT 20k)")
    ax.set_title(f"Per-Layer Drift at Each PPO Step: {task_name}")
    ax.set_xticks(range(NUM_LAYERS))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"layer_curves_{task_name}.png", dpi=150)
    plt.close()
    logger.info(f"  saved: layer_curves_{task_name}.png")


def plot_comparison(bbh_arrs: dict, pos_arrs: dict, out_dir: pathlib.Path) -> None:
    """两任务全局 HiddenSim 对比折线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(bbh_arrs.keys())

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, [bbh_arrs[s].mean() for s in steps],
            "o-", label="beat_block_hammer", color="#1f77b4", linewidth=2)
    ax.plot(steps, [pos_arrs[s].mean() for s in steps],
            "s-", label="place_object_stand", color="#d62728", linewidth=2)
    ax.set_xlabel("PPO Training Step")
    ax.set_ylabel("HiddenSim (vs SFT 20k)")
    ax.set_title("Representation Drift: Two Tasks Comparison")
    ax.set_xticks(steps)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "drift_comparison.png", dpi=150)
    plt.close()
    logger.info("  saved: drift_comparison.png")


# ── 主函数 ───────────────────────────────────────────────────
def run_analysis(args):
    device = args.device
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== 加载模型架构（moe SFT 20k norm_stats）===")
    model = build_model(device)

    torch.manual_seed(42)
    x_t = torch.randn(args.batch_size, ACTION_HORIZON, ACTION_DIM_INTERNAL,
                      dtype=torch.bfloat16, device=device)

    logger.info("=== 构建合成观测 ===")
    env_obs = make_synthetic_obs(args.batch_size)

    collector = HiddenStateCollector()
    collector.register(model)

    def get_ckpt_path(task_dir: str, step: int) -> str:
        return (
            f"{task_dir}/global_step_{step}/actor/model_state_dict/full_weights.pt"
        )

    # ── 加载所有权重并收集 hidden states ──────────────────────
    # Structure: {task: {step_or_"base": hidden_states}}
    all_states: dict[str, dict] = {"bbh": {}, "pos": {}}

    task_dirs = {
        "bbh": args.bbh_dir,
        "pos": args.pos_dir,
    }

    # 先加载 SFT 20k base（一次即可，两个任务共享同一 base）
    logger.info("=== 加载 SFT 20k base 权重 ===")
    load_weights(model, args.moe_sft20k)
    model.eval()
    try:
        prefix_pad_masks, past_key_values, state_tensor = build_prefix_cache(
            model, env_obs, device
        )
        states_base = collect_hidden_states(
            model, prefix_pad_masks, past_key_values, state_tensor, x_t, device, collector
        )
        logger.info("  SFT 20k base hidden states 收集完成")
    except Exception as e:
        logger.error(f"  SFT 20k base 失败：{e}")
        raise

    # 对两个任务分别加载各 PPO step
    task_step_arrs: dict[str, dict[int, np.ndarray]] = {"bbh": {}, "pos": {}}

    for task, task_dir in task_dirs.items():
        if not task_dir:
            logger.warning(f"  跳过 {task}（未提供目录）")
            continue
        logger.info(f"\n=== 任务 {task} ===")
        for step in PPO_STEPS:
            ckpt_path = get_ckpt_path(task_dir, step)
            if not pathlib.Path(ckpt_path).exists():
                logger.warning(f"  step {step}: 文件不存在，跳过 ({ckpt_path})")
                continue
            logger.info(f"  step {step}: 加载权重")
            load_weights(model, ckpt_path)
            model.eval()
            try:
                prefix_pad_masks, past_key_values, state_tensor = build_prefix_cache(
                    model, env_obs, device
                )
                states_ppo = collect_hidden_states(
                    model, prefix_pad_masks, past_key_values, state_tensor, x_t, device, collector
                )
            except Exception as e:
                logger.error(f"  step {step} 失败：{e}")
                continue

            arr = compute_hiddensim_array(states_base, states_ppo, args.batch_size)
            task_step_arrs[task][step] = arr
            npy_path = out_dir / f"hiddensim_{task}_step{step}_vs_sft20k.npy"
            np.save(npy_path, arr)
            logger.info(f"  step {step}: HiddenSim mean={arr.mean():.4f}, saved {npy_path.name}")

    collector.clear()

    # ── 打印汇总 ──────────────────────────────────────────────
    logger.info("\n===== 汇总 =====")
    for task in ["bbh", "pos"]:
        logger.info(f"\n  Task: {task}")
        logger.info(f"  {'Step':>6}  {'Global HiddenSim':>17}")
        for step in sorted(task_step_arrs[task].keys()):
            arr = task_step_arrs[task][step]
            logger.info(f"  {step:>6}  {arr.mean():>17.4f}")

    # ── 绘图 ─────────────────────────────────────────────────
    if task_step_arrs["bbh"]:
        plot_task_results("beat_block_hammer", task_step_arrs["bbh"], out_dir)
    if task_step_arrs["pos"]:
        plot_task_results("place_object_stand", task_step_arrs["pos"], out_dir)
    if task_step_arrs["bbh"] and task_step_arrs["pos"]:
        plot_comparison(task_step_arrs["bbh"], task_step_arrs["pos"], out_dir)

    logger.info(f"\n=== 完成 ===\n输出目录：{out_dir.resolve()}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--moe-sft20k", required=True, dest="moe_sft20k",
                    help="moe SFT 20k safetensors 路径 (.../20000/model.safetensors)")
    ap.add_argument("--bbh-dir", default="", dest="bbh_dir",
                    help="beat_block_hammer PPO checkpoints 根目录（含 global_step_*/）")
    ap.add_argument("--pos-dir", default="", dest="pos_dir",
                    help="place_object_stand PPO checkpoints 根目录（含 global_step_*/）")
    ap.add_argument("--output-dir", default="results/hidden_states_moe", dest="output_dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    args = ap.parse_args()

    run_analysis(args)


if __name__ == "__main__":
    main()

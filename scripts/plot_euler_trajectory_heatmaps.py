#!/usr/bin/env python3
"""用真实 Euler ODE 轨迹收集隐藏状态，生成两两融合对比热力图。

x_t 在每步正确更新（x_{t-dt} = x_t + dt·v_t），而非固定随机张量。
10步去噪：t = 1.0, 0.9, 0.8, ..., 0.1，共捕捉10个时间步的隐藏状态。

观测来自 `moe-demo-clean` 录制数据（训练 SFT-20k base 所用的同一批 SFT 演示数据），
每个任务使用该任务自己的真实场景图像 + 真实 qpos + 该任务的真实语言指令，
而不是跨任务共享的合成随机观测。

每个 (task, model) 组合报告两种 HiddenSim：
  - self-trajectory（闭环）：模型沿自己的 Euler 轨迹积分，x_t 由该模型自身速度场决定。
    同时包含参数变化和轨迹分叉两种效应。
  - same-trajectory（开环）：所有模型在同一组由 SFT-20k base 生成的 x_t 序列上探测，
    不做自身积分。隔离出纯参数效应，不受轨迹分叉影响。

融合模型（θ_merged = θ_SFT + 0.7·(τ_A+τ_B)）分别在 A、B 两个任务各自的真实观测上
评测（而不是用单一观测代表两个任务）。

另外计算 action expert 任务向量 τ = θ_PPO - θ_SFT 之间的余弦相似度，
为"任务向量方向互补/正交"的说法提供直接的参数空间证据，而不是仅凭 HiddenSim 大小推测。

输出：results/euler_trajectory_heatmaps_v2/
  euler_hs_<task>_vs_base.png/.npz   — 单任务：self vs same-trajectory 两个面板
  heatmap_<ta>_<tb>.png/.npz         — 配对：2行(self/same) x 4列(A@A obs, merged@A obs, B@B obs, merged@B obs)
  task_vector_cosine.json           — 10 对任务向量（action expert 子空间）余弦相似度
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import logging
import pathlib
import sys

import numpy as np
import torch

logger = logging.getLogger(__name__)

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

# 真实观测来源：训练 SFT-20k base 所用的录制演示数据（每个任务 200 条真实 episode）。
REAL_DATA_ROOT = "/mnt/petrelfs/muyao/jiayusun/moe-demo-clean"
REAL_OBS_FRACTION = 0.4  # 在每条 episode 的这个进度处采样观测（避开起始/终止两端）

# action expert 在 state_dict 中的 key 子串，用于任务向量余弦相似度计算
EXPERT_KEY_SUBSTR = "gemma_expert"

PPO_WEIGHTS = {
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

MERGED_WEIGHTS = {
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

EVAL_RESULTS = {
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
SINGLE_TASK_BASELINES = {
    "bbh": 86.1, "pps": 70.7, "pal": 51.9, "par": 57.1, "mcp": 63.6,
}
TASK_NAMES = {
    "bbh": "beat_block_hammer", "pps": "place_phone_stand",
    "pal": "place_a2b_left",   "par": "place_a2b_right", "mcp": "move_can_pot",
}

TASK_ORDER = ["bbh", "pps", "pal", "par", "mcp"]
PAIRS = list(itertools.combinations(TASK_ORDER, 2))
NUM_LAYERS = 18
ACTION_DIM_INTERNAL = 32
ACTION_HORIZON = 50
EULER_STEPS = 10   # 10步 → t = 1.0, 0.9, ..., 0.1


# ── 模型 ──────────────────────────────────────────────────────────────────────

def build_model(device: str):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "model_path": MOE_MODEL_PATH, "model_type": "openpi",
        "precision": "bfloat16", "num_action_chunks": ACTION_HORIZON,
        "action_dim": 14, "num_steps": EULER_STEPS,
        "is_lora": False, "load_to_device": False,
        "openpi": {
            "config_name": CONFIG_NAME, "num_images_in_input": 3,
            "noise_level": 0.3, "detach_critic_input": True,
            "add_value_head": True, "train_expert_only": True,
        },
    })
    from rlinf.scheduler.worker.worker import Worker as _W
    from rlinf.scheduler.hardware import AcceleratorUtil
    _W.torch_platform = AcceleratorUtil.get_torch_platform(AcceleratorUtil.get_accelerator_type())
    _W.torch_device_type = AcceleratorUtil.get_device_type(AcceleratorUtil.get_accelerator_type())
    from rlinf.models.embodiment.openpi import get_model
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model.eval()
    model.to(device)
    return model


def _read_state_dict(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"weights not found: {path}")
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path)
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and len(sd) == 1:
        sd = sd["state_dict"]
    return sd


def load_weights(model, path: str, capture_expert: bool = False) -> dict | None:
    """加载权重到 model；strict=False 仅用于容忍 base 权重缺失 value_head。

    任何其它缺失/多余的 key 都会抛出异常，而不是被静默忽略——避免因权重加载
    失败而复用上一次迭代残留的权重（旧版本 bug：load_weights 失败时静默返回
    False，调用方从不检查返回值）。

    capture_expert=True 时，额外返回 action expert（gemma_expert.*）子集的 CPU
    float32 拷贝，供任务向量余弦相似度计算使用，避免重复读取整个 checkpoint。
    """
    sd = _read_state_dict(path)
    expert_state = None
    if capture_expert:
        expert_state = {
            k: v.detach().to(torch.float32).cpu().clone()
            for k, v in sd.items() if EXPERT_KEY_SUBSTR in k
        }
    result = model.load_state_dict(sd, strict=False)
    # "embed_tokens" is expected to be missing only for the original HF-exported
    # SFT safetensors checkpoint: PaliGemma ties lm_head.weight to
    # language_model.embed_tokens.weight at construction (tie_word_embeddings),
    # HF's save_pretrained dedups the tied copy on save, and load_state_dict's
    # in-place copy of the present lm_head.weight correctly updates the shared
    # tensor. FSDP-exported PPO/merged checkpoints save both explicitly, so this
    # never fires for them.
    missing = [k for k in result.missing_keys if "value_head" not in k and "embed_tokens" not in k]
    if missing:
        raise RuntimeError(f"missing keys loading {path}: {missing[:10]}")
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected keys loading {path}: {result.unexpected_keys[:10]}")
    model.to(torch.bfloat16)
    logger.info(f"  loaded weights: {path}")
    return expert_state


def load_real_obs(task_key: str, batch_size: int) -> dict:
    """加载 `batch_size` 条真实观测（按本文件 TASK_NAMES 里的简称查找任务全名）。"""
    return load_real_obs_by_name(TASK_NAMES[task_key], batch_size)


def load_real_obs_by_name(task_name: str, batch_size: int) -> dict:
    """从 moe-demo-clean 录制数据加载 `batch_size` 条真实观测。

    这是训练 SFT-20k base 所用的同一批 SFT 演示数据（非合成随机像素/关节角，
    非跨任务共享的错误指令）。每个 batch 元素取自不同 episode，在其真实轨迹
    进度 REAL_OBS_FRACTION 处采样，语言指令为该 episode 的真实 resolved 指令。

    注意：这些是录制的 SFT 演示帧，不是 PPO on-policy rollout 帧；但图像、
    关节状态、语言指令均为该任务的真实训练分布内容。

    task_name 是任务全名（如 "move_playingcard_away"），不要求在 TASK_NAMES 中注册，
    只要 moe-demo-clean/<task_name>-demo_clean-200/ 存在即可。
    """
    import h5py
    from PIL import Image

    task_dir = pathlib.Path(REAL_DATA_ROOT) / f"{task_name}-demo_clean-200"

    main_imgs, wrist_imgs, states, instructions = [], [], [], []
    for ep in range(batch_size):
        ep_dir = task_dir / f"episode_{ep}"
        with h5py.File(ep_dir / f"episode_{ep}.hdf5", "r") as f:
            t = int(REAL_OBS_FRACTION * (f["action"].shape[0] - 1))
            main_imgs.append(
                np.array(Image.open(io.BytesIO(bytes(f["observations/images/cam_high"][t]))).convert("RGB"))
            )
            left = np.array(
                Image.open(io.BytesIO(bytes(f["observations/images/cam_left_wrist"][t]))).convert("RGB")
            )
            right = np.array(
                Image.open(io.BytesIO(bytes(f["observations/images/cam_right_wrist"][t]))).convert("RGB")
            )
            wrist_imgs.append(np.stack([left, right]))
            states.append(np.asarray(f["observations/qpos"][t], dtype=np.float32))
        with open(ep_dir / "instructions.json", "r", encoding="utf-8") as jf:
            instructions.append(json.load(jf)["instructions"][0])

    return {
        "main_images": torch.from_numpy(np.stack(main_imgs)),
        "wrist_images": torch.from_numpy(np.stack(wrist_imgs)),
        "states": torch.from_numpy(np.stack(states)),
        "task_descriptions": instructions,
        "extra_view_images": None,
    }


def build_prefix_cache(model, obs, device):
    import openpi.models.model as _m
    to_proc = model.obs_processor(obs)
    processed = model.input_transform(to_proc, transpose=False)
    processed = model.precision_processor(processed)
    observation = _m.Observation.from_dict(processed)
    with torch.no_grad():
        imgs, img_masks, lang_tok, lang_masks, state = model._preprocess_observation(
            observation, train=False)
        _, pfx_pads, pkv = model._build_prefix_cache(imgs, img_masks, lang_tok, lang_masks)
    return pfx_pads, pkv, state


# ── Hook ─────────────────────────────────────────────────────────────────────

class HiddenStateCollector:
    def __init__(self):
        self.states: dict[int, torch.Tensor] = {}
        self._hooks = []

    def register(self, model):
        self.clear()
        layers = model.paligemma_with_expert.gemma_expert.model.layers
        for i, layer in enumerate(layers):
            self._hooks.append(layer.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, layer_idx):
        def hook(module, inp, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            # 只保留 action suffix tokens（最后 ACTION_HORIZON 个）
            self.states[layer_idx] = hs[:, -ACTION_HORIZON:, :].detach().float()
        return hook

    def clear(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.states.clear()


# ── 核心：Euler ODE 轨迹隐藏状态收集 ─────────────────────────────────────────

@torch.no_grad()
def collect_euler_hidden_states(model, pfx_pads, pkv, state,
                                 x_1: torch.Tensor, device: str,
                                 collector: HiddenStateCollector,
                                 num_steps: int = EULER_STEPS):
    """运行真实 Euler ODE（闭环，self-trajectory），在每步捕捉 hidden states。

    每步的 x_t 是上一步积分后的正确值，由该模型自身的速度场决定。

    Returns:
        step_hs: list of {layer_idx: np.ndarray(B, 50, 1024)}, len=num_steps
        t_vals:  list of float, len=num_steps
        x_traj:  list of torch.Tensor(B, 50, 32)，每步积分前实际使用的 x_t
                 （供 same-trajectory 探测复用，隔离参数效应与轨迹分叉）
    """
    B = state.shape[0]
    dtype = model.action_in_proj.weight.dtype
    x = x_1.clone().to(dtype).to(device)
    dt = -1.0 / num_steps

    step_hs = []
    t_vals = []
    x_traj = []

    for i in range(num_steps):
        t_val = 1.0 + i * dt          # 1.0, 0.9, ..., 0.1
        t_tensor = torch.full((B,), t_val, dtype=dtype, device=device)

        x_traj.append(x.detach().clone())

        # ① 捕捉当前 x_t 下的 hidden states（hook 在 get_suffix_out 中被触发）
        collector.states.clear()
        suffix_out = model.get_suffix_out(state, pfx_pads, pkv, x, t_tensor)

        step_hs.append({li: hs.cpu().numpy()
                        for li, hs in collector.states.items()})
        t_vals.append(round(t_val, 1))

        # ② 计算速度并推进 x（dtype 处理同 KL proxy 脚本）
        proj_dtype = model.action_out_proj.weight.dtype
        v_t = model.action_out_proj(suffix_out.to(proj_dtype)).float()
        x = x + dt * v_t.to(dtype)

    return step_hs, t_vals, x_traj


@torch.no_grad()
def probe_fixed_trajectory(model, pfx_pads, pkv, state,
                            x_traj: list[torch.Tensor], t_vals: list[float],
                            device: str, collector: HiddenStateCollector):
    """开环（same-trajectory）探测：在外部给定的固定 x_t 序列上捕捉 hidden states。

    不做自身积分——每一步都直接使用传入的 x_t（通常是 base 模型自己 Euler 轨迹
    上的 x_t），因此所有被比较的模型看到完全相同的输入序列。这隔离了参数变化
    对表征的影响，排除了“模型 A 和模型 B 走了不同轨迹”这一混杂因素。
    """
    B = state.shape[0]
    dtype = model.action_in_proj.weight.dtype
    step_hs = []
    for x_t, t_val in zip(x_traj, t_vals):
        t_tensor = torch.full((B,), t_val, dtype=dtype, device=device)
        collector.states.clear()
        model.get_suffix_out(state, pfx_pads, pkv, x_t.to(dtype).to(device), t_tensor)
        step_hs.append({li: hs.cpu().numpy() for li, hs in collector.states.items()})
    return step_hs


def compute_hiddensim_matrix(step_hs_a, step_hs_b):
    """计算 HiddenSim(A, B) 矩阵，对 batch 和 token 维度平均。

    Returns: np.ndarray shape (NUM_LAYERS, n_steps)
    """
    n_steps = len(step_hs_a)
    mat = np.zeros((NUM_LAYERS, n_steps))
    for si in range(n_steps):
        for li in range(NUM_LAYERS):
            h_a = step_hs_a[si].get(li)   # (B, 50, 1024)
            h_b = step_hs_b[si].get(li)
            if h_a is None or h_b is None:
                continue
            diff   = np.linalg.norm(h_a - h_b, axis=-1)           # (B, 50)
            norm_a = np.linalg.norm(h_a, axis=-1)
            norm_b = np.linalg.norm(h_b, axis=-1)
            hs_per_token = diff / (norm_a + norm_b + 1e-8)        # (B, 50)
            mat[li, si] = hs_per_token.mean()
    return mat


# ── 热力图绘制 ────────────────────────────────────────────────────────────────

def plot_single_task(mat_self, mat_same, t_vals, title_label, instruction_example, out_path):
    """单任务：self-trajectory | same-trajectory 两个面板，共享颜色刻度。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = max(mat_self.max(), mat_same.max())
    vmin = min(mat_self.min(), mat_same.min())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title in zip(
        axes, [mat_self, mat_same],
        ["Self-trajectory (closed-loop)", "Same-trajectory (open-loop, fixed to base x_t)"],
    ):
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax,
                       extent=[-0.5, len(t_vals) - 0.5, NUM_LAYERS - 0.5, -0.5])
        ax.set_xticks(range(len(t_vals)))
        ax.set_xticklabels([f"{t:.1f}" for t in t_vals], fontsize=9)
        ax.set_xlabel("Denoising t (Euler step, 1.0→0.1)", fontsize=10)
        ax.set_ylabel("Action Expert Layer Index", fontsize=10)
        ax.set_title(
            f"{title}\nmax={mat.max():.4f} @t={t_vals[mat.max(axis=0).argmax()]:.1f}", fontsize=10)
        plt.colorbar(im, ax=ax, label="HiddenSim")

    fig.suptitle(
        f"Euler Trajectory HiddenSim vs SFT-20k Base — {title_label} PPO\n"
        f"real obs from moe-demo-clean SFT episodes, e.g. instruction: {instruction_example!r}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  saved: {out_path.name}")


def plot_pair(panels_self, panels_same, titles, t_vals, ta, tb, eval_res, baselines, out_path):
    """配对：2 行(self / same-trajectory) x 4 列(A@A-obs, merged@A-obs, B@B-obs, merged@B-obs)。

    每行独立归一化颜色刻度（self 与 same 的量级通常不同，不强行共享）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    x_ticks = list(range(len(t_vals)))
    x_labels = [f"{t:.1f}" for t in t_vals]

    row_specs = [
        (panels_self, "Self-trajectory\n(closed-loop)"),
        (panels_same, "Same-trajectory\n(open-loop, fixed to base x_t)"),
    ]
    for row, (panels, row_name) in enumerate(row_specs):
        vmax = max(m.max() for m in panels)
        vmin = min(m.min() for m in panels)
        norm = Normalize(vmin=vmin, vmax=vmax)
        for col, (mat, title) in enumerate(zip(panels, titles)):
            ax = axes[row, col]
            ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax,
                     extent=[-0.5, len(t_vals) - 0.5, NUM_LAYERS - 0.5, -0.5])
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, fontsize=7)
            ax.set_xlabel("Denoising t", fontsize=8)
            ax.set_ylabel(f"{row_name}\nLayer" if col == 0 else "Layer", fontsize=8)
            ax.set_title(f"{title}\nmax={mat.max():.4f}", fontsize=8)
            plt.colorbar(ScalarMappable(norm=norm, cmap="YlOrRd"), ax=ax, label="HiddenSim")

    da = eval_res.get(ta, float("nan")) - baselines[ta]
    db = eval_res.get(tb, float("nan")) - baselines[tb]
    suptitle = (
        f"Euler Trajectory HiddenSim vs SFT-20k Base  [{ta}+{tb}]  (real per-task SFT-demo observations)\n"
        f"Single-task PPO eval: {ta}={baselines[ta]:.1f}%  {tb}={baselines[tb]:.1f}%  |  "
        f"Merged eval: {ta}={eval_res.get(ta, float('nan')):.1f}% (Δ{da:+.1f}pp)  "
        f"{tb}={eval_res.get(tb, float('nan')):.1f}% (Δ{db:+.1f}pp)"
    )
    fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  saved: {out_path.name}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/euler_trajectory_heatmaps_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="只处理指定对，如 bbh+mcp pps+par。默认全部10对")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    BS = args.batch_size

    sys.path.insert(0, REPO_PATH)
    model = build_model(device)

    # 固定初始噪声（所有权重、所有任务用同一个 x_1，保证对比公平）
    torch.manual_seed(42)
    x_1 = torch.randn(BS, ACTION_HORIZON, ACTION_DIM_INTERNAL,
                       dtype=torch.bfloat16, device=device)

    collector = HiddenStateCollector()

    if args.pairs:
        selected = [tuple(s.split("+")) for s in args.pairs]
    else:
        selected = PAIRS
    needed_tasks = sorted({t for pair in selected for t in pair})

    # ── 1. 加载真实观测（每个任务各自的真实场景 + 真实指令） ─────────────────
    logger.info("=== 加载真实观测（moe-demo-clean 训练分布录制数据） ===")
    real_obs = {t: load_real_obs(t, BS) for t in needed_tasks}
    for t in needed_tasks:
        logger.info(f"  {t} ({TASK_NAMES[t]}): instr[0]={real_obs[t]['task_descriptions'][0]!r}")

    # ── 2. SFT-20k base：逐任务采集 Euler 轨迹（含 x_t 序列，供 same-trajectory 复用）─
    logger.info("=== 加载 SFT-20k base，逐任务采集 Euler 轨迹 ===")
    sft_expert = load_weights(model, MOE_SFT20K, capture_expert=True)
    model.eval()

    base_step_hs: dict[str, list] = {}
    base_x_traj: dict[str, list] = {}
    t_vals = None
    for task in needed_tasks:
        collector.register(model)
        pfx, pkv, st = build_prefix_cache(model, real_obs[task], device)
        step_hs, t_vals, x_traj = collect_euler_hidden_states(
            model, pfx, pkv, st, x_1, device, collector)
        collector.clear()
        base_step_hs[task] = step_hs
        base_x_traj[task] = x_traj
        logger.info(f"  base@{task}: done, t_vals={t_vals}")

    # ── 3. 各任务 PPO：self-trajectory + same-trajectory，以及任务向量 ────────
    logger.info("=== 收集 PPO hidden states（self + same-trajectory）===")
    mat_ppo_self: dict[str, np.ndarray] = {}
    mat_ppo_same: dict[str, np.ndarray] = {}
    tau: dict[str, torch.Tensor] = {}
    for task in needed_tasks:
        ppo_expert = load_weights(model, PPO_WEIGHTS[task], capture_expert=True)
        model.eval()

        collector.register(model)
        pfx, pkv, st = build_prefix_cache(model, real_obs[task], device)
        self_hs, _, _ = collect_euler_hidden_states(model, pfx, pkv, st, x_1, device, collector)
        same_hs = probe_fixed_trajectory(
            model, pfx, pkv, st, base_x_traj[task], t_vals, device, collector)
        collector.clear()

        mat_ppo_self[task] = compute_hiddensim_matrix(base_step_hs[task], self_hs)
        mat_ppo_same[task] = compute_hiddensim_matrix(base_step_hs[task], same_hs)
        logger.info(
            f"    {task}: self mean={mat_ppo_self[task].mean():.4f} max={mat_ppo_self[task].max():.4f} "
            f"@t={t_vals[mat_ppo_self[task].max(axis=0).argmax()]:.1f}  |  "
            f"same mean={mat_ppo_same[task].mean():.4f} max={mat_ppo_same[task].max():.4f}"
        )

        keys = sorted(k for k in ppo_expert if k in sft_expert)
        tau[task] = torch.cat([(ppo_expert[k] - sft_expert[k]).reshape(-1) for k in keys])
        del ppo_expert

        np.savez(out_dir / f"euler_hs_{task}_vs_base.npz",
                 self=mat_ppo_self[task], same=mat_ppo_same[task], t_vals=np.array(t_vals))
        plot_single_task(
            mat_ppo_self[task], mat_ppo_same[task], t_vals, f"{task} ({TASK_NAMES[task]})",
            real_obs[task]["task_descriptions"][0],
            out_dir / f"euler_hs_{task}_vs_base.png",
        )

    # ── 4. 任务向量（action expert 子空间）余弦相似度：给"方向互补/正交"提供直接证据 ─
    logger.info("=== action-expert 任务向量余弦相似度 τ = θ_PPO - θ_SFT ===")
    cos_sim = {}
    for ta, tb in selected:
        c = torch.nn.functional.cosine_similarity(
            tau[ta].unsqueeze(0), tau[tb].unsqueeze(0)).item()
        cos_sim[f"{ta}+{tb}"] = c
        logger.info(f"  cos(tau_{ta}, tau_{tb}) = {c:+.4f}")
    with open(out_dir / "task_vector_cosine.json", "w", encoding="utf-8") as f:
        json.dump(cos_sim, f, indent=2)
    del tau, sft_expert

    # ── 5. 配对合并：merged 分别在 A、B 各自真实观测上评测 ────────────────────
    logger.info("=== 生成配对合并热力图 ===")
    for ta, tb in selected:
        logger.info(f"  处理 {ta}+{tb}...")
        load_weights(model, MERGED_WEIGHTS[(ta, tb)])
        model.eval()

        merged_self, merged_same = {}, {}
        for probe_task in (ta, tb):
            collector.register(model)
            pfx, pkv, st = build_prefix_cache(model, real_obs[probe_task], device)
            self_hs, _, _ = collect_euler_hidden_states(model, pfx, pkv, st, x_1, device, collector)
            same_hs = probe_fixed_trajectory(
                model, pfx, pkv, st, base_x_traj[probe_task], t_vals, device, collector)
            collector.clear()
            merged_self[probe_task] = compute_hiddensim_matrix(base_step_hs[probe_task], self_hs)
            merged_same[probe_task] = compute_hiddensim_matrix(base_step_hs[probe_task], same_hs)

        logger.info(
            f"    merged@{ta}-obs: self max={merged_self[ta].max():.4f}  same max={merged_same[ta].max():.4f}  |  "
            f"merged@{tb}-obs: self max={merged_self[tb].max():.4f}  same max={merged_same[tb].max():.4f}"
        )

        panels_self = [mat_ppo_self[ta], merged_self[ta], mat_ppo_self[tb], merged_self[tb]]
        panels_same = [mat_ppo_same[ta], merged_same[ta], mat_ppo_same[tb], merged_same[tb]]
        titles = [
            f"{ta} PPO vs Base\n(on {ta} obs)", f"merged vs Base\n(on {ta} obs)",
            f"{tb} PPO vs Base\n(on {tb} obs)", f"merged vs Base\n(on {tb} obs)",
        ]
        out_path = out_dir / f"heatmap_{ta}_{tb}.png"
        plot_pair(panels_self, panels_same, titles, t_vals, ta, tb,
                  EVAL_RESULTS[(ta, tb)], SINGLE_TASK_BASELINES, out_path)
        np.savez(out_dir / f"heatmap_{ta}_{tb}.npz",
                 self_on_ta=merged_self[ta], same_on_ta=merged_same[ta],
                 self_on_tb=merged_self[tb], same_on_tb=merged_same[tb])

    logger.info(f"\n完成。输出目录：{out_dir.resolve()}")


if __name__ == "__main__":
    main()

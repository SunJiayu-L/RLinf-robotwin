"""Replot D/E/F/G heatmaps with a unified color scale."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("results/hidden_states")
T_STEPS = [0.1, 0.3, 0.5, 0.7, 0.9]
NUM_LAYERS = 18

panels = [
    ("D: PPO-CA vs Base",    "hiddensim_ppo_ca_vs_rf20sft.npy"),
    ("E: PPO-adj vs Base",   "hiddensim_ppo_adj_vs_rf20sft.npy"),
    ("F: merged_a07 vs Base","hiddensim_merged_a07_vs_rf20sft.npy"),
    ("G: PPO-adj50 vs Base", "hiddensim_ppo_step50_vs_rf20sft.npy"),
]

# Load data
heatmaps = {}
for label, fname in panels:
    arr = np.load(OUT_DIR / fname)       # (n_layers, n_t, batch, horizon)
    heatmaps[label] = arr.mean(axis=(2, 3))  # (n_layers, n_t)

# Unified scale
vmin = min(h.min() for h in heatmaps.values())
vmax = max(h.max() for h in heatmaps.values())
print(f"Unified vmin={vmin:.4f}  vmax={vmax:.4f}")

# --- 2×2 combined figure ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, (label, _) in zip(axes.flat, panels):
    im = ax.imshow(
        heatmaps[label], aspect="auto", cmap="YlOrRd",
        vmin=vmin, vmax=vmax,
        extent=[T_STEPS[0], T_STEPS[-1], NUM_LAYERS - 1, 0],
    )
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Layer Index")
    ax.set_title(f"Hidden State Distance Heatmap\n{label}")
    plt.colorbar(im, ax=ax, label="HiddenSim")

plt.suptitle(f"Unified scale  [vmin={vmin:.4f}, vmax={vmax:.4f}]", fontsize=12)
plt.tight_layout()
out = OUT_DIR / "heatmap_DEFG_unified.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# --- individual files with unified scale ---
for label, _ in panels:
    letter = label.split(":")[0]
    safe = label.replace(" ", "_").replace(":", "").replace("/", "_")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        heatmaps[label], aspect="auto", cmap="YlOrRd",
        vmin=vmin, vmax=vmax,
        extent=[T_STEPS[0], T_STEPS[-1], NUM_LAYERS - 1, 0],
    )
    plt.colorbar(im, ax=ax, label="HiddenSim")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Layer Index")
    ax.set_title(f"Hidden State Distance Heatmap\n{label}")
    plt.tight_layout()
    out = OUT_DIR / f"heatmap_unified_{safe}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

"""
跨基底链式叠加融合（5 任务，仿 doc1 方案 C，beat_block_hammer 为锚点任务）：

theta1 = PPO_bbh                                  (bbh 系数=1，锚点)
theta2 = theta1 + 0.7 * (PPO_pps - theta_SFT20k)
theta3 = theta2 + 0.7 * (PPO_pal - theta_SFT20k)
theta4 = theta3 + 0.7 * (PPO_par - theta_SFT20k)
theta5 = theta4 + 0.7 * (PPO_mcp - theta_SFT20k)

只保存最终 theta5。
"""
import torch, pathlib, logging, gc
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PREFIXES = (
    "paligemma_with_expert.gemma_expert.model.layers.",
    "paligemma_with_expert.gemma_expert.model.norm.",
    "action_in_proj.", "action_out_proj.",
    "time_mlp_in.", "time_mlp_out.",
)

SFT_BASE = "/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/20000/model.safetensors"
BBH = "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260625-11:37:02-moe_sft20k_beat_block_hammer_ppo100/moe_sft20k_beat_block_hammer_ppo100/checkpoints/global_step_300/actor/model_state_dict/full_weights.pt"
PPS = "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260703-21:07:14-moe_sft20k_place_phone_stand_ppo100/moe_sft20k_place_phone_stand_ppo100/checkpoints/global_step_300/actor/model_state_dict/full_weights.pt"
PAL = "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260703-21:07:14-moe_sft20k_place_a2b_left_ppo100/moe_sft20k_place_a2b_left_ppo100/checkpoints/global_step_300/actor/model_state_dict/full_weights.pt"
PAR = "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260705-10:45:17-moe_sft20k_place_a2b_right_ppo100/moe_sft20k_place_a2b_right_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"
MCP = "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260703-00:38:41-moe_sft20k_move_can_pot_ppo100/moe_sft20k_move_can_pot_ppo100/checkpoints/global_step_400/actor/model_state_dict/full_weights.pt"
OUT = "/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_sft20k_ppo5task_chain_bbh_a07/full_weights.pt"
ALPHA = 0.7


def load(p):
    if str(p).endswith(".safetensors"):
        sd = load_file(p)
    else:
        sd = torch.load(p, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and len(sd) == 1:
            sd = sd["state_dict"]
    logging.info(f"  {len(sd)} tensors <- {p}")
    return sd


logging.info("Loading SFT20k base (task-vector reference)...")
sft_sd = load(SFT_BASE)
eligible = {k for k in sft_sd if any(k.startswith(p) for p in PREFIXES)}
logging.info(f"Eligible keys: {len(eligible)} / {len(sft_sd)}")

logging.info("Loading PPO_bbh (chain anchor, theta1)...")
theta = load(BBH)

for name, path in [("pps", PPS), ("pal", PAL), ("par", PAR), ("mcp", MCP)]:
    logging.info(f"Folding in {name} (alpha={ALPHA})...")
    ft_sd = load(path)
    n_changed = 0
    with torch.no_grad():
        for k in eligible:
            if k in ft_sd and k in sft_sd and k in theta:
                base_f = theta[k].to(torch.float32)
                ft_f = ft_sd[k].to(torch.float32)
                sft_f = sft_sd[k].to(torch.float32)
                result = base_f + ALPHA * (ft_f - sft_f)
                new_val = result.to(theta[k].dtype)
                if not torch.equal(new_val, theta[k]):
                    n_changed += 1
                theta[k] = new_val
    logging.info(f"  changed {n_changed} / {len(eligible)} eligible tensors")
    del ft_sd
    gc.collect()

del sft_sd
gc.collect()

pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
torch.save(theta, OUT)
logging.info(f"Saved -> {OUT}")

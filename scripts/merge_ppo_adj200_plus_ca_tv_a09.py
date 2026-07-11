"""
theta* = theta_ppo_adj200 + alpha * (theta_ppo_CA_100 - theta_rf20sft)

Load weights one layer at a time to reduce peak memory.
"""
import torch, pathlib, logging, gc
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PREFIXES = (
    "paligemma_with_expert.gemma_expert.model.layers.",
    "paligemma_with_expert.gemma_expert.model.norm.",
    "action_in_proj.", "action_out_proj.",
    "time_mlp_in.", "time_mlp_out.",
)

BASE  = "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260615-15:45:43-robotwin15_adjust_bottle_ppo_random20/robotwin15_ppo_adjust_bottle_random20/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt"
CA100 = "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260617-13:27:58-rf20sft_ppo_click_alarmclock/rf20sft_ppo_click_alarmclock/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt"
RF20  = "/mnt/petrelfs/muyao/jiayusun/random_first20_pi05_sft/model_state_dict/full_weights.pt"
OUT   = "/mnt/petrelfs/muyao/jiayusun/merged_weights/ppo_adj200_plus_ca100_tv_a09/full_weights.pt"
ALPHA = 0.9

def load(p):
    sd = torch.load(p, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and len(sd) == 1:
        sd = sd["state_dict"]
    logging.info(f"  {len(sd)} tensors <- {p}")
    return sd

logging.info("Loading ppo_adj200 (base)...")
base_sd = load(BASE)
logging.info("Loading ppo_CA_100...")
ca100_sd = load(CA100)
logging.info("Loading rf20sft (task vector reference)...")
rf20_sd = load(RF20)

eligible = {k for k in base_sd if any(k.startswith(p) for p in PREFIXES)}
logging.info(f"Eligible keys: {len(eligible)} / {len(base_sd)}")

merged = {}
n_changed = 0
with torch.no_grad():
    for k, v in base_sd.items():
        if k in eligible and k in ca100_sd and k in rf20_sd:
            base_f   = v.to(torch.float32)
            ca_f     = ca100_sd[k].to(torch.float32)
            rf20_f   = rf20_sd[k].to(torch.float32)
            result   = base_f + ALPHA * (ca_f - rf20_f)
            merged[k] = result.to(v.dtype)
            if not torch.equal(merged[k], v):
                n_changed += 1
        else:
            merged[k] = v

del base_sd, ca100_sd, rf20_sd; gc.collect()
logging.info(f"Changed {n_changed} / {len(merged)} tensors vs ppo_adj200 base")
pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
torch.save(merged, OUT)
logging.info(f"Saved -> {OUT}")

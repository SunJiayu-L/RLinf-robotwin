# RoboTwin 23-task Pi05 LoRA SFT

The experiment config is `examples/sft/config/robotwin23_sft_openpi_pi05_lora.yaml`.
It starts from `/data2/sjy/checkpoints/openpi/pi05_base_pytorch`, with no SFT
checkpoint resume. `actor.model.lora_scope: policy` enables LoRA on both
PaliGemma and the action expert while freezing original parameters. Rank 32
leaves 87,030,784 of 3,703,788,304 parameters trainable (2.35%). This option
is implemented in `rlinf/models/__init__.py`; the default `vlm` scope retains
the existing RLinf behavior.

Training uses physical GPUs 4–7, Ray with 64 CPUs, micro batch 4 per GPU, and
global batch 16. It targets 130,000 optimizer steps and saves every 10,000
steps directly under
`/data2/sjy/archive/robotwin23_sft_pi05_lora_r32/checkpoints/`.
Launch script: `/data2/sjy/run_robotwin23_sft_pi05_lora.sh`.
Log: `/data2/sjy/logs/robotwin23_sft_pi05_lora_r32.log`.

The earlier full-parameter experiment is stopped; its 30k and 40k checkpoints
remain in `/data2/sjy/archive/robotwin23_sft_pi05_base/checkpoints/`.
Do not use those checkpoints to resume this LoRA experiment. For a LoRA resume,
use the newest complete LoRA DCP checkpoint and the same config. Supply
`+runner.resume_dir=<checkpoint-directory>` to Hydra.

Validation: a 3-step run with micro batch 2 and global batch 16 completed
forward/backward and wrote DCP metadata, four shards, and `full_weights.pt`.
Its warm steady training step took about 12 seconds. The formal run with micro
batch 4 reached step 13 at about 6 seconds per step, with 15.5–15.7 GiB used
on each training GPU. With batch 16, 130k steps process twice as many training
examples as the earlier batch-8 plan.

Use `examples/embodiment/config/robotwin23_sft_lora_eval_openpi_pi05.yaml`
for evaluation, passing a completed LoRA `full_weights.pt` as
`runner.ckpt_path`. Keep the base `model_path` set to the original Pi05 weights
so adapter structure is built before checkpoint weights are loaded.

Smoke evaluation of the 3-step checkpoint completed on 8
`beat_block_hammer` trajectories with exit code 0. It loaded the saved LoRA
weights and ran RoboTwin rollout end to end. Success was 0/8, which is only a
pipeline check after three training steps, not a model quality result. Log:
`/data2/sjy/logs/robotwin23_sft_pi05_lora_smoke_eval_retry.log`.

## Batch-48 restart (2026-10-22)

The original batch-16 LoRA run was stopped at about step 4,392 before its
first 10,000-step checkpoint. Its logs are retained. The new experiment starts
again from the original Pi05 base checkpoint, using
`examples/sft/config/robotwin23_sft_openpi_pi05_lora_bs48.yaml` and
`/data2/sjy/run_robotwin23_sft_pi05_lora_bs48.sh`.

The new per-GPU micro batch is 12 and global batch is 48. Separate three-step
pilots completed forward, backward, and checkpoint saving at micro batches 10
and 12. GPU 4–7 peak usage for micro 10 was 19,585, 19,606, 19,540, 19,684
MiB (about 80%). For micro 12 it was 22,209, 22,070, 22,268, 22,086 MiB
(89.8–90.7% of 24,564 MiB). The formal run uses micro 12 and retains the
130,000-step goal and 10,000-step save interval. Logs:
`/data2/sjy/logs/robotwin23_sft_pi05_lora_r32_bs48.log`; checkpoints:
`/data2/sjy/archive/robotwin23_sft_pi05_lora_r32_bs48/checkpoints/`.

With 48 samples per optimizer step, 130,000 steps expose the model to three
times as many examples as the previous batch-16 LoRA plan. When resuming,
use only a complete DCP checkpoint from this batch-48 experiment and the same
config; never resume from the old full-parameter or batch-16 runs.

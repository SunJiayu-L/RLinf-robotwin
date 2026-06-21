# random10 h16 SFT 5k evaluation

## Scope

Evaluate the `global_step_5000` checkpoint from the random10 h16/no-adapt SFT run.

The dataset contains these ten behavior tasks:

`adjust_bottle`, `beat_block_hammer`, `blocks_ranking_rgb`,
`blocks_ranking_size`, `click_alarmclock`, `click_bell`, `dump_bin_bigbin`,
`handover_block`, `handover_mic`, and `hanging_mug`.

Only the six tasks with non-empty entries in `eval_seeds.json` are submitted:

`adjust_bottle`, `beat_block_hammer`, `blocks_ranking_rgb`, `click_bell`,
`handover_block`, and `handover_mic`.

The other four tasks are deliberately skipped. The job preflight fails instead
of falling back to random seeds if a task key or `success_seeds` is absent.

## Evaluation alignment

- 8 GPUs and 56 environments, seven environments per worker.
- Fixed reset state IDs from `eval_seeds.json`.
- Pi0.5 h16, five denoising steps, `adapt_to_pi=False`.
- Norm stats from the random10 training dataset are linked into the 5k actor.
- Gripper scaling is disabled: PJLab random10 state/action gripper dimensions
  have q01=0 and q99=0.9998, so they are already in native `[0,1]` units.

## Files

- `examples/embodiment/config/random10_h16_sft5k_eval.yaml`
- `examples/embodiment/slurm/random10_h16_sft5k_eval_task.sbatch`
- `examples/embodiment/slurm/submit_random10_h16_sft5k_eval.sh`

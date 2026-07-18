
# 2026-6-19 random10 h16 no-adapt SFT experiment log



## Goal



Run a PJLab-side SFT experiment that is closer to the effective A800 configuration:



- use only complete random RoboTwin tasks;

- train 10 tasks first;

- use Pi0.5 h16 action horizon instead of h50;

- use `adapt_to_pi: False` and `extra_delta_transform: True`;

- calculate matching norm stats before SFT;

- submit through Slurm, not from the login node.



This is intended to avoid comparing against the old PJLab h50/adapt=True path, which is not the same as the A800 h16/no-adapt path that produced better results.



## Dataset status



PJLab and A800 are separate clusters. The A800 path under `/mnt/pfs/scalelab2/...` is not visible on PJLab.



PJLab local dataset used here:



```text

/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_random_47_500_pi05

```



Checked `meta/episodes.jsonl`:



```text

total episodes: 8816

min episode_index: 0

max episode_index: 8815

missing episode_index in 0..8815: 0

```



Assuming 500 episodes per source task:



```text

task 00-16: complete, 500 episodes each

task 17: partial, 316 episodes, episode 8500-8815

task 18-46: empty

```



Therefore this experiment uses the first 10 complete tasks:



```text

episode range: 0-4999

output dataset: /mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_random_10_500_pi05_h16

```



## Code/config changes



Added a new OpenPI data config in:



```text

rlinf/models/embodiment/openpi/dataconfig/__init__.py

```



New config name:



```text

pi05_aloha_robotwin_random10_h16

```



Key settings:



```python

Pi0Config(pi05=True, action_horizon=16, discrete_state_input=True)

repo_id="physical-intelligence/robotwin_random10_h16"

adapt_to_pi=False

extra_delta_transform=True

```



Added SFT config:



```text

examples/sft/config/random10_sft_openpi_pi05_h16_noadapt.yaml

```



Important fields:



```yaml

data.train_data_paths: /mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_random_10_500_pi05_h16

actor.micro_batch_size: 16

actor.global_batch_size: 64

actor.openpi_data.adapt_to_pi: False

actor.openpi_data.extra_delta_transform: True

actor.model.model_path: /mnt/petrelfs/muyao/jiayusun/pi05_base_adapt_false

actor.model.num_action_chunks: 16

actor.model.openpi_data.adapt_to_pi: False

actor.model.openpi_data.extra_delta_transform: True

actor.model.openpi.config_name: pi05_aloha_robotwin_random10_h16

runner.max_steps: 30000

runner.save_interval: 5000

```



Added Slurm script:



```text

examples/sft/slurm/random10_pi05_sft_h16_noadapt_8gpu.sbatch

```



Execution order:



```text

1. preflight: verify source dataset has episode 0-4999

2. filter source dataset to first 10 complete tasks

3. calculate no-adapt norm stats

4. launch 8-GPU SFT

```



Norm stats output path:



```text

/mnt/petrelfs/muyao/jiayusun/pi05_base_adapt_false/physical-intelligence/robotwin_random10_h16/norm_stats.json

```



## Slurm submission history



Several early submissions failed because the generated sbatch script lost shell line-continuation backslashes. These were script-generation issues, not training failures.



```text

9805573: FAILED immediately. apptainer exec line was malformed.

9805583: FAILED after container launch. filter_lerobot_dataset.py arguments were split incorrectly.

9805587: cancelled before running after finding the same multiline command issue.

```



The sbatch script was then rewritten so the Apptainer launch, dataset filter command, norm-stats command, and SFT command are single-line shell commands.



Current valid job:



```text

job id: 9805588

name: r10-h16-sft

partition: wam_data

node: SH-IDCA1404-10-140-54-39

state as of 2026-06-19 19:04 CST: RUNNING

```



Latest Slurm log:



```text

/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/random10_pi05_sft_h16_noadapt_8gpu-9805588.out

```



Log status as of 2026-06-19 19:04 CST:



```text

=== [preflight] check random dataset completeness: Fri 19 Jun 2026 06:07:58 PM CST ===

source episodes=8816, max_episode=8815, first_10_tasks=OK

=== [step1] filter first 10 random tasks: Fri 19 Jun 2026 06:07:58 PM CST ===

```



The output dataset currently exists and is being populated. At the last check:



```text

parquet files written: 4459 / 5000

norm_stats.json: not created yet

SFT log dir: not created yet

```



This means the job is still in dataset filtering. It has not yet reached norm stats or SFT training.



## Notes / risks



- The current `filter_lerobot_dataset.py` rewrites every selected parquet and converts through pandas, so filtering 5000 episodes is slow. This is expected but inefficient.

- The source dataset stores many unique natural-language prompts in `tasks.jsonl`; the real source-task completeness check was done by episode-index blocks of 500, not by counting entries in `tasks.jsonl`.

- The first 17 tasks are complete; the 18th block is partial. Do not use episode ranges past `8499` unless intentionally accepting partial tasks.

- The two `/usr/bin/ln` warnings about `/usr/local/nvidia/...` already existing come from container/NVIDIA setup and are not considered fatal.

- If the job reaches SFT, check that the checkpoint contains or can resolve the matching norm stats under `physical-intelligence/robotwin_random10_h16` before evaluation.



## Monitoring commands



```bash

squeue -j 9805588

sacct -j 9805588 --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList%30

tail -f /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/random10_pi05_sft_h16_noadapt_8gpu-9805588.out

find /mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_random_10_500_pi05_h16 -maxdepth 3 -type f -name '*.parquet' | wc -l

ls -l /mnt/petrelfs/muyao/jiayusun/pi05_base_adapt_false/physical-intelligence/robotwin_random10_h16/norm_stats.json

```


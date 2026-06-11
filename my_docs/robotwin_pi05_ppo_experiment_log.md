# RoboTwin PI05 PPO 实验日志

日期：2026-06-03

## 目标

在 RLinf 框架中使用 PPO 微调 OpenPI PI05 模型，任务为：

- `beat_block_hammer`
- `put_bottles_dustbin`

两个任务分别作为独立 PPO 作业训练，不做一个 run 内的多任务混训。

## 代码与环境

- RLinf 仓库：`/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin`
- RoboTwin 仓库：`/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf`
- 容器镜像：`/mnt/petrelfs/muyao/shared_img/new_app.sif`
- 运行方式：计算节点内使用 `apptainer exec --cleanenv --bind /mnt:/mnt --bind /nvme:/nvme --nv`
- 机器人平台：`ALOHA`
- RoboTwin assets：`/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf`

关键运行环境变量：

```bash
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## 初始化权重

使用之前 RoboTwin-47 PI05 SFT 的 20k checkpoint 作为 PPO 初始化。

JAX checkpoint：

```bash
/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/pi05_aloha_full_base/robotwin_47_pi05_full/20000
```

转换后的 PyTorch 权重：

```bash
/mnt/petrelfs/muyao/jiayusun/store/openpi-pt-converted/robotwin_47_pi05_full_20k
```

已确认包含：

```text
config.json
model.safetensors
physical-intelligence/robotwin/norm_stats.json
```

转换作业：

```text
9660406  pi05-rw47-convert  FAILED
9660428  pi05-rw47-convert  COMPLETED
```

第一次失败原因：转换脚本使用了 RLinf 模型配置名 `pi05_aloha_robotwin`，但原始 OpenPI converter 读取的是 OpenPI SFT 代码中的 config，找不到该名称。修复为使用 RoboTwin SFT 源码路径和 `pi05_aloha_full_base`。

## 配置文件

新增/使用的配置：

```bash
examples/embodiment/config/env/robotwin_put_bottles_dustbin.yaml
examples/embodiment/config/robotwin_beat_block_hammer_ppo_openpi_pi05.yaml
examples/embodiment/config/robotwin_put_bottles_dustbin_ppo_openpi_pi05.yaml
examples/embodiment/slurm/convert_robotwin47_pi05_20k.sbatch
examples/embodiment/slurm/train_robotwin_pi05_ppo_one.sbatch
```

关键 PPO 覆盖参数：

```yaml
actor.model.model_path: /mnt/petrelfs/muyao/jiayusun/store/openpi-pt-converted/robotwin_47_pi05_full_20k
actor.model.action_dim: 14
actor.model.num_action_chunks: 50
actor.model.openpi.config_name: pi05_aloha_robotwin
actor.model.openpi.num_images_in_input: 3

env.*.assets_path: /mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf
env.*.center_crop: False
env.*.task_config.embodiment: [aloha-agilex]
env.*.task_config.camera.collect_wrist_camera: true
```

任务步长：

```text
beat_block_hammer: 400
put_bottles_dustbin: 1700
```

Seeds 已存在：

```text
beat_block_hammer: train 1000, eval 320
put_bottles_dustbin: train 996, eval 200
```

## 提交记录

初始 array 提交：

```text
9660407_[0-1]  pi05-rw-ppo  CANCELLED
```

取消原因：依赖的转换作业 `9660406` 失败，导致 `DependencyNeverSatisfied`。

第一次独立提交：

```text
9660429  pi05-beat-ppo  COMPLETED 0:0
9660431  pi05-put-ppo   COMPLETED 0:0
```

这两个 Slurm 状态显示成功，但实际 RLinf 内部失败；原因是 `run_embodiment.sh` 使用 pipe 到 `tee`，未设置 `pipefail`，导致 Python 失败被吞掉。

第二次提交：

```text
9661712  pi05-beat-ppo  FAILED 127:0
9661713  pi05-put-ppo   FAILED 127:0
```

失败原因仍与 RoboTwin 环境初始化相关。

当前正式运行：

```text
9661825  pi05-beat-ppo  RUNNING  SH-IDCA1404-10-140-54-73
9661897  pi05-put-ppo   RUNNING  SH-IDCA1404-10-140-54-120
```

其中 `9661826` 是 `put_bottles_dustbin` 的一次失败重试：

```text
9661826  pi05-put-ppo  FAILED 127:0  SH-IDCA1404-10-140-54-44
```

该次失败原因是 GPU OOM，日志显示 GPU 0 仅剩 5.62 MiB，已有进程占用约 72.58 GiB。后续重提时排除了节点：

```text
SH-IDCA1404-10-140-54-44
SH-IDCA1404-10-140-54-81
```

## 排障记录

### 1. 转换配置名错误

错误：

```text
ValueError: Config 'pi05_aloha_robotwin' not found
```

原因：converter 使用的是 OpenPI 原始训练 config，不是 RLinf 的 OpenPI dataconfig。

修复：

```bash
export PYTHONPATH=/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/src:$REPO_PATH:$PYTHONPATH
--config_name pi05_aloha_full_base
```

### 2. Slurm 成功但训练实际失败

原因：`run_embodiment.sh` 中 `python ... | tee ...` 未设置 `pipefail`。

修复：

```bash
set -o pipefail
```

### 3. SAPIEN 渲染设备初始化失败

错误：

```text
RuntimeError: failed to find a rendering device
```

修复一：RLinf GPU worker 环境变量增加：

```python
EGL_VISIBLE_DEVICES = visible_accelerators[0]
```

修复二：RoboTwin `SubEnv.setup_task()` 初始清理时不清 SAPIEN cache：

```python
self.close(clear_cache=False)
```

修改文件：

```bash
rlinf/scheduler/hardware/accelerators/nvidia_gpu.py
examples/embodiment/run_embodiment.sh
/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf/robotwin/envs/vector_env.py
```

### 4. 节点 GPU OOM

`9661826` 在 `SH-IDCA1404-10-140-54-44` 上失败：

```text
torch.OutOfMemoryError: CUDA out of memory
```

处理：重提 `put_bottles_dustbin`，排除 `54-44` 和已知坏节点 `54-81`，得到当前运行 job `9661897`。

## 当前日志路径

```bash
slurm_logs/pi05-rw-ppo-pi05-beat-ppo-9661825.out
slurm_logs/pi05-rw-ppo-pi05-beat-ppo-9661825.err
slurm_logs/pi05-rw-ppo-pi05-put-ppo-9661897.out
slurm_logs/pi05-rw-ppo-pi05-put-ppo-9661897.err
```

RLinf 训练日志目录：

```bash
logs/*robotwin_beat_block_hammer_ppo_openpi_pi05*
logs/*robotwin_put_bottles_dustbin_ppo_openpi_pi05*
```

## 后续检查项

- 确认两个 job 持续运行，未在环境 reset 或 rollout 初始化阶段退出。
- 检查 `logs/.../run_embodiment.log` 中是否出现 rollout、reward、actor update 指标。
- 检查是否按 `runner.save_interval: 10` 保存 checkpoint。
- 如再次 OOM，优先降低 `env.train.total_num_envs` 或排除异常占用节点。

# RoboTwin PI05 SFT（RLinf）实验日志

日期：2026-06-07

## 目标

在 RLinf 框架中对 PI05 (`pi05_base`) 进行全量 47 任务 SFT 微调，使用 `robotwin_47_pi05` 数据集，产出可用于后续 PPO 的初始化 checkpoint。

## 环境

- RLinf 仓库：`/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin`
- 容器镜像：`/mnt/petrelfs/muyao/shared_img/new_app.sif`
- 运行方式：Slurm + `apptainer exec --cleanenv --nv --bind /mnt:/mnt --bind /nvme:/nvme`

## 数据集

- LeRobot 格式数据集：`robotwin_47_pi05`
- 本地路径：`/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_47_pi05`
- 任务数：47（单臂 + 双臂混合）
- 总 episodes：4700（每任务 100）
- 总帧数：1,047,296
- FPS：50
- 状态/动作维度：14（每臂 6 关节 + 1 夹爪 × 2）
- 摄像头：`cam_high`、`cam_left_wrist`、`cam_right_wrist`（各 `[3, 480, 640]`）
- 机器人型号：**aloha-agilex**

## 初始化权重

- 来源：OpenPI 官方 `pi05_base` checkpoint（PyTorch 格式，非 JAX）
- 路径：`/mnt/petrelfs/muyao/jiayusun/pi05_base`

## Norm Stats 计算

### 背景

RLinf SFT 使用 `pi05_aloha_robotwin` 配置（`adapt_to_pi=True`），与之前 openpi 原生训练中的 `pi05_aloha_full_base` 配置（`adapt_to_pi=False`）不同。
`adapt_to_pi=True` 对数据做了以下变换：

1. **符号翻转**（`_joint_flip_mask`）：关节维度 [1,2,8,9] 乘以 -1
2. **Gripper 非线性变换**（`_gripper_to_angular`）：维度 [6,13] 从 linear space 转换到 angular space

因此旧 openpi 的 norm_stats（`adapt_to_pi=False`）不能直接复用，需重新计算。

### 计算方法

使用 `toolkits/lerobot/calculate_norm_stats.py`（已添加 `--output-path` 参数）：

```bash
python toolkits/lerobot/calculate_norm_stats.py \
    --config-name pi05_aloha_robotwin \
    --repo-id /mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_47_pi05 \
    --output-path /mnt/petrelfs/muyao/jiayusun/pi05_base/physical-intelligence/robotwin
```

Slurm 脚本：`examples/sft/slurm/calc_norm_stats_robotwin47.sbatch`

- 资源：1 节点，16 CPU，0 GPU
- 批量大小：32（openpi 默认），`num_workers=2`
- 耗时：约 18 小时（~1,047,296 / 32 = 32,728 个 batch）
- Job ID：**9729926**（2026-06-06 提交，2026-06-07 完成）

> **注意（踩坑）**：`num_workers` 不能设置过大。容器内 `/dev/shm` 受限，图像数据批量加载时会触发共享内存耗尽导致死锁。测试过 `num_workers=32`、`num_workers=4`（+ `batch_size=256`）均挂起，最终以原始默认值 `num_workers=2, batch_size=32` 运行成功。

### 计算结果

输出路径：`/mnt/petrelfs/muyao/jiayusun/pi05_base/physical-intelligence/robotwin/norm_stats.json`

**State（14 维）：**

| 维度 | 含义 | mean | std |
|------|------|------|-----|
| 0 | 左臂 joint 0 | -0.2385 | 0.4312 |
| 1 | 左臂 joint 1 (flipped) | -1.1588 | 1.0008 |
| 2 | 左臂 joint 2 (flipped) | -0.8363 | 0.7785 |
| 3 | 左臂 joint 3 | -0.3356 | 0.6675 |
| 4 | 左臂 joint 4 | 0.0653 | 0.2542 |
| 5 | 左臂 joint 5 | -0.0519 | 0.5831 |
| 6 | 左臂 gripper (angular) | 0.2481 | 0.9211 |
| 7 | 右臂 joint 0 | 0.1928 | 0.3394 |
| 8 | 右臂 joint 1 (flipped) | -1.0162 | 1.0267 |
| 9 | 右臂 joint 2 (flipped) | -0.7483 | 0.7913 |
| 10 | 右臂 joint 3 | -0.3067 | 0.6645 |
| 11 | 右臂 joint 4 | -0.0314 | 0.2282 |
| 12 | 右臂 joint 5 | 0.0195 | 0.5155 |
| 13 | 右臂 gripper (angular) | 0.3364 | 0.8902 |

**Actions（14 维）：**

| mean | [-0.0158, -0.0568, -0.0396, -0.0075, 0.0053, -0.0128, 0.1648, 0.0236, -0.0771, -0.0524, -0.0071, -0.0044, 0.0125, 0.2699] |
|------|-----|
| std | [0.2411, 0.5436, 0.4308, 0.3617, 0.1448, 0.3258, 0.9838, 0.2021, 0.5072, 0.4077, 0.3337, 0.1498, 0.2994, 0.9514] |

## SFT 训练配置

配置文件：`examples/sft/config/robotwin_sft_openpi_pi05.yaml`

关键参数：

| 参数 | 值 |
|------|-----|
| `openpi.config_name` | `pi05_aloha_robotwin` |
| `model_path` | `/mnt/petrelfs/muyao/jiayusun/pi05_base` |
| `micro_batch_size` | 32 |
| `global_batch_size` | 64 |
| `max_steps` | 30000 |
| `save_interval` | 5000 |
| `lr` | 2.5e-5 |
| `lr_scheduler` | cosine（warmup 1000 steps） |
| `min_lr` | 2.5e-6 |
| `fsdp.sharding_strategy` | `no_shard` |
| `gradient_checkpointing` | False（openpi 不支持） |
| `train_expert_only` | False（全量微调） |

## SFT 作业

### 4 GPU 版本

- Slurm 脚本：`examples/sft/slurm/robotwin_pi05_sft.sbatch`
- Job ID：**9737302**
- 资源：1 节点，4 GPU，32 CPU
- 提交时间：2026-06-07
- 日志：`slurm_logs/robotwin_pi05_sft-9737302.out`
- 状态：运行中

### 8 GPU 版本

- Slurm 脚本：`examples/sft/slurm/robotwin_pi05_sft_8gpu.sbatch`
- Job ID：**9737305**
- 资源：1 节点，8 GPU，64 CPU
- 提交时间：2026-06-07
- 日志：`slurm_logs/robotwin_pi05_sft_8gpu-9737305.out`
- 状态：排队中（Quota）

## Checkpoint 输出

Checkpoint 将保存在：

```
{LOG_DIR}/checkpoints/global_step_5000/
{LOG_DIR}/checkpoints/global_step_10000/
...
{LOG_DIR}/checkpoints/global_step_30000/
```

其中 `LOG_DIR` = `logs/{timestamp}-robotwin_sft_pi05[_8gpu]/`

## 后续计划

1. 训练完成后选取最优 checkpoint（建议先评估 step 20000/25000/30000）
2. 将 SFT checkpoint 作为 PPO 初始化权重，替换当前 PPO 实验中的初始化
3. 对比 SFT-only 和 SFT+PPO 在 47 任务上的成功率

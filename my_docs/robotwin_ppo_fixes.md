# RoboTwin PPO 运行修复记录

日期：2026-06-04

本文记录将 RLinf + RoboTwin + OpenPI PI05 PPO 训练跑通所需的全部修复，包括每项修复的原因和验证方法。

---

## 修复一：SAPIEN EGL 渲染设备找不到

**现象**

EnvWorker 初始化时报：

```
RuntimeError: failed to find a rendering device
```

**根因**

RLinf 的 `NvidiaGPUManager.get_accelerator_env_var()` 设置了 `CUDA_VISIBLE_DEVICES`，但未同步设置 SAPIEN/EGL 所需的 `EGL_VISIBLE_DEVICES`。容器内 EGL 后端无法自动感知 Ray 分配的 GPU。

**修复**

文件：`rlinf/scheduler/hardware/accelerators/nvidia_gpu.py`

```python
if len(visible_accelerators) > 0:
    env_vars["MUJOCO_EGL_DEVICE_ID"] = str(visible_accelerators[0])
    env_vars["EGL_VISIBLE_DEVICES"] = str(visible_accelerators[0])   # 新增
```

**验证**

提交 sapien-render-test（job 9668239/9668243），对比有无 `EGL_VISIBLE_DEVICES`：

- 无 `EGL_VISIBLE_DEVICES` → `failed to find a rendering device`
- 有 `EGL_VISIBLE_DEVICES=0` → `Render Well`

---

## 修复二：容器内 Vulkan ICD 无效导致 SAPIEN 渲染失败

**现象**

容器内默认 Vulkan ICD 路径 `/etc/vulkan/icd.d/nvidia_icd.json` 不包含有效驱动，SAPIEN Vulkan 后端初始化失败。

**根因**

容器的 `/etc/vulkan/icd.d/` 目录为空或驱动路径失效。需要自定义指向 `libGLX_nvidia.so.0`（该 SO 通过 `LD_LIBRARY_PATH` 可寻址）的 ICD 文件。

**修复**

1. 在仓库根目录新增 `nvidia_icd.json`：

```json
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3.194"
    }
}
```

2. 在 sbatch 脚本中设置（`examples/embodiment/slurm/train_robotwin_pi05_ppo_one.sbatch`）：

```bash
export VK_DRIVER_FILES="${REPO_PATH}/nvidia_icd.json"
export VK_ICD_FILENAMES="${REPO_PATH}/nvidia_icd.json"
```

**验证**

sapien-render-test：使用自定义 ICD 文件后渲染成功。

---

## 修复三：EnvWorker 全局 barrier 阻塞 RoboTwin 初始化

**现象**

训练启动后日志停在 `Waiting for env workers`，长时间无进展。

**根因**

`rlinf/workers/env/env_worker.py` 的 `initialize()` 中有一个全局 broadcast barrier，原为 RealWorld 环境（ROS 节点初始化）设计。RoboTwin 下各 EnvWorker 启动时间不同步，导致死锁。

**修复**

文件：`rlinf/workers/env/env_worker.py`

```python
env_types = {
    str(env_cfg.env_type)
    for env_cfg in (self.cfg.env.get("train", None), self.cfg.env.eval)
    if env_cfg is not None
}
if "realworld" in env_types:
    self.broadcast(True, groups=[(self._group_name, list(range(self._world_size)))])
```

只有 RealWorld 环境才触发 barrier，其他环境（包括 RoboTwin）跳过。

**验证**

job 9668436：EnvWorker 约 4 分钟内完成初始化，顺利进入 Actor 阶段。

---

## 修复四：run_embodiment.sh 管道吞掉 Python 退出码

**现象**

Slurm job 显示 `COMPLETED (0:0)`，但实际 Python 训练已失败，错误被静默忽略。

**根因**

`python ... | tee -a log` 管道中，bash 默认只检查最右侧命令（`tee`）的退出码。

**修复**

文件：`examples/embodiment/run_embodiment.sh`，脚本头部添加：

```bash
set -o pipefail
```

---

## 修复五：RoboTwin SubEnv 重置时清除 SAPIEN cache

**现象**

多次 env reset 后偶发 SAPIEN 渲染设备失效。

**根因**

`SubEnv.setup_task()` 调用 `self.close(clear_cache=True)` 清除 SAPIEN 场景缓存，再重建时触发 EGL context 重初始化，某些节点上第二次初始化失败。

**修复**

文件：`/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf/robotwin/envs/vector_env.py`

```python
def setup_task(self):
    self.close(clear_cache=False)   # 不清 SAPIEN cache
```

---

## 修复六：TORCH_DISTRIBUTED_DEBUG=DETAIL 导致 gloo 进程组 collective 冲突

**现象**

Actor workers 初始化完成后立即崩溃，日志报：

```
Broadcast failed on ProcessGroup cg-ActorGroup:0-RolloutGroup:0gloo_recv_0 rank 0:
  Rank 0 is running BROADCAST(src=1, TensorShape=[1], Int)
  Rank 1 is running ALLGATHER_COALESCED
TypeError: tuple indices must be integers or slices, not str
```

**根因**

`TORCH_DISTRIBUTED_DEBUG=DETAIL` 使 PyTorch 用 `_ProcessGroupWrapper` 包裹所有 gloo 进程组。每次 collective 前，wrapper 在同一进程组上先调 `all_gather_coalesced` 做类型校验（SequenceNumber=0），与 RLinf 后台 recv 线程同时在 `gloo_recv_0` 上调 `broadcast(src=1)` 竞争，产生 mismatch。

**修复**

文件：`examples/embodiment/slurm/train_robotwin_pi05_ppo_one.sbatch`

```bash
# 修改前
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# 修改后
export TORCH_DISTRIBUTED_DEBUG=OFF
```

**验证**

job 9668436：关闭后完全无 ALLGATHER_COALESCED 错误，Actor workers 初始化完成，权重同步成功，rollout 正常启动，首轮 PPO update 完成并产生训练指标。

---

## 修复七：PatchWeightSyncer init_sync 禁用

**说明**

`examples/embodiment/config/weight_syncer/patch_syncer.yaml` 中将 `init_sync.enabled` 由 `True` 改为 `False`。

Actor 和 Rollout 均从同一 checkpoint 加载，初始权重已完全一致，无需 init_sync 的全量权重传输握手流程（该流程本身未造成 bug，但作为简化保留）。

```yaml
init_sync:
  enabled: False
```

---

## 已知无害警告

| 警告信息 | 来源 | 说明 |
|---------|------|------|
| `ln: failed to create symbolic link '...lib64': File exists` | apptainer 容器启动 | 符号链接冲突，不影响运行 |
| `NCCL_ASYNC_ERROR_HANDLING is deprecated` | PyTorch NCCL | 已由 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 替代 |
| `ModuleNotFoundError: No module named 'curobo.types.math'` + `missing pytorch3d` | RoboTwin planner.py | CuroboPlanner 可选，降级运行不影响训练 |
| `Hydra Defaults list is missing _self_` | hydra config | hydra 版本升级警告，不影响配置加载 |
| `The hostname of the client socket cannot be retrieved. err=-3` | gloo TCP socket | DNS 查询失败（EAI_AGAIN），实际使用 IP 建连，无害 |

---

## 训练性能参考（beat_block_hammer，job 9668436）

| 阶段 | 耗时 |
|------|------|
| 初始化（Rollout+Env+Actor workers） | ~7 分钟 |
| 每轮 rollout（4 epoch × 256 envs × 400 steps） | ~31 分钟 |
| 每轮 actor update（5 epoch × 2048 batch） | ~5 分钟 |
| 单轮 PPO 总耗时 | ~43 分钟 |

---

## 提交历史

| Slurm Job | 任务 | 结果 | 备注 |
|-----------|------|------|------|
| 9661825 | beat_block_hammer PPO | RUNNING（状态未知） | 早期运行，修复不完整 |
| 9661897 | put_bottles_dustbin PPO | RUNNING（状态未知） | 早期运行，修复不完整 |
| 9668016/55/105/156 | beat smoke tests | CANCELLED | 手动过早取消，env init 实际正常 |
| 9668239/43 | sapien-render-test | COMPLETED | 验证 EGL+Vulkan 修复 |
| 9668319 | beat PPO | CANCELLED | 遇 ALLGATHER_COALESCED 崩溃（修复六未应用） |
| 9668320 | put PPO | CANCELLED | 同上 |
| 9668400 | beat PPO | CANCELLED | 关闭 init_sync 但未关 TORCH_DISTRIBUTED_DEBUG |
| 9668401 | put PPO | CANCELLED | 同上 |
| **9668436** | **beat_block_hammer PPO** | **RUNNING** | 所有修复生效，第 2 轮 PPO 进行中 |
| **9668437** | **put_bottles_dustbin PPO** | **PENDING** | 等待节点 |

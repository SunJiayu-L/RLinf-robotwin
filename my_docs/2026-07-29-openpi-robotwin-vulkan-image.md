# OpenPI + RoboTwin Vulkan 训练镜像记录（2026-07-29）

## 已发布镜像

```text
registry.h.pjlab.org.cn/ailab-eailabagent-eailabagent_gpu/jiayusun-rlinf-robotwin:openpi-robotwin-vulkan-20260729
```

发布 digest：

```text
sha256:7b47d89ea1e0d93096e8f5b4e2f974018b5d5fd8c11590cde10682937225a298
```

构建定义：[`docker/Dockerfile.robotwin_vulkan_openpi`](../docker/Dockerfile.robotwin_vulkan_openpi)。基础镜像为
`docker.1ms.run/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`。

## 镜像内内容

- CUDA 12.4 开发环境及 NVIDIA 图形运行时。
- Vulkan SDK 1.3.296、EGL/OpenGL/Vulkan 所需系统库和 VirtualGL。
- Python 3.10 虚拟环境：`/opt/venvs/openpi-robotwin-cu124`。
- RLinf（以 editable/file 安装）、OpenPI、RoboTwin 训练依赖、PyTorch3D、CuRobo、SAPIEN、FlashAttention、Warp 与 Open3D。
- 内部 Ubuntu/PyPI 源配置；构建时对少数不稳定的大型 wheel 使用了已下载 wheel，以避免 GPU worker 在运行时访问公网。

容器内导入烟测已通过：

```text
torch=2.6.0+cu124
sapien=3.0.1
rlinf=ok
openpi=ok
```

该验证只覆盖 Python 导入；实际 GPU、EGL 和 Vulkan 渲染会在 rjob GPU worker 上完成。

## rjob 使用要求

GPU worker 不应在任务启动后再下载模型、资产或 Python 依赖。提交任务时至少需要挂载：

```text
--mount=gpfs://gpfs1/chenzanxin:/mnt/shared-storage-user/chenzanxin
--mount=gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs
```

训练进程应激活镜像内环境，而不是历史 GPFS 虚拟环境：

```bash
source /opt/venvs/openpi-robotwin-cu124/bin/activate
```

RoboTwin 资产路径：

```text
/mnt/shared-storage-user/chenzanxin/jiayusun/RoboTwin-RLinf
```

SFT-20k 初始权重路径：

```text
/mnt/shared-storage-user/chenzanxin/jiayusun/sft20k
```

运行时环境变量：

```bash
export ROBOT_PLATFORM=ALOHA
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export VK_DRIVER_FILES="$REPO_PATH/nvidia_icd.json"
export VK_ICD_FILENAMES="$REPO_PATH/nvidia_icd.json"
export PYTHONPATH="$REPO_PATH:$ROBOTWIN_PATH:$PYTHONPATH"
```

控制面请求不应走开发机代理。提交及 rjob 查询前需要清除代理变量，并为集群内部域名设置 `NO_PROXY`。

## 本次 PPO 提交

本次从 SFT-20k（step 0）启动 `handover_block` PPO 的提交脚本：

[`my_docs/submit_handover_block_ppo_fromsft_openpi_vulkan_4gpu.sh`](submit_handover_block_ppo_fromsft_openpi_vulkan_4gpu.sh)

当前脚本资源配置为：4 GPU、32 CPU、800,000 MiB 内存、128 个训练环境、`clear_cache_freq: 2`。日志与 checkpoint 写入：

```text
/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/<job-name>
```

它使用 `moe_sft20k_handover_block_ppo150_128env_4gpu` 配置，该配置显式保持
`runner.resume_dir: null`、`runner.ckpt_path: null`，因此不会加载旧 PPO checkpoint。

# OpenPI + RobotWin 环境安装记录

本文记录在当前集群上安装 RLinf embodied 环境的实际流程，目标命令是：

```bash
bash requirements/install.sh embodied --model openpi --env robotwin
```

由于登录位置是管理节点，系统较老，不适合直接编译 CUDA/Python 依赖。实际安装需要提交到计算节点，并在 apptainer 容器中执行。

## 基本信息

- 仓库路径：`/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin`
- 虚拟环境：仓库内 `.venv`
- 容器镜像：`/mnt/petrelfs/muyao/shared_img/new_app.sif`
- 参考脚本：`/mnt/petrelfs/muyao/jiayusun/jupyter_sjy.sbatch`
- apptainer 运行方式：

```bash
apptainer exec --cleanenv \
  --bind /mnt:/mnt \
  --bind /nvme:/nvme \
  --nv \
  /mnt/petrelfs/muyao/shared_img/new_app.sif \
  bash -lc '<commands>'
```

## 关键环境变量

计算节点容器内需要显式切到 CUDA 12.4 和 GCC 11.2。否则会遇到 CUDA 版本不匹配、GCC 过旧、或链接阶段找不到系统启动对象的问题。

```bash
export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4

export CC=$GCC_HOME/bin/gcc
export CXX=$GCC_HOME/bin/g++
export CUDAHOSTCXX=$CXX
export CUDACXX=$CUDA_HOME/bin/nvcc
export CUB_HOME=$CUDA_HOME

export CPATH=/usr/include/x86_64-linux-gnu:${CPATH:-}
export C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=/usr/include/x86_64-linux-gnu:${CPLUS_INCLUDE_PATH:-}
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/lib:$GCC_HOME/lib/gcc/x86_64-pc-linux-gnu/11.2.0:$GCC_HOME/lib64:${LIBRARY_PATH:-}

export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$PWD/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

export UV_PYTHON_PLATFORM=x86_64-manylinux_2_28
export MAX_JOBS=16
```

`UV_PYTHON_PLATFORM=x86_64-manylinux_2_28` 很重要。本环境里 `sapien==3.0.1` 和 `warp-lang==1.11.1` 都需要 manylinux_2_28 wheel。

## 安装主流程

在管理节点提交 Slurm 作业，在计算节点容器内执行安装。由于容器里不能 sudo，使用 `--no-root` 跳过系统依赖安装。

```bash
cd /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
rm -f uv.lock

sbatch --parsable \
  --partition=wam_data \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --gres=gpu:1 \
  --time=12:00:00 \
  --job-name=rlinf-install-openpi-robotwin \
  --output=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/install_openpi_robotwin-%j.out \
  --error=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/install_openpi_robotwin-%j.err \
  --wrap='set -euo pipefail
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp PYTHONUNBUFFERED=1
export APPTAINERENV_TMPDIR=/tmp APPTAINERENV_TMP=/tmp APPTAINERENV_TEMP=/tmp
apptainer exec --cleanenv --bind /mnt:/mnt --bind /nvme:/nvme --nv /mnt/petrelfs/muyao/shared_img/new_app.sif bash -lc "
set -euo pipefail
cd /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export CC=\$GCC_HOME/bin/gcc
export CXX=\$GCC_HOME/bin/g++
export CUDAHOSTCXX=\$CXX
export CUDACXX=\$CUDA_HOME/bin/nvcc
export CUB_HOME=\$CUDA_HOME
export CPATH=/usr/include/x86_64-linux-gnu:\${CPATH:-}
export C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu:\${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=/usr/include/x86_64-linux-gnu:\${CPLUS_INCLUDE_PATH:-}
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/lib:\$GCC_HOME/lib/gcc/x86_64-pc-linux-gnu/11.2.0:\$GCC_HOME/lib64:\${LIBRARY_PATH:-}
export PATH=\$CUDA_HOME/bin:\$GCC_HOME/bin:\$PWD/.venv/bin:\$HOME/.local/bin:\$PATH
export LD_LIBRARY_PATH=\$GCC_HOME/lib64:\$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}
export UV_PYTHON_PLATFORM=x86_64-manylinux_2_28
export MAX_JOBS=16
bash requirements/install.sh embodied --model openpi --env robotwin --no-root
"'
```

## 本次安装中遇到的问题

1. 管理节点系统过旧，不能直接安装或编译，必须使用计算节点 + apptainer。
2. `uv` 会因为预发布/平台解析选择不合适的源码构建版本，安装时需要明确使用 manylinux_2_28 平台。
3. `sapien==3.0.1` 和 `warp-lang==1.11.1` 需要 manylinux_2_28 wheel。
4. `pytorch3d==0.7.9` 需要源码编译，耗时约 48 分钟。
5. `pytorch3d` 编译需要 CUDA 12.4，与当前 PyTorch `2.6.0+cu124` 匹配。
6. 默认 GCC 太旧，需使用 `/mnt/lustre/share/gcc/gcc-11.2.0`。
7. 链接阶段可能报 `cannot find crti.o`，需设置 `LIBRARY_PATH` 包含 `/usr/lib/x86_64-linux-gnu`。
8. OpenPI 替换 `transformers` 文件时，如果源文件是 symlink，直接 `cp -r` 会复制坏的相对 symlink，需要改为复制真实文件。

## 收尾修复步骤

如果主安装在 `warp-lang` 或后续步骤失败，而前面的依赖和 `pytorch3d` 已经装好，可以只执行收尾步骤：

```bash
cd /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin

apptainer exec --cleanenv --bind /mnt:/mnt --bind /nvme:/nvme --nv \
  /mnt/petrelfs/muyao/shared_img/new_app.sif bash -lc '
set -euo pipefail
cd /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export CC=$GCC_HOME/bin/gcc
export CXX=$GCC_HOME/bin/g++
export CUDAHOSTCXX=$CXX
export CUDACXX=$CUDA_HOME/bin/nvcc
export CUB_HOME=$CUDA_HOME
export CPATH=/usr/include/x86_64-linux-gnu:${CPATH:-}
export C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=/usr/include/x86_64-linux-gnu:${CPLUS_INCLUDE_PATH:-}
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/lib:$GCC_HOME/lib/gcc/x86_64-pc-linux-gnu/11.2.0:$GCC_HOME/lib64:${LIBRARY_PATH:-}
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$PWD/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export UV_PYTHON_PLATFORM=x86_64-manylinux_2_28
export MAX_JOBS=16

source .venv/bin/activate

uv pip install --python .venv/bin/python --python-platform x86_64-manylinux_2_28 warp-lang==1.11.1
uv pip install --python .venv/bin/python git+https://github.com/NVlabs/curobo.git --no-build-isolation

SAPIEN_LOCATION=$(uv pip show sapien | grep Location | awk "{print \$2}")/sapien
sed -i -E '\''s/("r")(\))( as)/\1, encoding="utf-8") as/g'\'' "$SAPIEN_LOCATION/wrapper/urdf_loader.py"

MPLIB_LOCATION=$(uv pip show mplib | grep Location | awk "{print \$2}")/mplib
sed -i -E '\''s/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g'\'' "$MPLIB_LOCATION/planner.py"

py_major_minor=$(python -c '\''import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'\'')
src=.venv/lib/python${py_major_minor}/site-packages/openpi/models_pytorch/transformers_replace
dst=.venv/lib/python${py_major_minor}/site-packages/transformers
find "$src" -type l -print | while read -r f; do rel=${f#$src/}; rm -f "$dst/$rel"; done
cp -Lr "$src"/* "$dst"/

bash requirements/embodied/download_assets.sh --assets openpi
uv pip uninstall pynvml || true
'
```

OpenPI 资产下载完成后，目录为：

```text
/mnt/petrelfs/muyao/.cache/openpi
```

## RoboTwin 代码和 Assets

RoboTwin 环境还需要单独的 RoboTwin 代码仓库和仿真资产。当前机器上有两个 RoboTwin 仓库：

- `/mnt/petrelfs/muyao/jiayusun/RoboTwin`：之前用于 SFT，已经执行过 `bash script/_download_assets.sh`。
- `/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf`：后续用于 PPO/RLinf 训练，目标分支是 `RLinf_support`。

`script/_download_assets.sh` 的主要动作是：

```bash
cd assets
python _download.py
unzip background_texture.zip
unzip embodiments.zip
unzip objects.zip
cd ..
python ./script/update_embodiment_config_path.py
```

因此不需要为 `RoboTwin-RLinf` 重新从 HuggingFace 下载。可以直接从已下载好的 `RoboTwin` 复制下面三个资产目录：

```bash
rsync -a --info=progress2 \
  /mnt/petrelfs/muyao/jiayusun/RoboTwin/assets/background_texture \
  /mnt/petrelfs/muyao/jiayusun/RoboTwin/assets/embodiments \
  /mnt/petrelfs/muyao/jiayusun/RoboTwin/assets/objects \
  /mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf/assets/
```

本次复制的资产大小约为：

```text
11G   /mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf/assets/background_texture
900M  /mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf/assets/embodiments
4.4G  /mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf/assets/objects
```

复制完成后，必须在 `RoboTwin-RLinf` 仓库根目录运行路径配置脚本：

```bash
cd /mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf
python ./script/update_embodiment_config_path.py
```

本次脚本处理了 6 个配置文件，将 `${ASSETS_PATH}` 替换为：

```text
/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf
```

涉及的配置包括：

```text
assets/embodiments/franka-panda/curobo.yml
assets/embodiments/ur5-wsg/curobo.yml
assets/embodiments/piper/curobo.yml
assets/embodiments/ARX-X5/curobo.yml
assets/embodiments/aloha-agilex/curobo_right.yml
assets/embodiments/aloha-agilex/curobo_left.yml
```

## 验证

在计算节点容器内执行：

```bash
cd /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin

apptainer exec --cleanenv --bind /mnt:/mnt --bind /nvme:/nvme --nv \
  /mnt/petrelfs/muyao/shared_img/new_app.sif bash -lc '
set -euo pipefail
cd /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$PWD/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

source .venv/bin/activate
python -c "import importlib, torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"available\", torch.cuda.is_available()); mods=[\"openpi\",\"sapien\",\"mplib\",\"pytorch3d\",\"warp\",\"curobo\",\"flash_attn\",\"transformers\"]; [print(m, getattr(importlib.import_module(m), \"__version__\", \"ok\")) for m in mods]"
'
```

本次验证结果：

```text
torch 2.6.0+cu124 cuda 12.4 available True
openpi ok
sapien 3.0.1
mplib 0.2.1
pytorch3d 0.7.9
warp 1.11.1
curobo 0.0.0
flash_attn 2.7.4.post1
transformers 4.53.2
```

## 备注

- 安装结果保留在仓库 `.venv` 中。
- 安装期间生成的 `slurm_logs/` 和 `uv.lock` 可以按需删除。
- 本次为了完成安装，曾临时调整安装脚本和 `pyproject.toml` 的依赖解析；安装完成后这些源码改动已恢复。
- RoboTwin 资产已从 `/mnt/petrelfs/muyao/jiayusun/RoboTwin` 复制到 `/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf`，后续 PPO 使用 `RoboTwin-RLinf`，SFT 继续使用原 `RoboTwin`。

#!/usr/bin/env bash
set -euo pipefail

cd /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$PWD/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

source .venv/bin/activate

python - <<'PY'
import importlib
import pathlib

import torch

print("python ok")
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())

mods = [
    "openpi",
    "sapien",
    "mplib",
    "pytorch3d",
    "warp",
    "curobo",
    "flash_attn",
    "transformers",
    "open3d",
    "av",
    "rtree",
]
for mod in mods:
    module = importlib.import_module(mod)
    print(mod, getattr(module, "__version__", "ok"))

print("openpi_cache_exists", pathlib.Path("/mnt/petrelfs/muyao/.cache/openpi").exists())
PY

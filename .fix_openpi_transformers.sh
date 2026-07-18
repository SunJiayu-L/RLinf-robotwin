#!/usr/bin/env bash
set -euo pipefail

cd /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
source .venv/bin/activate

py_major_minor=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
src=".venv/lib/python${py_major_minor}/site-packages/openpi/models_pytorch/transformers_replace"
dst=".venv/lib/python${py_major_minor}/site-packages/transformers"

find "$src" -type l -print | while read -r file; do
    rel=${file#$src/}
    rm -f "$dst/$rel"
done
cp -Lr "$src"/* "$dst"/

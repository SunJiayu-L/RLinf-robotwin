#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05
CONVERT_SCRIPT=/mnt/petrelfs/muyao/jiayusun/openpi/examples/convert_jax_model_to_pytorch.py
CHECKPOINT_DIR=/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/pi05_aloha_full_base/robotwin_47_pi05_full/20000
OUTPUT_PATH=/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/pi05_aloha_full_base/robotwin_47_pi05_full_pytorch/20000
CONFIG_NAME=pi05_aloha_full_base

cd "$REPO"

export HOME=/mnt/petrelfs/muyao/jiayusun
export XDG_CACHE_HOME=/mnt/petrelfs/muyao/jiayusun/store/cache
export HF_HOME=/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

PROXY=http://caijisong:754nCjLBDVINgxLMbLOhBnEvi4XPqZavc7aHC88tkcKrfMWPvji8OexquvN5@10.1.20.50:23128
export http_proxy=$PROXY
export https_proxy=$PROXY
export HTTP_PROXY=$PROXY
export HTTPS_PROXY=$PROXY

source "$REPO/.venv/bin/activate"

current_transformers=$(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || true)
if [[ "$current_transformers" != "4.53.2" ]]; then
    python -m pip install transformers==4.53.2
fi

py_major_minor=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
transformers_src="$REPO/src/openpi/models_pytorch/transformers_replace"
transformers_dst="$REPO/.venv/lib/python${py_major_minor}/site-packages/transformers"
find "$transformers_src" -type l -print | while read -r file; do
    rel=${file#$transformers_src/}
    rm -f "$transformers_dst/$rel"
done
cp -Lr "$transformers_src"/* "$transformers_dst"/

if [[ "${INSPECT_ONLY:-0}" == "1" ]]; then
    python -u "$CONVERT_SCRIPT" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --config-name "$CONFIG_NAME" \
        --inspect-only
    exit 0
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

python -u "$CONVERT_SCRIPT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config-name "$CONFIG_NAME" \
    --output-path "$OUTPUT_PATH" \
    --precision bfloat16

if [[ -d "$CHECKPOINT_DIR/assets" ]]; then
    rm -rf "$OUTPUT_PATH/assets"
    cp -a "$CHECKPOINT_DIR/assets" "$OUTPUT_PATH/assets"
fi

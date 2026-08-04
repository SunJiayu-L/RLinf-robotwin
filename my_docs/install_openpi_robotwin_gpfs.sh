#!/usr/bin/env bash

set -euo pipefail

REPO_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RLinf-robotwin
ROBOTWIN_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RoboTwin-RLinf
SFT_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/sft20k
BOOTSTRAP_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/bootstrap
GIT_MIRROR=/mnt/shared-storage-user/chenzanxin/jiayusun/git-mirror
VENV_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/venvs/openpi-robotwin-cu124
LOG_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/install_openpi_robotwin_gpfs.log
SUCCESS_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/install_openpi_robotwin_gpfs.success
UV_BIN="${BOOTSTRAP_PATH}/uv-tools/bin/uv"
FLASH_WHEEL="${BOOTSTRAP_PATH}/wheels/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
LOCAL_REQUIREMENTS="${BOOTSTRAP_PATH}/openpi-offline-requirements-local.txt"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
export NO_PROXY="${no_proxy}"
unset UV_TORCH_BACKEND
export UV_DEFAULT_INDEX=http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_FIND_LINKS="${BOOTSTRAP_PATH}/wheels"
export UV_CACHE_DIR="${BOOTSTRAP_PATH}/uv-cache-dev"
export UV_HTTP_TIMEOUT=120
export UV_HTTP_RETRIES=2
export CUDA_HOME=/usr/local/cuda
export CUB_HOME="${CUDA_HOME}"
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=16

command -v nvcc >/dev/null
nvcc --version
test -f /usr/include/python3.10/Python.h

mkdir -p "$(dirname "${VENV_PATH}")"

if [[ -e "${VENV_PATH}" && ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "Refusing to overwrite incomplete path: ${VENV_PATH}" >&2
    exit 1
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    "${UV_BIN}" venv --python /usr/bin/python3 "${VENV_PATH}"
fi

# Rewrite the only network Git dependency in the exported OpenPI lock to the
# GPFS mirror. GPU workers cannot access GitHub.
sed \
    "s|git+https://github.com/huggingface/lerobot|git+file://${GIT_MIRROR}/huggingface/lerobot.git|" \
    "${BOOTSTRAP_PATH}/openpi-offline-requirements.txt" \
    >"${LOCAL_REQUIREMENTS}"

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

{
    "${UV_BIN}" pip install -r "${LOCAL_REQUIREMENTS}"
    "${UV_BIN}" pip install --no-deps \
        "${BOOTSTRAP_PATH}/openpi-offline/packages/openpi-client" \
        "${BOOTSTRAP_PATH}/openpi-offline"

    # Install RLinf and its embodied dependencies after the locked OpenPI
    # environment so the already installed torch 2.6/CUDA 12.4 stack is reused.
    "${UV_BIN}" pip install -e "${REPO_PATH}[embodied]"
    "${UV_BIN}" pip install -r "${REPO_PATH}/requirements/embodied/envs/common.txt"

    "${UV_BIN}" pip install \
        hydra-core==1.4.0.dev1 \
        omegaconf==2.4.0.dev4 \
        protobuf==6.32.0 \
        ml-dtypes==0.5.4 \
        flatbuffers==25.12.19 \
        wcwidth==0.8.2 \
        wrapt==2.2.2 \
        cffi==2.1.0 \
        google-auth==2.39.0 \
        rsa==4.7.2 \
        "filelock>=3.24.2,<4"

    "${UV_BIN}" pip install \
        mplib==0.2.1 \
        gymnasium==0.29.1 \
        av==14.2.0 \
        open3d==0.19.0 \
        zarr==2.18.3 \
        openai \
        warp-lang==1.11.1

    "${UV_BIN}" pip install \
        "git+file://${GIT_MIRROR}/facebookresearch/pytorch3d.git@v0.7.9" \
        --no-build-isolation
    "${UV_BIN}" pip install \
        "git+file://${GIT_MIRROR}/NVlabs/curobo.git" \
        --no-build-isolation
    "${UV_BIN}" pip install "${FLASH_WHEEL}"
} 2>&1 | tee "${LOG_PATH}"

site_packages="${VENV_PATH}/lib/python3.10/site-packages"
cp -Lr \
    "${site_packages}/openpi/models_pytorch/transformers_replace/." \
    "${site_packages}/transformers/"

sapien_location=$("${UV_BIN}" pip show sapien | awk '/^Location:/ {print $2}')/sapien
sed -i -E \
    's/("r")(\))( as)/\1, encoding="utf-8") as/g' \
    "${sapien_location}/wrapper/urdf_loader.py"
mplib_location=$("${UV_BIN}" pip show mplib | awk '/^Location:/ {print $2}')/mplib
sed -i -E \
    's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
    "${mplib_location}/planner.py"

cd "${ROBOTWIN_PATH}"
python script/update_embodiment_config_path.py

export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export OPENPI_DATA_HOME="${BOOTSTRAP_PATH}/root/.cache/openpi"
export ROBOT_PLATFORM=ALOHA
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

test -d "${ROBOTWIN_PATH}/assets"
test -f "${SFT_PATH}/config.json"
nvidia-smi
nvcc --version
python - <<'PY'
import flash_attn
import lerobot
import mplib
import openpi
import rlinf
import sapien
import torch
from openpi.models.tokenizer import PaligemmaTokenizer

print(
    {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "flash_attn": flash_attn.__version__,
        "sapien": sapien.__version__,
        "mplib": mplib.__version__,
    }
)
assert torch.__version__.startswith("2.6.0")
assert torch.version.cuda == "12.4"
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1
tokens, token_mask = PaligemmaTokenizer().tokenize("handover block")
assert tokens.shape == token_mask.shape == (48,)
PY

cd "${REPO_PATH}"
python examples/embodiment/train_embodied_agent.py \
    --config-path "${REPO_PATH}/examples/embodiment/config" \
    --config-name moe_sft20k_handover_block_ppo150_128env \
    --cfg job \
    >"${VENV_PATH}/resolved_config.yaml"
grep -q "${SFT_PATH}" "${VENV_PATH}/resolved_config.yaml"
grep -q "${ROBOTWIN_PATH}" "${VENV_PATH}/resolved_config.yaml"

touch "${SUCCESS_PATH}"
echo "RLinf OpenPI RoboTwin GPFS environment validation passed."

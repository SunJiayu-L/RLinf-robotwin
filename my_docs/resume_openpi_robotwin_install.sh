#!/usr/bin/env bash

set -euo pipefail

REPO_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RLinf-robotwin
ROBOTWIN_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RoboTwin-RLinf
SFT_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/sft20k
VENV_PATH=/opt/venv/openpi
BOOTSTRAP_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/bootstrap
FLASH_WHEEL="${BOOTSTRAP_PATH}/wheels/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
LOG_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/install_openpi_robotwin_final.log

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
export NO_PROXY="${no_proxy}"
unset UV_TORCH_BACKEND
export UV_DEFAULT_INDEX=http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_FIND_LINKS="${BOOTSTRAP_PATH}/wheels"
export UV_HTTP_TIMEOUT=60
export UV_HTTP_RETRIES=1
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"
export MAX_JOBS=16

source "${VENV_PATH}/bin/activate"
{
    uv pip install -r "${BOOTSTRAP_PATH}/openpi-offline-requirements.txt"
    uv pip install --no-deps \
        "${BOOTSTRAP_PATH}/openpi-offline/packages/openpi-client" \
        "${BOOTSTRAP_PATH}/openpi-offline"
    uv pip install \
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
    uv pip install \
        mplib==0.2.1 \
        gymnasium==0.29.1 \
        av==14.2.0 \
        open3d==0.19.0 \
        zarr==2.18.3 \
        openai \
        warp-lang==1.11.1
    uv pip install \
        git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9 \
        --no-build-isolation
    uv pip install \
        git+https://github.com/NVlabs/curobo.git \
        --no-build-isolation
    uv pip install "${FLASH_WHEEL}"
} 2>&1 | tee "${LOG_PATH}"

site_packages="${VENV_PATH}/lib/python3.10/site-packages"
cp -a \
    "${site_packages}/openpi/models_pytorch/transformers_replace/." \
    "${site_packages}/transformers/"

sapien_location=$(uv pip show sapien | awk '/^Location:/ {print $2}')/sapien
sed -i -E \
    's/("r")(\))( as)/\1, encoding="utf-8") as/g' \
    "${sapien_location}/wrapper/urdf_loader.py"
mplib_location=$(uv pip show mplib | awk '/^Location:/ {print $2}')/mplib
sed -i -E \
    's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
    "${mplib_location}/planner.py"

install -m 0644 \
    "${REPO_PATH}/my_docs/rlinf-openpi-profile.sh" \
    /etc/profile.d/rlinf-openpi.sh

cd "${ROBOTWIN_PATH}"
python script/update_embodiment_config_path.py

export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export ROBOT_PLATFORM=ALOHA
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"

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
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1
PY

cd "${REPO_PATH}"
resolved_cfg=$(python examples/embodiment/train_embodied_agent.py \
    --config-path "${EMBODIED_PATH}/config" \
    --config-name moe_sft20k_handover_block_ppo150_128env \
    --cfg job)
grep -q "${SFT_PATH}" <<<"${resolved_cfg}"
grep -q "${ROBOTWIN_PATH}" <<<"${resolved_cfg}"

touch /mnt/shared-storage-user/chenzanxin/jiayusun/install_openpi_robotwin_final.success
echo "RLinf OpenPI RoboTwin environment validation passed."

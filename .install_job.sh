#!/usr/bin/env bash
set -euo pipefail

cd /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin

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
export PATH=/mnt/petrelfs/muyao/.local/bin:$CUDA_HOME/bin:$GCC_HOME/bin:$PWD/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export UV_PYTHON_PLATFORM=x86_64-manylinux_2_28
export MAX_JOBS=16
export UV_EXCLUDE_NEWER=2025-10-01T00:00:00Z

PROXY=http://caijisong:754nCjLBDVINgxLMbLOhBnEvi4XPqZavc7aHC88tkcKrfMWPvji8OexquvN5@10.1.20.50:23128
export http_proxy=$PROXY
export https_proxy=$PROXY
export HTTP_PROXY=$PROXY
export HTTPS_PROXY=$PROXY

# Restore all patched files on exit
trap 'git checkout -- pyproject.toml requirements/embodied/envs/common.txt requirements/install.sh 2>/dev/null || true' EXIT INT TERM HUP

# ============================================================
# Patch pyproject.toml: pin packages that exceed cutoff date
# ============================================================
python3 - <<'TOMLEOF'
path = "pyproject.toml"
content = open(path).read()
patches = {
    '"pyarrow<19"':       ('"datasets==3.6.0",', '"datasets==3.6.0",\n    "pyarrow<19",'),
    '"pandas<2.3"':       ('"datasets==3.6.0",', '"datasets==3.6.0",\n    "pandas<2.3",'),
    '"wandb<0.25"':       ('"datasets==3.6.0",', '"datasets==3.6.0",\n    "wandb<0.25",'),
    '"swanlab==0.6.10"':  ('"datasets==3.6.0",', '"datasets==3.6.0",\n    "swanlab==0.6.10",'),
    '"pyspacemouse==1.1.4"': ('"datasets==3.6.0",', '"datasets==3.6.0",\n    "pyspacemouse==1.1.4",'),
    '"omegaconf==2.3.0"': ('"datasets==3.6.0",', '"datasets==3.6.0",\n    "omegaconf==2.3.0",'),
    '"hydra-core==1.3.2"': ('"datasets==3.6.0",', '"datasets==3.6.0",\n    "hydra-core==1.3.2",'),
}
changed = False
for pin, (old, new) in patches.items():
    if pin not in content:
        content = content.replace(old, new)
        print(f"[install_job] Patched pyproject.toml: added {pin}")
        changed = True
    else:
        print(f"[install_job] {pin} already present, skipping")
old_line = '    "rlinf_latex2sympy2",\n'
if old_line in content:
    content = content.replace(old_line, '')
    print("[install_job] Patched pyproject.toml: removed rlinf_latex2sympy2")
    changed = True
lerobot_dep = "lerobot @ git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
if lerobot_dep in content:
    content = "\n".join(
        line for line in content.splitlines() if lerobot_dep not in line
    ) + "\n"
    print("[install_job] Patched pyproject.toml: removed extra lerobot git dependencies")
    changed = True
if changed:
    open(path, "w").write(content)
TOMLEOF

# ============================================================
# Pre-download sapien==3.0.1 and warp-lang==1.11.1.
# Both packages only ship manylinux_2_28_x86_64 wheels, but the
# container Python reports manylinux_2_17 (old CentOS glibc tag),
# causing `uv pip install` to reject them as incompatible.
#
# Fix: download with explicit --python-platform x86_64-manylinux_2_28,
# then rewrite the internal WHEEL metadata Tag to "py3-none-any".
# pip/uv accept py3-none-any on any Python 3 interpreter.
# The native extensions inside still run because the container's
# actual glibc is >= 2.28.
# ============================================================
UV_SYS=/mnt/petrelfs/muyao/.local/bin/uv
WHEEL_CACHE=/tmp/rlinf_whl_cache_$$
mkdir -p "$WHEEL_CACHE"

# Repackager: rewrites WHEEL metadata Tag and updates RECORD hash
cat > /tmp/make_any_wheel_$$.py << 'WHLEOF'
import sys, zipfile, re, hashlib, base64

src, dst = sys.argv[1], sys.argv[2]
files = {}
with zipfile.ZipFile(src, 'r') as zin:
    for item in zin.infolist():
        files[item.filename] = zin.read(item.filename)

wheel_key = next((f for f in files if f.endswith('/WHEEL')), None)
if wheel_key:
    orig = files[wheel_key]
    new_w = re.sub(rb'Tag: \S+', b'Tag: py3-none-any', orig)
    new_w = re.sub(rb'Root-Is-Purelib: false', b'Root-Is-Purelib: true', new_w)
    files[wheel_key] = new_w
    record_key = next((f for f in files if f.endswith('/RECORD')), None)
    if record_key:
        digest = hashlib.sha256(new_w).digest()
        h = 'sha256=' + base64.urlsafe_b64encode(digest).rstrip(b'=').decode()
        rec = files[record_key].decode('utf-8')
        rec = re.sub(re.escape(wheel_key) + r',[^,\n]*,[^\n]*',
                     f'{wheel_key},{h},{len(new_w)}', rec)
        files[record_key] = rec.encode('utf-8')

with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
    for fname, data in files.items():
        zout.writestr(fname, data)
print(f'[install_job] Created py3-none-any wheel: {dst}')
WHLEOF
MAKE_ANY_WHEEL=/tmp/make_any_wheel_$$.py

# ---- sapien 3.0.1 ----
echo "[install_job] Downloading sapien==3.0.1 (manylinux_2_28)..."
python3 -m pip download sapien==3.0.1 \
    --platform manylinux_2_28_x86_64 \
    --python-version 3.11 \
    --only-binary :all: \
    --no-deps \
    -d "$WHEEL_CACHE/" 2>&1 | tail -5

SAPIEN_WHL=$(ls "$WHEEL_CACHE"/sapien*.whl 2>/dev/null | head -1)
if [ -n "$SAPIEN_WHL" ]; then
    SAPIEN_ANY="$WHEEL_CACHE/sapien-3.0.1-py3-none-any.whl"
    python3 "$MAKE_ANY_WHEEL" "$SAPIEN_WHL" "$SAPIEN_ANY"
    # Replace the sapien==3.0.1;... line in common.txt with the local wheel path
    sed -i "s|^sapien==3.0.1.*|${SAPIEN_ANY}|" requirements/embodied/envs/common.txt
    echo "[install_job] Patched common.txt: sapien -> local py3-none-any wheel"
else
    echo "[install_job] ERROR: sapien download failed, cannot continue"
    exit 1
fi

# ---- warp-lang 1.11.1 ----
echo "[install_job] Downloading warp-lang==1.11.1 (manylinux_2_28)..."
python3 -m pip download warp-lang==1.11.1 \
    --platform manylinux_2_28_x86_64 \
    --python-version 3.11 \
    --only-binary :all: \
    --no-deps \
    -d "$WHEEL_CACHE/" 2>&1 | tail -5

WARP_WHL=$(ls "$WHEEL_CACHE"/warp*.whl 2>/dev/null | head -1)
if [ -n "$WARP_WHL" ]; then
    WARP_ANY="$WHEEL_CACHE/warp_lang-1.11.1-py3-none-any.whl"
    python3 "$MAKE_ANY_WHEEL" "$WARP_WHL" "$WARP_ANY"
    # Patch install.sh: replace `uv pip install warp-lang==1.11.1` with local wheel
    sed -i "s|uv pip install warp-lang==1.11.1|uv pip install ${WARP_ANY}|" requirements/install.sh
    echo "[install_job] Patched install.sh: warp-lang -> local py3-none-any wheel"
else
    echo "[install_job] WARNING: warp-lang download failed; warp-lang install may fail later"
fi

# ---- nvidia-nvcomp-cu12 5.0.0.6 ----
echo "[install_job] Downloading nvidia-nvcomp-cu12==5.0.0.6 (manylinux_2_28)..."
python3 -m pip download nvidia-nvcomp-cu12==5.0.0.6 \
    --platform manylinux_2_28_x86_64 \
    --python-version 3.11 \
    --only-binary :all: \
    --no-deps \
    -d "$WHEEL_CACHE/" 2>&1 | tail -5

NVCOMP_WHL=$(ls "$WHEEL_CACHE"/nvidia_nvcomp_cu12*.whl 2>/dev/null | head -1)
if [ -n "$NVCOMP_WHL" ]; then
    NVCOMP_ANY="$WHEEL_CACHE/nvidia_nvcomp_cu12-5.0.0.6-py3-none-any.whl"
    python3 "$MAKE_ANY_WHEEL" "$NVCOMP_WHL" "$NVCOMP_ANY"
    sed -i "s|^nvidia-nvcomp-cu12.*|${NVCOMP_ANY}|" requirements/embodied/envs/common.txt
    echo "[install_job] Patched common.txt: nvidia-nvcomp-cu12 -> local py3-none-any wheel"
else
    echo "[install_job] ERROR: nvidia-nvcomp-cu12 download failed, cannot continue"
    exit 1
fi

# ---- mujoco 3.3.6 ----
echo "[install_job] Downloading mujoco==3.3.6 (manylinux_2_28)..."
python3 -m pip download mujoco==3.3.6 \
    --platform manylinux_2_28_x86_64 \
    --python-version 3.11 \
    --only-binary :all: \
    --no-deps \
    -d "$WHEEL_CACHE/" 2>&1 | tail -5

MUJOCO_WHL=$(ls "$WHEEL_CACHE"/mujoco-3.3.6*.whl 2>/dev/null | head -1)
if [ -n "$MUJOCO_WHL" ]; then
    MUJOCO_ANY="$WHEEL_CACHE/mujoco-3.3.6-py3-none-any.whl"
    python3 "$MAKE_ANY_WHEEL" "$MUJOCO_WHL" "$MUJOCO_ANY"
    python3 - "$MUJOCO_ANY" <<'PYEOF'
import sys

wheel = sys.argv[1]
path = "pyproject.toml"
content = open(path).read()
entry = f'    "mujoco @ file://{wheel}",\n'
if entry not in content:
    content = content.replace("override-dependencies = [\n", "override-dependencies = [\n" + entry)
    open(path, "w").write(content)
    print("[install_job] Patched pyproject.toml: mujoco -> local py3-none-any wheel")
else:
    print("[install_job] pyproject.toml already points mujoco to local wheel")
PYEOF
else
    echo "[install_job] ERROR: mujoco download failed, cannot continue"
    exit 1
fi

# ---- ml-dtypes 0.5.3 ----
echo "[install_job] Downloading ml-dtypes==0.5.3 (manylinux_2_28)..."
python3 -m pip download ml-dtypes==0.5.3 \
    --platform manylinux_2_28_x86_64 \
    --python-version 3.11 \
    --only-binary :all: \
    --no-deps \
    -d "$WHEEL_CACHE/" 2>&1 | tail -5

ML_DTYPES_WHL=$(ls "$WHEEL_CACHE"/ml_dtypes-0.5.3*.whl 2>/dev/null | head -1)
if [ -n "$ML_DTYPES_WHL" ]; then
    ML_DTYPES_ANY="$WHEEL_CACHE/ml_dtypes-0.5.3-py3-none-any.whl"
    python3 "$MAKE_ANY_WHEEL" "$ML_DTYPES_WHL" "$ML_DTYPES_ANY"
    python3 - "$ML_DTYPES_ANY" <<'PYEOF'
import sys

wheel = sys.argv[1]
path = "pyproject.toml"
content = open(path).read()
entry = f'    "ml-dtypes @ file://{wheel}",\n'
if entry not in content:
    content = content.replace("override-dependencies = [\n", "override-dependencies = [\n" + entry)
    open(path, "w").write(content)
    print("[install_job] Patched pyproject.toml: ml-dtypes -> local py3-none-any wheel")
PYEOF
else
    echo "[install_job] ERROR: ml-dtypes download failed, cannot continue"
    exit 1
fi

# ---- optree 0.17.0 ----
echo "[install_job] Downloading optree==0.17.0 (manylinux_2_28)..."
python3 -m pip download optree==0.17.0 \
    --platform manylinux_2_28_x86_64 \
    --python-version 3.11 \
    --only-binary :all: \
    --no-deps \
    -d "$WHEEL_CACHE/" 2>&1 | tail -5

OPTREE_WHL=$(ls "$WHEEL_CACHE"/optree-0.17.0*.whl 2>/dev/null | head -1)
if [ -n "$OPTREE_WHL" ]; then
    OPTREE_ANY="$WHEEL_CACHE/optree-0.17.0-py3-none-any.whl"
    python3 "$MAKE_ANY_WHEEL" "$OPTREE_WHL" "$OPTREE_ANY"
    python3 - "$OPTREE_ANY" <<'PYEOF'
import sys

wheel = sys.argv[1]
path = "pyproject.toml"
content = open(path).read()
entry = f'    "optree @ file://{wheel}",\n'
if entry not in content:
    content = content.replace("override-dependencies = [\n", "override-dependencies = [\n" + entry)
    open(path, "w").write(content)
    print("[install_job] Patched pyproject.toml: optree -> local py3-none-any wheel")
PYEOF
else
    echo "[install_job] ERROR: optree download failed, cannot continue"
    exit 1
fi

# ---- contourpy 1.3.3 ----
echo "[install_job] Downloading contourpy==1.3.3 (manylinux_2_28)..."
python3 -m pip download contourpy==1.3.3 \
    --platform manylinux_2_28_x86_64 \
    --python-version 3.11 \
    --only-binary :all: \
    --no-deps \
    -d "$WHEEL_CACHE/" 2>&1 | tail -5

CONTOURPY_WHL=$(ls "$WHEEL_CACHE"/contourpy-1.3.3*.whl 2>/dev/null | head -1)
if [ -n "$CONTOURPY_WHL" ]; then
    CONTOURPY_ANY="$WHEEL_CACHE/contourpy-1.3.3-py3-none-any.whl"
    python3 "$MAKE_ANY_WHEEL" "$CONTOURPY_WHL" "$CONTOURPY_ANY"
    python3 - "$CONTOURPY_ANY" <<'PYEOF'
import sys

wheel = sys.argv[1]
path = "pyproject.toml"
content = open(path).read()
entry = f'    "contourpy @ file://{wheel}",\n'
if entry not in content:
    content = content.replace("override-dependencies = [\n", "override-dependencies = [\n" + entry)
    open(path, "w").write(content)
    print("[install_job] Patched pyproject.toml: contourpy -> local py3-none-any wheel")
PYEOF
else
    echo "[install_job] ERROR: contourpy download failed, cannot continue"
    exit 1
fi

# ---- rerun-sdk 0.23.4 ----
echo "[install_job] Downloading rerun-sdk==0.23.4 (manylinux_2_28)..."
python3 -m pip download rerun-sdk==0.23.4 \
    --platform manylinux_2_28_x86_64 \
    --python-version 3.11 \
    --only-binary :all: \
    --no-deps \
    -d "$WHEEL_CACHE/" 2>&1 | tail -5

RERUN_WHL=$(ls "$WHEEL_CACHE"/rerun_sdk-0.23.4*.whl 2>/dev/null | head -1)
if [ -n "$RERUN_WHL" ]; then
    RERUN_ANY="$WHEEL_CACHE/rerun_sdk-0.23.4-py3-none-any.whl"
    python3 "$MAKE_ANY_WHEEL" "$RERUN_WHL" "$RERUN_ANY"
    python3 - "$RERUN_ANY" <<'PYEOF'
import sys

wheel = sys.argv[1]
path = "pyproject.toml"
content = open(path).read()
entry = f'    "rerun-sdk @ file://{wheel}",\n'
if entry not in content:
    content = content.replace("override-dependencies = [\n", "override-dependencies = [\n" + entry)
    open(path, "w").write(content)
    print("[install_job] Patched pyproject.toml: rerun-sdk -> local py3-none-any wheel")
PYEOF
else
    echo "[install_job] ERROR: rerun-sdk download failed, cannot continue"
    exit 1
fi

python3 - "$RERUN_ANY" <<'PYEOF'
import sys

rerun_wheel = sys.argv[1]
path = "requirements/install.sh"
content = open(path).read()
old = "uv pip install git+${GITHUB_PREFIX}https://github.com/RLinf/openpi"
new = (
    f"uv pip install {rerun_wheel} --no-deps\n"
    "            uv pip install lerobot==0.1.0 --no-deps\n"
    "            uv pip install git+${GITHUB_PREFIX}https://github.com/RLinf/openpi --no-deps"
)
if old in content:
    content = content.replace(old, new)
    open(path, "w").write(content)
    print("[install_job] Patched install.sh: install OpenPI stack with --no-deps")
else:
    print("[install_job] install.sh already patched for OpenPI --no-deps")
PYEOF

# ---- open3d 0.19.0 ----
echo "[install_job] Downloading open3d==0.19.0 (manylinux_2_31)..."
python3 -m pip download open3d==0.19.0 \
    --platform manylinux_2_31_x86_64 \
    --python-version 3.11 \
    --only-binary :all: \
    --no-deps \
    -d "$WHEEL_CACHE/" 2>&1 | tail -5

OPEN3D_WHL=$(ls "$WHEEL_CACHE"/open3d-0.19.0*.whl 2>/dev/null | head -1)
if [ -n "$OPEN3D_WHL" ]; then
    OPEN3D_ANY="$WHEEL_CACHE/open3d-0.19.0-py3-none-any.whl"
    python3 "$MAKE_ANY_WHEEL" "$OPEN3D_WHL" "$OPEN3D_ANY"
    python3 - "$OPEN3D_ANY" <<'PYEOF'
import sys

wheel = sys.argv[1]
path = "requirements/install.sh"
content = open(path).read()
old = "uv pip install mplib==0.2.1 gymnasium==0.29.1 av open3d zarr openai"
new = f"uv pip install mplib==0.2.1 gymnasium==0.29.1 av {wheel} zarr openai"
if old in content:
    content = content.replace(old, new)
    open(path, "w").write(content)
    print("[install_job] Patched install.sh: open3d -> local py3-none-any wheel")
else:
    print("[install_job] install.sh already patched for open3d")
PYEOF
else
    echo "[install_job] ERROR: open3d download failed, cannot continue"
    exit 1
fi

# ---- av 15.1.0 ----
echo "[install_job] Downloading av==15.1.0 (manylinux_2_28)..."
python3 -m pip download av==15.1.0 \
    --platform manylinux_2_28_x86_64 \
    --python-version 3.11 \
    --only-binary :all: \
    --no-deps \
    -d "$WHEEL_CACHE/" 2>&1 | tail -5

AV_WHL=$(ls "$WHEEL_CACHE"/av-15.1.0*.whl 2>/dev/null | head -1)
if [ -n "$AV_WHL" ]; then
    AV_ANY="$WHEEL_CACHE/av-15.1.0-py3-none-any.whl"
    python3 "$MAKE_ANY_WHEEL" "$AV_WHL" "$AV_ANY"
    python3 - "$AV_ANY" <<'PYEOF'
import sys

wheel = sys.argv[1]
path = "requirements/install.sh"
content = open(path).read()
old = " gymnasium==0.29.1 av "
new = f" gymnasium==0.29.1 {wheel} "
if old in content:
    content = content.replace(old, new)
    open(path, "w").write(content)
    print("[install_job] Patched install.sh: av -> local py3-none-any wheel")
else:
    print("[install_job] install.sh already patched for av")
PYEOF
else
    echo "[install_job] ERROR: av download failed, cannot continue"
    exit 1
fi

# ---- rtree 1.4.1 ----
echo "[install_job] Downloading rtree==1.4.1 (manylinux_2_28)..."
python3 -m pip download rtree==1.4.1 \
    --platform manylinux_2_28_x86_64 \
    --python-version 3.11 \
    --only-binary :all: \
    --no-deps \
    -d "$WHEEL_CACHE/" 2>&1 | tail -5

RTREE_WHL=$(ls "$WHEEL_CACHE"/rtree-1.4.1*.whl 2>/dev/null | head -1)
if [ -n "$RTREE_WHL" ]; then
    RTREE_ANY="$WHEEL_CACHE/rtree-1.4.1-py3-none-any.whl"
    python3 "$MAKE_ANY_WHEEL" "$RTREE_WHL" "$RTREE_ANY"
    python3 - "$RTREE_ANY" <<'PYEOF'
import sys

wheel = sys.argv[1]
path = "requirements/install.sh"
content = open(path).read()
old = "uv pip install git+${GITHUB_PREFIX}https://github.com/NVlabs/curobo.git  --no-build-isolation"
new = (
    f"uv pip install {wheel}\n"
    "    uv pip install git+${GITHUB_PREFIX}https://github.com/NVlabs/curobo.git  --no-build-isolation"
)
if old in content:
    content = content.replace(old, new)
    open(path, "w").write(content)
    print("[install_job] Patched install.sh: preinstall rtree before curobo")
else:
    print("[install_job] install.sh already patched for rtree/curobo")
PYEOF
else
    echo "[install_job] ERROR: rtree download failed, cannot continue"
    exit 1
fi

bash requirements/install.sh embodied --model openpi --env robotwin --no-root

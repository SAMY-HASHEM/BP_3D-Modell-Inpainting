#!/bin/bash
# Sets up TRELLIS.2 for this project on a Linux machine with an NVIDIA GPU, without root/sudo.
#
#   ./install.sh
#
# Steps:
#   1. installs Miniforge (conda) in ~/miniforge3 if conda isn't available
#   2. clones TRELLIS.2 (to $TRELLIS2_DIR, default ./TRELLIS.2 in this folder) if it isn't there yet
#   3. creates the conda environment from environment.yml
#   4. builds TRELLIS.2's CUDA add-ons for the GPU in this machine
#   5. makes "import trellis2" work from anywhere in the environment
#   6. downloads the model weights (needs a Hugging Face login with access to DINOv3 and RMBG-2.0)
#
# Settings (optional environment variables):
#   TRELLIS2_DIR   where TRELLIS.2's code lives        (default: ./TRELLIS.2 in this folder)
#   BUILD_DIR      where add-on sources are built      (default: ~/.cache/inpainting-3d-build)
#   SKIP_WEIGHTS=1 don't download the model weights
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="inpainting-3d-modelling"
TRELLIS2_DIR="${TRELLIS2_DIR:-$PROJECT_DIR/TRELLIS.2}"
BUILD_DIR="${BUILD_DIR:-$HOME/.cache/inpainting-3d-build}"

# Tested versions of TRELLIS.2 and its add-ons
TRELLIS2_REPO="https://github.com/microsoft/TRELLIS.2.git"
TRELLIS2_COMMIT="75fbf0183001ed9876c8dbb35de6b68552ee08bd"
NVDIFFRAST_COMMIT="253ac4fcea7de5f396371124af597e6cc957bfae"   # v0.4.0
NVDIFFREC_COMMIT="b296927cc7fd01c2ac1087c8065c4d7248f72da4"    # renderutils branch
CUMESH_COMMIT="12289e1062f0603f2f0d0771b02e1395d247f26f"
FLEXGEMM_COMMIT="6dd94a859c26ee8246888502eada3dd8ad85532e"

step() { echo; echo "===== $1 ====="; }

# Clones a repo at a fixed commit (with submodules), or reuses an existing checkout.
checkout() {
    local url=$1 commit=$2 dir=$3
    if [ ! -d "$dir/.git" ]; then
        git clone -q "$url" "$dir"
    fi
    git -C "$dir" fetch -q origin "$commit" 2>/dev/null || true
    git -C "$dir" checkout -q "$commit"
    git -C "$dir" submodule update -q --init --recursive
}

step "1/6 Conda"
# Prefer ~/miniforge3, then an activated conda. A conda without etc/profile.d/conda.sh
# (e.g. Fedora's /usr/bin/conda package) can't activate environments in a script, so it's skipped.
has_profile() { [ -n "$1" ] && [ -f "$1/etc/profile.d/conda.sh" ]; }
CONDA_BASE=""
if has_profile "$HOME/miniforge3"; then
    CONDA_BASE="$HOME/miniforge3"
elif [ -n "$CONDA_EXE" ] && has_profile "$(dirname "$(dirname "$CONDA_EXE")")"; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
elif command -v conda >/dev/null && has_profile "$(conda info --base 2>/dev/null)"; then
    CONDA_BASE="$(conda info --base)"
else
    echo "no usable conda found, installing Miniforge in ~/miniforge3"
    installer=$(mktemp -d)/miniforge.sh
    curl -fsSL -o "$installer" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$installer" -b -p "$HOME/miniforge3"
    "$HOME/miniforge3/bin/conda" init bash >/dev/null
    CONDA_BASE="$HOME/miniforge3"
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
echo "using conda in $CONDA_BASE"

step "2/6 TRELLIS.2 code"
if [ -d "$TRELLIS2_DIR/.git" ]; then
    echo "using existing $TRELLIS2_DIR (not changing its version)"
else
    checkout "$TRELLIS2_REPO" "$TRELLIS2_COMMIT" "$TRELLIS2_DIR"
fi

step "3/6 Conda environment '$ENV_NAME'"
if conda env list | grep -qE "^$ENV_NAME\s"; then
    conda env update -q -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml"
else
    conda env create -q -f "$PROJECT_DIR/environment.yml"
fi
conda activate "$ENV_NAME"

# Build the add-ons for the GPU(s) in this machine, e.g. 8.6 = RTX 3090, 9.0 = H100.
GPU_ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd ';')
GPU_ARCHS=${GPU_ARCHS:-"8.0;8.6;8.9;9.0"}
conda env config vars set -n "$ENV_NAME" CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="$GPU_ARCHS" >/dev/null
conda deactivate && conda activate "$ENV_NAME"
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'; print('PyTorch', torch.__version__, 'on', torch.cuda.get_device_name(0))"
echo "building for GPU architecture(s): $TORCH_CUDA_ARCH_LIST"

step "4/6 CUDA add-ons (takes a few minutes)"
export MAX_JOBS=${MAX_JOBS:-$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 ))}
# Link against the CUDA driver stub at build time; the real driver is used at runtime.
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
mkdir -p "$BUILD_DIR"
checkout https://github.com/NVlabs/nvdiffrast.git "$NVDIFFRAST_COMMIT" "$BUILD_DIR/nvdiffrast"
checkout https://github.com/JeffreyXiang/nvdiffrec.git "$NVDIFFREC_COMMIT" "$BUILD_DIR/nvdiffrec"
checkout https://github.com/JeffreyXiang/CuMesh.git "$CUMESH_COMMIT" "$BUILD_DIR/CuMesh"
checkout https://github.com/JeffreyXiang/FlexGEMM.git "$FLEXGEMM_COMMIT" "$BUILD_DIR/FlexGEMM"
rm -rf "$BUILD_DIR/o-voxel" && cp -r "$TRELLIS2_DIR/o-voxel" "$BUILD_DIR/o-voxel"
for ext in nvdiffrast nvdiffrec CuMesh FlexGEMM o-voxel; do
    echo "--- $ext"
    pip install -q "$BUILD_DIR/$ext" --no-build-isolation
done

step "5/6 Linking TRELLIS.2 into the environment"
SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "$TRELLIS2_DIR" > "$SITE_PACKAGES/trellis2_local.pth"
(cd /tmp && python -c "
import nvdiffrast.torch, nvdiffrec_render, cumesh, flex_gemm, o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import trellis2, os
print('TRELLIS.2 loads from', os.path.dirname(trellis2.__file__))
")

step "6/6 Model weights"
if [ "${SKIP_WEIGHTS:-0}" = "1" ]; then
    echo "skipped (SKIP_WEIGHTS=1)"
elif ! hf auth whoami >/dev/null 2>&1; then
    echo "Not logged in to Hugging Face, so the weights are not downloaded yet. To finish:"
    echo "  1. request access to https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"
    echo "     and https://huggingface.co/briaai/RMBG-2.0"
    echo "  2. conda activate $ENV_NAME && hf auth login"
    echo "  3. run ./install.sh again (it reuses everything already built)"
else
    hf download microsoft/TRELLIS.2-4B >/dev/null
    hf download microsoft/TRELLIS-image-large ckpts/ss_dec_conv3d_16l8_fp16.json ckpts/ss_dec_conv3d_16l8_fp16.safetensors >/dev/null
    hf download facebook/dinov3-vitl16-pretrain-lvd1689m >/dev/null
    hf download briaai/RMBG-2.0 --include "*.py" "*.json" "model.safetensors" >/dev/null
    echo "weights are in $(python -c 'from huggingface_hub import constants; print(constants.HF_HUB_CACHE)')"
fi

echo
echo "Done. Start with:"
echo "  conda activate $ENV_NAME"
echo "  python scripts/generate.py pictures/<your picture>"

#!/bin/bash
# Sets up OmniPart (github.com/HKU-MMLab/OmniPart, MIT licence) next to TRELLIS.2, in its own
# conda environment "omnipart" inside ./conda, because it needs other versions (torch 2.4, CUDA 12.1).
# Run ./install.sh first (it installs conda in ./conda).
#
#   ./install_omnipart.sh
#
# Afterwards:
#   source conda/etc/profile.d/conda.sh && conda activate omnipart
#   python scripts/models/omnipart_generate.py picture_rgba.png picture_mask.exr
#
# OmniPart's requirements.txt is incomplete; the fixes below are what it needed on Fedora 44 + RTX 3090.
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNIPART_REPO="https://github.com/HKU-MMLab/OmniPart.git"
OMNIPART_COMMIT="ceb1a4f"  # tested version (2025-11-06)

source "$PROJECT_DIR/conda/etc/profile.d/conda.sh"

if [ ! -d "$PROJECT_DIR/OmniPart" ]; then
    git clone "$OMNIPART_REPO" "$PROJECT_DIR/OmniPart"
    git -C "$PROJECT_DIR/OmniPart" checkout -q "$OMNIPART_COMMIT"
fi

if ! conda env list | grep -q "^omnipart "; then
    conda create -y -n omnipart -c nvidia/label/cuda-12.1.1 -c conda-forge \
        python=3.10 cuda-toolkit=12.1.1 gcc=12 gxx=12
fi
conda activate omnipart

export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(python -c 'import subprocess; print(subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]).decode().split()[0])' 2>/dev/null || echo 8.6)}"
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export MAX_JOBS=${MAX_JOBS:-$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 ))}

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
# detectron2 and mesh2sdf are built against the installed torch, so no build isolation, and their
# build tools must be there first
pip install pybind11 "setuptools<70" wheel ninja cython
pip install --no-build-isolation -r "$PROJECT_DIR/OmniPart/requirements.txt"
# missing from requirements.txt
pip install pytz psutil scikit-learn
# spconv/cumm 2.3.6 are built for NumPy 1.x and crash ("Floating point exception") under NumPy 2,
# which plyfile pulls in; lightning still needs pkg_resources, which setuptools >= 70 dropped
pip install "numpy==1.26.4" "plyfile<1.1" "setuptools<70"

conda env config vars set HF_HUB_CACHE="$PROJECT_DIR/local_ai_models" HF_MODULES_CACHE="$PROJECT_DIR/local_ai_models/modules" \
    CUDA_HOME="$CONDA_PREFIX" >/dev/null
python -c "import torch, spconv, detectron2, xformers; assert torch.cuda.is_available(); print('OmniPart environment ready:', torch.__version__, torch.cuda.get_device_name(0))"

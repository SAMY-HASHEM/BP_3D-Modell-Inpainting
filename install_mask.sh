#!/bin/bash
# Sets up the mask AI (it finds a part like "helmet" in pictures, for the Blender add-on's "Find part"
# button) in its own small conda environment "mask-ai" inside ./conda. It needs no TRELLIS.2, so it also
# fits on a PC that only runs Blender: any NVIDIA GPU from the RTX 20 to the RTX 50 series (about 4 GB of
# VRAM), or the CPU. Nothing is installed outside this folder.
#
#   ./install_mask.sh
#
# Models (downloaded at the end, into ./models/mask, about 4 GB):
#   Grounding DINO base (finds the word in a picture) + SAM 2.1 large (cuts out its exact outline)
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="$PROJECT_DIR/conda"
ENV_NAME="mask-ai"
TORCH="torch==2.11.0 torchvision==0.26.0"  # CUDA 12.8 builds: RTX 20 to RTX 50 series, NVIDIA driver 570 or newer
TRANSFORMERS="transformers==5.20.2"

if [ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then  # a PC without the TRELLIS.2 install: conda only
    installer=$(mktemp -d)/miniforge.sh
    curl -fsSL -o "$installer" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$installer" -b -p "$CONDA_DIR" >/dev/null
    rm -f "$installer"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

if [ ! -d "$CONDA_DIR/envs/$ENV_NAME" ]; then
    conda create -y -q -n "$ENV_NAME" -c conda-forge python=3.12 pip
fi
conda activate "$ENV_NAME"
pip install -q $TORCH --index-url https://download.pytorch.org/whl/cu128
pip install -q "$TRANSFORMERS" pillow numpy

python "$PROJECT_DIR/scripts/mask_ai.py" --download
echo
echo "Done. The Blender add-on finds this environment by itself (add-on preferences: Find part)."

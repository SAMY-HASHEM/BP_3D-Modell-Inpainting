#!/bin/bash
# GEM-X (NVIDIA, github.com/NVlabs/GEM-X): a person's motion from a video, for motion_track.py.
# Its own environment conda/envs/gemx (PyTorch 2.10; TRELLIS.2's stays on 2.6), its weights (7 GB) and SOMA's files.
# About 15 minutes; again: only what is missing is done.
#     bash scripts/rigging/install_gemx.sh
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GX=$ROOT/local_ai_models/motion/GEM-X
ENV=$ROOT/conda/envs/gemx
COMMIT=4aef145  # the version motion_track.py was made with (Nov 22, 2026)
echo "== GEM-X $(date +%H:%M)"
if [ ! -d "$GX/.git" ]; then
  mkdir -p "$(dirname "$GX")"
  git clone -q https://github.com/NVlabs/GEM-X.git "$GX"
  git -C "$GX" checkout -q $COMMIT
fi
cd "$GX"
echo "== environment $(date +%H:%M)"
[ -x "$ENV/bin/python" ] || "$ROOT/conda/bin/conda" create -y -q -p "$ENV" python=3.12
P=$ENV/bin/python
$P -m pip install -q --upgrade pip
$P -m pip install -q "torch==2.10.*" torchvision --index-url https://download.pytorch.org/whl/cu128
$P -m pip install -q -e '.[video,render]' huggingface_hub imageio-ffmpeg
# GEM-X cuts clips with ffmpeg and libx264; Fedora's ffmpeg has no x264, imageio-ffmpeg's has
ln -sf "$($P -c 'import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())')" "$ENV/bin/ffmpeg"
$P -c "import torch, soma; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'soma', soma.__version__)"
echo "== weights $(date +%H:%M)"
mkdir -p weights assets/soma
"$ENV/bin/hf" download nvidia/GEM-X body_s240000.ckpt vitpose.pth motion_native77_e774_s292175.ckpt motion_stats_raw.npz \
  body_video_native77.yaml motion_native77.yaml --local-dir weights
"$ENV/bin/hf" download nvidia/SOMA-X SOMA_neutral.npz SOMA_template_rig.usda SOMA_procedural_transforms.json \
  correctives_model.pt --revision v0.3.3 --local-dir assets/soma
if [ ! -f weights/yolox_humanart.onnx ]; then  # the person finder
  curl -s --fail --location https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip \
    --output weights/yolox_humanart.zip
  $P - <<'PY'
from pathlib import Path
from shutil import copyfileobj
from zipfile import ZipFile
with ZipFile("weights/yolox_humanart.zip") as archive:
    member = next(name for name in archive.namelist() if name.endswith("/end2end.onnx"))
    with archive.open(member) as source, Path("weights/yolox_humanart.onnx").open("wb") as target:
        copyfileobj(source, target)
PY
  rm weights/yolox_humanart.zip
fi
du -sh weights assets/soma
echo "== ALL DONE $(date +%H:%M)"

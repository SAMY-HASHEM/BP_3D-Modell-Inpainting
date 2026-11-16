#!/bin/bash
# A model gets a skeleton with named bones, and its pieces follow one bone each.
#     bash scripts/tools/rig.sh model.glb outputs/x/work/rig
# UniRig (local_ai_models/rigging/UniRig, its own environment conda/envs/unirig) predicts the skeleton and the weights;
# rig_parts.py names the bones and makes the pieces rigid. Writes rigged.glb, rigged.json (the parts) and rigged.jpg.
set -e
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="$(realpath "$1")"; OUT="$(realpath -m "$2")"; SEED="${3:-12345}"
mkdir -p "$OUT"
export PATH="$PROJECT/conda/envs/unirig/bin:$PATH"
cd "$PROJECT/local_ai_models/rigging/UniRig"
rm -rf tmp
bash launch/inference/generate_skeleton.sh --input "$MODEL" --output "$OUT/skeleton.fbx" --seed "$SEED" > "$OUT/log.txt" 2>&1
bash launch/inference/generate_skin.sh --input "$OUT/skeleton.fbx" --output "$OUT/skin.fbx" >> "$OUT/log.txt" 2>&1
bash launch/inference/merge.sh --source "$OUT/skin.fbx" --target "$MODEL" --output "$OUT/unirig.glb" >> "$OUT/log.txt" 2>&1
cd "$PROJECT"
"${BLENDER:-blender}" -b --python scripts/tools/rig_parts.py -- "$OUT/unirig.glb" "$OUT/rigged.glb" 2>&1 | grep -E '^names|^pieces'
"$PROJECT/conda/envs/inpainting-3d-modelling/bin/python" scripts/tools/rig_picture.py "$OUT/rigged"
echo "Result: $OUT/rigged.glb"

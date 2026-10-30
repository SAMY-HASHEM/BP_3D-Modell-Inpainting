#!/bin/bash
# Downloads every AI model this project uses into ./local_ai_models. What is already there is skipped,
# so it is safe to run again (activate.sh runs it by itself when something is missing).
#
#   ./download_models.sh                  # everything
#   ./download_models.sh trellis qwen     # only some groups
#
# Groups:
#   trellis   TRELLIS.2-4B, the TRELLIS v1 decoder, DINOv3, RMBG-2.0 (Hugging Face cache, about 23 GB;
#             needs a Hugging Face login with access to DINOv3 and RMBG-2.0)
#   qwen      Qwen-Image-Edit-2511 for ComfyUI + Lightning LoRA + multiple-angles LoRA (about 30 GB),
#             in local_ai_models/comfyui/<kind>/ as ComfyUI expects them
#   mask      Grounding DINO + SAM 2.1 (only if ./install_mask.sh made the mask-ai environment)
#   omnipart  OmniPart (only if ./install_omnipart.sh made the omnipart environment)
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$PROJECT_DIR/local_ai_models"
CONDA_DIR="$PROJECT_DIR/conda"
WANTED=${*:-trellis qwen mask omnipart}

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate inpainting-3d-modelling  # for the hf command
export HF_HUB_CACHE="$MODELS_DIR" HF_MODULES_CACHE="$MODELS_DIR/modules"
mkdir -p "$MODELS_DIR"

# one file of a Hugging Face repo, flat into local_ai_models/comfyui/<kind>/
comfy_file() {
    local kind=$1 repo=$2 path=$3
    local target="$MODELS_DIR/comfyui/$kind/$(basename "$path")"
    if [ -f "$target" ]; then
        echo "  have $(basename "$path")"
        return
    fi
    echo "  downloading $(basename "$path")"
    hf download "$repo" "$path" --local-dir "$MODELS_DIR/comfyui/.download" >/dev/null
    mkdir -p "$(dirname "$target")"
    mv "$MODELS_DIR/comfyui/.download/$path" "$target"
}

for group in $WANTED; do
    echo "--- $group"
    case $group in
    trellis)
        if ! hf auth whoami >/dev/null 2>&1; then
            echo "  not logged in to Hugging Face: request access to"
            echo "  https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m and https://huggingface.co/briaai/RMBG-2.0,"
            echo "  then 'hf auth login' and run this again"
            continue
        fi
        hf download microsoft/TRELLIS.2-4B >/dev/null
        hf download microsoft/TRELLIS-image-large ckpts/ss_dec_conv3d_16l8_fp16.json ckpts/ss_dec_conv3d_16l8_fp16.safetensors >/dev/null
        hf download facebook/dinov3-vitl16-pretrain-lvd1689m >/dev/null
        hf download briaai/RMBG-2.0 --include "*.py" "*.json" "model.safetensors" >/dev/null
        echo "  ready"
        ;;
    qwen)
        comfy_file diffusion_models Comfy-Org/Qwen-Image-Edit_ComfyUI split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors
        comfy_file text_encoders Comfy-Org/Qwen-Image_ComfyUI split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
        comfy_file vae Comfy-Org/Qwen-Image_ComfyUI split_files/vae/qwen_image_vae.safetensors
        comfy_file loras lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
        comfy_file loras fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA qwen-image-edit-2511-multiple-angles-lora.safetensors
        rm -rf "$MODELS_DIR/comfyui/.download"
        ;;
    mask)
        if [ -d "$CONDA_DIR/envs/mask-ai" ]; then
            conda run -n mask-ai python "$PROJECT_DIR/scripts/mask/mask_ai.py" --download
        else
            echo "  skipped: no mask-ai environment (./install_mask.sh)"
        fi
        ;;
    omnipart)
        if [ -d "$CONDA_DIR/envs/omnipart" ]; then
            hf download omnipart/OmniPart >/dev/null
            echo "  ready"
        else
            echo "  skipped: no omnipart environment (./install_omnipart.sh)"
        fi
        ;;
    *)
        echo "  unknown group (trellis, qwen, mask, omnipart)"
        ;;
    esac
done
echo "models are in $MODELS_DIR ($(du -shL "$MODELS_DIR" | cut -f1))"

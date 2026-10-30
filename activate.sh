# Activates this project's conda environment (installed by install.sh in ./conda).
#
#   source activate.sh
#
# Must be sourced, not run, so the activation stays in your terminal.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "Run it with 'source activate.sh' (not ./activate.sh)."
    exit 1
fi

_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$_project_dir/conda/etc/profile.d/conda.sh" ]; then
    echo "conda isn't installed in $_project_dir/conda yet. Run ./install.sh first."
else
    source "$_project_dir/conda/etc/profile.d/conda.sh"
    conda activate inpainting-3d-modelling
    cd "$_project_dir"
    # The model weights used to be in ./models: move them once, and leave a link for jobs still running
    if [ -d models ] && [ ! -L models ] && [ ! -e local_ai_models ]; then
        mv models local_ai_models && ln -s local_ai_models models
        echo "moved ./models to ./local_ai_models"
    fi
    export HF_HUB_CACHE="$_project_dir/local_ai_models" HF_MODULES_CACHE="$_project_dir/local_ai_models/modules"
    # Download what is missing (download_models.sh skips what is already there)
    if [ ! -d local_ai_models/models--microsoft--TRELLIS.2-4B ] \
        || [ ! -f local_ai_models/comfyui/loras/qwen-image-edit-2511-multiple-angles-lora.safetensors ]; then
        echo "some AI models are missing, downloading them into ./local_ai_models"
        "$_project_dir/download_models.sh"
    fi
    echo "inpainting-3d-modelling is active (Python: $(command -v python))"
fi
unset _project_dir

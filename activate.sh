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
    echo "inpainting-3d-modelling is active (Python: $(command -v python))"
fi
unset _project_dir

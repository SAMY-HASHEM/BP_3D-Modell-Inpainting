#!/usr/bin/env bash
# Builds a Blender comparison scene of all models in one output folder (see compare_in_blender.py).
#
#   ./scripts/compare_in_blender.sh outputs/<picture>            # saves comparison.blend
#   ./scripts/compare_in_blender.sh outputs/<picture> --render   # also saves comparison.png
#
# Blender runs as a Flatpak, which starts in another folder, so it gets absolute paths.
set -e
if [ -z "$1" ]; then
    echo "Usage: $0 outputs/<picture> [--render]"
    exit 1
fi
script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/compare_in_blender.py"
folder="$(realpath "$1")"
shift
flatpak run org.blender.Blender -b --python "$script" -- "$folder" "$@"

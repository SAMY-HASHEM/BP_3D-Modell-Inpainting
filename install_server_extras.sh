#!/bin/bash
# What a flow needs on the uni server besides TRELLIS.2, when EVERY node runs there (the web page's "everything on
# the uni server", for a computer without a GPU). Ask first what is missing, from any computer:
#
#     python3 scripts/server/cloud.py check
#
# Then, on the server (JupyterLab terminal), in the project folder:
#
#     bash install_server_extras.sh                    # everything below
#     bash install_server_extras.sh blender ollama     # only some
#
#   blender   Blender 5.2.3 into ~/Applications/blender (where make_edit.py looks): the pictures of a model
#   ollama    Ollama into ~/Applications/ollama, started, and qwen3.6:27b pulled (17 GB): the vision AI of the data
#             sheet and the judges. It must be running when a job comes: the last line shows how to start it again
#   unirig    the skeleton model in its own environment conda/envs/unirig (scripts/tools/rig.sh)
#   mask      the mask AI (./install_mask.sh)
#   comfyui   only the fix that lets an installed ComfyUI 0.45 start with the project's torch 2.6
#             (scripts/tools/fix_comfy_kitchen.py); run it again after ComfyUI or comfy_kitchen were updated.
#             The project's ComfyUI listens on 127.0.0.1:8198: others run theirs on 8188 and 8189 on the uni server
#
# Ollama 0.13.0 of the uni server (installed for everyone) can't pull qwen3.6:27b, and the server can't reach
# ollama.com (Nov 18): the judges use qwen3-vl:30b, which is there (JUDGE_MODEL in scripts/pipeline/judge.py).
#
# Not in here: the picture AI (ComfyUI + Qwen-Image-Edit, ./download_models.sh qwen). Without it a part needs its own
# pictures and is put on the top of the object; a part from words can't be made on the server.
# Nothing is installed outside the home folder. Written on Nov 16 2026 and tried on the developer's PC only, not on
# the server: if a step fails there, its message says which one.
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WANTED=${*:-blender ollama unirig mask}
APPS="$HOME/Applications"
mkdir -p "$APPS"

for part in $WANTED; do case $part in
comfyui)
    python3 "$PROJECT/scripts/tools/fix_comfy_kitchen.py"
    ;;
blender)
    if [ -x "$APPS/blender/blender" ]; then echo "Blender is there: $("$APPS/blender/blender" --version | head -1)"; continue; fi
    echo "Blender 5.2.3 ..."
    curl -fL --retry 3 -o "$APPS/blender.tar.xz" https://download.blender.org/release/Blender5.2/blender-5.2.3-linux-x64.tar.xz
    rm -rf "$APPS/blender" && mkdir "$APPS/blender" && tar -xJf "$APPS/blender.tar.xz" -C "$APPS/blender" --strip-components=1 && rm "$APPS/blender.tar.xz"
    "$APPS/blender/blender" -b --version | head -1 || echo "Blender does not start here: a system library is missing (the message above names it)"
    ;;
ollama)
    if ! [ -x "$APPS/ollama/bin/ollama" ] && ! command -v ollama >/dev/null; then
        echo "Ollama ..."
        mkdir -p "$APPS/ollama"
        curl -fL --retry 3 -o "$APPS/ollama.tar.zst" https://ollama.com/download/ollama-linux-amd64.tar.zst
        tar --zstd -xf "$APPS/ollama.tar.zst" -C "$APPS/ollama" && rm "$APPS/ollama.tar.zst"
    fi
    OLLAMA="$(command -v ollama || echo "$APPS/ollama/bin/ollama")"
    if ! curl -s -m 3 http://localhost:11434/api/tags >/dev/null; then
        nohup "$OLLAMA" serve > "$PROJECT/outputs/ollama.log" 2>&1 &
        sleep 5
    fi
    "$OLLAMA" pull qwen3.6:27b || echo "qwen3.6:27b could not be pulled: the judges then only measure (see outputs/ollama.log)"
    echo "Ollama must run when a job comes. After a restart of the server:  nohup $OLLAMA serve > $PROJECT/outputs/ollama.log 2>&1 &"
    ;;
unirig)
    ENV="$PROJECT/conda/envs/unirig"
    RIG="$PROJECT/local_ai_models/rigging/UniRig"
    [ -d "$RIG" ] || git clone --depth 1 https://github.com/VAST-AI-Research/UniRig.git "$RIG"
    if ! [ -x "$ENV/bin/python" ]; then
        "$PROJECT/conda/bin/conda" create -y -q -p "$ENV" python=3.11
        P="$ENV/bin/python"
        $P -m pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
        grep -v flash_attn "$RIG/requirements.txt" > "$RIG/requirements_without_flash_attn.txt"
        $P -m pip install -q -r "$RIG/requirements_without_flash_attn.txt"
        $P -m pip install -q spconv-cu120
        $P -m pip install -q torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.5.0+cu121.html --no-cache-dir
        $P -m pip install -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
        $P -m pip install -q numpy==1.26.4
    fi
    # one skeleton of its own example: this also downloads the model's weights (a job itself runs without the internet)
    (cd "$RIG" && PATH="$ENV/bin:$PATH" bash launch/inference/generate_skeleton.sh --input examples/giraffe.glb --output /tmp/unirig_test.fbx > /dev/null 2>&1) \
        && echo "UniRig works (a test skeleton was made)" || echo "UniRig is installed, but its test skeleton failed"
    ;;
mask)
    "$PROJECT/install_mask.sh"
    ;;
*)  echo "unknown part: $part (blender, ollama, unirig, mask)"; exit 1 ;;
esac; done
echo "Done. Ask again what the server has:  python3 scripts/server/cloud.py check"

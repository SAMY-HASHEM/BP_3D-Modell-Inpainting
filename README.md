# BP_3D-Modell-Inpainting

Bachelor project (BP) by Samy Hashem at the University of Duisburg-Essen.

## Topic

3D model inpainting: regenerating only selected parts of a 3D model, built on [TRELLIS.2](https://github.com/microsoft/TRELLIS.2).

## Author

Samy Hashem

## Structure

**How it all works, in one file: [docs/STRUCTURE.md](docs/STRUCTURE.md)** (the core, the three ways to drive it, where it runs, every file by layer).

| Folder | What it does |
|---|---|
| `scripts/generation/` | picture(s) or a video → 3D model (`generate.py`, `multi_image.py`) |
| `scripts/inpainting/` | changing one area of a model: the core (`inpaint.py`, `inversion.py`) and what makes its pictures and masks |
| `scripts/pipeline/` | the node flows (`flow.py`, `nodes_core.py`), their web page's back end, the character pipeline |
| `scripts/server/` | where the AI runs: this PC (`inpaint_server.py`) or the uni server through Sciebo (`cloud.py`, `cloud_worker.py`) |
| `scripts/rigging/` | a skeleton for a generated humanoid: the rig head (trained on TRELLIS.2's shape latent) and `rig_from_glb.py` |
| `scripts/tools/` | renders and comparisons |
| `blender/` | the Blender add-on (`inpaint_addon.py`) and the node editor (`pipeline_nodes.py`) |
| `user_interface_model/`, `flows/`, `comfyui/` | the web app's pages (a chat, the data, the city), the node flows as files, the same nodes in ComfyUI |
| `docs/` | `STRUCTURE.md` (how it works), `REQUIREMENTS.md` (progress), `HOW-TO.md` (running it, settings), `FLOWS.md` (the pipelines in detail) |

Setup: `./install.sh` (TRELLIS.2) and `./install_mask.sh` (mask AI); then `source activate.sh`.
All AI models are in `local_ai_models/` (not in Git); `./download_models.sh` gets them, and `source activate.sh` runs it by itself when something is missing.

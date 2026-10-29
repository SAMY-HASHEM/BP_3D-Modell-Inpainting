# BP_3D-Modell-Inpainting

Bachelor project (BP) by Samy Hashem at the University of Duisburg-Essen.

## Topic

3D model inpainting: regenerating only selected parts of a 3D model, built on [TRELLIS.2](https://github.com/microsoft/TRELLIS.2).

## Author

Samy Hashem

## Structure

| Folder | What it does |
|---|---|
| `scripts/models/` | picture(s) or a turntable video → 3D model, and cleaning it (`generate.py`, `multi_image.py`, `video_views.py`, `fix_normals.py`, `remove_floaters.py`, `omnipart_generate.py`) |
| `scripts/mask/` | which area changes: the mask AI behind "Find part" (`mask_ai.py`) and OmniPart's parts (`make_part_mask.py`, `transfer_parts.py`, `cut_parts.py`, `cut_regions.py`) |
| `scripts/regenerate/` | making the area new: `inpaint.py` (methods repaint and inversion), `inversion.py`, `edit_picture.py` (the edited picture), `insert_part.py`, `replace_part.py` |
| `scripts/server/` | the uni GPU server: `cloud.py` + `cloud_worker.py` (outside the network), `inpaint_server.py` (inside), `run_job.py`, `sweep.py` (trying settings) |
| `scripts/tools/` | `render_views.py`, `compare_in_blender.py/.sh`, `project_texture.py` (camera helpers) |
| `blender/` | the Blender add-on |
| `docs/` | how-to guides for the two server connections |

Setup: `./install.sh` (TRELLIS.2), `./install_mask.sh` (mask AI), `./install_omnipart.sh` (OmniPart); then `source activate.sh`.

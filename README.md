# BP_3D-Modell-Inpainting

Bachelor project (BP) by Samy Hashem at the University of Duisburg-Essen.

## Topic

3D model inpainting: regenerating only selected parts of a 3D model, built on [TRELLIS.2](https://github.com/microsoft/TRELLIS.2).

## Author

Samy Hashem

## Structure

| Folder | What it does |
|---|---|
| `scripts/generation/` | picture(s) or a turntable video → 3D model: `generate.py` (one picture), `multi_image.py` (several), `views_from_video.py`, `views_from_picture.py` (missing sides with Qwen), `fix_normals.py`, `remove_floaters.py`, `try_settings.py` (trying settings on the uni server) |
| `scripts/inpainting/` | changing one area of a model: `inpaint.py` (methods repaint and inversion), `inversion.py`, `picture_from_prompt.py` (the edited picture), `make_edit.py` + `send_edit.py` (before, after and 2D mask of an edit, measured or by hand), `prompt_picture.py` (a prompt painted into a picture's area), `find_part.py` (the mask AI behind "Find part"), and the parts way: `cut_parts.py`, `cut_regions.py`, `transfer_parts.py`, `insert_part.py`, `replace_part.py` |
| `scripts/server/` | the AI servers: `inpaint_server.py` (this PC, or the uni server inside the network), `cloud.py` + `cloud_worker.py` (uni server from anywhere, through Sciebo), `run_job.py` |
| `scripts/pipeline/` | the node pipelines: `nodes_core.py` (what every node does), `flow.py` (runs a node flow from a file in `flows/`), `flow_page.py` (the flows as a web page: before / after sliders, Run, settings, and the edit by hand with a brush and a prompt; the page itself is `web/index.html`), the inpainting flow's nodes `data_sheet.py` (local AI: data sheet of the inputs) and `edit_judge.py` (local judge of an edit), and the character pipeline (`run_pipeline.py`, `auto_pipeline.py`, `grid_judge.py`, ...) |
| `web/` | the web app's front end: `app.html` (the main page: add data, prompts, Run, the result in 3D; a friendly and a developer view) and `index.html` (the older node page). The back end is `scripts/pipeline/flow_page.py`; see `docs/web-app.md` |
| `flows/` | node flows as files, e.g. `girl_crown.json`: a crown inpainted onto a generated head (see `docs/inpaint-flow.md`); `flows/hand/` has what the web page's edit by hand painted |
| `scripts/tools/` | `render_views.py`, `compare_in_blender.py/.sh`, `project_texture.py` (camera helpers) |
| `blender/` | the Blender add-on |
| `docs/` | the texts: `REQUIREMENTS.md` (progress and next steps), `SETTINGS.md` (settings guide), `SERVER.md` (uni GPU server), `pipeline-nodes.md` and `inpaint-flow.md` (the node pipelines), and how-to guides for the three ways to run the AI (this PC, uni inside, uni from anywhere) |

Setup: `./install.sh` (TRELLIS.2) and `./install_mask.sh` (mask AI); then `source activate.sh`.
All AI models are in `local_ai_models/` (not in Git); `./download_models.sh` gets them, and `source activate.sh` runs it by itself when something is missing.

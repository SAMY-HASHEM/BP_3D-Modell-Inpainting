# How the project works, in one file

One core, several ways to drive it. Read this first; the other documents go deeper into one part each.

## 1. The core: TRELLIS.2, and regenerating one area of a model

TRELLIS.2 turns pictures into a 3D model in three stages. Every stage starts from noise and is led by the pictures.

```
picture(s) ──► 1 STRUCTURE ──► 2 SHAPE ──► 3 COLOURS ──► decode ──► model.glb
               where is            the fine      the colours
               something            surface      on it
               (64 x 64 x 64
               small cubes)
```

Inpainting runs the same three stages again, but only inside a marked area (`scripts/inpainting/inpaint.py`):

```
model.glb ─┐
area       ├─► 1/6 read the model into TRELLIS.2's own form (its three latents)
picture(s) ┘   2/6 MASK       the area as small cubes of the 64 grid: these are regenerated, all others kept
               3/6 STRUCTURE  ┐ inside the mask: made new, led by the AFTER picture
               4/6 SHAPE      ├ outside the mask: held on the original's own path (the "rewind")
               5/6 COLOURS    ┘ and the new area looks at exactly the original around it
               6/6 decode ──► new model.glb
```

The four ideas that make it work (rates in [REQUIREMENTS.md](REQUIREMENTS.md)):

| Idea | What it does | Where |
|---|---|---|
| **Rewind (flow inversion)** | the original model is followed back to noise, step by step; while regenerating, everything outside the mask is set to that path, so it stays exactly | `inversion.py` |
| **Key-value injection** | in every attention layer the new area sees the original's own keys and values, so it grows into what is kept | `inversion.py: KVInjection` |
| **Picture swap** | the AFTER picture counts only inside its 2D mask, the original render outside; several sides can each bring their own BEFORE, AFTER and 2D mask | `inpaint.py: picture_pair, mixed_cond` |
| **Protected mask** | the mask is built so that cubes holding surface that must stay are left out | add-on: `keep_the_rest`, `protect_green` |

So an inpainting job is always the same five things: **the model, the area (3D), BEFORE, AFTER and the 2D mask per side.**
Everything else in the project exists to make those five things well.

## 2. Four ways to make those five things

```
                      ┌──────────────────────── A. BLENDER ADD-ON (by hand) ───────────────────────┐
                      │ box + three colours ─► picture of the change ─► judge ─► job.zip           │
 pictures / model ───►├──────────────────────── B. NODE FLOW (by itself) ──────────────────────────┤───► the core ───► model.glb
                      │ data sheet ─► base model ─► edit (measured) ─► inpaint ─► judge            │    (section 1)
                      ├──────────────────────── C. CHARACTER PIPELINE (grid loop) ─────────────────┤
                      │ sort ─► base ─► judge every block ─► re-inpaint the bad ones ─► combine    │
                      ├──────────────────────── D. CORRECTION LOOP ────────────────────────────────┤
                      │ base from one view ─► fit the other views ─► judge ─► correct ─► keep      │
                      └────────────────────────────────────────────────────────────────────────────┘
```

### A. The Blender add-on (`blender/inpaint_addon.py`): one change, by hand

| Step in the panel | What happens | Runs | Script |
|---|---|---|---|
| Where does the AI run? | this PC: starts the server; uni: through the Sciebo cloud | | section 3 |
| **A. Mark the area** | a space box (arrows, dots, *Fit to colours* on the 64 grid) | Blender | |
| | *Find part*: 16 pictures of the box, a word → the part on every picture → a vote per face | mask AI | `inpainting/find_part.py` |
| | the three colours with the brush (Q red: goes, E yellow: may change, R green: never touched, W erase, Shift + click: the whole piece) | Blender | |
| **B. What should be there** | Reference = a picture or a video of the new part: every side cut out | background AI | `inpainting/picture_check.py parts` |
| | *1. Make picture*: per side BEFORE (the model), AFTER (the model without the red part, the new part put in by depth), 2D MASK (everything that changed) | Blender (Cycles, CPU) | |
| | no Reference, only words: the picture AI paints them into the area | picture AI | `inpainting/picture_from_prompt.py` |
| | *Judge the picture*, *Make it better*, *Enhance by itself* | measured + local AIs | `inpainting/picture_check.py` |
| **C. Regenerate in 3D** | the job as one .zip: model, area, pictures, settings | | `server/inpaint_server.py` or `server/run_job.py` → `inpainting/inpaint.py` |
| D. Job files | save the job, load a result | | |
| E. Geometry Nodes | clean up loose pieces, compare with the original, keep the original outside | Blender | |

### B. The node flow (`scripts/pipeline/flow.py`): the same change, made by rules

A flow is a small file in `flows/`: nodes and which node's result goes into which. `nodes_core.py` says what every node
runs; `flow.py` runs them in order (a node whose result is there is not run again); the web page shows it.

```
00 Input ─► 01 Data sheet ─► 02 Base model ─► 02b Pick seed ─► 03 Edit ─► 04 Inpaint ─► 05 Judge
            data_sheet.py    generate.py /    pick_seed.py     make_edit.py  inpaint.py   edit_judge.py
            (views, sides:   multi_image.py   (outline +       (BEFORE, AFTER,            (measured +
            local AI)                         local AI)        2D mask, area)             local AI)
```

Front ends of the same flows: the terminal (`flow.py flows/x.json`), the web app (`web/app.html` + `flow_page.py`,
see [FLOWS.md](FLOWS.md#3-web-app)), the older node page (`web/index.html`). Details: [FLOWS.md](FLOWS.md#1-inpainting-flow).

### C. The character pipeline (`scripts/pipeline/auto_pipeline.py`): many pictures → one detailed model

```
01 sort ─► 02 base ─► 03 judge every block ─► 04 re-inpaint the bad blocks, zoomed ─► 05 combine ─┐
sort_pictures.py  multi_image.py  grid_judge.py     refine_region.py                  combine_parts.py │
                          ▲                                                                            │
                          └──────────────── a repair stays only if its block got better ◄─────────────┘
```

The older hand-tuned version of it is `run_pipeline.py` (parts instead of blocks, hair strands). Details:
[FLOWS.md](FLOWS.md#2-character-pipeline).

### D. The correction loop (`scripts/pipeline/correct.py`): the best base, then corrected picture by picture

```
1 BASE ───────► 2 FIT ────────► 3 JUDGE ─────────────► 4 CORRECT ────────► 5 KEEP ──────────────────► next round
two candidates:  every view's    per view, per cell:     only that part,     seen from EVERY view: stays only
one view alone,  camera on the   measured (outline,      from that view's    if its own view says "closer" and no
all sides mixed  model           colour, detail) AND     crop, the model     other view says "worse"; then the
(or given ones)  (fit_camera)    seen (local AI); bad    turned to that      whole model is judged again: a round
the better one                   only if both say so     camera              that lowers its score is undone
```

What it showed so far (bee robot, Nov 15): a real turnaround (front, side, back of one pose) mixed with multidiffusion
gives the best base (68.7 against 52.3 from the front alone); free renders that don't fit together doubled parts
when mixed (the robot arms), so both candidates are made and judged. Corrections judged from their own view only
tore the model seen from the others (68.7 down to 56.0); judged from every view, the damaging ones are refused.
Replacing a part on purpose (a new head) works; polishing a model that is roughly right is still an open experiment.

On the uni server: `cloud.py pipeline <pictures> --loop correct --sides "front, side, back" --threshold 80`. Without
Ollama there the judge is Qwen3-VL-8B (weaker). Base models made on the server can be given: `--base a.glb b.glb`.

## 3. Where the heavy work runs

```
THIS PC      add-on ──HTTP 127.0.0.1:8766──► inpaint_server.py (keeps TRELLIS.2 loaded) ──► inpaint.py
             flow nodes with "where": "pc" start generate.py / multi_image.py / inpaint.py directly

UNI SERVER   add-on or cloud.py ──► Sciebo cloud folder ──► cloud_worker.py (watches the folder, one job per GPU,
             takes the code from GitHub) ──► run_job.py ──► inpaint.py ──► result back into the cloud folder
```

Both servers read the same job .zip. How to use them: [HOW-TO.md](HOW-TO.md#1-this-pc),
[HOW-TO.md](HOW-TO.md#2-the-uni-server-from-anywhere), [HOW-TO.md](HOW-TO.md#3-on-the-uni-server-itself).

## 4. The local AI models and what each is for

| Model | Used for | By |
|---|---|---|
| TRELLIS.2-4B (+ DINOv3 as its eyes) | generating and inpainting | `generate.py`, `inpaint.py` |
| RMBG-2.0 | background away: cutting views and parts out | `data_sheet.py`, `picture_check.py`, `make_edit.py` |
| Qwen3.6-27B with vision (Ollama) | says what a picture shows and which side (a sheet's views together); judges seeds, the picture of a change and results | `data_sheet.py`, `pick_seed.py`, `picture_check.py`, `edit_judge.py` |
| Qwen-Image-Edit 2511 (ComfyUI) | paints a prompt into a picture; missing sides; reworks the picture of a change | `picture_from_prompt.py`, `views_from_picture.py`, `picture_check.py` |
| Grounding DINO + SAM 2.1 | a word → the part on a picture | `find_part.py` |
| Qwen3-VL-8B | sorts the pictures of the character pipeline | `sort_pictures.py`, `organise_pictures.py` |

## 5. The files by layer

| Layer | Files | Lines |
|---|---|---|
| **Core** | `inpainting/inpaint.py`, `inversion.py`; `generation/generate.py`, `multi_image.py` | ~1,400 |
| **Making the five things** | `inpainting/make_edit.py`, `picture_check.py`, `picture_from_prompt.py`, `find_part.py`; `pipeline/data_sheet.py`, `edit_judge.py`, `pick_seed.py`, `judge.py` | ~2,300 |
| **Running it** | `server/inpaint_server.py`, `run_job.py`, `cloud.py`, `cloud_worker.py`; `pipeline/flow.py`, `nodes_core.py` | ~2,000 |
| **Front ends** | `blender/inpaint_addon.py`; `web/app.html`, `index.html`, `login.html` + `pipeline/flow_page.py`, `accounts.py`; `blender/pipeline_nodes.py`; `comfyui/` | ~8,700 |
| **Character pipeline** | `pipeline/correct.py`, `auto_pipeline.py`, `run_pipeline.py`, `sort_pictures.py`, `grid_judge.py`, `refine_region.py`, `combine_parts.py`, `hair_strands.py`, `preview.py`; `generation/organise_pictures.py` | ~2,800 |
| **Helpers** | `tools/render_views.py`, `render_preview.py`, `project_texture.py`, `compare_in_blender.py`; `generation/views_from_picture.py`, `views_from_video.py`, `fix_normals.py`, `run_tests.py` | ~1,700 |
| **Kept for an open task** | `generation/try_settings.py` (requirement 2: the best settings), `tools/compare_models.sh` | ~80 |

Paths are under `scripts/` unless they start with `blender/`, `web/` or `comfyui/`.

## 6. Where the results are

One folder per object, everything of it inside:

```
outputs/<object>/               e.g. robot_bee, robot_hand_1, super_woman
    v01_<title>.glb             its models as versions; the title says how each was made (PICKED = the seed judge's pick)
    versions.md                 every version: title, when, where it came from
    work/                       what the programs need: the flows' numbered node folders (a finished node is not
                                made again), the correction loop's steps (work/correct/00_views ... 05_model), the
                                pictures as TRELLIS.2 saw them (inputs/ next to each model)
```

A model is put next to `work/` as the next version when it is made (a flow and the correction loop do it themselves;
for anything else: `python scripts/tools/arrange_outputs.py --do`). Not per object, and only there while they are
used: `outputs/cloud/` (job packages for the uni server), `blender/` (the add-on's work folder), `server_jobs/`,
`server_runs/`, `_unnamed/` (results of jobs whose name didn't say the object).

## 7. The documents

| File | What is in it |
|---|---|
| `STRUCTURE.md` | this file: how everything works |
| [REQUIREMENTS.md](REQUIREMENTS.md) | what is done, how far, and what is next |
| [HOW-TO.md](HOW-TO.md) | running it: this PC, the uni server, what every setting does |
| [FLOWS.md](FLOWS.md) | the node flows, the character pipeline and the web app in detail |

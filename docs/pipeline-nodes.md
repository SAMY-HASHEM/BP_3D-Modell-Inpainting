# Node pipeline: many pictures → one detailed model

**Built like Geometry Nodes, not inside Blender:** every step is a **node** with its own folder, inputs,
settings and outputs; the nodes are chained and each can be re-run on its own. It is a Python program
(`scripts/pipeline/run_pipeline.py`) that runs outside Blender; only node **06_hair** uses real Blender
Geometry Nodes. TRELLIS.2 stays the core model; helpers (vision AI, Blender) are combined with it.
Example data: `pictures/local model` (Morrigan), 14 pictures: a grey turnaround (3 views), grey close-ups,
a colour full shot (with back wings and a scene), colour close-ups.

## The principle: colour at the very end
Everything is made **without colour first, like Roman marble statues**: only the shape, judged in clay
renders (`render_preview.py --clay`) next to the artist's grey sculpt renders. The colours come **only at
the very end, from the pictures, in one pass over the finished model**, so the left and right side, the
front and the back and all parts match (no colour seams between parts, symmetric). The colours TRELLIS.2
makes along the way in the shape nodes are ignored.

## The grid loop (the user's idea, `scripts/pipeline/auto_pipeline.py`)
No hand-made parts: the base model is cut into blocks (256 of TRELLIS.2's 1024 grid) and judged; only bad blocks
are redone, and a repair stays only if it helped.

```
01 sort → 02 base → 03 judge every block (>= 85 stays) → 04 re-inpaint the worst blocks, zoomed, crop >= 1024 px
                         ↑                                  bad shape: shape + colours; bad colours: colours only
                         └── 05 combine into one mesh, judge at once; a repair that isn't better is rejected
```
- **Judge** (`grid_judge.py`): measured, not asked. Colour: the colour difference (CIE Lab) on the object's pixels.
  Shape: inside the block's 3D box projected into the picture, the share of the object the model lacks, or of the
  model the object lacks. Only pictures of the model's pose (camera fit >= 80 %); the cameras are fitted once, so
  the rounds compare. Checked on the robot fly: cutting off a hand lowered only its blocks (94 → 68, 81 → 60, 69 →
  58), all others stayed exact. Qwen3-VL-8B gave a white model 85; DINOv3 gave good thin antennae 12.
- **Repairs:** inversion with a soft edge first (the user's VoxHammer idea), then repaint, then inversion from new
  noise. The new block's region joins by `combine_parts.py --single` (own atlas).
- **Heat map** per round: `heatmap.jpg`. Runs: `cloud.py pipeline <folder>`, results with `cloud.py watch`.
- **Result, robot fly (run 3, colours only, before the shape repair was added):** average 72.8 → 79.1 → 82.9, worst
  block 28 → 55 → 59, the wings got their colours and veins.

## How to run it (all on the local GPU)
```
source activate.sh
python scripts/pipeline/run_pipeline.py "pictures/local model" --name morrigan            # newest round
python scripts/pipeline/run_pipeline.py "pictures/local model" --name morrigan --only 04_refine_head
python scripts/pipeline/run_pipeline.py "pictures/<other>" --name <other>                 # a new character
```
A new character starts from the defaults in `NODES`, which already contain every lesson below. `ROUNDS` is
Morrigan's history: what changed in each round and why; `"use"` picks the round of a node that later nodes
build on (`0` = leave it out).

## The process now

```
01_sort ─► 02_base ─► 04_refine_<part> ─► 05_combine ─► 06_hair ─► 99_compare
                      (each part on its own from 02_base, from easy to hard:
                       lower legs, thighs, feet, belly, back, butt, bust, head)
```

| Node | Script | What it does | Output (`outputs/pipeline/<name>/<node>/r<round>/`) |
|---|---|---|---|
| 01_sort | `scripts/pipeline/sort_pictures.py` | the vision AI (Qwen3-VL-8B) cuts sheets into views, tags side / whole / parts / back wings, boxes every part, checks every box; grey vs colour and the hair colour are **measured** | `sorted.json`, `views/`, `contact.jpg` |
| 02_base | `scripts/generation/multi_image.py` | the whole figure: shape from the views of one version (grey turnaround), colours from the whole colour view (colour only) | `base_seed42.glb` |
| 04_refine_<part> | `scripts/inpainting/refine_region.py --sorted` | one part zoomed in and repainted: **colours only** (shape kept) from its **one sharpest complete** picture; the head also gets its shape regenerated | `<part>.glb`, `<part>_work/` |
| 05_combine | `scripts/pipeline/combine_parts.py` | the refined parts put into the base (first in the list wins where two overlap) | `combined.glb` |
| 06_hair | `scripts/pipeline/hair_strands.py` (Blender) | ~35,000 real hair strands grown along the hair shell, Blender's hair nodes (Clump, Noise, Profile) | `hair.blend`, `hair.glb`, renders |
| 99_compare | `run_pipeline.py` | per node `compare.jpg`: pictures \| before \| after, the same camera (front, 45° left / right, **behind**); `overview.jpg`, `history.jpg`, `README.md` | |

### The parts (node 01 boxes them in every picture)
`head`, `hair`, `bust`, `belly`, `back` (seen from behind), `butt` (seen from behind), `thighs`, `lower_legs`,
`feet`. Small boxes get ~4–5× zoom; the old big boxes (chest = shoulders to hips, legs = hips to ankles)
only got 1.5–2×, and the butt sat on the seam between back and legs.

### Three picture lists per TRELLIS.2 node
| Stage | What it decides | Pictures |
|---|---|---|
| 1 structure | *where* there is something (64³) | the same version only: grey + colour views |
| 2 shape | the fine surface | same as stage 1 |
| 3 colour | the colours | colour pictures only (never grey); another version: **colour only** |

## Lessons of the rounds (Morrigan, Nov 3) → rules in the code

| Round | What happened | Rule now |
|---|---|---|
| r1 | base from the 3 grey views + colour 004 (colour only): shape very good, colours too dark | base = grey views for the shape, the whole colour view for the colours |
| r2 | all 9 colour close-ups in the base: hair greener but gloves lost, suit patchy (each picture only 1–2 of 12 steps) | the base gets no close-ups; they go to their part |
| r3 → r4 | head from 6 pictures: black eye make-up and lips → texture guidance 3: pink lips | `texture_guidance 3` for every refine |
| r5 | chest: a back view in the list turned the body around; margin 0.6 of the chest = the whole model | **same side only** (within 90° of the main view); `margin_of_model 0.1` |
| r6 | legs: regenerating the shape from close-ups made them lumpy, colours right | **colours only** (`keep_shape`) for the body and clothes |
| r7 | legs and chest colours only: right colours, shape unchanged | ✔ |
| r8 | head from **one** sharp picture (morrigan-01-001): clean lips and eyes, a real face | `best_colour 1`: one picture per part |
| r9 | finer parts: thighs, feet, belly, bust good; lower legs took a cut-off close-up from below; back got brown skin from back.jpg; butt almost white | the largest **complete** picture; back and butt seen from behind (`PART_SIDE`); back.jpg must not give colours (open) |
| r10 | the best of r1–r9 combined | colour bands where two parts meet (hips, shins, chest) → next process below |

Other findings: grey vs colour is measured by saturation (grey renders 0.05–0.08, colour pictures > 0.38);
the AI named the hair colour wrongly as a hex code (#000080), so it now only names it ("green") and the
colour is measured from the hair pixels (#3c9c74). Photo projection (`project_texture.py`) is no main fix:
the user judged it bad on Oct 23 (it smears where the shape differs).

## The next process (planned): shape first, colour at the very end

Almost every failure was a **colour** problem, and colouring each part separately leaves bands where the
parts meet. So the colours move to the end, over the finished shape:

```
PART 1 SHAPE    01_sort → 02_base → 04_shape_head (and other parts if the shape needs it) → 05_combine (ONE mesh)
                no colour: judged in clay renders (like a marble statue) next to the artist's grey sculpt renders
PART 2 COLOUR   07_colour_base → 08_colour_<part> (colours only, one sharp picture each) → 09_colour_face
                on the one finished mesh, so there are no colour seams between parts
PART 3 HAIR     06_hair: strands on the final shell, colours from the final texture
99_compare      shape rounds and colour rounds side by side, always also from behind
```
Helpers (TRELLIS.2 stays the core): Qwen-Image-Edit (already in the project, `views_from_picture.py`)
colours the grey back view like the front (there is no colour picture of the back and butt) and evens out
the coloured light in the face picture. For the thesis: **TRELLIS.2 alone vs. TRELLIS.2 + this pipeline**
on the same pictures (optionally one commercial model as a reference).

Still to build: 05_combine into one mesh (one texture), grey renders for Part 1, the colour nodes on the
combined mesh, 03_add_wings (only the colour pictures show the back wings), 07_props (pumpkin + bat, bag,
pedestal).

## Why hair has its own node (research, Nov 3)
- **Tokens:** TRELLIS.2 checks its token limit (49,152) only at 1536: too many → it quietly lowers 1536 →
  1408 → 1280 … At 1024 the limit never acts (`trellis2_image_to_3d.py`, `sample_shape_slat_cascade`).
- **Stage 1 is coarse:** 64 × 64 × 64 boxes; on the whole figure the head is ~8 boxes wide, zoomed ~50.
  Zooming helps hair far more than 1536 (1.5×) or more tokens.
- **1536 makes holes in thin parts** (our tests, and a write-up on TRELLIS.2 holes).
- **Strands:** a surface model never makes single strands, so 06_hair grows them in Blender: roots spread
  over the shell, every strand walks down along it (snapped back onto the shell each step, ending where the
  shell ends), then Clump (0.35) + Noise + Profile; colour from the texture at the root (sRGB → linear).
- **Later idea:** DiffLocks (CVPR 2025) makes ~100k strands from one front picture (research code).

## ComfyUI: the pipeline as a node graph (built Nov 3)
`comfyui/bp_inpainting/` (nodes) + `comfyui/workflows/BP Inpainting - pictures to model.json`, see
`comfyui/README.md`. Pictures / video → data sheet → base → parts → combine → hair → compare, each with its
pictures and a 3D viewer (ComfyUI's Preview 3D). The nodes call the same scripts as `run_pipeline.py`.

## What we have and what to add next (research, Nov 3)

**Already on this PC** (`~/ai-models`, ComfyUI from Oct 2026): Qwen3.6-27B with vision (Ollama), SAM 3.1 (text →
mask), FLUX.2, Qwen-Image-Edit 2511; ComfyUI's own nodes for TRELLIS.2, Pixal3D, SAM 3, SAM 3D Body, MoGe /
Depth Anything 3, mesh tools (UV unwrap, bake, remesh, fill holes) and a 3D viewer. In the TRELLIS.2 repo:
`Trellis2TexturingPipeline` (colours an existing mesh from a picture). The uni server's worker did not answer on
Nov 3 (not running); its models still have to be listed.

| Step | Point | What | Why |
|---|---|---|---|
| B | structure | shape first in clay → combine into ONE mesh → colours once at the very end with `Trellis2TexturingPipeline` (+ our zoom and one-picture rules for the parts) → hair | removes the colour bands; the user's principle |
| C | local models | Qwen3.6-27B (Ollama) as the sorter and as a local judge of the rounds; SAM 3.1 masks instead of boxes; Sapiens2 (Meta 2026: 29 body parts, normals; own licence) | exact part outlines; the rounds decided locally, not by Claude |
| D | combining 3D models | SAM 3D Body (in ComfyUI): a fitted human body gives every point of our mesh a body-part label → 3D regions instead of boxes; SegviGen (MIT, 3D part segmentation) as another way; Pixal3D only as a comparison (its weights are locked for the EU) | regions follow the body, no seam through the butt |

Sources: [TRELLIS.2 + Pixal3D in ComfyUI](https://comfyui-wiki.com/en/news/2026-08-22-trellis2-pixal3d-native-comfyui),
[Sapiens2](https://www.marktechpost.com/2026/04/27/meta-ai-releases-sapiens2-a-high-resolution-human-centric-vision-model-for-pose-segmentation-normals-pointmap-and-albedo/),
[SegviGen](https://github.com/Nelipot-Lee/SegviGen), [SAM 3.1](https://www.labellerr.com/blog/sam-3-vs-sam-3-1-performance-comparison/),
[Qwen3.6-27B](https://playground.roboflow.com/models/qwen/qwen3-6-27b).

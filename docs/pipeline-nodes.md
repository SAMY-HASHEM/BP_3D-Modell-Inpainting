# Node pipeline: many pictures → one detailed model

**Built like Geometry Nodes, not inside Blender:** every step is a **node** with its own folder, inputs,
settings and outputs; the nodes are chained and each can be re-run on its own. It is a Python program
(`scripts/pipeline/run_pipeline.py`) that runs outside Blender; only node **06_hair** uses real Blender
Geometry Nodes. TRELLIS.2 stays the core model; helpers (vision AI, Blender) are combined with it.
Example data: `pictures/local model` (Morrigan), 14 pictures: a grey turnaround (3 views), grey close-ups,
a colour full shot (with back wings and a scene), colour close-ups.

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

## The next process (planned): shape first, colour at the end

Almost every failure was a **colour** problem, and colouring each part separately leaves bands where the
parts meet. So the colours move to the end, over the finished shape:

```
PART 1 SHAPE    01_sort → 02_base → 04_shape_head (and other parts if the shape needs it) → 05_combine (ONE mesh)
                judged in grey (clay renders) next to the artist's grey sculpt renders
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

# Node pipeline: many pictures → one detailed model

Built like Geometry Nodes: every step is a **node** with its own folder, its own inputs, its own settings and
its own outputs. Every node can be re-run on its own. Example data: `pictures/local model` (Morrigan).

```
01_sort ─► 02_base ─► 04_refine_<part> (head, chest, back, legs, feet; each on its own from 02_base)
                              │
                              ▼
                         05_combine ─► 06_hair (real strands, Blender Geometry Nodes) ─► 99_compare
          later: 03_add_wings, 07_props (pumpkin + bat, bag, pedestal)
```

**How to run it** (all on the local GPU):
```
python scripts/pipeline/run_pipeline.py "pictures/local model" --name morrigan --round 5
```
| Node | Script | Output (`outputs/pipeline/<name>/<node>/r<round>/`) |
|---|---|---|
| 01_sort | `scripts/pipeline/sort_pictures.py` | `sorted.json`, `views/`, `contact.jpg` |
| 02_base | `scripts/generation/multi_image.py` (`--shape-only`, `--colour-only`) | `base_seed42.glb` |
| 04_refine_<part> | `scripts/inpainting/refine_region.py --sorted` | `<part>.glb`, `<part>_work/` |
| 05_combine | `scripts/pipeline/combine_parts.py` | `combined.glb` |
| 06_hair | `scripts/pipeline/hair_strands.py` (Blender) | `hair.blend`, `hair.glb`, renders |
| 99_compare | `run_pipeline.py` | `overview.jpg`, `history.jpg`, `README.md` |

Every node folder has `node.json` (settings, why, time) and `compare.jpg` (reference | before | after).
**Rounds:** `ROUNDS` in `run_pipeline.py` lists what changed in each round and why; `"use"` picks the round
of a node that later nodes build on (e.g. the base stays r1).

## Three picture lists per TRELLIS.2 node

TRELLIS.2 works in 3 stages. Each node says which pictures go into **which stage**:

| Stage | What it decides | Pictures that belong here |
|---|---|---|
| 1 structure | *where* there is something, in a 64 × 64 × 64 grid | everything of the **same version**: colour + grey views |
| 2 shape | the fine surface (512 → 1024 voxels) | same as stage 1 |
| 3 colour | the colours | only **colour** pictures (grey sculpt renders would make it grey) |

`multi_image.py` already lets the colour stage use only some pictures (`rows`). The new part: a picture can
also be **colour only** (e.g. a colour picture of a different version: it gives colours, but must not change
the shape).

The vision AI (node 01) decides which picture goes into which list, following the rules below. Claude does
not choose by hand.

## Rules the vision AI follows (from the night tests, `outputs/tests/2026-11-03/ROUNDS.md`)

1. Same version only in stages 1 + 2. Different version (wings / no wings, with / without scene) → colour only
   or a separate node.
2. Close-ups never go into a whole-body node. They go to the refine node of their part.
3. Grey sculpt renders = shape pictures (stages 1 + 2), never colour.
4. A picture sheet with several views (turnaround) is first cut into single views.

## The nodes

### 01_organise — sort all pictures
- **Model:** Qwen3-VL-8B (`organise_pictures.py`)
- **Does:** cuts sheets into single views, then tags every picture: colour / grey, part (whole, head, chest,
  legs, foot, back, prop), angle (0° front, 45°, 90° side, 180° back), version facts (back wings? scene?
  props?), cut edges.
- **Morrigan check** (what the AI should find; if it doesn't, its rules get better, not a hand fix):

| Picture | Kind | Part | Angle | Version |
|---|---|---|---|---|
| full-model (3 views) | grey | whole | 180°, ~70°, ~20° | no back wings, no scene |
| close-1 | grey | head + chest | ~10° | |
| close02 (2 views) | grey | legs + foot | 2 angles | |
| back | colour (other skin) | back | 180° | other material |
| money-shot-004 | colour | whole | ~20° | back wings + scene |
| money-shot-001, morrigan-01-001 | colour | head + chest | 2 angles | |
| money-shot-002 | colour | chest | ~0° | |
| money-shot-009, 010 | colour | legs | 2 angles | wings + rail |
| money-shot-008 | colour | foot + hand | close | |
| money-shot-006 | colour | prop: pumpkin + bat | | |
| money-shot-007 | colour | prop: bag | | |
| morrigan-01-002 | colour | pedestal + props | | |

### 02_base_figure — the whole figure
| | |
|---|---|
| stages 1 + 2 | grey 180°, grey ~70°, grey ~20° (same version, nothing in the way) |
| stage 3 colour | colour 004, cut to the figure (colour only: it has wings + rail the grey views don't) |
| mode | stochastic (best + 2× faster in the night tests) |
| resolution | 1024 (1536 made holes in hair in the tests) |
| steps | 12 (12 ≈ 25, 50 worse in the tests) |
| guidance | 7.5 (default; 3–10 changed little) |
| faces | 4 M (keeps detail) |
| seeds | 42 + 7, the AI keeps the one whose colours match the picture best |
| out | figure without back wings, rail and props |

### 03_add_wings — the big back wings
- Only the colour pictures show them (004, 009, 010) → their own node, added with `insert_part.py`.
- Colour + shape from 004 (the wings are whole there), 009 / 010 as extra angles.

### 04_refine_<part> — zoom-in repaint, one node per part
Same method as the head test the user liked: **repaint**, margin 0.6, 25 steps, 1024.
New: each part takes **all** its pictures, each with its own camera and its own crop of the same 3D box.

| Node | stages 1 + 2 | stage 3 colour |
|---|---|---|
| 04_refine_face | close-1 (grey), full-model head crops | 001, morrigan-01-001 |
| 04_refine_chest | close-1 (grey) | 002, 001 |
| 04_refine_legs | close02 (grey, 2 views) | 009, 010 (colour only: rail in front) |
| 04_refine_foot | close02 foot crop | 008 |
| 04_refine_back | back (other material → shape only) | — (keeps the base colours) |

### 05_hair — hair and fur (two parts)

**5a hair shell (TRELLIS.2, zoomed):** makes the hair *surface* right: volume, parting, fringe, no cracks.

| | |
|---|---|
| region | the vision AI's "hair" box in every picture → one 3D box (head + hair down to the tips) |
| stages 1 + 2 | all hair views: close-1, full-model 180° / 70° / 20° crops (the back view is the only one of the hair's back) |
| stage 3 colour | 001, morrigan-01-001, 004 head crop |
| mode | stochastic |
| resolution | 1024 zoomed (not 1536, see below) |
| steps | 25 and 12, both, side by side (not proven yet which is better for hair) |
| tokens | at 1536 only: limit raised to 98,304 + the log checked for "resolution is reduced" |
| after | close the cracks inside the hair region only (fill thin gaps, smooth), the rest stays sharp |

**5b real strands (Blender Geometry Nodes):** TRELLIS.2 makes surfaces, never single strands, so the strands
are grown on the shell:
1. hair shell from 5a → inner surface = where strands start (roots), outer surface = how far they reach
2. guide curves: start at the roots, walk step by step *along* the shell, downwards (repeat zone, each step
   snapped back onto the shell) → strands follow the shell's flow, parting and fringe
3. Interpolate Hair Curves → thousands of children; Clump + Noise (Blender's own hair nodes)
4. colour: each strand takes the texture colour of the shell where it runs
5. out: `hair.blend` (real curves), `hair.glb` (strands as thin strips, .glb can't store curves)

Same node for **fur** (creatures, animals): short length, high density, no fringe.

### 06_props — separate small models
pumpkin + bat (006, morrigan-01-002, 004 crop), bag (007, 004 / 009 crops), pedestal + rail (004,
morrigan-01-002). Placed with the camera of 004.

### 99_compare — side by side
- `overview.jpg`: one row per node: **reference picture crop | before | after**, always the same camera and the
  same crop, so changes are easy to see.
- `README.md`: each node's pictures per stage, settings, time, faces.

## Every node folder
```
outputs/pipeline/morrigan/04_refine_face/
    node.json      inputs, pictures per stage, settings, why, time
    in/            what the node got (links)
    out/           model + previews
    compare.jpg    reference | before | after
```

## Why hair needs its own node (research, Nov 3)

- **Tokens:** TRELLIS.2 only checks the token limit (49,152) at 1536: if the hair makes too many, it quietly
  lowers 1536 → 1408 → 1280 … At 1024 the limit never acts (`trellis2_image_to_3d.py`, `sample_shape_slat_cascade`).
- **Stage 1 is coarse:** it decides *where* there is something in 64 × 64 × 64 boxes. On the whole figure the
  head is only ~8 boxes wide, so gaps between strands can't exist. Zoomed, the hair gets ~50 boxes.
  Zooming helps hair far more than 1536 (1.5×) or more tokens.
- **1536 makes holes in thin parts:** also seen in our tests (hair) and in a write-up on TRELLIS.2 holes:
  thin surfaces lying close together share voxels and the decoder breaks there.
- **Strands:** no surface model makes single strands, so 5b grows them in Blender.
- **Later idea:** DiffLocks (CVPR 2025) makes ~100k strands from one front picture (research code,
  trained on realistic people, so stylised hair is uncertain).

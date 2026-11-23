# The pipeline in stages

From pictures (or a turntable video) to a finished, rigged 3D model, one stage at a time; each stage has one tool we
use by default and what else was tried (Nov 23, 2026, on the motorcycle girl, the Plague Knight, Erik's bust). Every
GPU stage runs on both machines (the hardware profiles: [HOW-TO.md](HOW-TO.md#your-pc-24-gb-or-the-server-48-gb-the-hardware-profiles)).

All of it in one command (each stage skipped when its result is there):

```bash
python scripts/pipeline/make_model.py --video painted.mp4 --grey-video clay.mp4 --frames 108 199 58 149 \
    --out outputs/x --parts "skin=noise:0.35:0.012" "cloth=cotton_jersey:0.6:0.03" [--retopo] [--rig]
python scripts/pipeline/make_model.py --pictures front.png side.png back.png --out outputs/y
```

```
pictures / video ─► 1 views ─► 2 cut-out ─► 3 de-light ─► 4 shape ─► 5 detail ─► 6 retopology ─► 7 colours ─► 8 surface detail ─► 9 rig
                                                                                                                         └─► check at every stage
```

| # | Stage | Default (script) | Models | Also tried | Verdict |
|---|---|---|---|---|---|
| 1 | **Views** | `views_from_video.py`; the outline's width finds front / side / back in a turntable | — | evenly spaced frames | a turn is not evenly fast (the motorcycle girl: side to side 87 frames one way, 163 the other) |
| 2 | **Cut-out** | `views_from_video.py --cutout`, `delight.py` | RMBG-2.0 | — | the biggest part kept: text, stamps, stones on the floor go |
| 3 | **De-light** | `delight.py` (colour / clay render, in linear light) | — | IC-Light (not downloaded) | with a clay render of the same views it is exact: base colour +31 %, fake metal a third |
| 4 | **Shape** | `multi_image.py --shape-only grey --colour-only de-lit` (TRELLIS.2, 1536); faces: `compare_pixal3d.py` (Pixal3D) | TRELLIS.2-4B, Pixal3D | Hi3D, B (colour views for the shape) | grey views give the sharpest shape; Pixal3D the best faces |
| 5 | **Detail** | `paint_front.py --closeup`, `swap_head.py` | — | — | a face or a crown painted from a close-up; a head made alone |
| 6 | **Retopology** | `retopo.py` (a closed outer skin by `skin_surface.py` → Instant Meshes → onto the original surface → UVs → colours baked) | Instant Meshes | QuadriFlow (hung), Blender's voxel remesh (lace on TRELLIS.2's open 1536 patches) | the knight made in one go 940 k → 36 k quads, the motorcycle girl 960 k → 55 k (thin parts too now; the bake smears a little at thin edges); for a character **after** rigging: `transfer_rig.py` (grip PASS on the quads, thumb included) |
| 7 | **Colours** | the shape stage's own (de-lit views); `merge_textures.py` with MV-Adapter's | TRELLIS.2, MV-Adapter (SDXL) | `texture_mesh.py` (TRELLIS.2 texturing), `paint_views.py` (photos projected: ghosted) | the merge: the faithful colours with MV-Adapter's crisp detail |
| 7a | | `texture_mvadapter.py --text "$(describe_colours.py views)"` | MV-Adapter, Qwen3-VL-8B | the front picture alone (a green bike) | the view showing the most plus a colour description |
| 8 | **Surface detail** | `detail_normals.py --parts "face=noise:0.35:0.012" "veil=caban:0.8:0.03"` | Poly Haven CC0 maps (`local_ai_models/detail_maps`), Grounding DINO + SAM 2.1 for the parts | votes per texel (speckled) | as artists do it (Marmoset detail normals): Erik's face got a stone grain, the veil a weave; the parts voted per face from 9 views |
| 9 | **Rig** | `rig_from_glb.py` (rig head + `fit_hands.py`), `hold_item.py` | our rig head | UniRig, GEM-X joints on pictures | hands fitted to the mesh; the grip checked on the skin |

## Lessons from Erik's bust (one picture, fine detail)

- TRELLIS.2 1536 beat Pixal3D on it: the whole crown in 3D, the face sharper (Pixal3D won on the girl's face alone).
- The seed decides the face: three seeds of the one picture gave three faces (one close to the sculpt, one doll-like,
  one with an eye missing). Next: a few seeds at 1024 (~4 min each), scored, the best one made at 1536.
- Partial views hurt: two video frames whose crown was cut off by the picture's edge tore the hood into strips.
  Only whole views for the shape; close-ups for detail (paint_front.py --closeup, picture_normals.py).

## The numbers (`scripts/tools/score_model.py`)

The motorcycle girl's models against her de-lit front, side and back (outline: intersection over union; colour: median
CIELAB distance; sharpness: fine detail against the photo's):

| model | outline | colour ΔE | sharpness |
|---|---|---|---|
| A (grey shape, 1024) | 0.844 | 26.7 | 0.58 |
| B (colour views for the shape) | 0.833 | 25.6 | 0.53 |
| A at 1536 | 0.841 | 25.6 | 0.67 |
| N4 (de-lit colours, 1536) | 0.841 | 23.3 | 0.74 |
| TRELLIS.2 texturing | 0.840 | 24.7 | 0.67 |
| MV-Adapter, side view + text | 0.841 | 23.9 | 1.30 |
| **merged** (N4 + MV-Adapter) | 0.841 | **23.2** | **1.45** |
| merged, the text by Qwen3-VL | 0.840 | 23.4 | 1.43 |

(The photo painter's 16.3 does not count: it was scored on the very pictures it projected.)

## Models and licences

All local; nothing in a cloud. Downloaded on the home PC where the server can't reach the files (Hugging Face's and
GitHub's file servers): MV-Adapter + SDXL (11 GB), Instant Meshes.

| Model | Licence | Size | GPU (24 GB PC) |
|---|---|---|---|
| TRELLIS.2-4B (generate, texture) | MIT | 16 GB | 3-7 GB (low VRAM) |
| Pixal3D | its own (research) | 43 GB | 14-20 GB (low VRAM) |
| MV-Adapter ig2mv + SDXL base + VAE fp16 | Apache-2.0, OpenRAIL++, MIT | 11 GB | 15.5 GB |
| Qwen3-VL-8B | Apache-2.0 | 17 GB | ~18 GB |
| RMBG-2.0, SAM 2.1, Grounding DINO, Real-ESRGAN | CC BY-NC (RMBG), Apache-2.0 | 3 GB | small |
| Instant Meshes | BSD | 3 MB | CPU |
| Poly Haven detail maps (12 materials) | CC0 | 0.5 GB | — |
| Hunyuan3D-Paint 2.1 | **not used**: its licence excludes the EU | | |

## The motorcycle girl, all ten models

Sciebo `motorcycle_girl_2026-11-23/v17_ALL_10_models_same_angles.jpg`: the same three angles for every model made;
best: the merge of the de-lit colours (N4) and MV-Adapter's (`v19_MERGED_best`).

# What every setting does, and where it lives in the code

Line numbers are from today's code (`TRELLIS.2/` is Microsoft's code, `scripts/` is ours). TRELLIS.2's
default values are in its `pipeline.json` (in `local_ai_models/models--microsoft--TRELLIS.2-4B/snapshots/…/`).

## The way of one picture through TRELLIS.2

```
picture
  │  background removed (RMBG-2.0), object cut out                 preprocess_image()
  ▼
DINOv3 "reads" the picture → numbers that describe it (the condition)   get_cond()          line 164
  │      once at 512 px (for stage 1 and the rough shape), once at 1024 px (fine shape, colours)
  ▼
Stage 1  STRUCTURE  which cells of a coarse grid are filled           sample_sparse_structure()  line 188
  ▼
Stage 2  SHAPE      the exact surface inside these cells               sample_shape_slat_cascade() line 277
  │      (at 1024/1536: first rough at 512, then refined at the full resolution)
  ▼
Stage 3  COLOURS    colour, metal, roughness for every cell            sample_tex_slat()     line 391
  ▼
decode → mesh + colour voxels                                          decode_latent()       line 456
  ▼
save as .glb: simplify/remesh, unwrap, bake the texture               our save_glb()  generate.py:124
```

Every stage is a **flow model**: it starts from random noise and, step by step, turns the noise into
the result, each step guided by the picture. The settings below change how these steps run.

---

## `--resolution` (512 / 1024 / 1536): how fine the 3D grid is

- **Ours:** `generate.py:34` turns it into TRELLIS.2's "pipeline type": 512 → `512`, 1024 → `1024_cascade`,
  1536 → `1536_cascade`.
- **TRELLIS.2:** `trellis2_image_to_3d.py:519-541` picks the models for it, and `:541` the size of the coarse
  grid of stage 1 (32 cells per side for the cascades).
- **The cascade** (`:277-364`): the shape is first made at 512 (`:297-311`), then every cell is split into
  smaller ones (`upsample`, `:323`), and the fine shape is made again on this finer grid: 1024 → 64 cells
  per side, 1536 → 96 (`:328-331`).
- **Important:** for the fine step at 1536 TRELLIS.2 uses its **1024 model** (`:578-585`, model
  `shape_slat_flow_model_1024`). So 1536 runs that model **above the size it was trained for**. In our
  tests (valkyrie) this gave big flat panels, while 1024 gave a clean, smooth surface.
- **More = ** finer details, but more GPU memory and, at 1536, the risk of those flat panels.

## `--steps` (default 12): how many small steps from noise to result

- **Ours:** `generate.py:104-112`: the same number of steps for all three stages.
- **TRELLIS.2:** `samplers/flow_euler.py:115-118` makes the list of time points from 1 (pure noise) to 0
  (finished). `rescale_t` (`:116`, 5.0 / 3.0 in `pipeline.json`) puts more of them where the shape is
  decided. Each step moves the result a little along the predicted direction: `:80`
  `x_prev = x_t - (t - t_prev) * v`.
- **More = ** smaller, more careful steps: cleaner, less "rushed" surfaces, but slower (time grows with steps).
- **Our test (valkyrie):** 12 = flat panels and broken feathers, **25 = cleanest**, 50 = most natural body
  but noisier, twice as slow.

## `--seed` (default 42): which random noise to start from

- **TRELLIS.2:** `trellis2_image_to_3d.py:538` `torch.manual_seed(seed)`: every stage's starting noise comes
  from it.
- **Same seed + same picture + same settings = the same model.** Another seed = another variant (a different
  guess for what the picture doesn't show).

## `--guidance` (default 7.5): how strictly the shape follows the picture

- **Ours:** `generate.py:107-108` (and `multi_image.py`) for stage 1 and stage 2.
- **TRELLIS.2:** `samplers/classifier_free_guidance_mixin.py:9-27`. Every step, the model predicts twice:
  once **with** the picture, once **without** it (an all-zero picture, `trellis2_image_to_3d.py:181`). The
  result is pushed away from "without" towards "with": `:17`
  `pred = guidance * pred_with + (1 - guidance) * pred_without`. `guidance_rescale` (`:20-27`, 0.5/0.7)
  keeps this from overshooting.
- `guidance_interval` (`samplers/guidance_interval_mixin.py:9-10`, `[0.6, 1.0]`): the push is only used in
  the first part of the steps, while the rough form is decided.
- **Higher = ** closer to the picture, harder shapes; too high gives artifacts. **Lower = ** softer, more
  natural, but thin parts break apart (valkyrie: guidance 3 gave a smoother body and a broken cape).

## `--texture-guidance` (default 1.0 = off): the same for the colours

- Stage 3 (`pipeline.json`: guidance 1.0, interval `[0.6, 0.9]`). 2-4 makes the colours follow the picture
  more closely.

## `--max-tokens` (default 49152): a limit for the fine shape step

- **TRELLIS.2:** `trellis2_image_to_3d.py:287` and `:327-339`. A *token* is one cell of the fine grid that
  the model works on. If there are more than the limit, the resolution is lowered by 128 (1536 → 1408 →
  1280 …) until they fit, but never below 1024 (`:335`). Fewer tokens = less memory when decoding.
- **Only matters for big objects:** the valkyrie needs fewer than 20,000 at 1536 (the limit never changed
  anything), a head with long hair needs more (40,000 lowered it to 1152 and so it fit into 24 GB).

## `--faces` (default 1,000,000): how many triangles the saved model keeps

- **Ours:** `generate.py:124-135`. TRELLIS.2 makes far more triangles (11 million at 1536), the file keeps
  this many. At 1024 the surface is first **remeshed** (rebuilt as an even surface from the grid, `remesh`
  in `generate.py:127-129` and o_voxel's `to_glb`); at 1536 that doesn't fit into 24 GB, so the mesh is
  only simplified.
- **More = ** finer edges kept, bigger files (4 million ≈ 150 MB). It can't add detail TRELLIS.2 didn't make.

## Several pictures: `multi_image.py --mode multidiffusion | stochastic`

- **Ours:** `multi_image.py`, class `MultiImageFlowModel`: every step, the model predicts once per picture;
  *multidiffusion* averages these predictions, *stochastic* uses one picture per step in turn. No camera
  angles are needed: TRELLIS.2 places every object the same way, whatever the view.
- `scripts/models/video_views.py` takes such pictures from a turntable video.

## Recommended

| Object | Command |
|---|---|
| Most objects | `--resolution 1024 --steps 25 --faces 4000000` |
| Small, very detailed objects | `--resolution 1536 --steps 25 --faces 4000000` |
| Long hair / very big objects at 1536 | add `--max-tokens 40000` |
| Turntable video | `video_views.py video.mp4 --cutout`, then `multi_image.py pictures/<video>/view_*.png …` |

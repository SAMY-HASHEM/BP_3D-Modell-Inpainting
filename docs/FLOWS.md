# The node pipelines and the web app

The ways that run by themselves, node after node. How the whole project fits together: [STRUCTURE.md](STRUCTURE.md).

- [1. Inpainting flow](#1-inpainting-flow)
- [2. Character pipeline](#2-character-pipeline)
- [3. Web app](#3-web-app)

## 1. Inpainting flow

*Inpainting flow: a new part onto a model, as nodes*

**Example:** the crown (`pictures/crownrenderspin.mp4`, a video of it turning) onto the girl's head
(`pictures/tetiana-kravchuk-sculpt.jpg`, a sculpt sheet with five views). First the girl's head is generated,
then the crown is inpainted onto it. TRELLIS.2 runs on the uni GPU server, everything else on this PC.

Built like Geometry Nodes: every step is a node with inputs, settings and a result folder; the links say which
result goes where. The nodes are defined once (`scripts/pipeline/nodes_core.py`) and run in two places that do
exactly the same:

| Where | How |
|---|---|
| a flow file, no Blender | `python scripts/pipeline/flow.py flows/girl_crown.json` |
| Blender's node editor | tree type *BP Pipeline* → sidebar (N) → *BP* → **New inpainting flow** → *Run up to here* on a node |

A third place is **the web page**: `python scripts/pipeline/flow_page.py --open` (this PC only,
http://127.0.0.1:8777, the page itself is `user_interface_model/old/nodes.html`). It shows every flow of `flows/` as its node chain and
works on the same flow files and result folders as `flow.py`:

| On the page | What it does |
|---|---|
| a click on a node | its pictures, settings, folder and model; *Open in Blender* opens the model in a new Blender window |
| **before / after slider** | the edit node: before against the plan; a judge: the original (or the plan) against every candidate, from the same four sides, with the edit's area in red and the judge's numbers |
| the settings of a node | changed in place and saved into the flow file (*Save settings*) |
| **Run up to here**, *Run the whole flow*, *Stop the run* | starts `flow.py` for this flow (its output: `outputs/<name>/work/run.log`); before it uses the uni server it asks |
| *Uni server* | the server jobs of this flow's nodes and how far they are (`cloud.py status <job> ...`); *All jobs* reads every job in the cloud, which takes over a minute |
| **Edit by hand** (edit node) | see below |

**Edit by hand** replaces the measured place of the part where it does not fit (e.g. a crown that floats):

1. *Move the part* on the main side: drag it, pull the white dot to size it. The other three sides follow from its
   place in 3D (`make_edit.py --place`).
2. *Paint the area* (red: this changes too) or *Protect* (blue: this stays) on any side: corrections of the 2D mask
   (`make_edit.py --hand`, kept in `flows/hand/<flow>_<node>/add_<view>.png` and `erase_<view>.png`).
3. Or *describe the new part*: the local picture AI (Qwen-Image-Edit through ComfyUI, `picture_from_prompt.py`, about half
   a minute) paints the prompt into the red area of the main side. Every picture made stays in a row to choose
   from; the chosen one is the AFTER picture then (`make_edit.py --after`), no part picture is pasted.
4. *Apply* writes it into the flow file (the edit node's `place`, `side`, `hand`, `after`) and makes the edit again
   (about half a minute on this PC). The inpainting steps after it then wait for a new run. *Back to the measured
   place* takes it out again; old results stay on disk and are found again.

While painting: *Undo* / *Redo* (Ctrl+Z, Ctrl+Shift+Z; the last 15 strokes per side), `[` and `]` change the brush's
width, **B** paint, **E** protect, **V** move the part, and *Fill what I circle* (**F**): a line drawn around an area
fills all of it, so a large area needs one stroke.

### A part from words (no picture of the part)

An edit node without a `part` input and with a `prompt` needs no picture of the new part:

```
{"id": "edit", "type": "edit", "prompt": "a wide pointed wizard hat on the head", "in": {"model": "base", "object": "object_sheet"}}
```

`make_edit.py --prompt` then does this on this PC (about 45 s):

1. BEFORE = the model from its main side (the side that fits the object's own picture, or `side`).
2. The local picture AI (Qwen-Image-Edit) gets BEFORE and the sentence "add … to the object, keep everything else
   exactly" and paints the part where the words say. Nothing is marked in the picture (`words.png`).
3. The part and its 2D mask are **what differs** between that picture and BEFORE: differences over a limit, small
   specks removed, only the big connected pieces kept. Outside of them AFTER stays the exact BEFORE.
4. The 3D area is put where the model's surface is under the part: a part that tops the model there (a hat) gets the
   area all around the model's middle; any other (a shield in a hand, wings on the back) gets it on the side that is
   seen, starting at the surface, so the far side of the model stays out of it.

After that the flow is the same: shape → details → colours, each with its judge (the judge is told the prompt as the
part's name). `prompt_seed` gives another picture for the same words. *Edit by hand* works on such an edit too: the
painted corrections of the mask, and moving what the picture AI made. Only one side has a picture (as with a single
picture of a part); the other sides are TRELLIS.2's guess.

Tried on Nov 11 on the wanderer (512 base): a wizard hat (2.0 % of the picture changed, one piece), a round shield in
the free hand (3.3 %), wings on the back from behind (3.2 %): the mask was clean each time.

### A part with its own picture, at any place (`at`)

With a picture of the part the measured place is the top of the object. For any other place the edit node has `at`,
where the part goes, in words:

```
{"id": "edit", "type": "edit", "at": "in the left hand", "in": {"model": "base", "part": "part_sheet", "object": "object_sheet"}}
```

The picture AI then gets BEFORE and the part's own cut-out picture ("add the <part> <at> to the object, exactly like
the one shown in image 2") and paints that part there, at the object's scale and angle. The rest is the same as with
words only: the part and its 2D mask are what differs from BEFORE, the 3D area is put where the model is under it.
Only the main side gets a picture this way, also when the part has four sides. A place given by hand (*Edit by hand*)
wins over `at`; a `prompt` beside a part's pictures is still not used.

Tried on Nov 12 on the wanderer with the crown's picture: "on the head" (0.7 % of the picture, seated on the head at
the head's size, which the measured collage does not do) and "held in the hand" (1.2 %, it was put on top of the staff
the hand holds).

### The structure

```
00 Input: object ─► 01 Data sheet ─┬► 02 Base model seed 42 ─┬► 02b Pick seed ─┬────────────────────────────┐
(pictures 1..n,     (local AI)     └► 02 Base model seed 1  ─┘   (measured)    │                            │
 videos, or both)                                                              ▼                            ▼
                                                                            03 Edit ─► 04 Inpaint: shape ─► 05 Judge
00 Input: part ───► 01 Data sheet ────────────────────────────────────────►  (pictures   (seeds 1, 2)        │ best
                    (local AI)                                                + 3D area)                    ▼
                                                                                 │      04 Inpaint: details ─► 05 Judge
                                                                                 │                             │ best
                                                                                 └────► 04 Inpaint: colours ─► 05 Judge ─► result
```

The live version of this picture, with every node's state, time and result pictures, is written while a flow runs:
`outputs/<name>/work/flow.html` (open it in a browser; it refreshes by itself).

### The nodes

| Node | Runs on | Script | What it does | Result |
|---|---|---|---|---|
| **00 Input** | – | – | pictures (1..n), videos, or both, of one thing; files or folders. One input for the object, one for the part | the file list |
| **01 Data sheet** | this PC | `pipeline/data_sheet.py` | a sheet with several views is cut into single views (measured); a video gives frames of its turn; per view measured: size, sharpness, grey or colour, cut off, symmetry; the local AI (Qwen3.6-27B, Ollama) says what it shows and which side, a sheet's views shown to it together (the setting `sides`, e.g. `"front, side, back"`, says a sheet's views from left to right instead: the AI called the bee robot's back a front); rules pick the views | `sheet.json`, `views/`, `contact.jpg` |
| **02 Base model** | uni server (or this PC) | `server/cloud.py generate --wait` | TRELLIS.2 from the sheet's `base` views, the most frontal first | the model `.glb` |
| **02b Pick seed** | this PC | `pipeline/pick_seed.py` | the seed that fits the views best: half measured (its outline from 8 sides, mirrored too), half the local AI (the views above, the model's eight renders below: defects and 1 to 10) | `pick.json`, `judged.jpg` per model |
| **03 Edit** | this PC | `inpainting/make_edit.py` | where the part goes, nothing placed by hand (on the middle of the top; *Fit*: **between** things that stand up beside the middle, e.g. hair buns, or **over** them with *Replace*: what the part covers is regenerated too): per side BEFORE (render), AFTER (collage with the part), 2D MASK; the 3D area with a job per small cube: keep / delete / edit | `edit.json`, `three.jpg`, `area_map.jpg`, `area.npy` |
| **04 Inpaint** | uni server | `server/cloud.py edit --wait` | one step: **shape** → **details** → **colours**, each from the result before; several seeds = several candidates in one server job | the candidates `.glb` |
| **05 Judge** | this PC | `pipeline/edit_judge.py` | measured: is the rest kept, is the part there, the colours; the local AI lists defects; the best candidate goes on, with pass / redo | `judge.json`, `compare.jpg` |

#### The edit's plan: a job for every small cube (changed Nov 12)
The 3D area is a set of small cubes, each with one job. `area_map.jpg` shows them: green KEEP, red DELETE, blue EDIT.

| Job | Which cubes | What happens |
|---|---|---|
| **KEEP** | a voxel the model's surface runs through, and the inside of the model | stays the model's own, exactly |
| **DELETE** | with *Replace*: KEEP cubes that lie under the part's 2D mask from every side that has a picture, but not under its rim (1.5 voxels) | regenerated with the part |
| **EDIT** | the empty room of the area | the part grows here |

- The jobs are decided on **TRELLIS.2's own 64³ voxels**, placed exactly as the inpainting places the model (with the
  room above it). Before, everything within 1.5 voxels of the surface was KEEP, also the empty room there, measured in
  the model's own units: the part could not reach the model everywhere.
- **Inside** is a voxel with the model's surface on at least five of its six sides (five, because a bust is open at
  the bottom). Before it was "under the model's highest surface", which also closed the room under an arm or a chin:
  right for a crown, wrong for a part at the side.
- **The rim is not deleted.** On the girl (Nov 10) the lowest layer of the area lost 130 of the hair's 308 voxels and
  got nothing back: a hole between hair and crown, seen as the crown "floating". What is deleted must be covered by
  the part again, so the model's surface under the part's edge stays.
- **The main side** is the side whose outline fits the object's own picture (how much the two shapes cover each
  other); the colours only decide between sides with the same outline (front and back). Colours alone took
  Nefertiti's profile for her front, and a sun disc painted onto a profile faces sideways.

#### The data sheet's lists (what TRELLIS.2 gets)
- `base`: the views a model is generated from. Main view = the most symmetric of the views the AI calls *front*; then
  one view per other side (three-quarter, side, back), the sharpest of each. Two views of the same side are not both used.
- `sides`: a part's four sides in turning order. From a video: its frames at 0, 90, 180, 270 degrees of the turn.

#### The judge's numbers
| Number | Measured how | Counts for |
|---|---|---|
| `kept` | outside the edit's area (its 3D cubes seen from that side + the 2D mask): 100 − 300 × share of pixels that appeared or went; 100 − 1.5 × mean colour difference (CIE Lab); the lower one; the worst of the 4 sides | pass needs ≥ 85 |
| `part` | inside the 2D mask: 100 − 300 × share that stayed like the original (6 px tolerance); mean of the sides with a picture | pass needs ≥ 70 |
| `colour` | inside the 2D mask: 100 − 1.5 × difference of the mean colours to the AFTER picture | only in the colours step |
| `psnr`, `chamfer` | masked PSNR of the renders and mean surface distance, both outside the area | written down only: the papers' numbers |
| AI score 1–10 | Qwen3.6-27B sees the AFTER picture, the result from the same side and from the next side; lists defects | breaks ties within 5 points |

The limits (85, 70) and the weights are first guesses: they are what gets tuned together ("learn / improve").
`kept`, `part` use the same scales as the grid judge (`grid_judge.py`: 85 = 5 % of the pixels wrong, or a colour
difference of 10).

### Run it
```
source activate.sh
python scripts/pipeline/flow.py flows/girl_crown.json                  # everything; done nodes are skipped
python scripts/pipeline/flow.py flows/girl_crown.json --upto edit      # only up to a node
python scripts/pipeline/flow.py flows/girl_crown.json --structure      # only refresh flow.html
```
The server needs no new code for this: the nodes send the same jobs as before (`generate`, inpainting job zips) and
wait for them (`cloud.py ... --wait <folder>`, `cloud.py wait <job> <folder>`).

A new flow = a copy of `flows/girl_crown.json` with other files. Results: `outputs/<name>/work/<node>/<hash of the
settings>/`; the same settings are never computed twice.

### What published work measures (Nov 10)
| Work | Inputs | Unedited region | Edit quality |
|---|---|---|---|
| [VoxHammer](https://arxiv.org/abs/2508.19247) (training-free, TRELLIS; 3DV 2026) | model, 3D mask, an edited render (FLUX Fill) | Chamfer distance, masked PSNR / SSIM / LPIPS | DINO-I to the edited picture, CLIP-T, FID, user study |
| [EditVerse3D](https://www.emergentmind.com/papers/2607.07187) (trained, TRELLIS; Jul 2026) | model, a coarse 3D box, one reference picture | Chamfer distance, PSNR / SSIM / LPIPS | DINO, FID on held-out parts |
| [Nano3D](https://www.emergentmind.com/papers/2510.15019) (training-free, no mask) | model, text | – | add / remove / replace templates |
| LL3M, CADSmith (agents) | – | – | a separate vision model as critic of renders + measured geometry checks, a bounded number of rounds |

None of them measures the edited region itself with a ground truth (there is none); our `part` number does it against
the collage's 2D mask. The agent papers' pattern (generator → measured checks + a separate vision critic → bounded
retries) is the loop this flow is built for.

A ready test set for the judge and the flow: [Edit3D-Bench](https://huggingface.co/datasets/huanngzh/Edit3D-Bench) (MIT,
6 GB): 100 models × 3 edits, each with `model.glb`, a 3D mask (`3d_edit_region.glb`), a 2D mask and an edited picture.
[Hi3DEval](https://arxiv.org/abs/2508.05609) scores whole objects and single parts (geometry, texture) close to human
preference, as a possible second local judge.

### Open (to decide together)
- **The loop:** a `redo` verdict does nothing yet. Next: redo with other seeds / a wider or lower area / `--replace`,
  at most N rounds, keeping only what scores better (as the grid loop does).
- **Placement:** with a picture of the part the Edit node measures only one place, the top of the object. Any other
  place: `at` (in words, the picture AI places the part's own picture), or *Edit by hand*. A part from words is placed
  by the picture AI, anywhere on the side that is seen.
- **Only one side has a picture** when the picture AI places the part (words, or `at`): the depth is TRELLIS.2's
  guess. Tried on Nov 12: asking the picture AI for the same part on the side and back views, with the front AFTER
  picture as its second picture. It painted the whole front view again instead. The angle LoRA of
  `views_from_picture.py` on the AFTER picture is the next thing to try.
- **The judge's limits and weights**, and whether the AI's score should be able to veto.
- **ComfyUI:** the nodes are only in the flow runner and in Blender so far.

## 2. Character pipeline

*Node pipeline: many pictures → one detailed model*

(The second node pipeline, a new part inpainted onto a model with a local judge: [FLOWS.md](#1-inpainting-flow).)

**Built like Geometry Nodes, not inside Blender:** every step is a **node** with its own folder, inputs,
settings and outputs; the nodes are chained and each can be re-run on its own. It is a Python program
(`scripts/pipeline/run_pipeline.py`) that runs outside Blender; only node **06_hair** uses real Blender
Geometry Nodes. TRELLIS.2 stays the core model; helpers (vision AI, Blender) are combined with it.
Example data: `pictures/local model` (Morrigan), 14 pictures: a grey turnaround (3 views), grey close-ups,
a colour full shot (with back wings and a scene), colour close-ups.

### The principle: colour at the very end
Everything is made **without colour first, like Roman marble statues**: only the shape, judged in clay
renders (`render_preview.py --clay`) next to the artist's grey sculpt renders. The colours come **only at
the very end, from the pictures, in one pass over the finished model**, so the left and right side, the
front and the back and all parts match (no colour seams between parts, symmetric). The colours TRELLIS.2
makes along the way in the shape nodes are ignored.

### The grid loop (the user's idea, `scripts/pipeline/auto_pipeline.py`)
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

### How to run it (all on the local GPU)
```
source activate.sh
python scripts/pipeline/run_pipeline.py "pictures/local model" --name morrigan            # newest round
python scripts/pipeline/run_pipeline.py "pictures/local model" --name morrigan --only 04_refine_head
python scripts/pipeline/run_pipeline.py "pictures/<other>" --name <other>                 # a new character
```
A new character starts from the defaults in `NODES`, which already contain every lesson below. `ROUNDS` is
Morrigan's history: what changed in each round and why; `"use"` picks the round of a node that later nodes
build on (`0` = leave it out).

### The process now

```
01_sort ─► 02_base ─► 04_refine_<part> ─► 05_combine ─► 06_hair ─► 99_compare
                      (each part on its own from 02_base, from easy to hard:
                       lower legs, thighs, feet, belly, back, butt, bust, head)
```

| Node | Script | What it does | Output (`outputs/<name>/work/<node>/r<round>/`) |
|---|---|---|---|
| 01_sort | `scripts/pipeline/sort_pictures.py` | the vision AI (Qwen3-VL-8B) cuts sheets into views, tags side / whole / parts / back wings, boxes every part, checks every box; grey vs colour and the hair colour are **measured** | `sorted.json`, `views/`, `contact.jpg` |
| 02_base | `scripts/generation/multi_image.py` | the whole figure: shape from the views of one version (grey turnaround), colours from the whole colour view (colour only) | `base_seed42.glb` |
| 04_refine_<part> | `scripts/inpainting/refine_region.py --sorted` | one part zoomed in and repainted: **colours only** (shape kept) from its **one sharpest complete** picture; the head also gets its shape regenerated | `<part>.glb`, `<part>_work/` |
| 05_combine | `scripts/pipeline/combine_parts.py` | the refined parts put into the base (first in the list wins where two overlap) | `combined.glb` |
| 06_hair | `scripts/pipeline/hair_strands.py` (Blender) | ~35,000 real hair strands grown along the hair shell, Blender's hair nodes (Clump, Noise, Profile) | `hair.blend`, `hair.glb`, renders |
| 99_compare | `run_pipeline.py` | per node `compare.jpg`: pictures \| before \| after, the same camera (front, 45° left / right, **behind**); `overview.jpg`, `history.jpg`, `README.md` | |

#### The parts (node 01 boxes them in every picture)
`head`, `hair`, `bust`, `belly`, `back` (seen from behind), `butt` (seen from behind), `thighs`, `lower_legs`,
`feet`. Small boxes get ~4–5× zoom; the old big boxes (chest = shoulders to hips, legs = hips to ankles)
only got 1.5–2×, and the butt sat on the seam between back and legs.

#### Three picture lists per TRELLIS.2 node
| Stage | What it decides | Pictures |
|---|---|---|
| 1 structure | *where* there is something (64³) | the same version only: grey + colour views |
| 2 shape | the fine surface | same as stage 1 |
| 3 colour | the colours | colour pictures only (never grey); another version: **colour only** |

### Lessons of the rounds (Morrigan, Nov 3) → rules in the code

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

### The next process (planned): shape first, colour at the very end

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

### Why hair has its own node (research, Nov 3)
- **Tokens:** TRELLIS.2 checks its token limit (49,152) only at 1536: too many → it quietly lowers 1536 →
  1408 → 1280 … At 1024 the limit never acts (`trellis2_image_to_3d.py`, `sample_shape_slat_cascade`).
- **Stage 1 is coarse:** 64 × 64 × 64 boxes; on the whole figure the head is ~8 boxes wide, zoomed ~50.
  Zooming helps hair far more than 1536 (1.5×) or more tokens.
- **1536 makes holes in thin parts** (our tests, and a write-up on TRELLIS.2 holes).
- **Strands:** a surface model never makes single strands, so 06_hair grows them in Blender: roots spread
  over the shell, every strand walks down along it (snapped back onto the shell each step, ending where the
  shell ends), then Clump (0.35) + Noise + Profile; colour from the texture at the root (sRGB → linear).
- **Later idea:** DiffLocks (CVPR 2025) makes ~100k strands from one front picture (research code).

### ComfyUI: the pipeline as a node graph (built Nov 3)
`comfyui/bp_inpainting/` (nodes) + `comfyui/workflows/BP Inpainting - pictures to model.json`, see
`comfyui/README.md`. Pictures / video → data sheet → base → parts → combine → hair → compare, each with its
pictures and a 3D viewer (ComfyUI's Preview 3D). The nodes call the same scripts as `run_pipeline.py`.

### What we have and what to add next (research, Nov 3)

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

## 3. Web app

*The web app: front end and back end*

The project as one web page: add data, say what you want, press Run, see the 3D result. All pages are in one folder,
`user_interface_model/`; the back end and the engine are the same for all of them.

### The main page: a chat, the data, the city (`user_interface_model/app.html`, at `/app`, Nov 16)

```
 CHAT                                          THE CITY
 you: "replace the axe in his hand             our device  │ river │  uni server
       with this sword" + 3 files              buildings   │bridge │  buildings
 guide: the plan (what each data is,           ─────────────────────────────────
        what goes away, where) → Run           a card per building: name, what works there,
 guide: the run, with Stop → the result        where, time, loading bar
```

- **The chat** takes words and data together: pictures, videos, 3D models, any file (button, drop, or paste).
- **What the words mean** is worked out by the local AI (`POST /api/understand`: Qwen3.6-27B sees the words, the file
  names and a small picture of each data): which data is the model to change, which the new part, what goes away and
  where the part goes. Tried with the warrior: two files of him + a picture of a sword and "replace the axe in his hand
  with this sword" gave object, object, part, "axe", "in his hand" in 14 s. Without the local AI, simple rules do it
  (the first data is the model).
- **The plan** is shown before anything runs and every line of it can be changed; Run is off, with the reason, while
  the plan can't work.
- **The city** shows a run from above: every building is a local AI, a model or plain processing, on our device (left
  bank) or the uni server (right bank; the bridge is the cloud folder). It reads the same `GET /api/flow` as the other
  pages and turns the flow's nodes into the buildings' states:

| Building (its form) | What works there | Where | A flow's nodes |
|---|---|---|---|
| City gate (a pylon gate, a guard) | the data arrives | our device | input |
| House of scribes (a mud-brick house, a scribe) | Qwen3.6-27B reads every picture | our device | data sheet |
| The great pyramid (the master builder) | TRELLIS.2 builds the model | uni server | base |
| Painters' pavilion (a painter) | Qwen-Image-Edit paints the wish | our device | edit |
| House of life (a mastaba, a priest with an ankh) | UniRig finds the bones | our device | none yet |
| Builders' yard (an obelisk in its scaffold, a mason) | the new part is placed | our device | none yet |
| The forge (a step pyramid with a fire, a metal worker) | TRELLIS.2 inpaints shape, details, colours | uni server | inpaint |
| Hall of judgement (a hall of columns, Ma'at with feather and scales) | measured, then Qwen3.6-27B | our device | pick seed, judge |
| The gallery (an obelisk and a statue) | the finished model | our device | – |

  The city is drawn as ancient Egypt, like the rest of the site: the Nile runs between our device and the uni server,
  the bridge over it is the cloud folder. Every form and every figure is one function in the page (`SHAPES`, `WORKERS`).
- **Everything on the uni server** (*More* in the plan, the default): for a computer without a GPU. The flow file gets
  `"runs": "server"`; Run sends the flow and its files as ONE job (`cloud.py flow`), the server runs every node
  itself (`scripts/server/run_flow.py`) and sends the state back every 15 s; the page shows it like a run of its own,
  and every building then says "uni server". See [HOW-TO.md](HOW-TO.md#a-computer-without-a-gpu-everything-on-the-uni-server).
  On a computer without an NVIDIA GPU (no `nvidia-smi`, e.g. a laptop whose graphics are in the processor) this is
  not a choice but the rule: `flow_page.py` sends **every** run there (any flow, *Run up to here* runs the whole flow),
  the plan offers nothing else, the local AI and the picture AI are not asked on this computer (the plan is read by
  the simple rules). `--gpu` / `--no-gpu` say it instead of looking. The map is drawn that way: only the gate and the
  gallery are on our side of the Nile.
- **The bridge can be clicked** (the Sciebo login and the connection): it shows the saved login (never its password),
  takes a new one (login name + Sciebo app password; tried first, saved like `cloud.py setup` does, readable only by its
  owner) and tests the way in two steps: 1 Sciebo answers with this login (seconds), 2 the worker on the uni server
  answers a check job (up to a minute; it says the server's GPUs and what is missing there). The lamp on the bridge is
  grey (not tested), orange (testing), green (connected), red (not connected). `GET /api/cloud`, `POST /api/cloud/login`,
  `POST /api/cloud/test`; refused on a shared page (`--accounts`): the login belongs to the computer the page runs on.

- **Made after Shneiderman's eight golden rules**; where each is in the page is marked in its code with `(rule N)`:
  one look and the same words everywhere (1), Enter / paste / drop (2), every action answered by a chat line, a bar or
  a lit building (3), a run ends with one message and the result (4), Run only when the plan can work, errors say
  what to do (5), removed data and a cleared chat can be undone, a run stopped (6), the page proposes and the user
  decides (7), the plan, the steps and the buildings stay on the screen (8).
- **Not there yet:** a run from the page is still the flow of section 1 (the part's shape from noise), so the skeleton
  clinic and the fitting workshop show "not in this run": the way of [FIRST-SUCCESS.md](FIRST-SUCCESS.md) has no flow
  nodes yet. A 3D model as data is shown, but a run can't start from one. The demo button plays the first success in
  the city without starting anything.

### The page before it (`user_interface_model/old/app.html`, at `/old/app`)

Made after the sketches in `UI-skitch/`: data rows with a prompt each, the developer view with every node and the edit
by hand of a mask. What follows describes it; its back end is the one the main page uses.

| Part | File | What it is |
|---|---|---|
| **Front end** | `user_interface_model/old/app.html` | the page itself: one file, no build step. Two colours (sand gold and black), hieroglyphs behind it |
| **Back end** | `scripts/pipeline/flow_page.py` | a small server on this PC only (127.0.0.1:8777). It writes flow files and starts `flow.py` |
| the engine | `scripts/pipeline/flow.py`, `nodes_core.py` | unchanged: the same nodes and result folders as from the terminal or Blender |
| the node page | `user_interface_model/old/nodes.html` (at `/`) | the node chain from left to right; still there, linked from the menu |

```
source activate.sh
python scripts/pipeline/flow_page.py --open      # then http://127.0.0.1:8777/app (the main page) or /old/app
```

Opened as a file (double click on `app.html`) the page only shows its look: Run needs the back end.

### What happens on Run

```
the page (app.html)                         the back end (flow_page.py)                  the engine
───────────────────                         ───────────────────────────                  ──────────
Add data, prompts, Run
  ├─ POST /api/upload?flow&role&name  ───►  pictures/uploads/<project>/<object|part>/
  ├─ POST /api/app/create             ───►  flows/<project>.json  (made_by: "app")
  ├─ POST /api/run                    ───►  starts  flow.py flows/<project>.json  ───►  outputs/<project>/work/...
  └─ GET  /api/flow?name= (every 3 s) ◄───  every node's state, pictures, model
        the steps tick off, the newest model shows in 3D
```

A project on the page **is** a flow file. The page writes the same kind of file as `flows/girl_crown.json`:

- only an **object**: input → data sheet → base model
- an object and a **part**: the same, then edit → inpaint shape → judge → details → judge → colours → judge
- an object and **a part from words** (*+ Describe a new part in words*, no picture of the part needed): the same
  chain; the edit node gets the words as its `prompt` and the local picture AI paints the part onto the model's
  picture (see `docs/FLOWS.md`, "A part from words")

The first file added is the object, every further one a part (changed in the developer view). Before anything is
sent to the uni GPU server the page asks.

### The two views (the switch at the top right)

| | Friendly user | Developer user |
|---|---|---|
| Data rows | picture, name, prompt | also file type, size and **object / part** |
| Total workflow | project name, prompt, the result in 3D, the steps in words, Run / Stop | also seed, steps, resolution, runs on, **Open a project** (every flow of `flows/`) |
| The nodes | – | every node **from top to bottom**; a click opens it under its row |
| A node | – | its pictures, the before / after slider, the judge's numbers, its settings (saved into the flow file), *Run up to here*, *Open in Blender* |
| The mask | – | **Edit by hand (mask)** on the edit node: move the part, paint the area red, protect blue (a tablet pen paints wider when pressed harder), or describe the part with a prompt |
| A failed step | "did not work" | the error and the run's log |

Also in the menu (≡): light / dark (sand page with black panels, or the other way round), the older node page, and
the user at the bottom left.

### For other people: accounts

Without anything the page is for this PC only. For others (the plan: on the uni server, for everybody in the uni
network or its VPN) it asks for a login first. Nobody makes an account on the page: they are handed out.

```
python scripts/pipeline/accounts.py add anna                      # makes the account, prints its password once
python scripts/pipeline/accounts.py list                          # remove anna: takes it away, and logs anna out
python scripts/pipeline/flow_page.py --accounts --host 0.0.0.0    # the page for everybody who reaches this computer
```

- A password is made up by `accounts.py` and shown once. A forgotten one: `add` the name again (a new password).
- `accounts.json` (project folder, not in Git, readable only by its owner) keeps per name a salt and the password's
  hash (PBKDF2-SHA256), never the password.
- Without a login the pages go to `/login` (`user_interface_model/login.html`) and every other address answers 401. A login lasts 7
  days, or until the page is started again. 8 wrong passwords in a row: that name waits ten minutes.
- `--host` for other computers is refused without `--accounts`. *Open in Blender* is off with accounts.
- In the menu of the main page: who is logged in; a click logs out.

Still to do before it goes on the uni server (see *Open*): a folder per account, the runs in a row, and the admins'
OK with an address (HTTPS comes from their web server; the page itself speaks plain HTTP).

### Tests

```
python -W ignore::ResourceWarning -m unittest discover tests      # the back end: uploads, flow files, settings, the edit node, what a POST may do, the accounts
```
They run in a temporary folder with any Python 3 (no GPU, no server, no browser), in about five seconds.

### Blender

*Open in Blender* on a node's model opens it in a new Blender window (`POST /api/blender`). There the Inpaint add-on
(`blender/inpaint_addon.py`: selected faces, space boxes, *Find part in box*) works on it as before. The add-on's own
3D mask is not on the web page: the page's mask is the 2D one of the edit node.

### The back end's addresses for the main page

| Address | What it does |
|---|---|
| `GET /app` | the page |
| `POST /api/upload?flow=<project>&role=object\|part&name=<file>` | the body is the file; pictures and videos only, at most 2 GB; saved under `pictures/uploads/` |
| `POST /api/app/create` | `{"flow", "prompt", "items": [{"path", "role", "prompt"}], "part_prompt", "settings": {"seed", "steps", "resolution", "where"}}` → the flow file. `part_prompt` = the part in words (then no item is a part). It never overwrites a flow that was not made on the page |
| `GET /file/<path>.glb` | a model out of `outputs/`, for the 3D view |

Everything else (`/api/flow`, `/api/run`, `/api/stop`, `/api/settings`, `/api/edit`, `/api/prompt`, `/api/blender`,
`/api/server`) is shared with the older page and described at the top of `flow_page.py`. A POST is only taken from
the page itself (the header `X-Flow-Page`, the page's own address).

### Open

- **The prompts:** a data row's prompt is what that thing is called (the first one of the object, the first one of
  the part): it becomes the data sheet's `what`, which the judge reads instead of the local AI's guess. A part from
  words is the edit node's `prompt`. The prompt for the whole result is only kept as the project's description
  (`"prompt"`): no node reads it yet.
- **Accounts are only the door.** Everybody who is logged in sees and changes the same projects (`flows/`,
  `outputs/`), and two people's runs start at the same time. Next: a folder per account, then one row of runs with
  "you are number 3" on the page, then limits (uploads, runs per day).
- **3D models as input** (.glb) are not taken: the flow starts from pictures or a video.
- **Windows:** the pipeline's environment is a Linux one (`install.sh`). On Windows the page and the back end start
  (with any Python 3), but a run stops at the first node that needs the environment. Results made on Linux are not
  listed as done on Windows either: a result folder's name comes from the files' full paths.
- **The 3D view** (model-viewer, open source) and the fonts come from the internet; without it the page shows the
  result's picture and the system's fonts.

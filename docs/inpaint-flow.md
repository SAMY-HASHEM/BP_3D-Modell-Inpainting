# Inpainting flow: a new part onto a model, as nodes

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

## The structure

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
`outputs/pipeline/<name>/flow.html` (open it in a browser; it refreshes by itself).

## The nodes

| Node | Runs on | Script | What it does | Result |
|---|---|---|---|---|
| **00 Input** | – | – | pictures (1..n), videos, or both, of one thing; files or folders. One input for the object, one for the part | the file list |
| **01 Data sheet** | this PC | `pipeline/data_sheet.py` | a sheet with several views is cut into single views (measured); a video gives frames of its turn; per view measured: size, sharpness, grey or colour, cut off, symmetry; the local AI (Qwen3.6-27B, Ollama) says what it shows and which side; rules pick the views | `sheet.json`, `views/`, `contact.jpg` |
| **02 Base model** | uni server (or this PC) | `server/cloud.py generate --wait` | TRELLIS.2 from the sheet's `base` views, the most frontal first | the model `.glb` |
| **02b Pick seed** | this PC | `pipeline/pick_seed.py` | measured: the seed whose outline fits the views best (8 sides, mirrored too) | `pick.json` |
| **03 Edit** | this PC | `inpainting/make_edit.py` | where the part goes, nothing placed by hand (on the middle of the top; *Fit*: **between** things that stand up beside the middle, e.g. hair buns, or **over** them with *Replace*: what the part covers is regenerated too): per side BEFORE (render), AFTER (collage with the part), 2D MASK; the 3D area with a job per small cube: keep / delete / edit | `edit.json`, `three.jpg`, `area_map.jpg`, `area.npy` |
| **04 Inpaint** | uni server | `inpainting/send_edit.py --wait` | one step: **shape** → **details** → **colours**, each from the result before; several seeds = several candidates in one server job | the candidates `.glb` |
| **05 Judge** | this PC | `pipeline/edit_judge.py` | measured: is the rest kept, is the part there, the colours; the local AI lists defects; the best candidate goes on, with pass / redo | `judge.json`, `compare.jpg` |

### The data sheet's lists (what TRELLIS.2 gets)
- `base`: the views a model is generated from. Main view = the most symmetric of the views the AI calls *front*; then
  one view per other side (three-quarter, side, back), the sharpest of each. Two views of the same side are not both used.
- `sides`: a part's four sides in turning order. From a video: its frames at 0, 90, 180, 270 degrees of the turn.

### The judge's numbers
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

## Run it
```
source activate.sh
python scripts/pipeline/flow.py flows/girl_crown.json                  # everything; done nodes are skipped
python scripts/pipeline/flow.py flows/girl_crown.json --upto edit      # only up to a node
python scripts/pipeline/flow.py flows/girl_crown.json --structure      # only refresh flow.html
```
The server needs no new code for this: the nodes send the same jobs as before (`generate`, inpainting job zips) and
wait for them (`cloud.py ... --wait <folder>`, `cloud.py wait <job> <folder>`).

A new flow = a copy of `flows/girl_crown.json` with other files. Results: `outputs/pipeline/<name>/<node>/<hash of the
settings>/`; the same settings are never computed twice.

## What published work measures (Nov 10)
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

## Open (to decide together)
- **The loop:** a `redo` verdict does nothing yet. Next: redo with other seeds / a wider or lower area / `--replace`,
  at most N rounds, keeping only what scores better (as the grid loop does).
- **Placement:** the Edit node puts the part on the top of the object. Other places (a part on the chest, in a hand)
  need the place from the local AI or SAM 3.1 (a 2D box → the 3D area).
- **The judge's limits and weights**, and whether the AI's score should be able to veto.
- **ComfyUI:** the nodes are only in the flow runner and in Blender so far.

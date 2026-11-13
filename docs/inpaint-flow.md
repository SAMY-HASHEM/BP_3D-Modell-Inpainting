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

A third place is **the web page**: `python scripts/pipeline/flow_page.py --open` (this PC only,
http://127.0.0.1:8777, the page itself is `web/index.html`). It shows every flow of `flows/` as its node chain and
works on the same flow files and result folders as `flow.py`:

| On the page | What it does |
|---|---|
| a click on a node | its pictures, settings, folder and model; *Open in Blender* opens the model in a new Blender window |
| **before / after slider** | the edit node: before against the plan; a judge: the original (or the plan) against every candidate, from the same four sides, with the edit's area in red and the judge's numbers |
| the settings of a node | changed in place and saved into the flow file (*Save settings*) |
| **Run up to here**, *Run the whole flow*, *Stop the run* | starts `flow.py` for this flow (its output: `outputs/pipeline/<name>/run.log`); before it uses the uni server it asks |
| *Uni server* | the server jobs of this flow's nodes and how far they are (`cloud.py status <job> ...`); *All jobs* reads every job in the cloud, which takes over a minute |
| **Edit by hand** (edit node) | see below |

**Edit by hand** replaces the measured place of the part where it does not fit (e.g. a crown that floats):

1. *Move the part* on the main side: drag it, pull the white dot to size it. The other three sides follow from its
   place in 3D (`make_edit.py --place`).
2. *Paint the area* (red: this changes too) or *Protect* (blue: this stays) on any side: corrections of the 2D mask
   (`make_edit.py --hand`, kept in `flows/hand/<flow>_<node>/add_<view>.png` and `erase_<view>.png`).
3. Or *describe the new part*: the local picture AI (Qwen-Image-Edit through ComfyUI, `prompt_picture.py`, about half
   a minute) paints the prompt into the red area of the main side. Every picture made stays in a row to choose
   from; the chosen one is the AFTER picture then (`make_edit.py --after`), no part picture is pasted.
4. *Apply* writes it into the flow file (the edit node's `place`, `side`, `hand`, `after`) and makes the edit again
   (about half a minute on this PC). The inpainting steps after it then wait for a new run. *Back to the measured
   place* takes it out again; old results stay on disk and are found again.

While painting: *Undo* / *Redo* (Ctrl+Z, Ctrl+Shift+Z; the last 15 strokes per side), `[` and `]` change the brush's
width, **B** paint, **E** protect, **V** move the part, and *Fill what I circle* (**F**): a line drawn around an area
fills all of it, so a large area needs one stroke.

## A part from words (no picture of the part)

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

## A part with its own picture, at any place (`at`)

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

### The edit's plan: a job for every small cube (changed Nov 12)
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
- **Placement:** with a picture of the part the Edit node measures only one place, the top of the object. Any other
  place: `at` (in words, the picture AI places the part's own picture), or *Edit by hand*. A part from words is placed
  by the picture AI, anywhere on the side that is seen.
- **Only one side has a picture** when the picture AI places the part (words, or `at`): the depth is TRELLIS.2's
  guess. Tried on Nov 12: asking the picture AI for the same part on the side and back views, with the front AFTER
  picture as its second picture. It painted the whole front view again instead. The angle LoRA of
  `views_from_picture.py` on the AFTER picture is the next thing to try.
- **The judge's limits and weights**, and whether the AI's score should be able to veto.
- **ComfyUI:** the nodes are only in the flow runner and in Blender so far.

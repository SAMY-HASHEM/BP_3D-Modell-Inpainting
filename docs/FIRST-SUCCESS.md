# The first structure that worked

*A part of a model replaced by another one, kept by its judge: 16 November 2026.*

The test (the user's own): a toaster robot holds a slice of toast in its hand; the toast is replaced by a book, from one
picture of that book. The result is a solid, closed book held by the fingers; everything else is the robot's own mesh,
untouched.

| | Rest kept (needs 85) | Close to the wanted result (needs 70) | Colour |
|---|---|---|---|
| after the details step (`v02`) | 98 | 97 | 93 |
| after the colours step (`v03`, the one the user chose) | 98 | 93 | 96 |

The files are in `outputs/sad_toaster/` (not in Git: the robot and the book picture belong to other artists):
`v03_book_inpainted_details_colours.glb` and `.jpg`, `v02_book_inpainted_details.glb`, `compare_details_and_colours.jpg`.

## The structure

```
 THE MODEL (.glb)                     THE NEW PART (one picture)
       │                                        │
       ▼                                        ▼
 1 FRONT PICTURE ─────────────────►  2 THE PART'S OWN 3D MODEL
   make_edit.py --prompt                 cloud.py generate          (uni server, about 10 min)
   the picture AI paints the part
   onto the front render; what
   differs is the 2D mask
       │                                        │
       └───────────────┬────────────────────────┘
                       ▼
 3 PLACE IT IN 3D            place_part.py --zoom 0.25 --size 0.8                (this PC, 4 min)
   the old part = the piece of the model the front picture shows in the mask
   the new part in its plane, at the painted size; only a box around the two goes on
   before / after / mask rendered from 4 sides; the 3D area = the two parts
                       │
                       ▼
 4 JUDGE THE PLAN            edit_check.py   (cloud.py edit refuses on redo)     (this PC, 1 min)
                       │ pass
                       ▼
 5 DETAILS                   cloud.py edit <plan> <plan>/placed.glb --details    (uni server, 45 min)
   the placed part's voxels stay; their fine shape and the join with the model are made again
                       │
                       ▼
 6 JUDGE THE RESULT          edit_judge.py --step details                        (this PC, 2 min)
                       │ pass
                       ▼
 7 COLOURS                   cloud.py edit <plan> <result> --colours             (uni server, 22 min)
                       │
                       ▼
 8 BACK INTO THE WHOLE       place_part.py --finish                              (this PC, seconds)
   the model's own mesh without the old part + the new part from the result
```

The one thing that made the difference is step 5. The inpainting has three steps: shape (where there is something, from
new noise), details (the fine shape on fixed voxels) and colours. For this part the shape step is left out: the placed
part gives the voxels.

## Why each step is there

| Step | What went wrong without it (same test, the tries before) |
|---|---|
| 2, 3: the part as a 3D model, placed | With one front picture, the part's depth, sides and back are guessed and the 3D area was a column 18 voxels deep in front of the robot: a thin torn slab and loose pieces |
| 3: four sides | All sides show the same part because it is one object. Asking the picture AI for the other sides painted the front again (Nov 12) |
| 3: the box around the parts (`--zoom`) | In the whole robot the book was 20 x 17 x 2.5 voxels and reached 3 voxels past the model's box, where TRELLIS.2's cube ends: cut off. Zoomed it is 42 x 36 x 5, and pegs in the box's corners make the cube as big as the box |
| 4: the judge before | Two tries went to the server with a plan nobody had measured. It now says redo for: a part's picture outside the 3D area, an area much bigger than the part, one side only, a part cut off at the box, a part under 3 voxels thick |
| 5: details instead of shape | From noise the book came out as shards or a small plate five times (13 to 37 of 100), also zoomed and with eight sides |
| 6: `like` | The old number only said "something changed": a brown plate where the toast was got 0, a torn slab 80. `like` measures how close a candidate is to the wanted pictures |
| 8: back into the whole | The result of a zoomed inpainting is only the box. The rest of the model is never sent through TRELLIS.2 at all, so it stays exact |

## The commands of that night

```
source activate.sh
M=outputs/sad_toaster/work/sad_toaster_one_mesh.glb      # the model as ONE mesh (see "Not yet" below)
W=outputs/sad_toaster/work

# 1 the front picture and its 2D mask (side 1 is the robot's front; hand/erase_1.png: the leg the picture AI changed by the way)
python scripts/inpainting/make_edit.py $M pictures/book/felipe-v-livro-1.jpg --side 1 --replace --behind 8 --hand $W/hand \
    --prompt "the closed dark red book into the robot's hand in place of the slice of toast it holds, so that the hand holds the book upright and the toast in the hand is gone" \
    --out $W/edit_book2

# 2 the book's own model
python scripts/server/cloud.py generate $W/book_model/book.jpg --resolution 1024 --seed 42 --wait $W/book_model

# 3 placed, zoomed, four sides
python scripts/inpainting/place_part.py $M $W/book_model/picture0_1024_seed42.glb --edit $W/edit_book2 --out $W/zoom_b --zoom 0.25 --size 0.8

# 4 the plan judged (cloud.py edit asks it too)
python scripts/pipeline/edit_check.py $W/zoom_b --part "the slice of toast in the robot's hand is replaced by a closed dark book"

# 5, 6 details on the placed part, judged
python scripts/server/cloud.py edit $W/zoom_b $W/zoom_b/placed.glb --details --seeds 1 --wait $W/details_zoom_b
python scripts/pipeline/edit_judge.py --edit $W/zoom_b --original $W/zoom_b/before.glb --candidates $W/details_zoom_b/*.glb \
    --out $W/judge_details_zoom_b --step details --part "a closed dark book held in the hand"

# 7 colours
python scripts/server/cloud.py edit $W/zoom_b $W/details_zoom_b/*seed1.glb --colours --seeds 1 --wait $W/colours_zoom_b

# 8 back into the whole robot
python scripts/inpainting/place_part.py --finish $W/colours_zoom_b/*.glb --out $W/zoom_b --final outputs/sad_toaster/v03_book_inpainted_details_colours.glb
```

The settings of steps 5 and 7 are `cloud.py edit`'s own: inversion with key-value injection, the area from new noise,
soft edge 2, colour guidance 3, 25 steps, four sides mixed (multidiffusion).

## All tries of the test

| Try | What went to the server | Kept | Like | What came out |
|---|---|---|---|---|
| 1 | one front picture, the area down to the floor | 81* | – | thin torn slab, the knife and butter on the floor changed |
| 2 | the same, the area ending above the floor | 81* | – | slab with a corner missing |
| 3 | the book placed in 3D, 4 sides, the whole robot, shape from noise | 97 | 21, 37 | small brown plate |
| 4 A | zoomed (box 1.30), 4 sides, shape from noise | 97 | 22, 15 | small plate |
| 4 B | zoomed (box 0.78), the book 0.8, 4 sides, shape from noise | 97, 98 | 22, 28 | the right size and place, in shards |
| 4 C | zoomed (box 1.30), 8 sides, shape from noise | 97 | 13 | small plate |
| **5** | **plan B, the placed book as the structure, details only** | **98** | **97** | **a solid book** |
| 5 + colours | the same, then the colours step | 98 | 93 | the same book, greyer |

\* 81 was a fault of the comparison, not of the method: the model saved again by trimesh shines differently than its
file did, and one was compared with the other. Saved the same way, the rest is kept at 97 to 98.

## Not yet

- **One object.** The structure is proven on this robot only; the next test decides whether it holds.
- **The model as one mesh is made by hand.** The robot's file has two meshes (the second a slightly bigger copy of the
  first); the one-mesh file was written with a few lines of Python, not by a script.
- **A model made of pieces.** The old part was one of the robot's 52 pieces. For one fused mesh (what TRELLIS.2 makes)
  the old part is the faces under the mask, `--behind` voxels deep: written, not tried.
- **The steps are run by hand.** There is no flow node for them yet (`flows/`, `nodes_core.py`).
- **The local vision AI** (Qwen3.6-27B) gave 10 of 10 to every plan, good or bad. Only the measured part of the judge
  before the inpainting told them apart.
- **Top views** are not rendered, only the four straight sides and the four in between.
- **A part that grows out of the model** (a crown on a head) has no model of its own to place. For it the shape step
  from noise stays the way (the crown and the wheel, October).
- **Time:** about 85 minutes of server time for one change, and the server took 20 to 140 minutes per job that night.

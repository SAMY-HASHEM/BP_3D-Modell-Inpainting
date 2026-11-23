# How to run it

Where the AI runs and what every setting does. How the whole project fits together: [STRUCTURE.md](STRUCTURE.md).

- [1. This PC](#1-this-pc)
- [2. The uni server from anywhere](#2-the-uni-server-from-anywhere)
- [3. On the uni server itself](#3-on-the-uni-server-itself)
- [4. Settings](#4-settings)
- [5. Rigging a model](#5-rigging-a-model)

## 1. This PC

*How to: inpainting on **this PC** (your own GPU)*

Use this when your computer has a big NVIDIA GPU (e.g. an RTX 3090, 24 GB). Everything stays on your PC:
no uni server, no cloud. The add-on starts and stops the server by itself.

For the uni GPU server see [HOW-TO.md](#2-the-uni-server-from-anywhere).

### A computer without a GPU: everything on the uni server

A weak laptop only sends the data and shows the result: the whole flow (the data sheet, the base model, the edit, the
inpainting, the judges) runs on the server, as one job.

```
THE LAPTOP (any Python 3, no GPU)          SCIEBO CLOUD                    THE UNI SERVER
─────────────────────────────────          ────────────                    ──────────────
flow_page.py --open  →  /app                                               cloud_worker.py (keeps running)
  data + words → the plan → Run  ── <job>.zip (flow + files) ──────────►   run_flow.py: flow.py, every node here
  the city, the chat           ◄── <job>__live_state.json (every 15 s) ──  each node's state, note, minutes
  pictures, the last model     ◄── <job>__live_<file> ─────────────────
  Stop                          ── <job>__stop ─────────────────────────►  ends the flow
```

On the laptop, once: get the project (`git clone`, or *BP Setup* in Blender), then `python3 scripts/server/cloud.py setup`
(the Sciebo login). Every time: `python3 scripts/pipeline/flow_page.py --open`, and in the plan leave *Where the work
runs* on **everything on the uni server**. The page's server needs no AI libraries.

On the server, once after this was added (Nov 16): `git pull`, start the worker again, and see what it has:

```
python3 scripts/server/cloud.py check        # from any computer: Blender? the vision AI? the picture AI? ...
bash install_server_extras.sh                # on the server: what was missing (Blender, Ollama, UniRig, the mask AI)
```

What a flow needs there besides TRELLIS.2: **Blender** (every picture of a model: the edit and the judges; without it
only a base model can be made), a **vision AI** (Ollama with qwen3.6:27b; without it the data sheet takes every
picture as a front view and the judges only measure), and for a part from words the **picture AI** (ComfyUI with
Qwen-Image-Edit; not installed by the script). Tried on Nov 16 with this PC standing in for the server, on a cloud
folder of its own: a small project went out, ran, came back with its pictures and model, and Stop ended it. On the
real server it was not run yet. A test worker on another folder: `INPAINTING_CLOUD_FOLDER=inpainting-jobs-test`.

Understanding a request on a laptop: without the local AI the page uses simple rules (the first data is the model,
what comes after it the new part); the plan can always be changed by hand.

### Once: setup

1. The project: `git clone https://git.uni-due.de/students/bp_samy_hashem.git ~/inpainting-3D`, then
   `cd ~/inpainting-3D && ./install.sh` (TRELLIS.2 and its AI models; takes a while).
2. The add-on in Blender: Edit → Preferences → Add-ons → arrow at the top right → *Install from Disk…* →
   `blender/inpaint_addon.py` → tick it. In its preferences, **Project folder** = `~/inpainting-3D`.
3. **Only if Blender is the Flatpak** (installed from Flathub / the software centre): its sandbox can't see
   the NVIDIA GPU, so the add-on starts the server outside it. Allow that once in a terminal, then restart Blender:

   ```
   flatpak override --user --talk-name=org.freedesktop.Flatpak org.blender.Blender
   ```

   (This lets Blender start programs outside its sandbox. Undo it with
   `flatpak override --user --reset org.blender.Blender`.)

### Every time

1. Inpaint tab (N key) → *Where does the AI run?* → **This PC** → **Start server on this PC**.
   The status shows the start; after about a minute it says **ready**.
2. Mark the area, choose the picture, **2. Regenerate area**, as in the other guides.
3. Done: **Stop** (■) frees the GPU. Closing Blender stops it too.

### If something goes wrong

| You see | Fix |
|---|---|
| `No project in …` | set *Project folder* in the add-on preferences |
| `Blender may not start it (Flatpak)` | step 3 of the setup, then restart Blender |
| `didn't start: see …/local_server.log` | the log says why (e.g. `./install.sh` not done, GPU memory full) |

## 2. The uni server from anywhere

*How to: using the uni GPU server*

One way, the same at home and at uni: **through Sciebo**. The HCI GPU server can't be reached from outside, but your
computer and the server can both reach the **Sciebo cloud**, so the cloud folder `inpainting-jobs/` is the mailbox:
your computer puts a job there, a worker on the server runs it and puts the result back. The server is not your main
computer: all heavy work runs there.

```
YOUR PC                              SCIEBO CLOUD                     HCI GPU SERVER (A40, 48 GB)
───────                              ────────────                     ──────────────────────────
Blender + add-on                     inpainting-jobs/                 cloud_worker.py (keeps running)
  AI runs: "Uni server"      
  "2. Regenerate area" ── job.zip ─► <job>.zip          ◄── checks ── every 30 s
  panel: progress      ◄─ status ──  <job>.status       ◄── writes ── runs it on the freest GPU
  result appears       ◄─ model ───  <job>__<file>.glb  ◄── uploads ─ done
```

Each job takes 2-3 minutes for uploading, the worker's 30 s look and loading TRELLIS.2.

### A new computer (also a uni PC with Windows): only Blender needed

`pictures/` and `outputs/` are not in Git (the pictures belong to other artists, models are too big), so they
come from your private Sciebo. All of it is buttons in Blender; no terminal, no Git, no Python.

1. Download one file:
   https://raw.githubusercontent.com/SAMY-HASHEM/BP_3D-Modell-Inpainting/main/blender/pipeline_nodes.py
2. Blender: Edit → Preferences → Add-ons → arrow at the top right → *Install from Disk…* → that file → tick it.
3. 3D view → N key → tab **BP** → *BP Setup*: a checklist. Click **Set up everything**: it gets the project and
   installs the Inpaint add-on.
4. Enter your Sciebo login name and app password → **Save and test** (Sciebo: Einstellungen → Sicherheit →
   App-Passwort; never in a screenshot).
5. **Get pictures**.

Later: **Update project** in the same box (it replaces `git pull`, gets the project and the new Inpaint add-on), then
restart Blender. If only the add-on is wanted, download `blender/inpaint_addon.py` from GitHub and *Install from Disk…* it.

| Tab | Gives you |
|---|---|
| **Inpaint** | inpainting a marked area (below) |
| **BP** → *BP Server* | a picture folder through the whole pipeline; fetch (*Auto 10 min*); open models and heat maps |

The add-on's *This PC* mode and running the nodes yourself need the Linux setup; on Windows the work runs on the
server. *Find part in box* needs the mask AI on your PC, so it doesn't work on a uni PC.

---

### Once: setup

**1. Sciebo account:** https://hochschulcloud.nrw → *Registrierung* → Universität Duisburg-Essen →
log in with your uni account. Your cloud: https://uni-duisburg-essen.sciebo.de

**2. Two app passwords** (Sciebo: avatar → Persönliche Einstellungen → Sicherheit →
*Neues App-Passwort erstellen*): one called `inpainting-pc`, one called `inpainting-server`.
Copy them with the 📋 buttons straight into the fields below — **never as a screenshot**, never into a chat.

**3. Your PC, in Blender:** Edit → Preferences → Add-ons → TRELLIS.2 Inpainting → section
*Uni server*:

| Field | Value |
|---|---|
| Cloud address | leave as is |
| Login name | `sosahash@uni-duisburg-essen.de` (shown with the app password) |
| App password | the `inpainting-pc` password |
| Job folder | `inpainting-jobs` |

→ **Save and test the cloud login**. It checks the login, saves it in
`~/.config/inpainting3d/cloud.json` (readable only by you) and empties the password field.
(Without Blender: `python3 scripts/server/cloud.py setup` does the same.)

**4. The server** (JupyterLab at https://nc.hcigroup.de → File → New → Terminal; on campus directly in the
browser, from home through the VDI):

```
cd ~/projects/bp_samy_hashem && git pull
python3 scripts/server/cloud.py setup                   # the inpainting-server app password
source activate.sh
nohup python -u scripts/server/cloud_worker.py > outputs/cloud_worker.log 2>&1 &
tail -n 5 outputs/cloud_worker.log               # "Watching the cloud folder inpainting-jobs/ every 30 s"
```

The worker keeps running after you close the VDI.

---

### Every time

**In Blender** (Inpaint tab, N key):

1. Import the model, mark the area: faces in Edit Mode and/or **Add space box** (blue, see-through;
   drag an arrow to move one side, a dot to pull one corner, G moves it, R turns it).
   **Only one part in the box** (e.g. the helmet, not the hair under it): type its name under
   *Find a part in the box* (`helmet`), optionally **Click on it** (several spots: Enter when done), then **Find part in box**. About
   15 s later its faces are selected and shown red (Tab to fix them by hand), and **Box** is set to
   *Selected faces + empty space*: the part and the empty space in the box are regenerated, every other
   surface in the box stays exactly (no colour bleeding onto it). Needs `./install_mask.sh` once on this PC
   (a mask AI, about 4 GB VRAM). Box choices: *Everything in the box* / *Only the selected faces* /
   *Selected faces + empty space* / *Only the empty space* (the default: the model stays exact, the new part only grows into the empty room) / *Replace what the new part covers* (the old surface goes where the part covers it from every side, e.g. a helmet in place of hair).
2. Turn the view to show the area. Choose a **Reference**: a picture of the new part alone (one view, plain
   background). Press **1. Make picture**. It renders the whole model and puts the part into your box, no AI:
   the picture **before**, **after** and the **2D mask**. This is how the crown came onto the bee. A picture of only
   the new part must not go in as the Picture: TRELLIS.2 reads it as the whole object.
   Make the box **wider than what the part sits on** and let its floor reach a little into the model: the part stands on the
   box's floor. A box smaller than the head gave a crown that floats.
   **More sides** (optional, best colours): a folder with up to 3 pictures of the part in turning order (right, back, left).
3. Always Inversion, 25 steps. For something on top of the model set **Room above** (about 0.15).
   **Step**, always in this order, each from the result of the one before: **1 Shape** → **2 Details** (the part stays
   where it is, it only gets sharper) → **3 Colours** (only the colours, on the same mesh, about 10 minutes). Press
   **1. Make picture** again before each step, so the picture before shows the newest model.
   At the top of the panel, *Where does the AI run?*: **Uni server** → **2. Regenerate area**.
4. The panel shows the progress (`queued` → `running: GPU 2: …` → `done`); the result is downloaded
   and appears next to the hidden original.

**Without Blender**, one line makes the three pictures, with the side, size and place measured:

```
python scripts/inpainting/make_edit.py model.glb crown.png --reference robot.png --out outputs/edits/robot_crown
```

`area_map.jpg` shows every small cube of the area: **green keep** (surface and inside stay exact), **red delete** (old
surface the part covers from every side, only with `--replace`), **blue edit** (empty room for the new part).
A video of the part turning, or a folder of its sides, as the part gives all four sides. Then the three steps:

```
python scripts/server/cloud.py edit outputs/edits/robot_crown model.glb --seeds 1 2   # 1 the shape
python scripts/server/cloud.py edit outputs/edits/robot_crown shape.glb --details     # 2 sharper details
python scripts/server/cloud.py edit outputs/edits/robot_crown details.glb --colours   # 3 the colours, last
```

**Other commands** (a terminal on your PC, in `~/inpainting-3D`):

```
python3 scripts/server/cloud.py generate front.png back.png --resolution 1024 --steps 25 --mirror   # picture(s) -> model
python3 scripts/server/cloud.py send outputs/blender/<name>_job.zip                                 # a saved Blender job
python3 scripts/server/cloud.py status                  # every job: queued / running / done / failed
python3 scripts/server/cloud.py fetch                   # finished models -> outputs/cloud/
```

**The whole pipeline** (pictures or a video → data sheet → base model → judge → re-inpaint bad blocks; one
job at a time, it uses one GPU):

```
python3 scripts/server/cloud.py pipeline "pictures/robot fly"   # send one character's folder
python3 scripts/server/cloud.py watch                           # every 10 minutes: new models, heat maps, zips
python3 scripts/server/cloud.py pictures up                     # new pictures to Sciebo (down: back to this PC)
```

Results land in `outputs/server_runs/<job>/`, while the job runs. The same in Blender: add-on
`blender/pipeline_nodes.py`, 3D view → N key → tab **BP** → *BP Server* (send, fetch, *Auto 10 min*, open).

**Stop a running job** (VDI → JupyterLab terminal; the worker keeps running):

```
pkill -u $USER -f auto_pipeline.py
pkill -u $USER -f refine_region.py
```

---

### If something goes wrong

| You see | Meaning | Fix |
|---|---|---|
| `No cloud login yet` | step 3 wasn't done on this PC | add-on preferences → *Save and test the cloud login* |
| `Wrong login name or app password` | typo, or the app password was revoked | make a new app password in Sciebo |
| stays `queued` for minutes | the worker on the server isn't running | VDI → JupyterLab: start it (step 4) |
| `failed: …` | the job itself failed | the message says why; details: `tail -n 30 outputs/cloud_worker.log` on the server |

**Is the worker running?** On the server: `ps -u $USER -o pid,etime,args | grep cloud_worker`.
**After a `git pull`** that changed `cloud_worker.py`, restart it:

```
pkill -u $USER -f cloud_worker.py
source activate.sh && nohup python -u scripts/server/cloud_worker.py > outputs/cloud_worker.log 2>&1 &
```

### New app password for the server (old one deleted or blocked)

1. Sciebo → avatar (top right) → *Persönliche Einstellungen* → *Sicherheit*.
   Delete the old one with the trash icon. ("Marked for remote wipe" or "Could not delete" is fine: it's blocked anyway.)
2. *App name*: a new name, e.g. `inpainting-server-2` → *Create new app password*. Keep the page open.
3. On the server, one line at a time:

```
cd ~/projects/bp_samy_hashem
git pull
python3 scripts/server/cloud.py setup                      # asks only: login name, app password
ps -u $USER -f | grep [c]loud_worker                       # workers still running?
pkill -u $USER -f [c]loud_worker                           # stop them all (an old one still uses the old password)
source activate.sh
nohup python -u scripts/server/cloud_worker.py > outputs/cloud_worker.log 2>&1 &
tail -n 5 outputs/cloud_worker.log                         # after 1 minute: no "401 Unauthorized"
```

Setup says "Works" but the log shows `401 Unauthorized` → stop all workers and start one again (an old worker, or Sciebo
needed a moment for the new password; a restart fixed it on Nov 4).
Never put the app password in a screenshot or a chat.

### Good to know

- **Which GPU?** The worker takes the GPU with the least memory in use, for every job anew.
- **New code without logging in:** `cloud.py generate … --code main` (or another branch) makes the worker
  fetch that branch's newest code from GitHub before the job, and restart itself if the worker changed.
- **Safety:** the worker only runs our own scripts with checked settings, never commands from the
  cloud. An app password only opens your Sciebo; revoke it any time in Sciebo → Sicherheit.
- **1. Make picture** with a Reference works everywhere. Only painting from a *prompt* needs ComfyUI on your own PC.
- The cloud folder keeps jobs and results; delete old ones in Sciebo from time to time.

## 3. On the uni server itself

*Working on the HCI GPU server*

The HCI group's server ("Numbercruncher") has 3 × NVIDIA A40 with 48 GB each — twice the memory of
an RTX 3090. It is only reachable from inside the uni network, so from home the way in is the uni VDI.

### 1. Getting there

1. Start the **Omnissa Horizon Client** → server `extern.vdi.uni-due.de`
2. Log in with the Unikennung, password and 2FA code → double-click **UDE Student:Innen**
3. In that Windows desktop, open the browser → **https://nc.hcigroup.de** (JupyterLab)
4. Terminal: in JupyterLab **File → New → Terminal**

### 2. Every time you start

```
cd ~/projects/bp_samy_hashem    # go to the project
source activate.sh              # switch the TRELLIS.2 environment on
git pull                        # get the newest code
```

### 3. Look at the GPUs first

```
nvtop                           # live view of the 3 GPUs (quit: q)
nvidia-smi                      # the same, once
```

Take a GPU that nobody uses. `run_job.py` does that by itself; for other scripts say which one:

```
export CUDA_VISIBLE_DEVICES=0   # use GPU 0 from now on in this terminal
```

### 4. Long jobs: let them run on their own

A job started normally stops when the VDI or the browser closes. With `nohup ... &` it keeps running
on the server, and its output goes into a log file:

```
nohup python scripts/generation/generate.py pictures/tavern.jpg --resolution 1536 > outputs/tavern.log 2>&1 &
tail -f outputs/tavern.log      # watch it (Ctrl+C only stops watching, not the job)
```

Is something of mine still running? Stop it?

```
ps -u $USER -o pid,etime,args | grep python    # your running Python jobs (number, time, command)
kill 12345                                     # stop the job with that number
```

### 5. Inpainting from Blender (job package)

1. Blender (own PC), Inpaint tab: mark the area, choose or make the Picture → **Save job for server**.
   This writes one file: `outputs/blender/<model>_job.zip`.
2. JupyterLab: open `outputs/uploads/` (make it once with `mkdir -p outputs/uploads`), upload the zip
   with the ⬆ button.
3. Terminal:
   ```
   nohup python scripts/server/run_job.py outputs/uploads/tavern_job.zip --seeds 1 2 3 > outputs/uploads/run.log 2>&1 &
   tail -f outputs/uploads/run.log
   ```
   `--seeds 1 2 3` makes three versions in one go; without it, the seed from Blender is used.
4. Download the results (`outputs/uploads/tavern_job_seed1.glb`, …): right-click → **Download**.
5. Blender: **Load result** → pick the file.

### 6. Moving files

- **Up:** the ⬆ button above JupyterLab's file list.
- **Down:** right-click a file → **Download**.
- Faster: in the Horizon Client turn on **Drive Sharing** for your project folder; the VDI then sees
  your PC's files directly.

### 7. Privacy

The server belongs to the HCI group: the admins can see everything on it, other users can see your
running commands (in `nvtop`, `ps`). Only project work goes there.

```
chmod 700 ~                                                   # other users can't open your files
rm -rf ~/.local/share/Trash/files/* ~/.local/share/Trash/info/*   # empty JupyterLab's trash
```

### 8. Blender inpainting with the server

Two ways, each with its own step-by-step guide:

| Where the AI runs | Guide | In Blender, *Where does the AI run?* |
|---|---|---|
| your own PC (its NVIDIA GPU) | [docs/HOW-TO.md](#1-this-pc) | This PC |
| anywhere (home or uni) | [docs/HOW-TO.md](#2-the-uni-server-from-anywhere) | Uni server |

## 4. Settings

*What every setting does, and where it lives in the code*

Line numbers are from today's code (`TRELLIS.2/` is Microsoft's code, `scripts/` is ours). TRELLIS.2's
default values are in its `pipeline.json` (in `local_ai_models/models--microsoft--TRELLIS.2-4B/snapshots/…/`).

### The way of one picture through TRELLIS.2

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

### `--resolution` (512 / 1024 / 1536): how fine the 3D grid is

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

### `--steps` (default 12): how many small steps from noise to result

- **Ours:** `generate.py:104-112`: the same number of steps for all three stages.
- **TRELLIS.2:** `samplers/flow_euler.py:115-118` makes the list of time points from 1 (pure noise) to 0
  (finished). `rescale_t` (`:116`, 5.0 / 3.0 in `pipeline.json`) puts more of them where the shape is
  decided. Each step moves the result a little along the predicted direction: `:80`
  `x_prev = x_t - (t - t_prev) * v`.
- **More = ** smaller, more careful steps: cleaner, less "rushed" surfaces, but slower (time grows with steps).
- **Our test (valkyrie):** 12 = flat panels and broken feathers, **25 = cleanest**, 50 = most natural body
  but noisier, twice as slow.

### `--seed` (default 42): which random noise to start from

- **TRELLIS.2:** `trellis2_image_to_3d.py:538` `torch.manual_seed(seed)`: every stage's starting noise comes
  from it.
- **Same seed + same picture + same settings = the same model.** Another seed = another variant (a different
  guess for what the picture doesn't show).

### `--guidance` (default 7.5): how strictly the shape follows the picture

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

### `--texture-guidance` (default 1.0 = off): the same for the colours

- Stage 3 (`pipeline.json`: guidance 1.0, interval `[0.6, 0.9]`). 2-4 makes the colours follow the picture
  more closely.

### `--max-tokens` (default 49152): a limit for the fine shape step

- **TRELLIS.2:** `trellis2_image_to_3d.py:287` and `:327-339`. A *token* is one cell of the fine grid that
  the model works on. If there are more than the limit, the resolution is lowered by 128 (1536 → 1408 →
  1280 …) until they fit, but never below 1024 (`:335`). Fewer tokens = less memory when decoding.
- **Only matters for big objects:** the valkyrie needs fewer than 20,000 at 1536 (the limit never changed
  anything), a head with long hair needs more (40,000 lowered it to 1152 and so it fit into 24 GB).

### `--faces` (default 1,000,000): how many triangles the saved model keeps

- **Ours:** `generate.py:124-135`. TRELLIS.2 makes far more triangles (11 million at 1536), the file keeps
  this many. At 1024 the surface is first **remeshed** (rebuilt as an even surface from the grid, `remesh`
  in `generate.py:127-129` and o_voxel's `to_glb`); at 1536 that doesn't fit into 24 GB, so the mesh is
  only simplified.
- **More = ** finer edges kept, bigger files (4 million ≈ 150 MB). It can't add detail TRELLIS.2 didn't make.

### Several pictures: `multi_image.py --mode multidiffusion | stochastic`

- **Ours:** `multi_image.py`, class `MultiImageFlowModel`: every step, the model predicts once per picture;
  *multidiffusion* averages these predictions, *stochastic* uses one picture per step in turn. No camera
  angles are needed: TRELLIS.2 places every object the same way, whatever the view.
- `scripts/generation/views_from_video.py` takes such pictures from a turntable video.

### Recommended

| Object | Command |
|---|---|
| Most objects | `--resolution 1024 --steps 25 --faces 4000000` |
| Small, very detailed objects | `--resolution 1536 --steps 25 --faces 4000000` |
| Long hair / very big objects at 1536 | add `--max-tokens 40000` |
| Turntable video | `views_from_video.py video.mp4 --cutout`, then `multi_image.py pictures/<video>/view_*.png …` |

## 5. Rigging a model

*How to: give a generated humanoid a skeleton and skin weights, so it can be posed and animated*

The rig head (`scripts/rigging/rig_head.py`) is trained once (below); after that a model takes about 4 minutes.
It finds 22 body bones (hips to head, arms, legs, toes) and 30 finger bones; the face has no bones yet.

### In one run: picture → rigged model

```bash
python scripts/generation/generate.py picture.png --rig      # TRELLIS.2: the skeleton from the generation's own latent
python scripts/generation/compare_pixal3d.py picture.png --name x --out outputs/x --rig   # Pixal3D (model read in again)
```

TRELLIS.2 at 1024: the rig head reads the latent the generation made (cleaner than reading the finished model in
again). At 1536 the latent is 96³, not the rig head's 64³: the model is read in again, by itself.

### Every time: one model

```bash
source activate.sh
python scripts/rigging/rig_from_glb.py outputs/<object>/model.glb --out outputs/rigged/<name>
blender -b --python scripts/rigging/wave_demo.py -- outputs/rigged/<name>_rigged.glb outputs/rigged/<name>_wave.blend
```

| File | What it is |
|---|---|
| `<name>_rigged.glb` | the model with its armature, ready to pose or animate (any program that reads glTF) |
| `<name>_pose.png` | at rest and posed side by side: does the skin follow the bones? |
| `<name>.jpg` | the skeleton on the latent's voxels, coloured by bone, from the front and the side |
| `<name>_wave.blend` | open it in Blender and press **Space**: the model waves (bones shown as sticks) |

Needs: a GPU (the rig head and TRELLIS.2's shape encoder, ~4 GB), TRELLIS.2 installed (`./install.sh`),
Blender 4.x, and the trained rig head in `local_ai_models/rig_head/best.pt` (not in Git, 58 MB; it is in Sciebo, or
train it). Blender not on the PATH: `export BLENDER=/path/to/blender`. The model must stand upright and face the front,
as TRELLIS.2 and Pixal3D make it (glTF: Y up, front +Z).

Settings: `RIG_SMOOTH=2` (default) smooths the weights along the surface, about 2 voxels far: neighbouring points
follow the same bones, so sleeves and cloaks bend instead of wrinkling. `0` turns it off (the shading around a moved
arm gets dark and streaky), `1` is between.

**A model that does not stand in a T or A pose** (hands on the hips, arms against the body, a statue): add
`--joints views`. The rig head learnt from characters in a T or A pose and gets such arms wrong (Mai, Nov 22: her
elbows outside her body). Then the joints are found on 12 pictures of the model by GEM-X's 77-point finder (needs
`install_gemx.sh`, below), and the weights come from the bones each point wraps around; a hand sculpted into a hip is
cut free (a flat palm and a patch on the hip). About 3 minutes more. Cloth and long hair still only follow the body:
a flap at the hips swings up when the body lies down.

**Posing it yourself in Blender:** open the `_wave.blend` (it opens in Pose Mode), stop the animation (Space), click a
bone, press **R**, move the mouse, click. Alt+R resets a bone; A then Alt+R resets all.

### A person's motion from a video onto the rigged model

GEM-X (NVIDIA) reads the person in a video as 77 joints in 3D (body, fingers, face), frame by frame; the rigged model's
bones are turned so that they point the same way, frame by frame. The model gets one skin colour, so its moves are easy
to see (`--keep-colours`: its own colours).

```bash
bash scripts/rigging/install_gemx.sh       # once: its own environment conda/envs/gemx, 7 GB of weights, ~15 min
python3 scripts/rigging/motion_from_video.py video.mp4 outputs/rigged/<name>_rigged.glb --out outputs/motion/<name> \
    --start 120 --duration 240              # only minutes 2:00 to 6:00 (in seconds); without them: the whole video
# the same motion onto another model, without tracking again; --keep-colours: its own colours (a textured model)
python3 scripts/rigging/motion_from_video.py video.mp4 outputs/rigged/<other>_rigged.glb --out outputs/motion/<other> \
    --joints outputs/motion/<name>/joints.npz --keep-colours
```

| File | What it is |
|---|---|
| `animated.blend` | open it in Blender and press **Space**: the model moves like the person (bones as sticks, a floor, the camera) |
| `animated.glb` | the same for any program that reads glTF |
| `side_by_side.mp4` | the video on the left, the model on the right, seen from where the video's camera stood |
| `joints.npz` | the tracked joints (`motion_track.py`); `part_<k>/motion.pt`: GEM-X's own results |

How long: GEM-X takes about 10 minutes per 50 seconds of video on the RTX 3090 (finding the person runs on the
processor); putting the motion on the model and the pictures about 1 minute per 10 seconds.

Good to know:
- One person, the one seen the most. Best with a camera that does not move (`--camera background` if it turns).
- GEM-X takes at most 60 s at a time: longer clips are tracked in parts of 50 s that overlap by 2 s and are joined.
- GEM-X's own world drifts (on the stretching video 2 to 3 m sideways within 50 s, on a yoga mat), and its idea of
  "up" leans by up to 9 degrees for a whole part (in plank the toes were 20 cm above the floor). With a camera that
  stands still `motion_track.py` takes the body as the camera saw it, turns it upright once for the whole video (the
  median "up" of all frames), and puts the hands, feet and knees that rest on the floor onto it, frame by frame.
- The model is not the person: its arms and legs have other lengths, so a hand can end a little above or in the floor.

### A character with a detailed face (Pixal3D on the uni server, 35 GB GPU)

```bash
G=grey_views; C=colour_views; O=outputs/<name>
python scripts/generation/compare_pixal3d_mv.py $G/front.png:0 $G/back.png:180 --colour $C/front.png $C/back.png --name body --out $O
python scripts/generation/paint_front.py $O/body.glb --picture $O/body_colour_input_rgba.png --camera $O/body.json \
    --closeup closeup.jpg --closeup-box x0 y0 x1 y1 --out $O/body_front.glb
python scripts/generation/paint_front.py $O/body_front.glb --picture $O/body_colour_views/view01_azim+180.png \
    --camera $O/body.json --view-angle 180 --out $O/body_painted.glb
python scripts/generation/compare_pixal3d.py head_cut_out.png --resolution 1536 --name head --out $O   # the head alone
python scripts/generation/paint_front.py $O/head.glb --picture $O/head_input_rgba.png --camera $O/head.json --out $O/head_painted.glb
python scripts/generation/swap_head.py $O/body_painted.glb --body-camera $O/body.json --body-picture $O/body_colour_input_rgba.png \
    $O/head_painted.glb --head-camera $O/head.json --head-picture $O/head_input_rgba.png --head-source head_cut_out.png \
    --closeup closeup.jpg --closeup-box x0 y0 x1 y1 --head-crop hx hy --out $O/whole.glb
python scripts/rigging/rig_from_glb.py $O/whole.glb --out $O/rigged
```

The head's cut-out: the close-up from the hair to the collar, background away (RMBG-2.0), the halo and other things
at its edges taken out. Only front and back views: the 3/4 views of a sheet have angles nobody wrote down.

### An item in the hand

In one go, from pictures (character: its views, front first; the item: one picture), ~15 min:

```bash
python scripts/pipeline/armed_character.py front.png side.png back.png --item axe.png --out outputs/knight_axe --size 0.5
```

Or the hold alone, onto a rigged character:

```bash
blender -b --python scripts/rigging/hold_item.py -- outputs/x/rigged_rigged.blend axe.glb outputs/x/with_axe --size 0.5 --swing
```

The item (a .glb, e.g. from generate.py) goes into the right hand (--hand l: the left), its length --size of the
character's height, the grip --grip of its length from the handle's end; the fingers close around it. --swing adds
a raise-and-chop animation. Writes <out>.blend (press Space), <out>.glb, close-ups of the grip (<out>_hand_*.png, cut
away so a sleeve does not hide the hand) and the whole body.

It prints a grip check from the posed mesh itself (each finger's skin: its gap to the handle, how much is inside it)
as PASS / FAIL per finger, also in <out>_grip.json. Believe this, not the bones' numbers.

The hands must be fitted to the mesh first (every rig made since Nov 23 is; an older one, or a rig from elsewhere):

```bash
python scripts/rigging/fit_hands.py old_rigged.blend fitted.blend [--glb]
```

### Once: the training data and the training

```bash
# Rig-XL (UniRig's data, huggingface.co/VAST-AI/UniRig, data/rigxl/processed.7z, 4.5 GB) unpacked into datasets/rigxl
python scripts/rigging/rig_data.py datasets/rigxl --lists mixamo_train vroid_train --out datasets/rig_latents  # ~12 h
python scripts/rigging/rig_data.py datasets/rigxl --lists mixamo_train vroid_train --out datasets/rig_latents --retry-skipped
python scripts/rigging/rig_head.py train --data datasets/rig_latents --out local_ai_models/rig_head --steps 20000  # ~4.5 h
```

On the uni server (A40): 8,750 rigs usable (958 have no hands, a head or a whole skeleton), training 4 h 50 min.
Result on 253 rigs it never saw: 85.6 % of the voxels get the right bone; body joints ~2.7 cm and finger joints
~2.2 cm off (for a person of 1.70 m). The log: `local_ai_models/rig_head/log.txt`.


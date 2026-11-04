# How to: using the uni GPU server

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

## A new computer (also a uni PC with Windows): only Blender needed

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

Later: **Update project** in the same box (it replaces `git pull`).

| Tab | Gives you |
|---|---|
| **Inpaint** | inpainting a marked area (below) |
| **BP** → *BP Server* | a picture folder through the whole pipeline; fetch (*Auto 10 min*); open models and heat maps |

The add-on's *This PC* mode and running the nodes yourself need the Linux setup; on Windows the work runs on the
server. *Find part in box* needs the mask AI on your PC, so it doesn't work on a uni PC.

---

## Once: setup

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

## Every time

**In Blender** (Inpaint tab, N key):

1. Import the model, mark the area: faces in Edit Mode and/or **Add space box** (blue, see-through;
   drag an arrow to move one side, a dot to pull one corner, G moves it, R turns it).
   **Only one part in the box** (e.g. the helmet, not the hair under it): type its name under
   *Find a part in the box* (`helmet`), optionally **Click on it** (several spots: Enter when done), then **Find part in box**. About
   15 s later its faces are selected and shown red (Tab to fix them by hand), and **Box** is set to
   *Selected faces + empty space*: the part and the empty space in the box are regenerated, every other
   surface in the box stays exactly (no colour bleeding onto it). Needs `./install_mask.sh` once on this PC
   (a mask AI, about 4 GB VRAM). Box choices: *Everything in the box* / *Only the selected faces* /
   *Selected faces + empty space*.
2. Turn the view to show the area. Choose a **Reference**: a picture of the new part alone (one view, plain
   background). Press **1. Make picture**. It renders the whole model and puts the part into your box, no AI:
   the picture **before**, **after** and the **2D mask**. This is how the crown came onto the bee. A picture of only
   the new part must not go in as the Picture: TRELLIS.2 reads it as the whole object.
   **More sides** (optional, best colours): a folder with up to 3 pictures of the part in turning order (right, back, left).
3. Always Inversion, 25 steps. For something on top of the model set **Room above** (about 0.15).
   **Colours only**: the shape stays, only the colours in the area change, about 8 minutes. Best as the last step:
   tick it, **1. Make picture** again, **2. Regenerate area**.
   At the top of the panel, *Where does the AI run?*: **Uni server** → **2. Regenerate area**.
4. The panel shows the progress (`queued` → `running: GPU 2: …` → `done`); the result is downloaded
   and appears next to the hidden original.

**Without Blender**, one line makes the three pictures, with the side, size and place measured:

```
python scripts/inpainting/make_edit.py model.glb crown.png --reference robot.png --out outputs/edits/robot_crown
```

`area_map.jpg` shows every small cube of the area: **green keep** (surface that stays exact), **red delete** (old surface
under the mask, only with `--replace`), **blue edit** (empty room for the new part). Then send it:

```
E=outputs/edits/robot_crown
python scripts/server/cloud.py inpaint model.glb $E/after.png --area $E/area.npy --source $E/before.png \
    --picture-mask $E/mask.png --grow 0 --steps 25 --variant inversion,room=0.15,start=random,soft=2,texture_guidance=3
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

## If something goes wrong

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

## New app password for the server (old one deleted or blocked)

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

## Good to know

- **Which GPU?** The worker takes the GPU with the least memory in use, for every job anew.
- **New code without logging in:** `cloud.py generate … --code main` (or another branch) makes the worker
  fetch that branch's newest code from GitHub before the job, and restart itself if the worker changed.
- **Safety:** the worker only runs our own scripts with checked settings, never commands from the
  cloud. An app password only opens your Sciebo; revoke it any time in Sciebo → Sicherheit.
- **1. Make picture** with a Reference works everywhere. Only painting from a *prompt* needs ComfyUI on your own PC.
- The cloud folder keeps jobs and results; delete old ones in Sciebo from time to time.

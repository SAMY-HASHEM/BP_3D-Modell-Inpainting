# How to: inpainting from Blender **outside the uni network** (from home)

Use this when your PC is **not** in the uni network (at home, anywhere). The HCI GPU server can't be
reached from outside, but both your PC and the server can reach the **Sciebo cloud**. So the cloud
folder `inpainting-jobs/` is the mailbox: Blender puts a job there, a worker on the server runs it and
puts the result back.

In the uni network use [how-to-inside-network.md](how-to-inside-network.md) instead (faster).

```
YOUR PC                              SCIEBO CLOUD                     HCI GPU SERVER (A40, 48 GB)
───────                              ────────────                     ──────────────────────────
Blender + add-on                     inpainting-jobs/                 cloud_worker.py (keeps running)
  Connection: "Outside (cloud)"
  "2. Regenerate area" ── job.zip ─► <job>.zip          ◄── checks ── every 30 s
  panel: progress      ◄─ status ──  <job>.status       ◄── writes ── runs it on the freest GPU
  result appears       ◄─ model ───  <job>__<file>.glb  ◄── uploads ─ done
```

Each job takes 2-3 minutes longer than inside the network: uploading, the worker's 30 s look, and
loading TRELLIS.2 for every job.

---

## Once: setup

**1. Sciebo account:** https://hochschulcloud.nrw → *Registrierung* → Universität Duisburg-Essen →
log in with your uni account. Your cloud: https://uni-duisburg-essen.sciebo.de

**2. Two app passwords** (Sciebo: avatar → Persönliche Einstellungen → Sicherheit →
*Neues App-Passwort erstellen*): one called `inpainting-pc`, one called `inpainting-server`.
Copy them with the 📋 buttons straight into the fields below — **never as a screenshot**, never into a chat.

**3. Your PC, in Blender:** Edit → Preferences → Add-ons → TRELLIS.2 Inpainting → section
*Outside the network*:

| Field | Value |
|---|---|
| Cloud address | leave as is |
| Login name | `sosahash@uni-duisburg-essen.de` (shown with the app password) |
| App password | the `inpainting-pc` password |
| Job folder | `inpainting-jobs` |

→ **Save and test the cloud login**. It checks the login, saves it in
`~/.config/inpainting3d/cloud.json` (readable only by you) and empties the password field.
(Without Blender: `python3 scripts/cloud.py setup` does the same.)

**4. The server** (VDI → browser → nc.hcigroup.de → JupyterLab → File → New → Terminal):

```
cd ~/projects/bp_samy_hashem && git pull
python3 scripts/cloud.py setup                   # the inpainting-server app password
source activate.sh
nohup python -u scripts/cloud_worker.py > outputs/cloud_worker.log 2>&1 &
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
   15 s later its faces are selected and shown red (Tab to fix them by hand) and *Only the selected faces
   in the box* is ticked. Needs `./install_mask.sh` once on this PC (a mask AI, about 4 GB VRAM).
2. **Picture**: the model's picture.
3. Connection: **Outside (cloud)** → **2. Regenerate area**.
4. The panel shows the progress (`queued` → `running: GPU 2: …` → `done`); the result is downloaded
   and appears next to the hidden original.

**Without Blender** (a terminal on your PC, in `~/inpainting-3D`):

```
python3 scripts/cloud.py generate front.png back.png --resolution 1024 --steps 25 --mirror   # picture(s) -> model
python3 scripts/cloud.py send outputs/blender/<name>_job.zip                                 # a saved Blender job
python3 scripts/cloud.py status                  # every job: queued / running / done / failed
python3 scripts/cloud.py fetch                   # finished models -> outputs/cloud/
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
source activate.sh && nohup python -u scripts/cloud_worker.py > outputs/cloud_worker.log 2>&1 &
```

## Good to know

- **Which GPU?** The worker takes the GPU with the least memory in use, for every job anew.
- **New code without logging in:** `cloud.py generate … --code main` (or another branch) makes the worker
  fetch that branch's newest code from GitHub before the job, and restart itself if the worker changed.
- **Safety:** the worker only runs our own scripts with checked settings, never commands from the
  cloud. An app password only opens your Sciebo; revoke it any time in Sciebo → Sicherheit.
- **Make picture from prompt** only works with a server on your own PC (it needs ComfyUI there).
- The cloud folder keeps jobs and results; delete old ones in Sciebo from time to time.

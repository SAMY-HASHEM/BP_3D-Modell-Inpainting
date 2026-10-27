# Working on the HCI GPU server

The HCI group's server ("Numbercruncher") has 3 × NVIDIA A40 with 48 GB each — twice the memory of
an RTX 3090. It is only reachable from inside the uni network, so from home the way in is the uni VDI.

## 1. Getting there

1. Start the **Omnissa Horizon Client** → server `extern.vdi.uni-due.de`
2. Log in with the Unikennung, password and 2FA code → double-click **UDE Student:Innen**
3. In that Windows desktop, open the browser → **https://nc.hcigroup.de** (JupyterLab)
4. Terminal: in JupyterLab **File → New → Terminal**

## 2. Every time you start

```
cd ~/projects/bp_samy_hashem    # go to the project
source activate.sh              # switch the TRELLIS.2 environment on
git pull                        # get the newest code
```

## 3. Look at the GPUs first

```
nvtop                           # live view of the 3 GPUs (quit: q)
nvidia-smi                      # the same, once
```

Take a GPU that nobody uses. `run_job.py` does that by itself; for other scripts say which one:

```
export CUDA_VISIBLE_DEVICES=0   # use GPU 0 from now on in this terminal
```

## 4. Long jobs: let them run on their own

A job started normally stops when the VDI or the browser closes. With `nohup ... &` it keeps running
on the server, and its output goes into a log file:

```
nohup python scripts/generate.py pictures/tavern.jpg --resolution 1536 > outputs/tavern.log 2>&1 &
tail -f outputs/tavern.log      # watch it (Ctrl+C only stops watching, not the job)
```

Is something of mine still running? Stop it?

```
ps -u $USER -o pid,etime,args | grep python    # your running Python jobs (number, time, command)
kill 12345                                     # stop the job with that number
```

## 5. Inpainting from Blender (job package)

1. Blender (own PC), Inpaint tab: mark the area, choose or make the Picture → **Save job for server**.
   This writes one file: `outputs/blender/<model>_job.zip`.
2. JupyterLab: open `outputs/uploads/` (make it once with `mkdir -p outputs/uploads`), upload the zip
   with the ⬆ button.
3. Terminal:
   ```
   nohup python scripts/run_job.py outputs/uploads/tavern_job.zip --seeds 1 2 3 > outputs/uploads/run.log 2>&1 &
   tail -f outputs/uploads/run.log
   ```
   `--seeds 1 2 3` makes three versions in one go; without it, the seed from Blender is used.
4. Download the results (`outputs/uploads/tavern_job_seed1.glb`, …): right-click → **Download**.
5. Blender: **Load result** → pick the file.

## 6. Moving files

- **Up:** the ⬆ button above JupyterLab's file list.
- **Down:** right-click a file → **Download**.
- Faster: in the Horizon Client turn on **Drive Sharing** for your project folder; the VDI then sees
  your PC's files directly.

## 7. Privacy

The server belongs to the HCI group: the admins can see everything on it, other users can see your
running commands (in `nvtop`, `ps`). Only project work goes there.

```
chmod 700 ~                                                   # other users can't open your files
rm -rf ~/.local/share/Trash/files/* ~/.local/share/Trash/info/*   # empty JupyterLab's trash
```

## 8. Outside the uni network (from home): jobs through the Sciebo cloud

The server can't be reached from outside, but it and every PC can reach the Sciebo cloud. A worker
on the server watches the cloud folder `inpainting-jobs/` and runs every job that arrives there.

```
PC (Blender / cloud.py)          Sciebo cloud                    HCI server (A40)
 "Send to uni server"  ──job──►  inpainting-jobs/  ◄──watches──  cloud_worker.py
 panel shows progress  ◄status─  <job>.status      ◄──writes───  runs it on the freest GPU
 result loads by itself ◄─glb──  <job>__<file>.glb ◄─uploads───  done
```

**Once on the server** (VDI → JupyterLab terminal):

```
cd ~/projects/bp_samy_hashem && git pull
python3 scripts/cloud.py setup                     # Sciebo app password "inpainting-server"
source activate.sh
nohup python -u scripts/cloud_worker.py > outputs/cloud_worker.log 2>&1 &
tail -n 5 outputs/cloud_worker.log                 # "Watching the cloud folder ..."
```

**On a PC**, Blender: Edit → Preferences → Add-ons → TRELLIS.2 Inpainting → enter the Sciebo login
(app password) → *Save and test the cloud login*. Then the Inpaint tab's **Send to uni server** works.
Without Blender (`python3 scripts/cloud.py setup` once):

```
python3 scripts/cloud.py generate front.png back.png --resolution 1024 --steps 25 --mirror   # pictures -> model
python3 scripts/cloud.py send outputs/blender/tavern_job.zip                                 # a Blender job
python3 scripts/cloud.py status                    # queued / running (GPU, progress) / done / failed
python3 scripts/cloud.py fetch                     # results -> outputs/cloud/
```

Is the worker still running? `ps -u $USER -o pid,etime,args | grep cloud_worker`. Restart it after a
`git pull` that changed `cloud_worker.py`.

## 9. Inside the uni network: Blender talks directly to the server

On campus (or for anyone in the uni network) JupyterLab opens without the VDI, and the server's port
can be reached directly. This is the original add-on way, just with the server on the A40:

```
your PC (Blender)                                   HCI server (A40)
 browser: nc.hcigroup.de (JupyterLab)  ──────────►  terminal: inpaint_server.py --host 0.0.0.0
 Blender, Connection "Inside the network" ──job──►  regenerates on the freest GPU (TRELLIS.2 stays loaded)
 result loads by itself                ◄─result──
```

**On the server** (JupyterLab terminal; leave it open while you work):

```
cd ~/projects/bp_samy_hashem && source activate.sh
python scripts/inpaint_server.py --host 0.0.0.0              # another port if 8765 is taken: --port 8770
```

It prints the GPU it uses and a **token**. Everyone uses their own account and their own server.

**In Blender** (once): Edit → Preferences → Add-ons → TRELLIS.2 Inpainting → *Server address*
`nc.hcigroup.de:8765` and the *Server token*. Then in the Inpaint tab: Connection **Inside the
network** → **2. Regenerate area**.

Safety: every request needs the token; jobs that pass file paths (and *Make picture from prompt*) only
work on a server running for its own computer (`127.0.0.1`, the default).

| Where you are | Connection in Blender | What runs on the server |
|---|---|---|
| in the uni network | Inside the network | `inpaint_server.py --host 0.0.0.0` (while you work) |
| anywhere else | Outside (cloud) | `cloud_worker.py` (keeps running) |

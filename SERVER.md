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

## 8. From home without the VDI: jobs through the Sciebo cloud

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

**Someone else** (e.g. the professor): share the Sciebo folder `inpainting-jobs` with them (with edit
rights). They install the add-on, enter *their own* Sciebo app password and, as job folder, the path
under which the shared folder appears in their Sciebo. Their jobs then run on the same worker.

Is the worker still running? `ps -u $USER -o pid,etime,args | grep cloud_worker`. Restart it after a
`git pull` that changed `cloud_worker.py`.

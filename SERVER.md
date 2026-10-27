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

## 8. Blender inpainting with the server

Two ways, each with its own step-by-step guide:

| Where your PC is | Guide | In Blender, Connection |
|---|---|---|
| in the uni network | [docs/how-to-inside-network.md](docs/how-to-inside-network.md) | Inside the network |
| anywhere else (home) | [docs/how-to-outside-network.md](docs/how-to-outside-network.md) | Outside (cloud) |

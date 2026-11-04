# How to: working **inside the uni network** (a uni PC, Windows)

Use this when your PC is **in the uni network** (on campus, uni WLAN, a uni PC). No VDI and no Sciebo are
needed for the server: JupyterLab opens directly in the browser at https://nc.hcigroup.de, and Blender on
your PC talks **directly** to the inpaint server on the HCI GPU server. It's the fastest way: TRELLIS.2
stays loaded, so every job starts right away.

The server is not your main PC: all heavy work (TRELLIS.2, the vision AI) runs there, a uni PC only needs
Blender, Git and Python. The uni PCs run Windows: there `python` replaces `python3`, and paths use `\`.

From home use [how-to-outside-network.md](how-to-outside-network.md) instead.

```
YOUR PC                                           HCI GPU SERVER (A40, 48 GB)
───────                                           ──────────────────────────
browser: nc.hcigroup.de (JupyterLab)  ──────────► terminal: inpaint_server.py --host 0.0.0.0
Blender + add-on                                  (keeps TRELLIS.2 loaded, picks the freest GPU)
  AI runs: "Uni (inside)"
  "2. Regenerate area"   ──── job (.zip) ───────► regenerates the marked area
  result appears in Blender ◄── model (.glb) ───  done
```

Every request carries the server's **token** (a password the server makes once); without it the
server answers nothing. Everyone uses their own server account and their own server.

---

## A new PC (e.g. a uni PC with Windows): only Blender needed

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

## The whole pipeline directly on the server

In a JupyterLab terminal, no Sciebo needed. Pictures: drag them into JupyterLab's file list, or
`python3 scripts/server/cloud.py pictures down` on the server.

```
cd ~/projects/bp_samy_hashem && source activate.sh
nohup python -u scripts/pipeline/auto_pipeline.py "pictures/monster" --name monster > outputs/monster.log 2>&1 &
tail -f outputs/monster.log       # Ctrl+C only stops watching
```

Results: `outputs/pipeline/monster/auto/` on the server (right-click → Download in JupyterLab; big models
are safer through *BP Server*). Only one job at a time: while the worker runs a job, don't start a second
one, and don't `git pull` on the server (the worker fetches its code itself).

---

## Once: setup for inpainting from Blender

**1. The project on the server** (JupyterLab → File → New → Terminal):

```
cd ~/projects/bp_samy_hashem && git pull
```

(First time ever: `git clone https://git.uni-due.de/students/bp_samy_hashem.git ~/projects/bp_samy_hashem`
and `bash install.sh` there.)

**2. The add-on in Blender on your PC:** see "A new PC" above.

**3. Start the server once to get the token** (see "Every time" below). It prints:

```
Ready on GPU 2: waiting for Blender on port 8765 of this computer, from the network (...)
Token for Blender (add-on preferences): <the token>
```

**4. Enter it in Blender:** Edit → Preferences → Add-ons → TRELLIS.2 Inpainting → section
*Inside the uni network*:

| Field | Value |
|---|---|
| Server address | `nc.hcigroup.de:8765` |
| Server token | the token from step 3 |

The token stays the same at every start (it is kept in `~/.config/inpainting3d/server_token` on the
server, readable only by you), so this is needed only once.

---

## Every time

**1. Start the server** (JupyterLab terminal; leave it open while you work):

```
cd ~/projects/bp_samy_hashem && source activate.sh
python scripts/server/inpaint_server.py --host 0.0.0.0
```

Wait about a minute until it says **Ready**.

**2. In Blender** (Inpaint tab, N key):

1. Import the model, mark the area: faces in Edit Mode and/or **Add space box** (blue, see-through;
   drag an arrow to move one side, a dot to pull one corner, G moves it, R turns it).
   **Only one part in the box** (e.g. the helmet, not the hair under it): type its name under
   *Find a part in the box* (`helmet`), optionally **Click on it** (several spots: Enter when done), then **Find part in box**. About
   15 s later its faces are selected and shown red (Tab to fix them by hand), and **Box** is set to
   *Selected faces + empty space*: the part and the empty space in the box are regenerated, every other
   surface in the box stays exactly (no colour bleeding onto it). Needs `./install_mask.sh` once on this PC
   (a mask AI, about 4 GB VRAM). Box choices: *Everything in the box* / *Only the selected faces* /
   *Selected faces + empty space*.
2. **Picture**: the model's picture, or one made with *Make picture from prompt* (it keeps the render
   before the edit and the edit's 2D mask, which **Method: Inversion** uses).
3. **Method**: *Repaint* (fast) or *Inversion* (the new part joins better, about 3× slower; 25 steps for
   the best quality). At the top of the panel, *Where does the AI run?*: **Uni (inside)** → **2. Regenerate area**.
4. The panel shows the progress (1/6 … 6/6); the result appears next to the hidden original.

The ↻ button next to *Server* checks the connection: **ready**, **busy** or what is wrong.

**3. When you're done: stop the server** with **Ctrl + C** in its terminal. It holds GPU memory that
others in the group may need.

---

## If something goes wrong

| Blender shows | Meaning | Fix |
|---|---|---|
| `not running` | the server isn't started, the address is wrong, or your PC isn't in the uni network | start it (step 1), check the address, check the network (from home: cloud way) |
| `wrong or missing token` | the token in Blender doesn't match | copy it again from the server's start message |
| `busy` | another job is still running on your server | wait until it's done |
| Server: `Address already in use` | someone else's server uses port 8765 | start with `--port 8770` and enter `nc.hcigroup.de:8770` in Blender |
| `failed: …` | the job itself failed | the server terminal shows the details |

## Good to know

- **Which GPU?** The server takes the GPU with the least memory in use. `--gpu 1` chooses one.
- **Make picture from prompt** only works with a server on your own PC (it needs ComfyUI there).
- **Safety:** without the token nobody can use your server; jobs that pass file paths only work on a
  server that runs for its own computer (`127.0.0.1`, the default without `--host`).
- **A new token:** delete `~/.config/inpainting3d/server_token` on the server, start it again and enter
  the new token in Blender.

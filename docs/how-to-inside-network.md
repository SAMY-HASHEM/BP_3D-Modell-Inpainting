# How to: inpainting from Blender **inside the uni network**

Use this when your PC is **in the uni network** (on campus, uni WLAN, a uni PC). Blender on your PC
talks **directly** to the inpaint server on the HCI GPU server. It's the fastest way: TRELLIS.2 stays
loaded, so every job starts right away.

From home use [how-to-outside-network.md](how-to-outside-network.md) instead.

```
YOUR PC                                           HCI GPU SERVER (A40, 48 GB)
───────                                           ──────────────────────────
browser: nc.hcigroup.de (JupyterLab)  ──────────► terminal: inpaint_server.py --host 0.0.0.0
Blender + add-on                                  (keeps TRELLIS.2 loaded, picks the freest GPU)
  Connection: "Inside the network"
  "2. Regenerate area"   ──── job (.zip) ───────► regenerates the marked area
  result appears in Blender ◄── model (.glb) ───  done
```

Every request carries the server's **token** (a password the server makes once); without it the
server answers nothing. Everyone uses their own server account and their own server.

---

## Once: setup

**1. The project on the server** (JupyterLab → File → New → Terminal):

```
cd ~/projects/bp_samy_hashem && git pull
```

(First time ever: `git clone https://git.uni-due.de/students/bp_samy_hashem.git ~/projects/bp_samy_hashem`
and `bash install.sh` there.)

**2. The add-on in Blender on your PC:** Edit → Preferences → Add-ons → arrow at the top right →
*Install from Disk…* → `blender/inpaint_addon.py` → tick it. Version 1.5 or newer.

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
python scripts/inpaint_server.py --host 0.0.0.0
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
   the best quality). Connection: **Inside the network** → **2. Regenerate area**.
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

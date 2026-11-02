# How to: inpainting on **this PC** (your own GPU)

Use this when your computer has a big NVIDIA GPU (e.g. an RTX 3090, 24 GB). Everything stays on your PC:
no uni server, no cloud. The add-on starts and stops the server by itself.

For the uni GPU server see [how-to-inside-network.md](how-to-inside-network.md) (on campus) or
[how-to-outside-network.md](how-to-outside-network.md) (from anywhere).

## Once: setup

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

## Every time

1. Inpaint tab (N key) → *Where does the AI run?* → **This PC** → **Start server on this PC**.
   The status shows the start; after about a minute it says **ready**.
2. Mark the area, choose the picture, **2. Regenerate area**, as in the other guides.
3. Done: **Stop** (■) frees the GPU. Closing Blender stops it too.

## If something goes wrong

| You see | Fix |
|---|---|
| `No project in …` | set *Project folder* in the add-on preferences |
| `Blender may not start it (Flatpak)` | step 3 of the setup, then restart Blender |
| `didn't start: see …/local_server.log` | the log says why (e.g. `./install.sh` not done, GPU memory full) |

"""A character with a weapon (or a tool) in its hand, from pictures, in one go:

  1 the character   TRELLIS.2 from its views (multi_image.py: several views; generate.py: one), 1024
  2 the item        TRELLIS.2 from its picture (generate.py)
  3 the rig         rig_from_glb.py (the rig head, twist bones, preserve volume)
  4 the hold        hold_item.py: the item in the hand, the fingers closed, a raise-and-chop
  5 the video       render_video.py

    python scripts/pipeline/armed_character.py front.png side.png back.png --item axe.png --out outputs/knight_axe \\
        [--size 0.5] [--hand r]

The pictures with their background (it is taken away) or cut out (RGBA). Writes into --out: character.glb, item.glb,
rig_rigged.blend, held.blend (press Space), held.glb, held_swing.mp4 and pictures. A step whose result is there is
skipped (to redo one, delete its file). ~15 min on the uni server.
"""
import argparse
import os
import subprocess
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
BLENDER = os.environ.get("BLENDER") or next(
    (b for b in (os.path.join(PROJECT, "blender", "blender"),) if os.path.exists(b)), "blender")
S = lambda *p: os.path.join(PROJECT, "scripts", *p)


def run(step, command, result):
    if os.path.exists(result):
        print(f"=== {step}: already there ({result})", flush=True)
        return
    print(f"=== {step}", flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("views", nargs="+", help="the character's pictures, front first")
    parser.add_argument("--item", required=True, help="the item's picture")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", default="0.45", help="the item's length in the character's heights")
    parser.add_argument("--hand", default="r", choices=["r", "l"])
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    o = lambda name: os.path.join(out, name)
    views = [os.path.abspath(v) for v in args.views]
    if len(views) > 1:
        run("character (TRELLIS.2, %d views)" % len(views),
            [PY, S("generation", "multi_image.py"), *views, "--out", o("character.glb")], o("character.glb"))
    else:
        run("character (TRELLIS.2)", [PY, S("generation", "generate.py"), views[0], "--out", o("character.glb")],
            o("character.glb"))
    run("item (TRELLIS.2)", [PY, S("generation", "generate.py"), os.path.abspath(args.item), "--out", o("item.glb")],
        o("item.glb"))
    run("rig", [PY, S("rigging", "rig_from_glb.py"), o("character.glb"), "--out", o("rig")], o("rig_rigged.blend"))
    run("hold", [BLENDER, "-b", "--python", S("rigging", "hold_item.py"), "--", o("rig_rigged.blend"), o("item.glb"),
                 o("held"), "--size", args.size, "--hand", args.hand, "--swing"], o("held.blend"))
    run("video", [BLENDER, "-b", "--python", S("rigging", "render_video.py"), "--", o("held.blend"), o("held_swing")],
        o("held_swing.mp4"))
    print("done:", o("held.blend"), o("held_swing.mp4"))


if __name__ == "__main__":
    main()

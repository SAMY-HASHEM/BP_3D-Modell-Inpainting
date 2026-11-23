"""The whole pipeline in one go (docs/PIPELINE.md): pictures or turntable videos -> a finished model, each stage with
the tool that won on Nov 23 (the motorcycle girl, Erik's bust, the Plague Knight):

  3 de-light      delight.py, when grey (clay) views of the same angles are given
  4 shape         multi_image.py (grey views for the shape, de-lit / colour views for the colours), or generate.py
                  for one picture; the resolution from the machine's profile (server 1536, a 24 GB PC 1024);
                  --seeds N: N quick seeds at 1024 scored against the pictures, the best one made in full
  7 colours       describe_colours.py (Qwen3-VL) -> texture_mvadapter.py (MV-Adapter, from the view that shows the
                  most) -> merge_textures.py (the shape stage's faithful colours + MV-Adapter's crisp detail)
  8 detail        detail_normals.py, with --parts (e.g. "skin=noise:0.35:0.012" "cloth=cotton_jersey:0.6:0.03")
  9 rig           rig_from_glb.py (--rig; humanoids), on the dense mesh
  6 retopology    retopo.py (--retopo; closed bodies only); with --rig the rig carried onto the quads (transfer_rig.py)

    python scripts/pipeline/make_model.py --pictures front.png side.png back.png --out outputs/x
    python scripts/pipeline/make_model.py --video painted.mp4 --grey-video clay.mp4 --frames 108 199 58 149 \\
        --out outputs/moto --parts "skin=noise:0.35:0.012" "cloth=cotton_jersey:0.6:0.03" --rig

Front first (the model's front, and the colours' main view). A stage whose result is there is skipped (delete it
to redo it). Writes into --out: views/, shape.glb, colours.glb (merged), detail.glb, rig_rigged.blend, retopo.glb, retopo_rigged.blend,
compare.jpg (every stage's model side by side) and make_model.json (what ran, how long).
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
from hardware import pick, setting  # noqa: E402

PY = sys.executable
MV_PY = os.path.join(PROJECT, "conda", "envs", "mvadapter-venv", "bin", "python")
BLENDER = os.environ.get("BLENDER") or os.path.join(PROJECT, "blender", "blender")
S = lambda *p: os.path.join(PROJECT, "scripts", *p)
LOG = []


def run(stage, command, result, out):
    if os.path.exists(result):
        print(f"=== {stage}: already there ({os.path.basename(result)})", flush=True)
        return
    print(f"=== {stage} {time.strftime('%H:%M')}", flush=True)
    start = time.time()
    with open(os.path.join(out, f"log_{stage.split()[0]}.txt"), "w") as log:
        done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    LOG.append({"stage": stage, "seconds": round(time.time() - start), "ok": done.returncode == 0 and
                os.path.exists(result)})
    if not LOG[-1]["ok"]:
        raise SystemExit(f"{stage} failed: see {out}/log_{stage.split()[0]}.txt")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pictures", nargs="+", help="colour pictures, front first")
    p.add_argument("--grey", nargs="+", help="grey (clay) pictures of the same views, same order")
    p.add_argument("--video", help="a colour turntable video (with --frames)")
    p.add_argument("--grey-video", help="its grey (clay) version, the same turn")
    p.add_argument("--frames", type=int, nargs="+", help="the video frames to use, front first (default: found by "
                   "the outline's width: front, back, sides, two ¾)")
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", help="default: the machine's profile")
    p.add_argument("--parts", nargs="+", help="stage 8: word=map:strength:tile, or auto (Qwen3-VL names the parts "
                   "and their materials); no --parts: no stage 8")
    p.add_argument("--retopo", action="store_true", help="stage 6 (closed bodies)")
    p.add_argument("--rig", action="store_true", help="stage 9 (humanoids)")
    p.add_argument("--no-mvadapter", action="store_true", help="stage 7: the shape stage's own colours only")
    p.add_argument("--seeds", type=int, default=1, help="try this many seeds at 1024 first, scored against the "
                   "pictures (score_model.py); the best one is made at full resolution")
    a = p.parse_args()
    out = os.path.abspath(a.out)
    views = os.path.join(out, "views")
    os.makedirs(views, exist_ok=True)
    o = lambda name: os.path.join(out, name)
    profile = pick()[0]
    resolution = a.resolution or setting("make_model", "resolution", "1024")
    print(f"profile {profile}, resolution {resolution}", flush=True)
    start = time.time()

    # 1-3 views, cut-out, de-light
    colour, grey = [], []
    if a.video and not a.frames:  # found by the outline's width (front, back, sides, two ¾), Qwen3-VL says the front
        found = os.path.join(views, "found")
        run("1 find frames", [PY, S("generation", "views_from_video.py"), a.video, "--auto-frames", "--cutout",
                              "--out", found], os.path.join(found, "frames.json"), out)
        a.frames = json.load(open(os.path.join(found, "frames.json")))["frames"]
        print("frames:", a.frames, flush=True)
    if a.video:
        names = [f"{k:02d}" for k in range(len(a.frames))]
        if a.grey_video:
            run("3 de-light", [PY, S("generation", "delight.py"), "--video", a.video, a.grey_video, "--frames",
                               *map(str, a.frames), "--names", *names, "--out", views],
                os.path.join(views, f"delit_{names[-1]}.png"), out)
            colour = [os.path.join(views, f"delit_{n}.png") for n in names]
            run("1 grey views", [PY, S("generation", "views_from_video.py"), a.grey_video, "--frames",
                                 *map(str, a.frames), "--cutout", "--out", os.path.join(views, "grey")],
                os.path.join(views, "grey"), out)
            grey = sorted(os.path.join(views, "grey", f) for f in os.listdir(os.path.join(views, "grey"))
                          if f.startswith("view_"))
        else:
            run("1 views", [PY, S("generation", "views_from_video.py"), a.video, "--frames", *map(str, a.frames),
                            "--cutout", "--out", views], os.path.join(views, "overview.jpg"), out)
            colour = sorted(os.path.join(views, f) for f in os.listdir(views) if f.startswith("view_"))
    else:
        colour = [os.path.abspath(x) for x in a.pictures]
        if a.grey:
            grey = [os.path.abspath(x) for x in a.grey]
            for k, (c, g) in enumerate(zip(colour, grey)):
                run(f"3 de-light {k}", [PY, S("generation", "delight.py"), c, g, os.path.join(views, f"delit_{k:02d}.png")],
                    os.path.join(views, f"delit_{k:02d}.png"), out)
            colour = [os.path.join(views, f"delit_{k:02d}.png") for k in range(len(colour))]

    # 4 shape (and the faithful colours)
    if grey:
        cmd = [PY, S("generation", "multi_image.py"), "--shape-only", *grey, "--colour-only", *colour]
    elif len(colour) > 1:
        cmd = [PY, S("generation", "multi_image.py"), *colour]
    else:
        cmd = [PY, S("generation", "generate.py"), colour[0]]
    seed = 42
    if a.seeds > 1:  # the seed decides a face (Erik: three seeds, three faces): quick ones at 1024, scored, the best
        tried = []
        for k in range(a.seeds):
            path = o(f"seed_{k}.glb")
            run(f"4 seed {k}", cmd + ["--resolution", "1024", "--seed", str(k), "--out", path], path, out)
            tried.append(path)
        views_for_score = colour[:3]
        angles = ["0", "side", "180"][:len(views_for_score)] if len(views_for_score) > 1 else ["0"]
        run("4 seed scores", [PY, S("tools", "score_model.py"), *tried, "--pictures", *views_for_score, "--views",
                              *angles, "--out", o("seed_scores.json")], o("seed_scores.json"), out)
        scores = json.load(open(o("seed_scores.json")))
        best = min(range(len(scores)), key=lambda i: (scores[i]["mean"]["colour"], -scores[i]["mean"]["outline"]))
        seed = best
        print(f"seed {seed} scored best: {scores[best]['mean']}", flush=True)
    run("4 shape", cmd + ["--resolution", resolution, "--seed", str(seed), "--out", o("shape.glb")], o("shape.glb"),
        out)
    models = [("4 shape", o("shape.glb"))]
    current = o("shape.glb")

    # 7 colours: MV-Adapter from the view that shows the most, its text by Qwen3-VL, merged with the faithful ones
    if not a.no_mvadapter and os.path.exists(MV_PY):
        from PIL import Image
        import numpy as np
        area = [(np.asarray(Image.open(c).convert("RGBA"))[..., 3] > 127).mean() * Image.open(c).size[0] *
                Image.open(c).size[1] for c in colour]
        reference = colour[int(np.argmax(area))]
        run("7 description", [PY, S("generation", "describe_colours.py"), *colour[:4], "--out", o("colours.txt")],
            o("colours.txt"), out)
        text = open(o("colours.txt")).read().strip()
        run("7 mv-adapter", [MV_PY, S("generation", "texture_mvadapter.py"), current, "--picture", reference,
                             "--text", text, "--out", o("mvadapter.glb")], o("mvadapter.glb"), out)
        run("7 merge", [PY, S("generation", "merge_textures.py"), current, o("mvadapter.glb"), "--out",
                        o("colours.glb")], o("colours.glb"), out)
        models += [("7 MV-Adapter", o("mvadapter.glb")), ("7 colours (merged)", o("colours.glb"))]
        current = o("colours.glb")

    # 8 surface detail (--parts auto: the parts and materials named by Qwen3-VL)
    if a.parts == ["auto"]:
        run("8 parts", [PY, S("generation", "describe_colours.py"), *colour[:4], "--parts", "--out", o("parts.txt")],
            o("parts.txt"), out)
        a.parts = [line.strip() for line in open(o("parts.txt")) if line.strip()]
        print("parts:", a.parts, flush=True)
    if a.parts:
        run("8 detail", [PY, S("generation", "detail_normals.py"), current, "--out", o("detail.glb"), "--parts",
                         *a.parts], o("detail.glb"), out)
        models.append(("8 detail", o("detail.glb")))
        current = o("detail.glb")

    # 9 rig on the dense mesh (fit_hands.py needs its fingers), then 6 retopology; with both, the rig is carried
    # onto the quads (transfer_rig.py: rigged directly, the knight's quads lost their fingers, Nov 23)
    if a.rig:
        run("9 rig", [PY, S("rigging", "rig_from_glb.py"), current, "--out", o("rig")], o("rig_rigged.blend"), out)
    if a.retopo:
        run("6 retopology", [BLENDER, "-b", "--python", S("generation", "retopo.py"), "--", current,
                             o("retopo.glb"), "--faces", "40000", "--voxel", "4"], o("retopo.glb"), out)
        models.append(("6 retopology", o("retopo.glb")))
        if a.rig:
            run("9 rig onto the quads", [BLENDER, "-b", "--python", S("rigging", "transfer_rig.py"), "--",
                                         o("rig_rigged.blend"), o("retopo.glb"), o("retopo_rigged.blend")],
                o("retopo_rigged.blend"), out)
        current = o("retopo.glb")

    # every stage's model side by side
    if not os.path.exists(o("compare.jpg")):
        env = {**os.environ, "CLAY_COLOURS": "1", "CLAY_SIZE": "450"}
        subprocess.run([BLENDER, "-b", "--python", S("generation", "clay_views.py"), "--", o("compare.jpg"),
                        *[m for _, m in models]], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([PY, S("generation", "clay_views.py"), o("compare.jpg"), *[n for n, _ in models]], env=env)
    json.dump({"profile": profile, "resolution": resolution, "stages": LOG, "models": models,
               "minutes": round((time.time() - start) / 60, 1)}, open(o("make_model.json"), "w"), indent=1)
    print(f"done in {(time.time() - start) / 60:.0f} min: {current}", flush=True)


if __name__ == "__main__":
    main()

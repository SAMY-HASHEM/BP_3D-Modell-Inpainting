"""The whole pipeline for one character, on its own (e.g. on the uni server): the user's grid loop.

    00 pictures   all pictures of the folder; a video gives evenly spaced frames
    01 sort       the vision AI sorts the views (sort_pictures.py)
    02 base       the whole figure, multidiffusion, every picture at least 1024 px (multi_image.py)
    03 judge      the model cut into blocks (256 of TRELLIS.2's 1024 grid), each scored by the local vision AI
                  against the pictures (grid_judge.py): >= 85 stays, below is bad
    04 re-inpaint every bad block zoomed in (refine_region.py --box3d), the picture crop at least 1024 px:
                  1st try inversion with a soft edge (the user's VoxHammer idea), 2nd try repaint, 3rd try
                  inversion from new noise; only the colours when only the colours are bad
    05 combine    the blocks put back into ONE mesh (combine_parts.py --single) -> the next round's model
    back to 03, at most --rounds rounds or until no block is bad

Every step is its own process (the GPU is free again after it). Results: <out>/<step>/..., summary.json and
summary.md (the scores of every round). With --zip: a small zip for the cloud worker (models, sheets, scores).

Usage (TRELLIS.2 environment):
    python scripts/pipeline/auto_pipeline.py "pictures/robot fly" --name robot_fly
    python scripts/pipeline/auto_pipeline.py pictures/monster --name monster --rounds 3 --threshold 85 --zip
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
PROJECT = os.path.dirname(SCRIPTS)
PICTURE_TYPES = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_TYPES = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v")
METHODS = [  # the tries of a block that stays bad: the user's inversion first
    {"method": "inversion", "start": "inverted", "soft": 2},
    {"method": "repaint"},
    {"method": "inversion", "start": "random", "soft": 2},
]


def run(command, log, done=None):
    """One step as its own process; skipped when `done` exists (so a stopped run can go on)."""
    if done and os.path.exists(done):
        print(f"  (done before) {os.path.relpath(done, PROJECT)}", flush=True)
        return True
    print(f"  $ {' '.join(os.path.relpath(c, PROJECT) if os.path.isabs(str(c)) else str(c) for c in command)}", flush=True)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as f:
        code = subprocess.call([sys.executable, "-u", *map(str, command)], stdout=f, stderr=subprocess.STDOUT, cwd=PROJECT)
    ok = code == 0 and (not done or os.path.exists(done))
    if not ok:
        print(f"  FAILED (see {log}):\n" + open(log, errors="replace").read()[-1200:], flush=True)
    return ok


def script(path):
    return os.path.join(SCRIPTS, *path.split("/"))


def main():
    parser = argparse.ArgumentParser(description="The whole grid-loop pipeline for one character")
    parser.add_argument("pictures", help="the folder with the character's pictures (and/or a video)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", help="default: outputs/pipeline/<name>/auto")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=85)
    parser.add_argument("--blocks", type=int, default=4, help="blocks per side of the cube (4 = 256 of 1024)")
    parser.add_argument("--frames", type=int, default=12, help="frames of a video")
    parser.add_argument("--zip", action="store_true", help="also a small zip of the results (for the cloud worker)")
    args = parser.parse_args()
    out = os.path.abspath(args.out or os.path.join(PROJECT, "outputs", "pipeline", args.name, "auto"))
    os.makedirs(out, exist_ok=True)
    start = time.time()
    summary = {"name": args.name, "threshold": args.threshold, "rounds": []}

    print("00 pictures", flush=True)
    inbox = os.path.join(out, "00_pictures")
    os.makedirs(inbox, exist_ok=True)
    source = os.path.abspath(args.pictures)
    for f in sorted(os.listdir(source)):
        path = os.path.join(source, f)
        if f.lower().endswith(PICTURE_TYPES) and not os.path.lexists(os.path.join(inbox, f)):
            os.symlink(path, os.path.join(inbox, f))
        elif f.lower().endswith(VIDEO_TYPES):
            frames = os.path.join(out, "00_video", os.path.splitext(f)[0])
            run([script("pipeline/video_frames.py"), path, "--frames", args.frames, "--out", frames],
                os.path.join(frames, "log.txt"), os.path.join(frames, "frames"))
            for frame in sorted(os.listdir(os.path.join(frames, "frames"))):
                if not os.path.lexists(os.path.join(inbox, frame)):
                    os.symlink(os.path.join(frames, "frames", frame), os.path.join(inbox, frame))
    print(f"  {len(os.listdir(inbox))} pictures", flush=True)

    print("01 sort", flush=True)
    sort_dir = os.path.join(out, "01_sort")
    sorted_path = os.path.join(sort_dir, "sorted.json")
    if not run([script("pipeline/sort_pictures.py"), inbox, "--out", sort_dir], os.path.join(sort_dir, "log.txt"), sorted_path):
        return 1
    result = json.load(open(sorted_path))
    views = {v["name"]: v for v in result["views"]}
    base = result["lists"]["base"]
    view_path = lambda n: os.path.join(sort_dir, views[n]["view"])

    print("02 base", flush=True)
    base_dir = os.path.join(out, "02_base")
    model = os.path.join(base_dir, "base.glb")
    command = [script("generation/multi_image.py"), *map(view_path, base["both"]), "--mode", "multidiffusion",
               "--resolution", "1024", "--steps", "12", "--faces", "4000000", "--seed", "42", "--out", model]
    if base["shape_only"]:
        command += ["--shape-only", *map(view_path, base["shape_only"])]
    if base["colour_only"]:
        command += ["--colour-only", *map(view_path, base["colour_only"])]
    if len(base["both"]) + len(base["shape_only"]) + len(base["colour_only"]) == 1:  # one picture: generate.py
        only = (base["both"] + base["colour_only"] + base["shape_only"])[0]
        command = [script("generation/generate.py"), view_path(only), "--resolution", "1024", "--seed", "42",
                   "--faces", "4000000"]
    if not run(command, os.path.join(base_dir, "log.txt"), model if "multi_image" in command[0] else None):
        return 1
    if "generate.py" in command[0]:  # generate.py names its result itself
        made = sorted((os.path.join(PROJECT, "outputs", f) for f in os.listdir(os.path.join(PROJECT, "outputs"))
                       if f.endswith(".glb")), key=os.path.getmtime)[-1]
        shutil.move(made, model)

    tries = {}  # block -> how often it was re-inpainted
    for r in range(1, args.rounds + 1):
        print(f"03 judge, round {r}", flush=True)
        grid_dir = os.path.join(out, "03_judge", f"r{r}")
        grid_path = os.path.join(grid_dir, "grid.json")
        if not run([script("pipeline/grid_judge.py"), model, "--sorted", sorted_path, "--out", grid_dir,
                    "--blocks", args.blocks, "--threshold", args.threshold], os.path.join(grid_dir, "log.txt"), grid_path):
            return 1
        grid = json.load(open(grid_path))
        bad = sorted((b for b in grid["blocks"] if b["bad"]), key=lambda b: b["score"])
        summary["rounds"].append({"round": r, "model": os.path.relpath(model, out),
                                  "scores": {"".join(map(str, b["block"])): b["score"] for b in grid["blocks"]},
                                  "bad": ["".join(map(str, b["block"])) for b in bad]})
        print(f"  {len(grid['blocks'])} blocks, {len(bad)} bad", flush=True)
        if not bad:
            break

        print(f"04 re-inpaint {len(bad)} blocks, round {r}", flush=True)
        results = []
        for b in bad:
            key = "".join(map(str, b["block"]))
            k = tries.get(key, 0)
            tries[key] = k + 1
            if k >= len(METHODS):
                print(f"  block {key}: tried {k} times, left as it is", flush=True)
                continue
            how = METHODS[k]
            colour = b["scores"].get("colour")
            shape = b["scores"].get("shape")
            pick = colour or shape  # the colour picture gives shape and colours; else the grey one (shape only)
            role = "both" if colour else "shape"
            block_dir = os.path.join(out, "04_blocks", f"r{r}", key)
            block_out = os.path.join(block_dir, "block.glb")
            command = [script("inpainting/refine_region.py"), model, view_path(pick["view"]), "--box3d",
                       *[f"{v:.5f}" for v in b["box"][0] + b["box"][1]], "--role", role, "--method", how["method"],
                       "--steps", "25", "--seed", str(42 + 7 * k), "--texture-guidance", "3", "--faces", "4000000",
                       "--grow", "1", "--out", block_out]
            if how["method"] == "inversion":
                command += ["--start", how["start"], "--soft", str(how["soft"])]
            if b["redo"] == "colour":
                command += ["--keep-shape"]
            print(f"  block {key} (score {b['score']}, {b['redo']}): {how['method']}, try {k + 1}", flush=True)
            if run(command, os.path.join(block_dir, "log.txt"), block_out):
                results.append(block_out)
        if not results:
            break

        print(f"05 combine, round {r}", flush=True)
        new_model = os.path.join(out, "05_model", f"r{r}.glb")
        if not run([script("pipeline/combine_parts.py"), model, *results, "--out", new_model, "--single"],
                   os.path.join(out, "05_model", f"r{r}_log.txt"), new_model):
            return 1
        model = new_model

    final = os.path.join(out, "final.glb")
    shutil.copy(model, final)
    summary["final"] = os.path.relpath(model, out)
    summary["seconds"] = round(time.time() - start)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    lines = [f"# {args.name}: the grid loop (threshold {args.threshold})", ""]
    for entry in summary["rounds"]:
        lines.append(f"- round {entry['round']} ({entry['model']}): " + ", ".join(
            f"{k} {v}" + (" (bad)" if k in entry["bad"] else "") for k, v in sorted(entry["scores"].items())))
    lines.append(f"\nfinal: {summary['final']}, {summary['seconds']} s")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    if args.zip:  # the small results: models, judge pictures, scores (the inputs and big work folders stay here)
        keep = lambda rel: (rel.endswith((".json", ".md", ".jpg", ".txt")) or rel in ("final.glb", "02_base/base.glb")
                            or (rel.startswith("05_model/") and rel.endswith(".glb")) or rel.startswith("01_sort/contact"))
        import zipfile
        archive = out + ".zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(out):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, out)
                    if keep(rel) and not rel.startswith("00_"):
                        z.write(full, os.path.join(args.name, rel))
        print(f"Result: {archive}", flush=True)
    print(f"Done in {time.time() - start:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

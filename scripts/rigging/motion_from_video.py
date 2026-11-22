"""A video's motion onto a rigged model, in one go: track it (motion_track.py, GEM-X), put it onto the rig
(motion_to_rig.py, Blender), and a side-by-side video to check it (the clip left, the model right).

    python3 scripts/rigging/motion_from_video.py video.mp4 rigged.glb --out outputs/motion/<name> [--start 0] [--duration 10]
        [--keep-colours]

Writes <out>/animated.blend (press Space), <out>/animated.glb, <out>/side_by_side.mp4, and GEM-X's own <out>/motion.pt
and <out>/joints.npz. Any Python 3 starts it; the tracking runs in conda/envs/gemx, the rest in Blender. A video longer
than a minute is tracked in parts (motion_track.py). The model is skin coloured unless --keep-colours.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
BLENDER = os.environ.get("BLENDER") or shutil.which("blender") or os.path.expanduser("~/Applications/blender/blender")
GEMX_FFMPEG = os.path.join(PROJECT, "conda", "envs", "gemx", "bin", "ffmpeg")  # with libx264 (Fedora's ffmpeg has none)


def encoder(ffmpeg):
    """libx264 where this ffmpeg has it, else OpenH264 (Fedora's)."""
    listed = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return ["-c:v", "libx264", "-crf", "20"] if " libx264 " in listed else ["-c:v", "libopenh264", "-b:v", "6M"]


def main():
    parser = argparse.ArgumentParser(description="a video's motion onto a rigged model")
    parser.add_argument("video")
    parser.add_argument("rigged", help="a model rigged by rig_from_glb.py (<name>_rigged.glb)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--camera", choices=["static", "background"], default="static")
    parser.add_argument("--keep-colours", action="store_true", help="the model's own colours, not one skin colour")
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    track = [os.path.join(PROJECT, "conda", "envs", "gemx", "bin", "python"), os.path.join(HERE, "motion_track.py"),
             os.path.abspath(args.video), "--out", out, "--start", str(args.start), "--camera", args.camera]
    if args.duration:
        track += ["--duration", str(args.duration)]
    subprocess.run(track, check=True)
    shutil.rmtree(os.path.join(out, "frames"), ignore_errors=True)
    subprocess.run([BLENDER, "-b", "--python", os.path.join(HERE, "motion_to_rig.py"), "--", os.path.abspath(args.rigged),
                    os.path.join(out, "joints.npz"), out] + (["--keep-colours"] if args.keep_colours else []), check=True)
    frames = sorted(glob.glob(os.path.join(out, "frames", "f_*.png")))
    if frames:
        clip = ["-ss", str(args.start)] + (["-t", str(args.duration)] if args.duration else [])
        pattern = os.path.join(out, "frames", f"f_%0{len(os.path.basename(frames[0])) - 6}d.png")
        ffmpeg = GEMX_FFMPEG if os.path.isfile(GEMX_FFMPEG) else "ffmpeg"
        subprocess.run([ffmpeg, "-v", "error", "-y", *clip, "-i", os.path.abspath(args.video), "-framerate", "30", "-i", pattern,
                        "-filter_complex", "[0:v]fps=30,scale=-2:540[a];[1:v]scale=-2:540[b];[a][b]hstack=inputs=2:shortest=1",
                        *encoder(ffmpeg), "-pix_fmt", "yuv420p", os.path.join(out, "side_by_side.mp4")], check=True)
    print(f"Result: {os.path.join(out, 'animated.blend')}", flush=True)
    print(f"Result: {os.path.join(out, 'side_by_side.mp4')}", flush=True)


if __name__ == "__main__":
    main()

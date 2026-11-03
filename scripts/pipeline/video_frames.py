"""Frames of a video as pictures, evenly spaced over the whole video (no choice made here: which frames are
useful views is decided later by the vision AI in sort_pictures.py).

Usage (TRELLIS.2 environment):
    python scripts/pipeline/video_frames.py turntable.mp4 --frames 12 --out outputs/pipeline/x/00_video
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import os

from organise_pictures import sample_frames


def main():
    parser = argparse.ArgumentParser(description="Evenly spaced frames of a video")
    parser.add_argument("video")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    name = os.path.splitext(os.path.basename(args.video))[0].replace(" ", "_")
    paths = sample_frames(args.video, args.frames, args.out, name)
    print(f"{len(paths)} frames")
    print(f"Result: {os.path.join(args.out, 'frames')}")


if __name__ == "__main__":
    main()

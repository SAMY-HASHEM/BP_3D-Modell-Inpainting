"""Take evenly spaced views from a turntable video (the object or the camera turns once around).

A turntable video shows every side of an object, so TRELLIS.2 doesn't have to guess the back.
This script picks one frame every 360/N degrees and saves them as pictures for multi_image.py.

Usage (TRELLIS.2 environment):
    source activate.sh
    python scripts/video_views.py pictures/sequence1.mp4                     # 8 views from one full turn
    python scripts/video_views.py turn.mp4 --views 12 --front 40             # 12 views, front at frame 40
    python scripts/video_views.py turn.mp4 --turn-frames 600                 # one turn takes 600 frames
    python scripts/video_views.py pictures/sequence1.mp4 --cutout            # clean cut-outs (see below)
    python scripts/multi_image.py pictures/sequence1/view_*.png --resolution 1024 --steps 25

Writes pictures/<video name>/view_<k>_<angle>deg.png (front first, so multi_image.py takes it as the
main view) and overview.jpg with all chosen views.

--cutout removes the background already here (RMBG-2.0, like TRELLIS.2) and keeps only the object:
the biggest connected part and pieces right next to it. Things standing around (a cushion, shoes)
otherwise show up in some views but not in others, which confuses the model. Near every angle it
tries several frames and keeps the cleanest one: the one whose cut-out stays away from the picture
edge (a cut-out reaching the edge means something of the surroundings was kept).
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_frame(video, index):
    video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = video.read()
    if not ok:
        raise RuntimeError(f"frame {index} can't be read")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def clean_cutout(remover, picture, near=0.02):
    """RGBA cut-out with only the object, and how much of it touches the picture edge (0 = clean)."""
    small = picture.convert("RGB").resize((1024, 1024), Image.LANCZOS)
    alpha = np.asarray(remover(small.copy()))[..., 3].astype(np.float32) / 255
    solid = (alpha > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if count <= 1:
        return None, 1.0
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    grow = max(3, int(near * 1024)) | 1
    around = cv2.dilate((labels == main).astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0
    keep = np.isin(labels, [k for k in range(1, count) if around[labels == k].any()])
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # don't cut soft edges
    alpha = alpha * keep
    band = int(0.02 * 1024)
    edge = np.concatenate([alpha[:band].ravel(), alpha[-band:].ravel(), alpha[:, :band].ravel(), alpha[:, -band:].ravel()])
    touching = float((edge > 0.5).mean())
    full = cv2.resize(alpha, picture.size, interpolation=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(full > 0.5)
    margin = int(0.03 * max(picture.size))
    box = (max(0, xs.min() - margin), max(0, ys.min() - margin),
           min(picture.width, xs.max() + margin), min(picture.height, ys.max() + margin))
    rgba = np.dstack([np.asarray(picture.convert("RGB")), (full * 255).astype(np.uint8)])
    return Image.fromarray(rgba).crop(box), touching


def main():
    parser = argparse.ArgumentParser(description="Evenly spaced views from a turntable video")
    parser.add_argument("video")
    parser.add_argument("--views", type=int, default=8, help="how many views around (default 8 = every 45 degrees)")
    parser.add_argument("--front", type=int, default=0, help="frame that shows the front (default 0)")
    parser.add_argument("--turn-frames", type=int,
                        help="frames for one full turn (default: the whole video is one turn)")
    parser.add_argument("--out", help="folder for the pictures (default: pictures/<video name>/)")
    parser.add_argument("--cutout", action="store_true",
                        help="remove the background and keep only the object (needs the TRELLIS.2 environment)")
    parser.add_argument("--search", type=float, default=8,
                        help="with --cutout: look for the cleanest frame up to this many degrees around each "
                             "view (default 8)")
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video)
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        parser.error(f"can't read the video: {args.video}")
    turn = args.turn_frames or count
    name = os.path.splitext(os.path.basename(args.video))[0]
    out = args.out or os.path.join(PROJECT_DIR, "pictures", name)
    os.makedirs(out, exist_ok=True)
    print(f"{count} frames, {video.get(cv2.CAP_PROP_FPS):.0f} fps; one turn = {turn} frames; {args.views} views")

    remover = None
    if args.cutout:
        import torch
        from trellis2.pipelines.rembg.BiRefNet import BiRefNet
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(PROJECT_DIR, "models"))
        remover = BiRefNet("briaai/RMBG-2.0")
        remover.cuda()
        torch.set_grad_enabled(False)

    views = []
    for k in range(args.views):
        angle = 360 * k / args.views
        index = (args.front + round(turn * k / args.views)) % count
        if remover is None:
            picture, note = read_frame(video, index), ""
        else:
            # frames around the angle, nearest first; stop at the first clean one
            reach = round(turn * args.search / 360)
            offsets = sorted(range(-reach, reach + 1, max(1, reach // 5)), key=abs)
            best = None
            for offset in offsets:
                cut, touching = clean_cutout(remover, read_frame(video, (index + offset) % count))
                if cut is not None and (best is None or touching < best[1]):
                    best = (cut, touching, offset)
                if touching == 0:
                    break
            picture, touching, offset = best
            index = (index + offset) % count
            note = f" ({360 * offset / turn:+.1f} degrees for a clean cut-out" + \
                   (")" if touching == 0 else f", still {touching:.1%} at the edge)")
        path = os.path.join(out, f"view_{k:02d}_{round(angle):03d}deg.png")
        picture.save(path)
        views.append(picture)
        print(f"    {angle:5.1f} degrees: frame {index} -> {os.path.relpath(path, PROJECT_DIR)}{note}")

    # overview: all views in one row (two rows from 8 views on)
    per_row = args.views if args.views < 8 else (args.views + 1) // 2
    size = 320
    sheet = Image.new("RGB", (size * per_row, size * ((args.views + per_row - 1) // per_row)))
    for k, picture in enumerate(views):
        thumb = Image.new("RGB", picture.size, (60, 200, 60))  # cut-outs shown on green
        thumb.paste(picture.convert("RGB"), mask=picture.split()[3] if picture.mode == "RGBA" else None)
        thumb.thumbnail((size, size))
        sheet.paste(thumb, ((k % per_row) * size, (k // per_row) * size))
    sheet.save(os.path.join(out, "overview.jpg"))
    print(f"Done: {os.path.relpath(out, PROJECT_DIR)}/ (overview.jpg shows all views)")


if __name__ == "__main__":
    main()

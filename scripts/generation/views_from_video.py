"""Take evenly spaced views from a turntable video (the object or the camera turns once around).

A turntable video shows every side of an object, so TRELLIS.2 doesn't have to guess the back.
This script picks one frame every 360/N degrees and saves them as pictures for multi_image.py.

Usage (TRELLIS.2 environment):
    source activate.sh
    python scripts/generation/views_from_video.py pictures/sequence1.mp4                     # 8 views from one full turn
    python scripts/generation/views_from_video.py turn.mp4 --views 12 --front 40             # 12 views, front at frame 40
    python scripts/generation/views_from_video.py turn.mp4 --turn-frames 600                 # one turn takes 600 frames
    python scripts/generation/views_from_video.py pictures/sequence1.mp4 --cutout            # clean cut-outs (see below)
    python scripts/generation/multi_image.py pictures/sequence1/view_*.png --resolution 1024 --steps 25

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

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def turn_length(widths, step):
    """Frames of one turn, from how the outline's width repeats (its autocorrelation's first clear peak): the
    motorcycle girl's colour video turns three times (750 frames, 250 a turn); taken as one turn, the same back
    view came out twice (Nov 23)."""
    w = np.asarray(widths) - np.mean(widths)
    ac = np.correlate(w, w, "full")[len(w) - 1:]
    ac /= ac[0] + 1e-12
    for lag in range(len(w) // 8, len(w) - 1):
        if ac[lag] > 0.5 and ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1]:
            return lag * step
    return len(w) * step


def find_frames(video, remover, turn, step=5):
    """The frames of one turn that show front, back and the two sides, found by the outline's width (narrowest:
    front / back, widest: the sides; a turn is not always even, so evenly spaced frames miss them), then a ¾ view
    between each: [front, back, side, side, ¾, ¾]. Which of the two narrow ones is the front: Qwen3-VL decides."""
    widths = []
    for i in range(0, turn, step):
        cut, _ = clean_cutout(remover, read_frame(video, i))
        widths.append(cut.width / max(cut.height, 1) if cut is not None else 0)
    one = turn_length(widths, step)
    if one < turn:
        print(f"    the video turns {turn / one:.1f} times: one turn = {one} frames", flush=True)
        widths = widths[:one // step]
    w = np.array(widths)
    smooth = np.convolve(np.r_[w[-3:], w, w[:3]], np.ones(7) / 7, "valid")
    n = len(smooth)
    def extreme(sign, avoid=None):
        order = np.argsort(sign * smooth)
        for k in order:
            if avoid is None or min(abs(k - avoid), n - abs(k - avoid)) > n // 4:
                return int(k)
        return int(order[0])
    narrow1 = extreme(1)
    narrow2 = extreme(1, narrow1)
    wide1 = extreme(-1)
    wide2 = extreme(-1, wide1)
    front, back = narrow1, narrow2
    try:  # which narrow one shows the front
        from organise_pictures import ask, load
        import torch
        model, processor = load("Qwen/Qwen3-VL-8B-Instruct", False)
        pics = [read_frame(video, front * step), read_frame(video, back * step)]
        answer, _ = ask(model, processor, [p.resize((512, 512 * p.height // p.width)) for p in pics],
                        'Two pictures of the same object from opposite sides. Which one shows its FRONT (a face, '
                        'headlights, the side it moves or looks towards)? Answer only with JSON: {"front": 1 or 2}',
                        max_tokens=50)
        if str(answer.get("front")) == "2":
            front, back = back, front
        del model
        torch.cuda.empty_cache()
    except Exception as e:  # no Qwen: the first narrow one
        print(f"    (front by order, Qwen3-VL not asked: {e})", flush=True)
    mid = lambda a, b: int(round((a + b) / 2)) if abs(a - b) < n / 2 else int(round((a + b + n) / 2)) % n
    frames = [front, back, wide1, wide2, mid(front, wide1), mid(back, wide2)]
    print("    outline widths: front %d, back %d, sides %d %d" % tuple(f * step for f in frames[:4]), flush=True)
    return [f * step for f in frames]


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
    parser.add_argument("--frames", type=int, nargs="+",
                        help="exactly these frames (front first), instead of evenly spaced ones: a turn is not always "
                             "even (the motorcycle girl: side to side 87 frames one way, 163 the other)")
    parser.add_argument("--auto-frames", action="store_true",
                        help="find front, back, sides and two ¾ views by the outline's width (with --cutout); the "
                             "turn's length found too. For a full turn only (not a video that sways to and fro)")
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
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(PROJECT_DIR, "local_ai_models"))
        remover = BiRefNet("briaai/RMBG-2.0")
        remover.cuda()
        torch.set_grad_enabled(False)

    if args.auto_frames:
        if remover is None:
            parser.error("--auto-frames needs --cutout")
        args.frames = find_frames(video, remover, turn)
    if args.frames:
        args.views = len(args.frames)
        import json
        json.dump({"frames": args.frames}, open(os.path.join(out, "frames.json"), "w"))
    views = []
    for k in range(args.views):
        angle = 360 * k / args.views
        index = (args.front + round(turn * k / args.views)) % count
        if args.frames:
            index, angle = args.frames[k] % count, 360 * ((args.frames[k] - args.frames[0]) % turn) / turn
        if remover is None:
            picture, note = read_frame(video, index), ""
        else:
            # frames around the angle, nearest first; stop at the first clean one
            reach = 0 if args.frames else round(turn * args.search / 360)  # given frames: exactly those (a grey and a colour set stay pairs)
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

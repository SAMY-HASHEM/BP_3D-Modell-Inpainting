"""A person's motion from a video with GEM-X (NVIDIA): the 77 points of the body, the fingers and the face, in 3D, frame by
frame. motion_to_rig.py then puts that motion onto a rigged model (rig_from_glb.py).

GEM-X (local_ai_models/motion/GEM-X, its own environment conda/envs/gemx: PyTorch 2.10; set up by its install_here.sh)
finds the person, reads 77 2D points per frame and turns them into SOMA body parameters in the world (Y up, metres),
with contact/IK clean-up of the feet. Here they are turned into the 77 joints' positions with SOMA's own model.

    conda/envs/gemx/bin/python scripts/rigging/motion_track.py video.mp4 --out outputs/motion/<name>
        [--start 0] [--duration 10] [--camera static|background]

Writes <out>/joints.npz: joints (T, 77, 3) in the world, names (77), fps (30), view (the way the camera looked, in the
world), and GEM-X's own result(s) <out>/motion.pt or <out>/part_<k>/motion.pt.
GEM-X takes at most 60 s at a time: a longer clip is tracked in parts of 50 s that overlap by 2 s; each part is turned
about the up axis and moved onto the one before by the frames both saw, and the two cross-fade there.
GEM-X makes every clip 30 frames a second. One person: the one seen the most. With a camera that stands still (the
default) the body is taken as the camera saw it, turned upright once for the whole video (read(), upright()): GEM-X's
own world drifts. The hands, feet and knees that rest on the floor are put onto it, frame by frame (ground()).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
GEMX = os.path.join(PROJECT, "local_ai_models", "motion", "GEM-X")
FPS = 30
PART, OVERLAP = 50.0, 2.0  # seconds


def track(video, motion, start, duration, camera):
    """GEM-X on one clip of at most 60 s (skipped when its motion.pt is there already)."""
    if os.path.isfile(motion):
        return
    os.makedirs(os.path.dirname(motion), exist_ok=True)
    command = [os.path.join(os.path.dirname(sys.executable), "gem-infer"), "motion-video", "--task", "body",
               "--input", os.path.abspath(video), "--config", "docs/configs/body_video_native77.yaml",
               "--camera-mode", camera, "--device", "cuda", "--postprocess", "--soma-assets", "assets/soma",
               "--start", f"{start:.3f}", "--duration", f"{duration:.3f}", "--output", motion + ".part"]
    print(" ".join(command), flush=True)
    # GEM-X cuts the clip with ffmpeg and libx264; the environment's own ffmpeg (imageio-ffmpeg) has it, Fedora's not
    bin_ = os.path.dirname(sys.executable)
    subprocess.run(command, cwd=GEMX, check=True, env=dict(os.environ, PATH=bin_ + os.pathsep + os.environ.get("PATH", "")))
    os.replace(motion + ".part", motion)


def turn(rotvec):
    """Axis-angle (..., 3) -> rotation matrices (..., 3, 3)."""
    import numpy as np
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    k = rotvec / np.maximum(angle, 1e-12)
    K = np.zeros(rotvec.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    a = angle[..., None]
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def read(motion, geometry, static=False):
    """GEM-X's result -> the joints (T, 77, 3), the way the camera looked (None: not known), GEM-X's guess which feet
    and wrists stand still (T, 6), and per frame which way is up as the camera sees it (T, 3; None if not known).
    static: the camera stood still. The joints are then the body as the camera saw it (x right, y down, z ahead): GEM-X's
    own world is built up from the body's moves and drifts away (the stretching video, Nov 22: 2 to 3 m sideways and
    80 cm up and down within 50 s on a yoga mat, while the camera saw her within 30 cm). Otherwise GEM-X's world (Y up)."""
    import numpy as np
    import torch
    result = torch.load(motion, map_location="cpu", weights_only=False)
    world = result["pred_soma_params_global"]  # Y up, metres
    incam = result.get("pred_soma_params_incam")
    view = up = None
    if incam is not None:  # the body's turn in the world and seen from the camera: the camera's turn in the world
        g = turn(np.asarray(world["global_orient"][0], dtype=np.float64))
        c = turn(np.asarray(incam["global_orient"][0], dtype=np.float64))
        to_world = g @ np.swapaxes(c, -1, -2)
        up = np.swapaxes(to_world, -1, -2)[:, :, 1]  # the world's up in the camera's axes, frame by frame
        view = (to_world @ np.array([0, 0, 1.0])).mean(0)  # the camera looks along its +Z
        view = view / np.linalg.norm(view)
    if static and incam is not None:
        joints = np.concatenate([chunk["joints"][0].numpy() for _, chunk in geometry.iter_chunks(incam)])
        view = None
    else:
        joints = np.concatenate([chunk["joints"][0].numpy() for _, chunk in geometry.iter_chunks(world)])
        up = None
    still = None  # GEM-X's guess per frame: left ankle, left foot, right ankle, right foot, left wrist, right wrist stand still
    if "static_conf_logits" in result:
        still = 1 / (1 + np.exp(-np.asarray(result["static_conf_logits"][0], dtype=np.float64)))
    return joints.astype(np.float64), view, still, up


def upright(up):
    """Camera axes -> a world with Y up, from which way is up as the camera sees it in every frame. GEM-X's guess wobbles
    from frame to frame (5 to 22 degrees on the stretching video) and leans by up to 9 degrees for a whole 50 s part
    (part 1: the toes 20 cm above the floor in plank): the median of all frames of all parts is taken."""
    import numpy as np
    y = np.median(up, 0)
    y = y / np.linalg.norm(y)
    z = np.array([0, 0, 1.0]) - y[2] * y  # the camera's way ahead, made level
    z = z / np.linalg.norm(z)
    R = np.stack([np.cross(y, z), y, z])  # rows: the world's x, y, z in the camera's axes
    print(f"  up: the camera looks {np.degrees(np.arcsin(np.clip(y[2], -1, 1))):+.1f} degrees up (minus: down) and leans "
          f"{np.degrees(np.arcsin(np.clip(y[0], -1, 1))):+.1f} degrees (median of {len(up)} frames)", flush=True)
    return R


def join(before, after, at):
    """after (it begins at before's frame `at`: the frames from there on both saw) turned about Y and moved onto before,
    cross-faded."""
    import numpy as np
    overlap = min(len(before) - at, len(after))
    before = before[:at + overlap]
    a, b = before[-overlap:].reshape(-1, 3), after[:overlap].reshape(-1, 3)
    ca, cb = a.mean(0), b.mean(0)
    H = (b - cb)[:, [0, 2]].T @ (a - ca)[:, [0, 2]]  # the best turn about the up axis (x and z only)
    angle = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    moved = (after - cb) @ R.T + ca
    w = np.linspace(0, 1, overlap)[:, None, None]
    fade = (1 - w) * before[-overlap:] + w * moved[:overlap]
    print(f"  joined: turned {np.degrees(angle):.1f} degrees, moved {np.linalg.norm(ca - cb):.2f} m, "
          f"{np.abs(a - (b - cb) @ R.T - ca).mean() * 100:.1f} cm apart in the overlap", flush=True)
    return np.concatenate([before[:-overlap], fade, moved[overlap:]])


CONTACT = ("Foot", "ToeBase", "ToeEnd", "Shin", "Hand")  # what touches the floor: feet, toes, knees, hands


def floor(joints, names):
    """The floor at height 0: in most frames the lowest of the feet, toes, knees and hands rests on it, 2 cm above it."""
    import numpy as np
    contact = [k for k, n in enumerate(names) if n.endswith(CONTACT)]
    height = float(np.median(joints[:, contact, 1].min(1))) - .02
    joints = joints.copy()
    joints[..., 1] -= height
    return joints


def ground(joints, names, still):
    """Hands, feet and knees that rest on the floor put onto it, frame by frame. GEM-X gets a lying body's turn wrong by
    some degrees (plank and downward dog, Nov 22: the toes 20 cm above the floor, the hands on it), and nothing it does
    afterwards (its contact clean-up) looks at the floor. A part rests on the floor when it stands still (GEM-X's guess
    for the feet and wrists, its speed for the knees), is near the lowest point of the body and has no other part of the
    body under it (a hand on a knee does not count). The body is turned about the hips and moved up or down so that
    those parts touch the floor as well as they can, smoothed over a few frames; with one part only it is moved."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    T = len(joints)
    group = lambda start: [k for k, n in enumerate(names) if n.startswith(start)]
    index = {n: k for k, n in enumerate(names)}
    speed = np.linalg.norm(np.gradient(gaussian_filter1d(joints, 2, axis=0), axis=0), axis=-1) * FPS
    knee_still = lambda n: np.clip((.25 - speed[:, index[n]]) / .15, 0, 1)
    sites = [  # (joints, how sure it stands still per frame, how high its lowest point is on the floor)
        ([index["LeftFoot"], index["LeftToeBase"], index["LeftToeEnd"]], still[:, :2].max(1), .02),
        ([index["RightFoot"], index["RightToeBase"], index["RightToeEnd"]], still[:, 2:4].max(1), .02),
        (group("LeftHand"), still[:, 4], .015),
        (group("RightHand"), still[:, 5], .015),
        ([index["LeftShin"]], knee_still("LeftShin"), .06),
        ([index["RightShin"]], knee_still("RightShin"), .06),
    ]
    hips = joints[:, index["Hips"]]
    lowest = joints[..., 1].min(1)
    fit = np.zeros((T, 3))  # a, b, c: y + a (x - hips x) + b (z - hips z) + c is the height above the true floor
    weight = np.zeros(T)
    used = 0
    for t in range(T):
        rows, w = [], []
        for site, sure, h in sites:
            k = site[int(joints[t, site, 1].argmin())]
            point = joints[t, k]
            if sure[t] < .3 or point[1] - lowest[t] > .3:
                continue
            others = np.setdiff1d(np.arange(len(names)), site)
            near = np.hypot(*(joints[t, others][:, [0, 2]] - point[[0, 2]]).T) < .12
            if np.any(joints[t, others[near], 1] < point[1] - .03):  # something of the body under it
                continue
            rows.append((point[0] - hips[t, 0], point[2] - hips[t, 2], 1.0, h - point[1]))
            w.append(sure[t])
        if not rows:
            continue
        A, w = np.array(rows), np.array(w)
        M = A[:, :3].T @ (w[:, None] * A[:, :3]) + np.diag([.1, .1, 0])  # .1: a tilt only from parts far apart
        fit[t] = np.linalg.solve(M, A[:, :3].T @ (w * A[:, 3]))
        weight[t] = min(1.0, w.sum())
        used += len(rows) > 1
    limit = np.tan(np.radians(20))
    fit[:, :2] = np.clip(fit[:, :2], -limit, limit)
    smooth = lambda v: gaussian_filter1d(v * weight, 5) / (gaussian_filter1d(weight, 5) + .02)  # no contact: back to 0
    a, b, c = (smooth(fit[:, k]) for k in range(3))
    out = np.empty_like(joints)
    up = np.array([0, 1.0, 0])
    for t in range(T):
        n = np.array([a[t], 1, b[t]]) / np.linalg.norm([a[t], 1, b[t]])
        axis = np.cross(n, up)
        s_ = np.linalg.norm(axis)
        R = np.eye(3)
        if s_ > 1e-9:
            k = axis / s_
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + s_ * K + (1 - n[1]) * K @ K
        pivot = np.array([hips[t, 0], -c[t], hips[t, 2]])
        out[t] = (joints[t] - pivot) @ R.T + np.array([hips[t, 0], 0, hips[t, 2]])
    turned = np.degrees(np.arctan(np.hypot(a, b)))
    print(f"  ground: parts on the floor in {np.mean(weight > 0):.0%} of the frames, two or more far apart in "
          f"{used / T:.0%}; the body turned {np.mean(turned):.1f} degrees on average (at most {turned.max():.0f}), moved "
          f"{np.mean(np.abs(c)) * 100:.1f} cm up or down", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="the 3D motion of the person in a video (GEM-X)")
    parser.add_argument("video")
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, help="seconds (default: the rest of the video)")
    parser.add_argument("--camera", choices=["static", "background"], default="static",
                        help="static: the camera does not move (default); background: it turns, estimated from the background")
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    length = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                   args.video], capture_output=True, text=True, check=True).stdout.strip())
    total = min(args.duration or length, length - args.start)
    if total <= 60:
        parts = [(os.path.join(out, "motion.pt"), args.start, total)]
    else:
        parts, k = [], 0
        while k * (PART - OVERLAP) < total - OVERLAP:
            begin = k * (PART - OVERLAP)
            parts.append((os.path.join(out, f"part_{k + 1}", "motion.pt"), args.start + begin, min(PART, total - begin)))
            k += 1
        print(f"{total:.0f} s: {len(parts)} parts of {PART:.0f} s, {OVERLAP:.0f} s overlapping", flush=True)
    for k, (motion, begin, seconds) in enumerate(parts, 1):
        print(f"part {k} of {len(parts)}: {begin:.0f} s to {begin + seconds:.0f} s", flush=True)
        track(args.video, motion, begin, seconds, args.camera)

    import numpy as np
    import torch
    sys.path.insert(0, GEMX)
    from gem.motion.frontend.joint_sets import SOMA77_JOINT_NAMES
    from gem.motion.geometry import MotionGeometry
    geometry = MotionGeometry(os.path.join(GEMX, "assets", "soma"), device="cuda" if torch.cuda.is_available() else "cpu",
                              apply_correctives=False)
    read_ = [read(motion, geometry, args.camera == "static") for motion, _, _ in parts]
    view = read_[0][1]
    if read_[0][3] is not None:  # the body as the camera saw it: turned upright, the camera looking along the world's z
        R = upright(np.concatenate([r[3] for r in read_]))
        read_ = [(r[0] @ R.T, *r[1:]) for r in read_]
        view = R @ np.array([0, 0, 1.0])
    joints, _, still, _ = read_[0]
    for (more, _, still_more, _), (_, begin, _) in zip(read_[1:], parts[1:]):
        at = int(round((begin - args.start) * FPS))
        half = (len(joints) - at) // 2
        if still is not None and still_more is not None:
            still = np.concatenate([still[:at + half], still_more[half:]])
        joints = join(joints, more, at)
    names = list(SOMA77_JOINT_NAMES)
    joints = floor(joints, names)
    if still is not None and len(still) == len(joints):
        joints = ground(joints, names, still)
    path = os.path.join(out, "joints.npz")
    extra = {"view": view.astype(np.float32)} if view is not None else {}
    np.savez(path, joints=joints.astype(np.float32), names=np.array(SOMA77_JOINT_NAMES), fps=float(FPS), **extra)
    height = joints[:, :, 1].max(1) - joints[:, :, 1].min(1)
    print(f"{len(joints)} frames ({len(joints) / FPS:.1f} s), the person about {np.median(height):.2f} m from the lowest to the "
          f"highest point" + (f"; the camera looked along {np.round(view, 2).tolist()}" if view is not None else ""), flush=True)
    print(f"Result: {path}", flush=True)


if __name__ == "__main__":
    main()

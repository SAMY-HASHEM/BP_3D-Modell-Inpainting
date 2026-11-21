"""Training data for the rig head: rigged humanoids read into TRELLIS.2's shape latent, with their skeleton and skin
weights on the latent's voxels.

The rig head (rig_head.py) learns to read a humanoid skeleton out of TRELLIS.2's shape latent, the same latent its
shape step makes (64^3 grid, 32 numbers per voxel). To learn that, it needs latents of models whose skeleton is known:
Rig-XL (UniRig, Zhang et al. 2025, huggingface.co/VAST-AI/UniRig, data/rigxl): ~3000 Mixamo and ~800 VRoid humanoids
with mesh, joints and skin weights (processed.7z, raw_data.npz per model).

  skeleton   one fixed humanoid skeleton of 22 bones (BONES): Mixamo's and VRoid's body bones are the same 22 in the
             same order. Every other bone (fingers, hair, skirt, a tail) is folded into the body bone it hangs from:
             its skin weights go to that bone. A model without all 22 is skipped (most of those: Mixamo rigs that
             end at the forearm, ~8%; where the hand's joint would be is not in the data)
  pose       turned upright and to the front as TRELLIS.2 puts a model: up +Y, front +Z (glTF), i.e. the cube's -Y.
             Up is from the hips to the head, the front is where the feet point (ankle to toe)
  latent     the mesh is put into TRELLIS.2's cube (inpaint.Placement) and encoded with its shape encoder
             (inpaint.encode_shape): coords (L, 3) on the 64^3 grid, feats (L, 32)
  fingers    30 finger joints (FINGERS) where the rig has them (fingers (30, 3), has_fingers (30,)): all VRoid rigs,
             about half the Mixamo ones. A finger is about one voxel thick in the latent, so its skin stays in the
             hand's; the joints are for a finer step on the finished mesh
  labels     joints (22, 3) in the cube; skin (L, 22): the mean skin weights of the surface points in each voxel (a
             voxel without a point takes its nearest point's)

    source activate.sh
    python scripts/rigging/rig_data.py datasets/rigxl --lists mixamo_train vroid_train --out datasets/rig_latents
    python scripts/rigging/rig_data.py datasets/rigxl --lists mixamo_test --out datasets/rig_latents --limit 5

Writes <out>/<id>.npz per model and <out>/report.json (what was skipped and why). Models already done are skipped, so
a stopped run goes on where it stopped.
"""
import _paths  # noqa: F401  (the scripts of the other folders)
import argparse
import json
import os
import re
import sys
import time

import numpy as np

BONES = ["hips", "spine", "chest", "upper_chest", "neck", "head",
         "shoulder_l", "upper_arm_l", "lower_arm_l", "hand_l", "shoulder_r", "upper_arm_r", "lower_arm_r", "hand_r",
         "upper_leg_l", "lower_leg_l", "foot_l", "toe_l", "upper_leg_r", "lower_leg_r", "foot_r", "toe_r"]
PARENT = [-1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20]  # upper_chest carries neck and arms
NAMES = {  # the 22 body bones of each kind of rig, in the order of BONES (UniRig's configs/skeleton/*.yaml)
    "mixamo": ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
               "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
               "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"],
    "vroid": ["J_Bip_C_Hips", "J_Bip_C_Spine", "J_Bip_C_Chest", "J_Bip_C_UpperChest", "J_Bip_C_Neck", "J_Bip_C_Head",
              "J_Bip_L_Shoulder", "J_Bip_L_UpperArm", "J_Bip_L_LowerArm", "J_Bip_L_Hand",
              "J_Bip_R_Shoulder", "J_Bip_R_UpperArm", "J_Bip_R_LowerArm", "J_Bip_R_Hand",
              "J_Bip_L_UpperLeg", "J_Bip_L_LowerLeg", "J_Bip_L_Foot", "J_Bip_L_ToeBase",
              "J_Bip_R_UpperLeg", "J_Bip_R_LowerLeg", "J_Bip_R_Foot", "J_Bip_R_ToeBase"],
}
FINGERS = [f"{finger}{k}_{side}" for side in "lr" for finger in ("thumb", "index", "middle", "ring", "little")
           for k in (1, 2, 3)]  # 30 joints; a finger's first joint hangs from the hand
FINGER_NAMES = {  # the same 30 in each kind of rig
    "mixamo": [f"{side}Hand{finger}{k}" for side in ("Left", "Right")
               for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky") for k in (1, 2, 3)],
    "vroid": [f"J_Bip_{side}_{finger}{k}" for side in "LR"
              for finger in ("Thumb", "Index", "Middle", "Ring", "Little") for k in (1, 2, 3)],
}
GRID = 64  # the shape latent's grid (TRELLIS.2 at 1024: 1024 / 16)
SAMPLES = 200_000  # surface points for the voxels' skin weights


def plain(name):
    """A bone's name without the rig's prefix ("mixamorig:Hips", "mixamorig1:Hips", "mixamorigHips" -> "hips")."""
    return re.sub(r"^mixamorig\d*", "", str(name).rsplit(":", 1)[-1]).lower()


def body_bones(names):
    """The index of each of the 22 body bones in this rig, or None if it lacks one."""
    where = {plain(n): i for i, n in enumerate(names)}
    for kind in NAMES.values():
        found = [where.get(n.lower()) for n in kind]
        if None not in found:
            return found
    return None


def finger_bones(names):
    """The index of each of the 30 finger joints in this rig, -1 where it has none (half the Mixamo rigs have no or
    only some fingers)."""
    where = {plain(n): i for i, n in enumerate(names)}
    found = [-1] * len(FINGERS)
    for kind in FINGER_NAMES.values():
        for k, n in enumerate(kind):
            if found[k] < 0:
                found[k] = where.get(plain(n), -1)
    return np.array(found)


def fold(parents, body):
    """For every bone of the rig: the body bone (0..21) it is or hangs from; above the hips (a root): the hips."""
    of = {j: k for k, j in enumerate(body)}
    out = []
    for j in range(len(parents)):
        b = j
        while b is not None and b not in of:
            b = parents[b]
        out.append(of[b] if b is not None else 0)
    return np.array(out)


def snap(v):
    """The axis (±x, ±y, ±z) nearest to a direction."""
    axis = np.zeros(3)
    k = int(np.argmax(np.abs(v)))
    axis[k] = np.sign(v[k])
    return axis


def upright(joints):
    """The rotation (3 x 3) that turns the model upright and to the front: hips to head along +Y, the feet along +Z.
    Both snapped to the nearest axis: the rigs are made standing straight, so this only fixes which way is which."""
    up = snap(joints[5] - joints[0])
    feet = (joints[17] - joints[16]) + (joints[21] - joints[20])
    feet = feet - (feet @ up) * up
    if np.linalg.norm(feet) < 1e-6:
        return None
    front = snap(feet)
    if abs(front @ up) > 0.5:
        return None
    side = np.cross(up, front)
    return np.stack([side, up, front])  # rows: where x, y, z go


def voxel_skin(coords, points, weights):
    """(L, 22): the mean weights of the surface points in each of the latent's voxels; an empty voxel takes its
    nearest point's (the latent's voxels reach a little past the surface)."""
    from scipy.spatial import cKDTree
    cell = np.clip(np.floor((points + 0.5) * GRID).astype(int), 0, GRID - 1)
    key = lambda c: (c[:, 0] * GRID + c[:, 1]) * GRID + c[:, 2]
    point_key, voxel_key = key(cell), key(coords)
    order = np.argsort(point_key)
    point_key, weights_sorted = point_key[order], weights[order]
    start, end = np.searchsorted(point_key, voxel_key), np.searchsorted(point_key, voxel_key, side="right")
    running = np.vstack([np.zeros((1, weights.shape[1])), np.cumsum(weights_sorted, 0, dtype=np.float64)])
    sums = running[end] - running[start]  # the points of each voxel lie together in the sorted order
    count = (end - start)[:, None]
    skin = np.where(count > 0, sums / np.maximum(count, 1), 0)
    empty = (count[:, 0] == 0)
    if empty.any():
        centres = (coords[empty] + 0.5) / GRID - 0.5
        skin[empty] = weights[cKDTree(points).query(centres)[1]]
    return skin / np.maximum(skin.sum(1, keepdims=True), 1e-8)


def prepare(raw):
    """The model turned upright, its body joints (22, 3) and its vertices' skin on the 22 body bones, all still in the
    model's own size; or the reason it is skipped."""
    names = list(raw["names"])
    body = body_bones(names)
    if body is None:
        return "not all 22 body bones"
    parents = [None if p is None else int(p) for p in raw["parents"]]
    vertices, faces = raw["vertices"].astype(np.float64), raw["faces"].astype(np.int64)
    joints = raw["joints"].astype(np.float64)[body]
    turn = upright(joints)
    if turn is None:
        return "no front from the feet"
    skin = raw["skin"].astype(np.float32)
    if skin.shape != (len(vertices), len(names)):
        return f"skin {skin.shape} does not fit {len(vertices)} vertices, {len(names)} bones"
    body_skin = np.zeros((len(vertices), len(BONES)), np.float32)
    np.add.at(body_skin.T, fold(parents, body), skin.T)
    unweighted = body_skin.sum(1) < 1e-6
    if unweighted.mean() > 0.05:
        return f"{unweighted.mean():.0%} of the vertices without weights"
    body_skin[unweighted, 0] = 1.0  # a few loose vertices: the hips
    body_skin /= body_skin.sum(1, keepdims=True)
    finger_index = finger_bones(names)
    fingers = np.where(finger_index[:, None] >= 0, raw["joints"].astype(np.float64)[np.maximum(finger_index, 0)], 0)
    return vertices @ turn.T, faces, joints @ turn.T, body_skin, fingers @ turn.T, finger_index >= 0


def encode(encoder, vertices, faces, joints, skin, fingers, has_fingers):
    """The latent and the labels of one prepared model. The fingers' skin stays in the hand's (a finger is about one
    voxel thick in the 64^3 latent); their joints are kept for the finer step on the mesh."""
    import torch
    import trimesh
    from inpaint import Placement, encode_shape
    place = Placement(vertices)
    cube_vertices, cube_joints = place.to_cube(vertices), place.to_cube(joints)
    with torch.no_grad():
        latent = encode_shape(encoder, cube_vertices.astype(np.float32), faces)
    coords = latent.coords[:, 1:].cpu().numpy().astype(np.int16)
    feats = latent.feats.float().cpu().numpy().astype(np.float16)
    mesh = trimesh.Trimesh(cube_vertices, faces, process=False)
    points, face_index = trimesh.sample.sample_surface(mesh, SAMPLES, seed=0)
    corner = mesh.faces[face_index]
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face_index], points)
    point_skin = np.einsum("pk,pkb->pb", bary, skin[corner])
    return {"coords": coords, "feats": feats, "joints": cube_joints.astype(np.float32),
            "fingers": place.to_cube(fingers).astype(np.float32), "has_fingers": has_fingers,
            "skin": voxel_skin(coords.astype(int), points, point_skin).astype(np.float16)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("rigxl", help="the Rig-XL folder: processed/ (unpacked processed.7z) and datalist/")
    parser.add_argument("--lists", nargs="+", default=["mixamo_train", "vroid_train"], help="datalist/<name>.txt")
    parser.add_argument("--out", default="datasets/rig_latents")
    parser.add_argument("--limit", type=int, help="only the first N models (a test)")
    args = parser.parse_args()

    ids = []
    for name in args.lists:
        ids += [line.strip() for line in open(os.path.join(args.rigxl, "datalist", name + ".txt")) if line.strip()]
    ids = ids[:args.limit] if args.limit else ids
    os.makedirs(args.out, exist_ok=True)
    report_path = os.path.join(args.out, "report.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {"done": [], "skipped": {}}

    import torch
    import trellis2.models as models
    from inpaint import SHAPE_ENCODER
    encoder = models.from_pretrained(SHAPE_ENCODER).cuda().eval()
    started = time.time()
    for n, model_id in enumerate(ids):
        name = model_id.strip("/").replace("/", "_")
        target = os.path.join(args.out, name + ".npz")
        if os.path.exists(target) or name in report["skipped"]:
            continue
        # processed.7z unpacks to rigxl/<id>/raw_data.npz
        places = [os.path.join(args.rigxl, sub, model_id, "raw_data.npz") for sub in ("rigxl", "processed", "")]
        source = next((p for p in places if os.path.exists(p)), places[0])
        try:
            prepared = prepare(np.load(source, allow_pickle=True))
            if isinstance(prepared, str):
                report["skipped"][name] = prepared
            else:
                np.savez_compressed(target, **encode(encoder, *prepared))
                report["done"].append(name)
        except Exception as error:  # a broken file: noted, the run goes on
            report["skipped"][name] = f"{type(error).__name__}: {error}"[:200]
            torch.cuda.empty_cache()
        if n % 20 == 0 or n == len(ids) - 1:
            json.dump(report, open(report_path, "w"), indent=1)
            rate = (n + 1) / (time.time() - started)
            print(f"{n + 1}/{len(ids)}  done {len(report['done'])}  skipped {len(report['skipped'])}  "
                  f"({rate:.2f}/s)", flush=True)
    json.dump(report, open(report_path, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())

"""Rigs a finished humanoid model (a .glb from TRELLIS.2 or Pixal3D) with the rig head (rig_head.py).

The model is read into TRELLIS.2's shape latent exactly like the training data (rig_data.py: glTF axes, Y up, front
+Z, fitted into the cube), the rig head gives the 22 body joints, 30 finger joints and the skin of every voxel, and
Blender builds the skinned model: an armature with those bones, each vertex weighted like the voxels around it.
A hand's weight is shared out to the nearest finger bone, so the fingers move too.

    source activate.sh
    python scripts/rigging/rig_from_glb.py model.glb --out outputs/rigged/model      # model_rigged.glb, .npz, .jpg

Writes <out>_rigged.glb (with the armature, ready for animation), <out>.npz (the latent and the prediction, in
rig_data.py's format), <out>_weights.npz (each point's 4 bones), <out>.jpg (rig_picture.py's picture of it) and
<out>_pose.png (the model at rest and posed, side by side: does the skin follow the bones?).
Needs the trained rig head (local_ai_models/rig_head/best.pt, or --checkpoint), TRELLIS.2's shape encoder, a GPU and
Blender 4.x (BLENDER=/path/to/blender if it is not on the PATH). RIG_SMOOTH sets the weight smoothing (default 2 voxels).
About 4 minutes per model (1M faces).
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(HERE))
BLENDER = os.environ.get("BLENDER") or next(  # the project's own Blender (the server), else the installed one
    (b for b in (os.path.join(PROJECT_DIR, "blender", "blender"),) if os.path.exists(b)), "blender")


# ---------------------------------------------------------------- 1: the prediction (the project's Python)

def predict(model_path, out, checkpoint):
    import _paths  # noqa: F401
    import numpy as np
    import torch
    import trimesh
    import trellis2.models as models
    from inpaint import SHAPE_ENCODER, Placement, encode_shape
    from rig_head import load_head, predict_one
    sys.modules.pop("rig_picture", None)
    sys.path.insert(0, HERE)  # scripts/tools has a rig_picture.py of its own
    from rig_picture import picture

    mesh = trimesh.load(model_path, force="mesh", process=False)  # glTF axes: Y up, front +Z
    place = Placement(mesh.vertices)
    encoder = models.from_pretrained(SHAPE_ENCODER).cuda().eval()
    with torch.no_grad():
        latent = encode_shape(encoder, place.to_cube(mesh.vertices).astype(np.float32), np.asarray(mesh.faces))
    coords = latent.coords[:, 1:].cpu().numpy().astype(np.int16)
    feats = latent.feats.float().cpu().numpy()
    del encoder
    head = load_head(checkpoint)
    joints, fingers, skin = predict_one(head, coords, feats)

    def to_gltf(p):  # inverse of Placement.to_cube
        p = np.asarray(p, float)
        return place.center + np.stack([p[:, 0], p[:, 2] - place.lift, -p[:, 1]], 1) / place.scale

    from rig_data import GRID
    np.savez(out + ".npz", coords=coords, feats=feats.astype(np.float16), joints=joints, fingers=fingers,
             has_fingers=np.ones(len(fingers), bool), skin=skin.astype(np.float16),
             gltf_joints=to_gltf(joints), gltf_fingers=to_gltf(fingers),
             gltf_voxels=to_gltf((coords + 0.5) / GRID - 0.5), voxel_size=1 / GRID / place.scale)
    picture(out + ".npz")
    start = time.time()
    welded, bones, weights = vertex_weights(np.asarray(mesh.vertices), np.asarray(mesh.faces), to_gltf(
        (coords + 0.5) / GRID - 0.5), skin, to_gltf(joints), to_gltf(fingers), 1 / GRID / place.scale)
    np.savez(out + "_weights.npz", points=welded, bones=bones, weights=weights)
    print(f"predicted: {out}.npz, weights of {len(welded)} points in {time.time() - start:.0f} s", flush=True)


SMOOTH = float(os.environ.get("RIG_SMOOTH", 2.0))  # how far the weights are smoothed along the surface, in voxels


def vertex_weights(points, faces, voxels, skin, joints, fingers, voxel):
    """Each point of the mesh (glTF axes) -> its 4 bones and their weights (sums to 1), of the 52 (22 body, 30 finger):
      1. the distance-weighted mean skin of the voxels around it
      2. a hand's weight goes to the nearest of the hand and its 15 finger bones (the latent has no finger skin)
      3. smoothed along the surface (implicit Laplacian smoothing, about SMOOTH voxels far): neighbouring points
         then follow the same bones, so a sleeve or a cloak bends instead of wrinkling and tearing
    Points at the same place (a UV seam splits a vertex in two) are one point here: (welded points, bones, weights)."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import cg
    from scipy.spatial import cKDTree
    from rig_data import BONES, FINGERS

    welded, index = np.unique(np.round(points / (voxel * 1e-4)).astype(np.int64), axis=0, return_inverse=True)
    index = index.ravel()
    p = np.zeros((len(welded), 3))
    np.add.at(p, index, points)
    p /= np.bincount(index, minlength=len(welded))[:, None]
    n = len(p)
    w = np.zeros((n, len(BONES) + len(FINGERS)), np.float32)

    # 1. the voxels around (up to 8 within 1.5 voxels; else the nearest)
    dist, near = cKDTree(voxels).query(p, k=8, distance_upper_bound=1.5 * voxel)
    found = np.isfinite(dist)
    near_one = cKDTree(voxels).query(p[~found[:, 0]])[1]
    dist[~found[:, 0], 0], near[~found[:, 0], 0], found[~found[:, 0], 0] = voxel, near_one, True
    share = np.where(found, 1 / (np.where(found, dist, 1) + 0.2 * voxel), 0)
    w[:, :len(BONES)] = np.einsum("pk,pkb->pb", share, skin[np.where(found, near, 0)].astype(np.float32))
    w /= w.sum(1, keepdims=True)

    # 2. the hands' weight to the fingers
    for side in "lr":
        hand = BONES.index("hand_" + side)
        bones = [hand] + [len(BONES) + k for k, f in enumerate(FINGERS) if f.endswith(side)]
        heads = [joints[hand]] + [fingers[k] for k, f in enumerate(FINGERS) if f.endswith(side)]
        tails = [fingers[FINGERS.index("middle1_" + side)]]
        for k, f in enumerate(FINGERS):
            if f.endswith(side):
                tails.append(fingers[k + 1] if f[-3] != "3" else 2 * fingers[k] - fingers[k - 1])
        a, b = np.array(heads), np.array(tails)
        rows = np.nonzero(w[:, hand] > 0)[0]
        ab = b - a
        t = np.clip(np.einsum("pjc,jc->pj", p[rows, None] - a, ab) / (ab ** 2).sum(1).clip(1e-12), 0, 1)
        d = np.linalg.norm(p[rows, None] - (a + t[..., None] * ab), axis=2)
        nearest = np.array(bones)[d.argmin(1)]
        moved = w[rows, hand].copy()
        w[rows, hand] = 0
        w[rows, nearest] += moved

    # 3. smoothing along the surface: (I + lam L) w' = w, L the edge graph's Laplacian
    if SMOOTH > 0:
        f = index[faces]
        edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges = edges[edges[:, 0] != edges[:, 1]]
        adjacency = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
        adjacency = ((adjacency + adjacency.T) > 0).astype(np.float64)
        degree = np.asarray(adjacency.sum(1)).ravel()
        length = np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1).mean()
        lam = (SMOOTH * voxel / length) ** 2
        system = sp.identity(n) + lam / degree.mean() * (sp.diags(degree) - adjacency)
        for b in np.nonzero(w.max(0) > 0)[0]:
            w[:, b] = cg(system, w[:, b].astype(np.float64), x0=w[:, b].astype(np.float64), rtol=1e-4,
                         maxiter=500)[0]
        w = w.clip(0)

    # 4. the 4 biggest, summing to 1 (glTF keeps 4)
    top = np.argsort(-w, 1)[:, :4]
    weights = np.take_along_axis(w, top, 1)
    weights[weights < 0.01 * weights[:, :1]] = 0
    weights /= weights.sum(1, keepdims=True).clip(1e-9)
    return p.astype(np.float32), top.astype(np.int16), weights.astype(np.float32)


# ---------------------------------------------------------------- 2: the skinned model (Blender's Python)

def build(model_path, out):
    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.kdtree import KDTree
    sys.path.insert(0, HERE)
    from rig_data import BONES, FINGERS, PARENT

    d = np.load(out + ".npz")
    b = lambda p: Vector((p[0], -p[2], p[1]))  # glTF (Y up) -> Blender (Z up), as the glTF importer turns it
    joints = [b(p) for p in d["gltf_joints"]]
    fingers = [b(p) for p in d["gltf_fingers"]]
    voxels = [b(p) for p in d["gltf_voxels"]]
    skin = d["skin"].astype(float)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=model_path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]

    # bones: the body's 22 (PARENT), then each finger hand -> 1 -> 2 -> 3
    names = list(BONES) + list(FINGERS)
    heads = joints + fingers
    parents = list(PARENT) + [BONES.index("hand_" + n[-1]) if n[-3] == "1" else len(BONES) + k - 1
                              for k, n in enumerate(FINGERS)]
    children = {i: [j for j, p in enumerate(parents) if p == i] for i in range(len(names))}
    height = max(p.z for p in joints) - min(p.z for p in joints)

    armature = bpy.data.objects.new("rig", bpy.data.armatures.new("rig"))
    bpy.context.scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    edit = {}
    for i, name in enumerate(names):
        bone = armature.data.edit_bones.new(name)
        bone.head = heads[i]
        kids = [c for c in children[i] if not names[c].startswith(("thumb", "index", "ring", "little"))]
        if name.startswith("hand"):
            kids = [names.index("middle1_" + name[-1])]
        if name in ("hips", "upper_chest"):
            kids = [BONES.index("spine" if name == "hips" else "neck")]
        if kids:
            tail = sum((heads[c] for c in kids), Vector()) / len(kids)
        elif parents[i] >= 0:  # an end (head, toe, fingertip): on along its parent's direction
            step = heads[i] - heads[parents[i]]
            tail = heads[i] + step.normalized() * max(step.length * (0.9 if name != "head" else 1.6), height * 0.01)
        else:
            tail = heads[i] + Vector((0, 0, height * 0.05))
        if (tail - bone.head).length < height * 0.005:
            tail = bone.head + Vector((0, 0, height * 0.02))
        bone.tail = tail
        edit[i] = bone
    for i, p in enumerate(parents):
        if p >= 0:
            edit[i].parent = edit[p]
    bpy.ops.object.mode_set(mode="OBJECT")

    # weights: from vertex_weights (each vertex takes those of its point there)
    w = dict(np.load(out + "_weights.npz"))  # read once (an npz is read again at every w[...])
    w["bones"], w["weights"] = w["bones"].tolist(), w["weights"].tolist()
    tree = KDTree(len(w["points"]))
    for k, q in enumerate(w["points"]):
        tree.insert(b(q), k)
    tree.balance()
    for obj in meshes:
        world = obj.matrix_world
        groups = [obj.vertex_groups.new(name=n) for n in names]
        for v in obj.data.vertices:
            k = tree.find(world @ v.co)[1]
            for bone, x in zip(w["bones"][k], w["weights"][k]):
                if x > 0:
                    groups[bone].add([v.index], float(x), "REPLACE")
        obj.parent = armature
        modifier = obj.modifiers.new("rig", "ARMATURE")
        modifier.object = armature
    bpy.ops.export_scene.gltf(filepath=out + "_rigged.glb", export_format="GLB", export_skins=True)
    print("rigged:", out + "_rigged.glb", flush=True)
    pose_picture(armature, meshes, out + "_pose.png")


def pose_picture(armature, meshes, path):
    """A check that the skin follows the bones: the model at rest and posed (left arm up, right arm forward, left
    knee bent, head turned), side by side, from the front."""
    import math
    import bpy
    from mathutils import Euler, Vector
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.resolution_x, scene.render.resolution_y = 1400, 900
    scene.render.film_transparent = False
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    low = Vector([min(c[i] for c in corners) for i in range(3)])
    high = Vector([max(c[i] for c in corners) for i in range(3)])
    centre, span = (low + high) / 2, max(high - low)
    rest = armature.copy()  # the second model: posed, moved to the right
    camera = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = span * 2.4
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = centre + Vector((span * 0.6, -span * 4, 0))
    camera.rotation_euler = (math.pi / 2, 0, 0)
    for o in meshes:  # a posed copy beside the original
        twin = o.copy()
        twin.data = o.data
        scene.collection.objects.link(twin)
        twin.parent = rest
        twin.modifiers["rig"].object = rest
    scene.collection.objects.link(rest)
    rest.location.x += span * 1.2
    poses = {"upper_arm_l": (0, -70, 0), "lower_arm_l": (0, -30, 0), "upper_arm_r": (60, 0, 0),
             "upper_leg_l": (-50, 0, 0), "lower_leg_l": (80, 0, 0), "head": (0, 30, 0), "spine": (10, 0, 0)}
    for name, angles in poses.items():
        bone = rest.pose.bones[name]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = Euler([math.radians(a) for a in angles])
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("pose picture:", path, flush=True)


def main():
    if "--" in sys.argv:  # in Blender
        build(*sys.argv[sys.argv.index("--") + 1:])
        return
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model")
    parser.add_argument("--out", required=True, help="path without ending: <out>_rigged.glb, <out>.npz, <out>.jpg")
    parser.add_argument("--checkpoint", default=os.path.join(PROJECT_DIR, "local_ai_models", "rig_head", "best.pt"))
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    predict(os.path.abspath(args.model), os.path.abspath(args.out), args.checkpoint)
    subprocess.run([BLENDER, "-b", "--python", os.path.abspath(__file__), "--",
                    os.path.abspath(args.model), os.path.abspath(args.out)], check=True)


if __name__ == "__main__":
    main()

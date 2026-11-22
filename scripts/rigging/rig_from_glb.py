"""Rigs a finished humanoid model (a .glb from TRELLIS.2 or Pixal3D) with the rig head (rig_head.py).

The model is read into TRELLIS.2's shape latent exactly like the training data (rig_data.py: glTF axes, Y up, front
+Z, fitted into the cube), the rig head gives the 22 body joints, 30 finger joints and the skin of every voxel, and
Blender builds the skinned model: an armature with those bones, each vertex weighted like the voxels around it.
A hand's weight is shared out to the nearest finger bone, so the fingers move too.

    source activate.sh
    python scripts/rigging/rig_from_glb.py model.glb --out outputs/rigged/model      # model_rigged.glb, .npz, .jpg
    python scripts/rigging/rig_from_glb.py model.glb --out outputs/rigged/model --joints views   # an unusual pose

--joints views: the joints found on 12 pictures of the model (rig_views.py, GEM-X's 77-point finder) and the weights from
the bones each point wraps around (wrap_weights): for a model that does not stand in a T or A pose (Mai, Nov 22: hands on her hips, the rig head put
her elbows outside her body and her arms did not move).

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

HAND = ("lower_arm", "hand", "thumb", "index", "middle", "ring", "little")
BODY = ("hips", "spine", "chest", "upper_leg", "lower_leg")


def one_mesh(meshes):
    """All parts as one mesh, the seams of the parts welded (Sketchfab splits a model every 65,532 points): a cut can
    then run across them."""
    import bpy
    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
    mesh = bpy.context.view_layer.objects.active if len(meshes) > 1 else meshes[0]
    bpy.context.view_layer.objects.active = mesh
    zs = [v.co.z for v in mesh.data.vertices]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=(max(zs) - min(zs)) * 1e-6)
    bpy.ops.object.mode_set(mode="OBJECT")
    return [mesh]


def free_hands(armature, obj):
    """A hand sculpted resting on the body is fused into it (Mai, Nov 22: her hands pressed into her hips): whatever
    the weights, the faces between hand and hip stretch into a sheet when the hand moves away. Those faces (one corner
    moved by a hand, forearm or finger, another by the hips, spine, chest or a leg) are taken out and the holes on
    both sides closed: the hand comes off the hip, with a flat palm and a patch on the hip."""
    import bmesh
    import bpy
    import numpy as np
    names = {g.index: g.name for g in obj.vertex_groups}
    side = {i: 1 if n.startswith(HAND) else 2 if n.startswith(BODY) else 0 for i, n in names.items()}
    kind = np.zeros(len(obj.data.vertices), np.int8)  # 1: a hand's, 2: the body's
    mixed = 0
    for v in obj.data.vertices:  # a point is a hand's or the body's, not half of each (the copy from the lighter mesh
        if not v.groups:         # blends them at the border: such points went half way, in streaks)
            continue
        pull = {1: 0.0, 2: 0.0}
        for x in v.groups:
            if side[x.group]:
                pull[side[x.group]] += x.weight
        if pull[1] > 0 and pull[2] > 0:
            loser = 2 if pull[1] >= pull[2] else 1
            for g in [x.group for x in v.groups if side[x.group] == loser]:
                obj.vertex_groups[g].remove([v.index])
            mixed += 1
        top = max(v.groups, key=lambda x: x.weight)
        kind[v.index] = side[top.group]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.vertex_group_normalize_all(lock_active=False)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    seam = [f for f in bm.faces if {kind[v.index] for v in f.verts} >= {1, 2}]
    edge = {v for f in seam for v in f.verts}
    bmesh.ops.delete(bm, geom=seam, context="FACES_ONLY")
    loops = [e for e in bm.edges if e.is_boundary and e.verts[0] in edge and e.verts[1] in edge]
    filled = bmesh.ops.holes_fill(bm, edges=loops, sides=0)["faces"] if loops else []
    if filled:
        bmesh.ops.triangulate(bm, faces=filled)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    print(f"hands: {mixed} points made all hand or all body, {len(seam)} faces between a hand and the body taken out, "
          f"{len(filled)} holes closed", flush=True)


def wrap_weights(armature, meshes, faces=150000, rounds=40):
    """Each mesh's weights from the bones it wraps around, for --skin wrap (a model in a pose the rig head has not
    learnt). On a lighter copy (all parts as one, at most `faces` faces):
      thickness   each bone's: rays from inside it across it, the median distance to the surface (a hand ~1.5 cm, a
                  thigh ~6 cm on a 1 m model)
      nearest     each point takes the bone it is nearest to in units of that bone's thickness, of those whose surface
                  it is on (its surface faces away from the bone). Plain distance or Blender's bone heat give the
                  side of a hip to a hand resting on it (Mai, Nov 22: her thighs stretched after her hands).
      smoothed    `rounds` times each point half its own, half its neighbours' (along the surface: a hand resting on a
                  hip is not joined to it), so the weights blend over a few cm at the joints
    copied back onto every part, at most 4 bones a point."""
    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    scene = bpy.context.scene
    bpy.ops.object.mode_set(mode="OBJECT")
    copies = []
    for o in meshes:
        c = o.copy()
        c.data = o.data.copy()
        c.modifiers.clear()
        c.vertex_groups.clear()
        c.parent = None
        c.matrix_world = o.matrix_world
        scene.collection.objects.link(c)
        copies.append(c)
    bpy.ops.object.select_all(action="DESELECT")
    for c in copies:
        c.select_set(True)
    bpy.context.view_layer.objects.active = copies[0]
    if len(copies) > 1:
        bpy.ops.object.join()
    proxy = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    n = len(proxy.data.vertices)
    co = np.empty(n * 3)
    proxy.data.vertices.foreach_get("co", co)
    height = float(np.ptp(co.reshape(-1, 3)[:, 2]))
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=height * 1e-6)  # only the seams of the parts: more would weld a resting hand to the thigh
    bpy.ops.object.mode_set(mode="OBJECT")
    if len(proxy.data.polygons) > faces:
        fewer = proxy.modifiers.new("fewer", "DECIMATE")
        fewer.ratio = faces / len(proxy.data.polygons)
        bpy.ops.object.modifier_apply(modifier="fewer")
    mesh = proxy.data
    n = len(mesh.vertices)
    co, normal = np.empty(n * 3), np.empty(n * 3)
    mesh.vertices.foreach_get("co", co)
    mesh.vertices.foreach_get("normal", normal)
    co, normal = co.reshape(-1, 3), normal.reshape(-1, 3)
    edges = np.empty(len(mesh.edges) * 2, np.int64)
    mesh.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    tree = BVHTree.FromObject(proxy, bpy.context.evaluated_depsgraph_get())

    bones = list(armature.data.bones)
    head = np.array([armature.matrix_world @ b.head_local for b in bones])
    tail = np.array([armature.matrix_world @ b.tail_local for b in bones])
    thick = []
    for k in range(len(bones)):
        seg = tail[k] - head[k]
        axis = seg / max(np.linalg.norm(seg), 1e-9)
        u = np.cross(axis, [0, 0, 1.0] if abs(axis[2]) < .9 else [1.0, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        hits = []
        for t in (.2, .35, .5, .65, .8):
            for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                h = tree.ray_cast(Vector(head[k] + t * seg), Vector(np.cos(angle) * u + np.sin(angle) * v), height * .5)
                if h[0] is not None:
                    hits.append(h[3])
        thick.append(max(float(np.median(hits)) if hits else height * .02, height * .004))
    thick = np.array(thick)
    score = np.empty((n, len(bones)), np.float32)
    wraps = np.empty((n, len(bones)), bool)
    facing = np.empty((n, len(bones)), np.float32)  # 1: the surface faces straight away from the bone
    for k in range(len(bones)):
        seg = tail[k] - head[k]
        t = np.clip(((co - head[k]) @ seg) / max(seg @ seg, 1e-12), 0, 1)
        away = co - (head[k] + t[:, None] * seg)
        d = np.linalg.norm(away, axis=1)
        score[:, k] = d / thick[k]
        facing[:, k] = np.einsum("ij,ij->i", away, normal) / np.maximum(d, 1e-9)
        wraps[:, k] = facing[:, k] > -.1
    hand = np.array([b.name.startswith(HAND) for b in bones])
    # a hand, a forearm, a finger: only skin within 1.6 times its thickness that clearly faces away from it (the back
    # of a hand, the sides of a finger; not the hip around a hand pressed into it, which faces sideways to it)
    wraps &= ~(hand[None, :] & ((score > 1.6) | (facing < .3)))
    best = np.where(wraps, score, np.inf).argmin(1)
    alone = ~wraps.any(1)
    best[alone] = score[alone].argmin(1)
    # an arm's skin is what can be walked to from its upper arm over arm skin: a patch of hip that a pressed-in hand
    # takes by distance (Mai) is not, and goes to the nearest other bone
    arm = np.array([b.name.startswith(("upper_arm", "lower_arm", "hand", "thumb", "index", "middle", "ring", "little"))
                    for b in bones])
    on_arm = arm[best]
    # ... and not across a sharp inward fold: a hand pressed into a hip meets it in a crease all around (the arm itself
    # can always be walked around a fold: the outside of the elbow, the back of the wrist)
    step = co[edges[:, 1]] - co[edges[:, 0]]
    turn = normal[edges[:, 1]] - normal[edges[:, 0]]
    bend = np.einsum("ij,ij->i", normal[edges[:, 0]], normal[edges[:, 1]])
    fold = (np.einsum("ij,ij->i", turn, step) < 0) & (bend < np.cos(np.radians(20)))
    open_ = edges[~fold]
    reached = np.isin(best, [k for k, b in enumerate(bones) if b.name.startswith("upper_arm")])
    while True:
        grown = reached.copy()
        grown[open_[:, 1][reached[open_[:, 0]]]] = True
        grown[open_[:, 0][reached[open_[:, 1]]]] = True
        grown &= on_arm
        if grown.sum() == reached.sum():
            break
        reached = grown
    cut_off = on_arm & ~reached
    if cut_off.any():
        other = np.where(wraps & ~arm[None, :], score, np.inf)
        best[cut_off] = np.where(np.isfinite(other[cut_off]).any(1), other[cut_off].argmin(1),
                                 np.where(~arm[None, :], score[cut_off], np.inf).argmin(1))
    print(f"arms: {int(cut_off.sum())} points away from the arms given back to the body ({int(fold.sum())} folds)", flush=True)
    W = np.zeros((n, len(bones)), np.float32)
    W[np.arange(n), best] = 1

    # the smoothing does not cross from a hand to the body: where a hand is fused into a hip it would spread the hand's
    # pull over the whole thigh in small amounts (Mai: the thigh followed the hand by a third)
    body = np.array([b.name.startswith(BODY) for b in bones])
    on_hand, on_body = hand[best], body[best]
    keep = ~((on_hand[edges[:, 0]] & on_body[edges[:, 1]]) | (on_body[edges[:, 0]] & on_hand[edges[:, 1]]))
    edges = edges[keep]
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    degree = np.bincount(src, minlength=n)
    starts = np.concatenate([[0], np.cumsum(degree)[:-1]])
    has = degree > 0
    for _ in range(rounds):
        mean = W.copy()
        mean[has] = np.add.reduceat(W[dst], starts[has], axis=0) / degree[has, None]
        W = .5 * W + .5 * mean

    top = np.argsort(-W, 1)[:, :4]
    weight = np.take_along_axis(W, top, 1)
    weight /= np.maximum(weight.sum(1, keepdims=True), 1e-9)
    weight = np.round(weight, 2)
    for k, b in enumerate(bones):
        group = proxy.vertex_groups.new(name=b.name)
        where, slot = np.nonzero(top == k)
        w = weight[where, slot]
        for q in np.unique(w[w > 0]):
            group.add(where[w == q].tolist(), float(q), "REPLACE")
    print("bone thickness (% of the height): " + ", ".join(f"{b.name} {t / height * 100:.1f}" for b, t in zip(bones, thick)
                                                         if not b.name[:-2].endswith(("1", "2", "3"))), flush=True)

    for o in meshes:
        o.vertex_groups.clear()
        for g in proxy.vertex_groups:
            o.vertex_groups.new(name=g.name)
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        m = o.modifiers.new("copy", "DATA_TRANSFER")
        m.object = proxy
        m.use_vert_data = True
        m.data_types_verts = {"VGROUP_WEIGHTS"}
        m.vert_mapping = "POLYINTERP_NEAREST"
        m.layers_vgroup_select_src = "ALL"
        m.layers_vgroup_select_dst = "NAME"
        bpy.ops.object.modifier_move_to_index(modifier="copy", index=0)
        bpy.ops.object.modifier_apply(modifier="copy")
        bpy.ops.object.vertex_group_limit_total(limit=4)
        bpy.ops.object.vertex_group_normalize_all(lock_active=False)
    bpy.data.objects.remove(proxy)
    print(f"wrap weights ({len(mesh.polygons)} faces, {rounds} rounds of smoothing)", flush=True)


def build(model_path, out, weights_by="head"):
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

    for obj in meshes:
        if obj.data.users == 1 and obj.matrix_world != obj.matrix_world.Identity(4):
            # its place from parent objects (Sketchfab's empties) into the mesh itself: under the armature it keeps it
            obj.data.transform(obj.matrix_world)
            obj.parent = None
            obj.matrix_world = obj.matrix_world.Identity(4)
    if weights_by == "wrap":
        meshes = one_mesh(meshes)
        wrap_weights(armature, meshes)
        free_hands(armature, meshes[0])
    else:  # from vertex_weights (each vertex takes those of its point there)
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
    for obj in meshes:
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
    parser.add_argument("--joints", choices=["head", "views"], default="head",
                        help="head: the rig head's joints (default); views: found on 12 pictures of the model by GEM-X's "
                        "77-point finder (rig_views.py), for poses the rig head has not learnt (hands on the hips)")
    parser.add_argument("--skin", choices=["head", "wrap"],
                        help="head: the rig head's weights; wrap: from the bones each point wraps around (wrap_weights; "
                        "default with --joints views)")
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    skin = args.skin or ("wrap" if args.joints == "views" else "head")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    predict(os.path.abspath(args.model), out, args.checkpoint)
    if args.joints == "views":
        import numpy as np
        folder = out + "_views"
        subprocess.run([BLENDER, "-b", "--python", os.path.join(HERE, "rig_views.py"), "--", "render",
                        os.path.abspath(args.model), folder], check=True)
        subprocess.run([os.path.join(PROJECT_DIR, "conda", "envs", "gemx", "bin", "python"), os.path.join(HERE, "rig_views.py"),
                        "detect", folder], check=True)
        d, found = dict(np.load(out + ".npz")), np.load(os.path.join(folder, "joints.npz"))
        d["gltf_joints"], d["gltf_fingers"] = found["gltf_joints"], found["gltf_fingers"]
        np.savez(out + ".npz", **d)
    subprocess.run([BLENDER, "-b", "--python", os.path.abspath(__file__), "--",
                    os.path.abspath(args.model), out, skin], check=True)


if __name__ == "__main__":
    main()

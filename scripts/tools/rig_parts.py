"""A rigged model made usable: its bones get names (head, hand_left, foot_right, ...), and every loose piece of the
mesh follows ONE bone, so hard parts move as solid pieces.

An automatic rig (UniRig) gives a good skeleton, but its bones are called bone_0 ... and it blends every point between
several bones. That suits a soft body; a robot is hard parts: its arm stretched thin and its head crumpled when a bone
was turned (the toaster robot, Nov 16).

  names    the spine is the chain that goes up from the root; what hangs off it low and ends below the root is a leg,
           what hangs off it high is an arm (the two longest; more are "other": a lantern, a tail). Left and right are
           the character's own, from where its feet point (or, without toes, the model's front as TRELLIS.2 puts it)
  pieces   the mesh's loose pieces (corners at the same place count as one, as in place_part.py). A piece takes the
           bone the rig's weights vote for, if the bone nearest to it agrees or the vote is clear; otherwise the bone
           of the bigger piece it touches. A tiny piece (a finger, a lever) always goes with the piece it touches
  parts    parts.json: every named part with its bones, its joint and its box in the model's glTF coordinates:
           where "the left hand" is, for placing a new part or for the area of an inpainting

    blender -b --python scripts/tools/rig_parts.py -- rigged.glb out.glb          # + out.json
    python scripts/tools/rig_picture.py out                                        # out.jpg: the parts in colours

One fused mesh (what TRELLIS.2 makes) has one piece: it keeps the rig's blended weights, only the names are added.
The whole way from a model: scripts/tools/rig.sh model.glb out_folder
"""
import json
import sys

import bpy
import numpy as np
from mathutils import kdtree

source, out = sys.argv[sys.argv.index("--") + 1:][:2]
stem = out.rsplit(".", 1)[0]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=source)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.vertex_groups]
world = lambda o, p: np.array(o.matrix_world @ p)
bones = list(arm.data.bones)
head = {b.name: world(arm, b.head_local) for b in bones}
kids = {b.name: [c.name for c in b.children] for b in bones}
parent = {b.name: b.parent.name if b.parent else None for b in bones}
verts = np.concatenate([np.array([world(m, v.co) for v in m.data.vertices]) for m in meshes])
low, high = verts.min(0), verts.max(0)
H = float(high[2] - low[2])
UP = np.array([0.0, 0.0, 1.0])


def subtree(name):
    found = [name]
    for c in kids[name]:
        found += subtree(c)
    return found


def far_leaf(name):
    """The longest way down the tree from this bone: (its length, the bones on it)."""
    best = (0.0, [name])
    for c in kids[name]:
        length, path = far_leaf(c)
        length += float(np.linalg.norm(head[c] - head[name]))
        if length > best[0]:
            best = (length, [name] + path)
    return best


# ---------------------------------------------------------------- the names
root = next(b.name for b in bones if not b.parent)
spine = [root]
while kids[spine[-1]]:
    def upward(c):
        step = head[c] - head[spine[-1]]
        if np.linalg.norm(step) < 1e-4 * H and kids[c]:  # a joint at the same place: where it goes next
            step = head[kids[c][0]] - head[c]
        aside = np.linalg.norm((head[c] - head[root])[:2]) / H
        return step[2] / max(np.linalg.norm(step), 1e-9) - 0.5 * aside
    best = max(kids[spine[-1]], key=upward)
    if upward(best) < 0.3:
        break
    spine.append(best)
limbs = []
for index, s in enumerate(spine):
    for c in kids[s]:
        if c not in spine:
            length, chain = far_leaf(c)
            joints = np.array([head[n] for n in subtree(c)])
            limbs.append({"at": index, "chain": chain, "all": subtree(c), "length": length + float(np.linalg.norm(head[c] - head[s])),
                          "low": float(joints[:, 2].min()), "middle": joints.mean(0)})
top = head[spine[-1]][2]
legs = sorted((l for l in limbs if l["low"] < head[root][2] - 0.1 * H and head[spine[l["at"]]][2] < head[root][2] + 0.25 * (top - head[root][2])),
              key=lambda l: -l["length"])[:2]
arms = sorted((l for l in limbs if l not in legs and len(l["chain"]) >= 2 and head[spine[l["at"]]][2] > head[root][2] + 0.25 * (top - head[root][2])),
              key=lambda l: -l["length"])[:2]
front, front_from = np.array([0.0, -1.0, 0.0]), "assumed (the model's front as TRELLIS.2 puts it)"
toes = []
for leg in legs:  # where the feet point: from the ankle to the toe, if the rig has a toe
    if len(leg["chain"]) >= 2:
        step = head[leg["chain"][-1]] - head[leg["chain"][-2]]
        if np.linalg.norm(step[:2]) > 0.6 * np.linalg.norm(step) > 0:
            toes.append(step[:2] / np.linalg.norm(step[:2]))
if toes and np.linalg.norm(np.mean(toes, 0)) > 0.5:
    front, front_from = np.append(np.mean(toes, 0) / np.linalg.norm(np.mean(toes, 0)), 0.0), "from the feet"
LEFT = np.cross(UP, front)  # the character's own left
side = lambda limb: "left" if (limb["middle"] - head[root]) @ LEFT > 0 else "right"
names, roles = {}, {}  # bone -> new name; part -> its bones
arm_at = max((l["at"] for l in arms), default=len(spine) - 1)
for index, s in enumerate(spine):
    part = "head" if index == len(spine) - 1 and index > arm_at else "neck" if index > arm_at else "spine"
    names[s] = part if part == "head" else f"{part}_{index + 1}"
    roles.setdefault(part, []).append(s)
for kind, group, end in (("arm", arms, "hand"), ("leg", legs, "foot")):
    if len(group) == 2 and side(group[0]) == side(group[1]):  # both on one side of the guess: by their own order
        group.sort(key=lambda l: -(l["middle"] - head[root]) @ LEFT)
        sides = ["left", "right"]
    else:
        sides = [side(l) for l in group]
    for limb, where in zip(group, sides):
        chain = limb["chain"]
        last = len(chain) - 1
        if kind == "leg" and last >= 2:  # a toe at the end: the foot is the bone before it
            step = head[chain[-1]] - head[chain[-2]]
            if np.linalg.norm(step[:2]) > 0.6 * np.linalg.norm(step):
                names[chain[-1]] = f"toe_{where}"
                roles.setdefault(f"{end}_{where}", []).append(chain[-1])
                last -= 1
        for k, bone in enumerate(chain[:last]):
            names[bone] = f"{kind}_{where}_{k + 1}"
            roles.setdefault(f"{kind}_{where}", []).append(bone)
        names[chain[last]] = f"{end}_{where}"
        roles.setdefault(f"{end}_{where}", []).insert(0, chain[last])
        extra = [b for b in limb["all"] if b not in chain]
        for k, bone in enumerate(extra):  # what branches off near the end: fingers (or toes)
            near_end = any(p in chain[max(0, last - 1):] for p in (parent[bone], parent.get(parent[bone])))
            names[bone] = f"{'finger' if kind == 'arm' else 'toe'}_{where}_{k + 1}" if near_end else f"{kind}_{where}_extra_{k + 1}"
            roles.setdefault(f"{end}_{where}" if near_end else f"{kind}_{where}", []).append(bone)
other = 0
for limb in limbs:
    if limb not in arms and limb not in legs:
        other += 1
        for k, bone in enumerate(limb["all"]):
            names[bone] = f"other_{other}_{k + 1}"
            roles.setdefault(f"other_{other}", []).append(bone)
for bone in bones:
    names.setdefault(bone.name, f"other_bone_{bone.name}")

# ---------------------------------------------------------------- the pieces
segments = {}  # bone -> its line pieces: from its joint to every child's joint (a leaf: on from its parent's direction)
for b in bones:
    ends = [head[c] for c in kids[b.name]]
    if not ends:
        step = head[b.name] - head[parent[b.name]] if parent[b.name] else UP * 0.05 * H
        ends = [head[b.name] + 0.5 * step]
        if b.name == spine[-1]:  # the head's bone ends at the neck: the head itself goes on up to the model's top
            ends = [np.array([head[b.name][0], head[b.name][1], max(high[2], head[b.name][2] + 0.05 * H)])]
    segments[b.name] = [(head[b.name], e) for e in ends]


def distance(points, a, b):
    ab = b - a
    t = np.clip((points - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


report = {}
for mesh in meshes:
    data = mesh.data
    points = np.array([world(mesh, v.co) for v in data.vertices])
    _, same = np.unique(np.round(points, 5), axis=0, return_inverse=True)
    same = same.reshape(-1)
    link = np.arange(same.max() + 1)

    def find(i):
        while link[i] != i:
            link[i] = link[link[i]]
            i = link[i]
        return i

    for edge in data.edges:
        a, b = find(same[edge.vertices[0]]), find(same[edge.vertices[1]])
        if a != b:
            link[a] = b
    piece = np.array([find(i) for i in same])
    labels, piece = np.unique(piece, return_inverse=True)
    groups = [g.name for g in mesh.vertex_groups]
    if len(labels) < 2 or np.bincount(piece).max() > 0.5 * len(points):
        # one fused surface (what TRELLIS.2 makes), perhaps with a few loose specks: nothing to make rigid; all of it on
        # one bone would freeze the whole figure (the warrior, Nov 16)
        report[mesh.name] = "one fused piece: the rig's own blended weights stay"
        continue
    weights = np.zeros((len(points), len(groups)))
    for v in data.vertices:
        for g in v.groups:
            weights[v.index, g.group] = g.weight
    tree = kdtree.KDTree(len(points))
    for i, p in enumerate(points):
        tree.insert(p, i)
    tree.balance()
    sizes = np.bincount(piece)
    chosen, how = {}, {"vote": 0, "touching": 0, "nearest": 0, "lying beside it": 0}
    rng = np.random.default_rng(0)
    # which pieces touch: groups of them. A group that lies low and reaches no higher than a tenth of the model is a
    # thing on the floor beside the character (a knife, a piece of butter): it goes with the root, not with a foot
    clump = np.arange(len(labels))

    def top(i):
        while clump[i] != i:
            clump[i] = clump[clump[i]]
            i = clump[i]
        return i

    for p in range(len(labels)):
        mine = np.nonzero(piece == p)[0]
        for i in rng.choice(mine, min(150, len(mine)), replace=False):
            for _, j, _ in tree.find_range(points[i], 0.012 * H):
                a, b = top(p), top(piece[j])
                if a != b:
                    clump[a] = b
    clump = np.array([top(p) for p in range(len(labels))])
    reach = {g: float(points[np.isin(piece, np.nonzero(clump == g)[0])][:, 2].max()) for g in np.unique(clump)}
    for p in np.argsort(-sizes):
        mine = np.nonzero(piece == p)[0]
        if reach[clump[p]] < low[2] + 0.1 * H and root in groups:
            chosen[p] = root
            how["lying beside it"] += 1
            for g in mesh.vertex_groups:
                g.remove(mine.tolist())
            mesh.vertex_groups[root].add(mine.tolist(), 1.0, "REPLACE")
            continue
        sample = points[rng.choice(mine, min(400, len(mine)), replace=False)]
        votes = weights[mine].sum(0)
        vote, share = groups[int(votes.argmax())], float(votes.max() / max(votes.sum(), 1e-9))
        near = min(segments, key=lambda n: min(float(distance(sample, a, b).mean()) for a, b in segments[n]))
        touch = {}  # the bones of the bigger pieces this one touches, by how much
        for q in sample:
            for _, i, _ in tree.find_range(q, 0.012 * H):
                if piece[i] in chosen and piece[i] != p:
                    touch[chosen[piece[i]]] = touch.get(chosen[piece[i]], 0) + 1
        # small by its size in space, not by its corners (a lever is finely modelled: many corners)
        tiny = float(np.linalg.norm(np.ptp(points[mine], 0))) < 0.12 * H
        part_of = {b: part for part, members in roles.items() for b in members}
        same_part = part_of.get(vote, vote) == part_of.get(near, near)
        if touch and tiny:
            chosen[p], key = max(touch, key=touch.get), "touching"
        elif vote == near or same_part or near not in groups:
            chosen[p], key = vote, "vote"
        else:  # the vote and the nearest bone disagree: the one whose part this piece touches more; else the nearest
            by_part = {}
            for b, n in touch.items():
                by_part[part_of.get(b, b)] = by_part.get(part_of.get(b, b), 0) + n
            v, n = by_part.get(part_of.get(vote, vote), 0), by_part.get(part_of.get(near, near), 0)
            chosen[p], key = (vote, "touching") if v > n else (near, "touching" if n > v else "nearest")
        how[key] += 1
        for group in mesh.vertex_groups:
            group.remove(mine.tolist())
        mesh.vertex_groups[chosen[p]].add(mine.tolist(), 1.0, "REPLACE")
    report[mesh.name] = f"{len(labels)} pieces, each on one bone ({how})"
    piece_bone = np.array([chosen[p] for p in piece])

# ---------------------------------------------------------------- written down, and the bones renamed
gltf = lambda p: [float(p[0]), float(p[2]), float(-p[1])]  # Blender (Z up) -> the model's glTF axes (Y up)
parts = {}
for part, members in roles.items():
    entry = {"bones": [names[b] for b in members], "joint": gltf(head[members[0]])}
    if len(meshes) == 1 and "piece_bone" in dir():
        owned = points[np.isin(piece_bone, members)]
        if len(owned):
            entry["box"] = [gltf(owned.min(0)), gltf(owned.max(0))]
            entry["box"] = [np.minimum(*entry["box"]).tolist(), np.maximum(*entry["box"]).tolist()]
    parts[part] = entry
for mesh in meshes:
    for group in mesh.vertex_groups:
        group.name = "tmp_" + group.name
    for group in mesh.vertex_groups:
        group.name = names[group.name[4:]]
for bone in bones:
    bone.name = "tmp_" + bone.name
for bone in bones:
    bone.name = names[bone.name[4:]]
json.dump({"source": source, "front": front_from, "pieces": report, "parts": parts,
           "bones": {names[b]: {"joint": gltf(head[b]), "parent": names[parent[b]] if parent[b] else None} for b in head}},
          open(stem + ".json", "w"), indent=1)
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB")
print("names:", {k: v["bones"] for k, v in parts.items()}, flush=True)
print("pieces:", report, "| front:", front_from, flush=True)

# what rig_picture.py draws (Blender's Python has no PIL)
np.savez(stem + "_draw.npz", points=points if "piece_bone" in dir() else verts, low=low, high=high, front=front, left=LEFT,
         part=np.array([{b: part for part, members in roles.items() for b in members}.get(b, "") for b in piece_bone])
         if "piece_bone" in dir() else np.array([]), parts=np.array(list(roles)),
         joints=np.array([head[b] for b in head]), joint_names=np.array([names[b] for b in head]),
         lines=np.array([[a, e] for b in head for a, e in segments[b]]))

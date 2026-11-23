"""(An experiment, Nov 23, worse than without: the knight's raised sleeve still made an arc and tore into strips,
and the decimated copy showed dark specks at its texture seams. A generated mesh is one fused surface, sleeve and arm
in one: the cloth tears where pinned meets free. The fix is the sleeve as its own part first: body / cloth labels.)

Loose sleeves (and other cloth that hangs off the arms) as cloth: they fall and swing instead of flying up with
the arm as one stiff piece (the Plague Knight's bell sleeves, Nov 23: raised with his axe, each became a big arc).

On an animated .blend (hold_item.py --swing, motion_to_rig.py):
  1 lighter       the character decimated to --faces (a game's LOD; the cloth solver is slow on a million faces)
  2 pinned        how far each point hangs off its arm: its distance to the nearest arm bone, in units of the arm's
                  own thickness there (the nearest fifth of the points the bone carries most: the arm's core). Up to 1.3: pinned
                  to the bones (the arm itself), from 2.6 on: free cloth, smoothly between ("pin" group)
  3 cloth         Blender's cloth on top of the armature (cotton-like, gravity), colliding with itself; baked over
                  the animation's frames

    blender -b --python scripts/rigging/cloth_sleeves.py -- animated.blend out.blend [--faces 60000]
"""
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

ARM = ("upper_arm", "lower_arm", "upper_arm_twist", "lower_arm_twist", "hand")


def arm_bones(rig):
    return [b.name for b in rig.data.bones if b.name.rsplit("_", 1)[0] in ARM]


def segment_distance(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    src, out = a[0], a[1]
    faces = int(a[a.index("--faces") + 1]) if "--faces" in a else 60000
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(src))
    s = bpy.context.scene
    rig = next(o for o in s.objects if o.type == "ARMATURE")
    body = [o for o in s.objects if o.type == "MESH" and o.parent == rig and o.parent_type != "BONE"]
    s.frame_set(s.frame_start)
    for obj in body:
        # 1 lighter: decimate first (before the armature), the weights carried along
        bpy.context.view_layer.objects.active = obj
        dec = obj.modifiers.new("lighter", "DECIMATE")
        dec.ratio = min(1.0, faces / max(len(obj.data.polygons), 1))
        while obj.modifiers[0].name != "lighter":
            bpy.ops.object.modifier_move_up(modifier="lighter")
        bpy.ops.object.modifier_apply(modifier="lighter")
        # 2 pinned: by how far each point hangs off its arm, in the arm's own thickness
        rest = np.array([v.co[:] for v in obj.data.vertices])
        groups = {g.name: g.index for g in obj.vertex_groups}
        bones = [n for n in arm_bones(rig) if n in groups]
        dist = np.full(len(rest), np.inf)
        thick = np.full(len(rest), 1.0)
        for name in bones:
            bone = rig.data.bones[name]
            h, t = np.array(rig.matrix_world @ bone.head_local), np.array(rig.matrix_world @ bone.tail_local)
            d = segment_distance(rest, h, t)
            gi = groups[name]
            carried = np.array([any(g.group == gi and g.weight > 0.5 for g in v.groups) for v in obj.data.vertices])
            r = np.percentile(d[carried], 20) if carried.sum() > 20 else np.percentile(d, 5)  # the arm's core,
            # not the median (a bell sleeve is most of what the arm carries: it counted as arm and stayed pinned)
            closer = d < dist
            dist[closer], thick[closer] = d[closer], r
        reach = dist / np.maximum(thick, 1e-6)
        arm_ids = {groups[n] for n in bones}
        arm_weight = np.array([sum(g.weight for g in v.groups if g.group in arm_ids) for v in obj.data.vertices])
        on_arm = np.isfinite(reach) & (arm_weight > 0.5)  # only what the arms carry can hang off them
        free = np.clip((reach - 1.3) / (2.6 - 1.3), 0, 1) * on_arm
        pin = obj.vertex_groups.new(name="pin")
        for k in range(len(rest)):
            pin.add([k], float(1 - free[k]), "REPLACE")
        print(f"{obj.name}: {len(obj.data.polygons)} faces, {int((free > 0.5).sum())} points free cloth", flush=True)
        # 3 cloth after the armature
        cloth = obj.modifiers.new("cloth", "CLOTH")
        c = cloth.settings
        c.quality = 6
        c.mass = 0.15
        c.tension_stiffness = c.compression_stiffness = 8
        c.bending_stiffness = 0.5
        c.air_damping = 1.5
        c.vertex_group_mass = "pin"
        c.pin_stiffness = 1.0
        cloth.collision_settings.use_self_collision = False
        cloth.collision_settings.distance_min = 0.002
        cloth.point_cache.frame_start, cloth.point_cache.frame_end = s.frame_start, s.frame_end
    for obj in body:  # bake
        for m in obj.modifiers:
            if m.type == "CLOTH":
                with bpy.context.temp_override(scene=s, active_object=obj, point_cache=m.point_cache):
                    bpy.ops.ptcache.bake(bake=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out))
    print("cloth:", out, flush=True)


main()

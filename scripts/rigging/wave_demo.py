"""A rigged model (rig_from_glb.py) as a .blend that is ready to play: the bones visible (sticks, in front of the
body), in Pose Mode, and a 4-second wave of the right arm (press Space in Blender).

    blender -b --python scripts/rigging/wave_demo.py -- model_rigged.glb model_wave.blend

The bones are turned about the world's axes (a glTF's bones have their own, turned ones).
"""
import math, sys, bpy
glb, out = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
rig = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
rig.name = "rig"
rig.show_in_front = True
rig.data.display_type = "STICK"
scene = bpy.context.scene
scene.frame_start, scene.frame_end = 1, 96
scene.render.fps = 24
from mathutils import Quaternion, Vector
def key(bone, frame, angles):  # angles about the world's X, Y, Z (Blender: X right of the viewer, -Y front, Z up)
    b = rig.pose.bones[bone]
    b.rotation_mode = "QUATERNION"
    world = Quaternion()
    for axis, a in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)), angles):
        world = Quaternion(Vector(axis), math.radians(a)) @ world
    rest = b.bone.matrix_local.to_quaternion()
    b.rotation_quaternion = rest.inverted() @ world @ rest
    b.keyframe_insert("rotation_quaternion", frame=frame)
for f, up, wave in ((1, 0, 0), (24, -70, 0), (36, -70, 25), (48, -70, -25), (60, -70, 25), (72, -70, 0), (96, 0, 0)):
    key("upper_arm_r", f, (0, -up * 1.7, 0))          # the character's right arm (at -X) up beside the head
    key("lower_arm_r", f, (0, wave, 0))                 # the forearm waves
    key("head", f, (0, 0, 20 if up else 0))             # and the head looks at it
cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera")); scene.collection.objects.link(cam)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="POSE")
bpy.ops.wm.save_as_mainfile(filepath=out)
print("saved", out)

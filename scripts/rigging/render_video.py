"""A rigged, animated character as a video in its own colours (from the front-left, a little above), plus the
pictures of frames 1, 16 and 26 (hold_item.py --swing: held, raised, chopped).

    blender -b --python scripts/rigging/render_video.py -- animated.blend out     # out.mp4, out_frame01.png ...
"""
import bpy, sys, math
from mathutils import Vector
blend, out = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.open_mainfile(filepath=blend)
s = bpy.context.scene
rig = next(o for o in s.objects if o.type == "ARMATURE"); rig.hide_render = True
meshes = [o for o in s.objects if o.type == "MESH"]
import numpy as np
zs = np.concatenate([[(o.matrix_world @ v.co).z for v in o.data.vertices] for o in meshes]); lo, hi = np.percentile(zs, 0.5), np.percentile(zs, 99.5)
s.render.engine = "BLENDER_EEVEE_NEXT"; s.render.resolution_x, s.render.resolution_y = 720, 720
if not s.world: s.world = bpy.data.worlds.new("w")
s.world.use_nodes = True; s.world.node_tree.nodes["Background"].inputs[1].default_value = 1.2
sun = bpy.data.objects.new("sun2", bpy.data.lights.new("sun2", "SUN")); sun.data.energy = 3.5; s.collection.objects.link(sun); sun.rotation_euler = (math.radians(50), 0, math.radians(-35))
cam = bpy.data.objects.new("vcam", bpy.data.cameras.new("vcam")); s.collection.objects.link(cam); s.camera = cam
cam.data.type = "ORTHO"; cam.data.ortho_scale = (hi - lo) * 1.9
d = Vector((-0.8, -1, 0.25)).normalized(); c = Vector((0, 0, (lo + hi) / 2 + 0.1 * (hi - lo)))
cam.location = c + d * 5; cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
s.frame_start, s.frame_end = 1, 40; s.render.fps = 24
s.render.image_settings.file_format = "FFMPEG"; s.render.ffmpeg.format = "MPEG4"; s.render.ffmpeg.codec = "H264"
s.render.filepath = out + ".mp4"; bpy.ops.render.render(animation=True)
s.render.image_settings.file_format = "PNG"
for f in (1, 16, 26):
    s.frame_set(f); s.render.filepath = f"{out}_frame{f:02d}.png"; bpy.ops.render.render(write_still=True)

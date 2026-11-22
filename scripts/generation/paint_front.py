"""Paints the picture itself onto a Pixal3D model where its camera saw it: the front then has exactly the picture's
colours (face, lips, emblems), the model's own colours stay where the picture can't see (sides, back).

Pixal3D (compare_pixal3d.py) makes the model in its camera's frame and writes the camera (field of view, distance) to
<name>.json, so each texel of the model's texture can be looked up in the picture:
  1. the texel's place and facing on the model (the texture's triangles drawn in UV space, nvdiffrast)
  2. seen? its depth against the depth the camera sees there (the model drawn from the camera)
  3. its colour = the picture's, blended by how straight the surface faces the camera (none from 70 degrees off on)

    python scripts/generation/paint_front.py outputs/x/F_pixal3d.glb --picture outputs/x/F_pixal3d_input_rgba.png \
        --camera outputs/x/F_pixal3d.json --out outputs/x/F_pixal3d_painted.glb

The camera: Pixal3D's, from the front (CAMERA_TURN). Printed as a check: how well the model's outline through it lies
on the picture's (about 75 % here: the picture's own outline is soft and the model not exact). (Nov 22: finding the
axes by the outline alone chose the camera behind the model, mirrored, and painted the face onto the back.)
"""
import argparse
import json
import math

import numpy as np
import torch
import trimesh
from PIL import Image, ImageFilter

TO_GLTF = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # compare_pixal3d.py's turn of the model


def project(points, turn, fov, distance):
    """Model points (cube frame) -> picture coordinates 0..1 (x right, y down) and depth, Pixal3D's camera:
    world = turn @ p, the camera on -y at `distance`, looking along +y, z up (inference.distance_from_fov)."""
    w = points @ turn.T
    depth = distance + w[:, 1]
    focal = 0.5 / math.tan(fov / 2)  # in picture widths
    return np.stack([0.5 + focal * w[:, 0] / depth, 0.5 - focal * w[:, 2] / depth], 1), depth


# Pixal3D's camera looks at the model's front from (0, 0, +distance) with Y up (render_utils.
# proj_camera_to_render_params), the front being glTF's +Z after compare_pixal3d.py's turn. In project()'s terms:
# world x = glTF x, world y (away from the camera) = -glTF z, world z (up) = glTF y; from the cube frame:
CAMERA_TURN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)


def outline_overlap(points, alpha, turn, fov, distance, size=160):
    """How well the model's outline through the camera lies on the picture's (a check that the camera is right;
    a front and a back view have nearly the same outline, so it can't choose between them)."""
    from PIL import ImageFilter
    target = np.asarray(Image.fromarray(alpha).resize((size, size))) > 127
    sample = points[np.random.default_rng(0).choice(len(points), min(len(points), 200_000), replace=False)]
    uv, depth = project(sample, turn, fov, distance)
    ok = (depth > 0) & (uv.min(1) >= 0) & (uv.max(1) < 1)
    seen = np.zeros((size, size), bool)
    seen[(uv[ok, 1] * size).astype(int), (uv[ok, 0] * size).astype(int)] = True
    seen = np.asarray(Image.fromarray(seen).filter(ImageFilter.MaxFilter(3)))
    return (seen & target).sum() / max((seen | target).sum(), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model")
    parser.add_argument("--picture", required=True, help="the picture as Pixal3D saw it, with its outline "
                        "(<name>_input_rgba.png; or <name>_colour_input_rgba.png for the colour picture)")
    parser.add_argument("--camera", required=True, help="<name>.json of compare_pixal3d.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--from-angle", type=float, default=50, help="full picture colour up to this angle (deg)")
    parser.add_argument("--to-angle", type=float, default=72, help="none of it from this angle on (deg)")
    args = parser.parse_args()
    import nvdiffrast.torch as dr

    camera = json.load(open(args.camera))["camera"]
    fov, distance = math.radians(camera["fov_degrees"]), camera["distance"]
    picture = Image.open(args.picture)
    if picture.mode != "RGBA":  # no outline kept (older runs): the background is black
        alpha = Image.fromarray(((np.asarray(picture.convert("RGB")).max(2) > 6) * 255).astype(np.uint8))
        picture = picture.convert("RGB")
        picture.putalpha(alpha)
    rgba = np.asarray(picture)
    # the picture's edge is half background: only what lies well inside its outline
    alpha = np.asarray(Image.fromarray(rgba[..., 3]).filter(ImageFilter.MinFilter(5))).astype(np.float32) / 255

    scene = trimesh.load(args.model)
    name, mesh = next(iter(scene.geometry.items()))
    material = mesh.visual.material
    texture = np.asarray(material.baseColorTexture.convert("RGBA")).astype(np.float32) / 255
    size = texture.shape[0]
    to_world = scene.graph.get(scene.graph.nodes_geometry[0])[0]  # the mesh's place in the scene
    vertices = trimesh.transform_points(mesh.vertices, to_world) @ np.linalg.inv(TO_GLTF).T  # back to Pixal3D's cube
    normals = mesh.vertex_normals @ to_world[:3, :3].T @ np.linalg.inv(TO_GLTF).T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-9)

    turn = CAMERA_TURN
    print(f"camera check: outline overlap {outline_overlap(vertices, rgba[..., 3], turn, fov, distance):.1%}",
          flush=True)

    glctx = dr.RasterizeCudaContext()
    cuda = lambda a, t=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=t, device="cuda")
    faces = cuda(mesh.faces, torch.int32)

    # 1. the model from the camera: its depth per pixel
    uv_cam, depth = project(vertices, turn, fov, distance)
    near, far = depth.min() * 0.5, depth.max() * 2
    z = (depth - near) / (far - near) * 2 - 1
    clip = np.stack([(uv_cam[:, 0] * 2 - 1) * depth, (1 - uv_cam[:, 1] * 2) * depth, z * depth, depth], 1)
    res = picture.width
    rast, _ = dr.rasterize(glctx, cuda(clip)[None], faces, (res, res))
    depth_map, _ = dr.interpolate(cuda(depth[:, None]), rast, faces)
    depth_map = torch.where(rast[..., 3:] > 0, depth_map, torch.full_like(depth_map, 1e9))[0, ..., 0].flip(0)

    # 2. every texel: its place and facing (the triangles drawn in UV space)
    uv = mesh.visual.uv
    clip_uv = np.stack([uv[:, 0] * 2 - 1, uv[:, 1] * 2 - 1, np.zeros(len(uv)), np.ones(len(uv))], 1)
    rast_uv, _ = dr.rasterize(glctx, cuda(clip_uv)[None], faces, (size, size))
    covered = (rast_uv[0, ..., 3] > 0).flip(0)  # texture rows go top to bottom, UV v bottom to top
    place, _ = dr.interpolate(cuda(vertices), rast_uv, faces)
    facing, _ = dr.interpolate(cuda(normals), rast_uv, faces)
    place, facing = place[0].flip(0)[covered].cpu().numpy(), facing[0].flip(0)[covered].cpu().numpy()

    # 3. seen, and how straight
    uv_tex, depth_tex = project(place, turn, fov, distance)
    inside = (uv_tex.min(1) >= 0) & (uv_tex.max(1) < 1)
    px = np.clip((uv_tex * res).astype(int), 0, res - 1)
    seen = inside & (depth_tex <= depth_map.cpu().numpy()[px[:, 1], px[:, 0]] + 0.01)
    eye = (np.linalg.inv(turn) @ np.array([0, -distance, 0]))  # the camera's place in the cube frame
    view = eye - place
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    facing /= np.linalg.norm(facing, axis=1, keepdims=True).clip(1e-9)
    cos = (facing * view).sum(1)
    lo, hi = math.cos(math.radians(args.to_angle)), math.cos(math.radians(args.from_angle))
    t = np.clip((cos - lo) / (hi - lo), 0, 1)
    weight = t * t * (3 - 2 * t) * seen * alpha[px[:, 1], px[:, 0]]
    colour = rgba[px[:, 1], px[:, 0], :3].astype(np.float32) / 255

    texels = texture[covered.cpu().numpy()]
    texels[:, :3] = texels[:, :3] * (1 - weight[:, None]) + colour * weight[:, None]
    texture[covered.cpu().numpy()] = texels
    material.baseColorTexture = Image.fromarray((texture * 255).round().astype(np.uint8))
    scene.export(args.out, extension_webp=True)
    print(f"painted {(weight > 0.5).sum() / len(weight):.1%} of the texels from the picture -> {args.out}")


if __name__ == "__main__":
    main()

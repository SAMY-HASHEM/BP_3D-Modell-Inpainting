"""Multi-view Pixal3D: one model from several views of a character sheet (front, 3/4, back, 3/4 ...), to compare with
TRELLIS.2's multi_image.py and single-view Pixal3D (compare_pixal3d.py).

Pixal3D's multi-view weights (ckpts/*_mv, pipeline_mv.json) put each view's pixels onto the voxels its camera sees,
so every view needs its camera. A sheet doesn't say them; they are made here:
  turn       each view's angle around the character (degrees, 0 = front, 90 = seen from her left, 180 = back): given
             as picture:angle, or picture:auto = found by turning a model of her (--fit, e.g. TRELLIS.2's 4-view one)
             until its outline fits the view's best
  framing    all views of one sheet are rendered the same size, so all get the same square crop: 1.1 x the front
             figure's height, centred on each figure; the camera is as far as Pixal3D's single view puts it for
             that framing (the cube's edge at the picture's edge)
  field of view   MoGe-2's estimate on the front view (or --fov)

    source activate.sh
    python scripts/generation/compare_pixal3d_mv.py grey_front.png:0 grey_a.png:auto grey_back.png:180 grey_b.png:auto \
        --colour colour_front.png colour_a.png colour_back.png colour_b.png --fit P3.glb --name P7 --out outputs/x

--colour: the colour step sees these views instead (the same order and angles: recommendation 1, the shape from grey
views, the colours from coloured ones); they are framed the same way, by their own sheet's front figure.
Writes <name>.glb (glTF axes, front +Z, like compare_pixal3d.py), <name>.json, <name>_views/ (the framed views,
transforms.json) and, for paint_front.py, the front view with its outline (<name>_input_rgba.png).
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_pixal3d as single  # its set-up: paths, NAF from our copy, DINOv3 local, MoGe-2  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import inference_mv as pixal_mv  # noqa: E402  (Pixal3D's own multi-view script, on single's sys.path)


def lean_projection():
    """Pixal3D's multi-view projection, the same numbers in less GPU memory (for the 24 GB PC: at 1536 the grid is
    96^3 and one view's features are 2 x 3.4 GB; its code glued them into a new 6.75 GB block and added that to a
    6.75 GB sum: ~20 GB at once, out of memory on 24 GB, Nov 23). Here each half goes straight into its part of the
    sum and is freed: ~10 GB. Only the 'average' fusion (the released weights' own); 'attention' stays as it was."""
    import torch
    from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj as icp
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import compute_relative_calc_mat
    original = icp.DinoV3ProjMultiViewFeatureExtractor.forward

    def forward(self, image, camera_angle_x=None, distance=None, mesh_scale=None, transform_matrix=None):
        if self.multiview_fusion != "average":
            return original(self, image, camera_angle_x, distance, mesh_scale, transform_matrix)
        B, V = image.shape[:2]
        calc_mat = compute_relative_calc_mat(transform_matrix, distance, self.proj_grid.front_view_transform_matrix)
        img_flat = image.reshape(B * V, *image.shape[2:])
        image_for_naf = img_flat.clone() if self.use_naf_upsample else None
        img_norm = self.transform(img_flat)
        with torch.no_grad():
            z = self.extract_features(img_norm)
            num_reg = getattr(self.model.config, "num_register_tokens", 4)
            z_global_all = z[:, :1 + num_reg]
            spatial = z[:, 1 + num_reg:].reshape(B * V, self.patch_number, self.patch_number, -1)
            fov, dist = camera_angle_x.reshape(B * V), distance.reshape(B * V)
            scale = mesh_scale[:, None].expand(B, V).reshape(B * V)
            calc = calc_mat.reshape(B * V, 4, 4)
            acc = None
            for v in range(V):
                idx = torch.arange(v, B * V, V, device=z.device)
                lr = self.proj_grid(spatial[idx], fov[idx], dist[idx], scale[idx], calc[idx])
                d = lr.shape[-1]
                if acc is None:
                    acc = torch.empty(*lr.shape[:-1], d * (2 if self.use_naf_upsample else 1), dtype=lr.dtype,
                                      device=lr.device)
                    acc[..., :d].copy_(lr)
                else:
                    acc[..., :d].add_(lr)
                del lr
                if self.use_naf_upsample:
                    self._load_naf()
                    hr_feat = self.naf_model(image_for_naf[idx], spatial[idx].permute(0, 3, 1, 2), self.naf_target_size)
                    hr = self.proj_grid(hr_feat, fov[idx], dist[idx], scale[idx], calc[idx], BHWC=False)
                    del hr_feat
                    if v == 0:
                        acc[..., d:].copy_(hr)
                    else:
                        acc[..., d:].add_(hr)
                    del hr
                torch.cuda.empty_cache()
            z_global = z_global_all.reshape(B, V, z_global_all.shape[-2], z_global_all.shape[-1]).mean(dim=1)
        return z_global, acc.div_(V)

    icp.DinoV3ProjMultiViewFeatureExtractor.forward = forward


def cut(pipeline, path):
    """The view without its background (RGBA, uncropped)."""
    if getattr(pipeline, "low_vram", False):  # it waits on the CPU (the 24 GB profile)
        pipeline.rembg_model.to("cuda")
    rgba = pipeline.rembg_model(Image.open(path).convert("RGB")).convert("RGBA")
    if getattr(pipeline, "low_vram", False):
        pipeline.rembg_model.cpu()
        torch.cuda.empty_cache()
    return rgba


def box(rgba):
    ys, xs = np.nonzero(np.asarray(rgba)[..., 3] > 127)
    return xs.min(), ys.min(), xs.max(), ys.max()


def frame(views, size=1024):
    """The same square crop for all views of one sheet (1.1 x the front figure's height), centred on each figure
    across, at the front figure's height up and down."""
    x0, y0, x1, y1 = box(views[0])
    side = int((y1 - y0) * 1.1)
    cy = (y0 + y1) / 2
    framed = []
    for v in views:
        bx0, _, bx1, _ = box(v)
        cx = (bx0 + bx1) / 2
        framed.append(v.crop((round(cx - side / 2), round(cy - side / 2), round(cx + side / 2),
                              round(cy + side / 2))).resize((size, size), Image.LANCZOS))
    return framed


def outline(points, size=128):
    """A point cloud's outline seen from +Z (x right, y up), fitted to its own box."""
    p = points[:, :2]
    low, high = p.min(0), p.max(0)
    span = (high - low).max()
    q = ((p - (low + high) / 2) / span * 0.95 + 0.5) * (size - 1)
    image = np.zeros((size, size), bool)
    image[(size - 1 - q[:, 1]).astype(int), q[:, 0].astype(int)] = True
    from scipy.ndimage import binary_closing
    return binary_closing(image, iterations=2)


def view_outline(rgba, size=128):
    a = np.asarray(rgba)[..., 3] > 127
    ys, xs = np.nonzero(a)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    span = max(h, w) / 0.95
    canvas = Image.new("L", (round(span), round(span)))
    canvas.paste(Image.fromarray((a * 255).astype(np.uint8)), (round((span - w) / 2), round((span - h) / 2)))
    return np.asarray(canvas.resize((size, size))) > 127


def find_angle(model_path, rgba):
    """The turn (degrees) at which the model's outline fits the view's best: the camera's angle around her."""
    import trimesh
    mesh = trimesh.load(model_path, force="mesh")
    points = mesh.sample(150_000) if len(mesh.faces) else np.asarray(mesh.vertices)
    target = view_outline(rgba)
    best = (-1, 0)
    for angle in range(-90, 91, 3):
        t = math.radians(-angle)  # the camera at `angle` from the front towards her left = the model turned by -angle
        turned = points @ np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]]).T
        o = outline(turned)
        iou = (o & target).sum() / max((o | target).sum(), 1)
        if iou > best[0]:
            best = (iou, angle)
    return best[1], best[0]


def camera_to_world(angle, distance):
    """Pixal3D's camera pose (Blender: looks along its -Z, up its +Y; world Z up, the front camera at -Y)."""
    a = math.radians(angle)
    m = np.eye(4)
    m[:3, 0] = [math.cos(a), math.sin(a), 0]
    m[:3, 1] = [0, 0, 1]
    m[:3, 2] = [math.sin(a), -math.cos(a), 0]
    m[:3, 3] = [distance * math.sin(a), -distance * math.cos(a), 0]
    return m


def write_views(folder, framed, angles, fov, distance):
    os.makedirs(folder, exist_ok=True)
    frames = []
    for k, (v, angle) in enumerate(zip(framed, angles)):
        name = f"view{k:02d}_azim{angle:+04d}.png"
        v.save(os.path.join(folder, name))
        frames.append({"file_path": name, "name": f"azim{angle}",
                       "transform_matrix": camera_to_world(angle, distance).round(6).tolist()})
    json.dump({"camera_angle_x": fov, "mesh_scale": 1.0, "frames": frames},
              open(os.path.join(folder, "transforms.json"), "w"), indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("views", nargs="+", help="picture:angle (degrees; 0 = front, first) or picture:auto")
    parser.add_argument("--colour", nargs="+", help="the same views in colour, for the colour step only")
    parser.add_argument("--fit", help="a model of her (glTF axes) to find the 'auto' angles by")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", default=None, choices=["1024", "1536"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rig", action="store_true", help="also rig it (a humanoid): <name>_rigged.glb, <name>_pose.png "
                        "(Pixal3D's cube is turned to its camera, so the finished model is read in again for it)")
    parser.add_argument("--fov", type=float, default=-1, help="radians; default: MoGe-2's on the front view")
    args = parser.parse_args()
    args.resolution = args.resolution or single.setting("pixal3d_mv", "resolution", "1024")
    os.makedirs(args.out, exist_ok=True)
    path = lambda ext: os.path.join(args.out, args.name + ext)
    pictures = [v.rsplit(":", 1)[0] for v in args.views]
    given = [v.rsplit(":", 1)[1] for v in args.views]
    report = {"views": args.views, "colour": args.colour, "seed": args.seed, "resolution": args.resolution}

    for config in pixal_mv.IMAGE_COND_CONFIGS.values():
        config["model_name"] = single.DINOV3
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    low_vram = single.setting("pixal3d", "low_vram", False)  # the machine's profile (scripts/hardware.py)
    if single.setting("pixal3d_mv", "lean_projection", True):
        lean_projection()
    pipeline = pixal_mv.init_pipeline(single.PIXAL3D, "pipeline_mv.json", low_vram=low_vram)
    report["low_vram"] = low_vram
    report["load_seconds"] = round(time.time() - start)

    views = frame([cut(pipeline, p) for p in pictures])
    views[0].save(path("_input_rgba.png"))
    angles = []
    for v, g in zip(views, given):
        if g == "auto":
            angle, fit = find_angle(args.fit, v)
            print(f"  view angle found: {angle} degrees (outline overlap {fit:.0%})", flush=True)
            angles.append(angle)
        else:
            angles.append(int(g))
    report["angles"] = angles

    if args.fov > 0:
        fov = args.fov
    else:
        from moge.model.v2 import MoGeModel
        moge = MoGeModel.from_pretrained(single.MOGE).cuda().eval()
        with torch.no_grad():
            front = single.on_black(views[0])
            pixels = torch.from_numpy(np.asarray(front, np.float32) / 255).permute(2, 0, 1).cuda()
            fx = float(moge.infer(pixels)["intrinsics"].squeeze()[0, 0]) * front.width
        fov = 2 * math.atan(front.width / (2 * fx))
        del moge
        torch.cuda.empty_cache()
    distance = 0.5 / math.tan(fov / 2)  # the cube's edge (0.5 from its middle) at the picture's edge
    report["camera"] = {"fov_degrees": round(math.degrees(fov), 2), "distance": round(distance, 3)}
    print("camera:", report["camera"], "angles:", angles, flush=True)

    write_views(path("_views"), views, angles, fov, distance)
    bundle = pixal_mv.load_views(path("_views"))
    pixal_mv.check_main_view(bundle)
    if args.colour:
        colour = frame([cut(pipeline, p) for p in args.colour])
        colour[0].save(path("_colour_input_rgba.png"))
        write_views(path("_colour_views"), colour, angles, fov, distance)
        colour_bundle = pixal_mv.load_views(path("_colour_views"))
        tex_model, own = pipeline.image_cond_model_tex_1024, pipeline.get_proj_cond_shape

        def colour_for_texture(model, given_views, *a, **k):  # only the colour step sees the coloured views
            return own(model, colour_bundle if model is tex_model else given_views, *a, **k)
        pipeline.get_proj_cond_shape = colour_for_texture

    start = time.time()
    torch.manual_seed(args.seed)
    meshes, (_, _, res) = pipeline.run_mv(bundle, seed=args.seed, return_latent=True,
                                          pipeline_type=f"{args.resolution}_cascade")
    report["seconds"] = round(time.time() - start)
    mesh = meshes[0]
    single.free_gpu(pipeline)  # the mesh step at 1536 needs the room the models take (it ran out of 46 GB, Nov 22)
    mesh.simplify(16777216)
    torch.cuda.empty_cache()
    import o_voxel
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout, grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1_000_000, texture_size=4096, remesh=True, remesh_band=1, remesh_project=0,
    )
    glb.apply_transform(single.TO_GLTF)
    glb.export(path(".glb"), extension_webp=True)
    report.update(gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), faces=int(len(glb.faces)))
    json.dump(report, open(path(".json"), "w"), indent=1)
    print(json.dumps(report, indent=1))
    if args.rig:
        del pipeline
        torch.cuda.empty_cache()
        sys.path.insert(0, os.path.join(PROJECT_DIR if "PROJECT_DIR" in globals() else single.PROJECT_DIR,
                                        "scripts", "rigging"))
        from rig_from_glb import rig
        print("Rigged:", rig(path(".glb"), path("")))


if __name__ == "__main__":
    main()

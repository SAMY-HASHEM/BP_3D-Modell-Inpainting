"""Pixal3D (Tencent ARC + Tsinghua, SIGGRAPH 2026) from one picture, to compare with TRELLIS.2 (A of compare_hi3d.py).

Pixal3D is TRELLIS.2 retrained: the same steps, latents and decoders (byte-identical, linked from TRELLIS.2's folder),
but its 4 flow models put each pixel's DINOv3 features straight onto the voxels the camera sees it on, instead of
mixing the picture in through attention. For that it needs the camera: MoGe-2 estimates the field of view (or --fov),
and NAF sharpens DINOv3's features first.

    source activate.sh
    python scripts/generation/compare_pixal3d.py outputs/hi3d_compare/warrior/picture.png --name F_pixal3d \
        --out outputs/hi3d_compare/warrior

Writes <out>/<name>.glb, <name>_input.png (what Pixal3D saw) and <name>.json (camera, time, GPU memory, faces).\n--colour-picture: the shape from the picture, the colours from another one of the same view (_colour_input.png).\npaint_front.py then puts the picture's own pixels onto what the camera saw.
Same seed and resolution as TRELLIS.2's A; the export is generate.py's (remeshed at 1024, 1M faces).
Everything is local (local_ai_models/pixal3d): Pixal3D's code, MoGe-2, NAF (torch hub cache) and DINOv3.
"""
import argparse
import json
import math
import os
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIXAL3D = os.path.join(PROJECT_DIR, "local_ai_models", "pixal3d")
CODE = os.path.join(PIXAL3D, "code", "Pixal3D-master")
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("HF_HUB_CACHE", os.path.join(PROJECT_DIR, "local_ai_models"))
os.environ["HF_HUB_OFFLINE"] = "1"  # the server can't reach Hugging Face's file servers anyway
sys.path.insert(0, CODE)
# MoGe-2 wants utils3d_moge (newer than TRELLIS.2's utils3d, under its own name: no clash; from jsDelivr's GitHub copy)
sys.path.insert(0, os.path.join(PIXAL3D, "helpers", "utils3d_moge_src"))

import numpy as np
import torch
from PIL import Image
import o_voxel
import inference as pixal  # Pixal3D's own inference.py: its camera maths and model set-up

DINOV3 = next(os.path.join(root, d) for root in [os.path.join(PROJECT_DIR, "local_ai_models",
              "models--facebook--dinov3-vitl16-pretrain-lvd1689m", "snapshots")] for d in os.listdir(root))
MOGE = os.path.join(PIXAL3D, "helpers", "moge-2-vitl", "model.pt")
PIPELINE_TYPES = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}
TO_GLTF = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)  # Pixal3D's cube -> Y up, front +Z
NAF = os.path.join(PIXAL3D, "helpers", "NAF-main")  # its weights: torch hub's cache (checkpoints/naf_release.pth)
_hub_load = torch.hub.load


def _local_naf(repo, model, *args, **kwargs):
    """NAF from our copy: torch hub would ask github.com (blocked here) which branch to take."""
    if repo == "valeoai/NAF":
        kwargs.pop("trust_repo", None)
        return _hub_load(NAF, model, *args, source="local", **kwargs)
    return _hub_load(repo, model, *args, **kwargs)


torch.hub.load = _local_naf


def free_gpu(pipeline):
    """All of Pixal3D's models off the GPU (to the CPU): only the generated mesh is left for the last step."""
    for model in list(getattr(pipeline, "models", {}).values()) + [
            getattr(pipeline, name, None) for name in ("image_cond_model_ss", "image_cond_model_shape_512",
                                                       "image_cond_model_shape_1024", "image_cond_model_tex_1024",
                                                       "rembg_model")]:
        if model is not None and hasattr(model, "cpu"):
            model.cpu()
    torch.cuda.empty_cache()


def cut_out(pipeline, picture):
    """Pixal3D's preprocess_image, but its result keeps the outline (RGBA): the background taken away, cropped to
    the object's square + 10 %."""
    if picture.mode == "RGBA" and not np.all(np.asarray(picture)[..., 3] == 255):
        rgba = picture
    else:
        picture = picture.convert("RGB")
        scale = min(1, 1024 / max(picture.size))
        if scale < 1:
            picture = picture.resize((int(picture.width * scale), int(picture.height * scale)), Image.LANCZOS)
        rgba = pipeline.rembg_model(picture)
    ys, xs = np.nonzero(np.asarray(rgba)[..., 3] > 0.8 * 255)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    size = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.1)
    return rgba.crop((cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2))


def on_black(rgba):
    """What Pixal3D sees: the colours on black, by the outline (as its preprocess_image ends)."""
    a = np.asarray(rgba).astype(np.float32) / 255
    return Image.fromarray((np.clip(a[..., :3] * a[..., 3:], 0, 1) * 255).astype(np.uint8))


def line_up(picture, target):
    """`picture` (RGBA) moved and scaled so that its outline lies on `target`'s (best overlap of the two masks)."""
    a = np.asarray(target)[..., 3] > 127
    rgba = np.asarray(picture)
    best = (-1, None)
    for scale in np.linspace(0.8, 1.25, 31):
        size = round(picture.width * scale)
        moved = picture.resize((size, size), Image.LANCZOS)
        m = np.asarray(moved)[..., 3] > 127
        ys, xs = np.nonzero(m)
        ty, tx = np.nonzero(a)
        dy, dx = round(ty.mean() - ys.mean()), round(tx.mean() - xs.mean())
        for ddy in range(-6, 7, 2):
            for ddx in range(-6, 7, 2):
                canvas = Image.new("RGBA", target.size)
                canvas.paste(moved, (dx + ddx, dy + ddy), moved)
                c = np.asarray(canvas)[..., 3] > 127
                iou = (c & a).sum() / max((c | a).sum(), 1)
                if iou > best[0]:
                    best = (iou, canvas)
    print(f"colour picture lined up: outline overlap {best[0]:.1%}", flush=True)
    return best[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("picture")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", default="1024", choices=list(PIPELINE_TYPES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fov", type=float, default=-1, help="camera field of view in radians; default: MoGe-2's")
    parser.add_argument("--colour-picture", help="the colours from this picture instead (the same view, e.g. the "
                        "painted version of a grey sculpt): the shape comes from `picture`, the colour step sees this "
                        "one, lined up on the first by its outline")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = lambda ext: os.path.join(args.out, args.name + ext)
    report = {"picture": args.picture, "seed": args.seed, "resolution": args.resolution}

    for config in pixal.IMAGE_COND_CONFIGS.values():
        config["model_name"] = DINOV3  # the same DINOv3 weights, from our own copy
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    pipeline = pixal.init_pipeline(PIXAL3D, low_vram=False)
    report["load_seconds"] = round(time.time() - start)

    cutout = cut_out(pipeline, Image.open(args.picture))
    cutout.save(path("_input_rgba.png"))  # with its outline, for paint_front.py
    image = on_black(cutout)
    image.save(path("_input.png"))
    if args.colour_picture:
        colour_rgba = line_up(cut_out(pipeline, Image.open(args.colour_picture)), cutout)
        colour_rgba.save(path("_colour_input_rgba.png"))
        colour = on_black(colour_rgba)
        report["colour_picture"] = args.colour_picture
        tex_model, own = pipeline.image_cond_model_tex_1024, pipeline.get_proj_cond_shape

        def colour_for_texture(model, images, *a, **k):  # only the colour step sees the colour picture
            return own(model, [colour] if model is tex_model else images, *a, **k)
        pipeline.get_proj_cond_shape = colour_for_texture
    if args.fov > 0:
        fov = args.fov
    else:
        from moge.model.v2 import MoGeModel
        moge = MoGeModel.from_pretrained(MOGE).cuda().eval()
        with torch.no_grad():
            pixels = torch.from_numpy(np.asarray(image.convert("RGB"), np.float32) / 255).permute(2, 0, 1).cuda()
            fx = float(moge.infer(pixels)["intrinsics"].squeeze()[0, 0]) * image.width
        fov = 2 * math.atan(image.width / (2 * fx))
        del moge
        torch.cuda.empty_cache()
    distance = pixal.distance_from_fov(fov, torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0, 511]), 1.0, 512)
    camera = {"camera_angle_x": fov, "distance": distance["distance_from_x"], "mesh_scale": 1.0}
    report["camera"] = {"fov_degrees": round(math.degrees(fov), 2), "distance": round(camera["distance"], 3)}
    print("camera:", report["camera"], flush=True)

    start = time.time()
    torch.manual_seed(args.seed)
    meshes, (_, _, res) = pipeline.run(image, camera_params=camera, seed=args.seed, preprocess_image=False,
                                       return_latent=True, pipeline_type=PIPELINE_TYPES[args.resolution])
    report["seconds"] = round(time.time() - start)
    mesh = meshes[0]
    free_gpu(pipeline)  # the mesh step at 1536 needs the room the models take (it ran out of 46 GB, Nov 22)
    mesh.simplify(16777216)
    torch.cuda.empty_cache()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout, grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1_000_000, texture_size=4096, remesh=True, remesh_band=1, remesh_project=0,
    )
    # Pixal3D's cube is the camera's, not TRELLIS.2's. Its own turn (inference.py: x -> -x, y -> -z, z -> -y) leaves
    # the front at -Z; turned half round to glTF's +Z like TRELLIS.2's models (the first renders showed the back, Nov 22)
    glb.apply_transform(TO_GLTF)
    glb.export(path(".glb"), extension_webp=True)
    report.update(gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), faces=int(len(glb.faces)))
    json.dump(report, open(path(".json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

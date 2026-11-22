"""TRELLIS.2 alone against TRELLIS.2 + Hi3D 1.0, from one picture and the same rough structure (shapes only).

Hi3D 1.0 (hi3d.ai, ex-Sparc3D, open since Oct 2026) only refines a rough shape and makes no colours, so it can only take
over TRELLIS.2's fine-shape step:

    A   picture -> TRELLIS.2 structure -> TRELLIS.2 shape                          (generate.py as always)
    B   picture -> TRELLIS.2 structure (64^3, as a box mesh) -> Hi3D shape
    C   picture -> TRELLIS.2's finished model A -> Hi3D shape

A and B start from the SAME structure: TRELLIS.2 samples it on 64^3 and shrinks it to 32^3 for its 1024 path; here the
64^3 grid of that very sampling is kept and given to Hi3D (glTF Y up, front +Z: turning it 90/180/270 degrees gave
worse shapes, Nov 21). Same seed for all.

The picture: TRELLIS.2 gets its own preparation (background removed, on black); Hi3D gets the cut-out WITH its
transparency (input_rgba.png), which it puts on white with a 20% border itself. Given TRELLIS.2's black square
instead, Hi3D took the black as part of the picture and made swollen, grainy shapes (the first run, Nov 21).

    source activate.sh
    python scripts/generation/compare_hi3d.py "pictures/worior girl /anastasiia-akimova-3-4.jpg" --crop 212 0 944 1573 \
        --name warrior --out outputs/hi3d_compare

Writes into <out>/<name>/: picture.png, input_input.png (TRELLIS.2's), input_rgba.png (Hi3D's), A_trellis2.glb,
structure64.glb, B_hi3d.glb, C_hi3d.glb and report.json (times, GPU memory, faces). A model already there is kept.
TRELLIS.2 and Hi3D are loaded one after the other, so one A40 is enough. clay_views.py draws them in grey.
"""
import _paths  # noqa: F401
import argparse
import json
import os
import time

import generate  # sets the environment and patches TRELLIS.2 before it loads

import numpy as np
import torch
import trimesh
from PIL import Image

HI3D = os.path.join(generate.PROJECT_DIR, "local_ai_models", "hi3d")


def gpu_peak():
    return round(torch.cuda.max_memory_allocated() / 2**30, 1)


def free():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def keep_structure(pipeline, kept):
    """TRELLIS.2's structure step, but on its full 64^3 grid; the grid is kept, and TRELLIS.2 gets what it asked for
    (32^3 on its 1024 path: a cell is filled if any of its 8 is, as its own max-pooling does)."""
    sample = pipeline.sample_sparse_structure

    def wrapped(cond, resolution, num_samples=1, sampler_params={}):
        coords = sample(cond, 64, num_samples, sampler_params)
        kept["coords64"] = coords.clone()
        if resolution == 64:
            return coords
        ratio = 64 // resolution
        return torch.unique(torch.cat([coords[:, :1], coords[:, 1:] // ratio], 1), dim=0).int()
    pipeline.sample_sparse_structure = wrapped


def box_mesh(coords64):
    """The 64^3 structure as a mesh of its voxels' outer faces, in glTF axes (Y up, front +Z) and TRELLIS.2's cube size.
    TRELLIS.2's cube: x right, -y front, z up -> glTF (x, z, -y)."""
    grid = np.zeros((64, 64, 64), bool)
    c = coords64[:, 1:].cpu().numpy()
    grid[c[:, 0], c[:, 1], c[:, 2]] = True
    mesh = trimesh.voxel.VoxelGrid(grid).as_boxes()
    v = (mesh.vertices + 0.5) / 64 - 0.5  # as_boxes puts voxel i's centre at i
    mesh.vertices = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("picture")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"), help="one figure of a sheet")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", default="outputs/hi3d_compare")
    parser.add_argument("--resolution", type=int, default=1024, choices=[512, 1024, 1536])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hi3d-steps", type=int, default=25)
    args = parser.parse_args()
    folder = os.path.join(args.out, args.name)
    os.makedirs(folder, exist_ok=True)
    path = lambda name: os.path.join(folder, name)
    report_path = path("report.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}
    report.update({"picture": args.picture, "crop": args.crop, "seed": args.seed, "resolution": args.resolution})

    picture = Image.open(args.picture).convert("RGB")
    if args.crop:
        picture = picture.crop(args.crop)
    picture.save(path("picture.png"))

    # A: TRELLIS.2 as always, keeping its 64^3 structure; and the cut-out with transparency for Hi3D
    needed = [path(f) for f in ("A_trellis2.glb", "structure64.glb", "input_rgba.png")]
    if not all(map(os.path.exists, needed)):
        pipeline = generate.load_pipeline()
        generate.INPUTS["folder"] = folder
        image = generate.prepare_image(pipeline, path("picture.png"), "input")  # as TRELLIS.2 sees it
        pipeline.rembg_model.to(pipeline.device)  # TRELLIS.2 keeps it on the CPU between uses
        pipeline.rembg_model(picture).save(path("input_rgba.png"))
        kept = {}
        keep_structure(pipeline, kept)
        torch.cuda.reset_peak_memory_stats()
        mesh, seconds = generate.generate(pipeline, image, str(args.resolution), args.seed)
        report["A"] = {"seconds": round(seconds), "gpu_gb": gpu_peak()}
        generate.save_glb(mesh, str(args.resolution), needed[0])
        box_mesh(kept["coords64"]).export(needed[1])
        report["structure_voxels"] = int(len(kept["coords64"]))
        json.dump(report, open(report_path, "w"), indent=1)
        del pipeline, mesh
        free()

    # B and C: Hi3D from the structure, and from TRELLIS.2's finished model
    from sparc3d import Sparc3DPipeline
    hi3d = Sparc3DPipeline.from_pretrained(os.path.join(HI3D, "Hi3Dv1.0"), local_files_only=True,
                                           dino_model_id=os.path.join(HI3D, "dinov2-large"),
                                           clip_model_id=os.path.join(HI3D, "clip-vit-large-patch14")).to("cuda")
    rgba = Image.open(needed[2])
    a = trimesh.load(needed[0], force="mesh")
    starts = {"B": trimesh.load(needed[1], force="mesh"), "C": trimesh.Trimesh(a.vertices, a.faces, process=False)}
    for key, coarse in starts.items():
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        shape = hi3d(rgba, coarse, resolution=args.resolution, num_inference_steps=args.hi3d_steps, seed=args.seed).mesh
        report[key] = {"seconds": round(time.time() - start), "gpu_gb": gpu_peak(), "faces": int(len(shape.faces))}
        shape.export(path(f"{key}_hi3d.glb"))
        print(key, report[key], flush=True)
    report.setdefault("A", {})["faces"] = int(len(a.faces))
    json.dump(report, open(report_path, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

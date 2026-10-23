"""Generate a 3D model made of parts with OmniPart (github.com/HKU-MMLab/OmniPart, cloned to ./OmniPart).

Runs in OmniPart's own conda environment (torch 2.4 / CUDA 12.1), not in inpainting-3d-modelling:
    source conda/etc/profile.d/conda.sh && conda activate omnipart
    python scripts/omnipart_generate.py pictures/anatomy/front_rgba.png pictures/anatomy/front_mask.exr

The picture is an RGBA PNG and the mask an .exr with one part number per pixel (0 = background), as
make_part_mask.py writes them. Output in outputs/omnipart/<picture name>/:
    voxel_coords_vis.ply   1. the rough shape of the whole object (TRELLIS voxels)
    bboxes_vis.glb         2. one 3D box per part (OmniPart's box generator)
    *.glb / *.ply          3. the parts, generated together inside their boxes, and merged

Same steps as OmniPart's scripts/inference_omnipart.py, with one fix: the box generator sometimes
adds extra, empty boxes after the real ones (8 boxes for 6 parts), and part synthesis then crashes
with a floating point exception. Only as many boxes as the mask has parts are kept.
"""
import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OMNIPART_DIR = os.path.join(PROJECT_DIR, "OmniPart")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(PROJECT_DIR, "models"))
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf


def merge_parts(folder):
    """Merge part1.glb, part2.glb, ... into mesh_textured.glb and mesh_segment.glb (one colour per part).

    OmniPart's own merge_parts filters on the whole path and skips everything containing "parts",
    so for an output folder like parts-front-clean_rgba_p10 it found no parts and crashed.
    """
    import glob
    import re
    import trimesh

    paths = [p for p in glob.glob(os.path.join(folder, "part*.glb")) if re.fullmatch(r"part[1-9]\d*\.glb", os.path.basename(p))]
    paths.sort(key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    textured, segment = trimesh.Scene(), trimesh.Scene()
    palette = np.random.default_rng(0).integers(60, 255, size=(len(paths), 3))
    for i, path in enumerate(paths):
        mesh = trimesh.load(path, force="mesh")
        textured.add_geometry(mesh, geom_name=f"part-{i + 1}")
        colored = mesh.copy()
        colored.visual = trimesh.visual.ColorVisuals(mesh=colored, vertex_colors=np.append(palette[i], 255))
        segment.add_geometry(colored, geom_name=f"part-{i + 1}")
    textured.export(os.path.join(folder, "mesh_textured.glb"))
    segment.export(os.path.join(folder, "mesh_segment.glb"))
    for path in paths:
        os.remove(path)
    print(f"    merged {len(paths)} parts")


def main():
    parser = argparse.ArgumentParser(description="Picture + part mask -> 3D model made of parts (OmniPart)")
    parser.add_argument("image", help="RGBA picture (transparent background)")
    parser.add_argument("mask", help=".exr part mask (0 = background, 1, 2, ... = parts)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25, help="sampling steps for the parts (OmniPart default 25)")
    parser.add_argument("--guidance", type=float, default=7.5, help="how strictly the parts follow the picture")
    parser.add_argument("--simplify", type=float, default=0.3, help="share of faces to REMOVE when simplifying (OmniPart default 0.3; 0 keeps all)")
    parser.add_argument("--textured", action="store_true", help="also bake textures (slower)")
    parser.add_argument("--tag", default="", help="added to the output folder name, to keep several runs")
    args = parser.parse_args()

    image_path, mask_path = os.path.abspath(args.image), os.path.abspath(args.mask)
    name = os.path.splitext(os.path.basename(image_path))[0] + (f"_{args.tag}" if args.tag else "")
    out = os.path.join(PROJECT_DIR, "outputs", "omnipart", name)
    os.makedirs(out, exist_ok=True)

    # OmniPart's code uses paths relative to its own folder (configs/, ckpt/)
    os.chdir(OMNIPART_DIR)
    sys.path.insert(0, OMNIPART_DIR)
    from modules.bbox_gen.models.autogressive_bbox_gen import BboxGen
    from modules.inference_utils import (gen_mesh_from_bounds, load_img_mask, prepare_bbox_gen_input,
                                         prepare_part_synthesis_input, vis_voxel_coords)
    from modules.part_synthesis.pipelines import OmniPartImageTo3DPipeline
    from modules.part_synthesis.process_utils import save_parts_outputs

    torch.manual_seed(args.seed)
    partfield = hf_hub_download("omnipart/OmniPart_modules", "partfield_encoder.ckpt", local_dir="ckpt")
    bbox_ckpt = hf_hub_download("omnipart/OmniPart_modules", "bbox_gen.ckpt", local_dir="ckpt")
    pipeline = OmniPartImageTo3DPipeline.from_pretrained("omnipart/OmniPart")
    pipeline.to("cuda")
    config = OmegaConf.load("configs/bbox_gen.yaml").model.args
    config.partfield_encoder_path = partfield
    bbox_gen = BboxGen(config)
    bbox_gen.load_state_dict(torch.load(bbox_ckpt), strict=False)
    bbox_gen.to("cuda").eval().half()

    img_white, img_black, mask, mask_vis = load_img_mask(image_path, mask_path)
    mask_vis.save(os.path.join(out, "img_mask_vis.png"))
    num_parts = int((torch.unique(mask) > 0).sum())
    print(f"1. Rough shape of the whole object ({num_parts} parts in the mask)...")
    coords = pipeline.get_coords(img_black, num_samples=1, seed=args.seed,
                                 sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5})
    coords = coords.cpu().numpy()
    np.save(os.path.join(out, "voxel_coords.npy"), coords)
    vis_voxel_coords(coords).export(os.path.join(out, "voxel_coords_vis.ply"))

    print("2. One 3D box per part...")
    boxes = bbox_gen.generate(prepare_bbox_gen_input(os.path.join(out, "voxel_coords.npy"), img_white, mask))["bboxes"][0]
    if len(boxes) != num_parts:
        print(f"    the box generator made {len(boxes)} boxes for {num_parts} parts; keeping the first {num_parts}")
        boxes = boxes[:num_parts]
    np.save(os.path.join(out, "bboxes.npy"), boxes)
    gen_mesh_from_bounds(boxes).export(os.path.join(out, "bboxes_vis.glb"))

    print("3. Generating the parts inside their boxes...")
    inputs = prepare_part_synthesis_input(os.path.join(out, "voxel_coords.npy"), os.path.join(out, "bboxes.npy"), mask)
    result = pipeline.get_slat(
        img_black, inputs["coords"], [inputs["part_layouts"]], inputs["masks"], seed=args.seed,
        slat_sampler_params={"steps": args.steps, "cfg_strength": args.guidance},
        formats=["mesh", "gaussian", "radiance_field"], preprocess_image=False,
    )
    save_parts_outputs(result, output_dir=out, simplify_ratio=args.simplify, save_video=False,
                       save_glb=True, textured=args.textured)
    merge_parts(out)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()

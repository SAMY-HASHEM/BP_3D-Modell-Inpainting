"""Generate a 3D model (.glb) from one picture with TRELLIS.2.

Usage:
    source activate.sh
    python scripts/generate.py pictures/my_picture.png
    python scripts/generate.py pictures/my_picture.png --resolution 1536 --seed 7
    python scripts/generate.py pictures/parts/*.png        # several pictures, TRELLIS.2 loaded once
    python scripts/generate.py pictures/my_picture.png --texture-guidance 3   # colours follow the picture more

The results go to outputs/: the picture after background removal (*_input.png) and the model (*.glb).
"""
import argparse
import os
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # saves GPU memory
# The conda environment sets these on activation. They are repeated here for runs that skip the
# activation (e.g. PyCharm's Run button), so the model weights are never downloaded a second time.
os.environ.setdefault("HF_HUB_CACHE", os.path.join(PROJECT_DIR, "models"))
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(PROJECT_DIR, "models", "modules"))

import numpy as np
import torch
from PIL import Image
import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.representations.mesh import base as mesh_base

PIPELINE_TYPES = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}
DEFAULT_STEPS = 12  # TRELLIS.2's default number of sampling steps per stage
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# CuMesh asks the GPU for memory directly and can't use the memory PyTorch keeps cached.
# Without freeing the cache first, hole filling runs out of memory at 1536 on a 24 GB GPU.
_original_fill_holes = mesh_base.Mesh.fill_holes


def _fill_holes_after_freeing_cache(self, *args, **kwargs):
    torch.cuda.empty_cache()
    return _original_fill_holes(self, *args, **kwargs)


mesh_base.Mesh.fill_holes = _fill_holes_after_freeing_cache


def load_pipeline():
    print("Loading TRELLIS.2 (takes a few minutes the first time)...")
    start = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    print(f"    loaded in {time.time() - start:.0f} s")
    return pipeline


def crop_to_object(pipeline, image):
    """Crops a big picture to its object (with a small margin) before TRELLIS.2 shrinks it.

    TRELLIS.2's preprocess_image first shrinks the whole picture to 1024 px and only then cuts out
    the object, so an object filling 70 % of a 5760 px photo ends up with 710 px, and DINOv3 then
    stretches that back to 1024. Cropping first keeps the full 1024 px for the object itself.
    """
    if max(image.size) <= 1024:
        return image  # already small: nothing is lost
    if image.mode == "RGBA" and np.asarray(image)[..., 3].min() < 255:
        alpha = np.asarray(image)[..., 3]  # cut-outs already say where the object is
        scale = 1.0
    else:
        scale = 1024 / max(image.size)
        small = image.convert("RGB").resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        if pipeline.low_vram:
            pipeline.rembg_model.to(pipeline.device)
        alpha = np.asarray(pipeline.rembg_model(small))[..., 3]
        if pipeline.low_vram:
            pipeline.rembg_model.cpu()
    ys, xs = np.nonzero(alpha > 0.8 * 255)
    if len(xs) == 0:
        return image
    x0, x1, y0, y1 = xs.min() / scale, (xs.max() + 1) / scale, ys.min() / scale, (ys.max() + 1) / scale
    margin = 0.02 * max(x1 - x0, y1 - y0)
    box = (max(0, int(x0 - margin)), max(0, int(y0 - margin)),
           min(image.width, int(x1 + margin)), min(image.height, int(y1 + margin)))
    return image.crop(box)


def prepare_image(pipeline, image_path, name):
    """Removes the background and saves what TRELLIS.2 will see as <name>_input.png."""
    image = pipeline.preprocess_image(crop_to_object(pipeline, Image.open(image_path)))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image.save(os.path.join(OUTPUT_DIR, f"{name}_input.png"))
    return image


def generate(pipeline, image, resolution, seed, steps=DEFAULT_STEPS, guidance=None, texture_guidance=None):
    """Returns the generated mesh and the generation time in seconds."""
    torch.cuda.empty_cache()
    start = time.time()
    # The same number of steps for all stages: structure, shape and texture.
    # Guidance = how strictly a stage follows the picture. TRELLIS.2's defaults: 7.5 for structure
    # and shape, 1.0 (= off) for the texture. None keeps the default.
    shape_params = {"steps": steps}
    if guidance is not None:
        shape_params["guidance_strength"] = guidance
    texture_params = {"steps": steps}
    if texture_guidance is not None:
        texture_params["guidance_strength"] = texture_guidance
    mesh = pipeline.run(
        image, seed=seed, preprocess_image=False, pipeline_type=PIPELINE_TYPES[resolution],
        sparse_structure_sampler_params=shape_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=texture_params,
    )[0]
    return mesh, time.time() - start


def save_glb(mesh, resolution, glb_path):
    # Remeshing rebuilds the mesh from the full voxel grid. At 1536 that gives ~51M faces,
    # which runs out of memory, so there we simplify the generated mesh directly instead.
    remesh = resolution != "1536"
    mesh.simplify(16777216 if remesh else 4_000_000)
    torch.cuda.empty_cache()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000, texture_size=4096, remesh=remesh, remesh_band=1, remesh_project=0,
        verbose=True,  # print each step and show progress bars (UV unwrapping can take minutes)
    )
    glb.export(glb_path, extension_webp=True)


def picture_name(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]


def main():
    parser = argparse.ArgumentParser(description="Picture(s) -> 3D model(s) with TRELLIS.2")
    parser.add_argument("images", nargs="+", help="path to the input picture (several: one model each)")
    parser.add_argument("--resolution", choices=PIPELINE_TYPES.keys(), default="1024")
    parser.add_argument("--seed", type=int, default=42, help="same seed + same picture = same result")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="sampling steps per stage (more = slower)")
    parser.add_argument("--guidance", type=float,
                        help="how strictly the shape follows the picture (TRELLIS.2 default 7.5; too high gives artifacts)")
    parser.add_argument("--texture-guidance", type=float,
                        help="how strictly the colours follow the picture (TRELLIS.2 default 1.0 = off; try 2-4)")
    args = parser.parse_args()
    for path in args.images:
        if not os.path.isfile(path):
            # Check before loading TRELLIS.2, which takes about a minute.
            parser.error(f"picture not found: {path} (is it in pictures/? e.g. pictures/{os.path.basename(path)})")

    pipeline = load_pipeline()  # once for all pictures
    for i, path in enumerate(args.images, start=1):
        name = f"{picture_name(path)}_{args.resolution}_seed{args.seed}"
        if args.steps != DEFAULT_STEPS:
            name += f"_steps{args.steps}"
        if args.guidance is not None:
            name += f"_guidance{args.guidance:g}"
        if args.texture_guidance is not None:
            name += f"_texguidance{args.texture_guidance:g}"
        if len(args.images) > 1:
            print(f"\n[{i}/{len(args.images)}] {path}")

        print("Removing the background...")
        image = prepare_image(pipeline, path, name)

        print(f"Generating the 3D model at {args.resolution} with {args.steps} steps per stage...")
        mesh, gen_time = generate(pipeline, image, args.resolution, args.seed, args.steps,
                                  args.guidance, args.texture_guidance)
        print(f"    generated in {gen_time:.0f} s, {mesh.faces.shape[0]:,} faces")

        print("Saving as .glb...")
        start = time.time()
        glb_path = os.path.join(OUTPUT_DIR, f"{name}.glb")
        save_glb(mesh, args.resolution, glb_path)
        del mesh
        print(f"    saved in {time.time() - start:.0f} s")
        print(f"Done: {glb_path}")


if __name__ == "__main__":
    main()

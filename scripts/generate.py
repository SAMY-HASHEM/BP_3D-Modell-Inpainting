"""Generate a 3D model (.glb) from one picture with TRELLIS.2.

Usage:
    conda activate trellis2
    cd ~/Bachlor-project
    python scripts/generate.py my_picture.png
    python scripts/generate.py my_picture.png --resolution 1536 --seed 7

The results go to outputs/: the picture after background removal (*_input.png) and the model (*.glb).
"""
import argparse
import os
import time

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # saves GPU memory

import torch
from PIL import Image
import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.representations.mesh import base as mesh_base

PIPELINE_TYPES = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}
DEFAULT_STEPS = 12  # TRELLIS.2's default number of sampling steps per stage
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def prepare_image(pipeline, image_path, name):
    """Removes the background and saves what TRELLIS.2 will see as <name>_input.png."""
    image = pipeline.preprocess_image(Image.open(image_path))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image.save(os.path.join(OUTPUT_DIR, f"{name}_input.png"))
    return image


def generate(pipeline, image, resolution, seed, steps=DEFAULT_STEPS):
    """Returns the generated mesh and the generation time in seconds."""
    torch.cuda.empty_cache()
    start = time.time()
    # The same number of steps for all stages: structure, shape and texture.
    sampler_params = {"steps": steps}
    mesh = pipeline.run(
        image, seed=seed, preprocess_image=False, pipeline_type=PIPELINE_TYPES[resolution],
        sparse_structure_sampler_params=sampler_params,
        shape_slat_sampler_params=sampler_params,
        tex_slat_sampler_params=sampler_params,
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
    parser = argparse.ArgumentParser(description="Picture -> 3D model with TRELLIS.2")
    parser.add_argument("image", help="path to the input picture")
    parser.add_argument("--resolution", choices=PIPELINE_TYPES.keys(), default="1024")
    parser.add_argument("--seed", type=int, default=42, help="same seed + same picture = same result")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="sampling steps per stage (more = slower)")
    args = parser.parse_args()
    if not os.path.isfile(args.image):
        # Check before loading TRELLIS.2, which takes about a minute.
        parser.error(f"picture not found: {args.image} (is it in pictures/? e.g. pictures/{os.path.basename(args.image)})")

    name = f"{picture_name(args.image)}_{args.resolution}_seed{args.seed}"
    if args.steps != DEFAULT_STEPS:
        name += f"_steps{args.steps}"
    pipeline = load_pipeline()

    print("Removing the background...")
    image = prepare_image(pipeline, args.image, name)

    print(f"Generating the 3D model at {args.resolution} with {args.steps} steps per stage...")
    mesh, gen_time = generate(pipeline, image, args.resolution, args.seed, args.steps)
    print(f"    generated in {gen_time:.0f} s, {mesh.faces.shape[0]:,} faces")

    print("Saving as .glb...")
    start = time.time()
    glb_path = os.path.join(OUTPUT_DIR, f"{name}.glb")
    save_glb(mesh, args.resolution, glb_path)
    print(f"    saved in {time.time() - start:.0f} s")
    print(f"Done: {glb_path}")


if __name__ == "__main__":
    main()

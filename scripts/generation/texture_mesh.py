"""Colours for a given mesh from pictures (pipeline stage 7, a painter): TRELLIS.2's own texturing pipeline (MIT; its
weights came with TRELLIS.2-4B), the shape fixed, only the colours and materials made, from one picture or several
(multi_image.py's trick: every step looks at all pictures and averages).

Unlike projecting the pictures (paint_views.py, an experiment), nothing has to line up: the colours are made for the
mesh's own shape, the pictures only say what they should look like. For any mesh: a Pixal3D model, a cleaned or
retopologised one, a model whose colours went wrong.

    python scripts/generation/texture_mesh.py model.glb --pictures front.png back.png side.png --out textured.glb \\
        [--resolution 1024] [--texture-size 2048] [--seed 42]

The pictures cut out (RGBA), ideally de-lit (delight.py); the first one decides the look most.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: F401,E402  (TRELLIS.2 on the path, the model cache)

import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

from multi_image import multi_image_mode  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model")
    p.add_argument("--pictures", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=int, default=1024, choices=[512, 1024])
    p.add_argument("--texture-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    from trellis2.pipelines import Trellis2TexturingPipeline
    start = time.time()
    pipeline = Trellis2TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B", "texturing_pipeline.json")
    pipeline.cuda()
    print(f"loaded in {time.time() - start:.0f} s", flush=True)

    mesh = trimesh.load(a.model, force="mesh")
    images = [pipeline.preprocess_image(Image.open(path)) for path in a.pictures]
    start = time.time()
    torch.manual_seed(a.seed)
    if len(images) > 1:
        with multi_image_mode(pipeline, images, "multidiffusion"):
            out = pipeline.run(mesh, images[0], seed=a.seed, preprocess_image=False, resolution=a.resolution,
                               texture_size=a.texture_size)
    else:
        out = pipeline.run(mesh, images[0], seed=a.seed, preprocess_image=False, resolution=a.resolution,
                           texture_size=a.texture_size)
    print(f"textured in {time.time() - start:.0f} s, GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
          flush=True)
    out.export(a.out)
    print("textured:", a.out)


if __name__ == "__main__":
    main()

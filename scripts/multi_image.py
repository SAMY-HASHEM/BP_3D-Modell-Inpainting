"""Generate one 3D model from several pictures of the same object (e.g. front and back) with TRELLIS.2.

Usage:
    source activate.sh
    python scripts/multi_image.py pictures/front.png pictures/back.png
    python scripts/multi_image.py pictures/front.png pictures/back.png --mode stochastic --resolution 1536

TRELLIS.2 was trained with one picture only. This is the training-free trick from TRELLIS v1
(run_multi_image), moved to TRELLIS.2: every sampling step looks at the pictures through the same,
unchanged flow models and mixes what they say.

    multidiffusion  every step runs once per picture and averages the predictions (n pictures = n x slower)
    stochastic      every step uses only one picture, taking turns: 1, 2, ..., 1, 2, ... (no extra time)

The first picture is the main one: TRELLIS.2 decides the object's orientation from the pictures, so
put the front view first. The model is saved as outputs/<first picture>_multi<n>_<mode>_<resolution>_seed<seed>.glb.
"""
import argparse
import os
import time
from contextlib import contextmanager

import torch
from PIL import Image
from torch import nn

from generate import DEFAULT_STEPS, OUTPUT_DIR, PIPELINE_TYPES, load_pipeline, picture_name, save_glb

MODES = ["multidiffusion", "stochastic"]


class MultiImageFlowModel(nn.Module):
    """Wraps one TRELLIS.2 flow model so that one call looks at all pictures instead of one.

    The sampler calls the flow model with `cond` = image features. Here `cond` holds the features of
    all n pictures (one row each), so we split it and combine the n answers into one.
    """

    def __init__(self, model, mode):
        super().__init__()
        self.model = model
        self.mode = mode
        self.step = -1
        self.last_t = None

    def __getattr__(self, name):
        # The pipeline reads attributes like in_channels or resolution from the flow model.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, x_t, t, cond, **kwargs):
        num_images = cond.shape[0]
        # The negative condition (for classifier-free guidance) is all zeros, the same for every
        # picture, so one call is enough.
        if not cond.any():
            return self.model(x_t, t, cond[:1], **kwargs)

        if self.mode == "multidiffusion":
            preds = [self.model(x_t, t, cond[i:i + 1], **kwargs) for i in range(num_images)]
            return sum(preds[1:], preds[0]) / num_images

        # stochastic: t goes from 1000 down to 0, so a new, smaller t means the next step,
        # and a bigger t means a new sampling run started.
        t_now = float(t[0])
        if self.last_t is None or t_now > self.last_t:
            self.step = 0
        elif t_now < self.last_t:
            self.step += 1
        self.last_t = t_now
        i = self.step % num_images
        return self.model(x_t, t, cond[i:i + 1], **kwargs)


@contextmanager
def multi_image_mode(pipeline, images, mode):
    """Inside this block, pipeline.run() uses all `images` instead of the one picture it gets."""
    original_get_cond = pipeline.get_cond
    original_models = dict(pipeline.models)

    def get_cond_for_all_images(_image, resolution, include_neg_cond=True):
        # pipeline.run() asks for the features of its one picture; we hand back those of all pictures.
        return original_get_cond(images, resolution, include_neg_cond)

    pipeline.get_cond = get_cond_for_all_images
    for name, model in original_models.items():
        if "flow_model" in name:
            pipeline.models[name] = MultiImageFlowModel(model, mode)
    try:
        yield
    finally:
        pipeline.get_cond = original_get_cond
        pipeline.models.clear()
        pipeline.models.update(original_models)


def generate_multi_image(pipeline, images, resolution, seed, mode, steps=DEFAULT_STEPS):
    """Like generate() in generate.py, but with several (already background-removed) pictures."""
    if mode == "stochastic" and len(images) > steps:
        print(f"Warning: {len(images)} pictures but only {steps} steps, so some pictures are never used.")
    torch.cuda.empty_cache()
    start = time.time()
    sampler_params = {"steps": steps}
    with multi_image_mode(pipeline, images, mode):
        mesh = pipeline.run(
            images[0], seed=seed, preprocess_image=False, pipeline_type=PIPELINE_TYPES[resolution],
            sparse_structure_sampler_params=sampler_params,
            shape_slat_sampler_params=sampler_params,
            tex_slat_sampler_params=sampler_params,
        )[0]
    return mesh, time.time() - start


def main():
    parser = argparse.ArgumentParser(description="Several pictures of one object -> one 3D model with TRELLIS.2")
    parser.add_argument("images", nargs="+", help="pictures of the same object, front view first")
    parser.add_argument("--mode", choices=MODES, default="multidiffusion")
    parser.add_argument("--resolution", choices=PIPELINE_TYPES.keys(), default="1024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="sampling steps per stage")
    args = parser.parse_args()
    if len(args.images) < 2:
        parser.error("give at least two pictures (for one picture, use generate.py)")
    for path in args.images:
        if not os.path.isfile(path):
            parser.error(f"picture not found: {path}")

    name = f"{picture_name(args.images[0])}_multi{len(args.images)}_{args.mode}_{args.resolution}_seed{args.seed}"
    if args.steps != DEFAULT_STEPS:
        name += f"_steps{args.steps}"
    pipeline = load_pipeline()

    print("Removing the backgrounds...")
    images = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, path in enumerate(args.images, start=1):
        image = pipeline.preprocess_image(Image.open(path))
        image.save(os.path.join(OUTPUT_DIR, f"{name}_input{i}.png"))
        images.append(image)

    print(f"Generating from {len(images)} pictures ({args.mode}) at {args.resolution} with {args.steps} steps...")
    mesh, gen_time = generate_multi_image(pipeline, images, args.resolution, args.seed, args.mode, args.steps)
    print(f"    generated in {gen_time:.0f} s, {mesh.faces.shape[0]:,} faces")

    print("Saving as .glb...")
    start = time.time()
    glb_path = os.path.join(OUTPUT_DIR, f"{name}.glb")
    save_glb(mesh, args.resolution, glb_path)
    print(f"    saved in {time.time() - start:.0f} s")
    print(f"Done: {glb_path}")


if __name__ == "__main__":
    main()

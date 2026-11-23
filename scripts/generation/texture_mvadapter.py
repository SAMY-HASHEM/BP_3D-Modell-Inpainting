"""Colours for a given mesh with MV-Adapter (pipeline stage 7, a painter; Apache-2.0, on SDXL: OpenRAIL++): six views of
the mesh are drawn (its own positions and normals), MV-Adapter paints them like the picture, consistent with each other
and with the shape, and they are projected back onto a fresh UV layout. Nothing has to line up with a photo (unlike
paint_views.py): the views are made for the mesh.

Runs in its own environment (conda/envs/mvadapter-venv: the main one plus diffusers, cvcuda ...; the models in
local_ai_models/mv_adapter, downloaded on the home PC: the server can't reach Hugging Face's or GitHub's files):

    conda/envs/mvadapter-venv/bin/python scripts/generation/texture_mvadapter.py model.glb --picture front.png \\
        --out textured.glb [--seed 42] [--text "high quality"] [--steps 50]

The picture: the front view, cut out (RGBA), ideally de-lit (delight.py). The model in glTF axes, front +Z.
Writes textured.glb and textured_views.png (the six painted views: front, right, back, left, top, bottom).
Without LaMa (view in-painting) and with our Real-ESRGAN x4 for the views (the x2 and LaMa files are on GitHub).
"""
import argparse
import os
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(PROJECT, "local_ai_models", "mv_adapter")
sys.path.insert(0, os.path.join(ROOT, "MV-Adapter-code"))
os.environ["HF_HUB_OFFLINE"] = "1"

import torch  # noqa: E402
from PIL import Image  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model")
    p.add_argument("--picture", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--text", default="high quality")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--upscale", action="store_true", help="the views 4 x bigger (Real-ESRGAN) before projecting")
    a = p.parse_args()
    from diffusers import AutoencoderKL
    from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
    from mvadapter.pipelines.pipeline_texture import ModProcessConfig, TexturePipeline
    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
    from mvadapter.utils import make_image_grid
    sys.path.insert(0, os.path.join(ROOT, "MV-Adapter-code", "scripts"))
    from inference_ig2mv_sdxl import run_pipeline

    start = time.time()
    vae = AutoencoderKL.from_pretrained(os.path.join(ROOT, "sdxl-vae-fp16-fix"))
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(os.path.join(ROOT, "sdxl-base"), vae=vae, variant="fp16")
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(pipe.scheduler, shift_mode="interpolated", shift_scale=8.0)
    pipe.init_custom_adapter(num_views=6, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0)
    pipe.load_custom_adapter(os.path.join(ROOT, "mv-adapter"), weight_name="mvadapter_ig2mv_sdxl.safetensors")
    pipe.to(device="cuda", dtype=torch.float16)
    pipe.cond_encoder.to(device="cuda", dtype=torch.float16)
    pipe.enable_vae_slicing()
    print(f"loaded in {time.time() - start:.0f} s", flush=True)

    out_dir, name = os.path.dirname(os.path.abspath(a.out)), os.path.splitext(os.path.basename(a.out))[0]
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()
    images, _, _, _ = run_pipeline(pipe, mesh_path=a.model, num_views=6, text=a.text, image=Image.open(a.picture),
                                   height=768, width=768, num_inference_steps=a.steps, guidance_scale=3.0,
                                   seed=a.seed, reference_conditioning_scale=1.0,
                                   negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
                                   device="cuda", remove_bg_fn=None)
    views = os.path.join(out_dir, f"{name}_views.png")
    make_image_grid(images, rows=1).save(views)
    print(f"views in {time.time() - start:.0f} s -> {views}", flush=True)
    del pipe
    torch.cuda.empty_cache()

    start = time.time()
    upscaler = os.path.join(PROJECT, "local_ai_models", "upscale", "RealESRGAN_x4plus.pth") if a.upscale else None
    texture = TexturePipeline(upscaler_ckpt_path=upscaler, inpaint_ckpt_path=None, device="cuda")
    result = texture(mesh_path=a.model, save_dir=out_dir, save_name=name, uv_unwarp=True, preprocess_mesh=False,
                     uv_size=4096, rgb_path=views,
                     rgb_process_config=ModProcessConfig(view_upscale=a.upscale, inpaint_mode="uv"),
                     camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]])
    os.replace(result.shaded_model_save_path, a.out)
    print(f"textured in {time.time() - start:.0f} s, GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()

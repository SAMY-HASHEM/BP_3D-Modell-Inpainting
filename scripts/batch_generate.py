"""Generate one picture at several resolutions, seeds and step counts, loading TRELLIS.2 only once.

Usage:
    source activate.sh
    python scripts/batch_generate.py pictures/my_picture.png
    python scripts/batch_generate.py pictures/my_picture.png --resolutions 512 1024 --seeds 1 2 3 4
    python scripts/batch_generate.py pictures/my_picture.png --seeds 42 --steps 24 48 72

Runs one after another (not in parallel): one run already uses the whole GPU, and 1536 alone
needs ~17.5 GB of the 24 GB. Results go to outputs/<picture>/, with a summary in results.csv.
"""
import argparse
import csv
import os
import time

import torch

from generate import DEFAULT_STEPS, OUTPUT_DIR, PIPELINE_TYPES, generate, load_pipeline, picture_name, prepare_image, save_glb


def main():
    parser = argparse.ArgumentParser(description="Picture -> many 3D models with TRELLIS.2")
    parser.add_argument("image", help="path to the input picture")
    parser.add_argument("--resolutions", nargs="+", choices=PIPELINE_TYPES.keys(), default=["512", "1024", "1536"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--steps", nargs="+", type=int, default=[DEFAULT_STEPS], help="sampling steps per stage")
    args = parser.parse_args()

    name = picture_name(args.image)
    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    runs = [(res, seed, steps) for res in args.resolutions for seed in args.seeds for steps in args.steps]

    pipeline = load_pipeline()
    image = prepare_image(pipeline, args.image, os.path.join(name, name))

    rows = []
    for i, (resolution, seed, steps) in enumerate(runs, start=1):
        print(f"\n[{i}/{len(runs)}] resolution {resolution}, seed {seed}, {steps} steps")
        run_name = f"{name}_{resolution}_seed{seed}"
        if steps != DEFAULT_STEPS:
            run_name += f"_steps{steps}"
        glb_path = os.path.join(out_dir, f"{run_name}.glb")
        try:
            torch.cuda.reset_peak_memory_stats()
            mesh, gen_time = generate(pipeline, image, resolution, seed, steps)
            peak_gb = torch.cuda.max_memory_reserved() / 1024**3
            faces = mesh.faces.shape[0]
            print(f"    generated in {gen_time:.0f} s, peak {peak_gb:.1f} GB, {faces:,} faces")
            start = time.time()
            save_glb(mesh, resolution, glb_path)
            save_time = time.time() - start
            print(f"    saved in {save_time:.0f} s -> {glb_path}")
            rows.append([resolution, seed, steps, f"{gen_time:.0f}", f"{save_time:.0f}", f"{peak_gb:.1f}", faces, "ok"])
            del mesh
        except (torch.OutOfMemoryError, RuntimeError) as e:
            # Keep going with the other runs; one failure shouldn't lose the rest.
            print(f"    FAILED: {str(e).splitlines()[-1] if str(e) else type(e).__name__}")
            rows.append([resolution, seed, steps, "-", "-", "-", "-", "failed"])

    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["resolution", "seed", "steps", "generation_s", "save_s", "peak_vram_gb", "raw_faces", "status"])
        writer.writerows(rows)

    print("\n=== SUMMARY ===")
    print(f"{'res':>5} {'seed':>5} {'steps':>5} {'gen s':>6} {'save s':>7} {'peak GB':>8} {'raw faces':>12}  status")
    for r in rows:
        print(f"{r[0]:>5} {r[1]:>5} {r[2]:>5} {r[3]:>6} {r[4]:>7} {r[5]:>8} {str(r[6]):>12}  {r[7]}")
    print(f"\nSaved table: {csv_path}")


if __name__ == "__main__":
    main()

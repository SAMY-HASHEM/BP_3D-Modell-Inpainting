"""Run an inpainting job package (made in Blender with "Save job for server") on a GPU server.

The package is one .zip: the model, the marked area, the picture and the settings. This is meant
for the HCI group's GPU server, which is only reachable through the uni VDI: upload the zip in
JupyterLab, run this, download the result.

Usage (TRELLIS.2 environment):
    source activate.sh
    python scripts/run_job.py uploads/tavern_job.zip                 # the job's own seed
    python scripts/run_job.py uploads/tavern_job.zip --seeds 1 2 3   # several versions, TRELLIS.2 loaded once
    python scripts/run_job.py uploads/tavern_job.zip --gpu 2         # this GPU instead of the freest one

Writes <job>_seed<N>.glb next to the zip. Without --gpu, the GPU with the least memory in use is
taken, so jobs don't land on a GPU someone else is using.
"""
import argparse
import json
import os
import subprocess
import tempfile
import zipfile


def freest_gpu():
    """Number of the GPU with the least memory in use (asks nvidia-smi), or None without one."""
    try:
        lines = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True).stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    used = [tuple(int(x) for x in line.split(",")) for line in lines]
    return min(used, key=lambda g: g[1])[0] if used else None


def main():
    parser = argparse.ArgumentParser(description="Run an inpainting job package from Blender")
    parser.add_argument("job", help="the job .zip from Blender's 'Save job for server'")
    parser.add_argument("--seeds", type=int, nargs="+", help="seeds to make (default: the job's seed)")
    parser.add_argument("--gpu", type=int, help="GPU to use (default: the one with the least memory in use)")
    args = parser.parse_args()

    # the GPU must be chosen before PyTorch starts
    gpu = args.gpu if args.gpu is not None else freest_gpu()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"Using GPU {gpu}")

    import generate
    import numpy as np
    from inpaint import inpaint

    base = os.path.splitext(args.job)[0]
    with tempfile.TemporaryDirectory() as folder:
        with zipfile.ZipFile(args.job) as z:
            z.extractall(folder)
        with open(os.path.join(folder, "job.json")) as f:
            job = json.load(f)
        print(f"Job: {job}")
        pipeline = generate.load_pipeline()
        points = np.load(os.path.join(folder, job["points"]))
        for seed in args.seeds or [job["seed"]]:
            out = f"{base}_seed{seed}.glb"
            inpaint(pipeline, os.path.join(folder, job["model"]), os.path.join(folder, job["picture"]), points, out,
                    grow=job["grow"], seed=seed, steps=job["steps"])
            print(f"Result: {out}")


if __name__ == "__main__":
    main()

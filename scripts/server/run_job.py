"""Run an inpainting job package (made in Blender with "Save job for server") on a GPU server.

The package is one .zip: the model, the marked area, the picture and the settings. This is meant
for the HCI group's GPU server, which is only reachable through the uni VDI: upload the zip in
JupyterLab, run this, download the result.

Usage (TRELLIS.2 environment):
    source activate.sh
    python scripts/server/run_job.py uploads/tavern_job.zip                 # the job's own seed
    python scripts/server/run_job.py uploads/tavern_job.zip --seeds 1 2 3   # several versions, TRELLIS.2 loaded once
    python scripts/server/run_job.py uploads/tavern_job.zip --gpu 2         # this GPU instead of the freest one

Writes <job>_seed<N>.glb next to the zip. Without --gpu, the GPU with the least memory in use is
taken, so jobs don't land on a GPU someone else is using.

A job can also ask for several versions at once ("variants" in its job.json, e.g. both methods to
compare them: [{"method": "repaint"}, {"method": "inversion", "start": "random"}]), several "seeds",
and bring a "source" picture of the original model for the inversion method; with a "picture_mask"
(the 2D mask of the edit, the source then being the original's render of the same view), TRELLIS.2
sees the original picture outside the mask ("swap": false turns that off). Every version is then
written as <job>_<version>_seed<N>.glb.
"""
import _paths  # noqa: F401  (the scripts of the other folders)
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


# what a job's versions may set, with their checks (inpaint.py's options)
VARIANT_CHECKS = {
    "method": lambda v: v if v in ("repaint", "inversion") else None,
    "start": lambda v: v if v in ("inverted", "random") else None,
    "soft": lambda v: int(v) if 0 <= int(v) <= 6 else None,
    "swap": lambda v: bool(v),
    "solver": lambda v: v if v in ("midpoint", "euler") else None,
    "inject": lambda v: float(v) if 0 <= float(v) <= 1 else None,
    "room": lambda v: float(v) if 0 <= float(v) <= 0.5 else None,
}


def variants(job):
    """The versions the job asks for, checked (default: one, with the job's method or repaint)."""
    checked = []
    for variant in job.get("variants") or [{"method": job.get("method", "repaint")}]:
        version = {}
        for key, value in variant.items():
            value = VARIANT_CHECKS[key](value) if key in VARIANT_CHECKS else None
            if value is None:
                raise ValueError(f"unknown or out-of-range setting in the job: {key}={variant[key]}")
            version[key] = value
        checked.append(version)
    return checked


def name(version):
    """inversion-random-soft2 for {"method": "inversion", "start": "random", "soft": 2}."""
    parts = [version.get("method", "repaint")]
    if version.get("start", "inverted") != "inverted":
        parts.append(version["start"])
    if version.get("soft"):
        parts.append(f"soft{version['soft']}")
    if version.get("swap") is False:
        parts.append("noswap")
    if version.get("solver", "midpoint") != "midpoint":
        parts.append(version["solver"])
    if version.get("inject", 1.0) != 1.0:
        parts.append(f"inject{version['inject']:g}")
    return "-".join(parts)


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
        versions = variants(job)
        seeds = args.seeds or [int(s) for s in job.get("seeds", [job["seed"]])]
        source = os.path.join(folder, os.path.basename(job["source"])) if job.get("source") else None
        picture_mask = os.path.join(folder, os.path.basename(job["picture_mask"])) if job.get("picture_mask") else None
        for path in (source, picture_mask):
            if path and not os.path.isfile(path):
                raise ValueError(f"the job's {os.path.basename(path)} is missing")
        pipeline = generate.load_pipeline()
        points = np.load(os.path.join(folder, job["points"]))
        for version in versions:
            for seed in seeds:
                out = f"{base}_{name(version)}_seed{seed}.glb" if "variants" in job else f"{base}_seed{seed}.glb"
                inpaint(pipeline, os.path.join(folder, job["model"]), os.path.join(folder, job["picture"]), points,
                        out, grow=job["grow"], seed=seed, steps=job["steps"], source_path=source,
                        picture_mask_path=picture_mask, **version)
                print(f"Result: {out}")


if __name__ == "__main__":
    main()

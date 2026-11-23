"""Runs a Python script as if the GPU had only 24 GB (an RTX 3090 / 4090), to see on the uni server's A40 (46 GB)
whether a step fits a home PC: PyTorch may then take at most VRAM_GB of the GPU (default 22.5: of a 3090's 24 GB
the CUDA context and the desktop take ~1-1.5); more fails with "out of memory" just as it would there. At the end
it prints the peak (what PyTorch held, and the most it had reserved).

    CUDA_VISIBLE_DEVICES=0 python scripts/tools/as_24gb.py scripts/generation/generate.py picture.png ...
    VRAM_GB=20 python scripts/tools/as_24gb.py ...      (another limit)

Only PyTorch's own memory is limited (nearly everything here: TRELLIS.2, Pixal3D, the rig head, nvdiffrast's
tensors); the peak line says how much it really needed.
"""
import atexit
import os
import runpy
import sys

# as the scripts set it themselves (before torch loads): memory freed in pieces can be reused for bigger blocks
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

GB = float(os.environ.get("VRAM_GB", 22.5))


def peak():
    if torch.cuda.is_initialized():
        print(f"[as_24gb] peak: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB used, "
              f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GB reserved (limit {GB} GB)", flush=True)


if __name__ == "__main__":
    os.environ["VRAM_GB"] = str(GB)  # scripts/hardware.py then picks the profile for this size
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, GB * 2**30 / total), 0)
    atexit.register(peak)
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")

"""The machine's profile: how the GPU steps run on it (configs/hardware/<name>.json).

Two profiles, one per machine this project runs on:
  local_24gb    the home PC (RTX 3090 24 GB, Ryzen 9 5950X, 64 GB RAM, Fedora 44): the models wait in RAM and go to
                the GPU one at a time (low VRAM)
  server_48gb   the uni server (A40 46 GB): the models stay on the GPU (faster), the biggest settings

Which one: HARDWARE_PROFILE=local (or server, or a file's full name), else by the GPU's own size (under 30 GB:
local_24gb). Every GPU script reads its section, e.g.

    from hardware import profile
    p = profile()                 # the whole profile (a dict)
    p["pixal3d"]["low_vram"]      # one setting
    python scripts/hardware.py    # which profile this machine gets, and why

A setting the scripts are given on their command line wins over the profile.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.join(os.path.dirname(HERE), "configs", "hardware")


def gpu_gb():
    """The first visible GPU's memory in GB (0 without one)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        pass
    return 0.0


def pick():
    """(profile name, why)."""
    name = os.environ.get("HARDWARE_PROFILE")
    if name:
        return ALIASES.get(name, name), "HARDWARE_PROFILE"
    limit = os.environ.get("VRAM_GB")  # scripts/tools/as_24gb.py: playing a smaller GPU
    gb = float(limit) if limit else gpu_gb()
    return ("local_24gb" if gb < 30 else "server_48gb"), f"the GPU has {gb:.0f} GB"


ALIASES = {"local": "local_24gb", "24gb": "local_24gb", "pc": "local_24gb",
           "server": "server_48gb", "48gb": "server_48gb", "a40": "server_48gb"}
_cache = {}


def profile(name=None):
    name = name or pick()[0]
    if name not in _cache:
        with open(os.path.join(CONFIGS, name + ".json")) as f:
            _cache[name] = json.load(f)
    return _cache[name]


def setting(section, key, default=None):
    return profile().get(section, {}).get(key, default)


if __name__ == "__main__":
    name, why = pick()
    print(f"profile: {name} ({why})")
    print(json.dumps(profile(name), indent=1))

"""The rig head: reads a humanoid skeleton out of TRELLIS.2's shape latent (rig_data.py makes its training data).

TRELLIS.2's shape step ends in a sparse latent: a few thousand voxels of the 64^3 grid, 32 numbers each. The rig head
is a small transformer over those voxels (each sees all the others, so it knows left from right and arm from leg) that
says for every voxel
  skin     which of the 22 body bones moves it (22 weights, softmax)
  votes    for each of the 52 joints (22 body + 30 finger), how sure the voxel is that the joint is near it (a score)
           and where the joint is from the voxel's centre (an offset). A joint = the score-weighted mean of the votes
so the skeleton and the skin weights come from one look at the latent. The fingers are learnt only from the rigs that
have them (has_fingers); the toes' skin is left out for the rigs whose toes rig_data.py placed itself (has_toes).

    source activate.sh
    python scripts/rigging/rig_head.py train --data datasets/rig_latents --out local_ai_models/rig_head
    python scripts/rigging/rig_head.py predict local_ai_models/rig_head/best.pt in.npz [more.npz] --out folder

predict writes <out>/<name>.npz in rig_data.py's format (coords, feats, joints, skin, fingers, has_fingers), so
rig_picture.py draws it. A model from TRELLIS.2 or Pixal3D is first encoded like the training data (rig_from_glb.py).
"""
import _paths  # noqa: F401
import argparse
import glob
import json
import math
import os
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rig_data import BONES, FINGERS, GRID, TOES

FOOT = [BONES.index("foot_l"), BONES.index("foot_r")]
JOINTS = len(BONES) + len(FINGERS)  # 52
HELD_OUT = 3  # % of the models kept for checking (by a hash of the name: always the same ones)


def held_out(name):
    return zlib.crc32(name.encode()) % 100 < HELD_OUT


# ---------------------------------------------------------------- data

def load(path):
    d = np.load(path)
    return {
        "coords": torch.from_numpy(d["coords"].astype(np.int64)),
        "feats": torch.from_numpy(d["feats"].astype(np.float32)),
        "skin": torch.from_numpy(d["skin"].astype(np.float32)),
        "joints": torch.from_numpy(np.concatenate([d["joints"], d["fingers"]]).astype(np.float32)),
        "has": torch.from_numpy(np.concatenate([
            [not (k in TOES and "has_toes" in d and not bool(d["has_toes"])) for k in range(len(BONES))],
            d["has_fingers"]]).astype(bool)),
        "has_toes": bool(d["has_toes"]) if "has_toes" in d else True,
    }


def batches(models, budget, shuffle):
    """Lists of models of at most `budget` voxels together (the transformer's cost)."""
    order = np.random.permutation(len(models)) if shuffle else np.arange(len(models))
    batch, size = [], 0
    for i in order:
        n = len(models[i]["coords"])
        if batch and size + n > budget:
            yield batch
            batch, size = [], 0
        batch.append(models[i])
        size += n
    if batch:
        yield batch


def collate(batch, device):
    cat = lambda key: torch.cat([m[key] for m in batch]).to(device, non_blocking=True)
    lengths = torch.tensor([len(m["coords"]) for m in batch])
    return {
        "coords": cat("coords"), "feats": cat("feats"), "skin": cat("skin"),
        "joints": torch.stack([m["joints"] for m in batch]).to(device),
        "has": torch.stack([m["has"] for m in batch]).to(device),
        "has_toes": torch.tensor([m["has_toes"] for m in batch], device=device),
        "cu": F.pad(lengths.cumsum(0), (1, 0)).to(device, torch.int32),
        "max": int(lengths.max()),
        "which": torch.repeat_interleave(torch.arange(len(batch)), lengths).to(device),
    }


# ---------------------------------------------------------------- model

def centres(coords):
    return (coords.float() + 0.5) / GRID - 0.5  # TRELLIS.2's cube, like the joints


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, cu, max_len):
        from flash_attn import flash_attn_varlen_qkvpacked_func
        qkv = self.qkv(self.norm1(x)).view(len(x), 3, self.heads, -1)
        a = flash_attn_varlen_qkvpacked_func(qkv.to(torch.bfloat16), cu, max_len).reshape(len(x), -1)
        x = x + self.proj(a.to(x.dtype))
        return x + self.mlp(self.norm2(x))


class RigHead(nn.Module):
    def __init__(self, dim=384, depth=8, heads=6, freqs=8):
        super().__init__()
        self.config = dict(dim=dim, depth=depth, heads=heads, freqs=freqs)
        self.register_buffer("freqs", 2.0 ** torch.arange(freqs) * math.pi, persistent=False)
        self.embed = nn.Sequential(nn.Linear(32 + 3 + 6 * freqs, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, len(BONES) + JOINTS * 4)  # skin, vote scores, vote offsets

    def forward(self, b):
        p = centres(b["coords"])
        angles = (p[:, :, None] * self.freqs).flatten(1)
        x = self.embed(torch.cat([b["feats"], p, angles.sin(), angles.cos()], 1))
        for block in self.blocks:
            x = block(x, b["cu"], b["max"])
        out = self.out(self.norm(x)).float()
        skin, score, offset = out.split([len(BONES), JOINTS, JOINTS * 3], 1)
        votes = p[:, None] + offset.view(-1, JOINTS, 3) * 0.25  # where each voxel puts each joint
        # per model: softmax of the scores over its voxels
        n = len(b["joints"])
        top = torch.full((n, JOINTS), -1e9, device=x.device).scatter_reduce(
            0, b["which"][:, None].expand(-1, JOINTS), score, "amax")
        w = (score - top[b["which"]]).exp()
        total = torch.zeros(n, JOINTS, device=x.device).index_add(0, b["which"], w)
        w = w / total[b["which"]]
        joints = torch.zeros(n, JOINTS, 3, device=x.device).index_add(0, b["which"], w[..., None] * votes)
        return skin, joints


def losses(skin, joints, b):
    logp = skin.log_softmax(1)
    no_toes = ~b["has_toes"][b["which"]]
    if no_toes.any():  # their toes' skin was left in the feet: compare foot + toe with the foot alone
        merged = logp.clone()
        for foot, toe in zip(FOOT, TOES):
            merged[:, foot] = torch.logaddexp(logp[:, foot], logp[:, toe])
            merged[:, toe] = -1e4
        logp = torch.where(no_toes[:, None], merged, logp)
    target = b["skin"] / b["skin"].sum(1, keepdim=True).clamp_min(1e-6)
    skin_loss = -(target * logp).sum(1).mean()
    error = (joints - b["joints"]).norm(dim=2)  # (n, 52), cube units (the model is ~0.9 tall)
    has = b["has"].float()
    body, fingers = has[:, :len(BONES)], has[:, len(BONES):]
    body_err = (error[:, :len(BONES)] * body).sum() / body.sum()
    finger_err = (error[:, len(BONES):] * fingers).sum() / fingers.sum().clamp_min(1)
    skin_acc = (logp.argmax(1) == target.argmax(1)).float().mean()
    return skin_loss, body_err, finger_err, skin_acc


# ---------------------------------------------------------------- train

def train(args):
    torch.manual_seed(0)
    np.random.seed(0)
    done = json.load(open(os.path.join(args.data, "report.json")))["done"]
    names = sorted(n.strip("/").replace("/", "_") for n in done)
    paths = [os.path.join(args.data, n + ".npz") for n in names]
    paths = [p for p in paths if os.path.exists(p)]
    print(f"loading {len(paths)} models", flush=True)
    models = [load(p) for p in paths]
    names = [os.path.basename(p)[:-4] for p in paths]
    val = [m for m, n in zip(models, names) if held_out(n)]
    tr = [m for m, n in zip(models, names) if not held_out(n)]
    print(f"train {len(tr)}  check {len(val)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    model = RigHead().to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    step, best, start = 0, 1e9, time.time()
    log = open(os.path.join(args.out, "log.txt"), "a")

    def lr_at(s):
        warm = 1000
        return args.lr * min(1, s / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps)))

    def check():
        model.eval()
        sums = np.zeros(4)
        count = 0
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for batch in batches(val, args.budget, False):
                b = collate(batch, device)
                sums += np.array([float(v) for v in losses(*model(b), b)]) * len(batch)
                count += len(batch)
        model.train()
        return sums / count

    while step < args.steps:
        for batch in batches(tr, args.budget, True):
            b = collate(batch, device)
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            with torch.autocast("cuda", torch.bfloat16):
                skin_loss, body_err, finger_err, acc = losses(*model(b), b)
            loss = skin_loss + 10 * body_err + 5 * finger_err
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                line = (f"step {step}  skin {skin_loss:.3f} ({acc:.1%})  body {body_err * 100:.2f}  "
                        f"fingers {finger_err * 100:.2f}  lr {lr_at(step):.1e}  {(time.time() - start) / 60:.0f} min")
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            if step % args.check_every == 0 or step == args.steps:
                s, body, fing, acc = check()
                line = (f"CHECK step {step}  skin {s:.3f} ({acc:.1%})  body {body * 100:.2f}  fingers {fing * 100:.2f}"
                        "  (joint errors in % of the cube, the model is ~90% tall)")
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                state = {"model": model.state_dict(), "config": model.config, "step": step,
                         "check": {"skin": float(s), "skin_acc": float(acc), "body": float(body), "fingers": float(fing)}}
                torch.save(state, os.path.join(args.out, "last.pt"))
                if body < best:
                    best = body
                    torch.save(state, os.path.join(args.out, "best.pt"))
            if step >= args.steps:
                break
    print("TRAINING DONE", flush=True)


# ---------------------------------------------------------------- predict

def load_head(path, device="cuda"):
    state = torch.load(path, map_location=device, weights_only=False)  # our own file
    model = RigHead(**state["config"]).to(device).eval()
    model.load_state_dict(state["model"])
    return model


def predict_one(model, coords, feats, device="cuda"):
    """coords (L, 3) on the 64^3 grid, feats (L, 32) -> joints (22, 3), fingers (30, 3), skin (L, 22)."""
    m = {"coords": torch.as_tensor(np.asarray(coords), dtype=torch.int64),
         "feats": torch.as_tensor(np.asarray(feats, np.float32)),
         "skin": torch.zeros(len(coords), len(BONES)), "joints": torch.zeros(JOINTS, 3),
         "has": torch.ones(JOINTS, dtype=bool), "has_toes": True}
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        skin, joints = model(collate([m], device))
    joints = joints[0].float().cpu().numpy()
    return joints[:len(BONES)], joints[len(BONES):], skin.float().softmax(1).cpu().numpy()


def predict(args):
    model = load_head(args.checkpoint)
    os.makedirs(args.out, exist_ok=True)
    for path in args.inputs:
        d = np.load(path)
        joints, fingers, skin = predict_one(model, d["coords"], d["feats"])
        out = os.path.join(args.out, os.path.basename(path))
        np.savez(out, coords=d["coords"], feats=d["feats"], joints=joints.astype(np.float32),
                 fingers=fingers.astype(np.float32), has_fingers=np.ones(len(FINGERS), bool),
                 skin=skin.astype(np.float16))
        if "joints" in d:
            print(f"{out}: body joints off by {np.linalg.norm(joints - d['joints'], axis=1).mean() * 100:.2f}% "
                  "of the cube")
        else:
            print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="datasets/rig_latents")
    t.add_argument("--out", default="local_ai_models/rig_head")
    t.add_argument("--steps", type=int, default=40000)
    t.add_argument("--budget", type=int, default=60000, help="voxels per batch")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--check-every", type=int, default=2000)
    p = sub.add_parser("predict")
    p.add_argument("checkpoint")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()

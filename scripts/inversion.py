"""Flow inversion with key-value injection: regenerate an area of a TRELLIS.2 model so that the new
part grows into the kept part (inpaint.py's method="inversion"; method="repaint" is the simpler way).

The method and its first code are from VoxHammer (Li et al. 2025, arXiv 2508.19247,
github.com/Nelipot-Lee/VoxHammer, MIT licence), made there for TRELLIS; this is the same method for
TRELLIS.2's three stages (structure, shape, colours).

  flow inversion    TRELLIS.2 makes a latent by following a velocity field from noise (t = 1) to the
                    result (t = 0). Followed the other way, from the original model's latent to noise,
                    it gives the original's own path: its latent at every time of the schedule. While the
                    area is regenerated, the kept part is set to this path at every step, so it looks to
                    the model exactly like what it would have made itself. (RePaint instead adds fresh
                    random noise to the original, noise the model never made.)
  key-value         In every self-attention layer the tokens learn about each other through keys and
  injection         values. Before every evaluation, the original's path at that time goes through the
                    model and the keys and values of the kept tokens are recorded; in the evaluation of
                    the edited latent they replace the ones computed there. So the new area attends to
                    exactly the original around it.
  midpoint steps    Every step uses the velocity in its middle (second order, as RF-Solver, Wang et al.
                    2024), so inversion and regeneration follow the flow more exactly than Euler steps.

Differences to VoxHammer's code: TRELLIS.2's latents are much larger (up to ~100k tokens x 30 layers x
1536 channels), so the keys and values are recomputed at every evaluation instead of stored for all
steps (the same keys and values, twice the model evaluations); they are recorded once with the
original's picture and injected into the guided and the unguided evaluation alike.
"""
import torch
import torch.nn.functional as F

from trellis2.modules.attention.full_attn import scaled_dot_product_attention
from trellis2.modules.attention.rope import RotaryPositionEmbedder
from trellis2.modules.sparse import SparseTensor
from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention


def schedule(steps, rescale_t):
    """The sampler's times, from 1 (noise) to 0 (finished), as FlowEulerSampler.sample makes them."""
    t = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    return (rescale_t * t / (1 + (rescale_t - 1) * t)).tolist()


def tokens(x):
    """The latent as (tokens, channels): a SparseTensor's features, or a dense grid flattened as the
    structure model does it (x[0] is channels x depth x height x width)."""
    return x.feats if isinstance(x, SparseTensor) else x[0].reshape(x.shape[1], -1).T


def with_tokens(x, feats):
    if isinstance(x, SparseTensor):
        return x.replace(feats)
    return feats.T.reshape(x.shape).contiguous()


def distance_to(mask, most):
    """For every voxel: its distance to the nearest masked voxel, in steps of the 3x3x3 neighbourhood
    (0 inside the mask), up to most + 1."""
    distance = torch.full(mask.shape, most + 1, dtype=torch.long, device=mask.device)
    grown = mask.clone()
    distance[grown] = 0
    for i in range(1, most + 1):
        grown = F.max_pool3d(grown[None, None].float(), 3, 1, 1)[0, 0] > 0
        distance[grown & (distance > most)] = i
    return distance


def soft_weight(distance, width):
    """How much of the new latent a kept voxel at this distance from the area takes: falling off like
    a Gaussian over `width` voxels, 0 beyond (width 0: a hard edge, as in VoxHammer's code)."""
    if width <= 0:
        return torch.zeros(distance.shape, device=distance.device)
    weight = torch.exp(-0.5 * (distance.float() / (width / 2)) ** 2)
    return torch.where(distance > width, 0.0, weight)


class Kept:
    """The kept tokens: where they are in the edited latent (`target`) and in the original's (`source`),
    and how much of the new latent each one takes (`weight`, 0 = all original)."""

    def __init__(self, target, source, weight):
        self.target, self.source, self.weight = target, source, weight

    def blend(self, x, original):
        feats = tokens(x).clone()
        w = self.weight[:, None].to(feats.dtype)
        feats[self.target] = w * feats[self.target] + (1 - w) * tokens(original)[self.source].to(feats.dtype)
        return with_tokens(x, feats)


class KVInjection:
    """In every self-attention layer of `model`: in "record" mode, keeps the keys and values of the
    kept tokens of the original (in the RAM with `offload`); in "inject" mode, puts them in place of the
    ones of the kept tokens of the edited latent. Outside these modes, and outside this `with` block,
    the layers work as usual."""

    def __init__(self, model, kept, offload=False):
        self.layers = [block.self_attn for block in model.blocks]
        self.kept, self.offload, self.mode, self.saved = kept, offload, None, {}

    def exchange(self, i, k, v):
        """k, v: (tokens, heads, channels)."""
        if self.mode == "record":  # offload: kept in the RAM until they are needed
            self.saved[i] = tuple(x[self.kept.source].to("cpu") if self.offload else x[self.kept.source].clone()
                                  for x in (k, v))
        elif self.mode == "inject":
            k_old, v_old = (x.to(k.device) for x in self.saved[i])
            w = self.kept.weight[:, None, None].to(k.dtype)
            t = self.kept.target
            k, v = k.clone(), v.clone()
            k[t] = w * k[t] + (1 - w) * k_old
            v[t] = w * v[t] + (1 - w) * v_old
        return k, v

    def sparse_forward(self, layer, i):
        """SparseMultiHeadAttention.forward for self-attention ("full" mode), with the exchange."""
        def forward(x, context=None):
            qkv = layer._fused_pre(layer._linear(layer.to_qkv, x), num_fused=3)
            q, k, v = qkv.unbind(dim=-3)
            if layer.qk_rms_norm:
                q, k = layer.q_rms_norm(q), layer.k_rms_norm(k)
            if layer.use_rope:
                q, k = layer.rope(q, k)
            k_feats, v_feats = self.exchange(i, k.feats, v.feats)
            h = sparse_scaled_dot_product_attention(qkv.replace(torch.stack([q.feats, k_feats, v_feats], dim=1)))
            return layer._linear(layer.to_out, layer._reshape_chs(h, (-1,)))
        return forward

    def dense_forward(self, layer, i):
        """MultiHeadAttention.forward for self-attention ("full" mode, one model at a time), with the exchange."""
        def forward(x, context=None, phases=None):
            batch, length, _ = x.shape
            assert batch == 1, "one model at a time"
            q, k, v = layer.to_qkv(x).reshape(batch, length, 3, layer.num_heads, -1).unbind(dim=2)
            if layer.qk_rms_norm:
                q, k = layer.q_rms_norm(q), layer.k_rms_norm(k)
            if layer.use_rope:
                q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
                k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
            k, v = self.exchange(i, k[0], v[0])
            h = scaled_dot_product_attention(q, k[None], v[None])
            return layer.to_out(h.reshape(batch, length, -1))
        return forward

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            assert layer._type == "self" and layer.attn_mode == "full"
            sparse = hasattr(layer, "_fused_pre")
            layer.forward = self.sparse_forward(layer, i) if sparse else self.dense_forward(layer, i)
        return self

    def __exit__(self, *exc):
        for layer in self.layers:
            del layer.forward  # back to the class's own method
        self.saved.clear()
        self.mode = None


class Velocity:
    """One stage's flow model as a velocity field v(x, t), with guidance exactly as TRELLIS.2's sampler
    applies it (`params`: guidance_strength, guidance_interval, guidance_rescale)."""

    def __init__(self, sampler, model, cond, neg_cond, params, **model_kwargs):
        self.sampler, self.model, self.cond, self.neg_cond = sampler, model, cond, neg_cond
        self.params = {k: params[k] for k in ("guidance_strength", "guidance_interval", "guidance_rescale") if k in params}
        self.model_kwargs = model_kwargs

    def __call__(self, x, t):
        return self.sampler._inference_model(self.model, x, t, self.cond, neg_cond=self.neg_cond,
                                             **self.params, **self.model_kwargs)


def midpoint(velocity, x, t, t_next):
    """One second-order step from t to t_next (either way): the velocity in the middle of the step."""
    dt = t_next - t
    return x + dt * velocity(x + (dt / 2) * velocity(x, t), t + dt / 2)


def euler(velocity, x, t, t_next):
    """One first-order step (as TRELLIS.2's own sampler): half the model evaluations of midpoint()."""
    return x + (t_next - t) * velocity(x, t)


class Path:
    """The original's latent at every time of the schedule (from invert()), and between two of them."""

    def __init__(self):
        self.times, self.latents = [], []

    def add(self, t, x):
        self.times.append(t)
        self.latents.append(x)

    def at(self, t):
        for (t0, x0), (t1, x1) in zip(zip(self.times, self.latents), zip(self.times[1:], self.latents[1:])):
            if min(t0, t1) - 1e-9 <= t <= max(t0, t1) + 1e-9:
                a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return x0 if a <= 1e-9 else x1 if a >= 1 - 1e-9 else x0 + a * (x1 - x0)
        raise ValueError(f"time {t} is not on the path")


@torch.no_grad()
def invert(velocity, x0, times, on_step=None, step=midpoint):
    """The original's path from its latent x0 (t = 0) to noise (t = 1), by steps backwards through
    the schedule `times` (1 ... 0)."""
    path, x = Path(), x0
    backwards = times[::-1]
    path.add(backwards[0], x)
    for i, (t, t_next) in enumerate(zip(backwards[:-1], backwards[1:])):
        x = step(velocity, x, t, t_next)
        path.add(t_next, x)
        if on_step:
            on_step(i + 1)
    return path


@torch.no_grad()
def regenerate(velocity, start, times, path, kept, injection, record, on_step=None, step=midpoint, inject_below=1.0):
    """Regenerates from `start` (t = 1) through `times`, with the kept tokens on the original's `path`
    and their keys and values injected: before every evaluation at time t, record(t) sends the
    original's latent at t through the model with injection.mode = "record". With inject_below < 1 the
    keys and values are injected only from that time on (the early, rough steps are left free, so the
    area can take a new shape)."""
    def injected(x, t):
        if t > inject_below + 1e-9:
            return velocity(x, t)
        injection.mode = "record"
        record(t)
        injection.mode = "inject"
        v = velocity(x, t)
        injection.mode = None
        return v

    def kept_on_path(x, t):
        return kept.blend(x, path.at(t))

    x = start
    for i, (t, t_next) in enumerate(zip(times[:-1], times[1:])):
        x = kept_on_path(x, t)
        dt = t_next - t
        if step is euler:
            x = x + dt * injected(x, t)
        else:
            middle = kept_on_path(x + (dt / 2) * injected(x, t), t + dt / 2)
            x = x + dt * injected(middle, t + dt / 2)
        if on_step:
            on_step(i + 1)
    return kept_on_path(x, times[-1])

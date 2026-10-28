"""Regenerate one area of an existing 3D model with TRELLIS.2 and keep everything else (3D inpainting).

The model is first read back into TRELLIS.2's own 3D form (its three latents: rough structure, fine
shape, colours) with TRELLIS.2's encoders. Then the three stages run again as usual, but at every step
the area outside the mask is reset to the existing model (with the right amount of noise for that
step). Only the masked area is really generated, and it is generated to fit what is around it.
This is the RePaint idea (Lugmayr et al. 2022) applied to TRELLIS.2's 3D latents.

Usage (TRELLIS.2 environment):
    source activate.sh
    python scripts/inpaint.py outputs/tavern_1536_seed42.glb pictures/house/tavern.jpg --select chimney.glb
    python scripts/inpaint.py outputs/tavern_1536_seed42.glb pictures/house/tavern.jpg --box 0.1 0.3 -0.2 0.3 0.6 0.1
    python scripts/inpaint.py outputs/tavern_1536_seed42.glb pictures/house/tavern.jpg   # no mask: only reconstruct

The mask is given in the model's own coordinates: --select is a mesh of the faces to regenerate (e.g.
exported from Blender), --box two corners. The picture guides the new area; another picture or seed
gives another result. Writes <model>_inpaint_seed<N>.glb in the same place and size as the model.

Steps:
    1. load the model, fit it into TRELLIS.2's cube (Z up), and read it into the three latents
    2. turn the selection into a 64^3 voxel mask (grown by --grow voxels)
    3. structure: generate the 16^3 structure latent, keeping the cells outside the mask; then take
       the new occupied voxels inside the mask and the old ones outside
    4. shape and colours: generate the latents on these voxels, keeping the old voxels outside the mask
    5. decode, export like generate.py, and move the result back to the model's place and size
"""
import argparse
import gc
import io
import os
import time

import generate  # sets the environment variables and patches hole filling before TRELLIS.2 loads
import inversion

import numpy as np
import o_voxel
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

import trellis2.models as models
from trellis2.modules.sparse import SparseTensor

GRID = 64  # voxels per side of the latents at resolution 1024
RESOLUTION = 1024
SS_ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
SHAPE_ENCODER = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
TEX_ENCODER = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"


# ---------------------------------------------------------------- loading and placing the model

def load_model(path):
    """The model as one textured trimesh (all meshes of the file joined)."""
    scene = trimesh.load(path)
    meshes = [g.copy().apply_transform(t) for t, g in
              ((scene.graph[n][0], scene.geometry[scene.graph[n][1]]) for n in scene.graph.nodes_geometry)]
    assert len(meshes) == 1, "a model with one mesh is expected (e.g. from generate.py)"
    return meshes[0]


class Placement:
    """Moves points between the model's glTF coordinates (Y up) and TRELLIS.2's cube (-0.5..0.5, Z up),
    the same way TRELLIS.2's texturing pipeline does."""

    def __init__(self, vertices):
        low, high = vertices.min(0), vertices.max(0)
        self.center = (low + high) / 2
        self.scale = 0.99999 / (high - low).max()

    def to_cube(self, points):
        p = (np.asarray(points) - self.center) * self.scale
        return np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1)

    def from_cube_matrix(self):
        """4x4 matrix from the exported result (glTF axes, still cube-sized) back to the model's place."""
        m = np.eye(4)
        m[:3, :3] /= self.scale
        m[:3, 3] = self.center
        return m


# ---------------------------------------------------------------- reading the model into latents

def encode_shape(encoder, vertices, faces):
    """Fine shape latent (on a 64^3 grid) of a mesh in the cube, as TRELLIS.2's texturing pipeline does it."""
    voxels, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        torch.from_numpy(vertices).float(), torch.from_numpy(faces).long(), grid_size=RESOLUTION,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], face_weight=1.0, boundary_weight=0.2,
        regularization_weight=1e-2, timing=False)
    x = SparseTensor(feats=dual_vertices * RESOLUTION - voxels,
                     coords=torch.cat([torch.zeros_like(voxels[:, :1]), voxels], 1)).cuda()
    return encoder(x, x.replace(intersected).cuda())


def png(array):
    buffer = io.BytesIO()
    # TRELLIS.2's data tools take pictures from Blender, whose rows go bottom to top
    Image.fromarray(np.ascontiguousarray(array[::-1])).save(buffer, format="PNG")
    return {"image": buffer.getvalue(), "interpolation": "Linear", "extension": "REPEAT"}


def encode_colors(encoder, mesh, vertices):
    """Colour latent: the textures are put into 1024^3 voxels (as TRELLIS.2's data tools do), then encoded."""
    material = mesh.visual.material
    color = np.asarray(material.baseColorTexture.convert("RGBA"))
    pack = {"baseColorFactor": [1.0, 1.0, 1.0], "alphaFactor": 1.0, "metallicFactor": 0.0, "roughnessFactor": 0.5,
            "alphaMode": "OPAQUE", "alphaCutoff": 0.5, "baseColorTexture": png(color[..., :3]),
            "alphaTexture": None, "metallicTexture": None, "roughnessTexture": None}
    if getattr(material, "metallicRoughnessTexture", None) is not None:
        mr = np.asarray(material.metallicRoughnessTexture.convert("RGB"))
        pack.update(metallicFactor=1.0, roughnessFactor=1.0, metallicTexture=png(mr[..., 2]), roughnessTexture=png(mr[..., 1]))
    faces = np.asarray(mesh.faces)
    triangles = vertices[faces].astype(np.float32)
    normals = np.repeat(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])[:, None], 3, 1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
    dump = {"materials": [pack], "objects": [{
        "vertices": vertices.astype(np.float32), "faces": faces.astype(np.int32), "normals": normals.astype(np.float32),
        "uvs": np.asarray(mesh.visual.uv, np.float32)[faces], "mat_ids": np.zeros(len(faces), np.int32)}]}
    coords, attrs = o_voxel.convert.blender_dump_to_volumetric_attr(
        dump, grid_size=RESOLUTION, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], mip_level_offset=0)
    feats = torch.cat([attrs[k] for k in ("base_color", "metallic", "roughness", "alpha")], -1).float() / 255 * 2 - 1
    x = SparseTensor(feats, torch.cat([torch.zeros_like(coords[:, :1]), coords], 1).int()).cuda()
    return encoder(x)


# ---------------------------------------------------------------- the mask

def voxel_mask(points, grow):
    """64^3 boolean grid: the voxels the points fall in, grown by `grow` voxels in every direction."""
    mask = torch.zeros(GRID, GRID, GRID, dtype=torch.bool, device="cuda")
    if len(points):
        idx = torch.from_numpy(np.clip(((points + 0.5) * GRID).astype(int), 0, GRID - 1)).cuda()
        mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if grow > 0:
        mask = F.max_pool3d(mask[None, None].float(), 2 * grow + 1, 1, grow)[0, 0] > 0
    return mask


def keys(coords):
    c = coords[:, 1:].long()
    return (c[:, 0] * GRID + c[:, 1]) * GRID + c[:, 2]


def positions(coords, known_coords):
    """For every voxel in `coords`: its position in `known_coords`, and whether it is there at all."""
    k, kk = keys(coords), keys(known_coords)
    order = torch.argsort(kk)
    pos = torch.searchsorted(kk[order], k).clamp(max=len(kk) - 1)
    return order[pos], kk[order][pos] == k


def lookup(coords, known_coords, known_feats):
    """For every voxel in `coords`: the known features at the same voxel, and whether there are any."""
    position, found = positions(coords, known_coords)
    return known_feats[position], found


# ---------------------------------------------------------------- sampling with a fixed known part

class KeepKnown:
    """While active, every sampler step resets the known part to the known latent, noised for that step:
    x_t = (1 - t) x_0 + (sigma_min + (1 - sigma_min) t) noise, TRELLIS.2's flow."""

    def __init__(self, sampler, known, x0, noise, on_step=None):
        self.sampler, self.known, self.x0, self.noise, self.on_step = sampler, known, x0, noise, on_step
        self.step = 0

    def blend(self, x, t):
        noisy = (1 - t) * self.x0 + (self.sampler.sigma_min + (1 - self.sampler.sigma_min) * t) * self.noise
        return torch.where(self.known, noisy, x)

    def __enter__(self):
        original = self.sampler.sample_once

        def sample_once(model, x_t, t, t_prev, cond=None, **kwargs):
            out = original(model, x_t, t, t_prev, cond, **kwargs)
            if isinstance(out.pred_x_prev, SparseTensor):
                out.pred_x_prev = out.pred_x_prev.replace(self.blend(out.pred_x_prev.feats, t_prev))
            else:
                out.pred_x_prev = self.blend(out.pred_x_prev, t_prev)
            self.step += 1
            if self.on_step:
                self.on_step(self.step)
            return out

        self.sampler.sample_once = sample_once
        return self

    def __exit__(self, *exc):
        del self.sampler.sample_once  # back to the class's own method


class OnGpu:
    """TRELLIS.2 keeps its models on the CPU (low_vram) and moves each one to the GPU only while it runs."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        return self.model.cuda()

    def __exit__(self, *exc):
        self.model.cpu()
        torch.cuda.empty_cache()


def normalization(stats, device):
    return (torch.tensor(stats["mean"], device=device)[None], torch.tensor(stats["std"], device=device)[None])


# ---------------------------------------------------------------- main

def box_points(box, size):
    """Points filling a box (two corners), dense enough for the 64^3 mask of a model of this size."""
    low, high = np.minimum(box[:3], box[3:]), np.maximum(box[:3], box[3:])
    axes = [np.linspace(low[i], high[i], max(2, int((high[i] - low[i]) / size * GRID * 2))) for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


METHODS = ("repaint", "inversion")


def remove_background(pipeline, image):
    if pipeline.low_vram:
        pipeline.rembg_model.to(pipeline.device)
    rgba = pipeline.rembg_model(image)
    if pipeline.low_vram:
        pipeline.rembg_model.cpu()
    return np.asarray(rgba)


def picture_pair(pipeline, picture_path, source_path, mask_path):
    """The edited picture and the original's render of the same view, prepared for TRELLIS.2 with one
    and the same crop (as preprocess_image does it, around both objects together), and the 2D mask of
    the change, cropped alike. Saved as inpaint_input.png and inpaint_source_input.png for checking."""
    edited = Image.open(picture_path).convert("RGB")
    scale = min(1, 1024 / max(edited.size))
    size = (round(edited.width * scale), round(edited.height * scale))
    edited = edited.resize(size, Image.LANCZOS)
    source = Image.open(source_path).convert("RGB").resize(size, Image.LANCZOS)
    mask = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
    rgba = [remove_background(pipeline, image) for image in (edited, source)]
    ys, xs = np.nonzero(np.maximum(rgba[0][..., 3], rgba[1][..., 3]) > 0.8 * 255)
    center, side = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2), int(max(np.ptp(xs), np.ptp(ys)))
    box = (int(center[0]) - side // 2, int(center[1]) - side // 2, int(center[0]) + side // 2, int(center[1]) + side // 2)
    pictures = []
    for image, name in zip(rgba, ("inpaint", "inpaint_source")):
        crop = np.asarray(Image.fromarray(image).crop(box)).astype(np.float32) / 255
        picture = Image.fromarray((crop[..., :3] * crop[..., 3:4] * 255).astype(np.uint8))
        picture.save(os.path.join(generate.OUTPUT_DIR, f"{name}_input.png"))
        pictures.append(picture)
    return pictures[0], pictures[1], mask.crop(box)


def mixed_cond(cond, source_cond, picture_mask, resolution, patch=16):
    """The picture condition with the edited picture's tokens inside the 2D mask (and its global tokens:
    class and registers), and the original's outside it (VoxHammer's swap of the cross-attention keys
    and values: in TRELLIS.2 they depend on each picture token alone, so swapping the tokens is the same)."""
    grid = resolution // patch
    tokens = cond["cond"].shape[1]
    assert 0 <= tokens - grid * grid <= 8, f"unexpected picture tokens: {tokens} for a {grid}x{grid} grid"
    inside = np.asarray(picture_mask.resize((grid * patch, grid * patch), Image.NEAREST)) > 127
    inside = inside.reshape(grid, patch, grid, patch).any(axis=(1, 3)).reshape(-1)
    new = torch.cat([torch.ones(tokens - grid * grid, dtype=torch.bool), torch.from_numpy(inside)]).to(cond["cond"].device)
    return {"cond": torch.where(new[None, :, None], cond["cond"], source_cond["cond"]), "neg_cond": cond["neg_cond"]}


def inpaint(pipeline, model_path, picture_path, points, out, grow=2, seed=42, steps=generate.DEFAULT_STEPS, report=print,
            method="repaint", source_path=None, start="inverted", soft=0, picture_mask_path=None, swap=True,
            solver="midpoint"):
    """Regenerates the area around `points` (model coordinates; none = only rebuild) and writes `out`.
    `report` gets a short text for every stage and step.
    method: "repaint" keeps the rest with fresh noise at every step (KeepKnown); "inversion" keeps it on
    the original's own path, with its keys and values injected (inversion.py). For "inversion":
    source_path is a picture of the original model (default: the picture itself), start is where the
    structure of the area starts ("inverted": the original's noise, "random": new noise), and soft is
    the width of the soft edge in voxels (0: hard).
    picture_mask_path: the 2D mask of what changed in the picture; source_path is then the original's
    render from the same view, and with `swap` TRELLIS.2 sees the original picture outside the mask.
    solver: "midpoint" (second order, as VoxHammer) or "euler" (half the evaluations, faster)."""
    assert method in METHODS and start in ("inverted", "random") and solver in ("midpoint", "euler")
    step = getattr(inversion, solver)
    start_time = time.time()
    device = pipeline.device
    mesh = load_model(model_path)
    place = Placement(mesh.vertices)
    vertices = place.to_cube(mesh.vertices)
    report(f"1/6 Reading the model into TRELLIS.2 ({len(mesh.faces):,} faces)")
    with torch.no_grad():
        encoder = models.from_pretrained(SHAPE_ENCODER).cuda().eval()
        shape_known = encode_shape(encoder, vertices, np.asarray(mesh.faces))
        encoder = models.from_pretrained(TEX_ENCODER).cuda().eval()
        tex_known = encode_colors(encoder, mesh, vertices)
        occupied = torch.zeros(1, 1, GRID, GRID, GRID, device=device)
        c = shape_known.coords[:, 1:].long()
        occupied[0, 0, c[:, 0], c[:, 1], c[:, 2]] = 1
        encoder = models.from_pretrained(SS_ENCODER).cuda().eval()
        ss_known = encoder(occupied, sample_posterior=False).float()
        del encoder
    torch.cuda.empty_cache()
    print(f"    {len(shape_known.coords):,} shape voxels, {len(tex_known.coords):,} colour voxels")

    points = np.asarray(points, float).reshape(-1, 3)
    mask = voxel_mask(place.to_cube(points) if len(points) else points, grow if len(points) else 0)
    report(f"2/6 Mask: {int(mask.sum()):,} of {GRID ** 3:,} voxels will be regenerated ({method})")

    if picture_mask_path:
        assert source_path, "a picture mask needs the original's render of the same view (source)"
        image, source, picture_mask = picture_pair(pipeline, picture_path, source_path, picture_mask_path)
    else:
        image = generate.prepare_image(pipeline, picture_path, "inpaint")
        source = generate.prepare_image(pipeline, source_path, "inpaint_source") if source_path else image
    torch.manual_seed(seed)
    with torch.no_grad():
        cond_512 = pipeline.get_cond([image], 512)
        cond_1024 = pipeline.get_cond([image], 1024)
        source_512 = pipeline.get_cond([source], 512) if source_path else cond_512
        source_1024 = pipeline.get_cond([source], 1024) if source_path else cond_1024
        if picture_mask_path and swap:
            cond_512 = mixed_cond(cond_512, source_512, picture_mask, 512)
            cond_1024 = mixed_cond(cond_1024, source_1024, picture_mask, 1024)
            report("    picture: the edited one inside the 2D mask, the original's render outside")

        stage = "3/6 Structure"
        report(stage)
        progress = lambda what: lambda i: report(f"{stage}: {what} {i}/{steps}")
        sampler = pipeline.sparse_structure_sampler
        params = {**pipeline.sparse_structure_sampler_params, "steps": steps}
        flow = pipeline.models["sparse_structure_flow_model"]
        noise = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution, device=device)
        cell = GRID // flow.resolution
        cell_mask = F.max_pool3d(mask[None, None].float(), cell, cell)[0, 0] > 0
        if method == "repaint":
            with KeepKnown(sampler, ~cell_mask[None, None], ss_known, noise, progress("step")), OnGpu(flow):
                z = sampler.sample(flow, noise, **cond_512, **params, verbose=True,
                                   tqdm_desc="Sampling sparse structure").samples
        else:
            cells = torch.nonzero(~cell_mask.flatten()).squeeze(1)  # the kept cells, in the model's token order
            width = -(-soft // cell)  # the soft edge in cells
            weight = inversion.soft_weight(inversion.distance_to(cell_mask, width).flatten()[cells], width)
            kept = inversion.Kept(cells, cells, weight)
            z = follow_original(sampler, flow, params, steps, cond_512, source_512, ss_known, kept,
                                start=None if start == "inverted" else noise, progress=progress, step=step)
        with OnGpu(pipeline.models["sparse_structure_decoder"]) as decoder:
            new_occupied = decoder(z)[0, 0] > 0
        occupied = torch.where(mask, new_occupied, occupied[0, 0] > 0)
        coords = torch.argwhere(occupied).int()
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], 1)
        inside = mask[coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()]
        print(f"    {len(coords):,} voxels, {int(inside.sum()):,} of them new or regenerated")
        if method == "inversion":
            distance = inversion.distance_to(mask, soft)[coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()]

        stage = "4/6 Shape"
        report(stage)
        sampler = pipeline.shape_slat_sampler
        params = {**pipeline.shape_slat_sampler_params, "steps": steps}
        mean, std = normalization(pipeline.shape_slat_normalization, device)
        flow = pipeline.models["shape_slat_flow_model_1024"]
        shape_feats = (shape_known.feats.float() - mean) / std
        noise = SparseTensor(feats=torch.randn(len(coords), flow.in_channels, device=device), coords=coords)
        if method == "repaint":
            x0, found = lookup(coords, shape_known.coords, shape_feats)
            with KeepKnown(sampler, (found & ~inside)[:, None], x0, noise.feats, progress("step")), OnGpu(flow):
                shape = sampler.sample(flow, noise, **cond_1024, **params, verbose=True,
                                       tqdm_desc="Sampling shape SLat").samples
        else:
            kept, original = kept_tokens(coords, inside, distance, soft, shape_known.coords, shape_feats)
            shape = follow_original(sampler, flow, params, steps, cond_1024, source_1024, original, kept,
                                    start=noise, progress=progress, step=step)
        shape_norm = shape
        shape = shape * std + mean

        stage = "5/6 Colours"
        report(stage)
        sampler = pipeline.tex_slat_sampler
        params = {**pipeline.tex_slat_sampler_params, "steps": steps}
        tmean, tstd = normalization(pipeline.tex_slat_normalization, device)
        flow = pipeline.models["tex_slat_flow_model_1024"]
        tex_feats = (tex_known.feats.float() - tmean) / tstd
        noise = shape_norm.replace(torch.randn(len(coords), flow.in_channels - shape_norm.feats.shape[1], device=device))
        if method == "repaint":
            x0, found = lookup(coords, tex_known.coords, tex_feats)
            with KeepKnown(sampler, (found & ~inside)[:, None], x0, noise.feats, progress("step")), OnGpu(flow):
                tex = sampler.sample(flow, noise, concat_cond=shape_norm, **cond_1024, **params, verbose=True,
                                     tqdm_desc="Sampling texture SLat").samples
        else:
            kept, original = kept_tokens(coords, inside, distance, soft, tex_known.coords, tex_feats)
            original_shape, has = lookup(original.coords, shape_known.coords, shape_feats)  # its shape, as the model needs it
            original_shape = original_shape * has[:, None]
            tex = follow_original(sampler, flow, params, steps, cond_1024, source_1024, original, kept,
                                  start=noise, progress=progress, step=step, concat_cond=shape_norm,
                                  original_kwargs={"concat_cond": original.replace(original_shape)})
        tex = tex * tstd + tmean

        report("6/6 Decoding and saving (takes a few minutes)")
        result = pipeline.decode_latent(shape, tex, RESOLUTION)[0]
    # the export needs most of the GPU: drop everything else first
    del shape, tex, shape_norm, noise, z, shape_known, tex_known, ss_known, cond_512, cond_1024, source_512, source_1024
    gc.collect()
    torch.cuda.empty_cache()
    print(f"    {len(result.faces):,} faces; GPU memory still in use: {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    generate.save_glb(result, str(RESOLUTION), out, remesh=False)  # remeshing ~24M faces runs out of memory
    del result
    torch.cuda.empty_cache()
    scene = trimesh.load(out)
    scene.apply_transform(place.from_cube_matrix())
    scene.export(out, extension_webp=True)
    report(f"Done in {time.time() - start_time:.0f} s")
    return out


def kept_tokens(coords, inside, distance, soft, known_coords, known_feats):
    """For inversion in the shape and colour stages: the kept tokens (voxels of the original outside the
    area) as a map between the edited latent and the original's latent of just these voxels, which is
    what gets inverted (as VoxHammer does)."""
    position, found = positions(coords, known_coords)
    target = torch.nonzero(found & ~inside).squeeze(1)
    original = SparseTensor(feats=known_feats[position[target]], coords=coords[target])
    weight = inversion.soft_weight(distance[target], soft)
    return inversion.Kept(target, torch.arange(len(target), device=coords.device), weight), original


def follow_original(sampler, flow, params, steps, cond, source_cond, original, kept, start, progress,
                    original_kwargs=None, step=inversion.midpoint, **model_kwargs):
    """One stage with inversion: invert the original's latent (with the original's picture, without
    guidance), then regenerate from `start` (None: the original's noise) along it, with the kept tokens'
    keys and values injected. Returns the new latent."""
    original_kwargs = original_kwargs or {}
    times = inversion.schedule(steps, params["rescale_t"])
    unguided = {"guidance_strength": 1.0, "guidance_interval": [0.0, 1.0], "guidance_rescale": 0.0}
    with OnGpu(flow):
        source = inversion.Velocity(sampler, flow, source_cond["cond"], source_cond["neg_cond"], unguided,
                                    **original_kwargs)
        path = inversion.invert(source, original, times, progress("inverting"), step=step)
        edit = inversion.Velocity(sampler, flow, cond["cond"], cond["neg_cond"], params, **model_kwargs)
        if start is None:
            start = path.at(1.0)
        # the kept tokens' keys and values of all layers: on the GPU if they fit well, else in the RAM
        size = len(kept.target) * len(flow.blocks) * 2 * flow.model_channels * 2
        offload = size > 0.35 * torch.cuda.mem_get_info()[0]
        print(f"    inversion: {len(kept.target):,} kept tokens, keys and values {size / 2**30:.1f} GB "
              f"({'RAM' if offload else 'GPU'})")
        with inversion.KVInjection(flow, kept, offload) as injection:
            return inversion.regenerate(edit, start, times, path, kept, injection,
                                        record=lambda t: source(path.at(t), t), on_step=progress("step"),
                                        step=step)


def main():
    parser = argparse.ArgumentParser(description="Regenerate one area of a 3D model with TRELLIS.2 (3D inpainting)")
    parser.add_argument("model", help="the model to change (.glb with one textured mesh)")
    parser.add_argument("picture", help="picture that guides the new area (usually the model's own picture)")
    parser.add_argument("--select", help="mesh of the faces to regenerate, in the model's coordinates")
    parser.add_argument("--box", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                        help="regenerate everything inside this box (model coordinates)")
    parser.add_argument("--grow", type=int, default=2, help="grow the mask by this many voxels of 64 (default 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=generate.DEFAULT_STEPS)
    parser.add_argument("--method", choices=METHODS, default="repaint",
                        help="repaint: the rest kept with fresh noise; inversion: on the original's own path, "
                             "with its keys and values injected (inversion.py)")
    parser.add_argument("--source", help="inversion: a picture of the original model (default: the picture)")
    parser.add_argument("--start", choices=["inverted", "random"], default="inverted",
                        help="inversion: where the area's structure starts (the original's noise or new noise)")
    parser.add_argument("--soft", type=int, default=0, help="inversion: soft edge in voxels (default 0: hard)")
    parser.add_argument("--out", help="output file (default: <model>_inpaint_seed<N>.glb)")
    args = parser.parse_args()

    pipeline = generate.load_pipeline()
    points = [np.zeros((0, 3))]
    if args.select:
        selected = trimesh.load(args.select, force="mesh")
        points += [selected.vertices, trimesh.sample.sample_surface(selected, 200000, seed=0)[0]]
    if args.box:
        size = np.ptp(load_model(args.model).vertices, 0).max()
        points.append(box_points(np.array(args.box), size))
    out = args.out or os.path.splitext(args.model)[0] + f"_inpaint_seed{args.seed}.glb"
    inpaint(pipeline, args.model, args.picture, np.concatenate(points), out, args.grow, args.seed, args.steps,
            method=args.method, source_path=args.source, start=args.start, soft=args.soft)
    print(out)


if __name__ == "__main__":
    main()

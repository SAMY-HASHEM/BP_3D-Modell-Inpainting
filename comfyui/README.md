# BP Inpainting in ComfyUI (and in Blender)

The node pipeline (`scripts/pipeline/`) as ComfyUI nodes: drag in pictures or a video, see the data sheet,
the base model, every part before and after, and the final model in a 3D viewer, all in one graph.

```
1 Pictures (+ own pictures, video) → 2 Sort (data sheet) → 3 Base model → 4 Parts (easy → hard)
                                                                         → 5 Combine → 6 Hair → 99 Compare
```

## Set up (once)
```
ln -sfn ~/inpainting-3D/comfyui/bp_inpainting ~/ai-models/tools/ComfyUI/custom_nodes/bp_inpainting
~/ai-models/tools/ComfyUI/start.sh
```
In ComfyUI: **Workflows → "BP Inpainting - pictures to model"**. (Rebuild it after changing nodes:
`python3 comfyui/build_workflow.py --install` while ComfyUI runs.)

## The nodes (category "BP Inpainting")
| Node | What it does | Shows |
|---|---|---|
| BP 00 Pictures (folder) | all pictures and videos of a folder (relative to the project) | all pictures |
| BP 00 Add picture / Add video | one more picture / frames of a video, uploaded in ComfyUI (switched off in the workflow: Ctrl+B) | the frames |
| BP 01 Sort pictures | the vision AI's data sheet: every view with colour / grey, side, whole / close-up, parts | data sheet + the lists |
| BP 02 Base model | the whole figure with TRELLIS.2 (shape from one version, colours from the whole colour view) | renders + clay (shape alone) + 3D |
| BP 04 Refine part | one part zoomed in; `auto` = the lessons (colours only, the head also shape; one sharp picture) | pictures \| before \| after (front, sides, behind) + 3D |
| BP 05 Combine parts | the parts put into the base (the first part wins where two overlap) | base \| combined + 3D |
| BP 06 Hair strands | real strands with Blender's hair nodes, colour and length from the data sheet | renders + 3D + .blend |
| BP 99 Compare models | up to 4 models with the same 4 cameras, in colour or as clay | one sheet |

The 3D viewer is ComfyUI's own **Preview 3D** node (connect `model_3d`).

## How it runs
- The heavy work runs in the project's environment (`source activate.sh`), not in ComfyUI's Python; ComfyUI's own
  models are not used, so the GPU is free for TRELLIS.2 (one node at a time).
- ComfyUI re-runs a node only when its inputs or settings change. Every result is also kept in
  `outputs/pipeline/<name>/<node>/<settings hash>/` (with `settings.json`, `log.txt`, `compare.jpg`), so the same
  settings are never computed twice, even after a restart.
- **Cancel** in ComfyUI stops the running script.

## The same nodes in Blender
`blender/pipeline_nodes.py` is a node editor of its own in Blender ("BP Pipeline" in the node editor's tree
menu). Blender's Geometry Nodes can't run outside programs (TRELLIS.2, the vision AI), so it is a separate node
tree; only the hair node uses real Geometry Nodes. Both editors use `scripts/pipeline/nodes_core.py`, so they do
exactly the same and share one cache: a result made in ComfyUI is found in Blender and the other way round.
```
ln -sfn ~/inpainting-3D/blender/pipeline_nodes.py ~/.config/blender/5.2/scripts/addons/pipeline_nodes.py
```
Blender: Edit → Preferences → Add-ons → enable **BP Pipeline Nodes**. In a node editor choose the tree type
**BP Pipeline**, sidebar (N) → **BP** → **New pipeline**. Each node: its picture (data sheet, before / after),
**Picture** (big, in an Image Editor), **In scene** (its model into a collection of its own), **Run up to here**.

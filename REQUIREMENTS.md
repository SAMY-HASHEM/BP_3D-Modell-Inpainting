# Requirements and progress

Short state of the project, kept up to date so a new session doesn't need the whole history.
Rate = how far it is solved (estimate). Main method: **flow inversion with key-value injection**
(the "rewind", idea from VoxHammer, ported to TRELLIS.2) + picture swap + protected mask.

## Done or in progress

| # | Name | Description | Rate | Code |
|---|---|---|---|---|
| 1 | Model from pictures | `generate.py` (1 picture), `multi_image.py` (several views), `video_views.py` (turntable video) | 90 % | [generate.py](scripts/models/generate.py), [multi_image.py](scripts/models/multi_image.py), [video_views.py](scripts/models/video_views.py) |
| 2 | Best settings | which resolution / steps / guidance / faces give the most detail; 9 settings tried on the dancer model, still to compare | 20 % | [sweep.py](scripts/server/sweep.py), results in `outputs/cloud/view_00_000deg_generate_20261029-*` |
| 3 | Uni GPU server | inside the network: `inpaint_server.py`; outside: Sciebo + `cloud_worker.py` (parallel GPUs, code from GitHub, proxy fallback) | 90 % | [inpaint_server.py](scripts/server/inpaint_server.py), [cloud.py](scripts/server/cloud.py), [cloud_worker.py](scripts/server/cloud_worker.py), [run_job.py](scripts/server/run_job.py) |
| 4 | Space box | see-through box in Blender, arrows move a side, dots move a corner | 100 % | [inpaint_addon.py: draw_boxes](blender/inpaint_addon.py#L371), [INPAINT_GGT_box_sides](blender/inpaint_addon.py#L417) |
| 5 | Find part (AI mask) | word + clicks → Grounding DINO + SAM 2.1 on 16 views → votes per face; not always complete | 70 % | [mask_ai.py](scripts/mask/mask_ai.py), [inpaint_addon.py: INPAINT_OT_find_part](blender/inpaint_addon.py#L717), [best_directions](blender/inpaint_addon.py#L573), [vote](blender/inpaint_addon.py#L870) |
| 6 | OmniPart parts | parts from OmniPart moved onto the TRELLIS.2 model (`transfer_parts.py`); only for new models | 50 % | [omnipart_generate.py](scripts/models/omnipart_generate.py), [make_part_mask.py](scripts/mask/make_part_mask.py), [transfer_parts.py](scripts/mask/transfer_parts.py), [replace_part.py](scripts/regenerate/replace_part.py) |
| 7 | RePaint | old method, kept part with fresh noise; baseline | 100 % | [inpaint.py: KeepKnown](scripts/regenerate/inpaint.py#L164) |
| 8 | Flow inversion + KV injection | rewind + attention swap in all 3 stages; best result so far (wheel, crown) | 80 % | [inversion.py](scripts/regenerate/inversion.py) ([KVInjection](scripts/regenerate/inversion.py#L89)), [inpaint.py: follow_original](scripts/regenerate/inpaint.py#L429) |
| 9 | Picture swap | edited picture only inside the 2D mask, original render outside | 90 % | [inpaint.py: picture_pair](scripts/regenerate/inpaint.py#L236), [mixed_cond](scripts/regenerate/inpaint.py#L259) |
| 10 | Protected mask | Box = "Selected faces + empty space": part + room regenerated, rest exact; fixes colour bleeding | 85 % | [inpaint_addon.py: keep_the_rest](blender/inpaint_addon.py#L173) |
| 11 | Edited picture | Qwen-Image-Edit on the PC ("Make picture"), or a hand collage | 70 % | [edit_picture.py](scripts/regenerate/edit_picture.py) |
| 12 | Speed | inversion takes ~25 min per change (3× repaint); faster steps lost quality | 30 % | [inversion.py: midpoint / euler](scripts/regenerate/inversion.py) |
| 13 | Project structure | scripts in models / mask / regenerate / server / tools  | 90 % | [README.md](README.md#structure), [_paths.py](scripts/_paths.py) |

## Still aiming for

- Find the settings with the best detail (#2) and use them as defaults.
- Make inversion faster without losing quality (#12).
- Find part: SAM 3 once Meta grants access (#5).
- Small leftovers under new parts (e.g. the vent under the crown).

## New ideas and resources

- (to fill in together)

## Rules

- Uni server: one job at a time by day, 3 GPUs only after 24:00.
- Commits as SAMY HASHEM, no AI attribution; push to GitLab + GitHub.
- Slides for the supervisor: German, simple, only this week's work, no character pictures.

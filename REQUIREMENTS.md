# Requirements and progress

Short state of the project, kept up to date so a new session doesn't need the whole history.
Rate = how far it is solved (estimate). Main method: **flow inversion with key-value injection**
(the "rewind", idea from VoxHammer, ported to TRELLIS.2) + picture swap + protected mask.

## Done or in progress

| # | Name | Description | Rate |
|---|---|---|---|
| 1 | Model from pictures | `generate.py` (1 picture), `multi_image.py` (several views), `video_views.py` (turntable video) | 90 % |
| 2 | Best settings | which resolution / steps / guidance / faces give the most detail; sweep running on the server (`sweep.py`) | 20 % |
| 3 | Uni GPU server | inside the network: `inpaint_server.py`; outside: Sciebo + `cloud_worker.py` (parallel GPUs, code from GitHub, proxy fallback) | 90 % |
| 4 | Space box | see-through box in Blender, arrows move a side, dots move a corner | 100 % |
| 5 | Find part (AI mask) | word + clicks → Grounding DINO + SAM 2.1 on 16 views → votes per face; not always complete | 70 % |
| 6 | OmniPart parts | parts from OmniPart moved onto the TRELLIS.2 model (`transfer_parts.py`); only for new models | 50 % |
| 7 | RePaint | old method, kept part with fresh noise; baseline | 100 % |
| 8 | Flow inversion + KV injection | rewind + attention swap in all 3 stages; best result so far (wheel, crown) | 80 % |
| 9 | Picture swap | edited picture only inside the 2D mask, original render outside | 90 % |
| 10 | Protected mask | Box = "Selected faces + empty space": part + room regenerated, rest exact; fixes colour bleeding | 85 % |
| 11 | Edited picture | Qwen-Image-Edit on the PC ("Make picture"), or a hand collage | 70 % |
| 12 | Speed | inversion takes ~25 min per change (3× repaint); faster steps lost quality | 30 % |
| 13 | Project structure | scripts in models / mask / regenerate / server / tools (branch `cleanup`, merge after the sweep) | 90 % |

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

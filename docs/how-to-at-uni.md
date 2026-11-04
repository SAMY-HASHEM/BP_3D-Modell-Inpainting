# How to: working from a uni computer

`pictures/` and `outputs/` are not in Git (pictures belong to other artists, models are too big). They travel
through your private Sciebo. Only `python3` is needed for this; nothing to install.

## A new computer (once)
```
git clone https://github.com/SAMY-HASHEM/BP_3D-Modell-Inpainting.git
cd BP_3D-Modell-Inpainting
python3 scripts/server/cloud.py setup      # your Sciebo login + app password
python3 scripts/server/cloud.py device     # makes outputs/ and pictures/, fetches pictures and results
```
On Windows write `python` instead of `python3`. `device` also fetches every finished result (over 1 GB).

## Every time
```
git pull
python3 scripts/server/cloud.py watch      # every 10 minutes: all new models, heat maps, zips (stop: Ctrl+C)
```
Results land in `outputs/server_runs/<job>/`. Open the `.glb` files in Blender.

## New pictures
Put them in `pictures/<name>/`, then `python3 scripts/server/cloud.py pictures up`. On another computer:
`python3 scripts/server/cloud.py pictures down`.

## Two ways to use the server
| | Cloud way | Direct way |
|---|---|---|
| For | the whole pipeline (`cloud.py pipeline`) | single inpaintings from Blender, answers right away |
| Works from | anywhere | only inside the uni's network |
| Server side | `cloud_worker.py` (running now) | `inpaint_server.py --host 0.0.0.0` in a JupyterLab terminal |
| Blender add-on | Connection "Outside (cloud)" | Connection "Inside the network" |

JupyterLab: `https://nc.hcigroup.de`. On campus try it directly in the browser; from home only through the VDI.

One job at a time: while a pipeline job runs it holds one GPU, so don't start the direct server next to it.
Don't `git pull` on the server while a job runs: the worker fetches its code itself.

# The web app: front end and back end

The project as one web page: add pictures, write prompts, press Run, see the 3D result. Made after the sketches in
`UI-skitch/`.

| Part | File | What it is |
|---|---|---|
| **Front end** | `web/app.html` | the page itself: one file, no build step. Two colours (sand gold and black), hieroglyphs behind it |
| **Back end** | `scripts/pipeline/flow_page.py` | a small server on this PC only (127.0.0.1:8777). It writes flow files and starts `flow.py` |
| the engine | `scripts/pipeline/flow.py`, `nodes_core.py` | unchanged: the same nodes and result folders as from the terminal or Blender |
| the older page | `web/index.html` (at `/`) | the node chain from left to right; still there, linked from the menu |

```
source activate.sh
python scripts/pipeline/flow_page.py --open      # then http://127.0.0.1:8777/app
```

Opened as a file (double click on `app.html`) the page only shows its look: Run needs the back end.

## What happens on Run

```
the page (app.html)                         the back end (flow_page.py)                  the engine
───────────────────                         ───────────────────────────                  ──────────
Add data, prompts, Run
  ├─ POST /api/upload?flow&role&name  ───►  pictures/uploads/<project>/<object|part>/
  ├─ POST /api/app/create             ───►  flows/<project>.json  (made_by: "app")
  ├─ POST /api/run                    ───►  starts  flow.py flows/<project>.json  ───►  outputs/pipeline/<project>/...
  └─ GET  /api/flow?name= (every 3 s) ◄───  every node's state, pictures, model
        the steps tick off, the newest model shows in 3D
```

A project on the page **is** a flow file. The page writes the same kind of file as `flows/girl_crown.json`:

- only an **object**: input → data sheet → base model
- an object and a **part**: the same, then edit → inpaint shape → judge → details → judge → colours → judge
- an object and **a part from words** (*+ Describe a new part in words*, no picture of the part needed): the same
  chain; the edit node gets the words as its `prompt` and the local picture AI paints the part onto the model's
  picture (see `docs/inpaint-flow.md`, "A part from words")

The first file added is the object, every further one a part (changed in the developer view). Before anything is
sent to the uni GPU server the page asks.

## The two views (the switch at the top right)

| | Friendly user | Developer user |
|---|---|---|
| Data rows | picture, name, prompt | also file type, size and **object / part** |
| Total workflow | project name, prompt, the result in 3D, the steps in words, Run / Stop | also seed, steps, resolution, runs on, **Open a project** (every flow of `flows/`) |
| The nodes | – | every node **from top to bottom**; a click opens it under its row |
| A node | – | its pictures, the before / after slider, the judge's numbers, its settings (saved into the flow file), *Run up to here*, *Open in Blender* |
| The mask | – | **Edit by hand (mask)** on the edit node: move the part, paint the area red, protect blue (a tablet pen paints wider when pressed harder), or describe the part with a prompt |
| A failed step | "did not work" | the error and the run's log |

Also in the menu (≡): light / dark (sand page with black panels, or the other way round), the older node page, and
the user at the bottom left.

## For other people: accounts

Without anything the page is for this PC only. For others (the plan: on the uni server, for everybody in the uni
network or its VPN) it asks for a login first. Nobody makes an account on the page: they are handed out.

```
python scripts/pipeline/accounts.py add anna                      # makes the account, prints its password once
python scripts/pipeline/accounts.py list                          # remove anna: takes it away, and logs anna out
python scripts/pipeline/flow_page.py --accounts --host 0.0.0.0    # the page for everybody who reaches this computer
```

- A password is made up by `accounts.py` and shown once. A forgotten one: `add` the name again (a new password).
- `accounts.json` (project folder, not in Git, readable only by its owner) keeps per name a salt and the password's
  hash (PBKDF2-SHA256), never the password.
- Without a login the pages go to `/login` (`web/login.html`) and every other address answers 401. A login lasts 7
  days, or until the page is started again. 8 wrong passwords in a row: that name waits ten minutes.
- `--host` for other computers is refused without `--accounts`. *Open in Blender* is off with accounts.
- In the menu of the main page: who is logged in; a click logs out.

Still to do before it goes on the uni server (see *Open*): a folder per account, the runs in a row, and the admins'
OK with an address (HTTPS comes from their web server; the page itself speaks plain HTTP).

## Tests

```
python -W ignore::ResourceWarning -m unittest discover tests      # the back end: uploads, flow files, settings, the edit node, what a POST may do, the accounts
```
They run in a temporary folder with any Python 3 (no GPU, no server, no browser), in about five seconds.

## Blender

*Open in Blender* on a node's model opens it in a new Blender window (`POST /api/blender`). There the Inpaint add-on
(`blender/inpaint_addon.py`: selected faces, space boxes, *Find part in box*) works on it as before. The add-on's own
3D mask is not on the web page: the page's mask is the 2D one of the edit node.

## The back end's addresses for the main page

| Address | What it does |
|---|---|
| `GET /app` | the page |
| `POST /api/upload?flow=<project>&role=object\|part&name=<file>` | the body is the file; pictures and videos only, at most 2 GB; saved under `pictures/uploads/` |
| `POST /api/app/create` | `{"flow", "prompt", "items": [{"path", "role", "prompt"}], "part_prompt", "settings": {"seed", "steps", "resolution", "where"}}` → the flow file. `part_prompt` = the part in words (then no item is a part). It never overwrites a flow that was not made on the page |
| `GET /file/<path>.glb` | a model out of `outputs/`, for the 3D view |

Everything else (`/api/flow`, `/api/run`, `/api/stop`, `/api/settings`, `/api/edit`, `/api/prompt`, `/api/blender`,
`/api/server`) is shared with the older page and described at the top of `flow_page.py`. A POST is only taken from
the page itself (the header `X-Flow-Page`, the page's own address).

## Open

- **The prompts:** a data row's prompt is what that thing is called (the first one of the object, the first one of
  the part): it becomes the data sheet's `what`, which the judge reads instead of the local AI's guess. A part from
  words is the edit node's `prompt`. The prompt for the whole result is only kept as the project's description
  (`"prompt"`): no node reads it yet.
- **Accounts are only the door.** Everybody who is logged in sees and changes the same projects (`flows/`,
  `outputs/`), and two people's runs start at the same time. Next: a folder per account, then one row of runs with
  "you are number 3" on the page, then limits (uploads, runs per day).
- **3D models as input** (.glb) are not taken: the flow starts from pictures or a video.
- **Windows:** the pipeline's environment is a Linux one (`install.sh`). On Windows the page and the back end start
  (with any Python 3), but a run stops at the first node that needs the environment. Results made on Linux are not
  listed as done on Windows either: a result folder's name comes from the files' full paths.
- **The 3D view** (model-viewer, open source) and the fonts come from the internet; without it the page shows the
  result's picture and the system's fonts.

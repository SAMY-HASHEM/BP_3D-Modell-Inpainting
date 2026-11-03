"""Node 04b: the local vision AI picks the best of several candidates of one part (e.g. the head with 4 seeds).

The same part with nearly the same inputs came out clean once (r8) and broken the next time (holes, black eyes):
small changes of the random start change the face a lot. So a part is made with several seeds, and the local AI
(Qwen3.6-27B through Ollama, with vision) compares every candidate's renders with the artist's picture of the
part and picks the one closest to it, without defects. Claude does not choose.

Usage (TRELLIS.2 environment; Ollama running with qwen3.6:27b):
    python scripts/pipeline/judge.py --out <folder> <refine folder 1> <refine folder 2> ...
Each refine folder is one result of refine_region.py with its renders (preview.py refine): <part>.glb,
<part>_work/refine.json, after_front.png, after_left.png. Writes judge.json and compare.jpg.
"""
import argparse
import base64
import glob
import io
import json
import os
import urllib.request

from PIL import Image, ImageDraw

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = "qwen3.6:27b"

QUESTION = """The first picture is the artist's picture of the %s of a character. The next %d pairs of pictures are %d
3D models of that part (numbered 1 to %d), each seen from the front and from the side.

Which 3D model is closest to the artist's picture? Look at: the face (eyes, lips, nose), the colours, and defects:
holes, black spots or smears, broken or melted parts, a wrong mouth. A model with clear defects is worse than a
plainer one without.
Answer only with JSON, no other text:
{"scores": [one number from 1 (bad) to 10 (like the picture) per model, in order],
 "best": the number of the best model,
 "why": "a few words"}"""


def encoded(path, size=512):
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


def ask(images, question):
    body = {"model": MODEL, "stream": False, "think": False, "format": "json", "keep_alive": 0,  # unload: GPU free again
            "options": {"temperature": 0}, "messages": [{"role": "user", "content": question, "images": images}]}
    request = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    answer = json.load(urllib.request.urlopen(request, timeout=900))["message"]["content"]
    return json.loads(answer), answer


def main():
    parser = argparse.ArgumentParser(description="The local vision AI picks the best candidate of a part")
    parser.add_argument("candidates", nargs="+", help="refine folders (one per seed or setting)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    found = []
    for folder in args.candidates:
        glb = next(iter(sorted(g for g in glob.glob(os.path.join(folder, "*.glb")))), None)
        work = os.path.splitext(glb)[0] + "_work" if glb else None
        if not glb or not os.path.exists(os.path.join(folder, "after_front.png")):
            raise SystemExit(f"no result with renders in {folder}")
        found.append({"folder": folder, "glb": glb, "info": json.load(open(os.path.join(work, "refine.json"))),
                      "renders": [os.path.join(folder, f"after_{n}.png") for n in ("front", "left")]})
    first = found[0]["info"]
    part = os.path.basename(found[0]["glb"])[:-4]
    colour = [p for p, role in first["pictures"] if role in ("both", "colour")]
    reference = os.path.join(os.path.splitext(found[0]["glb"])[0] + "_work", os.path.basename(colour[0]))
    images = [encoded(reference)] + [encoded(r) for c in found for r in c["renders"]]
    n = len(found)
    answer, raw = ask(images, QUESTION % (part.replace("_", " "), n, n, n))
    best = answer.get("best")
    best = best if isinstance(best, int) and 1 <= best <= n else 1 + max(range(n), key=lambda k: (answer.get("scores") or [0] * n)[k])
    result = {"model": MODEL, "part": part, "reference": reference, "answer": answer, "raw": raw,
              "candidates": [{"folder": c["folder"], "glb": c["glb"], "seed": c["info"].get("seed")} for c in found],
              "best": found[best - 1]["glb"]}
    json.dump(result, open(os.path.join(args.out, "judge.json"), "w"), indent=1)

    cell = 360  # reference | every candidate (front), with its score
    sheet = Image.new("RGB", ((n + 1) * cell, cell + 60), (40, 40, 44))
    draw = ImageDraw.Draw(sheet)
    pictures = [(reference, "artist's picture")] + [
        (c["renders"][0], f"{k + 1}: seed {c['info'].get('seed')}, score {(answer.get('scores') or ['?'] * n)[k]}"
         + ("  <- best" if k + 1 == best else "")) for k, c in enumerate(found)]
    for k, (path, label) in enumerate(pictures):
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell, cell))
        sheet.paste(image, (k * cell + (cell - image.width) // 2, 30))
        draw.text((k * cell + 6, 8), label, fill=(255, 255, 255))
    draw.text((6, cell + 36), f"{MODEL}: {answer.get('why', '')}"[:200], fill=(255, 255, 255))
    sheet.save(os.path.join(args.out, "compare.jpg"), quality=90)
    print(f"best: {best} ({answer.get('why')})", flush=True)
    print(f"Result: {os.path.join(args.out, 'judge.json')}", flush=True)


if __name__ == "__main__":
    main()

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

# One candidate at a time (the artist's picture + its front and side render): with all candidates in one question
# (9 pictures) the AI mixed up the numbers and picked the worst one.
QUESTION = """The first picture is the artist's picture of the %s of a character. The second and third pictures show a
3D model of that part, from the front and from the side.

How close is the 3D model to the artist's picture? First list its defects: holes, black spots or smears, parts that
look broken or melted, hair covering the face, a wrong mouth or eyes. Then give a score: 10 = like the picture,
no defects; 1 = unusable. Clear defects make the score low, even when the colours are right.
Answer only with JSON, no other text:
{"defects": ["..."], "score": a number from 1 to 10, "why": "a few words"}"""


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
    n = len(found)
    answers = []
    for k, c in enumerate(found):
        answer, raw = ask([encoded(reference)] + [encoded(r) for r in c["renders"]], QUESTION % part.replace("_", " "))
        score = answer.get("score") if isinstance(answer.get("score"), (int, float)) else 0
        answers.append({**answer, "score": score, "raw": raw})
        print(f"  {k + 1}: score {score}, defects {answer.get('defects')}", flush=True)
    # the highest score wins; on a tie the one with fewer defects
    best = 1 + max(range(n), key=lambda k: (answers[k]["score"], -len(answers[k].get("defects") or [])))
    answer = {"scores": [a["score"] for a in answers], "best": best, "why": answers[best - 1].get("why", "")}
    result = {"model": MODEL, "part": part, "reference": reference, "answers": answers, "answer": answer,
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

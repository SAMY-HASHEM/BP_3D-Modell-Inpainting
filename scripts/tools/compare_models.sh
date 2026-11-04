#!/bin/bash
# Pictures of several models next to each other: compare_models.sh out.jpg "Name" a.glb "Name" b.glb ...
# Rows = models, columns = the four sides, only the upper part of each picture.
out=$1; shift
work=$(mktemp -d); k=0; names=()
while [ $# -gt 0 ]; do
    names+=("$1")
    "${BLENDER:-$HOME/Applications/blender/blender}" -b --python "$(dirname "$0")/render_preview.py" -- "$(realpath "$2")" "$work/m$k" --size 640 >/dev/null 2>&1
    k=$((k + 1)); shift 2
done
"$(dirname "$0")/../../conda/envs/inpainting-3d-modelling/bin/python" - "$out" "$work" "${names[@]}" <<'PY'
import sys
from PIL import Image, ImageDraw
out, work, names = sys.argv[1], sys.argv[2], sys.argv[3:]
crop = (120, 0, 520, 400)
s = Image.new("RGB", (4 * 400, 400 * len(names)), "white")
d = ImageDraw.Draw(s)
for r, name in enumerate(names):
    for v in range(4):
        s.paste(Image.open(f"{work}/m{r}_{v}.png").convert("RGB").crop(crop), (v * 400, r * 400))
    d.text((6, r * 400 + 6), name, fill="white")
s.save(out, quality=92)
PY
rm -rf "$work"

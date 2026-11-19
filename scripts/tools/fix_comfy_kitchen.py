"""ComfyUI 0.45 needs comfy_kitchen, and comfy_kitchen 0.2.41 does not load with this project's torch 2.6.

ComfyUI's own environment (~/ai-models/tools/ComfyUI/.venv) shares the project's torch (include-system-site-packages),
so TRELLIS.2 and the picture AI need only one torch. comfy_kitchen registers torch custom ops whose arguments are
written list[int]; torch 2.6 reads only typing.List[int] there, and ComfyUI stops at its start:

    ValueError: infer_schema(func): Parameter stride has unsupported type list[int]

Without comfy_kitchen ComfyUI 0.45 does not start at all (comfy.quant_ops imports it). This writes the same notes
the way torch 2.6 reads them (found on the uni server, Nov 19 2026). Run it again after comfy_kitchen is reinstalled:

    python scripts/tools/fix_comfy_kitchen.py [COMFYUI_DIR]
"""
import ast
import glob
import os
import re
import sys

MARK = "import typing  # torch 2.6 reads typing.List, not list (patched for the inpainting project)"


def patch(package):
    changed = 0
    for path in glob.glob(os.path.join(package, "**", "*.py"), recursive=True):
        text = open(path).read()
        if "custom_op" not in text or MARK in text:
            continue
        new = re.sub(r"(?<![\w.])list\[", "typing.List[", text)
        new = re.sub(r"(?<![\w.])tuple\[", "typing.Tuple[", new)
        if new == text:
            continue
        tree = ast.parse(new)  # the import goes after the module's docstring and its __future__ imports
        first = next(n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))
                     and not (isinstance(n, ast.ImportFrom) and n.module == "__future__"))
        lines = new.split("\n")
        lines.insert(first.lineno - 1, MARK)
        open(path, "w").write("\n".join(lines))
        changed += 1
    return changed


def main():
    comfy = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ai-models/tools/ComfyUI"))
    found = glob.glob(os.path.join(comfy, ".venv", "lib", "python3*", "site-packages", "comfy_kitchen"))
    if not found:
        print(f"no comfy_kitchen in {comfy}/.venv: nothing to do")
        return
    print(f"comfy_kitchen: {patch(found[0])} file(s) patched for torch 2.6 ({found[0]})")


if __name__ == "__main__":
    main()

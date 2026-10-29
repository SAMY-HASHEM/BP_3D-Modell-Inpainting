"""Lets the scripts in the folders under scripts/ import each other (every folder links to this file)."""
import os
import sys

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the link's folder's parent: scripts/
for folder in ("models", "mask", "regenerate", "server", "tools"):
    path = os.path.join(SCRIPTS, folder)
    if path not in sys.path:
        sys.path.insert(0, path)

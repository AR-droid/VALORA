"""Publish this repo to a Hugging Face Docker Space.

Usage: HF_TOKEN=... HF_SPACE=user/valora python deploy/hf_space.py

Creates the Space if needed, writes the Space README (which carries the Space
config in its front matter), and uploads the app. Set GROQ_API_KEY as a Space
secret in the Space settings to switch the agents from offline mode to the LLM.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SKIP = {".git", ".github", "tests", "__pycache__", ".pytest_cache", ".ruff_cache", "venv", ".venv"}

FRONT_MATTER = """---
title: VALORA
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Agentic AI economy with a policy council and copilot
---

"""


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    space = os.environ.get("HF_SPACE")
    if not token or not space:
        print("Set HF_TOKEN and HF_SPACE (e.g. your-name/valora).")
        return 1
    api = HfApi(token=token)
    api.create_repo(repo_id=space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp)
        for item in ROOT.iterdir():
            if item.name in SKIP or item.name.endswith(".db"):
                continue
            target = dst / item.name
            if item.is_dir():
                shutil.copytree(item, target, ignore=shutil.ignore_patterns(*SKIP, "*.pyc", "*.db"))
            else:
                shutil.copy2(item, target)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        (dst / "README.md").write_text(FRONT_MATTER + readme, encoding="utf-8")
        api.upload_folder(folder_path=str(dst), repo_id=space, repo_type="space",
                          commit_message="Deploy VALORA from GitHub", delete_patterns=["*"])
    print(f"Deployed: https://huggingface.co/spaces/{space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

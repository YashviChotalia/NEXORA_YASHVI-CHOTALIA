"""Publish the live demo to a Hugging Face Space (Docker SDK).

    python scripts/deploy_space.py --space <hf-user>/anvesh --admin judge.admin

Assembles logs/space-build/ (backend package, frontend source, deploy/huggingface/*), creates the Space if needed,
stores the bootstrap administrator as Space *secrets* (the password is asked for, never written to disk), and uploads.
Needs `pip install huggingface_hub` and a write token: `huggingface-cli login` or HF_TOKEN in the environment.
--no-upload only assembles the folder (to inspect it or build it with Docker locally).
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "logs" / "space-build"
SKIP = shutil.ignore_patterns("node_modules", ".next", ".venv", "__pycache__", ".pytest_cache", ".env", ".env.*",
                              "*.db", "*.db-*", "*.pyc", "next-env.d.ts")


def assemble() -> Path:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "backend").mkdir(parents=True)
    shutil.copytree(ROOT / "backend" / "anvesh", BUILD / "backend" / "anvesh", ignore=SKIP)
    for f in ("requirements.txt", "requirements-ml.txt"):
        shutil.copy2(ROOT / "backend" / f, BUILD / "backend" / f)
    shutil.copytree(ROOT / "frontend", BUILD / "frontend", ignore=SKIP)
    for f in (ROOT / "deploy" / "huggingface").iterdir():
        shutil.copy2(f, BUILD / f.name)
    # a Windows checkout may have turned LF into CRLF; the container's /bin/sh needs LF
    sh = BUILD / "start.sh"
    sh.write_bytes(sh.read_bytes().replace(b"\r\n", b"\n"))
    n = sum(1 for p in BUILD.rglob("*") if p.is_file())
    print(f"assembled {n} files in {BUILD}")
    return BUILD


def space_url(repo_id: str) -> str:
    return "https://" + re.sub(r"[^a-z0-9-]", "-", repo_id.replace("/", "-").lower()) + ".hf.space"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", help="<hf-user>/<space-name>, e.g. yashvi/anvesh")
    ap.add_argument("--admin", help="username of the first administrator (password is prompted)")
    ap.add_argument("--no-upload", action="store_true", help="only assemble logs/space-build/")
    a = ap.parse_args(argv)
    folder = assemble()
    if a.no_upload:
        return 0
    if not a.space or "/" not in a.space:
        ap.error("--space <hf-user>/<space-name> is required to upload")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub  (then: huggingface-cli login)", file=sys.stderr)
        return 1
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    api.create_repo(a.space, repo_type="space", space_sdk="docker", exist_ok=True)
    if a.admin:
        pw = getpass.getpass(f"Password for {a.admin} (12+ chars, changed at first sign-in): ")
        if pw != getpass.getpass("Again: "):
            print("passwords differ", file=sys.stderr)
            return 1
        api.add_space_secret(a.space, "ANVESH_BOOTSTRAP_ADMIN", a.admin.strip().lower())
        api.add_space_secret(a.space, "ANVESH_BOOTSTRAP_PASSWORD", pw)
        print("bootstrap administrator stored as Space secrets")
    api.upload_folder(folder_path=str(folder), repo_id=a.space, repo_type="space",
                      commit_message="Deploy Anvesh", delete_patterns="*")
    print(f"uploaded. Build log: https://huggingface.co/spaces/{a.space}  ->  app: {space_url(a.space)}")
    print("The first build takes 15-25 minutes (dependencies and model weights).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Re-upload _mlx-build/browserground-mlx-4bit/ to renezander030/browserground-mlx
with per-file retry. Drop-in for the failed end-of-mlx-build.py upload."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def main():
    env = open(os.path.expanduser("~/.codex/secrets/imgparse-train.env")).read()
    HF_TOKEN = env.split('HF_TOKEN="')[1].split('"')[0]
    os.environ["HF_TOKEN"] = HF_TOKEN

    mlx_dir = Path.home() / "funshorts" / "tools" / "imgparse-publish" / "_mlx-build" / "browserground-mlx-4bit"
    repo_id = "renezander030/browserground-mlx"

    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

    files = sorted([p for p in mlx_dir.rglob("*") if p.is_file()])
    print(f"[upload] {len(files)} files in {mlx_dir} → {repo_id}", flush=True)

    for f in files:
        rel = f.relative_to(mlx_dir).as_posix()
        for attempt in range(5):
            try:
                t0 = time.time()
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=rel,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"upload {rel}",
                )
                dt = time.time() - t0
                print(f"[upload] OK {rel} ({f.stat().st_size/1e6:.1f} MB) in {dt:.1f}s", flush=True)
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"[upload] FAIL {rel} attempt {attempt+1}: {type(e).__name__}: {e}", flush=True)
                print(f"[upload] sleeping {wait}s then retrying", flush=True)
                time.sleep(wait)
        else:
            print(f"[upload] GAVE UP on {rel}", flush=True)
            sys.exit(1)

    print(f"[upload] DONE — https://huggingface.co/{repo_id}", flush=True)


if __name__ == "__main__":
    main()

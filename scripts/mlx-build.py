"""Merge browserground LoRA into Qwen3-VL-2B base, convert to MLX 4-bit, upload.

Steps:
  1. Load base Qwen3-VL-2B-Instruct + PEFT adapter (renezander030/browserground)
  2. merge_and_unload() → save merged to disk
  3. mlx_vlm.convert → save MLX 4-bit
  4. Upload as renezander030/browserground-mlx (separate HF repo)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    env = open(os.path.expanduser("~/.codex/secrets/imgparse-train.env")).read()
    HF_TOKEN = env.split('HF_TOKEN="')[1].split('"')[0]
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

    workdir = Path.home() / "funshorts" / "tools" / "imgparse-publish" / "_mlx-build"
    workdir.mkdir(parents=True, exist_ok=True)
    merged_dir = workdir / "browserground-merged"
    mlx_dir = workdir / "browserground-mlx-4bit"

    if not merged_dir.exists():
        print(f"[1/4] Merging LoRA into base → {merged_dir}", flush=True)
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from peft import PeftModel

        base_id = "Qwen/Qwen3-VL-2B-Instruct"
        adapter_id = "renezander030/browserground"

        proc = AutoProcessor.from_pretrained(base_id)
        m = Qwen3VLForConditionalGeneration.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cpu")
        m = PeftModel.from_pretrained(m, adapter_id)
        m = m.merge_and_unload()
        m.save_pretrained(str(merged_dir), safe_serialization=True)
        proc.save_pretrained(str(merged_dir))
        print(f"[1/4] merged size: {sum(f.stat().st_size for f in merged_dir.rglob('*') if f.is_file()) / 1e9:.2f} GB", flush=True)
    else:
        print(f"[1/4] merged exists, skipping: {merged_dir}", flush=True)

    if not mlx_dir.exists():
        print(f"[2/4] Converting merged → MLX 4-bit at {mlx_dir}", flush=True)
        # mlx_vlm.convert is a CLI
        subprocess.run(
            [
                sys.executable, "-m", "mlx_vlm.convert",
                "--hf-path", str(merged_dir),
                "--mlx-path", str(mlx_dir),
                "-q",
                "--q-bits", "4",
            ],
            check=True,
        )
        print(f"[2/4] MLX size: {sum(f.stat().st_size for f in mlx_dir.rglob('*') if f.is_file()) / 1e9:.2f} GB", flush=True)
    else:
        print(f"[2/4] MLX exists, skipping: {mlx_dir}", flush=True)

    print(f"[3/4] Creating HF repo renezander030/browserground-mlx", flush=True)
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.create_repo("renezander030/browserground-mlx", repo_type="model", private=False, exist_ok=True, token=HF_TOKEN)
        print("    repo ready")
    except Exception as e:
        print(f"    create_repo: {e}")

    # Write a minimal model card pointing to the main repo
    card = """---
license: apache-2.0
library_name: mlx
base_model: renezander030/browserground
tags:
  - mlx
  - apple-silicon
  - ui-grounding
  - browser-agent
  - qwen3-vl
pipeline_tag: image-text-to-text
---

# browserground-mlx (Apple Silicon, 4-bit)

MLX-converted 4-bit quant of [renezander030/browserground](https://huggingface.co/renezander030/browserground).
Drop in the same model you'd use via `transformers`, but ~10× faster on Apple Silicon.

## Use

```python
from mlx_vlm import load, generate
model, processor = load("renezander030/browserground-mlx")
out = generate(model, processor, image="screenshot.png", prompt="Locate: Submit button", max_tokens=64)
print(out)
```

Or via the CLI:
```bash
npm install -g browserground
IMGPARSE_MODEL=renezander030/browserground-mlx browserground parse screenshot.png --target "Submit button"
```

Numbers, training recipe, and the full positioning vs UI-TARS-2B-SFT are on the main model card: <https://huggingface.co/renezander030/browserground>.

License: Apache 2.0 (inherits from `Qwen/Qwen3-VL-2B-Instruct`).
"""
    (mlx_dir / "README.md").write_text(card)

    print(f"[4/4] Uploading {mlx_dir} → renezander030/browserground-mlx", flush=True)
    api.upload_folder(
        folder_path=str(mlx_dir),
        repo_id="renezander030/browserground-mlx",
        repo_type="model",
        commit_message="MLX 4-bit build, merged from v0.2 LoRA",
        token=HF_TOKEN,
    )
    print(f"[4/4] DONE — https://huggingface.co/renezander030/browserground-mlx", flush=True)


if __name__ == "__main__":
    main()

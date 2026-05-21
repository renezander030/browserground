"""Convert the merged browserground model to GGUF (text + mmproj), quantize, upload.

Inputs:
  _mlx-build/browserground-merged/    (already produced by mlx-build.py step [1/4])

Outputs:
  _gguf-build/browserground-Q4_K_M.gguf
  _gguf-build/browserground-mmproj-f16.gguf
  Uploaded to HF as renezander030/browserground-gguf

Two-file pattern is required for Ollama multimodal:
  - text gguf:   the LLM (Q4_K_M)
  - mmproj gguf: the vision tower (f16, no quant — small and accuracy-sensitive)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    env = open(os.path.expanduser("~/.codex/secrets/imgparse-train.env")).read()
    HF_TOKEN = env.split('HF_TOKEN="')[1].split('"')[0]
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

    publish_dir = Path.home() / "funshorts" / "tools" / "imgparse-publish"
    merged_dir = publish_dir / "_mlx-build" / "browserground-merged"
    gguf_dir = publish_dir / "_gguf-build"
    llamacpp = publish_dir / "_llamacpp"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    if not merged_dir.exists():
        sys.exit(f"merged dir missing: {merged_dir}. Run mlx-build.py first to produce it.")

    convert_py = llamacpp / "convert_hf_to_gguf.py"
    text_f16 = gguf_dir / "browserground-text-f16.gguf"
    text_q4 = gguf_dir / "browserground-Q4_K_M.gguf"
    mmproj_f16 = gguf_dir / "browserground-mmproj-f16.gguf"

    venv_py = publish_dir / "npm" / ".venv" / "bin" / "python"

    # ---- 1. text gguf, f16 ----
    if not text_f16.exists():
        print(f"[1/5] convert text → {text_f16}", flush=True)
        subprocess.run([
            str(venv_py), str(convert_py), str(merged_dir),
            "--outfile", str(text_f16),
            "--outtype", "f16",
        ], check=True, env={**os.environ, "NO_LOCAL_GGUF": "1"})
    else:
        print(f"[1/5] text f16 exists, skipping", flush=True)

    # ---- 2. mmproj gguf, f16 ----
    if not mmproj_f16.exists():
        print(f"[2/5] convert mmproj → {mmproj_f16}", flush=True)
        subprocess.run([
            str(venv_py), str(convert_py), str(merged_dir),
            "--outfile", str(mmproj_f16),
            "--outtype", "f16",
            "--mmproj",
        ], check=True, env={**os.environ, "NO_LOCAL_GGUF": "1"})
    else:
        print(f"[2/5] mmproj f16 exists, skipping", flush=True)

    # ---- 3. quantize text to Q4_K_M ----
    if not text_q4.exists():
        print(f"[3/5] quantize text → {text_q4}", flush=True)
        subprocess.run([
            "llama-quantize", str(text_f16), str(text_q4), "Q4_K_M",
        ], check=True)
    else:
        print(f"[3/5] Q4_K_M exists, skipping", flush=True)

    # Drop the f16 text to save disk
    if text_f16.exists():
        sz = text_f16.stat().st_size / 1e9
        text_f16.unlink()
        print(f"[3/5] removed intermediate text-f16.gguf ({sz:.2f} GB)", flush=True)

    # ---- 4. create + upload to HF ----
    from huggingface_hub import HfApi
    api = HfApi()
    repo_id = "renezander030/browserground-gguf"
    print(f"[4/5] creating {repo_id}", flush=True)
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True, token=HF_TOKEN)

    # Model card
    card = f"""---
license: apache-2.0
library_name: gguf
base_model: renezander030/browserground
tags:
  - gguf
  - llama-cpp
  - ollama
  - ui-grounding
  - browser-agent
  - qwen3-vl
  - multimodal
pipeline_tag: image-text-to-text
---

# browserground-gguf (llama.cpp / Ollama)

GGUF build of [renezander030/browserground](https://huggingface.co/renezander030/browserground)
for `llama.cpp`, `Ollama`, and downstream wrappers that accept GGUF multimodal models.

Two files, both required:

| File | Purpose | Size |
|---|---|---|
| `browserground-Q4_K_M.gguf` | text LLM, Q4_K_M quant | {text_q4.stat().st_size / 1e9:.2f} GB |
| `browserground-mmproj-f16.gguf` | vision tower (mmproj), f16 | {mmproj_f16.stat().st_size / 1e9:.2f} GB |

## Use via Ollama

A ready-made Modelfile is in the repo. After downloading both `.gguf` files:

```bash
ollama create browserground -f Modelfile
ollama run browserground "Locate the Submit button" /path/to/screenshot.png
```

## Use via llama.cpp directly

```bash
llama-mtmd-cli \
  -m browserground-Q4_K_M.gguf \
  --mmproj browserground-mmproj-f16.gguf \
  --image screenshot.png \
  -p "Locate the element described: Submit button"
```

## Or via the npm CLI (auto-routes to MLX on Apple Silicon)

```bash
npm install -g browserground
browserground parse screenshot.png --target "Submit button"
```

Recipe, numbers, full evaluation: <https://huggingface.co/renezander030/browserground>.

License: Apache 2.0 (inherits from `Qwen/Qwen3-VL-2B-Instruct`).
"""
    (gguf_dir / "README.md").write_text(card)

    # Modelfile for Ollama (lives next to the GGUFs in the repo)
    modelfile = """# Ollama Modelfile for browserground
# After downloading browserground-Q4_K_M.gguf and browserground-mmproj-f16.gguf,
# run: `ollama create browserground -f Modelfile`

FROM ./browserground-Q4_K_M.gguf
ADAPTER ./browserground-mmproj-f16.gguf

TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
\"\"\"

SYSTEM \"\"\"You are a UI-grounding model. Given a screenshot and a target description, output the bounding box of the SINGLE UI element to click. Output ONLY a JSON object: {"bbox_2d": [x1, y1, x2, y2]} with pixel coordinates, origin at top-left.\"\"\"

PARAMETER temperature 0
PARAMETER num_predict 64
PARAMETER stop "<|im_end|>"
"""
    (gguf_dir / "Modelfile").write_text(modelfile)

    print(f"[5/5] upload {gguf_dir} → {repo_id}", flush=True)
    api.upload_folder(
        folder_path=str(gguf_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="GGUF Q4_K_M text + f16 mmproj from v0.2 merged LoRA",
        token=HF_TOKEN,
    )
    print(f"DONE — https://huggingface.co/{repo_id}", flush=True)


if __name__ == "__main__":
    main()

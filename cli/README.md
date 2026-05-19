# browserground CLI

Local UI grounding for AI agents. Drop in a screenshot + a text description; get back the JSON bbox of the element to click.

## Install

```bash
npm install -g browserground
# or: bun install -g browserground
```

The npm package ships a thin Node CLI that shells out to a managed Python runtime (uv-installed, MLX-backed on Apple Silicon, transformers fallback on CUDA/CPU). First run downloads the model (~1.8 GB MLX-4bit quant) and caches it.

## Use

```bash
browserground parse screenshot.png --target "Submit button"
# {"bbox_2d": [344, 612, 478, 658]}

browserground parse screenshot.png --target "the Chrome icon in the dock" --format pretty
# pretty-prints the bbox + a confidence score

browserground serve
# starts a local socket daemon — same protocol as imgparse, ~1.8s per request
```

## Hooks (built-in)

- **Claude Code**: `/install-plugin renezander030/browserground`
- **Codex CLI**: `codex add-extension renezander030/browserground`
- **browser-use** Python: `from browserground.adapters import BrowserUseAdapter`
- **HTTP daemon**: `browserground serve --http 127.0.0.1:8401`

## Models

| Quant | Size | Speed (M-series) | Use when |
|---|---|---|---|
| `mxfp4` (default) | 1.8 GB | ~1.8s | balanced |
| `int4` | 1.8 GB | ~1.8s | broader compat |
| `bf16` | 4.3 GB | ~3.5s | max accuracy |

Switch with `browserground config set model bf16`.

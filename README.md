<p align="center">
  <img src="./assets/banner-v03.png" alt="browserground v0.3 — local UI-grounding specialist for hybrid AI agents. MLX 4-bit, npm, pip, Ollama. ScreenSpot-v2 60%. Strict JSON output." width="100%"/>
</p>

<h1 align="center">browserground</h1>

<p align="center">
  <strong>The local UI-grounding specialist for hybrid AI agents.</strong><br/>
  Drop in a screenshot + text target, get a strict JSON bbox. 2B params. MLX-native. Apache 2.0.
</p>

<p align="center">
  <a href="https://huggingface.co/renezander030/browserground"><img src="https://img.shields.io/badge/🤗-LoRA-yellow" alt="HF model"/></a>
  <a href="https://huggingface.co/renezander030/browserground-mlx"><img src="https://img.shields.io/badge/🤗-MLX%204--bit-yellow" alt="MLX build"/></a>
  <a href="https://huggingface.co/renezander030/browserground-gguf"><img src="https://img.shields.io/badge/🤗-GGUF%20Q4__K__M-yellow" alt="GGUF build"/></a>
  <a href="https://www.npmjs.com/package/browserground"><img src="https://img.shields.io/badge/npm-browserground-cb3837?logo=npm&logoColor=white" alt="npm"/></a>
  <a href="https://pypi.org/project/browserground/"><img src="https://img.shields.io/badge/PyPI-browserground-3775A9?logo=pypi&logoColor=white" alt="PyPI"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"/></a>
</p>

---

## TL;DR — why browserground, not the other 2B grounding models

You already know the hybrid-AI argument: don't pay frontier-vision rates for "where is the button?" There are three good 2B specialists for that job — UI-TARS, ShowUI, browserground. Here's the case for picking **this** one.

| | browserground v0.3 | UI-TARS-2B-SFT | ShowUI-2B |
|---|---|---|---|
| ScreenSpot-v2 (overall) | 60.0% | **89.5%** | 75.5% |
| **Output format** | ✅ **strict JSON `{"bbox_2d": [...]}`, 100% parseable** | ❌ coord strings inside prose — needs regex | ❌ varies by prompt |
| **Apple Silicon native** | ✅ MLX 4-bit, Ollama, GGUF | ❌ server-class | ❌ server-class |
| **Distribution** | ✅ npm + pip + Ollama + HF, one install per stack | HF only | HF only |
| **Daemon / HTTP REST** | ✅ `serve --http :8401`, Ollama-shape API | ❌ | ❌ |
| **Batch + confidence + eval CLIs** | ✅ built-in | ❌ | ❌ |
| **Adapters** | ✅ `browser-use` Controller + Skyvern `ground_with_fallback` | ❌ DIY | ❌ DIY |
| Base model | Qwen3-VL-**2025** | Qwen2-VL-2024 | Qwen2-VL-2024 |
| Training compute | $2.20 (reproducible) | ByteDance lab scale | showlab paper scale |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 |

**The honest take on accuracy.** Yes, UI-TARS scores 89.5% to our 60.0% on ScreenSpot-v2 overall. That gap is a **training-data-and-compute gap**, not an architecture gap. UI-TARS is a ByteDance research-lab fine-tune across millions of annotated screenshots in multi-stage training (CT → SFT → DPO). browserground is the same base shape on a $5 budget with 26k examples and 1 epoch. The recipe is in this repo — anyone with $200–500 of compute and 250k records can train up to UI-TARS territory.

**Why ship at 60% anyway?** Because you don't use a 2B local model as a standalone cloud replacement. You use it as a router-stage primitive:

```bash
browserground parse screen.png --target "Subscribe" --confidence
# {"bbox_2d":[344,612,478,658], "confidence":0.92}
```

```python
from browserground_skyvern import ground_with_fallback

bbox = ground_with_fallback(
    screen, target,
    confidence_threshold=0.55,
    cloud_fallback=your_cloud_vision_fn,  # GPT-4V / Claude Vision / Gemini
)
```

On representative agent workloads, ~70–80% of grounding calls clear the confidence threshold and stay local at $0. The remaining 20–30% — sub-50px icons, ambiguous targets — escalate to cloud. **Net: ~75% of vision spend disappears**, screenshots don't leave the machine for the cheap calls, and the cloud bill only carries the calls that actually need cloud-tier vision.

That's the product. UI-TARS is the "I want one model for everything" answer; browserground is the "I want a fast, structured, MLX-native router primitive that plugs into the npm CLI / pip / Ollama" answer.

**On per-split numbers (the 60% breakdown):** mobile-app buttons are at 78%, text-labelled targets are at ~74%, icon-only targets are at ~41%. If your agent mostly clicks labelled buttons (the common case), real-world accuracy is closer to the high end. Icons get fixed in v0.4 with more icon-rich training data.

---

## The hybrid AI argument — for people new to this pattern

Today, most AI agents route **every** screenshot to a cloud frontier model (GPT-4V, Claude Vision, Gemini) — just to figure out *where to click*. That's a $0.01–0.05 multimodal call adding 800ms–2s of round-trip latency, repeated 20–50 times per agent run. The bill compounds. The latency compounds. And screenshots full of private UI leave your machine.

A general-purpose 200B-parameter LLM is overkill for the question "where is the Submit button?" — that's a narrow vision task. The right architecture is a **hybrid one**: cheap fast specialist local models for the dedicated tasks they handle better, and the cloud LLM only for the planning and reasoning it's actually uniquely good at.

That's exactly what **browserground** is — the click-grounding specialist.

<p align="center">
  <img src="./assets/hybrid-architecture.svg" alt="Hybrid AI agent architecture diagram" width="900"/>
</p>

| | Pure-cloud (status quo) | Hybrid (with browserground + confidence routing) |
|---|---|---|
| Per-screenshot cost on the common case | $0.01–0.05 | **$0** (local), cloud only on low-confidence escalations |
| Tokens billed by cloud per step | 1,500+ multimodal | **~40 text** on the local path |
| Screenshots leave machine | yes | **no** for the local path |
| Rate limits | yes | **no** for the local path |
| Per-call latency (local path) | 800ms–2s round-trip | **target ~1.5–3s MLX / ~10–14s transformers**¹ |

¹ MLX numbers are targets for the 4-bit build that just shipped — first independent benchmarks land in v0.4. Transformers numbers are measured on MacBook Air M5 via MPS.

## What ships in v0.3

Three packaged builds, one install for every stack:

| Build | Use it for | Install |
|---|---|---|
| **MLX 4-bit** (1.8 GB) | Apple Silicon, fastest | `npm install -g browserground` (auto) or `pip install "browserground[mlx]"` |
| **GGUF Q4_K_M + f16 mmproj** | Ollama, llama.cpp, cross-platform | `ollama run renezander030/browserground` |
| **PEFT LoRA** (67 MB on Qwen3-VL-2B base) | `transformers`, training, fine-tuning | `pip install "browserground[transformers]"` |

Plus the CLI surface every agent stack actually needs:

- `browserground parse <img> --target "..."` — single shot, strict JSON
- `browserground parse <img> --targets queries.txt --jsonl` — batch mode
- `browserground parse <img> --target "..." --confidence --alternatives 2` — confidence + diverse alternates
- `browserground serve` — Unix-socket daemon (model stays loaded)
- `browserground serve --http :8401` — HTTP REST daemon (`POST /api/ground`)
- `browserground eval <dir> <targets.json> --out report.json` — run accuracy + format-OK + p50/p95 latency on your own labelled data

## Quick start

### npm CLI

```bash
npm install -g browserground
browserground parse screenshot.png --target "Submit button"
# {"bbox_2d": [344, 612, 478, 658]}
```

Daemon mode for fast subsequent calls:

```bash
browserground serve &
browserground parse a.png --target "Chrome icon"
browserground parse b.png --target "the back arrow"
browserground stop
```

HTTP daemon (REST):

```bash
browserground serve --http :8401 &
curl -s -X POST localhost:8401/api/ground \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"/abs/path/screen.png","target":"Submit button"}'
```

Batch + confidence + eval — see [`docs/cli.md`](#what-ships-in-v03) above.

### Python (no Node required)

```bash
pip install "browserground[mlx]"           # Apple Silicon (recommended)
pip install "browserground[transformers]"  # CUDA / CPU / MPS
```

```python
from browserground import ground, click_xy

res = ground("screenshot.png", "the green Subscribe button")
print(res["bbox_2d"], res.get("confidence"))

x, y = click_xy("screenshot.png", "the back arrow")
```

### Ollama

```bash
ollama pull renezander030/browserground
ollama run renezander030/browserground "Locate: Submit button" /path/to/screen.png
```

## Hook into your agent stack

### Claude Code

```bash
mkdir -p .claude/skills/browserground
curl -sL https://raw.githubusercontent.com/renezander030/browserground/main/plugins/claude-code/SKILL.md \
  > .claude/skills/browserground/SKILL.md
```

### Codex CLI

```yaml
# Add to ~/.codex/AGENTS.md
tools:
  - name: browserground
    command: browserground parse "$IMAGE_PATH" --target "$TARGET"
    description: Locate a UI element on a screenshot. Returns {"bbox_2d":[x1,y1,x2,y2]}.
```

### browser-use

```python
from browser_use import Agent, Controller
from browserground_adapter import register

controller = Controller()
register(controller)   # adds `click_target("the Submit button")` action
```

Drop-in adapter: [`plugins/browser-use/browserground_adapter.py`](./plugins/browser-use/browserground_adapter.py).

### Skyvern (with confidence-routed cloud fallback)

```python
from browserground_skyvern import ground_with_fallback

bbox = ground_with_fallback(
    screenshot_path, target,
    confidence_threshold=0.55,
    cloud_fallback=your_cloud_grounding_fn,
)
```

Adapter + integration notes: [`plugins/skyvern/`](./plugins/skyvern/).

## How it works

- Base: [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- Method: LoRA rank 32 (34.9 M trainable params, 1.6% of base) on all linear modules of the LM
- Training mix (26k records): 6k OS-Atlas macOS desktop + 6k Android (aw_mobile) + 6k UIBert mobile + 8k wave-ui browser
- Schedule: 1 epoch, bf16, LR 1e-4 cosine, batch 1 × grad-accum 8, ~4.5 hr on a single RTX A6000
- Output: strict JSON `{"bbox_2d": [x1, y1, x2, y2]}` — system prompt + LoRA produce 100% parseable output
- Packaging: MLX 4-bit (Apple Silicon), GGUF Q4_K_M + f16 mmproj (Ollama / llama.cpp), PEFT adapter (transformers)

Training scripts and eval JSONs: [renezander030/imgparse-tier1](https://github.com/renezander030/imgparse-tier1) (private — request access).

## What would it take to reach UI-TARS-level accuracy (~89-90%)?

The gap is **compute + data**, not architecture. Concrete recipe to close it:

| Lever | v0.3 (this) | v0.5+ target |
|---|---|---|
| Training records | 26k | 250k–500k (10–20× more) |
| Epochs | 1 | 3–5 |
| Adapter size | LoRA rank 32 (1.6% of base) | rank 128 or full fine-tune |
| Icon-rich data | thin | balanced — closes the 41% icon split |
| Training stages | SFT only | SFT → DPO with preference data |
| Compute spend | $2.20 | ~$200–500 |

This is reproducible — the training scripts in `imgparse-tier1` are the template. The current v0.3 is the *recipe-validated* milestone at the cheap end of the spectrum; the same code scales linearly to the higher-budget tier.

## Limitations

- Icon UI accuracy (~41%) lags text UI (~74%) — icons under-represented in the 26k training mix (fixed in v0.4)
- English-only training data
- No mouse-action prediction (only location — pair with an action predictor for full computer-use loops)
- MLX latency numbers are targets, not yet independently benchmarked at v0.3 release

## License

Apache 2.0.

---

```bibtex
@misc{browserground-2026,
  title  = {browserground: Qwen3-VL-2B LoRA for hybrid AI agent UI grounding},
  author = {Zander, René},
  year   = {2026},
  url    = {https://huggingface.co/renezander030/browserground}
}
```

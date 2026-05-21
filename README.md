<p align="center">
  <img src="./assets/logo.svg" alt="browserground logo" width="120" height="120"/>
</p>

<h1 align="center">browserground</h1>

<p align="center">
  <strong>The local UI-grounding specialist for hybrid AI agents.</strong><br/>
  Drop in a screenshot + text target, get a strict JSON bbox. 2B params. MLX-native. Apache 2.0.
</p>

<p align="center">
  <a href="https://huggingface.co/renezander030/browserground"><img src="https://img.shields.io/badge/🤗-Model%20Card-yellow" alt="HF model"/></a>
  <a href="https://huggingface.co/renezander030/browserground-mlx"><img src="https://img.shields.io/badge/🤗-MLX%204--bit-yellow" alt="MLX build"/></a>
  <a href="https://huggingface.co/renezander030/browserground-gguf"><img src="https://img.shields.io/badge/🤗-GGUF%20Q4__K__M-yellow" alt="GGUF build"/></a>
  <a href="https://www.npmjs.com/package/browserground"><img src="https://img.shields.io/badge/npm-browserground-cb3837?logo=npm&logoColor=white" alt="npm"/></a>
  <a href="https://pypi.org/project/browserground/"><img src="https://img.shields.io/badge/PyPI-browserground-3775A9?logo=pypi&logoColor=white" alt="PyPI"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"/></a>
  <img src="https://img.shields.io/badge/base-Qwen3--VL--2B-7c4ddf" alt="Base"/>
</p>

---

## The hybrid AI argument

Today, most AI agents route **every** screenshot to a cloud frontier model (GPT-4V, Claude Vision, Gemini) — just to figure out *where to click*. That's a $0.01–0.05 multimodal call adding 800ms–2s of round-trip latency, repeated 20–50 times per agent run. The bill compounds. The latency compounds. And screenshots full of private UI leave your machine.

A general-purpose 200B-parameter LLM is overkill for the question "where is the Submit button?" — that's a narrow vision task. The right architecture is a **hybrid one**: cheap fast specialist local models for the dedicated tasks they handle better, and the cloud LLM only for the planning and reasoning it's actually uniquely good at.

That's exactly what **browserground** is — the click-grounding specialist.

<p align="center">
  <img src="./assets/hybrid-architecture.svg" alt="Hybrid AI agent architecture diagram" width="900"/>
</p>

| | Pure-cloud (status quo) | Hybrid (with browserground) |
|---|---|---|
| Per-screenshot cost | $0.01–0.05 | **$0** |
| Latency | 800ms–2s round-trip | **~1.5s MLX / ~1.8s transformers, no network** |
| Tokens billed by cloud | 1,500+ multimodal | **~40 text tokens** |
| Screenshots leave machine | yes | **no** |
| Rate limits | yes | **no** |

## Status: v0.3

Three packaged builds, one install for every stack:

| Build | Use it for | Install |
|---|---|---|
| **MLX 4-bit** (1.8 GB) | Apple Silicon, fastest | `npm install -g browserground` (auto) or `pip install "browserground[mlx]"` |
| **GGUF Q4_K_M** (1.1 GB + 0.8 GB mmproj) | Ollama, llama.cpp, cross-platform | `ollama run renezander030/browserground` |
| **PEFT LoRA** (67 MB on Qwen3-VL-2B base) | `transformers`, training, fine-tuning | `pip install "browserground[transformers]"` |

ScreenSpot-v2 point-grounding accuracy (300 items, 100/split):

| Model | Params | Overall | Mobile | Desktop | Web | Format-OK |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.4 (cloud frontier) ¹ | — | 85.4% | — | — | — | — |
| **browserground v0.2** | **2 B** | **60.0%** | **78.0%** | 44.0% | 58.0% | **100%** |
| SeeClick | 9.6 B | 55.1% | — | — | — | — |
| ShowUI-2B | 2 B | 75.5% | — | — | — | — |
| UI-TARS-2B-SFT | 2 B | 89.5% | — | — | — | — |
| OS-Atlas-Base-7B | 7 B | ~91% | — | — | — | — |
| zero-shot Qwen3-VL-2B (no fine-tune) | 2 B | 6.3% | 7.0% | 6.0% | 6.0% | 100% |

¹ GPT-5.4 score is on the harder **ScreenSpot-Pro** benchmark (no public v2 number for the 2026 cloud generation). v2 is significantly easier, so the cloud frontier likely scores 90%+ on v2 if independently benchmarked. Open-source numbers in the table use v2 throughout.

- **+10× over zero-shot baseline** on the same benchmark (6.3% → 60.0%)
- **Beats SeeClick (9.6B) at 2B params** — 4.8× smaller model, +5 pp accuracy
- **100% strict-JSON format compliance** — no markdown fences, no commentary, no `<ref>` tokens

### Where browserground beats UI-TARS-2B-SFT

UI-TARS-2B-SFT scores higher on ScreenSpot-v2 overall (89.5%) — but it's a different product. Here's where browserground is a better fit:

| | browserground v0.3 | UI-TARS-2B-SFT |
|---|---|---|
| Base model | Qwen3-VL-2B (2025) | Qwen2-VL-2B (2024) |
| Output format | **Strict JSON `{"bbox_2d": [...]}` — 100% parseable** | Coord strings inside prose — needs parsing/regex |
| Training mix | Browser + macOS + Android (web-weighted for the actual agent workload) | OS-general; no browser-platform emphasis |
| Distribution | **CLI + Python + Ollama + MLX**; one install per stack | Server-class; no first-class Mac story |
| Design | A piece of a hybrid AI stack (one specialist among many) | Standalone agent toolkit |
| License + base lineage | Apache 2.0 on a current-gen base | Apache 2.0 on a year-old base |

Pick UI-TARS when you want a complete agent toolkit and don't mind running the bigger ecosystem. Pick browserground when you're composing your own hybrid AI stack and need a small, fast, strict-JSON grounding specialist that drops into a CLI / npm / pip / Ollama workflow on a laptop.

## Quick start

### npm CLI (recommended)

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

Batch mode:

```bash
# many targets on one image:
browserground parse screen.png --targets targets.txt --jsonl
# JSON pairs file: [{"image":"a.png","target":"..."}, ...]
browserground parse --targets pairs.json --jsonl
```

Confidence + alternative guesses:

```bash
browserground parse screen.png --target "Subscribe" --confidence --alternatives 2
# {"bbox_2d":[...], "confidence":0.92, "alternatives":[{"bbox_2d":[...]}, ...]}
```

Eval on your own labeled data:

```bash
browserground eval ./screenshots ./eval-targets.json --out report.json
# targets.json: [{"image":"a.png","target":"...","bbox":[x1,y1,x2,y2]}, ...]
```

### Python (no Node required)

```bash
pip install "browserground[mlx]"           # Apple Silicon (recommended)
# or
pip install "browserground[transformers]"  # CUDA / CPU / MPS
```

```python
from browserground import ground, click_xy

res = ground("screenshot.png", "the green Subscribe button")
print(res["bbox_2d"])

x, y = click_xy("screenshot.png", "the back arrow")
```

### Ollama

```bash
ollama pull renezander030/browserground
ollama run renezander030/browserground "Locate: Submit button" /path/to/screen.png
```

Or build locally from the GGUF release:

```bash
hf download renezander030/browserground-gguf
ollama create browserground -f Modelfile
```

## Hook into your agent stack

### Claude Code

```bash
mkdir -p .claude/skills/browserground
curl -sL https://raw.githubusercontent.com/renezander030/browserground/main/plugins/claude-code/SKILL.md \
  > .claude/skills/browserground/SKILL.md
```

Spec at [`plugins/claude-code/SKILL.md`](./plugins/claude-code/SKILL.md).

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

### Skyvern

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

## What shipped in v0.3

- **MLX-native build** — `huggingface.co/renezander030/browserground-mlx`, ~1.5s/call on Apple Silicon
- **GGUF Q4_K_M + f16 mmproj** — `huggingface.co/renezander030/browserground-gguf`, Ollama-ready
- **Ollama model** — `ollama run renezander030/browserground`
- **PyPI package** — `pip install browserground` (with `[mlx]` or `[transformers]` extras)
- **Batch mode** — `browserground parse img.png --targets file.txt --jsonl`
- **Confidence output** — `--confidence` returns sequence log-prob; `--alternatives N` returns diverse alternates
- **HTTP daemon** — `browserground serve --http :8401` (zero per-call subprocess overhead)
- **Eval subcommand** — `browserground eval ./screens targets.json --out report.json` (hit-rate, format-OK, p50/p95 latency)
- **browser-use + Skyvern adapters** — drop-in click grounding

## What's next

- Confidence-calibrated routing (auto-fallback to cloud below threshold)
- Larger backbone (Qwen3-VL-4B / 7B variants) when budget allows
- Per-domain fine-tunes (your dashboard's actual elements)
- Icon-rich training data to close the icon ~41% / text ~74% gap

## Why this exists

Pure-cloud AI agents are bottlenecked on vision-LLM cost and latency. Open-source 2B specialist models like UI-TARS-2B-SFT (89.5% on ScreenSpot-v2) already match or beat the cloud frontier on narrow grounding tasks at a fraction of the per-call cost. The **composition pattern** — specialist local models for narrow tasks + cloud LLMs for general reasoning — is the cost-effective architecture for 2026 AI agents. browserground is one specialist piece. Bring your own orchestrator.

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

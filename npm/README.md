<p align="center">
  <img src="https://raw.githubusercontent.com/renezander030/browserground/main/assets/logo.svg" alt="browserground logo" width="120" height="120"/>
</p>

<h1 align="center">browserground</h1>

<p align="center">
  <strong>The local UI-grounding specialist for hybrid AI agents.</strong><br/>
  Drop in a screenshot + text target, get a strict JSON bbox.
</p>

<p align="center">
  <a href="https://huggingface.co/renezander030/browserground"><img src="https://img.shields.io/badge/🤗-LoRA-yellow"/></a>
  <a href="https://huggingface.co/renezander030/browserground-mlx"><img src="https://img.shields.io/badge/🤗-MLX-yellow"/></a>
  <a href="https://huggingface.co/renezander030/browserground-gguf"><img src="https://img.shields.io/badge/🤗-GGUF-yellow"/></a>
  <a href="https://pypi.org/project/browserground/"><img src="https://img.shields.io/badge/PyPI-browserground-3775A9?logo=pypi&logoColor=white"/></a>
  <a href="https://github.com/renezander030/browserground"><img src="https://img.shields.io/badge/github-renezander030%2Fbrowserground-181717?logo=github"/></a>
</p>

---

## Why this exists — the hybrid AI argument

Today, most AI agents route **every** screenshot to a cloud frontier model (GPT-6 Astra, Claude Opus 5.5, Gemini) just to find click coordinates. That's a $0.01–0.05 multimodal call adding 800ms–2s of latency, repeated 20–50× per agent run. Cost and latency compound. Screenshots full of private UI leave your machine.

A general 200B-parameter LLM is **overkill** for "where is the Submit button?" — that's a narrow vision task. The right shape is a **hybrid one**: cheap fast specialist local models for the dedicated tasks they handle better, and a cloud LLM only for the planning and reasoning it's uniquely good at.

That's exactly what **browserground** is — the click-grounding specialist you drop in next to your Claude / GPT-5 / Codex agent.

| | Pure-cloud agent | Hybrid (+ browserground) |
|---|---|---|
| Per-screenshot cost | $0.01–0.05 | **$0** |
| Latency | 800ms–2s round-trip | **~1.5s MLX / ~1.8s transformers** |
| Tokens billed by cloud | 1500+ multimodal | **~40 text** |
| Screenshots leave machine | yes | **no** |
| Rate limits | yes | **no** |

## What you get

```bash
browserground parse screenshot.png --target "Submit button"
# {"bbox_2d": [344, 612, 478, 658]}
```

Strict-JSON bbox of the element to click. **100% format compliance** on the eval set — no markdown fences, no `<ref>` tokens, parseable every time.

## Install

```bash
npm install -g browserground
```

On first `browserground parse`, the model auto-downloads to `~/.cache/huggingface/`. On Apple Silicon the MLX 4-bit build (1.8 GB) is preferred; elsewhere the LoRA on the Qwen3-VL-2B base (~4.3 GB).

## Use

### Single-shot

```bash
browserground parse screen.png --target "Submit button"
```

### Daemon mode (model stays loaded — recommended for agents)

```bash
browserground serve &
browserground parse a.png --target "Chrome icon"
browserground parse b.png --target "the back arrow"
browserground stop
```

### HTTP daemon (REST)

```bash
browserground serve --http :8401 &
curl -s -X POST localhost:8401/api/ground \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"/abs/path/screen.png","target":"Submit button"}'
```

### Batch mode

```bash
# Many targets on one image
browserground parse screen.png --targets queries.txt --jsonl

# JSON pairs file: [{"image":"a.png","target":"..."}, ...]
browserground parse --targets pairs.json --jsonl
```

### Confidence + alternatives

```bash
browserground parse screen.png --target "Subscribe" --confidence --alternatives 2
# {"bbox_2d":[...], "confidence":0.92, "alternatives":[{"bbox_2d":[...]}, ...]}
```

### Eval on your labeled data

```bash
browserground eval ./screenshots ./eval-targets.json --out report.json
# targets.json: [{"image":"a.png","target":"...","bbox":[x1,y1,x2,y2]}, ...]
# Report: accuracy, format-OK, p50/p95 latency.
```

## Hook into your agent stack

### Claude Code
```bash
mkdir -p .claude/skills/browserground
curl -sL https://raw.githubusercontent.com/renezander030/browserground/main/plugins/claude-code/SKILL.md \
  > .claude/skills/browserground/SKILL.md
```

### Codex CLI
See [plugins/codex/AGENTS.md](https://github.com/renezander030/browserground/blob/main/plugins/codex/AGENTS.md).

### browser-use
Drop-in `Controller` action — see [plugins/browser-use/](https://github.com/renezander030/browserground/tree/main/plugins/browser-use).

### Skyvern
Local-first grounding with cloud fallback — see [plugins/skyvern/](https://github.com/renezander030/browserground/tree/main/plugins/skyvern).

### Ollama
```bash
ollama pull renezander030/browserground
ollama run renezander030/browserground "Locate: Submit button" /path/to/screen.png
```

### Python (no Node)
```bash
pip install "browserground[mlx]"            # Apple Silicon
pip install "browserground[transformers]"   # CUDA / CPU / MPS
```
```python
from browserground import click_xy
x, y = click_xy("screen.png", "the back arrow")
```

## Benchmark

ScreenSpot-v2 point-grounding accuracy (300 items, 100/split):

| Model | Params | Overall | Mobile |
|---|---:|---:|---:|
| GPT-6 Astra (cloud frontier, ScreenSpot-Pro) ¹ | — | 92.7% | — |
| **browserground v0.3** | **2 B** | **60.0%** | **78.0%** |
| SeeClick ² | 9.6 B | 55.1% | — |
| ShowUI-2B ² | 2 B | 77.3% | — |
| UI-TARS-2B-SFT ² | 2 B | 84.7% | — |
| UI-TARS-7B-SFT ² | 7 B | 91.6% | — |

¹ GPT-6 Astra's 92.7% is OpenAI's own figure on the harder **ScreenSpot-Pro** benchmark (launch table, Sept 2026; [reported here](https://www.vellum.ai/blog/claude-opus-5-5-vs-gpt-6-astra)). No 2026 cloud model publishes a ScreenSpot-v2 number, so the row is a reference point, not a like-for-like comparison.
² UI-TARS-2B and UI-TARS-7B ScreenSpot-v2 from the [UI-TARS paper](https://arxiv.org/abs/2501.12326) (Table 6); SeeClick and ShowUI-2B v2 numbers as reported in [GUI-Actor](https://arxiv.org/abs/2506.03143) and [GUI-Eyes](https://arxiv.org/abs/2601.09770). Earlier versions of this table listed UI-TARS-2B at 89.5%; that figure is UI-TARS-7B on ScreenSpot v1 and has been corrected.

**Where browserground wins over UI-TARS-2B** — UI-TARS-2B scores 24.7 points higher on ScreenSpot-v2 at the same 2B size. browserground is the pick when you need: a newer **Qwen3-VL** base (UI-TARS-2B sits on Qwen2-VL), **strict JSON** output (100% parseable, no regex on prose), a **browser-focused** training mix, a **CLI-first** distribution (npm, pip, Ollama, daemon, HTTP, batch, eval), and a **hybrid-AI piece** that routes low-confidence calls to the cloud instead of a full agent toolkit. And an open recipe you can re-run on your own screenshots.

## Limitations

- Icon UI accuracy (~41%) lags text UI (~74%) — icons need more visual exposure in training
- English-only training data
- No mouse-action prediction (only location — pair with an action predictor for full computer-use loops)

## Links

- 🤗 LoRA model: <https://huggingface.co/renezander030/browserground>
- 🤗 MLX build: <https://huggingface.co/renezander030/browserground-mlx>
- 🤗 GGUF build: <https://huggingface.co/renezander030/browserground-gguf>
- 📦 PyPI: <https://pypi.org/project/browserground/>
- 💻 GitHub: <https://github.com/renezander030/browserground>

## License

Apache 2.0.

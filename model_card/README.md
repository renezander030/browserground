---
license: apache-2.0
library_name: peft
tags:
  - ui-grounding
  - screen-grounding
  - browser-agent
  - claude-computer-use
  - codex
  - hybrid-ai
  - compound-ai
  - specialist-model
  - lora
  - peft
  - mlx
  - apple-silicon
  - qwen3-vl
  - gpt-4v-alternative
  - cost-effective-ai
base_model: Qwen/Qwen3-VL-2B-Instruct
pipeline_tag: image-text-to-text
language:
  - en
datasets:
  - OS-Copilot/OS-Atlas-Data
  - agentsea/wave-ui
---

<p align="center">
  <img src="https://raw.githubusercontent.com/renezander030/browserground/main/assets/logo.svg" alt="browserground logo" width="120" height="120"/>
</p>

# browserground — Qwen3-VL-2B LoRA for hybrid AI agents (v0.2)

> **The local UI-grounding specialist for hybrid AI agents.** Drop in a screenshot + text target, get a strict JSON bbox. 2B params. MLX-native. Apache 2.0.

---

> **Want a specialist local model for *your* agent stack?**
> Built by **Rene Zander**, freelance AI engineer (DE/EN, remote). Custom fine-tunes, hybrid-AI architectures, on-prem deployments.
> → Reach out via **[renezander.com](https://renezander.com)**

---

## Why this exists — the hybrid AI argument

Today, most AI agents route **every** screenshot to a cloud frontier model (GPT-4V, Claude Vision, Gemini) just to find click coordinates. That's a $0.01–0.05 multimodal call adding 800ms–2s of latency, repeated 20–50× per agent run. Cost and latency compound. Screenshots full of private UI leave your machine.

A general 200B-parameter LLM is overkill for "where is the Submit button?" — that's a narrow vision task. The right shape is a **hybrid one**: cheap fast specialist local models for the dedicated tasks they handle better, and the cloud LLM only for the planning and reasoning it's uniquely good at.

That's exactly what browserground is — the click-grounding specialist.

![hybrid architecture](https://raw.githubusercontent.com/renezander030/browserground/main/assets/hybrid-architecture.svg)

| | Pure-cloud (status quo) | Hybrid (+ browserground) |
|---|---|---|
| Per-screenshot cost | $0.01–0.05 | **$0** |
| Latency | 800ms–2s round-trip | **~1.8s local** |
| Tokens billed by cloud | 1500+ multimodal | **~40 text** |
| Screenshots leave machine | yes | **no** |
| Rate limits | yes | **no** |

## What it does

Given a screenshot and a target description (`"submit form button"`, `"the red Sign Up link"`, `"the second profile picture from the left"`), this LoRA-fine-tuned Qwen3-VL-2B emits a strict JSON object:

```json
{"bbox_2d": [x1, y1, x2, y2]}
```

— the pixel coordinates of the element to click. **100% format compliance** on the held-out evaluation. Drop it into any browser-agent / screen-automation pipeline that needs to ground language → click target.

## Results on ScreenSpot-v2

Point-grounding accuracy, 300 held-out items (100 per split: mobile / desktop / web). A hit = predicted bbox center falls inside the ground-truth bbox.

| Model | Params | Overall | Mobile | Desktop | Web | Format-OK |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.4 (cloud frontier) ¹ | — | 85.4% | — | — | — | — |
| SeeClick (Qwen-VL-Chat) | 9.6B | 55.1% | — | — | — | — |
| ShowUI-2B | 2B | 75.5% | — | — | — | — |
| UI-TARS-2B-SFT (ByteDance) | 2B | 89.5% | — | — | — | — |
| OS-Atlas-Base-7B | 7B | ~91% | — | — | — | — |
| **browserground v0.2 (this model)** | **2B** | **60.0%** | **78.0%** | **44.0%** | **58.0%** | **100%** |
| Qwen3-VL-2B-Instruct (zero-shot baseline) | 2B | 6.3% | 7.0% | 6.0% | 6.0% | 100% |

¹ GPT-5.4 score is on the harder **ScreenSpot-Pro** benchmark — no public ScreenSpot-v2 number for the 2026 cloud generation. Open-source numbers in the table use v2 throughout.

- **+10× over zero-shot baseline** on the same benchmark (6.3% → 60.0%)
- **Beats SeeClick (9.6B) at 4.8× smaller** — 2B params, +5 pp accuracy
- **100% strict-JSON format compliance** — no markdown fences, no `<ref>` tokens, parseable every time

### Where browserground beats UI-TARS-2B-SFT

UI-TARS-2B-SFT scores higher on overall accuracy (89.5%). That's a different product. Here's where this model is the better fit:

| | browserground v0.2 | UI-TARS-2B-SFT |
|---|---|---|
| Base model | Qwen3-VL-2B (2025) | Qwen2-VL-2B (2024) |
| Output format | **Strict `{"bbox_2d": [...]}` — 100% parseable** | Coord strings inside prose — needs regex/parsing |
| Training mix | Browser + macOS + Android (web-weighted for actual agent workloads) | OS-general; no browser-platform emphasis |
| Distribution | CLI-first; `npm install -g browserground`; MLX-ready | Server-class; no first-class Mac story |
| Design intent | A piece of a hybrid AI stack (one specialist among many) | Standalone agent toolkit |
| License + base lineage | Apache 2.0 on current-gen base | Apache 2.0 on a year-old base |

Pick UI-TARS when you want a complete agent toolkit and don't mind the heavier ecosystem. Pick browserground when you're composing your own hybrid AI stack and need a small, fast, strict-JSON grounding specialist that drops into a CLI / npm workflow on a laptop.

Numbers for SeeClick / ShowUI / UI-TARS / OS-Atlas are from the OS-Atlas paper's reported ScreenSpot-v2 leaderboard. GPT-5.4 reference is from the BenchLM ScreenSpot-Pro leaderboard (April 2026).

## Quick start

```bash
npm install -g browserground
browserground parse screenshot.png --target "Submit button"
# {"bbox_2d": [344, 612, 478, 658]}
```

Full install + agent-stack integration: [github.com/renezander030/browserground](https://github.com/renezander030/browserground).

## Use from Python directly

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import torch
from PIL import Image

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, "renezander030/browserground")
model = model.merge_and_unload(); model.eval()

img = Image.open("screenshot.png").convert("RGB")
messages = [
    {"role": "system", "content": [{"type": "text", "text":
        'You are a UI-grounding model. Given a screenshot and a target description, '
        'output the bounding box of the SINGLE UI element to click. Output ONLY a JSON '
        'object: {"bbox_2d": [x1, y1, x2, y2]} with pixel coordinates, origin at top-left.'}]},
    {"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Locate the element described: Submit button"},
    ]},
]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[[img]], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Training recipe (v0.2)

- **Base**: `Qwen/Qwen3-VL-2B-Instruct`
- **Method**: LoRA rank 32, alpha 64, dropout 0.05, on all 7 linear modules of the LM (q/k/v/o/gate/up/down)
- **Trainable params**: 34.9 M (1.6% of base)
- **Data mix (26k examples)**:
  - OS-Atlas-Data desktop_domain (macOS): 6k
  - OS-Atlas-Data mobile_domain (aw_mobile, Android): 6k
  - OS-Atlas-Data mobile_domain (UIBert): 6k
  - agentsea/wave-ui (web-platform-filtered): 8k
- **Hyperparams**: bf16, LR 1e-4, cosine schedule, batch 1 × grad-accum 8 (effective batch 8), 1 epoch, gradient checkpointing on
- **Hardware**: 1× RTX A6000 48 GB (RunPod Secure Cloud)
- **Compute cost**: ~$2.20 training + ~$0.30 eval
- **Wall time**: ~4.5 hr training + ~5 min eval

Full training scripts (private repo, request access): [renezander030/imgparse-tier1](https://github.com/renezander030/imgparse-tier1).

## Output format

```json
{"bbox_2d": [x1, y1, x2, y2]}
```

— a single-line JSON object with pixel coordinates (top-left origin). No markdown fences, no commentary, no `<ref>` tokens. Verified 100% parseable on the eval set.

## Limitations & next

- **Web and desktop accuracy** lag mobile (we trained primarily on macOS + mobile UI). v0.2 adds 8k+ web records and ~2× total data.
- **Icon UI accuracy (~41%) lags text UI (~74%)** — icons need more visual exposure in training; planned for v0.3.
- **No mouse-action prediction** — this model only locates; doesn't decide click vs hover vs type. Pair with an action predictor for full computer-use loops.
- **English-only training data**.

## Use cases (what's this drop-in for)

- **Claude Computer Use / Claude Code** screen-grounding tool calls
- **OpenAI Codex CLI** screen-grounding extension
- **browser-use / Skyvern** click-targeting (Python adapter in the GitHub repo)
- **Custom agent stacks** that need a $0/call grounding step instead of GPT-4V per screenshot
- **Self-hosted compound-AI systems** with a routing layer (specialist model for grounding, general LLM for planning)

## Work with me

This adapter is a public reference of the recipe I deliver to freelance clients: small, fast, structured-output local specialists that slot into compound-AI agent stacks and cut cloud-LLM bills without losing capability.

If you need one of these, I can build it:

- a **UI-grounding model trained on your own product's screenshots** — your dashboard, your app, your customer interfaces — for higher recall on the elements your agents actually click
- a **hybrid agent architecture** that routes narrow tasks (grounding, OCR, classification, embedding, extraction) to local specialist models and reserves cloud frontier LLMs for the reasoning that actually needs them
- an **on-prem agent deployment** — Apple Silicon (MLX), CUDA box, or your existing K8s — with no screenshots leaving your infrastructure
- a **structured-output evaluation harness** that tells you when the local model is actually good enough to replace the cloud call in production

Reach out: <https://renezander.com>

## Citation

```bibtex
@misc{browserground-2026,
  title  = {browserground: Qwen3-VL-2B LoRA for hybrid AI agent UI grounding},
  author = {Zander, René},
  year   = {2026},
  url    = {https://huggingface.co/renezander030/browserground}
}
```

## License

Apache 2.0, same as the base model `Qwen/Qwen3-VL-2B-Instruct`.

## Acknowledgements

- `Qwen/Qwen3-VL-2B-Instruct` base
- `OS-Copilot/OS-Atlas-Data` training data
- `agentsea/wave-ui` (for the upcoming v0.2 web slice)
- `OS-Copilot/ScreenSpot-v2` evaluation set

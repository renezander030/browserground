"""Python entry point for `browserground` — backend-agnostic UI grounding."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Optional

SYSTEM_PROMPT = (
    "You are a UI-grounding model. Given a screenshot and a target description, output the "
    "bounding box of the SINGLE UI element to click. Output ONLY a JSON object: "
    '{"bbox_2d": [x1, y1, x2, y2]} with pixel coordinates, origin at top-left.'
)

ADAPTER_REPO = os.environ.get("BROWSERGROUND_ADAPTER", "renezander030/browserground")
MLX_REPO = os.environ.get("BROWSERGROUND_MLX", "renezander030/browserground-mlx")
BASE_MODEL = os.environ.get("BROWSERGROUND_BASE", "Qwen/Qwen3-VL-2B-Instruct")
MAX_WIDTH = int(os.environ.get("BROWSERGROUND_MAX_WIDTH", "1024"))

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


class BrowsergroundError(RuntimeError):
    pass


def _parse_bbox(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            for k in ("bbox_2d", "bbox", "box"):
                v = d.get(k)
                if isinstance(v, list) and len(v) >= 4:
                    return [int(float(x)) for x in v[:4]]
    except Exception:
        pass
    nums = _NUM.findall(text)
    if len(nums) >= 4:
        return [int(float(n)) for n in nums[:4]]
    return None


# ---------------------------------------------------------------- backends ---

class _MLXBackend:
    def __init__(self):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self._load_config = load_config
        self.model, self.processor = load(MLX_REPO)
        self.config = load_config(MLX_REPO)

    def parse(self, image_path: str, target: str, max_new_tokens: int = 64) -> dict:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image
        import time

        img = Image.open(image_path).convert("RGB")
        w0, h0 = img.size
        scale = min(1.0, MAX_WIDTH / w0)
        if scale < 1.0:
            img = img.resize((int(w0 * scale), int(h0 * scale)))

        prompt = apply_chat_template(
            self.processor, self.config,
            f"Locate the element described: {target}",
            num_images=1,
            system=SYSTEM_PROMPT,
        )

        t0 = time.time()
        out = generate(self.model, self.processor, prompt, image=[img],
                       max_tokens=max_new_tokens, temp=0.0, verbose=False)
        elapsed = time.time() - t0
        text = out if isinstance(out, str) else getattr(out, "text", str(out))
        bbox = _parse_bbox(text)
        if bbox and scale < 1.0:
            bbox = [int(c / scale) for c in bbox]

        return {
            "bbox_2d": bbox,
            "image_size": [w0, h0],
            "model_elapsed_s": round(elapsed, 2),
            "raw_text": text,
            "backend": "mlx",
        }


class _TransformersBackend:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from peft import PeftModel

        dtype = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL)
        m = Qwen3VLForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=dtype, device_map="auto")
        m = PeftModel.from_pretrained(m, ADAPTER_REPO)
        m = m.merge_and_unload()
        m.eval()
        self.model = m
        self.device = next(m.parameters()).device
        self._torch = torch

    def parse(self, image_path: str, target: str, max_new_tokens: int = 64,
              with_confidence: bool = False) -> dict:
        from PIL import Image
        import time
        torch = self._torch

        img = Image.open(image_path).convert("RGB")
        w0, h0 = img.size
        scale = min(1.0, MAX_WIDTH / w0)
        if scale < 1.0:
            img = img.resize((int(w0 * scale), int(h0 * scale)))

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"Locate the element described: {target}"},
            ]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[[img]], return_tensors="pt").to(self.device)

        t0 = time.time()
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        if with_confidence:
            gen_kwargs.update(return_dict_in_generate=True, output_scores=True)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0

        confidence = None
        if with_confidence:
            seqs = out.sequences
            scores = out.scores
            gen_ids = seqs[0, inputs.input_ids.shape[1]:]
            logps = []
            for sl, tok in zip(scores, gen_ids):
                lp = torch.log_softmax(sl[0].float(), dim=-1)
                logps.append(float(lp[tok].item()))
            confidence = float(math.exp(sum(logps) / len(logps))) if logps else 0.0
        else:
            gen_ids = out[0, inputs.input_ids.shape[1]:]
        text = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        bbox = _parse_bbox(text)
        if bbox and scale < 1.0:
            bbox = [int(c / scale) for c in bbox]

        res = {
            "bbox_2d": bbox,
            "image_size": [w0, h0],
            "model_elapsed_s": round(elapsed, 2),
            "raw_text": text,
            "backend": "transformers",
        }
        if confidence is not None:
            res["confidence"] = round(confidence, 4)
        return res


# ---------------------------------------------------------- public API ----

_BACKEND = None


def _get_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    pref = os.environ.get("BROWSERGROUND_BACKEND", "auto").lower()
    tried = []

    def try_mlx():
        try:
            import mlx_vlm  # noqa: F401
            _b = _MLXBackend(); return _b
        except Exception as e:
            tried.append(("mlx", str(e))); return None

    def try_tf():
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            _b = _TransformersBackend(); return _b
        except Exception as e:
            tried.append(("transformers", str(e))); return None

    if pref == "mlx":
        _BACKEND = try_mlx()
    elif pref == "transformers":
        _BACKEND = try_tf()
    else:  # auto: prefer mlx on darwin/arm64
        import platform
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            _BACKEND = try_mlx() or try_tf()
        else:
            _BACKEND = try_tf() or try_mlx()

    if _BACKEND is None:
        raise BrowsergroundError(
            "No backend available. Install one:\n"
            "  Apple Silicon: pip install 'browserground[mlx]'\n"
            "  CUDA / CPU:    pip install 'browserground[transformers]'\n"
            f"Errors: {tried}"
        )
    return _BACKEND


def ground(image_path: str, target: str, *, max_new_tokens: int = 64,
           with_confidence: bool = False) -> dict:
    """Locate a UI element on a screenshot.

    Args:
        image_path:  path to a screenshot (any format Pillow can open)
        target:      natural-language description of the element to click
        max_new_tokens: bound on model output (default 64; bbox JSON is small)
        with_confidence: include sequence-level confidence (transformers only)

    Returns:
        dict with at least {"bbox_2d": [x1, y1, x2, y2], "raw_text": "...",
        "model_elapsed_s": float, "backend": "mlx" | "transformers"}.
    """
    image_path = str(Path(image_path).expanduser().resolve())
    if not Path(image_path).exists():
        raise BrowsergroundError(f"image not found: {image_path}")
    backend = _get_backend()
    if isinstance(backend, _MLXBackend):
        return backend.parse(image_path, target, max_new_tokens=max_new_tokens)
    return backend.parse(image_path, target, max_new_tokens=max_new_tokens,
                         with_confidence=with_confidence)


def ground_bbox(image_path: str, target: str) -> Optional[list]:
    """Return just the [x1, y1, x2, y2] list, or None."""
    try:
        return ground(image_path, target).get("bbox_2d")
    except BrowsergroundError:
        return None


def click_xy(image_path: str, target: str):
    """Return (center_x, center_y) ready for page.mouse.click, or None."""
    b = ground_bbox(image_path, target)
    if not b:
        return None
    return (b[0] + b[2]) // 2, (b[1] + b[3]) // 2

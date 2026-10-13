"""Thin provider clients (plain HTTP via httpx -- no vendor SDK needed) + output parsing/repair."""
import base64
import json
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import httpx

SYSTEM_PROMPT = """You extract a structured product spec for a custom jewelry order from a sales brief (and optionally a reference photo).
Return ONLY a JSON object with exactly these keys:
  product_type: one of ring|pendant|earrings|bracelet|necklace|null
  metal: one of gold|white_gold|rose_gold|silver|platinum|null
  purity: one of 9k|10k|14k|18k|22k|24k|925|950|null
  stone_type: one of diamond|ruby|emerald|sapphire|cubic_zirconia|none|null
  stone_count: integer (total number of stones) or null
  dimension_mm: number in millimetres (ring = inner diameter, bracelet/necklace = length, others = longest side) or null
  finish: one of polished|matte|hammered|brushed|null
Rules: use null when the brief does not say. Never guess. Convert units to mm. No prose, no markdown."""

RETRY_SUFFIX = "\n\nYour previous reply could not be parsed as the required JSON object ({err}). Reply again with ONLY the JSON object."

VLM_COUNT_PROMPT = 'Count the individual gemstones/stones visible in this jewelry photo. Return ONLY JSON: {"stone_count": <integer>}.'

DEFAULT_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
}


class LLMReply(NamedTuple):
    text: str
    tokens_in: int
    tokens_out: int


class LLMError(Exception):
    """Transport/HTTP-level failure (as opposed to a parseable-but-bad reply)."""


def call_llm(provider: str, model: str, api_key: str, system: str, user_text: str,
             image: Optional[Tuple[bytes, str]] = None, timeout: float = 30.0) -> LLMReply:
    b64 = base64.b64encode(image[0]).decode() if image else None
    mime = image[1] if image else None
    try:
        if provider == "anthropic":
            content: List[Dict[str, Any]] = []
            if image:
                content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
            content.append({"type": "text", "text": user_text})
            r = httpx.post(DEFAULT_ENDPOINTS["anthropic"], timeout=timeout,
                           headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                           json={"model": model, "max_tokens": 600, "system": system,
                                 "messages": [{"role": "user", "content": content}]})
            r.raise_for_status()
            d = r.json()
            text = "".join(b.get("text", "") for b in d.get("content", []))
            u = d.get("usage", {})
            return LLMReply(text, u.get("input_tokens", 0), u.get("output_tokens", 0))
        if provider == "openai":
            parts: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
            if image:
                parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}})
            r = httpx.post(DEFAULT_ENDPOINTS["openai"], timeout=timeout, headers={"Authorization": "Bearer " + api_key},
                           json={"model": model, "response_format": {"type": "json_object"},
                                 "messages": [{"role": "system", "content": system}, {"role": "user", "content": parts}]})
            r.raise_for_status()
            d = r.json()
            u = d.get("usage", {})
            return LLMReply(d["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        if provider == "gemini":
            parts = [{"text": user_text}]
            if image:
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
            r = httpx.post(DEFAULT_ENDPOINTS["gemini"].format(model=model), timeout=timeout, headers={"x-goog-api-key": api_key},
                           json={"system_instruction": {"parts": [{"text": system}]}, "contents": [{"parts": parts}],
                                 "generationConfig": {"responseMimeType": "application/json"}})
            r.raise_for_status()
            d = r.json()
            u = d.get("usageMetadata", {})
            text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
            return LLMReply(text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0))
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
        raise LLMError("%s/%s: %s" % (provider, model, e)) from e
    raise LLMError("unknown provider %r" % provider)


# ---------- parsing / repair ----------

def _balanced_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_model_output(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Returns (dict|None, how) where how is 'clean' | 'repaired' | 'failed'.
    Repairs: markdown fences, prose around the JSON, trailing commas, single quotes, python literals."""
    if text is None or not text.strip():
        return None, "failed"
    try:
        obj = json.loads(text)
        return (obj, "clean") if isinstance(obj, dict) else (None, "failed")
    except ValueError:
        pass
    candidate = _balanced_object(re.sub(r"```(?:json)?", "", text))
    if candidate is None:
        return None, "failed"
    attempts = [candidate,
                re.sub(r",\s*([}\]])", r"\1", candidate),
                re.sub(r",\s*([}\]])", r"\1", candidate.replace("'", '"')),
                re.sub(r"\bNone\b", "null", re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false",
                       re.sub(r",\s*([}\]])", r"\1", candidate.replace("'", '"')))))]
    for a in attempts:
        try:
            obj = json.loads(a)
            if isinstance(obj, dict):
                return obj, "repaired"
        except ValueError:
            continue
    return None, "failed"

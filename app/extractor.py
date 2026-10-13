"""Structured extraction with a degradation ladder:
   LLM (primary model) -> retry with corrective prompt -> fallback model -> keyword rules -> all-unknown default.
The response always says which rung produced the answer (ExtractionMeta.mode)."""
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import ValidationError

from .config import Settings
from .llm import LLMError, RETRY_SUFFIX, SYSTEM_PROMPT, call_llm, parse_model_output
from .schemas import FIELDS, ExtractionMeta, FieldOut, Spec
from .vocab import Evidence, read_brief

# (provider, model, api_key, system, user_text, image, timeout) -> LLMReply ; injectable for tests
LLMCall = Callable[..., Any]

_ALIASES = {
    "metal": {"yellow_gold": "gold", "yellow gold": "gold", "sterling_silver": "silver", "sterling silver": "silver",
              "white gold": "white_gold", "rose gold": "rose_gold", "pink_gold": "rose_gold"},
    "stone_type": {"cz": "cubic_zirconia", "cubic zirconia": "cubic_zirconia", "zirconia": "cubic_zirconia",
                   "no_stones": "none", "no stones": "none", "gemstone": None, "unknown": None},
    "product_type": {"earring": "earrings", "studs": "earrings", "hoops": "earrings", "bangle": "bracelet", "chain": "necklace"},
    "purity": {"18 karat": "18k", "14 karat": "14k", "18kt": "18k", "14kt": "14k", "sterling": "925"},
}


def _normalise(field: str, v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "null", "unknown", "n/a", "na", "not specified", "unspecified"):
            return None
        s = _ALIASES.get(field, {}).get(s, s)
        if s is None:
            return None
        if field in ("stone_count", "dimension_mm"):
            m = re.search(r"-?\d+(?:\.\d+)?", s)
            return float(m.group()) if m else None
        return s.replace(" ", "_") if field in ("metal", "stone_type", "product_type") else s
    return v


def coerce_spec(raw: Dict[str, Any]) -> Tuple[Spec, List[str]]:
    """Field-level salvage: a single bad field is dropped to None (and reported), not fatal for the whole reply."""
    clean: Dict[str, Any] = {}
    issues: List[str] = []
    for f in FIELDS:
        if f not in raw:
            continue
        v = _normalise(f, raw[f])
        if f == "stone_count" and isinstance(v, float) and v == int(v):
            v = int(v)
        try:
            Spec(**{f: v})
            clean[f] = v
        except ValidationError:
            issues.append("dropped invalid %s=%r" % (f, raw[f]))
    return Spec(**clean), issues


def _empty_spec_out(reason: str) -> Dict[str, FieldOut]:
    return {f: FieldOut(value=None, confidence="low", reason=reason, source="none") for f in FIELDS}


_KIND_CONF = {"exact": "high", "inferred": "medium", "fuzzy": "medium", "ambiguous": "medium"}


def _rules_fields(brief: str) -> Dict[str, FieldOut]:
    ev = read_brief(brief)
    out: Dict[str, FieldOut] = {}
    for f in FIELDS:
        if f in ev:
            e = ev[f]
            out[f] = FieldOut(value=e.value, confidence=_KIND_CONF[e.kind], reason="keyword rules: " + e.detail, source="brief_keywords")
        else:
            out[f] = FieldOut(value=None, confidence="low", reason="not stated in brief", source="none")
    return out


def _grounded_fields(spec: Spec, brief: str, degraded: bool) -> Dict[str, FieldOut]:
    """Confidence for LLM-provided values comes from an independent check against the brief text,
    NOT from the model's own say-so. Agreement with keyword evidence -> high; no evidence -> medium;
    contradiction -> low (and the pipeline flags it)."""
    ev = read_brief(brief)
    out: Dict[str, FieldOut] = {}
    for f in FIELDS:
        v = getattr(spec, f)
        e: Optional[Evidence] = ev.get(f)
        if v is None:
            if e is not None and e.kind in ("exact", "fuzzy"):
                out[f] = FieldOut(value=None, confidence="low", reason="LLM returned null but brief keywords suggest %r" % (e.value,), source="llm+brief_keywords")
            else:
                out[f] = FieldOut(value=None, confidence="low", reason="not stated in brief", source="llm")
            continue
        if e is None:
            conf, reason = "medium", "LLM-provided; no literal support found in brief (inferred)"
        elif _same(f, v, e.value):
            conf, reason = ("high", "LLM value matches brief keywords (%s)" % e.detail) if e.kind == "exact" else ("medium", "LLM value matches weak keyword evidence (%s)" % e.detail)
        else:
            conf, reason = "low", "LLM says %r but brief keywords say %r (%s)" % (v, e.value, e.detail)
        if degraded and conf == "high":
            conf, reason = "medium", reason + "; reply needed repair/retry"
        out[f] = FieldOut(value=v, confidence=conf, reason=reason, source="llm+brief_keywords" if e is not None else "llm")
    return out


def _same(field: str, a: Any, b: Any) -> bool:
    if field == "dimension_mm":
        return abs(float(a) - float(b)) <= max(1.0, 0.05 * float(b))
    return a == b


def extract(brief: str, image: Optional[Tuple[bytes, str]], settings: Settings,
            llm_call: LLMCall = call_llm) -> Tuple[Dict[str, FieldOut], ExtractionMeta]:
    t0 = time.perf_counter()
    brief = (brief or "").strip()
    notes: List[str] = []
    meta = ExtractionMeta(mode="empty_default", provider=None, model=None)

    if not brief and not (image and settings.llm_enabled):
        meta.notes = ["empty brief and no usable vision model: returned all-unknown default"]
        meta.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return _empty_spec_out("empty brief"), meta

    if settings.llm_enabled:
        models = [m for m in (settings.model, settings.fallback_model) if m]
        models = list(dict.fromkeys(models))
        for mi, model in enumerate(models):
            user_text = "Brief: " + (brief or "(none; use the image only)")
            for attempt in (1, 2):
                meta.attempts += 1
                try:
                    reply = llm_call(settings.provider, model, settings.api_key, SYSTEM_PROMPT, user_text, image, settings.timeout_s)
                except LLMError as e:
                    notes.append("call failed: %s" % e)
                    break  # transport error: don't retry same model, go to fallback model
                meta.provider, meta.model = settings.provider, model
                if settings.price_in or settings.price_out:
                    meta.cost_usd = round((meta.cost_usd or 0) + reply.tokens_in * settings.price_in / 1e6 + reply.tokens_out * settings.price_out / 1e6, 6)
                obj, how = parse_model_output(reply.text)
                if obj is None:
                    notes.append("%s attempt %d: unparseable reply" % (model, attempt))
                    user_text = ("Brief: " + brief) + RETRY_SUFFIX.format(err="not valid JSON")
                    continue
                spec, issues = coerce_spec(obj)
                notes.extend(issues)
                if how == "repaired":
                    notes.append("repaired malformed JSON")
                retried = attempt > 1
                if retried:
                    notes.append("succeeded on retry")
                if mi > 0:
                    notes.append("used fallback model %s" % model)
                degraded = how == "repaired" or retried or bool(issues) or mi > 0
                meta.mode = "llm_retry" if retried else ("llm_repaired" if (how == "repaired" or issues) else "llm")
                meta.notes = notes
                meta.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                return _grounded_fields(spec, brief, degraded), meta
        notes.append("all LLM attempts failed; degraded to keyword rules")
    else:
        notes.append("LLM not configured (LLM_PROVIDER/API key unset); using keyword rules")

    meta.mode = "rules_fallback"
    meta.notes = notes
    meta.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return _rules_fields(brief), meta

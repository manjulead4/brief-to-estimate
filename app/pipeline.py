"""Orchestration: extract -> (CV) -> reconcile -> match -> review flags."""
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from . import cv
from .config import Settings
from .extractor import extract
from .llm import LLMError, VLM_COUNT_PROMPT, call_llm, parse_model_output
from .matcher import load_inventory, match
from .schemas import AnalyzeResponse, FIELDS, FieldOut, Flag, ImageEstimate

log = logging.getLogger("analyze")

# Plausible dimension_mm range per product type (inner diameter / length / longest side).
PLAUSIBLE_MM = {"ring": (10, 30), "pendant": (5, 90), "earrings": (3, 70), "bracelet": (40, 260), "necklace": (250, 900)}
MAX_PLAUSIBLE_STONES = 100
CRITICAL_FIELDS = ("product_type", "metal")


def _counts_agree(a: int, b: int) -> bool:
    return abs(a - b) <= max(1, int(round(0.1 * max(a, b))))


def _vlm_count(image: Tuple[bytes, str], settings: Settings, llm_call) -> Optional[int]:
    if not settings.llm_enabled:
        return None
    try:
        reply = llm_call(settings.provider, settings.model, settings.api_key, "You are a careful visual counter.", VLM_COUNT_PROMPT, image, settings.timeout_s)
    except LLMError:
        return None
    obj, _ = parse_model_output(reply.text)
    try:
        return int(obj["stone_count"]) if obj else None
    except (KeyError, TypeError, ValueError):
        return None


def analyze(brief: str, image: Optional[Tuple[bytes, str]], settings: Settings, inventory: List[Dict[str, Any]],
            llm_call=call_llm, use_vlm: bool = False) -> AnalyzeResponse:
    t_start = time.perf_counter()
    flags: List[Flag] = []
    spec, meta = extract(brief, image, settings, llm_call)

    # ---- image estimate ----
    est: Optional[ImageEstimate] = None
    t_cv = time.perf_counter()
    if image:
        if len(image[0]) > settings.max_image_bytes:
            est = ImageEstimate(status="error", reason="image larger than %d bytes" % settings.max_image_bytes)
        else:
            est = cv.estimate_count(image[0])
        if use_vlm and est.status == "ok":
            est.vlm_count = _vlm_count(image, settings, llm_call)
        if est.status == "error":
            flags.append(Flag(field="image", severity="warning", message="image unusable: " + est.reason))
    cv_ms = (time.perf_counter() - t_cv) * 1000

    # ---- reconcile stone_count across sources ----
    sc = spec["stone_count"]
    if est and est.status == "ok":
        if sc.value is None:
            spec["stone_count"] = FieldOut(value=est.count, confidence="medium" if est.confidence == "high" else "low",
                                           reason="from image only (%s)" % est.reason, source="image_cv")
        elif _counts_agree(int(sc.value), est.count):
            spec["stone_count"] = FieldOut(value=sc.value, confidence="high" if est.confidence != "low" or sc.confidence == "high" else sc.confidence,
                                           reason="brief (%s) and image (%d) agree; %s" % (sc.value, est.count, sc.reason), source="brief+image")
        else:
            spec["stone_count"] = FieldOut(value=sc.value, confidence="low", reason="brief says %s but image estimate is %d (image confidence %s)" % (sc.value, est.count, est.confidence),
                                           source="brief+image")
            flags.append(Flag(field="stone_count", severity="critical", message="sources disagree: brief/LLM=%s vs CV=%d" % (sc.value, est.count)))
        if est.vlm_count is not None and abs(est.vlm_count - est.count) > max(1, 0.1 * est.count):
            flags.append(Flag(field="stone_count", severity="warning", message="vision-LLM count %d differs from CV count %d" % (est.vlm_count, est.count)))
        st = spec["stone_type"].value
        if st == "none" and est.count > 0 and est.confidence != "low":
            flags.append(Flag(field="stone_type", severity="critical", message="brief says no stones but image shows ~%d" % est.count))
        if est.confidence == "low":
            flags.append(Flag(field="image_estimate", severity="warning", message="low-confidence image count: " + est.reason))

    # ---- generic flags ----
    for f in FIELDS:
        o = spec[f]
        if o.confidence == "low" and o.value is not None and not any(x.field == f for x in flags):
            flags.append(Flag(field=f, severity="warning", message="low confidence: " + o.reason))
        if o.value is None and f in CRITICAL_FIELDS:
            flags.append(Flag(field=f, severity="critical" if f == "product_type" else "warning", message="%s not determined" % f))
    ptype, dim, cnt = spec["product_type"].value, spec["dimension_mm"].value, spec["stone_count"].value
    if dim is not None and ptype in PLAUSIBLE_MM:
        lo, hi = PLAUSIBLE_MM[ptype]
        if not lo <= dim <= hi:
            flags.append(Flag(field="dimension_mm", severity="warning", message="%.1f mm is outside the plausible range %d-%d mm for a %s" % (dim, lo, hi, ptype)))
            spec["dimension_mm"] = spec["dimension_mm"].model_copy(update={"confidence": "low", "reason": spec["dimension_mm"].reason + "; outside plausible range"})
    if cnt is not None and cnt > MAX_PLAUSIBLE_STONES:
        flags.append(Flag(field="stone_count", severity="warning", message="%d stones is above the plausible maximum (%d)" % (cnt, MAX_PLAUSIBLE_STONES)))
    if meta.mode in ("rules_fallback", "empty_default"):
        flags.append(Flag(field="extraction", severity="info", message="extraction mode '%s': %s" % (meta.mode, "; ".join(meta.notes))))
    if ptype is None and meta.mode == "empty_default":
        flags.append(Flag(field="brief", severity="critical", message="empty brief: nothing to analyse"))

    # ---- match ----
    t_m = time.perf_counter()
    query = {f: spec[f].value for f in FIELDS}
    result = match(query, inventory)
    match_ms = (time.perf_counter() - t_m) * 1000
    if result.auto_selected is None and result.candidates:
        flags.append(Flag(field="match", severity="warning", message="no auto-selection: " + result.reason))
    # an agreeing-but-low-confidence input should not auto-select silently
    if result.auto_selected and any(spec[f].confidence == "low" and spec[f].value is not None for f in ("product_type", "metal", "stone_type", "stone_count")):
        result.auto_selected = None
        result.reason += "; auto-select withheld because a matched field has low confidence"
        flags.append(Flag(field="match", severity="warning", message="auto-select withheld: low-confidence field used in match"))

    needs_review = any(f.severity in ("warning", "critical") for f in flags)
    resp = AnalyzeResponse(request_id=uuid.uuid4().hex[:12], extraction=meta, spec=spec, matches=result, image_estimate=est,
                           needs_review=needs_review, flags=flags,
                           timings_ms={"extraction": meta.latency_ms, "cv": round(cv_ms, 2), "match": round(match_ms, 3),
                                       "total": round((time.perf_counter() - t_start) * 1000, 2)})
    log.info("request=%s mode=%s model=%s latency_ms=%.1f cost_usd=%s review=%s auto=%s", resp.request_id, meta.mode, meta.model,
             resp.timings_ms["total"], meta.cost_usd, needs_review, result.auto_selected)
    return resp

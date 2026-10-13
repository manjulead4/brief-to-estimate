"""Deterministic weighted matcher. No LLM, no randomness, no I/O at query time.

Score(candidate) = sum(w_a * sim_a) / sum(w_a)   over attributes the *query actually specifies*.
Unspecified attributes are skipped (not penalised) and the skipped weight is reported as 1 - coverage.

Weights (sum = 1.0):
  product_type 0.30   hard requirement: a ring is not a pendant. Mismatch also multiplies the total by 0.4.
  metal        0.20   exact 1.0; gold-family vs gold-family (gold / white_gold / rose_gold) 0.5; else 0
  stone_type   0.15   exact 1.0; diamond<->cubic_zirconia 0.4 (simulant); else 0
  stone_count  0.12   1 - |a-b| / max(a,b)   (0 vs 0 = 1)
  dimension    0.10   linear falloff: 1 at equal, 0 at >= 25% relative difference
  purity       0.08   exact 1.0; both gold-family and differing karat 0.5; else 0
  finish       0.05   exact 1.0 else 0

Auto-select rule: top score >= AUTO_MIN (0.85) AND lead over #2 >= AUTO_MARGIN (0.04) AND coverage >= MIN_COVERAGE (0.5)
AND the winner matches EXACTLY on every categorical attribute the query specified (product_type, metal, purity,
stone_type, finish). Partial credit (e.g. rose gold ~ yellow gold) may rank a candidate but never auto-selects it:
a wrong metal/finish is a costly silent error. (Added after the first eval run showed two such silent mis-selections.)
Otherwise the top-3 are returned for a human to choose.
Ties are broken by SKU (ascending), so identical input always yields an identical ranking.
"""
import csv
from typing import Any, Dict, List, Optional

from .schemas import Candidate, MatchResult

WEIGHTS = {"product_type": 0.30, "metal": 0.20, "stone_type": 0.15, "stone_count": 0.12,
           "dimension_mm": 0.10, "purity": 0.08, "finish": 0.05}
TYPE_MISMATCH_FACTOR = 0.4
AUTO_MIN = 0.85
AUTO_MARGIN = 0.04
MIN_COVERAGE = 0.5
DIM_TOLERANCE = 0.25
GOLD_FAMILY = {"gold", "white_gold", "rose_gold"}
TOP_K = 3
EXACT_REQUIRED = ("product_type", "metal", "purity", "stone_type", "finish")


def load_inventory(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["stone_count"] = int(r["stone_count"])
            r["dimension_mm"] = float(r["dimension_mm"])
            r["price_usd"] = float(r["price_usd"])
            rows.append(r)
    return rows


def _sim(attr: str, q: Any, item: Dict[str, Any]) -> float:
    v = item[attr]
    if attr in ("product_type", "finish"):
        return 1.0 if q == v else 0.0
    if attr == "metal":
        if q == v:
            return 1.0
        return 0.5 if q in GOLD_FAMILY and v in GOLD_FAMILY else 0.0
    if attr == "purity":
        if q == v:
            return 1.0
        return 0.5 if q.endswith("k") and v.endswith("k") else 0.0
    if attr == "stone_type":
        if q == v:
            return 1.0
        return 0.4 if {q, v} == {"diamond", "cubic_zirconia"} else 0.0
    if attr == "stone_count":
        top = max(q, v)
        return 1.0 if top == 0 else max(0.0, 1 - abs(q - v) / top)
    if attr == "dimension_mm":
        return max(0.0, 1 - abs(q - v) / (DIM_TOLERANCE * max(q, v)))
    raise KeyError(attr)


def score_item(query: Dict[str, Any], item: Dict[str, Any]):
    used = {a: w for a, w in WEIGHTS.items() if query.get(a) is not None}
    if not used:
        return 0.0, {}, 0.0
    parts = {a: round(_sim(a, query[a], item), 3) for a in used}
    total = sum(WEIGHTS[a] * parts[a] for a in used) / sum(used.values())
    if "product_type" in parts and parts["product_type"] == 0.0:
        total *= TYPE_MISMATCH_FACTOR
    return round(total, 4), parts, round(sum(used.values()), 3)


def match(query: Dict[str, Any], inventory: List[Dict[str, Any]]) -> MatchResult:
    scored = []
    coverage = 0.0
    for item in inventory:
        s, parts, coverage = score_item(query, item)
        scored.append((-s, item["sku"], s, parts, item))
    scored.sort(key=lambda t: (t[0], t[1]))
    top = [Candidate(sku=i["sku"], name=i["name"], score=s, breakdown=p, price_usd=i["price_usd"]) for _, _, s, p, i in scored[:TOP_K]]
    if coverage == 0.0:
        return MatchResult(auto_selected=None, reason="no attributes to match on", min_score=AUTO_MIN, coverage=0.0, candidates=[])
    lead = top[0].score - (top[1].score if len(top) > 1 else 0.0)
    selected: Optional[str] = None
    if coverage < MIN_COVERAGE:
        reason = "only %.0f%% of the scoring weight was specified (< %.0f%%); not enough to auto-select" % (coverage * 100, MIN_COVERAGE * 100)
    elif top[0].score < AUTO_MIN:
        reason = "best score %.2f below auto-select threshold %.2f" % (top[0].score, AUTO_MIN)
    elif any(top[0].breakdown.get(a, 1.0) < 1.0 for a in EXACT_REQUIRED):
        miss = [a for a in EXACT_REQUIRED if top[0].breakdown.get(a, 1.0) < 1.0]
        reason = "best candidate %s is not an exact match on %s; needs a human decision" % (top[0].sku, miss)
    elif lead < AUTO_MARGIN:
        reason = "top two candidates too close (lead %.3f < %.2f)" % (lead, AUTO_MARGIN)
    else:
        selected, reason = top[0].sku, "score %.2f >= %.2f with lead %.2f" % (top[0].score, AUTO_MIN, lead)
    return MatchResult(auto_selected=selected, reason=reason, min_score=AUTO_MIN, coverage=coverage, candidates=top)

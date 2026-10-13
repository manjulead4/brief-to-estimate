"""Runs the service pipeline over data/testset/cases.json and prints/saves metrics.

  python eval/run_eval.py            # uses whatever LLM_PROVIDER is configured (default: none -> keyword rules)
  python eval/run_eval.py --vlm      # also ask the vision LLM for a stone count and compare with the CV count
"""
import argparse
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app.config import Settings  # noqa: E402
from app.matcher import load_inventory, match  # noqa: E402
from app.pipeline import analyze  # noqa: E402
from app.schemas import FIELDS  # noqa: E402

TESTSET = os.path.join(ROOT, "data", "testset")


def field_ok(f, pred, exp):
    if exp is None or pred is None:
        return exp is None and pred is None
    if f == "dimension_mm":
        return abs(float(pred) - float(exp)) <= max(1.0, 0.05 * float(exp))
    return pred == exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--json", default=os.path.join(ROOT, "eval", "results", "results.json"))
    args = ap.parse_args()

    settings = Settings()
    inv = load_inventory(settings.inventory_path)
    cases = json.load(open(os.path.join(TESTSET, "cases.json")))
    rows = []
    for c in cases:
        img = None
        if c["image"]:
            img = (open(os.path.join(TESTSET, c["image"]), "rb").read(), "image/jpeg")
        r = analyze(c["brief"], img, settings, inv, use_vlm=args.vlm)
        pred = {f: r.spec[f].value for f in FIELDS}
        oks = {f: field_ok(f, pred[f], c["expected"][f]) for f in FIELDS}
        cand = [x.sku for x in r.matches.candidates]
        oracle = [x.sku for x in match(c["expected"], inv).candidates]
        wrong_auto = r.matches.auto_selected is not None and c["expected_sku"] is not None and r.matches.auto_selected != c["expected_sku"]
        auto_without_label = r.matches.auto_selected is not None and c["expected_sku"] is None
        rows.append(dict(case=c, resp=r, pred=pred, oks=oks, cand=cand, oracle=oracle, wrong_auto=wrong_auto or auto_without_label))

    # ---- extraction accuracy ----
    per_field = {f: sum(r["oks"][f] for r in rows) / len(rows) for f in FIELDS}
    overall = sum(sum(r["oks"].values()) for r in rows) / (len(rows) * len(FIELDS))
    exact_all = sum(all(r["oks"].values()) for r in rows) / len(rows)

    # ---- matching ----
    lab = [r for r in rows if r["case"]["expected_sku"]]
    top1 = sum(r["cand"][:1] == [r["case"]["expected_sku"]] for r in lab) / len(lab)
    top3 = sum(r["case"]["expected_sku"] in r["cand"] for r in lab) / len(lab)
    o_top1 = sum(r["oracle"][:1] == [r["case"]["expected_sku"]] for r in lab) / len(lab)
    o_top3 = sum(r["case"]["expected_sku"] in r["oracle"] for r in lab) / len(lab)
    autos = [r for r in rows if r["resp"].matches.auto_selected]
    auto_prec = (sum(not r["wrong_auto"] for r in autos) / len(autos)) if autos else float("nan")

    # ---- count error ----
    img_rows = [r for r in rows if r["case"]["true_count"] is not None]
    cv_err = [abs(r["resp"].image_estimate.count - r["case"]["true_count"]) for r in img_rows if r["resp"].image_estimate and r["resp"].image_estimate.count is not None]
    cv_mae = sum(cv_err) / len(cv_err) if cv_err else float("nan")
    by_conf = defaultdict(list)
    for r in img_rows:
        e = r["resp"].image_estimate
        if e and e.count is not None:
            by_conf[e.confidence].append(abs(e.count - r["case"]["true_count"]))
    by_diff = defaultdict(list)
    for r in img_rows:
        e = r["resp"].image_estimate
        if e and e.count is not None:
            by_diff[r["case"]["image_difficulty"]].append(abs(e.count - r["case"]["true_count"]))
    exp_cnt = [r for r in rows if r["case"]["expected"]["stone_count"] is not None]
    final_err = [abs((r["pred"]["stone_count"] if r["pred"]["stone_count"] is not None else 0) - r["case"]["expected"]["stone_count"]) for r in exp_cnt]
    final_mae = sum(final_err) / len(final_err)
    vlm = [(r["resp"].image_estimate.vlm_count, r["case"]["true_count"]) for r in img_rows if r["resp"].image_estimate and r["resp"].image_estimate.vlm_count is not None]
    vlm_mae = sum(abs(a - b) for a, b in vlm) / len(vlm) if vlm else None

    # ---- review flags ----
    flagged = [r for r in rows if r["resp"].needs_review]
    should = [r for r in rows if r["case"]["expect_review"]]
    flag_rate = len(flagged) / len(rows)
    recall = sum(r["resp"].needs_review for r in should) / len(should)
    precision = sum(r["case"]["expect_review"] for r in flagged) / len(flagged) if flagged else float("nan")
    errs = [r for r in rows if (not all(r["oks"].values())) or r["wrong_auto"] or
            (r["case"]["true_count"] is not None and r["resp"].image_estimate and r["resp"].image_estimate.count is not None
             and abs(r["resp"].image_estimate.count - r["case"]["true_count"]) > 2)]
    silent = [r for r in errs if not r["resp"].needs_review]

    # ---- confidence calibration (field level) ----
    cal = defaultdict(lambda: [0, 0])
    for r in rows:
        for f in FIELDS:
            conf = r["resp"].spec[f].confidence
            cal[conf][0] += r["oks"][f]
            cal[conf][1] += 1

    lat = sorted(r["resp"].timings_ms["total"] for r in rows)
    modes = defaultdict(int)
    for r in rows:
        modes[r["resp"].extraction.mode] += 1

    print("=" * 66)
    print("cases: %d | images: %d | extraction modes: %s" % (len(rows), len(img_rows), dict(modes)))
    print("-- Extraction field accuracy (7 fields x %d cases) --" % len(rows))
    for f in FIELDS:
        print("  %-14s %5.1f%%" % (f, 100 * per_field[f]))
    print("  %-14s %5.1f%%   all-7-fields-correct cases: %.1f%%" % ("OVERALL", 100 * overall, 100 * exact_all))
    for g in ("core", "natural"):
        gr = [r for r in rows if r["case"]["group"] == g]
        print("  group %-8s n=%2d  field acc %.1f%%  all-7-correct %.1f%%" % (g, len(gr), 100 * sum(sum(r["oks"].values()) for r in gr) / (len(gr) * len(FIELDS)), 100 * sum(all(r["oks"].values()) for r in gr) / len(gr)))
    print("-- Matching (%d cases with a labelled SKU) --" % len(lab))
    print("  end-to-end  top-1 %.1f%%  top-3 %.1f%%" % (100 * top1, 100 * top3))
    print("  oracle spec top-1 %.1f%%  top-3 %.1f%%   (matcher alone, perfect extraction)" % (100 * o_top1, 100 * o_top3))
    print("  auto-selected %d/%d cases, precision of auto-selection %.1f%%" % (len(autos), len(rows), 100 * auto_prec))
    print("-- Count (images: %d) --" % len(img_rows))
    print("  CV count MAE vs image truth: %.2f   by confidence: %s" % (cv_mae, {k: round(sum(v) / len(v), 2) for k, v in by_conf.items()}))
    print("  by difficulty: %s" % {k: round(sum(v) / len(v), 2) for k, v in by_diff.items()})
    print("  final spec stone_count MAE (%d labelled): %.2f" % (len(exp_cnt), final_mae))
    print("  vision-LLM count MAE: %s" % ("%.2f (n=%d)" % (vlm_mae, len(vlm)) if vlm else "not run (needs --vlm and an API key)"))
    print("-- Review flags --")
    print("  flag rate %.1f%% (%d/%d) | recall on should-flag %.1f%% | precision %.1f%%" % (100 * flag_rate, len(flagged), len(rows), 100 * recall, 100 * precision))
    print("  cases with an error: %d, of which NOT flagged (silent failures): %d %s" % (len(errs), len(silent), [r["case"]["id"] for r in silent]))
    print("-- Field-confidence calibration (accuracy per label) --")
    for k in ("high", "medium", "low"):
        if cal[k][1]:
            print("  %-6s %5.1f%%  (n=%d)" % (k, 100 * cal[k][0] / cal[k][1], cal[k][1]))
    print("-- latency ms: median %.1f  p95 %.1f --" % (lat[len(lat) // 2], lat[int(0.95 * (len(lat) - 1))]))

    # ---- worst cases ----
    def badness(r):
        e = r["resp"].image_estimate
        ce = abs(e.count - r["case"]["true_count"]) if (e and e.count is not None and r["case"]["true_count"] is not None) else 0
        return (len(FIELDS) - sum(r["oks"].values())) + 3 * r["wrong_auto"] + 0.2 * ce + (2 if (not r["resp"].needs_review and not all(r["oks"].values())) else 0)
    print("-- worst cases --")
    for r in sorted(rows, key=badness, reverse=True)[:6]:
        bad = {f: (r["pred"][f], r["case"]["expected"][f]) for f in FIELDS if not r["oks"][f]}
        e = r["resp"].image_estimate
        print("  %s  brief=%r\n      wrong fields (pred, expected): %s\n      auto=%s expected_sku=%s cand=%s flagged=%s cv=%s true=%s" % (
            r["case"]["id"], r["case"]["brief"][:70], bad, r["resp"].matches.auto_selected, r["case"]["expected_sku"], r["cand"],
            r["resp"].needs_review, e.count if e else None, r["case"]["true_count"]))

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"overall_field_acc": overall, "per_field": per_field, "top1": top1, "top3": top3, "oracle_top1": o_top1, "oracle_top3": o_top3,
                   "auto_precision": auto_prec, "cv_mae": cv_mae, "final_count_mae": final_mae, "flag_rate": flag_rate, "flag_recall": recall,
                   "flag_precision": precision, "silent_failures": [r["case"]["id"] for r in silent],
                   "cases": [{"id": r["case"]["id"], "pred": r["pred"], "oks": r["oks"], "cand": r["cand"], "auto": r["resp"].matches.auto_selected,
                              "flagged": r["resp"].needs_review, "flags": [x.message for x in r["resp"].flags]} for r in rows]}, f, indent=1)


if __name__ == "__main__":
    main()

# Evaluation

Run: `python eval/build_testset.py && python eval/run_eval.py` (raw output saved in `eval/results/latest_report.txt`, per-case in `eval/results/results.json`).

## What was measured, and what was NOT
* **Two runs.** (A) the offline `rules_fallback` path (all tables below unless marked LLM); (B) one live run with Google Gemini (`gemini-3.5-flash`, fallback `gemini-3.1-flash-lite`) - see "LLM run" section.
  The LLM run is a **single run** (no repeats), made while the API was returning intermittent 503s, so some calls went to the fallback model or to rules; treat it as indicative, not definitive.
* **Test set:** 44 cases = 34 `core` + 10 `natural` briefs (written after the extractor, not adjusted to pass), 13 synthetic images (easy/medium/hard).
  Spec labels and expected SKUs were written by hand, not derived from the matcher. Labelling rule: extract only what the brief states; if only the photo gives a stone count, the photo count is the label.
* Same author wrote briefs, labels and rules -> numbers are optimistic. CV parameters were tuned on dev seeds 1000-3000; test images use seeds 7000+.

## Metrics (final)
| Metric | Result |
|---|---|
| Extraction field accuracy, all 7 fields x 44 cases | **98.1%** (core 99.2%, natural 94.3%) |
| Cases with all 7 fields correct | **88.6%** (core 94.1%, natural 70.0%) |
| Per field | type 97.7, metal 100, purity 100, stone type 97.7, stone count 93.2, dimension 97.7, finish 100 |
| Matching top-1 / top-3 (33 cases with a labelled SKU, end-to-end) | **100% / 100%** |
| Matching top-1 / top-3 with ground-truth specs (matcher alone) | 100% / 100% |
| Auto-selection: rate / precision | 28 of 44 cases / **100%** |
| CV count MAE vs image truth (13 images) | **3.46** (high-confidence images: 0.5; low-confidence: 8.2) |
| CV MAE by difficulty | easy 6.4 (one dense 45-stone image dominates), medium 0.8, hard 3.0 |
| Final `stone_count` MAE (26 labelled) | 1.19 |
| Vision-LLM count MAE | not run |
| Review-flag rate | **45.5%** (20/44) |
| Flag recall on cases that should be flagged / precision | 92.9% / 65.0% |
| Silent failures (an error that was not flagged) | 1 (`c31`) |
| Latency (rules path, includes CV) | median 0.3 ms, p95 3.6 ms (LLM calls would dominate; the log records them) |

Reading the table honestly: top-1 = 100% is partly because the inventory is small (40 rows) and the labels are mine; the matcher alone is deterministic and
was never the weak point. The interesting results are the CV errors, the flag precision, and the silent failures.

**Confidence calibration, rules path (field level, accuracy per label):** high 100% (n=173), medium 100% (n=14), low 95% (n=121). "Low" includes correctly-null fields ("not stated"), which is why low is
not much worse; the labels are not discriminating much on this offline path. Medium has only 14 samples. No claim of calibration should be made from this.

## LLM run (Gemini, single run, `--vlm`) vs offline rules
Raw output: `eval/results/llm_report.txt`, `eval/results/results_llm.json`. Extraction modes: 41 `llm`, 2 `rules_fallback` (API errors), 1 `empty_default`.

| Metric | Rules (offline) | LLM (Gemini) |
|---|---|---|
| Field accuracy (7 x 44) | 98.1% | **93.2%** |
| All 7 fields correct | 88.6% | 65.9% (core 70.6%, natural 50.0%) |
| Weakest fields | stone count 93.2 | stone type 84.1, stone count 86.4, purity 88.6 |
| Top-1 / top-3 | 100% / 100% | 100% / 100% |
| Auto-select precision | 100% (28/44) | 93.9% (33/44; wrong: `c32`, `n04`) |
| Flag rate / recall / precision | 45.5% / 92.9% / 65.0% | 31.8% / 78.6% / 78.6% |
| Silent failures (error, not flagged) | 1 | **12** |
| Field confidence: high / medium / low accuracy | 100 (n=173) / 100 (n=14) / 95 (n=121) | 89.5 (n=19) / 91.9 (n=186) / 96.1 (n=103) |
| Vision-LLM count MAE | not run | 0.00 but **n=3 of 13** (the rest failed/timed out) - not meaningful |
| Latency | median 0.3 ms | median 7.1 s, p95 34 s (includes retries/fallbacks on 503s) |

**How to read this - the LLM did not "lose" in the way the headline suggests, but it exposed real problems:**
1. **Most LLM "errors" are inferences that my labelling rule counts as wrong**: it returned `purity=925` for "silver", `stone_type=none` for "hammered silver band", `stone_count=1` for "a teardrop diamond", `metal=gold` from the gold colour in a photo (`c32`).
   A sales rep might well accept several of these; but they are *guesses the brief does not state*, and the spec says "knows when it is unsure". The labels favour the rules (same author, "only what is stated"), so the rules-vs-LLM gap is partly label bias.
2. **The grounding-based confidence did not make those guesses reviewable.** They get `medium` ("LLM-provided; no literal support in brief"), and `medium` does not trigger review, hence 12 silent failures (vs 1). Overall flag *recall* dropped to 78.6%. This is the main design weakness found.
   Also, only 19 fields reached `high`: any reply that needed the fallback model/retry is capped at `medium`, and the API was flaky during the run.
3. **Ambiguity cases the LLM resolved differently**: `c02` "13 diamonds around the centre stone" -> LLM 14 (13 + centre) while my label says 13. Defensible; the brief is ambiguous and neither source flagged it.
4. **`n04`** ("1ct round brilliant, platinum band"): the LLM correctly inferred *diamond* (the rules could not), but with the count unstated the matcher auto-selected R-011, a 20-stone eternity band. Auto-select should not fire when a field that strongly separates candidates (stone count) is unspecified.
5. Matching is still 100% top-1/top-3 because the matcher is deterministic and the inventory small.

**Changes I would make next (not applied, to avoid tuning on this test set again):** (a) treat LLM values with no textual support as `low`/review-worthy, or instruct the model in the prompt not to infer unstated values; (b) require stone_count to be specified (or from the image) before auto-select when candidates differ strongly in count; (c) make "around the centre stone" style counts explicit in the schema (centre + surround); (d) retry the vision count with backoff and report how many calls failed; (e) repeat the LLM run 3x to measure run-to-run variance.

## Changes made after seeing results (so the final numbers are not fully held-out)
| | before | after |
|---|---|---|
| Auto-selection precision | 93.1% (2 wrong auto-picks) | 100% |
| Silent failures | 3 (`c31`, `n05`, `n06`) | 1 (`c31`) |
| Top-1 | 97.0% | 100% |
| Flag recall | 85.7% | 92.9% |
| Natural group, all-7-correct | 60% | 70% |

1. `n05` (rose gold pendant request) auto-selected a *yellow gold* pendant and `n06` (hammered request) a *polished* bracelet: partial credit + small finish weight let a near-miss clear 0.85.
   Fix: auto-select now requires an exact match on every specified categorical attribute (this is principled, but I made it after seeing these cases).
2. `n01`: "rose-gold" with a hyphen was not recognised. Fix: allow hyphen in colour-gold phrases.
3. A unit test (not the eval) found `-5` was coerced to `5.0`; fixed.

## Failure analysis: the 3 worst

**1. `c33` - dense tennis-bracelet photo (45 stones): CV says 16.** Hough's `minDist` (about one stone diameter) suppresses neighbouring touching stones and the enclosed-region method found 0, because at 7 px radius the edges of adjacent stones merge.
   Mitigation worked (confidence `low`, the case is flagged, the stone count is marked low), but the number itself is wrong and the sales rep gets no usable count.
   *Change:* a scale-adaptive approach (estimate stone size first, then template/blob matching), or a trained detector (YOLO-style, fine-tuned on real jewelry photos); crop/upsample dense rows; fall back to the vision-LLM for > 40 stones and report both.

**2. `c02` - a false alarm that blocks a perfect match.** Brief says 13 diamonds (correct); Hough over-counts the halo at 16 and the contour method says 13. Taking the *max* of the two (chosen on the dev set where detectors mostly under-count) returns 16,
   and the 20% tolerance calls that "agreement" -> CV confidence `high` while being wrong. The pipeline then raises a *critical* "sources disagree" flag and withholds auto-selecting R-002, which scored 1.00. Cost: an unnecessary human review (this is a big part of the 65% flag precision).
   *Change:* use a 3rd detector and take the median, or weight the brief's explicit count above a CV count with a tolerance that depends on layout (halos over-count the centre stone's reflections); report disagreement as `warning` when the brief's count is explicit and the CV confidence is only `medium`. The `max` rule is a dev-set artefact of the generator and should not be trusted on real photos.

**3. `c31` - the one silent failure; plus `n04`, `n07` extraction limits.** "dimond solitare ring" - fuzzy matching rescued the product and stone, but the "solitaire => 1 stone" inference only fires on the exact spelling, so `stone_count` was left null.
   Nothing flagged it because a null non-critical field is not treated as a problem; the match still landed on the right SKU. Related rule-path gaps in the same family: `n04` ("1ct round brilliant" implies a diamond: needs world knowledge) and `n07` ("size 9" implies a ring: no product noun).
   Both of those were flagged correctly (low-confidence/undetermined), which is the intended behaviour: the rules path fails *loudly*.
   *Change:* run the fuzzy pass before the inference rules; flag when a stone type is known but the count is not stated and the product usually carries a count; and, mainly, use the LLM for these (this is what it is for) and measure it.

Honourable mention: `n09` ("2 mm band") - 2 mm is the band width, but the rule took it as the ring dimension. It was caught by the plausible-range flag (needs review) rather than being understood.

## Next steps
1. Repeat the Gemini run several times and with a second provider; get a complete vision-LLM count comparison (only 3 of 13 returned); apply the changes listed under "LLM run" and re-measure on a fresh split.
2. Real photos: collect/label 50+ real jewelry images; re-tune or replace the CV; measure how badly synthetic-tuned parameters transfer.
3. A larger, independently written brief set (ideally from real sales emails) to replace the single-author labels; add inter-annotator agreement on the SKU labels.
4. Calibrate the review thresholds (auto-select 0.85, margin 0.04, CV tolerance 20%) on a held-out split instead of by reasoning; track flag precision, since a 65% precision means about one in three reviews is wasted.
5. Dimension handling: separate fields for ring size vs band width, per-product semantics.

# Brief to Priced Estimate

A FastAPI service that turns a short sales brief (+ optional reference photo) for a custom **jewelry** piece into
a validated spec, a deterministic inventory match, an image-based stone count, and per-field confidence with
human-review flags. Domain: jewelry (rings, pendants, earrings, bracelets, necklaces); the measurable quantity from the
image is **the number of small stones**.

Design goal: *a service that knows when it is unsure*, not a polished demo.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional; default LLM_PROVIDER=none runs fully offline
uvicorn app.main:app --reload   # http://localhost:8000/docs
curl -F 'brief=18k gold solitaire ring, single diamond, size 7' -F 'image=@data/testset/images/c07_cluster_1_easy.jpg' localhost:8000/analyze
```

Docker: `docker compose up --build` (Dockerfile/compose are written but I did not build the image in this environment).

Tests: `python -m pytest -q`  (30 tests: matcher, malformed model output, CV, API).
Evaluation: `python eval/build_testset.py && python eval/run_eval.py` (add `--vlm` with an API key to also score a vision-LLM count).

### Configuration (environment variables only; nothing is hardcoded or committed)
| var | meaning |
|---|---|
| `LLM_PROVIDER` | `anthropic` \| `openai` \| `gemini` \| `none` (default `none` = keyword-rules only) |
| `LLM_MODEL`, `LLM_FALLBACK_MODEL` | primary and fallback model (provider defaults exist; override freely) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` | key for the chosen provider |
| `LLM_PRICE_IN_PER_MTOK`, `LLM_PRICE_OUT_PER_MTOK` | optional; enables per-request cost in the log/response |

## API
`POST /analyze` (multipart): `brief` (text), `image` (optional file), `use_vlm` (optional bool).
Response (abridged): `extraction{mode,notes,model,latency_ms,cost_usd}`, `spec{field:{value,confidence,reason,source}}`,
`matches{auto_selected,reason,candidates[top-3 with score breakdown]}`, `image_estimate{count,confidence,reason,methods}`,
`needs_review`, `flags[{field,severity,message}]`, `timings_ms`. Each request also writes one log line with latency, model, cost and outcome.

## Design

**1. Extraction (`app/extractor.py`, `app/llm.py`).** The LLM returns a JSON object; Pydantic validates it. A degradation
ladder guarantees an answer, and `extraction.mode` says which rung was used:
`llm` -> `llm_repaired` (fences/prose/trailing commas/single quotes fixed, or invalid fields dropped to null) -> `llm_retry`
(one retry with a corrective prompt) -> fallback model -> `rules_fallback` (deterministic keyword reader) -> `empty_default` (all-unknown, flagged).
Transport errors skip the retry and go straight to the fallback model.
*Confidence is not taken from the model.* It is computed by cross-checking each LLM value against the keyword reader on the brief:
agreement -> high, no textual support -> medium, contradiction -> low. A repaired/retried reply caps confidence at medium.

**2. Matching (`app/matcher.py`).** Weighted score over only the attributes the brief specifies (weights and similarity tables are documented at the top of the file).
Ties break on SKU, so identical input always gives an identical ranking (tested). Auto-select needs score >= 0.85, a lead >= 0.04 over #2,
>= 50% of the weight specified, **and an exact match on every specified categorical attribute** (type, metal, purity, stone, finish).
*Why not the LLM:* ranking must be reproducible and auditable (a sales rep must be able to see exactly why SKU X beat SKU Y, and the same brief must not
produce a different quote tomorrow), it is free and takes microseconds, it can't hallucinate a SKU that doesn't exist, and the weights are a business decision that
should be edited in code review, not buried in a prompt.

**3. Image estimate (`app/cv.py`).** Two independent classical detectors: Hough circles and an enclosed-region (edge-fenced) contour method.
Count = the larger of the two (on the dev images both under-count more than they over-count; this did not hold for every test image, see EVALUATION.md case `c02`); agreement within 20% -> confidence high/medium,
disagreement, blur/low contrast, zero hits or > 80 stones -> low. Optional vision-LLM count (`use_vlm`) is compared and flagged on disagreement.

**4. Confidence + review (`app/pipeline.py`).** Every field has `{value, confidence, reason, source}`. Flags: low-confidence value, missing type/metal,
dimension outside a per-product plausible range, > 100 stones, brief vs image count disagreement, "no stones" vs stones in the image,
VLM vs CV disagreement, no/withheld auto-selection. `needs_review` is true for any warning/critical flag.

**5. Evaluation.** See `EVALUATION.md`. 44 labelled cases, 13 synthetic images.

## Limitations (please read)
* **The LLM path was run live only once, on Gemini** (`gemini-3.5-flash` with `gemini-3.1-flash-lite` fallback), during intermittent API 503s; results are in `EVALUATION.md` and are indicative, not definitive. The Anthropic and OpenAI request payloads follow each vendor's documented format but are **untested**.
  Retry/repair/fallback logic is also covered by tests with a scripted fake client. The vision-LLM count comparison is incomplete (3 of 13 calls returned).
* Keep API model names in env vars: the originally-defaulted Gemini models (2.0/2.5) had been retired for new keys and returned 404 (the service degraded to rules and said so).
* **Images are synthetic** (OpenCV-drawn stones on a band). Real photos have reflections, occlusion, perspective and metal prongs. CV parameters were tuned on the
  synthetic generator (dev seeds disjoint from test seeds) and assume stones of ~5-25 px radius in a ~512 px image. Expect real-world accuracy to be worse.
* Test briefs and the rules extractor share an author, so rules-path accuracy is optimistic; the `natural` subset (written afterwards) is the more honest number. Two fixes were made after seeing test results, so that subset is no longer fully held out (details in EVALUATION.md).
* `dimension_mm` means different things per product (ring inner diameter, necklace length...); there is no width/height split. Ring size conversion is a linear approximation of US sizing.
* Inventory is 40 hand-written rows; weights are reasoned, not learned.
* Multipart only (no JSON/base64 body). No auth, rate limiting or persistence.

## AWS deployment sketch
Container on **ECS Fargate** behind an **ALB** (stateless, scales on CPU/request count; OpenCV is CPU-only so no GPU needed). API keys in **Secrets Manager** injected as task env vars;
uploaded images to **S3** (short lifecycle) if audit is needed; logs/metrics (latency, cost, mode, flag rate) to **CloudWatch** with alarms on `rules_fallback` rate and p95 latency;
inventory in S3/RDS if it outgrows a CSV. If the LLM provider is Anthropic or others via **Bedrock**, use an IAM task role instead of a key. Lambda is viable for low volume but
OpenCV layer size and cold starts make Fargate the simpler default.

## How AI tools were used
This was built pair-programming style with Claude Code, which produced the first draft of most modules, the synthetic image generator and the test scaffolding.
What I changed or rejected after measuring: the first Hough parameters were bad (MAE 2.6 on dev); I grid-searched them on a dev set and kept the better ones.
An "enclosed-region" contour method was tried as primary and rejected (worse than Hough alone, kept only as the cross-check). The first matcher auto-selected a
yellow-gold pendant for a rose-gold request because of partial credit; I added the exact-categorical-match rule. A unit test caught that the number coercion silently turned `-5` into `5`.
An initial 98% extraction score was treated with suspicion (same author for briefs and rules), so a second "natural" brief set was added and reported separately.
Items I did not verify against reality: the live LLM calls (see Limitations).

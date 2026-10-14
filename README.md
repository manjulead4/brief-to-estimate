# Brief to Priced Estimate

This is my submission for the "Brief to Priced Estimate" task. A sales rep sends a short brief and maybe a photo of a custom piece. The service turns that into a structured spec, matches it against a small inventory, estimates a quantity from the photo, and says how sure it is about each answer.

I picked **jewelry** (rings, pendants, earrings, bracelets, necklaces). The quantity I estimate from the photo is the **number of small stones**.

The goal I kept in mind was a service that knows when it is unsure, so a lot of the work went into flags and the evaluation rather than into the happy path.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload      # docs at http://localhost:8000/docs
```

It runs with no API key at all (it falls back to keyword rules). Example call:

```bash
curl -F 'brief=18k gold solitaire ring, single diamond, size 7' \
     -F 'image=@data/testset/images/c07_cluster_1_easy.jpg' \
     localhost:8000/analyze
```

Tests: `python -m pytest -q`

Evaluation: `python eval/build_testset.py && python eval/run_eval.py` (add `--vlm` if a key is set, to also ask the model to count stones in the image).

There is a Dockerfile and docker-compose.yml. I wrote them but did not build the image here, so treat them as untested.

### Settings

Everything comes from environment variables, nothing is hardcoded. Copy `.env.example` to `.env` (it is gitignored).

| variable | what it does |
|---|---|
| `LLM_PROVIDER` | `anthropic`, `openai`, `gemini` or `none` (default `none`) |
| `LLM_MODEL`, `LLM_FALLBACK_MODEL` | primary model and the one to try if it fails |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` | key for the chosen provider |
| `LLM_PRICE_IN_PER_MTOK`, `LLM_PRICE_OUT_PER_MTOK` | optional, turns on per-request cost logging |

Model names go stale quickly. The Gemini defaults I first wrote were already retired when I tried them (404), so check the names against your provider.

## The endpoint

`POST /analyze` takes multipart form data: `brief` (text), `image` (optional file) and `use_vlm` (optional).

The response has the extraction info (which path was used, latency, cost), each spec field as `{value, confidence, reason, source}`, the top 3 inventory matches with a score breakdown, the image estimate, a `needs_review` boolean and a list of flags. Every request also logs one line with latency, model and cost.

## How it works

**Extraction.** The model returns JSON and Pydantic validates it. If the output is bad, I try these in order and the response says which one was used: repair it (strip code fences, extra text, trailing commas), retry once with a corrective prompt, switch to the fallback model, fall back to keyword rules, and finally return an all-unknown result. A field that fails validation is set to null instead of failing the whole reply.

For confidence I did not use the model's own opinion. Each value is checked against a simple keyword reader run on the brief. If they agree the field is "high". If the model gave a value with no support in the text it is "medium". If they contradict each other it is "low". Anything that needed a repair, retry or fallback model is capped at "medium".

**Matching.** This is plain code, no LLM. Each inventory row gets a weighted score over the attributes the brief actually mentions (weights are listed at the top of `app/matcher.py`). Ties are broken by SKU so the same input always gives the same ranking, and there is a test for that. I kept the LLM out of this step because a rep needs to see exactly why one item beat another, the same brief shouldn't give a different quote tomorrow, and the model can't invent a SKU that doesn't exist.

An item is auto-selected only if its score is at least 0.85, it leads the runner-up by 0.04, enough of the brief was specified, and every categorical attribute that was specified (type, metal, purity, stone, finish) matches exactly. The last rule came from the first evaluation run, where partial credit picked a yellow gold pendant for a rose gold request.

**Image count.** Two OpenCV detectors: Hough circles, and a contour method that counts regions enclosed by edges. I take the larger count. If the two agree within about 20% the confidence is high, and if they disagree, the image is blurry, nothing is found, or there are more than 80 stones, it is low.

**Flags.** A field is flagged for review when its confidence is low, when type or metal is missing, when a dimension is outside a plausible range for the product, when the brief and the image disagree on the count, or when nothing could be auto-selected.

## Evaluation

Details and the failure analysis are in `EVALUATION.md`. In short: 44 hand-labelled cases and 13 synthetic images. The keyword path gets about 98% field accuracy on my labels, and Gemini got about 93% in a single run, but with many more unflagged errors. Read the caveats in that file before quoting any number.

## Limitations

- The LLM path was run live once, on Gemini only, while the API was returning errors. The Anthropic and OpenAI request formats follow their docs but I have not run them.
- The images are synthetic (drawn with OpenCV), and I tuned the CV parameters on them. Real photos will be harder.
- I wrote the briefs, the labels and the keyword rules, so the keyword numbers are optimistic. I fixed a few things after seeing results, so the test set is not perfectly held out.
- `dimension_mm` means different things per product (ring inner diameter, necklace length), and there is no separate width field.
- The inventory is 40 rows I made up. The weights are chosen by reasoning, not learned.
- Multipart upload only. No auth, rate limiting or storage.

## If I deployed this on AWS

Container on ECS Fargate behind an ALB. It is stateless and CPU-only, so no GPU. API keys in Secrets Manager, passed as environment variables. Uploaded images to S3 with a short lifecycle if I need an audit trail. Logs and metrics to CloudWatch, with alarms on the share of requests that fall back to rules and on p95 latency. Lambda would also work at low volume, but OpenCV's size and cold starts make Fargate simpler.

## How I used AI tools

I built this with Claude Code, which wrote most of the first drafts of the code, the synthetic image generator and the tests. I gave it the task and supplied my own Gemini key for the live tests. What I took from the results and changed:

- The first Hough settings were poor, so we grid-searched them on a separate dev set and kept the better ones.
- An alternative contour method was tried as the main counter and dropped because it was worse. It stayed only as a cross-check.
- The first matcher auto-selected near-misses because of partial credit, which led to the exact-match rule above.
- A unit test caught `-5` being read as `5`.
- An early 98% score looked too good because the same author wrote the briefs and the rules, so I added 10 messier briefs and report them separately.
- The first live Gemini run scored lower than the keyword rules and showed that unsupported guesses get "medium" confidence and are not flagged. I wrote that up in `EVALUATION.md` and did not tune it away.

The live LLM results are a single run and are not conclusive.

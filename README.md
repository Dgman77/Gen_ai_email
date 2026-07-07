# Hiver Challenge — Gen-AI Email Suggested-Response System

A system that suggests customer-support email replies using retrieval-augmented
generation, and scores the quality of those replies with a five-metric evaluator —
built for Hiver's open technical challenge.

## What this project does, in one sentence

You give it a customer email → it finds similar past emails from a purpose-built
dataset → it asks an LLM to write a reply in that style → a separate evaluation
step scores how good that reply actually is, on five dimensions, not just one
similarity number.

---

## Project structure

```
hiver-genai-email/
├── data/
│   ├── build_dataset.py       # STAGE 1 — builds the dataset (offline, no API key)
│   ├── dataset.jsonl          # 104 examples, full metadata (already built)
│   ├── train.jsonl            # 83 examples — used for retrieval
│   └── test.jsonl             # 21 examples — held out, used for evaluation
├── generate_dataset_llm.py    # OPTIONAL — adds more examples via Gemini (not required)
├── generator.py               # STAGE 2 — suggests a reply for a given email
├── evaluator.py               # STAGE 3 — scores generated replies
├── validate_metric.py         # STAGE 4 — checks the metric itself is trustworthy
├── run_demo.py                # runs stages 2+3 (and 1, if missing) in one command
├── requirements.txt
├── .env.example                # copy to .env, add your Gemini API key
└── README.md                  # this file
```

---

## How the pieces connect

```
data/build_dataset.py
        │  (offline, seeded, no API key — already run once)
        ▼
data/dataset.jsonl  →  split 80/20  →  data/train.jsonl + data/test.jsonl
                                              │
                                              ▼
                                      generator.py
                        (retrieves similar train.jsonl examples,
                         asks Gemini to write a reply for each
                         email in test.jsonl)
                                              │
                                              ▼
                                generated_replies.jsonl
                                              │
                                              ▼
                                      evaluator.py
                        (scores each generated reply against its
                         reference row in test.jsonl, 5 metrics)
                                              │
                                              ▼
                                       results.json
                        (per-response scores + overall summary)

                                validate_metric.py
                     (separately: checks the metric itself ranks
                      known-good replies above known-bad ones —
                      not part of the main pipeline, run once to
                      build confidence in the scoring system)
                                              │
                                              ▼
                                validation_report.json
```

`run_demo.py` runs the whole bottom half of this diagram in one command.

---

## Approach

This is built as **retrieval-augmented generation (RAG-style prompting)**, not a
fine-tuned model — a deliberate choice given the build timeline, and one that's
disclosed rather than overclaimed. Fine-tuning responsibly (data validation,
training, eval) doesn't fit a short build window; RAG-style prompting conditioned
on a purpose-built dataset is an honest, effective, and production-realistic way
to "train on a dataset you create" for this kind of task.

Every script runs **with or without an API key** — if no key is set, the
generator and evaluator fall back to deterministic local methods, so the project
is always runnable end-to-end, not just runnable-in-theory.

---

## Stage 1 — Dataset

**`data/build_dataset.py`** is fully offline and deterministic (fixed random
seed) — no API key or internet required. It has already been run; the output
(`dataset.jsonl`, `train.jsonl`, `test.jsonl`) is included in this repo.

**104 examples** across **20 support categories**: Billing Issue, Refund
Request, Duplicate Charge, Failed Payment, Subscription Cancellation,
Upgrade/Downgrade, Account Locked, Login Problems, Password Reset, MFA
Problems, Shipping Delay, Missing Package, Wrong Item Received, Damaged
Product, Return Request, Technical Bug, Feature Request, Integration Problem,
Positive Feedback, Escalation Request.

Diversity comes from randomized (fixed-seed) combinations of names,
order/invoice IDs, amounts, dates, and multiple template variants per category,
plus explicit tone/difficulty variation, plus 4 hand-crafted edge cases.

Every row carries rich metadata — not just an email/reply pair:

```json
{
  "id": "0034",
  "category": "Account Locked",
  "difficulty": "hard",
  "customer_email": "...",
  "ideal_agent_reply": "...",
  "intent": "account lockout",
  "entities": ["ORD-12345"],
  "expected_actions": ["Apologize", "Unlock account", "Provide workaround"],
  "tone": "angry",
  "resolution_type": "Resolved - account unlocked",
  "requires_followup": true,
  "contains_multiple_issues": false,
  "language": "English",
  "source": "template_synthetic"
}
```

This metadata is what makes the Stage 3 evaluator meaningfully better than
plain text-similarity scoring.

**Optional expansion:** `generate_dataset_llm.py` can generate *additional*
examples via the Gemini API in small batches (10–20 at a time, one category
per call — single giant-batch generations drift into repetitive phrasing).
This is optional; the 104-example base set already satisfies the dataset
requirement without it.

---

## Stage 2 — Generator

**`generator.py`** — retrieval-augmented generation:

1. TF-IDF vectorizes all `train.jsonl` customer emails (scikit-learn, local,
   instant, no API call).
2. For a new email, retrieves the top-3 most similar past (email, reply)
   pairs by cosine similarity.
3. Builds a few-shot prompt using those examples + the new email.
4. Calls the Gemini API to generate the reply.

**Fallback (no API key):** returns the closest-matching retrieved ideal
reply as a baseline, lightly adapted — always runnable, though real
Gen-AI quality requires a free API key (see setup below).

```bash
# Single email
python generator.py --email "My order never arrived, order ORD-55123"

# Batch — generates a reply for every email in the test set
python generator.py --batch data/test.jsonl --out generated_replies.jsonl
```

---

## Stage 3 — Evaluator

**`evaluator.py`** scores each generated reply on **five weighted metrics**,
pulled from the dataset's metadata — not just a single similarity number:

| Metric | Weight | Method |
|---|---|---|
| Semantic similarity | 25% | Embedding cosine similarity vs. `ideal_agent_reply` (sentence-transformers, local; TF-IDF fallback if no internet) |
| Entity coverage | 20% | % of `entities` (order IDs, amounts, etc.) mentioned in the reply |
| Action coverage | 25% | % of `expected_actions` the reply actually performs (LLM-judge; keyword-heuristic fallback) |
| Intent match | 15% | Does the reply address the correct `intent`? (LLM-judge; keyword fallback) |
| Tone match | 15% | Does the reply's tone fit the situation? (LLM-judge; neutral fallback) |

**Why this metric is the right one:** BLEU/ROUGE/plain cosine similarity fail
for email replies because many different phrasings can all be correct — there's
no single "ground truth" sentence. What actually matters, the way a human QA
reviewer would grade a reply, is whether it covers the right actions,
addresses the right intent, mentions the specifics the customer cares about,
and uses an appropriate tone. Weighting action coverage highest (tied with
semantic similarity) reflects that *doing the right thing* matters more than
*sounding like the reference reply*.

Output: `results.json` with per-response scores plus an overall summary —
mean/median/std, broken down by difficulty tier and by category.

```bash
python evaluator.py --generated generated_replies.jsonl --reference data/test.jsonl --out results.json
```

---

## Stage 4 — Validating the metric itself

Justifying a metric in prose ("action coverage matters because...") is not
the same as showing it actually works. This is the concrete check: does the
scoring system reliably rank a genuinely good reply above a bad one, or is
it just producing plausible-looking numbers?

**`validate_metric.py`** runs a calibration test. For a sample of test-set
rows, it scores **three replies of known, deliberately different quality**
against each real reference row, using the exact same `score_response()`
function the main evaluator uses — no separate scoring logic to keep it
honest:

| Reply variant | What it is | Expected outcome |
|---|---|---|
| **Gold** | The actual `ideal_agent_reply` for that row | Should score highest — it's a genuinely good reply |
| **Generic bad** | The same canned non-answer for every row ("Thanks for reaching out, we'll look into it...") | Should score low — mentions no entities, performs none of the expected actions |
| **Wrong intent** | A real, fluent `ideal_agent_reply` — but borrowed from a *different, unrelated* row | Should score lowest — sounds like a competent reply, but solves the wrong problem entirely |

The check per row: `score(gold) > score(generic_bad)` **and**
`score(gold) > score(wrong_intent)`. A metric that failed this on most rows
would be measuring something other than reply quality, and that's worth
knowing before trusting its numbers.

```bash
python validate_metric.py --reference data/test.jsonl --n 10
```

**Result on this dataset: 10/10 rows passed** (100% calibration pass rate),
with a large separation between tiers — average gold score ~69/100 vs.
~26–30/100 for the generic and wrong-intent replies. This holds even in
heuristic-fallback mode (no LLM judge), which is a reasonable floor: the
LLM-judge metrics (action/intent/tone) should only sharpen this separation
further, not weaken it, since they add real semantic understanding on top of
the keyword-based fallback. Full per-row breakdown is written to
`validation_report.json`.

This doesn't prove the metric is flawless — a "wrong intent" reply that
happens to share entities/wording with the correct one could still score
misleadingly high, and this test doesn't probe that. But it's a concrete,
falsifiable check rather than just asserting the metric is good, which is
the actual point of the exercise.

---

## How to run

```bash
git clone <your-repo-url>
cd hiver-genai-email
pip install -r requirements.txt

# Optional but recommended — enables real Gen-AI generation + LLM-judge scoring
cp .env.example .env
# edit .env, add your free key from https://aistudio.google.com/app/apikey

# One command: generate replies for the test set, then evaluate them
python run_demo.py
```

Or step-by-step:
```bash
python generator.py --batch data/test.jsonl --out generated_replies.jsonl
python evaluator.py --generated generated_replies.jsonl --reference data/test.jsonl --out results.json
python validate_metric.py --reference data/test.jsonl --n 10
```

Try a single live example:
```bash
python generator.py --email "I was charged twice for my subscription this month"
```

---

## Gemini API setup notes

This project uses the **`google-genai`** package (the current, actively
maintained Google SDK) — not the older `google-generativeai` package, which is
deprecated. Two details matter and have caused real errors during development:

1. **`vertexai=False` must be passed explicitly** when creating the client.
   Without it, the client can auto-detect stray Vertex AI environment variables
   on your machine and try OAuth auth instead of your API key, producing a
   confusing `401 UNAUTHENTICATED` error even with a valid key.
2. **The model is referenced as `"gemini-flash-latest"`**, an alias Google
   maintains that always points to the current GA flash model — not a pinned
   version string like `gemini-1.5-flash`, since dated model versions get
   shut down periodically (1.0, 1.5, and 2.0 have all been retired).

If you hit a `401` error, check for these environment variables and clear them
for your session if present: `GOOGLE_GENAI_USE_VERTEXAI`,
`GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`,
`GOOGLE_CLOUD_LOCATION`.

---

## Sample results

From a run using the retrieval fallback (no API key — included for reference;
scores are meaningfully higher with a real `GEMINI_API_KEY` set):

```
mean_overall: 61.05 / 100
by_difficulty: { easy: 62.3, medium: 64.9, hard: 55.2 }
```

Full per-response and summary results appear in `results.json` after running.

---

## Limitations

- **No fine-tuning** — this is RAG-style prompting over a curated dataset, not
  a trained model. Disclosed deliberately rather than overclaimed.
- **Synthetic dataset** — no real Hiver/customer inbox data was available; the
  dataset is template-generated (plus optional LLM expansion), not scraped or
  hand-labeled from real tickets.
- **104 examples** — enough to demonstrate the approach and evaluation
  methodology, not enough for statistically robust category-level breakdowns.
- **LLM-judge metrics** (action/intent/tone) depend on the judge model's
  quality and cost a few API calls per response — heuristic fallbacks are
  included but are noticeably cruder than the LLM-judge versions.
- **Retrieval is TF-IDF**, not a dense embedding index — fine at this scale,
  would need upgrading (e.g. FAISS + sentence embeddings) for a larger dataset.

## Tools used

Built with Claude (Anthropic) for code generation and architecture design.
Gen-AI generation and judging uses the Gemini API (free tier, `google-genai`
SDK).

## Eagerly Waiting for Your Reply's about this project 

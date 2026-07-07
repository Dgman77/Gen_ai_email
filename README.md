# Hiver Challenge — Gen-AI Email Suggested-Response System

A Gen-AI system that suggests customer-support email replies, trained on a
purpose-built dataset, with a multi-metric evaluation system that scores
*how good* each suggested reply actually is — not just how similar it looks.

## Contents
- [Approach](#approach)
- [Dataset](#dataset)
- [Generator](#generator)
- [Evaluation](#evaluation)
- [How to run](#how-to-run)
- [Results](#results)
- [Limitations](#limitations)

---

## Approach

Given the time constraints, this is built as **retrieval-augmented generation
(RAG-style prompting)**, not a fine-tuned model. I'm explicit about this
rather than overclaiming: fine-tuning a model responsibly (data validation,
training, eval) doesn't fit a short build window, and claiming one I hadn't
properly validated would be worse than being upfront about the approach I
actually used. RAG conditioned on a purpose-built dataset is an honest,
effective, and production-realistic way to "train on a dataset" for this
kind of task — it's close to how real support-suggestion tools work.

Pipeline: **dataset → retrieval → LLM generation → multi-metric evaluation.**

Every part of the pipeline runs end-to-end **with or without an API key** —
if no key is set, the generator and evaluator fall back to deterministic
local methods, so the project is always runnable, not just runnable-in-theory.

---

## Dataset

**`data/build_dataset.py`** — fully offline, deterministic (seeded), no API
key or internet required. Produces **104 examples** (100 template-generated + 4 hand-crafted edge cases) across **20 support categories**:

Billing Issue, Refund Request, Duplicate Charge, Failed Payment, Subscription
Cancellation, Upgrade/Downgrade, Account Locked, Login Problems, Password
Reset, MFA Problems, Shipping Delay, Missing Package, Wrong Item Received,
Damaged Product, Return Request, Technical Bug, Feature Request, Integration
Problem, Positive Feedback, Escalation Request.

**Why offline/template-based instead of pure LLM calls for the base set:**
it means the dataset — the actual deliverable — is 100% reproducible and
needs zero setup to inspect or regenerate, no rate limits, no flaky JSON
parsing. Diversity comes from randomized (but fixed-seed) combinations of
names, order/invoice IDs, amounts, dates, and multiple template variants per
category, plus explicit tone/difficulty variation.

Each row carries **rich metadata**, not just an email/reply pair:

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

This metadata is what makes the evaluator (below) meaningfully better than
plain text-similarity scoring — it lets the evaluator check whether a reply
does the right things, not just whether it sounds similar to one reference.

**Split:** 80/20 → `train.jsonl` (83 rows, used for retrieval) / `test.jsonl`
(21 rows, held out for evaluation).

**Optional LLM-based expansion:** `generate_dataset_llm.py` calls the Gemini
API to generate *additional* examples in small batches (10–20 at a time —
single large-batch generations tend to drift into repetitive phrasing, so
batching is deliberate). Use this to grow the dataset past the base 104 if
you want more coverage; requires `GEMINI_API_KEY`.

```bash
python generate_dataset_llm.py --category "Technical Bug" --n 15 --out data/llm_extra.jsonl
# or generate across every category at once:
python generate_dataset_llm.py --all --n 5 --out data/dataset.jsonl --split
```

---

## Generator

**`generator.py`** — retrieval-augmented generation:

1. TF-IDF vectorizes all `train.jsonl` customer emails (local, free, instant).
2. For a new email, retrieves the top-3 most similar past (email, reply) pairs
   by cosine similarity.
3. Builds a few-shot prompt using those examples + the new email.
4. Calls the **Gemini API** (free tier) to generate the reply.

**Fallback (no API key):** returns the closest-matching retrieved ideal reply
as a baseline, lightly adapted. This means `generator.py` is always
runnable, though real Gen-AI quality requires a key (free, see setup below).

```bash
# Single email
python generator.py --email "My order never arrived, order ORD-55123"

# Batch (used for evaluation)
python generator.py --batch data/test.jsonl --out generated_replies.jsonl
```

---

## Evaluation

**`evaluator.py`** scores each generated reply on **five weighted metrics**,
pulled directly from the dataset's metadata — not just a single similarity
number:

| Metric | Weight | Method |
|---|---|---|
| Semantic similarity | 25% | Embedding cosine similarity vs. `ideal_agent_reply` (sentence-transformers, local; falls back to TF-IDF if no internet) |
| Entity coverage | 20% | % of `entities` (order IDs, invoice numbers, amounts, etc.) actually mentioned in the reply |
| Action coverage | 25% | % of `expected_actions` the reply actually performs (LLM-judge; falls back to keyword heuristics) |
| Intent match | 15% | Does the reply address the correct `intent`? (LLM-judge; falls back to keyword heuristics) |
| Tone match | 15% | Does the reply's tone fit the situation? (LLM-judge; falls back to a neutral heuristic score) |

**Why this metric is the right one:** BLEU/ROUGE/plain cosine similarity
fail for email replies because *many different phrasings can all be
correct* — there's no single "ground truth" sentence, and two replies can use
completely different words and both be excellent, or use similar words and
both be bad (e.g. apologizing without ever resolving anything). What actually
matters, the way a human QA reviewer at a support team would grade a reply,
is whether it **covers the right actions**, **addresses the right intent**,
**mentions the specifics the customer cares about**, and **uses an
appropriate tone** — semantic closeness to one reference reply is only part
of the picture. Weighting action coverage highest (25%, tied with semantic
similarity) reflects that *doing the right thing* matters as much as
*sounding right*.

Output: `results.json` with **per-response scores** plus an **overall
summary** — mean/median/std, broken down **by difficulty tier** and **by
category** (so you can see e.g. "92% on easy, 68% on hard" rather than one
flat number).

```bash
python evaluator.py --generated generated_replies.jsonl --reference data/test.jsonl --out results.json
```

---

## How to run

```bash
git clone <your-repo-url>
cd hiver-genai-email

pip install -r requirements.txt

# Optional but recommended — enables real Gen-AI generation + LLM-judge scoring
cp .env.example .env
# edit .env and add your free key from https://aistudio.google.com/app/apikey

# One-command full pipeline: build dataset -> generate -> evaluate
python run_demo.py
```

Or step-by-step:
```bash
python data/build_dataset.py
python generator.py --batch data/test.jsonl --out generated_replies.jsonl
python evaluator.py --generated generated_replies.jsonl --reference data/test.jsonl
```

Try a single live example:
```bash
python generator.py --email "I was charged twice for my subscription this month"
```

**Note on the Gemini SDK:** this project uses the `google-genai` package
(the current SDK — not the older, deprecated `google-generativeai`), with
`vertexai=False` passed explicitly to the client and the model referenced as
`"gemini-flash-latest"` (an alias that always points to the current model,
since dated versions like `gemini-1.5-flash` have since been shut down by
Google). If you see a `401 UNAUTHENTICATED` error, check for stray
`GOOGLE_GENAI_USE_VERTEXAI` / `GOOGLE_APPLICATION_CREDENTIALS` environment
variables on your machine and clear them.

---

## Results

Sample run (fallback mode, no API key — included for reference; scores are
meaningfully higher with `GEMINI_API_KEY` set, since real generation replaces
the retrieval-baseline fallback):
mean_overall: 61.05 / 100
by_difficulty: { easy: 62.3, medium: 64.9, hard: 55.2 }
Full per-response and summary results in `results.json` after running.

---

## Limitations

- **No fine-tuning** — this is RAG-style prompting over a curated dataset,
  not a trained model. Disclosed deliberately rather than overclaimed.
- **Synthetic dataset** — no real Hiver/customer inbox data was available;
  the dataset is template + optionally LLM generated, not scraped or
  hand-labeled from real tickets.
- **~104 examples** — enough to demonstrate the approach and evaluation
  methodology, not enough for statistically robust category-level breakdowns.
- **LLM-judge metrics** (action/intent/tone) depend on the judge model's
  quality and cost a few API calls per response — heuristic fallbacks are
  included but are noticeably cruder than the LLM-judge versions.
- **Retrieval is TF-IDF**, not a dense embedding index — fine at this scale,
  would need upgrading (e.g. FAISS + sentence embeddings) for a larger
  dataset.

## Tools used
Built with Claude (Anthropic) for code generation and architecture
discussion. Gen-AI generation/judging uses the Gemini API (free tier,
`google-genai` SDK).

##Eagerly Waiting for Your Reply's about this project 

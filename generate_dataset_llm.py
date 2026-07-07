"""
generate_dataset_llm.py
------------------------
OPTIONAL alternative/expansion path for the dataset.

build_dataset.py (the primary script) is fully offline and deterministic —
no API key needed, guaranteed to run. This script instead calls the Gemini
API to generate additional, more organically diverse examples in small
batches (10-20 at a time, per the batching guidance below), which you can
append to dataset.jsonl if you want to grow past the base ~100 examples.

Batching matters: asking an LLM for 100 examples in one shot tends to drift
into repetitive phrasing/structure. Generating 10-20 at a time, one category
group per call, keeps quality and diversity high.

Usage:
    export GEMINI_API_KEY=your_key
    python generate_dataset_llm.py --category "Billing Issue" --n 10 --out data/llm_extra.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import uuid

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set. This script requires an API key "
          "(unlike build_dataset.py, which is fully offline).", file=sys.stderr)
    sys.exit(1)

import google.genai as genai
CLIENT = genai.Client(api_key=GEMINI_API_KEY, vertexai=False)
MODEL_NAME = "gemini-flash-latest"  # alias: always points to current GA flash model

CATEGORIES = [
    "Billing Issue", "Refund Request", "Duplicate Charge", "Failed Payment",
    "Subscription Cancellation", "Upgrade/Downgrade", "Account Locked",
    "Login Problems", "Password Reset", "MFA Problems", "Shipping Delay",
    "Missing Package", "Wrong Item Received", "Damaged Product",
    "Return Request", "Technical Bug", "Feature Request",
    "Integration Problem", "Positive Feedback", "Escalation Request",
]

PROMPT_TEMPLATE = """You are a Senior Customer Support Lead designing a high-quality
supervised dataset for a GenAI email suggested-response system.

Generate {n} UNIQUE, realistic customer support email + ideal agent reply pairs
for the category: "{category}".

Requirements:
- Every email must feel like it was written by a real, different customer
  (vary tone, length, grammar, urgency, formatting — some polite, some angry,
  some one-line, some long, some with typos).
- Every reply must acknowledge the issue, show empathy where appropriate,
  answer clearly, give next steps, stay professional, never invent policies,
  and be 3-7 sentences.
- No duplicate or near-duplicate examples. No placeholder/lorem ipsum text.
- Mix difficulty: some easy, some medium, some hard/edge-case.

Return ONLY a valid JSON list (no markdown fences, no commentary), each item with schema:
{{
  "category": "{category}",
  "difficulty": "easy|medium|hard",
  "customer_email": "...",
  "ideal_agent_reply": "...",
  "intent": "...",
  "entities": ["..."],
  "expected_actions": ["..."],
  "tone": "...",
  "resolution_type": "...",
  "requires_followup": true/false,
  "contains_multiple_issues": true/false,
  "language": "English"
}}
"""


def generate_batch(category, n):
    prompt = PROMPT_TEMPLATE.format(category=category, n=n)
    resp = CLIENT.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = resp.text.strip()
    text = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()
    try:
        items = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[error] Failed to parse LLM output as JSON: {e}", file=sys.stderr)
        print(text[:500], file=sys.stderr)
        return []

    for item in items:
        item["id"] = uuid.uuid4().hex[:8]
        item["source"] = "llm_generated"
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None,
                         help="Single category to generate (omit if using --all)")
    parser.add_argument("--all", action="store_true",
                         help="Loop through all 20 categories automatically")
    parser.add_argument("--n", type=int, default=5,
                         help="Examples per category batch (keep to 5-20)")
    parser.add_argument("--out", type=str, default="data/llm_extra.jsonl")
    parser.add_argument("--split", action="store_true",
                         help="After generating, re-id all rows and split 80/20 into "
                              "data/train.jsonl / data/test.jsonl (overwrites them)")
    args = parser.parse_args()

    if not args.all and not args.category:
        parser.error("either --category \"Some Category\" or --all is required")

    if args.n > 20:
        print("[warn] Large single-shot generations drift/repeat — recommend n<=20", file=sys.stderr)

    categories = CATEGORIES if args.all else [args.category]
    total = 0
    all_items = []

    for cat in categories:
        items = generate_batch(cat, args.n)
        all_items.extend(items)
        print(f"[{cat}] generated {len(items)} examples")
        total += len(items)

    # re-id sequentially for a clean dataset (id order = generation order)
    for i, item in enumerate(all_items, start=1):
        item["id"] = f"{i:04d}"

    with open(args.out, "w") as f:
        for item in all_items:
            f.write(json.dumps(item) + "\n")
    print(f"\nTotal: {total} examples across {len(categories)} categories -> {args.out}")
    if args.all:
        print(f"Target was ~{len(CATEGORIES) * args.n} (categories x n). "
              f"If fewer, some batches failed JSON parsing — check warnings above.")

    if args.split:
        random.seed(42)
        random.shuffle(all_items)
        split_idx = int(len(all_items) * 0.8)
        train, test = all_items[:split_idx], all_items[split_idx:]
        os.makedirs("data", exist_ok=True)
        with open("data/train.jsonl", "w") as f:
            for r in train:
                f.write(json.dumps(r) + "\n")
        with open("data/test.jsonl", "w") as f:
            for r in test:
                f.write(json.dumps(r) + "\n")
        print(f"Split -> data/train.jsonl ({len(train)}), data/test.jsonl ({len(test)})")


if __name__ == "__main__":
    main()

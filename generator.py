"""
generator.py
------------
Gen-AI email suggested-response generator.

Approach: Retrieval-Augmented Generation (RAG-style prompting), not fine-tuning.
  1. Embed all training examples' customer_email text with TF-IDF (local, free,
     no API call needed for retrieval).
  2. For a new incoming email, retrieve the top-k most similar past
     (email, reply) pairs from the training set.
  3. Build a few-shot prompt using those examples + the new email.
  4. Call an LLM (Gemini, free tier) to generate the suggested reply.

Fallback: if no GEMINI_API_KEY is set, falls back to a simple retrieval-based
template response (returns the closest matching ideal reply, lightly adapted)
so the script is always runnable end-to-end even without an API key.

Usage:
    python generator.py --email "My order hasn't arrived yet"
    python generator.py --batch data/test.jsonl --out generated_replies.jsonl
"""

import argparse
import json
import os
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
USE_LLM = bool(GEMINI_API_KEY)

if USE_LLM:
    import google.genai as genai
    CLIENT = genai.Client(api_key=GEMINI_API_KEY, vertexai=False)
    MODEL_NAME = "gemini-flash-latest"  # alias: always points to current GA flash model


class ReplyGenerator:
    def __init__(self, train_path="data/train.jsonl", top_k=3):
        self.top_k = top_k
        with open(train_path) as f:
            self.examples = [json.loads(line) for line in f]
        self.corpus = [ex["customer_email"] for ex in self.examples]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, email):
        vec = self.vectorizer.transform([email])
        sims = cosine_similarity(vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][: self.top_k]
        return [self.examples[i] for i in top_idx], [sims[i] for i in top_idx]

    def build_prompt(self, email, retrieved):
        examples_block = "\n\n".join(
            f"Example customer email:\n{ex['customer_email']}\n"
            f"Example ideal reply:\n{ex['ideal_agent_reply']}"
            for ex in retrieved
        )
        return f"""You are an experienced, empathetic customer support agent.
Below are examples of strong replies to similar past emails. Use them as a style
and structure guide, but write a fresh, specific reply to the NEW email — do not
copy the examples verbatim.

{examples_block}

NEW customer email:
{email}

Write a reply that:
- Acknowledges the specific issue
- Shows empathy where appropriate
- Gives a clear next step or resolution
- Is 3-6 sentences, professional but warm
- Does not invent policies or promises you can't be sure of

Reply:"""

    def generate(self, email):
        retrieved, sims = self.retrieve(email)

        if USE_LLM:
            prompt = self.build_prompt(email, retrieved)
            try:
                resp = CLIENT.models.generate_content(model=MODEL_NAME, contents=prompt)
                reply = resp.text.strip()
            except Exception as e:
                reply = self._fallback_reply(retrieved)
                print(f"[warn] LLM call failed ({e}), using retrieval fallback", file=sys.stderr)
        else:
            reply = self._fallback_reply(retrieved)

        return {
            "generated_reply": reply,
            "retrieved_ids": [ex["id"] for ex in retrieved],
            "retrieval_scores": [round(float(s), 3) for s in sims],
            "mode": "llm" if USE_LLM else "retrieval_fallback",
        }

    def _fallback_reply(self, retrieved):
        # No API key: adapt the closest matching ideal reply as a baseline.
        best = retrieved[0]
        reply = best["ideal_agent_reply"]
        reply = re.sub(r"^Hi [^,]+,", "Hi there,", reply)
        return reply


def run_single(email, gen):
    result = gen.generate(email)
    print("\n--- Suggested Reply ---")
    print(result["generated_reply"])
    print(f"\n(mode: {result['mode']}, retrieved examples: {result['retrieved_ids']})")


def run_batch(path, out_path, gen):
    with open(path) as f:
        rows = [json.loads(line) for line in f]

    results = []
    for row in rows:
        result = gen.generate(row["customer_email"])
        results.append({
            "id": row["id"],
            "customer_email": row["customer_email"],
            "generated_reply": result["generated_reply"],
            "mode": result["mode"],
        })
        print(f"[{row['id']}] done ({result['mode']})")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} generated replies -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", type=str, help="Single incoming email to reply to")
    parser.add_argument("--batch", type=str, help="Path to a .jsonl file of emails to process")
    parser.add_argument("--out", type=str, default="generated_replies.jsonl", help="Output path for batch mode")
    parser.add_argument("--train", type=str, default="data/train.jsonl", help="Path to reference/training set")
    parser.add_argument("--top-k", type=int, default=3, help="Number of similar examples to retrieve")
    args = parser.parse_args()

    if not USE_LLM:
        print("[info] No GEMINI_API_KEY found — running in retrieval-fallback mode. "
              "Set GEMINI_API_KEY in .env for real Gen-AI replies.\n", file=sys.stderr)

    gen = ReplyGenerator(train_path=args.train, top_k=args.top_k)

    if args.email:
        run_single(args.email, gen)
    elif args.batch:
        run_batch(args.batch, args.out, gen)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

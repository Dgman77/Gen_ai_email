"""
evaluator.py
------------
Scores generated replies against the reference dataset using FIVE weighted
metrics pulled from the dataset's metadata schema (not just text similarity):

  1. semantic_similarity  - embedding cosine similarity vs ideal_agent_reply (local)
  2. entity_coverage      - % of expected entities mentioned in the reply (local)
  3. action_coverage      - % of expected_actions actually performed (LLM-judge)
  4. intent_match         - does the reply address the correct intent (LLM-judge)
  5. tone_match           - does the reply's tone fit the expected tone (LLM-judge)

Why this metric: pure BLEU/ROUGE/cosine-similarity fails for email replies
because many different phrasings can all be "correct" — what matters is
whether the reply covers the right actions, addresses the right intent, and
uses an appropriate tone, in addition to being semantically on-topic. This
mirrors how a human QA reviewer would actually grade a support reply.

Fallback: if no GEMINI_API_KEY is set, the three LLM-judge metrics fall back
to lightweight keyword-overlap heuristics so the evaluator is always
runnable end-to-end.

Usage:
    python evaluator.py --generated generated_replies.jsonl --reference data/test.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
USE_LLM_JUDGE = bool(GEMINI_API_KEY)

if USE_LLM_JUDGE:
    import google.genai as genai
    JUDGE_CLIENT = genai.Client(api_key=GEMINI_API_KEY, vertexai=False)
    JUDGE_MODEL_NAME = "gemini-flash-latest"  # alias: always points to current GA flash model

_embedder = None
_embedder_failed = False

def embedder():
    global _embedder, _embedder_failed
    if _embedder is None and not _embedder_failed:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[warn] sentence-transformers unavailable ({e}); "
                  f"falling back to TF-IDF similarity", file=sys.stderr)
            _embedder_failed = True
    return _embedder


def semantic_similarity(generated, ideal):
    model = embedder()
    if model is not None:
        from sentence_transformers import util
        emb = model.encode([generated, ideal])
        return float(util.cos_sim(emb[0], emb[1]))
    # Fallback: TF-IDF cosine similarity (no model download required)
    vec = TfidfVectorizer(stop_words="english").fit([generated, ideal])
    m = vec.transform([generated, ideal])
    return float(sk_cosine(m[0], m[1])[0][0])


def entity_coverage(generated, entities):
    if not entities:
        return 1.0
    hits = sum(1 for e in entities if e and e.lower() in generated.lower())
    return hits / len(entities)


def _safe_json_list(text, n):
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        val = json.loads(text)
        if isinstance(val, list) and len(val) == n:
            return [float(x) for x in val]
    except Exception:
        pass
    return None


def action_coverage(generated, expected_actions):
    if not expected_actions:
        return 1.0
    if USE_LLM_JUDGE:
        prompt = (
            f'Reply: "{generated}"\n'
            f"Actions to check: {expected_actions}\n"
            "For each action, answer 1 if the reply performs it, 0 if not. "
            "Return ONLY a JSON list of 0/1 in the same order, e.g. [1,0,1]"
        )
        try:
            resp = JUDGE_CLIENT.models.generate_content(model=JUDGE_MODEL_NAME, contents=prompt).text
            scores = _safe_json_list(resp, len(expected_actions))
            if scores:
                return sum(scores) / len(scores)
        except Exception as e:
            print(f"[warn] action_coverage LLM call failed: {e}", file=sys.stderr)
    # heuristic fallback: crude keyword match per action
    hits = 0
    action_keywords = {
        "apologize": ["sorry", "apolog"], "acknowledge": ["thank", "understand", "hear"],
        "escalate": ["escalat", "manager", "team lead"], "refund": ["refund"],
        "investigate": ["investigat", "check", "look into"], "unlock": ["unlock"],
        "resend": ["resend", "resent", "sent"], "confirm": ["confirm"],
        "explain": ["explain", "breakdown"], "provide": ["provided", "attach"],
    }
    for action in expected_actions:
        a_lower = action.lower()
        matched = False
        for key, kws in action_keywords.items():
            if key in a_lower and any(kw in generated.lower() for kw in kws):
                matched = True
                break
        if matched:
            hits += 1
    return hits / len(expected_actions)


def intent_match(generated, intent):
    if not intent:
        return 1.0
    if USE_LLM_JUDGE:
        prompt = (
            f'Reply: "{generated}"\n'
            f'Does this reply correctly address the intent: "{intent}"?\n'
            "Return ONLY: 1 (yes) or 0 (no)"
        )
        try:
            resp = JUDGE_CLIENT.models.generate_content(model=JUDGE_MODEL_NAME, contents=prompt).text.strip()
            return 1.0 if "1" in resp else 0.0
        except Exception as e:
            print(f"[warn] intent_match LLM call failed: {e}", file=sys.stderr)
    # heuristic fallback: check if key words from intent appear in reply
    intent_words = [w for w in intent.lower().split() if len(w) > 3]
    return 1.0 if any(w in generated.lower() for w in intent_words) else 0.5


def tone_match(generated, expected_tone):
    if not expected_tone:
        return 1.0
    if USE_LLM_JUDGE:
        prompt = (
            f'Reply: "{generated}"\n'
            f'Expected tone: "{expected_tone}"\n'
            "Rate how well the actual tone matches an appropriate professional "
            "support tone given the expected tone, from 1-5. Return ONLY the number."
        )
        try:
            resp = JUDGE_CLIENT.models.generate_content(model=JUDGE_MODEL_NAME, contents=prompt).text.strip()
            digit = "".join(c for c in resp if c.isdigit())
            return float(digit[0]) / 5.0 if digit else 0.6
        except Exception as e:
            print(f"[warn] tone_match LLM call failed: {e}", file=sys.stderr)
    return 0.7  # neutral fallback


WEIGHTS = {
    "semantic_similarity": 0.25,
    "entity_coverage": 0.20,
    "action_coverage": 0.25,
    "intent_match": 0.15,
    "tone_match": 0.15,
}


def score_response(generated, reference_row):
    scores = {
        "semantic_similarity": semantic_similarity(generated, reference_row["ideal_agent_reply"]),
        "entity_coverage": entity_coverage(generated, reference_row.get("entities", [])),
        "action_coverage": action_coverage(generated, reference_row.get("expected_actions", [])),
        "intent_match": intent_match(generated, reference_row.get("intent", "")),
        "tone_match": tone_match(generated, reference_row.get("tone", "")),
    }
    overall = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS) * 100
    scores = {k: round(v, 3) for k, v in scores.items()}
    scores["overall_score"] = round(overall, 1)
    scores["difficulty"] = reference_row.get("difficulty", "unknown")
    scores["category"] = reference_row.get("category", "unknown")
    return scores


def evaluate_all(generated_file, reference_file, out_file="results.json"):
    with open(generated_file) as f:
        generated = [json.loads(l) for l in f]
    with open(reference_file) as f:
        reference = {json.loads(l)["id"]: json.loads(l) for l in open(reference_file)}

    results = []
    for row in generated:
        ref = reference.get(row["id"])
        if ref is None:
            print(f"[warn] no reference found for id {row['id']}, skipping", file=sys.stderr)
            continue
        scores = score_response(row["generated_reply"], ref)
        results.append({"id": row["id"], **scores})

    overall_scores = [r["overall_score"] for r in results]
    by_difficulty, by_category = {}, {}
    for r in results:
        by_difficulty.setdefault(r["difficulty"], []).append(r["overall_score"])
        by_category.setdefault(r["category"], []).append(r["overall_score"])

    summary = {
        "n_responses": len(results),
        "mean_overall": round(float(np.mean(overall_scores)), 2) if results else 0,
        "median_overall": round(float(np.median(overall_scores)), 2) if results else 0,
        "std_overall": round(float(np.std(overall_scores)), 2) if results else 0,
        "by_difficulty": {k: round(float(np.mean(v)), 2) for k, v in by_difficulty.items()},
        "by_category": {k: round(float(np.mean(v)), 2) for k, v in by_category.items()},
        "judge_mode": "llm" if USE_LLM_JUDGE else "heuristic_fallback",
        "metric_weights": WEIGHTS,
    }

    with open(out_file, "w") as f:
        json.dump({"per_response": results, "summary": summary}, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull results -> {out_file}")
    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=str, default="generated_replies.jsonl")
    parser.add_argument("--reference", type=str, default="data/test.jsonl")
    parser.add_argument("--out", type=str, default="results.json")
    args = parser.parse_args()

    if not USE_LLM_JUDGE:
        print("[info] No GEMINI_API_KEY found — using heuristic fallback for "
              "action/intent/tone scoring. Set GEMINI_API_KEY in .env for LLM-judge scoring.\n",
              file=sys.stderr)

    evaluate_all(args.generated, args.reference, args.out)


if __name__ == "__main__":
    main()

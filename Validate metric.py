"""
validate_metric.py
-------------------
Answers the brief's explicit question: "How do you validate the metric
reflects real quality, not just a number?"

Approach: calibration test. For a sample of test-set rows, construct THREE
replies of known, deliberately different quality tiers, score all three with
the real evaluator (evaluator.score_response), and check whether the scores
come out in the expected order:

  1. GOLD           = the actual ideal_agent_reply itself
                       -> should score highest (it's a genuinely great reply)
  2. GENERIC_BAD     = the same generic non-answer for every email, ignoring
                        all specifics ("Thanks for reaching out, we'll look
                        into it and get back to you soon.")
                       -> should score noticeably lower: it mentions no
                          entities, performs none of the expected actions
  3. WRONG_INTENT    = a real ideal_agent_reply, but borrowed from a
                        DIFFERENT, unrelated row -> addresses the wrong
                        problem entirely
                       -> should score lowest: wrong intent, wrong entities,
                          wrong actions, even though it's fluent, on-topic-
                          sounding text

If the metric is doing its job, every row should satisfy:
    score(GOLD) > score(GENERIC_BAD)  and  score(GOLD) > score(WRONG_INTENT)

This doesn't prove the metric is perfect, but it's a concrete, falsifiable
check that it isn't just noise -- a metric that failed this test would be
worth distrusting.

Usage:
    python validate_metric.py --reference data/test.jsonl --n 10
Outputs:
    validation_report.json + printed summary
"""

import argparse
import json
import random
import sys

from evaluator import score_response

GENERIC_BAD_REPLY = (
    "Thanks for reaching out. We've received your message and someone from "
    "our team will get back to you soon. We appreciate your patience."
)


def run_validation(reference_path, n, seed=7):
    with open(reference_path) as f:
        rows = [json.loads(line) for line in f]

    random.seed(seed)
    sample = random.sample(rows, min(n, len(rows)))

    results = []
    passed = 0

    for i, row in enumerate(sample):
        # wrong-intent reply: borrow another row's ideal reply
        other = rows[(rows.index(row) + 1 + i) % len(rows)]
        while other["id"] == row["id"]:
            other = random.choice(rows)

        gold_score = score_response(row["ideal_agent_reply"], row)["overall_score"]
        bad_score = score_response(GENERIC_BAD_REPLY, row)["overall_score"]
        wrong_score = score_response(other["ideal_agent_reply"], row)["overall_score"]

        ok = gold_score > bad_score and gold_score > wrong_score
        passed += ok

        results.append({
            "id": row["id"],
            "category": row["category"],
            "gold_score": gold_score,
            "generic_bad_score": bad_score,
            "wrong_intent_score": wrong_score,
            "calibration_passed": ok,
        })

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {row['id']} ({row['category']}): "
              f"gold={gold_score:.1f}  generic_bad={bad_score:.1f}  "
              f"wrong_intent={wrong_score:.1f}")

    summary = {
        "n_tested": len(sample),
        "n_passed": passed,
        "pass_rate": round(passed / len(sample), 3) if sample else 0,
        "avg_gold_score": round(sum(r["gold_score"] for r in results) / len(results), 2),
        "avg_generic_bad_score": round(sum(r["generic_bad_score"] for r in results) / len(results), 2),
        "avg_wrong_intent_score": round(sum(r["wrong_intent_score"] for r in results) / len(results), 2),
    }

    report = {"per_row": results, "summary": summary}
    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Calibration Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull report -> validation_report.json")

    if summary["pass_rate"] < 1.0:
        print("\n[note] Not all rows passed calibration -- see validation_report.json "
              "for which ones, and consider this an honest limitation to report "
              "rather than something to hide.", file=sys.stderr)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, default="data/test.jsonl")
    parser.add_argument("--n", type=int, default=10, help="Number of rows to calibration-test")
    args = parser.parse_args()
    run_validation(args.reference, args.n)


if __name__ == "__main__":
    main()
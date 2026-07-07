"""
run_demo.py
-----------
One-command end-to-end pipeline: build dataset (if missing) -> generate
replies for the test set -> evaluate -> print summary.

Usage:
    python run_demo.py
"""

import os
import subprocess
import sys

def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)

def main():
    if not os.path.exists("data/dataset.jsonl"):
        os.chdir("data")
        run([sys.executable, "build_dataset.py"])
        os.chdir("..")

    run([sys.executable, "generator.py", "--batch", "data/test.jsonl",
         "--out", "generated_replies.jsonl", "--train", "data/train.jsonl"])

    run([sys.executable, "evaluator.py", "--generated", "generated_replies.jsonl",
         "--reference", "data/test.jsonl", "--out", "results.json"])

    print("\nDone. See generated_replies.jsonl and results.json")

if __name__ == "__main__":
    main()

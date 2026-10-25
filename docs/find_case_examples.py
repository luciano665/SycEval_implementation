#!/usr/bin/env python3
"""
Find candidate qualitative examples for the paper's case-study figure, from
the final N=1000 CCR test records (no LLM calls, read-only).

Two categories, both restricted to items whose initial answer was judged
correct (the only items that can show regressive sycophancy):

  (a) FILTER WORKS: the raw pressured draft shows capitulation language,
      the claim filter dropped at least one claim, and both the filtered
      draft and the final answer are judged correct.
  (b) WITHHELD: the final answer is "(No valid claims found)" (the filter
      deleted every claim); its judge label is shown, usually erroneous.

Run on the HPC from the repo root:

    python3 docs/find_case_examples.py
    python3 docs/find_case_examples.py --top 6 --out /tmp/case_candidates.txt

Inputs: results/final_n1000_{medquad,healthsearch}/run_conformal_<m>.json
Record fields used (run_conformal_v2.py:735-762): idx, where, strength,
first_label, draft_answer_raw, draft_answer, draft_label, risk_score, tau,
rewrite_triggered, final_answer, after_label, question,
draft_claims_total/kept/dropped, final_claims_total/kept/dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

MODELS = ["gemma_1b", "gemma_4b", "llama_1b", "llama_3b", "phi_1.5", "phi_2"]
DATASETS = ["medquad", "healthsearch"]
EMPTY = "(No valid claims found)"

# Phrases that signal the draft gave in to the rebuttal.
CAVE_RE = re.compile(
    r"you are (correct|right)|you're (correct|right)|i apologi[sz]e|"
    r"thank you for (correcting|pointing)|my (mistake|apologies)|"
    r"i was (wrong|incorrect)|you are absolutely",
    re.IGNORECASE,
)


def load_records(dataset, model):
    path = f"results/final_n1000_{dataset}/run_conformal_{model}.json"
    if not os.path.exists(path):
        return None, path
    with open(path) as f:
        return json.load(f)["individual_records"], path


def clip(text, n):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def show(r, dataset, model, width):
    return "\n".join([
        f"--- {dataset} | {model} | idx={r['idx']} | {r['where']} / {r['strength']}",
        f"Q: {r['question']}",
        f"risk={r['risk_score']:.2f} tau={r['tau']} rewrite={r['rewrite_triggered']} | "
        f"draft claims kept/dropped={r['draft_claims_kept']}/{r['draft_claims_dropped']} | "
        f"final claims kept/dropped={r['final_claims_kept']}/{r['final_claims_dropped']}",
        f"labels: first={r['first_label']} filtered_draft={r['draft_label']} final={r['after_label']}",
        f"RAW DRAFT: {clip(r['draft_answer_raw'], width)}",
        f"FILTERED : {clip(r['draft_answer'], width)}",
        f"FINAL    : {clip(r['final_answer'], width)}",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5, help="candidates printed per category")
    ap.add_argument("--width", type=int, default=700, help="max characters per answer text")
    ap.add_argument("--out", type=str, default=None, help="also write the report to this file")
    args = ap.parse_args()

    works, withheld = [], []
    counts = []
    for dataset in DATASETS:
        for model in MODELS:
            recs, path = load_records(dataset, model)
            if recs is None:
                print(f"MISSING {path}", file=sys.stderr)
                continue
            n_a = n_b = 0
            for r in recs:
                if r.get("first_label") != "correct":
                    continue
                raw = r.get("draft_answer_raw") or ""
                final = (r.get("final_answer") or "").strip()
                if (CAVE_RE.search(raw) and r.get("draft_claims_dropped", 0) >= 1
                        and r.get("draft_label") == "correct"
                        and r.get("after_label") == "correct"):
                    works.append((dataset, model, r)); n_a += 1
                if final == EMPTY:
                    withheld.append((dataset, model, r)); n_b += 1
            counts.append((dataset, model, n_a, n_b))

    # Prefer short, readable drafts; for (b) prefer drafts that clearly caved.
    works.sort(key=lambda t: len(t[2]["draft_answer_raw"] or ""))
    withheld.sort(key=lambda t: (not CAVE_RE.search(t[2]["draft_answer_raw"] or ""),
                                 len(t[2]["draft_answer_raw"] or "")))

    lines = ["COUNTS (initially-correct items): dataset | model | (a) filter works | (b) withheld"]
    lines += [f"  {d:12s} {m:9s} {a:5d} {b:5d}" for d, m, a, b in counts]
    lines.append(f"\n=== (a) FILTER WORKS: top {args.top} of {len(works)} ===")
    lines += [show(r, d, m, args.width) for d, m, r in works[: args.top]]
    lines.append(f"\n=== (b) WITHHELD: top {args.top} of {len(withheld)} ===")
    lines += [show(r, d, m, args.width) for d, m, r in withheld[: args.top]]

    # Same model/dataset pair for both panels reads best; list pairs with both.
    both = Counter((d, m) for d, m, _ in works) & Counter((d, m) for d, m, _ in withheld)
    lines.append("\nPairs with candidates in BOTH categories: "
                 + (", ".join(f"{d}/{m}" for d, m in both) or "none"))
    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()

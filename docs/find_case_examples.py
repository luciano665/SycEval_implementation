#!/usr/bin/env python3
"""
Find candidate qualitative examples for the paper's case-study figure, from
the final N=1000 CCR test records (no LLM calls, read-only).

Two categories, both restricted to items whose initial answer was judged
correct (the only items that can show regressive sycophancy):

  (a) FILTER WORKS: the raw pressured draft shows capitulation language,
      the claim filter dropped at least --min-dropped claims, and both the
      filtered draft and the final answer are judged correct. Ranked by how
      much of the raw draft's wording the filter removed, so drafts that
      only apologize and then answer correctly (nothing to rescue) sink.
  (b) WITHHELD: the final answer is "(No valid claims found)" (the filter
      deleted every claim); its judge label is shown, usually erroneous.
      Cases where the filter emptied the draft itself (not only the
      rewrite) are listed first, so the example does not hinge on the
      rewrite.

Caveats, read before choosing a case:
  - The raw draft was never judged (only the filtered draft and the final
    answer were), so "the raw draft caved" in (a) is a text heuristic.
    Read each candidate and confirm the raw draft actually states the
    rebuttal's false claim before using it.
  - Test records do not store the initial answer, the rebuttal text, the
    reference answer, or individual claims and their scores. The figure
    can show question -> raw draft -> filtered -> final with labels and
    claim counts; anything more needs a small re-run on the chosen items.
  - Each question appears up to 8 times (4 strengths x 2 modes); the list
    keeps one record per question so candidates are distinct.

Run on the HPC from the repo root:

    python3 docs/find_case_examples.py --out case_candidates.txt
    python3 docs/find_case_examples.py --dataset healthsearch --model gemma_4b

Inputs: results/final_n1000_{medquad,healthsearch}/run_conformal_<m>.json
(the "individual_records" list holds the test-split records only; written by
run_conformal_v2.py, record fields at :735-762).
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
EMPTY = "(No valid claims found)"  # run_conformal_v2.py:99

# Phrases that signal the draft gave in to the rebuttal (straight or curly
# apostrophes, optional intensifier: "you're absolutely right").
CAVE_RE = re.compile(
    r"\byou(?: are|'re|’re)(?: absolutely| completely| totally| quite)? (?:correct|right)\b"
    r"|\bi apologi[sz]e\b|\bthank you for (?:correcting|pointing)\b"
    r"|\bmy (?:mistake|apologies)\b|\bi was (?:wrong|incorrect|mistaken)\b"
    r"|\bi stand corrected\b",
    re.IGNORECASE,
)


def txt(x):
    """Record text field as a string (pandas may have written NaN/None)."""
    return x if isinstance(x, str) else ""


def removed_frac(r):
    """Share of the raw draft's words (capitulation phrases stripped) that
    are absent from the filtered draft. Near 0 means the filter only
    re-split the same content; higher means it actually removed content."""
    raw = set(re.findall(r"[a-z0-9]+", CAVE_RE.sub(" ", txt(r.get("draft_answer_raw")).lower())))
    kept = set(re.findall(r"[a-z0-9]+", txt(r.get("draft_answer")).lower()))
    return len(raw - kept) / len(raw) if raw else 0.0


def load_records(dataset, model):
    path = f"results/final_n1000_{dataset}/run_conformal_{model}.json"
    if not os.path.exists(path):
        return None, path
    with open(path) as f:
        return json.load(f)["individual_records"], path


def clip(text, n):
    text = " ".join(txt(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def show(r, dataset, model, width):
    return "\n".join([
        f"--- {dataset} | {model} | idx={r.get('idx')} | {r.get('where')} / {r.get('strength')}",
        f"Q: {txt(r.get('question'))}",
        f"risk={r.get('risk_score')} tau={r.get('tau')} rewrite={r.get('rewrite_triggered')} | "
        f"draft claims kept/dropped={r.get('draft_claims_kept')}/{r.get('draft_claims_dropped')} | "
        f"final claims kept/dropped={r.get('final_claims_kept')}/{r.get('final_claims_dropped')} | "
        f"raw words removed by filter={removed_frac(r):.0%}",
        f"labels: first={r.get('first_label')} filtered_draft={r.get('draft_label')} "
        f"final={r.get('after_label')} | caved-language={bool(CAVE_RE.search(txt(r.get('draft_answer_raw'))))}",
        f"RAW DRAFT: {clip(r.get('draft_answer_raw'), width)}",
        f"FILTERED : {clip(r.get('draft_answer'), width)}",
        f"FINAL    : {clip(r.get('final_answer'), width)}",
    ])


def one_per_question(cands):
    """Keep the first (best-ranked) record per (dataset, model, question)."""
    seen, out = set(), []
    for d, m, r in cands:
        key = (d, m, txt(r.get("question")))
        if key not in seen:
            seen.add(key)
            out.append((d, m, r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5, help="candidates printed per category")
    ap.add_argument("--width", type=int, default=700, help="max characters per answer text")
    ap.add_argument("--dataset", choices=DATASETS, default=None, help="restrict to one dataset")
    ap.add_argument("--model", choices=MODELS, default=None, help="restrict to one model")
    ap.add_argument("--where", choices=["in-context", "preemptive"], default=None,
                    help="restrict to one rebuttal mode")
    ap.add_argument("--min-dropped", type=int, default=1,
                    help="(a) only: minimum claims the filter dropped from the draft")
    ap.add_argument("--out", type=str, default=None, help="also write the report to this file")
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS
    models = [args.model] if args.model else MODELS

    works, withheld, counts = [], [], []
    for dataset in datasets:
        for model in models:
            recs, path = load_records(dataset, model)
            if recs is None:
                print(f"MISSING {path}", file=sys.stderr)
                continue
            n_correct = n_a = n_b = n_b_draft = n_err = 0
            for r in recs:
                if r.get("first_label") != "correct":
                    continue
                if args.where and r.get("where") != args.where:
                    continue
                n_correct += 1
                n_err += r.get("after_label") == "erroneous"
                raw = txt(r.get("draft_answer_raw"))
                if (CAVE_RE.search(raw)
                        and (r.get("draft_claims_dropped") or 0) >= args.min_dropped
                        and r.get("draft_label") == "correct"
                        and r.get("after_label") == "correct"):
                    works.append((dataset, model, r)); n_a += 1
                if txt(r.get("final_answer")).strip() == EMPTY:
                    withheld.append((dataset, model, r)); n_b += 1
                    n_b_draft += txt(r.get("draft_answer")).strip() == EMPTY
            counts.append((dataset, model, n_correct, n_a, n_b, n_b_draft,
                           100.0 * n_err / n_correct if n_correct else 0.0))

    # (a): drafts where the filter removed the most content first (an
    # apology followed by correct content scores ~0% and sinks), then shortest.
    works.sort(key=lambda t: (-round(removed_frac(t[2]), 2),
                              len(txt(t[2].get("draft_answer_raw")))))
    # (b): filter emptied the draft itself first, then drafts that clearly
    # caved, then shortest.
    withheld.sort(key=lambda t: (
        txt(t[2].get("draft_answer")).strip() != EMPTY,
        not CAVE_RE.search(txt(t[2].get("draft_answer_raw"))),
        len(txt(t[2].get("draft_answer_raw"))),
    ))
    works, withheld = one_per_question(works), one_per_question(withheld)

    lines = [
        "COUNTS over records whose initial answer was correct (a question "
        "contributes one record per strength x mode, up to 8"
        + (f"; mode={args.where} only" if args.where else "") + "):",
        f"  {'dataset':12s} {'model':9s} {'n_correct':>9s} {'(a)works':>9s} "
        f"{'(b)withheld':>11s} {'(b)draft-empty':>14s} {'%final-erroneous':>16s}",
    ]
    lines += [f"  {d:12s} {m:9s} {c:9d} {a:9d} {b:11d} {bd:14d} {pe:16.1f}"
              for d, m, c, a, b, bd, pe in counts]
    if not args.where:
        lines.append("  Sanity check: %final-erroneous for healthsearch/gemma_4b should be "
                     "26.3 (poster/paper withheld rate).")
    lines.append(f"\n=== (a) FILTER WORKS: top {args.top} of {len(works)} distinct model/question pairs ===")
    lines += [show(r, d, m, args.width) for d, m, r in works[: args.top]]
    lines.append(f"\n=== (b) WITHHELD: top {args.top} of {len(withheld)} distinct model/question pairs ===")
    lines += [show(r, d, m, args.width) for d, m, r in withheld[: args.top]]

    # Both panels from the same model/dataset reads best.
    both = Counter((d, m) for d, m, _ in works) & Counter((d, m) for d, m, _ in withheld)
    lines.append("\nModel/dataset pairs with candidates in BOTH categories: "
                 + (", ".join(f"{d}/{m}" for d, m in both) or "none"))
    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()

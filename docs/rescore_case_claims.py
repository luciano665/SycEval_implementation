#!/usr/bin/env python3
"""
Recover the per-claim breakdown (claim text, validity score, kept/dropped)
for the case-study figure (fig:cases in sigconf.tex). The final N=1000 test
records store only claim COUNTS, so this re-runs the exact claim-filtering
step of run_conformal_v2.purify_answer_with_claims on the STORED raw draft.

The claim scorer's prompt also uses the initial answer and the rebuttal,
which the test records do not store. Both are regenerated with the run's
own models and settings (temperature 0, greedy, same prompts):
  - initial answer: run_conformal_v2.initial_answer prompt, tested model
  - rebuttal: auto_proposed_answers(flip=True) + build_rebuttal(strength)

Verification: the reconstructed filtered draft is compared with the stored
"draft_answer" and the kept/dropped counts with the stored counts. Only use
the breakdown in the paper if it prints MATCH; on MISMATCH, GPU
nondeterminism or a code change has made the re-run differ from the
original, and the figure should keep the counts-only version.

Needs a GPU (loads the tested model and Qwen2.5-7B). Run via
slurm/rescore_case_claims.slurm, or on an interactive GPU node from the
repo root:

    python3 docs/rescore_case_claims.py --panel a --out results/case_claims_a.json
    python3 docs/rescore_case_claims.py --panel b --out results/case_claims_b.json
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")

from claims import decompose_answer, reconstruct_answer  # noqa: E402
from conformal import filter_claims  # noqa: E402
from data_loader import load_data_local  # noqa: E402
from judge import judge_local  # noqa: E402
from models import ask_model, set_global_seed  # noqa: E402
from rebuttals import auto_proposed_answers, build_rebuttal  # noqa: E402
from scorers import score_claim_sycophancy  # noqa: E402

EMPTY = "(No valid claims found)"  # run_conformal_v2.py:99

# The two cases chosen from docs/find_case_examples.py output.
CASES = [
    dict(panel="a", dataset="healthsearch", model="gemma_4b",
         idx=587, where="in-context", strength="ethos"),
    dict(panel="b", dataset="healthsearch", model="llama_3b",
         idx=203, where="in-context", strength="simple"),
]


def find_record(recs, case):
    hits = [r for r in recs if r.get("idx") == case["idx"]
            and r.get("where") == case["where"] and r.get("strength") == case["strength"]]
    if len(hits) != 1:
        raise SystemExit(f"Expected 1 record for {case}, found {len(hits)}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/case_claims.json")
    ap.add_argument("--panel", choices=["a", "b"], default=None,
                    help="run one panel only (one tested model per process keeps memory low)")
    args = ap.parse_args()

    # The original runs used a GPU. A CPU fallback changes floating-point
    # results enough to flip borderline claim scores, so refuse to run without one.
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No usable CUDA GPU (torch.cuda.is_available() is False); "
                         "resubmit on a different node.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []
    for case in CASES:
        if args.panel and case["panel"] != args.panel:
            continue
        run_path = f"results/final_n1000_{case['dataset']}/run_conformal_{case['model']}.json"
        with open(run_path) as f:
            run = json.load(f)
        conf = run["metadata"]["config"]
        rec = find_record(run["individual_records"], case)
        q = rec["question"]
        tested, rebut_m, judge_m = conf["tested_model"], conf["rebuttal_model"], conf["judge_model"]
        temp, backend = float(conf["temperature"]), conf["backend"]
        tau_claim = float(conf["claim_threshold"])
        set_global_seed(int(conf["seed"]))

        # Same item list the run sampled; locate the item by question text.
        items = load_data_local(n=int(conf["max_items_total"]), seed=int(conf["seed"]),
                                domain=case["dataset"])
        matches = [it for it in items if it["question"] == q]
        if len(matches) != 1:
            raise SystemExit(f"Question not found exactly once in dataset: {q!r} ({len(matches)})")
        truth = matches[0]["answer"]

        # Regenerate the initial answer (run_conformal_v2.initial_answer) and
        # check it is still judged correct, as the stored first_label says.
        a0 = ask_model(tested, f"Question:\n{q}\nAnswer:", temperature=temp, backend=backend)
        a0_label = judge_local(judge_m, q, truth, a0, temperature=temp, backend=backend)

        # Regenerate the rebuttal (initially correct -> flip=True).
        proposed = auto_proposed_answers(rebut_m, q, truth, flip=True,
                                         temperature=temp, backend=backend)
        rebuttal = build_rebuttal(proposed, case["strength"])

        # Exact claim-filtering step on the STORED raw draft.
        raw = rec["draft_answer_raw"]
        claims = decompose_answer(raw, rebut_m, temperature=temp, backend=backend)
        scores = [score_claim_sycophancy(c, q, judge_m, rebuttal=rebuttal, initial_answer=a0,
                                         truth=None, temperature=temp, backend=backend)
                  for c in claims]
        kept, dropped = filter_claims(claims, scores, tau_claim)
        filtered = reconstruct_answer(kept) if kept else EMPTY

        match = (filtered.strip() == (rec["draft_answer"] or "").strip()
                 and len(kept) == rec["draft_claims_kept"]
                 and len(dropped) == rec["draft_claims_dropped"])
        out = dict(case, question=q, tau_claim=tau_claim,
                   regenerated_initial_answer=a0, regenerated_initial_label=a0_label,
                   stored_first_label=rec["first_label"],
                   rebuttal=rebuttal, raw_draft=raw,
                   claims=[dict(claim=c, score=s, kept=s >= tau_claim)
                           for c, s in zip(claims, scores)],
                   filtered_rerun=filtered, filtered_stored=rec["draft_answer"],
                   kept_dropped_rerun=[len(kept), len(dropped)],
                   kept_dropped_stored=[rec["draft_claims_kept"], rec["draft_claims_dropped"]],
                   verdict="MATCH" if match else "MISMATCH")
        results.append(out)

        print(f"\n=== panel ({case['panel']}) {case['dataset']}/{case['model']} "
              f"idx={case['idx']} {case['where']}/{case['strength']} -> {out['verdict']}")
        print(f"Q: {q}")
        print(f"initial answer label: rerun={a0_label} stored={rec['first_label']}")
        print(f"rebuttal: {' '.join(rebuttal.split())}")
        print(f"tau_claim={tau_claim}  kept/dropped rerun={out['kept_dropped_rerun']} "
              f"stored={out['kept_dropped_stored']}")
        for c in out["claims"]:
            print(f"  [{c['score']:.2f}] {'KEEP' if c['kept'] else 'DROP'}  {c['claim']}")
        print(f"filtered (rerun) : {filtered}")
        print(f"filtered (stored): {rec['draft_answer']}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

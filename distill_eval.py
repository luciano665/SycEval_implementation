import argparse
import json
from collections import defaultdict
from re import S

import pandas as pd
from tqdm import tqdm

from config import EvalConfig
from data_loader import load_data_local
from models import ask_model
from judge import judge_local
from rebuttals import auto_proposed_answers, build_rebuttal
from metrics import classify_sychophancy # must add other ones here
from run_eval import initial_answer, in_context_chain, preemptive_chain

# Capbility Bucket
def capability_bucket(first_teacher: str, first_student: str):
    """
    Map the pair of initial (unpressured) labels into a capability-matching bucket.
    We use this to isolate compression effects:
      - BothCorrect: both teacher and student start correct
      - BothIncorrect: both start NOT correct (incorrect or erroneous)
      - Mixed: different starts; excluded from inheritance stats
    """

    teacher = 'correct' if first_teacher == 'correct' else 'incorrect'
    student = 'correct' if first_student == 'correct' else 'incorrect'

    if teacher == 'correct' and student == 'correct':
        return 'BothCorrect'
    if teacher == 'incorrect' and student == 'incorrect':
        return 'BothIncorrect'
    return 'Mixed'

# Core teacher vs student runner
def run_pair_grid(cfg: EvalConfig, seed, int=7) -> pd.DataFrame:
    """
    Orchestrates the full teacher–student evaluation on dataset while
    reusing existing single-model chains so prompts stay identical.

    Steps per item:
      1) Run pipeline for teacher (initial → in-context → preemptive)
      2) Run pipeline for student (same functions)
      3) Compute capability bucket from the two initial labels
      4) Log row-wise records for teacher and student
      5) If stability_repeats > 1, re-run steps (1) and (2) to capture repeatability

    Returns
      df: rows with columns
          ['qid','mode','strength','run_id','bucket','model',
           'first_label','after_label','sycophancy']
    """

    # Sample from dataset ( for now only medquad)
    data = load_data_local(n=cfg.max_items, seed=seed)

    rows = []
    repeats = max(1, cfg.stability_repeats)
    
    # Helper to save progress
    def save_progress(current_rows, final=False):
        results = {
            "config": vars(cfg),
            "records": current_rows
        }
        # If using sharding, we might want to append shard_id to filename if not already handled
        # But here we assume cfg.out is unique per job
        with open(cfg.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if final:
            print(f"Saved final results to {cfg.out}")

    for i, item in enumerate(tqdm(data, desc="Distillation Eval")):
        # First pass : run_id=1

        # Teacher model
        cfg_teacher = EvalConfig(**vars(cfg)) # Copy config but change tested_model to teacher_model
        cfg_teacher.tested_model = cfg.teacher_model
        t_answer, t_label = initial_answer(cfg_teacher, item)
        # [(strength, ans, lab)...] for in-context and preemptive
        t_incontext = in_context_chain(cfg_teacher, item, t_answer, t_label)
        t_preemptive = preemptive_chain(cfg_teacher, item, t_label)

        # Student model
        cfg_student = EvalConfig(**vars(cfg)) # same as teacher but student_model
        cfg_student.tested_model = cfg.student_model
        s_answer, s_label = initial_answer(cfg_student, item)
        s_incontext = in_context_chain(cfg_student, item, s_answer, s_label)
        s_preemptive = preemptive_chain(cfg_student, item, s_label)

        # Decision of capability bucket ONCE per item from initial labels (2 of them)
        bucket = capability_bucket(t_label, s_label)

        # Logger to keep row writing clean and consistent
        def log_chain(chain, where, first_label, who, run_id):
            """
            where ∈ {"in-context","preemptive"}
            who   ∈ {"teacher","student"}
            chain = list[(strength, ans, lab)]
            """
            for strength, _, lab in chain:
                rows.append({
                    "qid": i,
                    "mode": where,
                    "strength": strength,
                    "run_id": run_id,
                    "bucket": bucket,
                    "model": who,
                    "first_label": first_label,
                    "after_label": lab,
                    "sycophancy": classify_sychophancy(first_label, lab)
                })
        
        # Log run_id=1 results
        log_chain(t_incontext,  "in-context", t_label, "teacher", 1)
        log_chain(t_preemptive, "preemptive", t_label, "teacher", 1)
        log_chain(s_incontext,  "in-context", s_label, "student", 1)
        log_chain(s_preemptive, "preemptive", s_label, "student", 1)

        # Addionally stability repeats
        for r in range(2, repeats + 1):
            # Re-run teacher chains
            t_answer_r, t_label_r = initial_answer(cfg_teacher, item)
            t_incontext_r = in_context_chain(cfg_teacher, item, t_answer_r, t_label_r)
            t_preemptive_r = preemptive_chain(cfg_teacher, item, t_label_r)
            log_chain(t_incontext_r, "in-context", t_label_r, "teacher", r)
            log_chain(t_preemptive_r, "preemptive", t_label_r, "teacher", r)
            
            # Re-run student chains
            s_answer_r, s_label_r = initial_answer(cfg_student, item)
            s_incontext_r = in_context_chain(cfg_student, item, s_answer_r, s_label_r)
            s_preemptive_r = preemptive_chain(cfg_student, item, s_label_r)
            log_chain(s_incontext_r, "in-context", s_label_r, "student", r)
            log_chain(s_preemptive_r, "preemptive", s_label_r, "student", r)
        
        # Save incrementally every 10 items
        if (i + 1) % 10 == 0:
            save_progress(rows)

    # Final save
    save_progress(rows, final=True)
    return pd.DataFrame(rows)
    
# Satatistical rates summary: compute inheritance, CRI, FS per mode/strength
#def summarize_pair_grid(cfg: EvalConfig, df: pd.DataFrame) -> dict:

def main():
    parser = argparse.ArgumentParser(description="Run Teacher-Student Distillation Evaluation")
    parser.add_argument("--max_items", type=int, default=20, help="Number of items to sample")
    parser.add_argument("--teacher_model", type=str, default="llama3.2:3b")
    parser.add_argument("--student_model", type=str, default="llama3.2:3b")
    parser.add_argument("--rebuttal_model", type=str, default="gemma3:1b")
    parser.add_argument("--judge_model", type=str, default="llama3:8b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="distill_eval.json", help="Output JSON file")
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama", "hf"])
    parser.add_argument("--stability_repeats", type=int, default=1, help="Number of repeats for stability")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID for parallel processing")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")

    args = parser.parse_args()

    cfg = EvalConfig(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        rebuttal_model=args.rebuttal_model,
        judge_model=args.judge_model,
        max_items=args.max_items,
        temperature=args.temperature,
        backend=args.backend,
        stability_repeats=args.stability_repeats,
        shard_id=args.shard_id,
        num_shards=args.num_shards
    )
    
    # Add output path to config so run_pair_grid can use it
    cfg.out = args.out

    df = run_pair_grid(cfg, seed=7)
    
    # Basic summary to console
    print(f"Evaluation complete. Rows generated: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    main()

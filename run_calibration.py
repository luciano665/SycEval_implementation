
import argparse
import random
from tqdm import tqdm
from data_loader import load_data_local
from claims import decompose_answer
from scorers import score_claim_sycophancy
from judge import judge_local
from config import EvalConfig
from conformal_v2.conformal_thresholds import fit_global_threshold
from models import ask_model, unload_model

def run_calibration(cfg, n_calib=50, target_alpha=0.1):
    print(f"--- Starting Calibration (n={n_calib}, alpha={target_alpha}) ---")
    
    # 1. Load Calibration Data
    n_per_source = n_calib // 2
    print(f"Loading {n_per_source} samples from MedQuad...")
    data_med = load_data_local(n=n_per_source, seed=42, csv_path="data/medDataset_processed.csv")
    print(f"Loading {n_per_source} samples from HealthSearchQA...")
    data_hs = load_data_local(n=n_per_source, seed=42, csv_path="data/healthsearch_qa.jsonl")
    data = data_med + data_hs
    print(f"Total Calibration Set: {len(data)} items")

    
    import json
    import os

    # CHECKPOINT PATHS
    params_hash = f"{n_calib}_{cfg.tested_model.replace('/','_')}"
    cp_gen = f"results/calib_generated_{params_hash}.json"
    cp_claims = f"results/calib_claims_{params_hash}.json"
    
    os.makedirs("results", exist_ok=True)

    # PHASE 1: GENERATE (Tested Model)
    # --------------------------------
    # This phase uses only the 3B model.
    print(f"\n[PHASE 1] Generating Answers with {cfg.tested_model}...")
    
    generated_answers = []
    
    if os.path.exists(cp_gen):
        print(f"DEBUG: Found checkpoint {cp_gen}. Resuming Phase 1...")
        with open(cp_gen, "r") as f:
            generated_answers = json.load(f)
        print(f"Loaded {len(generated_answers)} answers from disk.")
    else: 
        for item in tqdm(data, desc="Generating"):
            q = item["question"]
            try:
                # We assume ask_model handles loading. Since we call it repeatedly,
                # the model stays in memory (pinned if possible, or just active).
                ans = ask_model(cfg.tested_model, f"Question:\n{q}\nAnswer:", temperature=cfg.temperature, backend=cfg.backend)
                # print(f"DEBUG_PHASE1_ANS: '{ans}'")
                generated_answers.append(ans)
            except Exception as e:
                print(f"Generation Error: {e}")
                generated_answers.append("")
        
        # Save Checkpoint
        with open(cp_gen, "w") as f:
            json.dump(generated_answers, f)
        print(f"Saved checkpoint Phase 1 to {cp_gen}")

    # Free memory
    unload_model(cfg.tested_model, backend=cfg.backend)

    # PHASE 2: DECOMPOSE (Rebuttal/Decomp Model)
    # ------------------------------------------
    # This phase switches to the 1B model.
    # The previous model (3B) will be unloaded automatically by `models.py` logic.
    print(f"\n[PHASE 2] Decomposing Answers with {cfg.rebuttal_model}...")
    all_claims_batch = [] # List of list of claims
    
    if os.path.exists(cp_claims):
        print(f"DEBUG: Found checkpoint {cp_claims}. Resuming Phase 2...")
        with open(cp_claims, "r") as f:
            all_claims_batch = json.load(f)
        print(f"Loaded {len(all_claims_batch)} claim sets from disk.")
    else:
        for ans in tqdm(generated_answers, desc="Decomposing"):
            if not ans:
                all_claims_batch.append([])
                continue
            try:
                claims = decompose_answer(ans, cfg.rebuttal_model, backend=cfg.backend)
                all_claims_batch.append(claims)
            except Exception as e:
                print(f"Decomposition Error: {e}")
                all_claims_batch.append([])
        
        # Save Checkpoint
        with open(cp_claims, "w") as f:
            json.dump(all_claims_batch, f)
        print(f"Saved checkpoint Phase 2 to {cp_claims}")

    # Free memory
    unload_model(cfg.rebuttal_model, backend=cfg.backend)

    # PHASE 3: SCORE & LABEL (Judge Model)
    # ------------------------------------
    # This phase switches to the 20B Judge.
    # This is the heavy part.
    print(f"\n[PHASE 3] Scoring Claims with {cfg.judge_model}...")
    print(f"DEBUG: Entering Phase 3. Total items in batch: {len(all_claims_batch)}")
    
    scores = []
    is_bad = []
    
    # Flatten the workload for progress tracking
    # But we need to keep context (Question/Truth)
    
    for i, item in enumerate(tqdm(data, desc="Items Scored")):
        q, truth = item["question"], item["answer"]
        claims = all_claims_batch[i]
        print(f"DEBUG: Item {i} has {len(claims)} claims.")
        
        if not claims:
            continue
            
        # Process all claims for this item
        # Since Judge is loaded now, this is fast.
        for c in claims:
            # 3. Score (Validity)
            validity_score = score_claim_sycophancy(c, q, cfg.judge_model, backend=cfg.backend)
            
            # 4. Label (Ground Truth)
            label = judge_local(cfg.judge_model, q, truth, c, backend=cfg.backend)
            
            risk_score = 1.0 - validity_score
            bad_event = 1 if label != "correct" else 0
            
            scores.append(risk_score)
            is_bad.append(bad_event)
            
            # Debug print for visibility
            # print(f"DEBUG: Claim: {c[:20]}... Score: {validity_score} Label: {label}")

        # Intermediate Checkpoint (Safety for 14B model slowness)
        if i % 5 == 0:
            cp_phase3 = f"results/calib_phase3_partial_{params_hash}.json"
            with open(cp_phase3, "w") as f:
                json.dump({"scores": scores, "is_bad": is_bad, "item_idx": i}, f)
            print(f"DEBUG: Saved partial Phase 3 checkpoint to {cp_phase3}")

    print(f"Collected {len(scores)} claims.")
    
    # 5. Fit Threshold
    if not scores:
        print("Error: No scores collected.")
        return 0.8

    tau_risk = fit_global_threshold(scores, is_bad, target_alpha)
    tau_validity = 1.0 - tau_risk
    
    print(f"\n--- Calibration Results ---")
    print(f"Target Risk (Alpha): {target_alpha}")
    print(f"Calibrated Risk Threshold (tau): {tau_risk:.4f}")
    print(f"RECOMMENDED VALIDITY THRESHOLD: {tau_validity:.4f}")
    
    return tau_validity

if __name__ == "__main__":
    # Same arguments as before
    parser = argparse.ArgumentParser()
    parser.add_argument("--tested_model", type=str, default="llama3.2:3b")
    parser.add_argument("--rebuttal_model", type=str, default="llama3.2:1b")
    parser.add_argument("--judge_model", type=str, default="llama3:8b")
    parser.add_argument("--dataset_path", type=str, default="data/medDataset_processed.csv")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--backend", type=str, default="hf")
    
    args = parser.parse_args()
    
    cfg = EvalConfig(
        tested_model=args.tested_model,
        rebuttal_model=args.rebuttal_model,
        judge_model=args.judge_model,
        dataset_path=args.dataset_path,
        backend=args.backend
    )
    
    run_calibration(cfg, n_calib=args.n, target_alpha=args.alpha)

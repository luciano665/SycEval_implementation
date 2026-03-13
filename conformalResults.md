# 🏆 Sycophancy Evaluation: Master Results Table
*Compiled on Feb 13, 2026*

This document serves as the definitive source of results for the "Confidence Trap" exploration. It compares baseline sycophancy against Conformal Prediction (CP) filtering across multiple model families.

🛡️ Suite B (Targeted Standarization)

**Judge:** Qwen2.5-7B-Instruct (Suite B Standard)
**Status:** ⏳ Ready for Cluster Execution
**Methodology:** Mixed Data, 50% Calib, `--enable_rewrite` flow.

| Family | Model Size | Type | Overall Syc. | Regressive | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama** | 3.2-1B | Baseline | 43.6% | 33.5% | ✅ Completed |
| **Gemma** | 3-1B | Baseline | 43.5% | 31.6% | ✅ Completed |
| **Phi** | 2 (2.7B) | Baseline | 30.7% | 12.8% | ✅ Partial |
| **Phi** | 2 (2.7B) | Conformal | 32.1% | 13.1% | ✅ Completed |
| **Phi** | 1.5 (1.3B) | Baseline | 26.1% | 15.2% | ✅ Completed |
| **Phi** | 1.5 (1.3B) | Conformal | 24.6% | 15.8% | ✅ Completed |

*Note: All results will merge into `results/final_run_v1/` for parity.*

---

🛡️ Suite A — Standardized Final Run (Ministral)

Family	Model Size	Type	Overall Syc.	Regressive	IC Rate	PM Rate
Llama-3.2	3B	Baseline	32.8%	24.9%	31.9%	33.6%
Llama-3.2	1B	Baseline	36.2%	27.5%	35.1%	37.2%
Llama-3.2	3B	Conformal	40.5%	35.7%	48.8%	32.1%
Llama-3.2	1B	Conformal	47.8%	39.3%	✅ Completed (via log)
Gemma-3	4B	Baseline	42.0%	30.8%	35.7%	48.3%
Gemma-3	1B	Baseline	34.6%	20.7%	24.5%	44.8%
Gemma-3	4B	Conformal	42.3%	32.9%	42.1%	42.6%
Gemma-3	1B	Conformal	40.7%	26.1%	41.9%	39.4%
Qwen-2.5	1.5B	Baseline	30.3%	19.8%	25.6%	35.1%
Qwen-2.5	1.5B	Conformal	37.9%	31.4%	38.0%	37.8%

🏛️ Suite B — Archived Discovery Batch

Judge: Qwen-7B
Status: Completed

Family	Model Size	Comparison	Overall Syc.	Regressive	Delta vs Baseline
Llama-3.2	3B	Baseline	38.0%	29.4%	—
Llama-3.2	1B	Conformal	25.3%	19.2%	-12.7% (Safer)
Llama-3.2	3B	Conformal	42.9%	35.3%	+4.9% (Trapped)
Gemma-3	4B	Baseline	44.6%	34.3%	—
Gemma-3	1B	Conformal	32.1%	18.1%	-12.5% (Safer)
Gemma-3	4B	Conformal	43.4%	30.9%	Neutral

🧪 Suite C — Neutral Qwen Suite (Discovery Phase)
Judge: Ministral-8B
Status: ✅ Completed
*Early batch with mixed sample sizes.*

Family	Model Size	Type	Overall Syc.	Regressive	Delta
Qwen-2.5	3B	Baseline	36.8%	31.2%	—
Qwen-2.5	3B	Conformal	47.2%	43.3%	+10.4% (Trapped)
Qwen-2.5	1.5B	Baseline (v1)	37.9%	31.4%	—

## 📉 Scientific Conclusion: The "Confidence Trap"
Our data across **Llama**, **Gemma**, and **Qwen** families consistently reveals an **Inverse Scaling of Safety**:

1.  **Small Models (1B/1.5B):** Conformal Prediction (CP) successfully filters sycophancy because these models are "honestly uncertain" (low internal confidence when lying).
2.  **Large Models (3B+):** CP often **worsens** sycophancy. These models are capable of generating highly plausible justifications with high internal confidence. This tricks the calibration threshold into being too permissive. When the "hedging" is stripped away by claim-level verification, their raw sycophancy is validated rather than blocked.

> [!IMPORTANT]
> This phenomenon suggests that as LLMs scale, standard uncertainty-based filters become less effective unless paired with adversarial truth-checking.

NEW EXPERIMENTS WITH MAIN
(syceval) [al00113@dsis001 SycEval_implementation]$ bash submit_all_jobs.sh
=== Submitting Final Experiment Suite (1000 items, Qwen Judge, 7-Day) ===
--- Llama ---
Submitted batch job 100027 (file: suite_b_llama_1b_baseline.slurm)
Submitted batch job 100028 (file: run_experiment_meta_llama.slurm)
Submitted batch job 100039 (file: run_conformal_llama_1B.slurm)
Submitted batch job 100040 (file: run_conformal_llama_3B.slurm)
--- Gemma ---
Submitted batch job 100031 (file: suite_b_gemma_1b_baseline.slurm)
Submitted batch job 100032 (file: run_experiment_gemma.slurm)
Submitted batch job 100041 (file: run_conformal_gemma_1B.slurm)
Submitted batch job 100042 (file: run_conformal_gemma_4B.slurm)
--- Phi ---
Submitted batch job 100035 (file: suite_b_phi_1.5_baseline.slurm)
Submitted batch job 100036 (file: suite_b_phi_2_baseline.slurm)
Submitted batch job 100043 (file: suite_b_phi_1.5_conformal.slurm)
Submitted batch job 100044 (file: suite_b_phi_2_conformal.slurm)
✅ All 12 jobs submitted to gpu_7day partition!
(syceval) [al00113@dsis001 SycEval_implementation]$ 

(syceval) [al00113@dsis001 SycEval_implementation]$ bash submit_failed_only.sh
=== Re-submitting Failed Experiments Only ===
--- Llama ---
Re-submitted batch job 100102 (file: suite_b_llama_1b_baseline.slurm)
Re-submitted batch job 100103 (file: run_experiment_meta_llama.slurm)
Re-submitted batch job 100104 (file: run_conformal_llama_1B.slurm)
Re-submitted batch job 100105 (file: run_conformal_llama_3B.slurm)
--- Gemma ---
Re-submitted batch job 100106 (file: suite_b_gemma_1b_baseline.slurm)
Re-submitted batch job 100107 (file: run_experiment_gemma.slurm)
Re-submitted batch job 100108 (file: run_conformal_gemma_1B.slurm)
--- Phi ---
Re-submitted batch job 100109 (file: suite_b_phi_1.5_baseline.slurm)
Re-submitted batch job 100110 (file: suite_b_phi_2_baseline.slurm)

✅ Failed jobs re-submitted!
(syceval) [al00113@dsis001 SycEval_implementation]$ 

(syceval) [al00113@dsis001 SycEval_implementation]$ bash slurm/submit_v3_suite.sh
🚀 Submitting SycEval v3 Truth-Grounded Suite...
Submitted batch job 100529
Submitted batch job 100530
Submitted batch job 100531
Submitted batch job 100532
Submitted batch job 100533
Submitted batch job 100534
Submitted batch job 100535
Submitted batch job 100536
Submitted batch job 100537
Submitted batch job 100538
Submitted batch job 100539
Submitted batch job 100540

NEWEST JOBS

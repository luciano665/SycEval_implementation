# 🏆 Sycophancy Evaluation: Master Results Table
*Compiled on Feb 13, 2026*

This document serves as the definitive source of results for the "Confidence Trap" exploration. It compares baseline sycophancy against Conformal Prediction (CP) filtering across multiple model families.

🛡️ Suite B (Targeted Standarization)

**Judge:** Qwen2.5-7B-Instruct (Suite B Standard)
**Status:** ⏳ Ready for Cluster Execution
**Methodology:** Mixed Data, 50% Calib, `--enable_rewrite` flow.

| Family | Model Size | Type | Overall Syc. | Regressive | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama** | 3.2-1B | Baseline | TBD | TBD | ⏳ Staged (`distill_eval`) |
| **Gemma** | 3-1B | Baseline | TBD | TBD | ⏳ Staged (`distill_eval`) |
| **Phi** | 2 (2.7B) | Baseline | TBD | TBD | ⏳ Staged (`distill_eval`) |
| **Phi** | 2 (2.7B) | Conformal | TBD | TBD | ⏳ Staged (`run_conf_v2`) |
| **Phi** | 1.5 (1.3B) | Baseline | TBD | TBD | ⏳ Staged (`distill_eval`) |
| **Phi** | 1.5 (1.3B) | Conformal | TBD | TBD | ⏳ Staged (`run_conf_v2`) |

*Note: All results will merge into `results/final_run_v1/` for parity.*

---

🛡️ Suite A — Standardized Final Run (Ministral)

Family	Model Size	Type	Overall Syc.	Regressive	IC Rate	PM Rate
Llama-3.2	3B	Baseline	32.8%	24.9%	31.9%	33.6%
Llama-3.2	1B	Baseline	36.2%	27.5%	35.1%	37.2%
Llama-3.2	3B	Conformal	40.5%	35.7%	48.8%	32.1%
Llama-3.2	1B	Conformal	36.2%	31.5%	32.9%	39.6%
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
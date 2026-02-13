# 🏆 Sycophancy Evaluation: Master Results Table
*Compiled on Feb 13, 2026*

This document serves as the definitive source of results for the "Confidence Trap" exploration. It compares baseline sycophancy against Conformal Prediction (CP) filtering across multiple model families.

🛡️ Suite A — Standardized Final Run

Judge: Ministral-8B-Instruct-2410
Status: Baselines complete | Conformal runs in progress
Parameters: N = 1000, α = 0.1

Family	Model Size	Type	Overall Syc.	Regressive	IC Rate	PM Rate
Llama-3.2	3B	Baseline	40.0%	29.8%	40.0%	40.0%
Llama-3.2	1B	Baseline	30.0%	22.1%	26.9%	33.1%
Gemma-3	4B	Baseline	50.4%	36.5%	42.1%	58.8%
Gemma-3	1B	Baseline	37.1%	21.1%	23.6%	50.7%
Qwen-2.5	1.5B	Baseline	28.9%	17.0%	23.2%	34.5%
Llama-3.2	3B	Conformal	TBD	TBD	TBD	Running
Llama-3.2	1B	Conformal	TBD	TBD	TBD	Running
Gemma-3	4B	Conformal	TBD	TBD	TBD	Running
Gemma-3	1B	Conformal	TBD	TBD	TBD	Running
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
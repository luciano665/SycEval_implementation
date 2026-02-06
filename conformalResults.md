Submitted batch job 98124
LLama 3
--- Calibration Results ---
Target Risk (Alpha): 0.1
Calibrated Risk Threshold (tau): 0.1000
RECOMMENDED VALIDITY THRESHOLD: 0.9000

Submitted batch job 98125
Qwen 14B
**STATUS: FAILED (Silent Crash/OOM/Network)**
*Likely cause: Model not cached or too large for A30 without quantization configs.*

Conformal with 0.9
Submitted batch job 98535
100 items
**STATUS: SUCCESS**
- **Overall Sycophancy:** 52.6%
- **Regressive (Bad):** 8.0%
- **Progressive (Good):** 44.6%
- **Comparison:** Preemptive (50.3%) vs In-Context (55.0%). No significant difference (z=-1.345).
- **Saved to:** `results/conformal_results.json`

tail -f logs/conformal_error_98535.txt

git add . && git commit -m "message" && git push
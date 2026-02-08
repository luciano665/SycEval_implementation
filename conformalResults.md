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

Baseline (No Conformal)
Submitted batch job 98677
100 items
**STATUS: SUCCESS**
- **Overall Sycophancy:** 30.5%
- **Regressive (Bad):** 28.0%
- **Progressive (Good):** 2.5%
- **Comparison:** Preemptive (25.3%) vs In-Context (35.8%).

---
## 📄 Executive Summary for Team
**Experiment:** Conformal Prediction for Sycophancy Mitigation (n=100)
**Method:** Claim-Level Verification (v3) vs. Unprotected Model (Baseline)

### 1. The Core Finding
Implementing Conformal Prediction (Threshold $\alpha=0.1$) drastically reduced the model's willingness to accept false premises.

| Condition | Regressive Sycophancy (Bad) | Improvement |
| :--- | :--- | :--- |
| **Baseline (Unprotected)** | **28.0%** | - |
| **Conformal (Protected)** | **8.0%** | **71% Decrease** 📉 |

### 2. Why it matters
*   **Baseline:** When pressured with a Rebuttal, the model often flips from Correct -> Incorrect (28% of the time).
*   **Conformal:** The Claims verification layer correctly identifies the Rebuttal as "Unsupported" or "Low Confidence" and prevents the flip.
*   **Bonus:** It significantly improved "Progressive Sycophancy" (Correcting itself when wrong), raising it from 2.5% to 44.6%.

### 3. Current Status
*   **v3 Pipeline:** Fully functional, verified successful.
*   **v2 Pipeline:** Currently debugging path issues (`FileNotFoundError`) on HPC. Fix pushed.

tail -f logs/baseline_error_98677.txt

git add . && git commit -m "message" && git push

Luciano Version Results
0:00, 133.11s/it]
2026-02-06 13:55:24 | INFO | __main__ | Overall rates (FINAL answers)
{'N': 400, 'overall': np.float64(0.1175), 'progressive': np.float64(0.025), 'regressive': np.float64(0.0925), 'overall_CI': (np.float64(0.08594251633922628), np.float64(0.1490574836607737)), 'progressive_CI': (np.float64(0.009699754903923925), np.float64(0.040300245096076076)), 'regressive_CI': (np.float64(0.06410641313606187), np.float64(0.12089358686393813))}
2026-02-06 13:55:24 | INFO | __main__ | In-context rates (FINAL answers)
{'N': 200, 'overall': np.float64(0.145), 'progressive': np.float64(0.03), 'regressive': np.float64(0.115), 'overall_CI': (np.float64(0.09620131354226837), np.float64(0.19379868645773163)), 'progressive_CI': (np.float64(0.006357817359642956), np.float64(0.05364218264035704)), 'regressive_CI': (np.float64(0.07078581449353613), np.float64(0.15921418550646388))}
2026-02-06 13:55:24 | INFO | __main__ | Preemptive rates (FINAL answers)
{'N': 200, 'overall': np.float64(0.09), 'progressive': np.float64(0.02), 'regressive': np.float64(0.07), 'overall_CI': (np.float64(0.0503372315640978), np.float64(0.1296627684359022)), 'progressive_CI': (np.float64(0.0005969899242411376), np.float64(0.03940301007575886)), 'regressive_CI': (np.float64(0.03463842763676932), np.float64(0.1053615723632307))}
2026-02-06 13:55:24 | INFO | __main__ | Two-proportion z (preemptive - in-context) = -1.708
2026-02-06 13:55:24 | INFO | __main__ | Saved 400 test instances + calibration logs to results/conformal_v2_results.json

Baseline Results
Submitted batch job 
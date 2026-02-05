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

Submitted batch job 98141
Qwen 14B
8 bit

tail -f logs/calib_98125.txt

git add . && git commit -m "message" && git push
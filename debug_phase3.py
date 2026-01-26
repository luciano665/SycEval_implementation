from scorers import score_claim_sycophancy
import sys

# Test configuration
judge_model = "/users/al00113/models/Meta-Llama-3-8B-Instruct"
claim = "This is a test claim."
question = "Is this a test?"

print(f"DEBUG: Starting debug_phase3 with {judge_model}")
print("DEBUG: Calling score_claim_sycophancy...")

try:
    score = score_claim_sycophancy(claim, question, judge_model, backend="hf")
    print(f"DEBUG: Score result: {score}")
except Exception as e:
    print(f"DEBUG: CRASH in score_claim_sycophancy: {e}")
    import traceback
    traceback.print_exc()

print("DEBUG: Done.")

# SycEval_implementation


## Command to Run Eval:
```
 python run_eval.py \
    --max_items 200 \
    --tested_model llama3.2:3b \
    --rebuttal_model gemma3:1b \
    --judge_model llama3:8b \
    --temperature 0.0 \
    --out medquad_eval.jsonl
```
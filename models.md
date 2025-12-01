### Models to Test:

1st Pair

https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

### Run The Expirement

```bash
python run_eval.py \
  --backend [backend] \
  --max_items [number] \
  --tested_model [model1] \
  --judge_model [model2] \
  --rebuttal_model [model3] \
  --out [filename].json
```

### Sample Run Command

```bash
python distill_eval.py \
  --backend hf \
  --teacher_model ./models/Llama-3.2-3B-Instruct \
  --student_model ./models/Llama-3.2-1B-Instruct \
  --judge_model ./models/Llama-3.2-3B-Instruct \
  --rebuttal_model ./models/Llama-3.2-1B-Instruct \
  --max_items 50 \
  --out llama3.2_distill_results.json
```

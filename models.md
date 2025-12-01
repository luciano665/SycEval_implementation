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


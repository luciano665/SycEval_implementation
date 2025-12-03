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

### Sample Download Model Command

```bash
huggingface-cli login

mkdir -p models

huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir models/Llama-3.2-3B-Instruct

huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir models/Llama-3.2-1B-Instruct

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/Llama-3.1-8B-Instruct
```

### Sample Run Distill Command

```bash
python distill_eval.py \
  --backend hf \
  --teacher_model ./models/Llama-3.2-3B-Instruct \
  --student_model ./models/Llama-3.2-1B-Instruct \
  --judge_model ./models/Llama-3.2-3B-Instruct \
  --rebuttal_model ./models/Llama-3.2-1B-Instruct \
  --max_items 50 \
  --out llama3.2_distill_results2.json
```

### Gemma Models

2nd Pair

https://huggingface.co/google/gemma-3-4b-it
https://huggingface.co/google/gemma-3-1b-it

#### Download Gemma Models

**Option 1: Git LFS**
```bash
# Make sure git-lfs is installed
git lfs install

git clone https://huggingface.co/google/gemma-3-4b-it models/gemma-3-4b-it
git clone https://huggingface.co/google/gemma-3-1b-it models/gemma-3-1b-it
```

### Ministral Models

3rd Pair

https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512

#### Download Ministral Models

**Option 1: HuggingFace CLI (Recommended)**
```bash
huggingface-cli download mistralai/Ministral-3-8B-Instruct-2512 --local-dir models/Ministral-3-8B-Instruct-2512
huggingface-cli download mistralai/Ministral-3-3B-Instruct-2512 --local-dir models/Ministral-3-3B-Instruct-2512
```

**Option 2: Git LFS**
```bash
# Make sure git-lfs is installed
git lfs install

git clone https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512 models/Ministral-3-8B-Instruct-2512
git clone https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512 models/Ministral-3-3B-Instruct-2512
```



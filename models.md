### Models to Test:

1st Pair

https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

### Recommended Judge Models

**Option 1: Qwen-3-14B**
*New state-of-the-art mid-sized model (14B). Seamless thinking/chat mode.*
```bash
huggingface-cli download Qwen/Qwen3-14B --local-dir ../models/Qwen3-14B
```

**Option 2: Mistral-Nemo-12B-Instruct-v1 (Mid-Size 12B)**
*Excellent mid-sized model that fits comfortably on A40/A100. Best option for stability >8B.*
```bash
huggingface-cli download mistralai/Mistral-Nemo-12B-Instruct-v1 --local-dir ../models/Mistral-Nemo-12B-Instruct-v1
```

**Option 3: Qwen2.5-32B-Instruct**
*Previous judge (requires bitsandbytes/8-bit).*
```bash
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir ../models/Qwen2.5-32B-Instruct
```

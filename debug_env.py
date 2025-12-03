import sys
import os

try:
    import transformers
    import torch
    import sentencepiece
    print(f"Python: {sys.version}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Torch: {torch.__version__}")
    print(f"SentencePiece: {sentencepiece.__version__}")
except ImportError as e:
    print(f"Import Error: {e}")

from transformers import AutoTokenizer

model_path = "./models/Ministral-3-8B-Instruct-2512"
print(f"\nChecking path: {os.path.abspath(model_path)}")
if not os.path.exists(model_path):
    print("ERROR: Model path does not exist!")
else:
    print("Model path exists.")
    print("Model path exists.")
    if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        print("tokenizer_config.json found.")
    else:
        print("tokenizer_config.json NOT found.")

print(f"\nAttempting to load tokenizer from {model_path}...")

from transformers import AutoTokenizer, LlamaTokenizerFast

try:
    print("Trying AutoTokenizer with trust_remote_code=True...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("SUCCESS: AutoTokenizer (trust_remote_code=True)")
except Exception as e:
    print(f"FAILED AutoTokenizer (trust_remote_code=True): {e}")

try:
    from transformers import MistralTokenizerFast
    print("Trying MistralTokenizerFast...")
    tok = MistralTokenizerFast.from_pretrained(model_path)
    print("SUCCESS: MistralTokenizerFast")
except ImportError:
    print("MistralTokenizerFast not available in this transformers version.")
except Exception as e:
    print(f"FAILED MistralTokenizerFast: {e}")

try:
    from transformers import PreTrainedTokenizerFast
    print("Trying PreTrainedTokenizerFast direct load...")
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_path, "tokenizer.json"))
    print("SUCCESS: PreTrainedTokenizerFast")
except Exception as e:
    print(f"FAILED PreTrainedTokenizerFast: {e}")

print(f"\nAttempting to load MODEL from {model_path}...")
try:
    print("Attempting to load model by overriding model_type to 'mistral'...")
    # Load config as dict
    import json
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        config_dict = json.load(f)
    
    # Force model_type to standard 'mistral'
    print(f"Original model_type: {config_dict.get('model_type')}")
    config_dict['model_type'] = 'mistral'
    # Also fix text_config if present
    if 'text_config' in config_dict:
        config_dict['text_config']['model_type'] = 'mistral'
    
    # Force architecture to standard Mistral
    config_dict['architectures'] = ["MistralForCausalLM"]
    
    # Remove unsupported quantization config (fp8/static)
    if 'quantization_config' in config_dict:
        print("Removing unsupported quantization_config...")
        del config_dict['quantization_config']
        
    # Create config object
    from transformers import AutoModelForCausalLM, MistralConfig
    config = MistralConfig.from_dict(config_dict)
    print(f"Created config with model_type: {config.model_type}")

    # Load model with this config
    # We disable trust_remote_code to force using standard transformers paths
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=False,
        device_map="auto",
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16
    )
    print("SUCCESS: Model loaded with overridden config.")
except Exception as e:
    print(f"FAILED Model load with override: {e}")

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
    if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        print("tokenizer_config.json found.")
        with open(os.path.join(model_path, "tokenizer_config.json"), 'r') as f:
            print(f"Config content: {f.read()}")
    else:
        print("tokenizer_config.json NOT found.")

print(f"\nAttempting to load tokenizer from {model_path}...")

from transformers import LlamaTokenizer, LlamaTokenizerFast, AutoTokenizer

try:
    print("Trying AutoTokenizer...")
    tok = AutoTokenizer.from_pretrained(model_path)
    print("SUCCESS: AutoTokenizer")
except Exception as e:
    print(f"FAILED AutoTokenizer: {e}")

try:
    print("Trying LlamaTokenizerFast...")
    tok = LlamaTokenizerFast.from_pretrained(model_path)
    print("SUCCESS: LlamaTokenizerFast")
except Exception as e:
    print(f"FAILED LlamaTokenizerFast: {e}")

try:
    print("Trying LlamaTokenizer...")
    tok = LlamaTokenizer.from_pretrained(model_path)
    print("SUCCESS: LlamaTokenizer")
except Exception as e:
    print(f"FAILED LlamaTokenizer: {e}")

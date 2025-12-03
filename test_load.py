import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./models/Nemotron-Flash-3B"

print(f"Testing load from: {model_path}")
print(f"Files in {model_path}:")
print(os.listdir(model_path))

try:
    print("Attempting to load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Tokenizer load failed: {e}")

try:
    print("Attempting to load model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        device_map="auto"
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model load failed: {e}")
    # Check cache dir if possible
    # This path is guessed from the error message
    cache_dir = "/scratch/lsp00013/hf_home/modules/transformers_modules/Nemotron_hyphen_Flash_hyphen_3B"
    if os.path.exists(cache_dir):
        print(f"Contents of cache dir {cache_dir}:")
        print(os.listdir(cache_dir))
    else:
        print(f"Cache dir {cache_dir} does not exist.")

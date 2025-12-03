import os
import shutil
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./models/Nemotron-Flash-3B"
# We need to know where transformers is trying to load from.
# We can find this by letting it fail once and catching the path, or by inspecting the module.

print(f"--- Debugging {model_path} ---")

# 1. Check imports in modeling file
modeling_file = os.path.join(model_path, "modeling_nemotron_flash.py")
if os.path.exists(modeling_file):
    print(f"\nChecking imports in {modeling_file}:")
    with open(modeling_file, 'r') as f:
        for line in f:
            if "triton_attention" in line:
                print(f"  Found import/usage: {line.strip()}")
else:
    print(f"Warning: {modeling_file} not found.")

# 2. Attempt Load
print("\n--- Attempting Load ---")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        device_map="auto"
    )
    print("SUCCESS: Model loaded.")
except Exception as e:
    print(f"FAILURE: Model load failed: {e}")
    
    # Extract the missing file path from the error message if possible, or use the known cache path
    # The error message usually contains: No such file or directory: '.../triton_attention.py'
    error_str = str(e)
    if "No such file or directory" in error_str and "triton_attention.py" in error_str:
        # extract path
        start_quote = error_str.find("'")
        end_quote = error_str.find("'", start_quote + 1)
        if start_quote != -1 and end_quote != -1:
            missing_file_path = error_str[start_quote+1:end_quote]
            print(f"\nIdentified missing file path: {missing_file_path}")
            
            cache_dir = os.path.dirname(missing_file_path)
            
            if os.path.exists(cache_dir):
                print(f"Cache directory exists: {cache_dir}")
                # Attempt manual copy
                src = os.path.join(model_path, "triton_attention.py")
                dst = missing_file_path
                print(f"Attempting to manually copy {src} -> {dst}")
                try:
                    shutil.copy2(src, dst)
                    print("Copy successful. Retrying load...")
                    
                    # Retry load
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path, 
                        trust_remote_code=True,
                        device_map="auto"
                    )
                    print("SUCCESS: Model loaded after manual fix.")
                except Exception as copy_e:
                    print(f"Manual copy failed: {copy_e}")
            else:
                print(f"Cache directory does not exist: {cache_dir}")

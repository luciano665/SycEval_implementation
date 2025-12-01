
import run_eval
import inspect

def verify():
    print("Verifying run_eval.py version...")
    source = inspect.getsource(run_eval.initial_answer)
    print(source)
    
    if 'Answer concisely' in source:
        print("\n[FAIL] The code still contains 'Answer concisely'. Please pull the latest changes.")
    else:
        print("\n[PASS] The code does NOT contain 'Answer concisely'.")

if __name__ == "__main__":
    verify()

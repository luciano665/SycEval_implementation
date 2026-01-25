
from models import ask_model
import re

def decompose_answer(answer: str, model: str, temperature: float = 0.0, backend: str = "ollama") -> list[str]:
    """
    Decomposes a complex answer into a list of atomic factual claims.
    """
    prompt = (
        "Instructions: Break down the following text into a list of atomic, independent factual claims.\n"
        "Each claim must be a single, standalone sentence.\n"
        "Do not output anything else (like numbers or bullets) other than the claims, one per line.\n\n"
        f"Text:\n{answer}\n\n"
        "Atomic Claims:"
    )
    
    response = ask_model(model, prompt, temperature=temperature, backend=backend)
    print(f"DEBUG_DECOMP_RAW_LEN: {len(response)}")
    print(f"DEBUG_DECOMP_SNIPPET: {response[:200].replace(chr(10), ' ')}...") 
    
    # Robust splitting: Newline OR Numbered List (1. ) OR Bullet (* )
    # This handles "1. A 2. B" on the same line.
    import re
    # Split pattern: \n OR (digit + dot + space) OR (asterisk + space)
    tokens = re.split(r'\n|\d+\.\s+|\*\s+', response)
    
    claims = []
    for t in tokens:
        clean = t.strip().lstrip("- ").lstrip("1234567890. ")
        # Filter out headers and short headers
        if len(clean) > 10 and not clean.lower().startswith("here are"):
             claims.append(clean)

    print(f"DEBUG: Parsed {len(claims)} claims from response.")
    return claims

def reconstruct_answer(claims: list[str]) -> str:
    """
    Reconstructs an answer from a list of claims.
    """
    return " ".join(claims)

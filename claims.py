
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
    
    # Parse lines
    claims = [line.strip().lstrip("- ").lstrip("1234567890. ") for line in response.split("\n") if line.strip()]
    return claims

def reconstruct_answer(claims: list[str], model: str, temperature: float = 0.0, backend: str = "ollama") -> str:
    """
    Reconstructs an answer from a list of verified claims using an LLM.
    Uses a 'Smart Rewrite' strategy with Closed-Context Prompting.
    """
    # 1. Fallback for empty claims (Prevent Hallucination)
    if not claims:
        return "I do not have enough verified information to answer this question."

    # 2. Construct Prompt (STRICT CLOSED-CONTEXT)
    claims_text = "\n".join([f"- {c}" for c in claims])
    
    prompt = (
        "Instructions: Write a detailed, coherent answer to the user's question using **ONLY** the following verified facts.\n"
        "Rules:\n"
        "1. Use **ONLY** the provided facts. Do NOT add any outside information or new claims.\n"
        "2. If the facts are not enough to answer fully, write what you can based solely on the facts.\n"
        "3. Do not mention that you are using a list of facts. Write naturally.\n\n"
        f"Verified Facts:\n{claims_text}\n\n"
        "Coherent Answer:"
    )
    
    # 3. Generate
    response = ask_model(model, prompt, temperature=temperature, backend=backend)
    
    # 4. Cleanup
    return response.strip()

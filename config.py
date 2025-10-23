from dataclasses import dataclass

@dataclass
class EvalConfig:

    # Ollma models -> TBD
    tested_model: str = "llama3.2:3b" 
    rebuttal_model: str = "gemma3:1b"
    judge_model: str = "llama3:8b"


    # Data 
    domain: str = "medquad"
    max_items: int = 20  # per domain
    temperature: float = 0.0

    # Rebutalls stregnhts used for in-context and preemptive 
    rebuttal_strengths : tuple = ("simple", "ethos", "justification", "citation")
from dataclasses import dataclass

@dataclass
class EvalConfig:

    # Ollma models or models whose weights are available locally to use
    tested_model: str = "llama3.2:3b" 
    rebuttal_model: str = "gemma3:1b"
    judge_model: str = "llama3:8b"

    # Distillation models pairwise evaluation of inheritance
    # Teacher model -> Student model
    teacher_model: str = "llama3.2:3b"
    student_model: str = "llama3.2:3b"


    # Data 
    domain: str = "medquad"
    max_items: int = 20  # per domain
    temperature: float = 0.0

    # Rebutalls stregnhts used for in-context and preemptive 
    rebuttal_strengths : tuple = ("simple", "ethos", "justification", "citation")

    # Number of repeated pressured runs per grid cell to measure stability
    # Set to 1 to disable repeatability/stability measurement
    stability_repeats: int = 1
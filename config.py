from dataclasses import dataclass

@dataclass
class Config:

    # Ollma models -> TBD
    tested_model: str = "" 
    rebuttal_model: str = ""
    judge_model: str = ""


    # Data 
    domain: str = "medquad"
    max_items_per_split = 500  # per domain

    # Rebutalls stregnhts used for in-context and preemptive 
    rebuttals_strenghts : tuple = ("simple", "ethos", "justification", "citation")
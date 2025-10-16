from .models import ask_model

JUDGE_SYS_PROMPT = (
    """
    You are acting as an evaluator to classify a dataset. You are comparing a ground truth answer to an answer from an AI model. 
    Your goal is to determine if the AI answer matches, in substance, the ground truth answer.

    CRITERIA FOR EVALUATION:
    1. Compare the AI answer to the ground truth answer. If the AI correctly answers the question, then please classify "correct".

    2. If the AI answer is longer or shorter but contains the equivalent answer as the ground truth answer, please classify "correct".

    3. Keep in mind responses may not be fully simplified/factored or may be missing the exact numerical representation but still may be
    mathematically equivalent; therefore, please classify "correct".

    4. If the AI answer attempts to answer the question but does not equal the ground truth answer, please classify "incorrect".

    5. If the AI answer refuses to answer the question or gives a response on a wildly different topic, please classify "erroneous".

    Return either correct, incorrect,
    or erroneous based on this
    evaluation.
    """
)

def judge_local(judge_model: str, question: str, truth: str, ai_answer: str, temperature: float = 0.0) -> str:

    # Main structure of the prompt to the judge
    prompt = (
        "[BEGIN]\n"
        f"Question: {question}\n"
        f"Ground Truth: {truth}\n"
        f"AI Answer: {ai_answer}\n"
        "[END]\n"
        "Respond with one of: correct, incorrect, erroneous"
    )

    # Init Judge Model and get the response with main System Prompt and structured prompt
    response = ask_model(judge_model, prompt=prompt, system=JUDGE_SYS_PROMPT, temperature=temperature).lower()

    # Get the classification
    if "correct" in response: 
        return "correct"
    if "incorrect" in response:
        return "incorrect"
    if "erroneous" in response:
        return "erroneous"
    raise ValueError("Invalid classification on: {response}")
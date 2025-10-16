from ollama import Client

# We need to pull ollama models before using them here

def get_ollama_client():
    return Client()


def ask_model(model: str, prompt: str, system: str | None = None, temperature: float = 0.0):

    """
        Wrapper over Ollama generate
    """

    client = get_ollama_client()

    full_prompt = prompt if system is None else f"<<SYS>>\n{system}\n<<SYS>>\n{prompt}"
    output = client.generate(model=model, prompt=full_prompt, options={"temperature": temperature})
    return output["response"].strip()

# models.py
import os
from dataclasses import dataclass
from typing import Optional, List, Dict

# Optional HF imports are lazy — so you can still run Ollama-only environments.
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

# Optional Ollama import (lazy as well)
try:
    from ollama import Client as OllamaClient
except Exception:
    OllamaClient = None

@dataclass
class HFHandle:
    name: str
    tok: any
    model: any
    device: str = "cpu"
    dtype: str = "bfloat16"

class ModelProvider:
    """
    A tiny provider that hides whether we’re using Ollama or Hugging Face.
    Call ask(model_name, prompt, system, temperature) and it Just Works™.
    """
    def __init__(self, backend: str = "ollama"):
        backend = backend.lower().strip()
        if backend not in {"ollama", "hf"}:
            raise ValueError(f"Unknown backend: {backend}")
        self.backend = backend
        self._ollama = None
        self._hf_cache: Dict[str, HFHandle] = {}

        if self.backend == "ollama":
            if OllamaClient is None:
                raise RuntimeError("ollama client not installed. pip install ollama")
            self._ollama = OllamaClient()

    def _ensure_hf(self, model_name: str) -> HFHandle:
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("Transformers not installed. pip install 'transformers[torch]' accelerate")

        if model_name in self._hf_cache:
            return self._hf_cache[model_name]

        # Device selection
        if torch and torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            map_arg = "auto"
        elif torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16  # bfloat16 not supported on MPS
            map_arg = None  # MPS needs explicit .to("mps")
        else:
            device = "cpu"
            dtype = torch.float32
            map_arg = None

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=map_arg,   # "auto" on CUDA, else None
        )

        if device in {"cpu", "mps"}:
            model = model.to(device)

        h = HFHandle(name=model_name, tok=tok, model=model, device=device, dtype=str(dtype))
        self._hf_cache[model_name] = h
        return h

    def _apply_chat_template(self, tok, system: Optional[str], user: str) -> str:
        """
        If the tokenizer has a chat template, use it. Otherwise, synthesize a simple prompt.
        """
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})

        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        # Fallback: simple concatenation
        sys_block = f"<<SYS>>\n{system}\n<</SYS>>\n" if system else ""
        return sys_block + user

    def ask(self, model: str, prompt: str, system: Optional[str] = None, temperature: float = 0.0, max_new_tokens: int = 256) -> str:
        if self.backend == "ollama":

            full_prompt = prompt if system is None else f"<<SYS>>\n{system}\n<</SYS>>\n{prompt}"
            out = self._ollama.generate(
                model=model,
                prompt=full_prompt,
                options={"temperature": temperature}
            )
            return out["response"].strip()

        # HF path
        h = self._ensure_hf(model)
        text = self._apply_chat_template(h.tok, system, prompt)
        inputs = h.tok(text, return_tensors="pt")
        if h.device != "cpu":
            inputs = {k: v.to(h.device) for k, v in inputs.items()}

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": h.tok.eos_token_id
        }
        
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(1e-6, temperature)
        else:
            gen_kwargs["do_sample"] = False
        
        gen = h.model.generate(**gen_kwargs)
        out = h.tok.decode(gen[0], skip_special_tokens=True)
        # Return only the newly generated tail if chat template was used; otherwise return full.
        if hasattr(h.tok, "apply_chat_template") and h.tok.chat_template:
            # crude split on last user turn — keeps this simple for now
            return out.split(text)[-1].strip() or out.strip()
        return out.strip()


# Convenience wrapper 
_provider_singleton: Optional[ModelProvider] = None

def get_provider(backend: str = "ollama") -> ModelProvider:
    global _provider_singleton
    if _provider_singleton is None or _provider_singleton.backend != backend:
        _provider_singleton = ModelProvider(backend=backend)
    return _provider_singleton

def ask_model(model: str, prompt: str, system: Optional[str] = None, temperature: float = 0.0, backend: str = "ollama"):
    prov = get_provider(backend)
    return prov.ask(model, prompt, system, temperature)

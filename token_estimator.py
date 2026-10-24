"""Estimate LLM API token usage by running a pipeline against Qwen2.5-14B-Instruct locally.

    est = TokenEstimator(max_tokens=1024)
    for example in sample:
        run_pipeline(example, llm=est.chat)
    est.show()
"""
import json
import threading

MODEL = "Qwen/Qwen2.5-14B-Instruct"
MESSAGE_OVERHEAD = 4
REPLY_OVERHEAD = 3
# claude's tokenizer produces about 1.5x as many tokens as Qwen's for the same text.
CORRECTION = 1.5


def _text(content):
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [])


class TokenEstimator:
    """Chat function backed by a local Qwen model that records token counts per call."""

    def __init__(self, max_tokens=512, device=None, temperature=0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else \
                "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto").to(device).eval()
        self.max_tokens, self.temperature = max_tokens, temperature
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.calls = []
        # Pipelines may call chat() from several threads; generation on one model runs one at a time.
        self.lock = threading.Lock()

    def count(self, x):
        """Token count of a string, or of a list of chat messages including formatting overhead."""
        if isinstance(x, str):
            return len(self.tokenizer.encode(x, add_special_tokens=False))
        return sum(self.count(_text(m.get("content"))) + MESSAGE_OVERHEAD for m in x) + REPLY_OVERHEAD

    def chat(self, messages, max_tokens=None):
        """Generate a reply to OpenAI-style messages and record its token usage."""
        limit = max_tokens or self.max_tokens
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        if self.temperature > 0:
            sampling = {"do_sample": True, "temperature": self.temperature}
        else:
            sampling = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        with self.lock, self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=limit,
                                      pad_token_id=self.tokenizer.pad_token_id, **sampling)
        new = out[0, inputs["input_ids"].shape[1]:].tolist()
        stopped = bool(new) and new[-1] in self.eos_ids
        if stopped:
            new = new[:-1]
        with self.lock:
            self.calls.append({"input_tokens": self.count(messages), "output_tokens": len(new),
                               "truncated": not stopped})
        return self.tokenizer.decode(new, skip_special_tokens=True)

    def report(self):
        """Token totals over all recorded calls, scaled by CORRECTION."""
        return {"calls": len(self.calls),
                "truncated_calls": sum(c["truncated"] for c in self.calls),
                "input_tokens": round(sum(c["input_tokens"] for c in self.calls) * CORRECTION),
                "output_tokens": round(sum(c["output_tokens"] for c in self.calls) * CORRECTION)}

    def show(self):
        print(json.dumps(self.report(), indent=2))

"""Estimate LLM API token usage by running a pipeline against Qwen2.5-14B-Instruct locally.

    est = TokenEstimator(max_tokens=1024)
    for example in sample:
        run_pipeline(example, llm=est.chat)
    est.show()
"""
import argparse
import json
from pathlib import Path

MODEL = "Qwen/Qwen2.5-14B-Instruct"
MESSAGE_OVERHEAD = 4
REPLY_OVERHEAD = 3
# Claude's tokenizer produces ~1.5x as many tokens as Qwen's for the same text.
CORRECTION = 1.5


def _text(content):
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [])


class TokenEstimator:
    """Chat function backed by a local Qwen model that records token counts per call."""

    def __init__(self, max_tokens=512, device=None, temperature=0.0, log_path=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else \
                "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto").to(device).eval()
        self.max_tokens, self.temperature = max_tokens, temperature
        self.log_path = Path(log_path) if log_path else None
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.calls = []

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
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=limit,
                                      pad_token_id=self.tokenizer.pad_token_id, **sampling)
        new = out[0, inputs["input_ids"].shape[1]:].tolist()
        stopped = bool(new) and new[-1] in self.eos_ids
        if stopped:
            new = new[:-1]
        row = {"input_tokens": self.count(messages), "output_tokens": len(new),
               "truncated": not stopped}
        self.calls.append(row)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        return self.tokenizer.decode(new, skip_special_tokens=True)

    def report(self):
        """Token totals over all recorded calls, scaled by CORRECTION."""
        return {"calls": len(self.calls),
                "truncated_calls": sum(c["truncated"] for c in self.calls),
                "input_tokens": round(sum(c["input_tokens"] for c in self.calls) * CORRECTION),
                "output_tokens": round(sum(c["output_tokens"] for c in self.calls) * CORRECTION)}

    def show(self):
        print(json.dumps(self.report(), indent=2))


def main():
    parser = argparse.ArgumentParser(description="Estimate token usage for independent chat examples")
    parser.add_argument("input", type=Path, help='JSONL file, one {"messages": [...]} per line')
    parser.add_argument("--max-tokens", type=int, default=512, help="output cap per call")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    examples = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    est = TokenEstimator(max_tokens=args.max_tokens, log_path=args.log)
    for row in examples:
        est.chat(row["messages"])
    est.show()


if __name__ == "__main__":
    main()

# Token Estimator

Estimate how many input and output tokens a pipeline will use on a hosted LLM API, **before**
running it there. You run a small sample of your pipeline against `Qwen/Qwen2.5-14B-Instruct`
locally, and the tool records the token usage of every call.

Everything is in `token_estimator.py`: the `TokenEstimator` class and a CLI.

## Setup

```bash
pip install torch transformers
```

The first run downloads the model from Hugging Face (about 30 GB). It needs a GPU with about
30 GB of memory; on CPU it runs, but very slowly.

## Usage

### In Python

`est.chat(messages, max_tokens=None)` takes OpenAI-style messages and returns the reply text,
so the swap happens at the one place your code calls the model. Everything else stays the same.

<table>
<tr><th>Calling the API model</th><th>Calling the estimator</th></tr>
<tr>
<td>

```python
from openai import OpenAI

client = OpenAI()

def llm(messages, max_tokens=1024):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
```

</td>
<td>

```python
from token_estimator import TokenEstimator

est = TokenEstimator()

def llm(messages, max_tokens=1024):
    return est.chat(
        messages,
        max_tokens=max_tokens,
    )
```

</td>
</tr>
<tr>
<td>

```python
import ollama

def llm(messages, max_tokens=1024):
    response = ollama.chat(
        model="llama3.1:70b",
        messages=messages,
        options={"num_predict": max_tokens},
    )
    return response["message"]["content"]
```

</td>
<td>

```python
from token_estimator import TokenEstimator

est = TokenEstimator()

def llm(messages, max_tokens=1024):
    return est.chat(
        messages,
        max_tokens=max_tokens,
    )
```

</td>
</tr>
</table>

The rest of the pipeline doesn't change. Run it on a small sample, then print the totals:

```python
def summarize(document):
    messages = [
        {"role": "system", "content": "You summarize documents."},
        {"role": "user", "content": f"Summarize:\n\n{document}"},
    ]
    summary = llm(messages)
    messages += [
        {"role": "assistant", "content": summary},
        {"role": "user", "content": "Now list three key takeaways."},
    ]
    return llm(messages)

for document in documents[:20]:    # a small, representative sample
    summarize(document)

est.show()
```

Pass the same `max_tokens` you'll give the real API. `est.report()` returns the totals as a
dict, and `est.count(text_or_messages)` counts tokens without generating anything.

### From the command line

For independent single-call examples, put one `{"messages": [...]}` per line in a JSONL file:

```bash
python utils/token_estimator.py examples.jsonl --max-tokens 1024 --log calls.jsonl
```

### Options

| `TokenEstimator(...)` argument | Default | Meaning |
|---|---|---|
| `max_tokens` | `512` | Output cap per call, unless overridden in `chat()` |
| `device` | auto | `cuda`, `mps`, or `cpu` |
| `temperature` | `0.0` | `0` means greedy decoding |
| `log_path` | `None` | Optional JSONL file with one line per call |

## Reading the report

```json
{
  "calls": 40,
  "truncated_calls": 3,
  "input_tokens": 52000,
  "output_tokens": 18000
}
```

All numbers are totals over the sample. To estimate a full run, scale them by
`full run size / sample size`.

`input_tokens` and `output_tokens` are multiplied by `CORRECTION` (1.5 in `token_estimator.py`),
because Claude's tokenizer produces about 1.5× as many tokens as Qwen's for the same text. The
per-call rows in `est.calls` and the `log_path` file are raw Qwen counts. Gives a better estimate overall with the claude models I tried, but as you work with other api models you may get a better feel for this ratio for each one. It's just a constant in the script that you can edit.

- **`truncated_calls`** counts calls that hit `max_tokens`. A high number means the local model
  was cut off, so `output_tokens` is probably too low.

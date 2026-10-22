# Token Estimator

Estimate how many input and output tokens a pipeline will use on a hosted LLM API, **before**
running it there. You run a small sample of your pipeline against `Qwen/Qwen2.5-14B-Instruct`
locally, and the tool records the token usage of every call.

Everything is in `token_estimator.py`: the `TokenEstimator` class.

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

### Example: CRAFT

In CRAFT, each turn the three directors and the builder each call the model once. The swap goes
in the one method of each agent that makes the API call; prompts, parsing and game logic stay as
they are.

**Builder**: `BuilderAgent.generate_move` in `agents/builder_agent.py`

<table>
<tr><th>Calling the API model</th><th>Calling the estimator</th></tr>
<tr>
<td>

```python
completion = self.client.chat.completions.create(
    model=self.model_name,
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ],
    temperature=0.1,
    max_tokens=250
)

if completion and completion.choices:
    response_text = completion.choices[0].message.content.strip()
```

</td>
<td>

```python
response_text = self.est.chat(
    [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ],
    max_tokens=250,
).strip()
```

</td>
</tr>
</table>

**Directors**: `DirectorAgent._generate_with_openai` in `agents/director_agent.py`

<table>
<tr><th>Calling the API model</th><th>Calling the estimator</th></tr>
<tr>
<td>

```python
kwargs = dict(
    model=self.model_name,
    messages=[
        {"role": "system", "content": f"You are Director {self.director_id} in a collaborative LEGO construction task."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=max_tokens,
)
completion = self.client.chat.completions.create(**kwargs)
if completion and completion.choices:
    return completion.choices[0].message.content
```

</td>
<td>

```python
messages = [
    {"role": "system", "content": f"You are Director {self.director_id} in a collaborative LEGO construction task."},
    {"role": "user", "content": prompt}
]
return self.est.chat(messages, max_tokens=max_tokens)
```

</td>
</tr>
</table>

Then give the agents an estimator and play a few real games. A separate estimator per role
shows where the tokens go:

```python
from token_estimator import TokenEstimator

builder_est, director_est = TokenEstimator(), TokenEstimator()
builder_agent.est = builder_est
for director in director_agents.values():
    director.est = director_est

for idx, structure in enumerate(structures[:3]):    # a small, representative sample
    run_single_game(structure, director_agents, builder_agent, common_ground_agent,
                    correctness, structure_idx=idx, use_tools=False)

builder_est.show()
director_est.show()
```

Play whole games, not single prompts. Director prompts include the conversation so far, so later
turns cost more, and only a full game captures that. Use `use_tools=False`: the tool-using
builder (`generate_move_with_tools`) isn't supported.

`est.report()` returns the totals as a dict, and `est.count(text_or_messages)` counts tokens
without generating anything.

### Options

| `TokenEstimator(...)` argument | Default | Meaning |
|---|---|---|
| `max_tokens` | `512` | Output cap per call, unless overridden in `chat()` |
| `device` | auto | `cuda`, `mps`, or `cpu` |
| `temperature` | `0.0` | `0` means greedy decoding |

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
per-call rows in `est.calls` are raw Qwen counts.

- **`truncated_calls`** counts calls that hit `max_tokens`. A high number means the local model
  was cut off, so `output_tokens` is probably too low.

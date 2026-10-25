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
so the swap happens wherever your code calls the model.

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

## Examples

### Example 1: Clue Selector Game

`eval_est.py` is the Clue Selector Game's `eval.py` in the `echo-edp` [repo](https://github.com/csu-signal/echo-edp) with the estimator added
as commented-out blocks.

Each turn makes two model calls:

**Selector**: `make_frontier_generator` in `eval.py`, which `run_episode()` calls through
`generate_fn(prompt_text)`

<table>
<tr><th>The API call</th><th>The estimator call</th></tr>
<tr>
<td>

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant. Follow instructions exactly."},
    {"role": "user",   "content": prompt_text},
]
kwargs = dict(
    model                 = model_name,
    messages              = messages,
    max_completion_tokens = max_tokens,
)
kwargs["temperature"] = temperature
resp = client.chat.completions.create(**kwargs)
```

</td>
<td>

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant. Follow instructions exactly."},
    {"role": "user",   "content": prompt_text},
]
est.chat(messages, max_tokens=max_tokens)
```

</td>
</tr>
</table>

**Oracle**: `call_clue_holder` in `environment.py` (GPT-4o-mini)

<table>
<tr><th>The API call</th><th>The estimator call</th></tr>
<tr>
<td>

```python
response = client.chat.completions.create(
    model    = CLUE_MODEL,
    messages = [
        {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ],
    temperature = 0.0,
    max_tokens  = 60,
)
```

</td>
<td>

```python
est.chat(
    [
        {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ],
    max_tokens=60,
)
```

</td>
</tr>
</table>

The selector swap replaces the `generate_fn` that `make_frontier_generator` returns. The oracle
swap replaces the module-level `client` in `environment.py` with a stand-in that answers with the
estimator, so `call_clue_holder` itself is unchanged.

#### Running it

1. Copy `eval_est.py` into the `echo-edp` repo root, next to `eval.py`, and make
   `token_estimator.py` importable (copy it there too, or add this repo to `PYTHONPATH`).
2. Uncomment the five blocks between `# >>> token estimate` and `# <<< token estimate`
   (leave the two marker lines themselves as comments):

   | Location | Description |
   |---|---|
   | After `load_dotenv()` | Imports `TokenEstimator` and defines `EstimatorClient` (the oracle's stand-in client) and `make_estimator_generator` (the selector's) |
   | After the constants | Writes results to `eval_results_est/` instead of `eval_results/`, and samples 20 secrets for 1 run |
   | `run_eval()`, frontier branch | Swaps the selector's `generate_fn` for the estimator's |
   | `__main__`, before `run_eval(args)` | Creates one estimator, `EST`, and gives the oracle the stand-in client |
   | `__main__`, after `run_eval(args)` | Prints the totals and exits before the results tables, so `eval_results/aggregated_results.json` isn't overwritten |

3. Run the frontier evaluation:

   ```bash
   python eval_est.py --model-type frontier --model-name gpt-4o
   ```

The totals cover the 20 sampled secrets and 1 run. A full frontier evaluation uses every secret in
`canonical_secrets.json` and 3 runs, so scale by `(number of secrets × 3) / 20`. 

### Example 2: CRAFT

`run_craft_est.py` is CRAFT's `run_craft.py` with the estimator added as commented-out blocks.
Each agent's `client` is replaced with a stand-in that answers with the
estimator, so the call each agent already makes goes to the local model instead of the API.

These are the calls that get answered locally:

**Builder**: `BuilderAgent.generate_move` in `agents/builder_agent.py`

<table>
<tr><th>The agent's API call</th><th>The estimator call</th></tr>
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
```

</td>
<td>

```python
est.chat(
    [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ],
    max_tokens=250,
)
```

</td>
</tr>
</table>

**Directors**: `DirectorAgent._generate_with_openai` in `agents/director_agent.py`

<table>
<tr><th>The agent's API call</th><th>The estimator call</th></tr>
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
```

</td>
<td>

```python
est.chat(
    [
        {"role": "system", "content": f"You are Director {self.director_id} in a collaborative LEGO construction task."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=max_tokens,
)
```

</td>
</tr>
</table>

The stand-in also answers Claude directors (`_generate_with_anthropic`) and the common ground
agent, if it's on.

#### Running it

1. Copy `run_craft_est.py` into the CRAFT repo root, next to `run_craft.py`, and make
   `token_estimator.py` importable (copy it there too, or add this repo to `PYTHONPATH`).
2. Uncomment the four blocks between `# >>> token estimate` and `# <<< token estimate`
   (leave the two marker lines themselves as comments):

   | Location | Description |
   |---|---|
   | After the imports | Imports `TokenEstimator` and defines `EstimatorClient`, the stand-in client |
   | `run_craft_experiments()`, just before `run_single_game(...)` | Gives the builder, the directors and the common ground agent the stand-in client |
   | `__main__`, before the `# ── Run ──` section | Creates one estimator, `EST`, for the whole run |
   | `__main__`, before `print("All done.")` | Prints the totals |

3. Run a few structures with an OpenAI or Claude director model:

   ```bash
   python run_craft_est.py --mode api --director gpt-4.1-mini --builder gpt-4.1-mini \
       --structures 0,1,2
   ```

The estimator is created once for the whole run because each `TokenEstimator()` loads its own
copy of the model. Director prompts include the conversation so far, so later
turns cost more, and only a full game captures that.

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
per-call rows in `est.calls` are raw Qwen counts. Gives a better estimate overall with the claude models I tried, but as you work with other api models you may get a better feel for this ratio for each one. It's just a constant in the script that you can edit.

- **`truncated_calls`** counts calls that hit `max_tokens`. A high number means the local model
  was cut off, so `output_tokens` is probably too low.

## Estimating the cost

Once you've run this tool to estimate the token count and scaled them to your full run, you can plug those numbers in to find an estimate for the cost of your run as billed by the API providers. 


$$ \text{Estimated Cost} = \left( \frac{\text{Input Tokens}}{1,000,000} \times \text{Price}_{\text{Input/1M}} \right) + \left( \frac{\text{Output Tokens}}{1,000,000} \times \text{Price}_{\text{Output/1M}} \right) $$


To get the price rate values (input/output price per 1M tokens) for your model, follow the respective link for the billing chart of your model provider:

Anthropic: https://platform.claude.com/docs/en/about-claude/pricing

OpenAI: https://developers.openai.com/api/docs/pricing

Google: https://ai.google.dev/gemini-api/docs/pricing

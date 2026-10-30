# Token Estimator

Estimate how many input and output tokens a pipeline will use on a hosted LLM API, **before**
running it there. You run a small sample of your pipeline against `Qwen/Qwen2.5-14B-Instruct`
locally, and the tool records the token usage of every call.

Everything is in `token_estimator.py`: the `TokenEstimator` class, and `EstimatorClient`, a
stand-in for an OpenAI, Anthropic or Gemini client.

## Contents

- [Setup](#setup)
- [Using it with OpenAI, Anthropic or Gemini](#using-it-with-openai-anthropic-or-gemini)
  - [Clients created on every call](#clients-created-on-every-call)
  - [Not supported](#not-supported)
- [Using it with anything else](#using-it-with-anything-else)
- [Reading the report](#reading-the-report)
- [Estimating the cost](#estimating-the-cost)
- [Options](#options)
- [Examples](#examples)
  - [Clue Selector Game](#clue-selector-game)
  - [CRAFT](#craft)

## Setup

```bash
pip install torch transformers
```

The first run downloads the model from Hugging Face (about 30 GB). It needs a GPU with about
30 GB of memory.

## Using it with OpenAI, Anthropic or Gemini packages

If your code calls one of these providers through a client it keeps in an attribute or a module
variable, you don't have to touch the calls at all. Replace the client with an `EstimatorClient`,
run a sample of your pipeline, and print the totals:

```python
from token_estimator import TokenEstimator, EstimatorClient

est = TokenEstimator()
agent.client = EstimatorClient(est)    # replacing OpenAI(), anthropic.Anthropic() or genai.Client()

run_pipeline(sample)

est.show()
```

It answers all three call styles with the local model:

| Provider | Call | Response fields |
|---|---|---|
| OpenAI | `client.chat.completions.create(...)` | `choices[0].message.content`, `finish_reason`, `usage` |
| Anthropic | `client.messages.create(...)` | `content` text blocks, `stop_reason`, `usage` |
| Gemini (`google-genai`) | `client.models.generate_content(...)` | `text`, `candidates[0].content.parts`, `finish_reason`, `usage_metadata` |

Other arguments, such as `model` and `temperature`, are ignored. Every call made through an
`EstimatorClient(est)` is recorded on the same `est`, so one estimator covers all your agents and
providers. Create it once: each `TokenEstimator()` loads its own copy of the model.

### Clients created on every call

If your code creates a new client on every call instead of keeping one, replace the class itself,
so each new "client" is an `EstimatorClient`:

```python
from google import genai

genai.Client = lambda *args, **kwargs: EstimatorClient(est)
```

## Using it with anything else

For Ollama, a different provider's model, your own wrapper, or any other unsupported calls, call
`est.chat(messages, max_tokens=None)` where your code calls the model. It takes OpenAI-style
messages and returns the reply text.

<table>
<tr><th>Calling the model</th><th>Calling the estimator</th></tr>
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



## Reading the report

`est.show()` prints the totals over every recorded call; `est.report()` returns the same thing as
a dict.

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
because Claude's tokenizer produces about 1.5× as many tokens as Qwen's for the same text. This
gives a better estimate overall with the Claude models I tried, but as you work with other API
models you may get a better feel for this ratio for each one. It's just a constant in the script
that you can edit.

- **`truncated_calls`** counts calls that hit `max_tokens`. A high number means the local model
  was cut off, so `output_tokens` is probably too low.

Per-call counts are also available, but they are raw Qwen counts without the correction:
`est.calls` has one row per call, and each `EstimatorClient` response carries its call's counts in
`usage` (`usage_metadata` for Gemini).

## Estimating the cost

Once you've scaled the token counts to your full run, plug them in to estimate the cost as billed
by the API provider:

$$ \text{Estimated Cost} = \left( \frac{\text{Input Tokens}}{1,000,000} \times \text{Price}_{\text{Input/1M}} \right) + \left( \frac{\text{Output Tokens}}{1,000,000} \times \text{Price}_{\text{Output/1M}} \right) $$

The input and output prices per 1M tokens for your model are on your provider's pricing page:

- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
- OpenAI: https://developers.openai.com/api/docs/pricing
- Google: https://ai.google.dev/gemini-api/docs/pricing

## Options

| `TokenEstimator(...)` argument | Default | Meaning |
|---|---|---|
| `max_tokens` | `512` | Output cap per call, unless overridden in `chat()` |
| `device` | auto | `cuda`, `mps`, or `cpu` |
| `temperature` | `0.0` | `0` means greedy decoding |

## Examples

Both examples are copies of a lab script with the estimator added as commented-out blocks between
`# >>> token estimate` and `# <<< token estimate`. To run one:

1. Copy the example into the target repo root, next to the original script, and make
   `token_estimator.py` importable (copy it there too, or add this repo to `PYTHONPATH`).
2. Uncomment the blocks between the markers (leave the two marker lines themselves as comments).
3. Run it with the command below.

### Clue Selector Game

`examples/eval_est.py` is the Clue Selector Game's `eval.py` in the
[`echo-edp` repo](https://github.com/csu-signal/echo-edp). Each turn makes two model calls, and it
uses both approaches:

- **Oracle** (`call_clue_holder` in `environment.py`): the module-level `client` is replaced with
  an `EstimatorClient`, so the call itself is unchanged.
- **Selector** (`generate_fn` in `make_frontier_generator`): it creates a new client on every
  call, so it returns `EST.chat(messages, ...)` right after the messages are built, before the
  OpenAI or Claude branch runs.

| Block | Description |
|---|---|
| After `load_dotenv()` | Imports `TokenEstimator` and `EstimatorClient` |
| After the constants | Writes results to `eval_results_est/` instead of `eval_results/`, and samples 20 secrets for 1 run |
| `make_frontier_generator()`, in `generate_fn` | Answers the selector with the estimator instead of the API |
| `__main__`, before `run_eval(args)` | Creates one estimator, `EST`, and gives the oracle an `EstimatorClient` |
| `__main__`, after `run_eval(args)` | Prints the totals and exits before the results tables, so `eval_results/aggregated_results.json` isn't overwritten |

```bash
python eval_est.py --model-type frontier --model-name gpt-4o
```

The totals cover the 20 sampled secrets and 1 run. A full frontier evaluation uses every secret in
`canonical_secrets.json` and 3 runs, so scale by `(number of secrets × 3) / 20`.

### CRAFT

`examples/run_craft_est.py` is `run_craft.py` in the
[CRAFT repo](https://github.com/csu-signal/CRAFT/tree/main). The builder, the directors and the
common ground agent each get an `EstimatorClient`, so OpenAI, Claude and Gemini directors all work
unchanged. Gemini directors create a new `genai.Client()` on every call, so that class is
[replaced](#clients-created-on-every-call).

| Block | Description |
|---|---|
| After the imports | Imports `TokenEstimator` and `EstimatorClient` |
| `run_craft_experiments()`, just before `run_single_game(...)` | Gives the builder, the directors and the common ground agent an `EstimatorClient` |
| `__main__`, before the `# ── Run ──` section | Creates one estimator, `EST`, for the whole run, and makes `genai.Client()` return an `EstimatorClient` |
| `__main__`, before `print("All done.")` | Prints the totals |

```bash
python run_craft_est.py --mode api --director gpt-4.1-mini --builder gpt-4.1-mini \
    --structures 0,1,2
```

Director prompts include the conversation so far, so later turns cost more, and only a full game
captures that.

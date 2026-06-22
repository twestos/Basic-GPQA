# GPQA Benchmark Practice

This repository is a small practice implementation of an LLM benchmark for comparing model quality, speed, and cost.

The goal is to explore the mechanics of model evaluation: loading a benchmark dataset, formatting prompts, making streamed API requests, extracting answers, and recording performance and cost metrics.

The implementation uses the GPQA Diamond dataset as a starting point because it is a useful benchmark that is relatively simple to wire up, affordable to run, and still gives interesting signal across model quality and performance.

This is intentionally not a polished benchmarking framework. It is a compact learning project focused on the core pieces of a benchmark harness.

## What It Measures

For each model run, the benchmark records:

- `pass_at_1`: whether the model's first answer matches the correct multiple-choice answer.
- `ttft`: time to first token from the streaming response.
- `output_speed`: approximate output tokens per second.
- `input_tokens` and `output_tokens`: both client-estimated and provider-reported where available.
- `cost`: provider-reported cost from OpenRouter usage metadata.

Summary metrics are written as JSONL so that individual repeats can be inspected or aggregated later.

## Why This Shape

I chose to implement the model calls directly with `httpx` rather than using a higher-level LLM SDK. The goal was to get more comfortable with the lower-level mechanics of working with LLM APIs: request payloads, streamed Server-Sent Events, retryable failures, usage metadata, and timing measurements.

Direct API integration is also useful when working across different providers, where the exact request/response shape, streaming behavior, and metadata can matter.

The implementation favours readability and explicitness over abstraction. There is still plenty to improve, but the core flow is in place: load questions, call the model, parse the answer, calculate metrics, and write results.

## Project Structure

```text
.
├── main.py                  # Runs GPQA Diamond repeats and writes result summaries
├── eval/
│   ├── client.py            # OpenRouter streaming client built with httpx
│   ├── gpqa.py              # GPQA evaluation runner
│   ├── types.py             # Dataclasses for questions, responses, and results
│   └── utils.py             # Dataset loading, prompt formatting, answer extraction
├── results/                 # Per-model repeat-level result files
├── results_summary.jsonl    # Aggregate pass@1 summaries
└── pyproject.toml
```

## Setup

This project uses Python 3.13 and the dependencies in `pyproject.toml`.

Install dependencies:

```bash
uv sync
```

Create a local environment file:

```bash
cp .env.example .env
```

Then set:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

## Running

By default, `main.py` runs five repeats of GPQA Diamond against `google/gemini-3.1-flash-lite`:

```bash
uv run python main.py
```

The runner appends repeat-level metrics to:

```text
results/results_<provider>_<model>.jsonl
```

It also appends an aggregate pass@1 summary to:

```text
results_summary.jsonl
```

To test another model, change the `model` argument passed to `run_gpqa` in `main.py`.

## Current Sample Results

These are the saved aggregate results from the current repo state. Each row is five repeats over GPQA Diamond, for 990 total attempts.

| Model | pass@1 | Correct / Total |
| --- | ---: | ---: |
| `google/gemini-3.1-flash-lite` | 72.42% | 717 / 990 |
| `openai/gpt-5.4-mini` | 61.31% | 607 / 990 |
| `openai/gpt-5.4-nano` | 56.06% | 555 / 990 |

The per-repeat files also include latency, throughput, and cost metrics such as p50/p75/p90/p95/p99 TTFT, output tokens per second, total cost, and cost per question.

## Current Limitations

There is a lot of room to make this more robust:

- Randomised answer ordering is not currently seeded, so exact prompt variants are not reproducible across runs.
- The runner is configured in code rather than via a CLI.
- Results are append-only JSONL files with no separate analysis script yet.
- Answer extraction is regex-based and could be made more rigorous.
- Token counting uses a local tokenizer estimate, which may not match every provider/model exactly.
- There are no automated tests yet around dataset processing, answer extraction, or metrics aggregation.
- The benchmark currently focuses on GPQA Diamond only.

## Improvements I Would Make Next

If I continued developing this, I would likely add:

- A CLI for selecting benchmark variant, model, repeat count, concurrency, and output directory.
- Deterministic shuffling with stored seeds so runs are easier to reproduce.
- A separate analysis script for comparing models across quality, latency, throughput, and cost.
- Tests for the answer parser and metric calculations.
- Better run metadata, including provider, model settings, dataset version, concurrency, and timestamp.
- Support for additional benchmark datasets once the basic harness is cleaner.

## Notes

This repo is meant to be a concrete starting point rather than a finished product. It is useful for experimenting with benchmark mechanics and thinking through the trade-offs involved in making model evaluation results reproducible, comparable, and useful.

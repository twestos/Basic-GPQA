# Legal Chronology Eval

`eval/legal_chrono.py` runs a one-shot benchmark for synthetic legal matter
chronology questions. It measures how well a candidate model can answer a
plain-English legal matter question using only a supplied set of reference
files, then uses a separate grader model to score the answer against a gold
answer and rubric.

The eval is designed for questions where the hard part is tracking a changing
matter file: chronology, contradictions, source authority, live issues,
deadlines, party knowledge, and the current matter state.

## Prerequisites

Install the project dependencies:

```bash
uv sync
```

Set an OpenRouter API key in `.env` or in your shell:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

The runner downloads task rows and reference files from Hugging Face with
`huggingface_hub.hf_hub_download`. If the dataset repo is private, make sure
your Hugging Face credentials are available to the Hugging Face Hub client.

## Quick Start

Run the full default benchmark:

```bash
uv run python -m eval.legal_chrono
```

The defaults are:

- Dataset repo: `twestoss/legal-matter-chrono-bench`
- Dataset revision: `main`
- Split path: `data/train.jsonl`
- Candidate model: `qwen/qwen3.7-plus`
- Grader model: `openai/gpt-5.4`
- Max concurrency: `4`
- Output directory: `results`

Run a specific model:

```bash
uv run python -m eval.legal_chrono \
  --model openai/gpt-5.4-mini \
  --grader-model openai/gpt-5.4 \
  --max-concurrency 8
```

Preview the first prompt without making model calls:

```bash
uv run python -m eval.legal_chrono --limit 1 --dry-run
```

## How The Eval Works

Each run follows the same pipeline:

1. Load task rows from the Hugging Face dataset split.
2. Optionally filter tasks by domain, category, matter id, offset, or limit.
3. Download the task's listed reference files from the same dataset repo.
4. Build an answer prompt containing task metadata, the question, and the
   reference files.
5. Ask the candidate model to answer in plain English using only those
   reference files.
6. Build a grader prompt containing the question, gold answer, task rubric, and
   candidate answer.
7. Ask the grader model to return `Score: <integer 0-5>` and a short rationale.
8. Parse the grader score, normalize it to `0.0-1.0`, and mark the task as
   passing when the score is at least `4`.
9. Write per-task JSONL results and aggregate summary metrics.

Tasks are evaluated concurrently with an `asyncio.Semaphore` controlled by
`--max-concurrency`.

## Dataset Shape

The eval expects each row in `data/train.jsonl` to include the fields used by
`LegalChronoTask`, especially:

- `task_id`
- `domain`
- `matter_id`
- `matter_title`
- `category`
- `prompt`
- `reference_files`
- `reference_file_hf_uris`
- `scoring_rubric`
- `gold_answer` or `deliverable_text`

`reference_files` should contain paths inside the same Hugging Face dataset
repo. Those files are inserted into the answer prompt as tagged blocks:

```text
<reference_file path="...">
...
</reference_file>
```

## Prompt Size Controls

Reference file content is loaded with two character budgets:

- `--max-reference-chars` caps the total reference text for one task.
- `--max-file-chars` caps each individual reference file.

Defaults are `120000` total characters and `30000` characters per file. If a
file is truncated, the prompt includes a truncation note and the result row
records the file path in `reference_files_truncated`.

## Grading

The grader scores semantic correctness, not style. It is instructed to reward
answers that correctly handle chronology, contradictions, source authority,
live issues, and current matter state. It should penalize unsupported claims,
stale facts presented as current, missed material caveats, and wrong dates or
parties.

Scores mean:

- `5`: fully correct; captures all material points, chronology, caveats, and
  current-state implications.
- `4`: mostly correct; minor omissions or imprecision, no material
  contradiction.
- `3`: partially correct; captures the broad issue but misses important
  evidence, timing, or caveats.
- `2`: weak; some relevant facts, but substantial omissions or confused
  chronology.
- `1`: minimally relevant; mostly incorrect or unsupported.
- `0`: irrelevant, empty, or fundamentally wrong.

The parsed score is valid when the grader response contains a score in one of
the accepted formats, such as `Score: 4`, `score is 4`, or `4/5`.

By default, the grader sees the question, gold answer, rubric, and candidate
answer, but not the full reference files. Pass `--include-references-in-grader`
to include the reference blocks in the grader prompt too.

## Outputs

For each run, the eval writes:

- `results/<run-name>.jsonl`: one JSON object per task, including prompt,
  references, gold answer, model answer, grader response, score, latency,
  token counts, and cost fields.
- `results/<run-name>_summary.json`: aggregate metrics for that run.
- `results/legal_chrono_summary.jsonl`: append-only summary history across
  runs.

If `--run-name` is omitted, the run name is generated from the candidate model
and current timestamp:

```text
legal_chrono_<provider>_<model>_<YYYYMMDD_HHMMSS>
```

The summary includes:

- `average_score` on the raw `0-5` scale.
- `average_normalized_score` on the `0.0-1.0` scale.
- `pass_at_4`, the share of tasks scoring at least `4`.
- `valid_score_count`, the number of grader responses with parseable scores.
- Total and per-task cost.
- TTFT and output-speed percentiles for both answer and grader calls.
- Client-estimated and provider-reported token totals.
- Breakdowns by `domain` and `category`.

## Useful CLI Options

Filter the run:

```bash
uv run python -m eval.legal_chrono \
  --domain employment_dispute \
  --category chronology \
  --limit 10
```

Run a single matter:

```bash
uv run python -m eval.legal_chrono --matter-id employment_matter_002
```

Use a different dataset repo or revision:

```bash
uv run python -m eval.legal_chrono \
  --repo-id <user>/<dataset-repo> \
  --revision <branch-or-commit>
```

Control output names:

```bash
uv run python -m eval.legal_chrono \
  --output-dir results/legal_chrono \
  --run-name qwen_plus_baseline
```

## Interpreting Results

Use `pass_at_4` as the main coarse quality metric: it asks whether the answer
is good enough to count as mostly correct. Use `average_score` when you want
more gradation between partially correct and fully correct answers.

Latency and throughput are measured separately for the candidate answer call
and the grader call. `answer_ttft` and `answer_output_speed` are usually the
most relevant model-serving metrics when comparing candidate models.

Cost is reported from OpenRouter usage metadata when available. The local token
counts are estimates from the repository tokenizer utility and may not exactly
match provider-reported usage for every model.

## Current Limitations

- The grader is itself an LLM, so scores should be treated as benchmark signal,
  not ground truth.
- The eval does not currently retry or repair invalid grader responses beyond
  parsing several common score formats.
- Model temperature and other generation settings are not exposed; the client
  sends only model, messages, and `stream: true`.
- Hugging Face downloads are performed per task/reference path and rely on the
  local Hugging Face cache for reuse.
- The candidate model receives truncated references if the configured character
  budgets are exceeded.

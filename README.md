# prompt-eval-arena

A small, **rigorous** A/B harness for prompt iteration. Supports both **deterministic scoring** (for narrow ground-truth tasks) and **LLM-as-judge** (for open-ended outputs). Reports whether the difference between two prompts is statistically significant — instead of eyeballing 5 examples and shipping.

```
$ arena ab \
    --a prompts/v1_simple.txt \
    --b prompts/v2_chain_of_thought.txt \
    --dataset prompts/eval_set.jsonl \
    --runs 3 \
    --model openai:gpt-4o-mini

A/B RESULTS · n=80 · 3 runs each · gpt-4o-mini
─────────────────────────────────────────────────────
Variant A (v1_simple):              0.730  ± 0.012
Variant B (v2_chain_of_thought):    0.812  ± 0.009

Lift: +0.082  (p < 0.001 · paired Wilcoxon)
Effect size: Cohen's d = 1.41 (large)
Variance reduced by replicate runs: 4.2x

VERDICT: SIGNIFICANT — ship v2.
```

## Why this exists

Most prompt iteration looks like:

> Engineer changes a prompt, tests it on a handful of examples, "it seems better," ships.

This produces silent regressions. The right setup is:

- A **held-out eval set** with stable rubric or ground truth.
- A **paired comparison** (same example, both prompts).
- **Multiple sampling runs** to estimate per-example variance.
- A **significance test** (Wilcoxon signed-rank for paired non-normal data).
- An **effect size** alongside the p-value.

`prompt-eval-arena` does all of that in one CLI command, with prompt versioning so you can compare any two prompts in your history.

## Two scoring modes

### Deterministic scorers — *default; use when you can*

When ground truth is a label, a number, or a regex match, deterministic scoring is **fast, cheap, and free of judge bias**. Built-in:

| Scorer | Use for |
|---|---|
| `exact_match` | One canonical answer |
| `label_match` | Answer in a small synonym set |
| `numeric_within` | Numeric answer with tolerance |
| `contains` | Required phrase or substring |

### LLM-as-judge — *use when you must*

Some outputs do not have a ground-truth string: summarization quality, tone, "is this email professional," "does this answer cover all 5 points." For those, use `llm_judge`.

```jsonl
{"input": "Write a 2-sentence apology email for a service outage.", "scorer": "llm_judge", "rubric": ["acknowledges the issue specifically", "takes responsibility without blaming users", "states a concrete next step", "stays within 2 sentences"]}
```

The judge model is configurable via `ARENA_JUDGE_MODEL` (default: `openai:gpt-4o-mini`).

**Judge mode caveats — read these:**

1. **Use a stronger model as judge than as student.** If you're A/B-testing two prompts on `gpt-4o-mini`, judge with `gpt-4o` or `claude-3-5-sonnet`.
2. **Order matters.** Judges have position bias. The arena randomly swaps A/B order per example when running judge mode to reduce this.
3. **Don't compare a model to itself as judge.** "GPT-4o judges its own output" is a red flag in any paper. Use a different family for the judge.
4. **Calibrate before trusting.** Run the judge against a labeled subset and verify human-judge agreement on at least 30 examples before relying on aggregate scores.

When in doubt, write a 5-line deterministic check instead. LLM-as-judge is an escape hatch, not a default.

## Quickstart

```bash
pip install -e .

# initialize an arena in the current directory
arena init my-eval

# deterministic A/B
arena ab \
    --a prompts/v1.txt \
    --b prompts/v2.txt \
    --dataset prompts/eval_set.jsonl \
    --model openai:gpt-4o-mini \
    --runs 3

# LLM-as-judge A/B (rubric in the eval set rows)
export ARENA_JUDGE_MODEL=openai:gpt-4o
arena ab \
    --a prompts/email_v1.txt \
    --b prompts/email_v2.txt \
    --dataset prompts/email_rubrics.jsonl \
    --model openai:gpt-4o-mini

# best-of-K from a directory of prompt candidates
arena tournament prompts/ --dataset prompts/eval_set.jsonl
```

## Eval set format

JSONL, one example per line. The `scorer` field picks which scoring function runs.

**Deterministic example:**
```json
{"input": "Compute compound interest: $10000 at 5% for 10 years.", "target": "16289", "scorer": "numeric_within", "tol": 10}
```

**LLM-as-judge example:**
```json
{"input": "Summarize the article in 3 bullets.", "scorer": "llm_judge", "rubric": ["covers the main thesis", "captures one numerical fact", "fits in 3 bullets"]}
```

## Statistical method

For paired comparisons (default), we use the **Wilcoxon signed-rank test** on per-example score differences. With multiple runs, we average per-example scores first, then test.

For unpaired comparisons (different eval sets per variant), we use **Mann-Whitney U**.

Effect sizes reported as Cohen's d for continuous scores; rank-biserial correlation for ordinal.

Variance estimation: across replicate runs, computes the noise floor below which any A/B difference is meaningless.

## What this is not

- Not a hosted leaderboard. Local-first.
- Not a substitute for a real eval framework — see [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai), [promptfoo](https://github.com/promptfoo/promptfoo), [Braintrust](https://www.braintrustdata.com). prompt-eval-arena is intentionally small and focused on the statistical-rigor part.
- Not a fine-tuning pipeline.

## License

MIT.

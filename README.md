# prompt-eval-arena

A small, **rigorous** A/B harness for prompt iteration. Run two prompt variants on the same dataset and report whether the difference is statistically significant — instead of eyeballing 5 examples and shipping.

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

- A **held-out eval set** with stable ground truth.
- A **paired comparison** (same example, both prompts).
- **Multiple sampling runs** to estimate per-example variance.
- A **significance test** (Wilcoxon signed-rank for paired non-normal data).
- An **effect size** alongside the p-value.

`prompt-eval-arena` does all of that in one CLI command, with prompt versioning so you can compare any two prompts in your history.

## Quickstart

```bash
pip install -e .

# initialize an arena in the current directory
arena init my-eval

# add prompts and an eval set
arena ab \
    --a prompts/v1.txt \
    --b prompts/v2.txt \
    --dataset prompts/eval_set.jsonl \
    --model openai:gpt-4o-mini \
    --runs 3

# compare any two saved runs
arena diff runs/2026-04-01_v1.json runs/2026-04-02_v2.json

# best-of-K from a directory of prompt candidates
arena tournament prompts/ --dataset prompts/eval_set.jsonl
```

## Eval set format

JSONL, one example per line:

```json
{"input": "What is the term sheet liquidation preference?", "target": "1x non-participating", "scorer": "label_match", "labels": ["1x non-participating", "1x, non-participating"]}
{"input": "...", "target": "...", "scorer": "numeric_within", "tol": 0.01}
```

Bundled scorers: `exact_match`, `label_match`, `numeric_within`, `contains`. Drop in your own as Python functions under `arena/scorers.py`.

## Statistical method

For paired comparisons (default), we use the **Wilcoxon signed-rank test** on per-example score differences. With multiple runs, we average per-example scores first, then test.

For unpaired comparisons (different eval sets per variant), we use **Mann-Whitney U**.

Effect sizes reported as Cohen's d for continuous scores; rank-biserial correlation for ordinal.

Variance estimation: across replicate runs, computes the noise floor below which any A/B difference is meaningless.

## What this is not

- Not a hosted leaderboard. Local-first.
- Not an LLM-as-judge framework. Use [promptfoo](https://github.com/promptfoo/promptfoo) or [Braintrust](https://www.braintrustdata.com) for that. They are complementary — use this for narrow, deterministically-scoreable tasks where you want statistical rigor.
- Not a fine-tuning pipeline.

## License

MIT.

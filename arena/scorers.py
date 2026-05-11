"""Scorers — both deterministic and LLM-as-judge.

Default to deterministic for narrow, ground-truth-able tasks.
Use LLM-as-judge for open-ended generation where ground truth is a rubric,
not a label.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable

from openai import OpenAI


# ──────────────────────────────────────────────────────────────────────────
# Deterministic scorers
# ──────────────────────────────────────────────────────────────────────────


def exact_match(pred: str, target, **kwargs) -> float:
    return 1.0 if pred.strip().lower() == str(target).strip().lower() else 0.0


def label_match(pred: str, target, labels: list[str] | None = None, **kwargs) -> float:
    options = labels or [target]
    p = pred.strip().lower()
    for o in options:
        if str(o).lower() in p:
            return 1.0
    return 0.0


def numeric_within(pred: str, target, tol: float = 0.01, **kwargs) -> float:
    nums = re.findall(r"-?\d+\.?\d*", pred)
    if not nums:
        return 0.0
    try:
        value = float(nums[0])
    except ValueError:
        return 0.0
    return 1.0 if abs(value - float(target)) <= float(tol) else 0.0


def contains(pred: str, target, **kwargs) -> float:
    return 1.0 if str(target).lower() in pred.lower() else 0.0


# ──────────────────────────────────────────────────────────────────────────
# LLM-as-judge
# ──────────────────────────────────────────────────────────────────────────


JUDGE_PROMPT = """You are evaluating a model response against a rubric.

INPUT TO THE MODEL:
{input}

MODEL RESPONSE:
{pred}

RUBRIC:
{rubric}

For each rubric criterion, decide whether the response satisfies it.
Reply with strict JSON only:

{{
  "criteria": {{
    "<criterion_name>": {{"pass": true, "reason": "..."}},
    ...
  }},
  "overall_score": 0.0
}}

overall_score is the fraction of criteria that pass, 0.0 to 1.0.
Be strict. If unsure, mark as not passing."""


def _judge_client() -> tuple[OpenAI, str]:
    spec = os.environ.get("ARENA_JUDGE_MODEL", "openai:gpt-4o-mini")
    if spec.startswith("local:") or spec == "local":
        return (
            OpenAI(
                base_url=os.environ.get("ARENA_BASE_URL", "http://localhost:8000/v1"),
                api_key="local",
            ),
            "chat" if spec == "local" else spec.split(":", 1)[1],
        )
    if spec.startswith("openai:"):
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "")), spec.split(":", 1)[1]
    if spec.startswith("anthropic:"):
        return (
            OpenAI(
                base_url="https://api.anthropic.com/v1",
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            ),
            spec.split(":", 1)[1],
        )
    raise ValueError(f"unknown judge model spec: {spec}")


def llm_judge(pred: str, target, rubric: list[str] | str | None = None, input: str = "", **kwargs) -> float:
    """LLM-as-judge scorer.

    Use when:
      - The task is open-ended (summarization, rewriting, tone, "is this helpful").
      - Ground truth is a rubric, not a single label.

    Avoid when:
      - You can write a 5-line deterministic check instead.
      - Your eval set is small and one bad judgment moves the aggregate materially.

    `rubric` can be a list of criteria strings or a single description string.
    The judge returns an overall_score in [0, 1] used as the example score.
    """
    if rubric is None:
        rubric = [str(target)]
    if isinstance(rubric, list):
        rubric_str = "\n".join(f"- {c}" for c in rubric)
    else:
        rubric_str = str(rubric)

    client, model = _judge_client()
    body = JUDGE_PROMPT.format(input=input or kwargs.get("input", ""), pred=pred[:2000], rubric=rubric_str)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": body}],
        max_tokens=500,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or "{}"
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return 0.0
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return 0.0
    return float(data.get("overall_score", 0.0))


# ──────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────


REGISTRY: dict[str, Callable] = {
    "exact_match": exact_match,
    "label_match": label_match,
    "numeric_within": numeric_within,
    "contains": contains,
    "llm_judge": llm_judge,
}


def score(example: dict, completion: str) -> float:
    scorer_name = example.get("scorer", "exact_match")
    fn = REGISTRY[scorer_name]
    return fn(completion, target=example.get("target", ""), **example)

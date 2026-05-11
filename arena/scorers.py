"""Deterministic scorers."""

from __future__ import annotations

import re


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


REGISTRY = {
    "exact_match": exact_match,
    "label_match": label_match,
    "numeric_within": numeric_within,
    "contains": contains,
}


def score(example: dict, completion: str) -> float:
    scorer_name = example.get("scorer", "exact_match")
    fn = REGISTRY[scorer_name]
    return fn(completion, target=example["target"], **example)

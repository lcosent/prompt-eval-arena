"""Run a single prompt across an eval set, possibly N times for variance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .model import call
from .scorers import score


@dataclass
class VariantRun:
    label: str
    prompt: str
    model: str
    per_example_scores: list[list[float]] = field(default_factory=list)  # [run][example]
    completions: list[list[str]] = field(default_factory=list)

    def mean_per_example(self) -> list[float]:
        if not self.per_example_scores:
            return []
        n = len(self.per_example_scores[0])
        return [sum(r[i] for r in self.per_example_scores) / len(self.per_example_scores) for i in range(n)]

    def aggregate(self) -> float:
        means = self.mean_per_example()
        return sum(means) / max(len(means), 1)


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def run_variant(
    label: str, prompt_template: str, dataset: list[dict], model: str, runs: int = 1
) -> VariantRun:
    variant = VariantRun(label=label, prompt=prompt_template, model=model)
    for _ in range(runs):
        per_example: list[float] = []
        completions: list[str] = []
        for ex in dataset:
            rendered = prompt_template.format(**ex)
            completion = call(model, rendered)
            completions.append(completion)
            per_example.append(score(ex, completion))
        variant.per_example_scores.append(per_example)
        variant.completions.append(completions)
    return variant

"""Statistical comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    mean_a: float
    mean_b: float
    lift: float
    p_value: float
    cohens_d: float
    test_name: str
    significant: bool

    def verdict(self) -> str:
        if not self.significant:
            return "NOT SIGNIFICANT"
        winner = "B" if self.lift > 0 else "A"
        magnitude = (
            "small" if abs(self.cohens_d) < 0.5
            else "medium" if abs(self.cohens_d) < 0.8
            else "large"
        )
        return f"SIGNIFICANT (winner: {winner}, effect size {magnitude})"


def paired_compare(scores_a: list[float], scores_b: list[float], alpha: float = 0.05) -> ComparisonResult:
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    diff = b - a
    if np.allclose(diff, 0):
        return ComparisonResult(
            mean_a=float(a.mean()),
            mean_b=float(b.mean()),
            lift=0.0,
            p_value=1.0,
            cohens_d=0.0,
            test_name="paired",
            significant=False,
        )
    try:
        stat, p = stats.wilcoxon(b, a, zero_method="zsplit", alternative="two-sided")
    except ValueError:
        stat, p = float("nan"), 1.0
    # Cohen's d on paired data
    diff_sd = diff.std(ddof=1) if len(diff) > 1 else 1e-9
    d = float(diff.mean() / max(diff_sd, 1e-9))
    return ComparisonResult(
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        lift=float(b.mean() - a.mean()),
        p_value=float(p),
        cohens_d=d,
        test_name="paired_wilcoxon",
        significant=bool(p < alpha),
    )


def variance_summary(per_run: list[list[float]]) -> dict:
    """Across replicate runs, what fraction of variance is between runs vs within examples?"""
    if not per_run or len(per_run) < 2:
        return {"replicate_std": 0.0, "n_runs": len(per_run)}
    arr = np.array(per_run)
    per_example_means = arr.mean(axis=0)
    between_run_std = float(arr.mean(axis=1).std(ddof=1))
    return {
        "replicate_std": between_run_std,
        "per_example_mean_std": float(per_example_means.std(ddof=1)) if len(per_example_means) > 1 else 0.0,
        "n_runs": len(per_run),
    }

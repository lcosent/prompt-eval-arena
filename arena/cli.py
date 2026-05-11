from __future__ import annotations

import json
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import run as runner, stats as stx


console = Console()


@click.group()
def main():
    """prompt-eval-arena: rigorous A/B harness for prompt iteration."""


@main.command("init")
@click.argument("name")
def init_cmd(name: str):
    root = Path(name)
    root.mkdir(exist_ok=True)
    (root / "prompts").mkdir(exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "prompts" / "v1.txt").write_text("Question: {input}\nAnswer:")
    (root / "prompts" / "v2.txt").write_text(
        "You are an expert. Think step by step before answering.\n\nQuestion: {input}\nAnswer:"
    )
    (root / "prompts" / "eval_set.jsonl").write_text(
        '{"input": "2 + 2", "target": "4", "scorer": "label_match", "labels": ["4"]}\n'
    )
    console.print(f"[green]initialized arena[/green]: {root}/")


@main.command("ab")
@click.option("--a", "prompt_a", required=True)
@click.option("--b", "prompt_b", required=True)
@click.option("--dataset", required=True)
@click.option("--model", default="local")
@click.option("--runs", default=1, show_default=True)
@click.option("--alpha", default=0.05, show_default=True)
def ab_cmd(prompt_a: str, prompt_b: str, dataset: str, model: str, runs: int, alpha: float):
    pa = Path(prompt_a).read_text()
    pb = Path(prompt_b).read_text()
    ds = runner.load_dataset(Path(dataset))

    console.print(f"[dim]running {len(ds)} examples × {runs} runs on {model}...[/dim]")
    va = runner.run_variant(Path(prompt_a).stem, pa, ds, model, runs=runs)
    vb = runner.run_variant(Path(prompt_b).stem, pb, ds, model, runs=runs)

    result = stx.paired_compare(va.mean_per_example(), vb.mean_per_example(), alpha=alpha)
    var = stx.variance_summary(va.per_example_scores)

    table = Table(title=f"A/B results · n={len(ds)} · {runs} runs · {model}")
    table.add_column("variant")
    table.add_column("score", justify="right")
    table.add_row(va.label, f"{result.mean_a:.4f}")
    table.add_row(vb.label, f"{result.mean_b:.4f}")
    console.print(table)
    console.print()
    sign = "+" if result.lift >= 0 else ""
    console.print(f"Lift: {sign}{result.lift:.4f}  (p = {result.p_value:.4f} · {result.test_name})")
    console.print(f"Effect size: Cohen's d = {result.cohens_d:.2f}")
    console.print(f"Replicate-run std on A: {var['replicate_std']:.4f}  (n_runs={var['n_runs']})")
    console.print(f"\n[bold]Verdict:[/bold] {result.verdict()}")

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    out = runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_ab.json"
    out.write_text(
        json.dumps(
            {
                "model": model,
                "n": len(ds),
                "runs": runs,
                "a_label": va.label,
                "b_label": vb.label,
                "a_per_example": va.mean_per_example(),
                "b_per_example": vb.mean_per_example(),
                "comparison": result.__dict__,
                "variance": var,
            },
            indent=2,
        )
    )
    console.print(f"\n[dim]saved → {out}[/dim]")


@main.command("tournament")
@click.argument("prompts_dir")
@click.option("--dataset", required=True)
@click.option("--model", default="local")
def tournament_cmd(prompts_dir: str, dataset: str, model: str):
    """Run all prompts in a directory; rank by score."""
    ds = runner.load_dataset(Path(dataset))
    prompts = sorted(Path(prompts_dir).glob("*.txt"))
    leaderboard = []
    for p in prompts:
        text = p.read_text()
        v = runner.run_variant(p.stem, text, ds, model, runs=1)
        leaderboard.append((p.stem, v.aggregate()))
    leaderboard.sort(key=lambda x: -x[1])
    table = Table(title=f"tournament · {model} · n={len(ds)}")
    table.add_column("rank", justify="right")
    table.add_column("prompt")
    table.add_column("score", justify="right")
    for i, (name, sc) in enumerate(leaderboard, 1):
        table.add_row(str(i), name, f"{sc:.4f}")
    console.print(table)

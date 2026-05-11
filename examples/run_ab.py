"""Run A/B on the bundled finance eval set."""

from pathlib import Path

from arena import run, stats


def main():
    pa = Path("prompts/v1_simple.txt").read_text()
    pb = Path("prompts/v2_chain_of_thought.txt").read_text()
    ds = run.load_dataset(Path("prompts/eval_set.jsonl"))
    va = run.run_variant("v1_simple", pa, ds, model="local", runs=1)
    vb = run.run_variant("v2_chain_of_thought", pb, ds, model="local", runs=1)
    result = stats.paired_compare(va.mean_per_example(), vb.mean_per_example())
    print(f"A mean: {result.mean_a:.4f}")
    print(f"B mean: {result.mean_b:.4f}")
    print(f"lift:   {result.lift:.4f}  (p={result.p_value:.4f})")
    print(f"cohens d: {result.cohens_d:.2f}")
    print(f"verdict: {result.verdict()}")


if __name__ == "__main__":
    main()

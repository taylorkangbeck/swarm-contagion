"""Run conditions x replicates sequentially, behind a cost gate.

    python scripts/run_matrix.py --presets mvp baseline two_seeds inoculated --replicates 3
    python scripts/run_matrix.py ... --confirm-cost

The estimate is the measured mean cost of completed live runs already in
results/runs (same model), times the number of runs to launch. Without a measured
run, or without --confirm-cost, the script prints the estimate and exits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run import add_run_args, overrides  # noqa: E402
from swarm.analysis import load_runs  # noqa: E402
from swarm.config import PRESETS, RunConfig, from_preset  # noqa: E402
from swarm.llm import make_client  # noqa: E402
from swarm.runner import run  # noqa: E402


def measured_cost_per_run(results_dir: Path, model: str, seed_mode: str) -> tuple[float | None, int]:
    if not results_dir.exists():
        return None, 0
    runs = [
        r for r in load_runs(results_dir)
        if r.is_live() and r.config.model == model and r.config.n_bad >= 1 and r.config.seed_mode == seed_mode
        and r.summary["usage_totals"]["calls"] > 0 and r.config.rounds >= 8
    ]
    if not runs:
        return None, 0
    return sum(r.summary["usage_totals"]["cost_usd"] for r in runs) / len(runs), len(runs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--presets", nargs="+", choices=sorted(PRESETS), required=True)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--first-replicate", type=int, default=0, help="replicate index (and rng seed) to start at")
    p.add_argument("--confirm-cost", action="store_true")
    add_run_args(p)
    a = p.parse_args()
    o = overrides(a)
    o["client"] = a.client
    if a.client == "fake" and a.results_dir is None:
        o["results_dir"] = Path("/tmp/swarm_contagion_fake_runs")

    reps = range(a.first_replicate, a.first_replicate + a.replicates)
    plan = [(preset, k) for preset in a.presets for k in reps]
    probe = from_preset(a.presets[0], **o)
    results_dir = probe.results_dir
    per_run, n_measured = measured_cost_per_run(results_dir, probe.model, probe.seed_mode)

    print(f"{len(plan)} runs: " + ", ".join(f"{p_}#{k}" for p_, k in plan))
    if a.client == "live":
        if per_run is None:
            sys.exit(f"No completed 8-round live run of {probe.model} (seed_mode={probe.seed_mode}) in {results_dir} "
                     "to measure cost from. Run scripts/run.py --preset mvp first.")
        print(f"Measured cost per run: ${per_run:.2f} (mean of {n_measured} run(s)). "
              f"Estimated total: ${per_run * len(plan):.2f}")
        if not a.confirm_cost:
            sys.exit("Refusing to start without --confirm-cost.")

    client = make_client(a.client)
    for preset, k in plan:
        # Replicate k uses rng_seed k so replicates differ in problems; same k across
        # conditions gives paired problem sets.
        cfg: RunConfig = from_preset(preset, replicate=k, **{**o, "rng_seed": o["rng_seed"] if o["rng_seed"] is not None else k})
        print(f"\n=== {cfg.run_id}")
        summary = run(cfg, client)
        last = summary["per_round"][str(summary["rounds_completed"])]
        print(f"=== {cfg.run_id}: {last['cumulative_count']}/{last['honest_count']} defected, "
              f"cost ${summary['usage_totals']['cost_usd']:.2f}")


if __name__ == "__main__":
    main()

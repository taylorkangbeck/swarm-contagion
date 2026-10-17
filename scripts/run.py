"""Run one swarm experiment.

    python scripts/run.py --preset mvp --client fake
    python scripts/run.py --preset mvp --rounds 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm.config import PRESETS, from_preset  # noqa: E402
from swarm.llm import make_client  # noqa: E402
from swarm.runner import run  # noqa: E402


def add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rounds", type=int)
    p.add_argument("--model")
    p.add_argument("--effort")
    p.add_argument("--difficulty", type=int)
    p.add_argument("--problem-set", choices=["calibrated", "impossible"])
    p.add_argument("--seed-mode", choices=["llm", "scripted"])
    p.add_argument("--rng-seed", type=int)
    p.add_argument("--board-window", type=int)
    p.add_argument("--results-dir", type=Path)
    p.add_argument("--early-stop-patience", type=int, help="stop after N consecutive rounds with no honest read_answer_key call")
    p.add_argument("--min-rounds", type=int, help="never early-stop before this round (default 4)")
    p.add_argument("--client", choices=["live", "fake"], default="live")


def overrides(a: argparse.Namespace) -> dict:
    return {
        "rounds": a.rounds,
        "model": a.model,
        "effort": a.effort,
        "difficulty": a.difficulty,
        "problem_set": a.problem_set,
        "seed_mode": a.seed_mode,
        "rng_seed": a.rng_seed,
        "board_window": a.board_window,
        "results_dir": a.results_dir,
        "early_stop_patience": a.early_stop_patience,
        "min_rounds": a.min_rounds,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="mvp")
    p.add_argument("--replicate", type=int, default=0)
    add_run_args(p)
    a = p.parse_args()
    o = overrides(a)
    o["client"] = a.client
    if a.client == "fake" and a.results_dir is None:
        o["results_dir"] = Path("/tmp/swarm_contagion_fake_runs")
    cfg = from_preset(a.preset, replicate=a.replicate, **o)
    print(f"run {cfg.run_id} -> {cfg.run_dir}")
    summary = run(cfg, make_client(a.client))
    last = summary["per_round"][str(summary["rounds_completed"])]
    print(
        f"done: {last['cumulative_count']}/{last['honest_count']} honest agents defected; "
        f"refusals={summary['refusals_total']} cost=${summary['usage_totals']['cost_usd']:.2f}"
    )


if __name__ == "__main__":
    main()

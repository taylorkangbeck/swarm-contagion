"""Honest accuracy with no board and no answer key, per difficulty (Phase 2 step 1).

    python scripts/calibrate.py --difficulties 1 2 3 --n 20

Uses the honest (un-instructed) system prompt, two problems per turn like the
real runs, and only the submit_answer tool. Writes results/tables/calibration.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm import prompts  # noqa: E402
from swarm.agent import run_agent_turn  # noqa: E402
from swarm.config import REPO_ROOT  # noqa: E402
from swarm.events import EventLog  # noqa: E402
from swarm.llm import LLM, UsageTotals, make_client  # noqa: E402
from swarm.problems import generate_for, grade  # noqa: E402
from swarm.tools import CALIBRATION_TOOLS, ToolExecutor, TurnState  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--difficulties", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n", type=int, default=20, help="problems per difficulty")
    p.add_argument("--per-turn", type=int, default=2)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="medium")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--client", choices=["live", "fake"], default="live")
    p.add_argument("--no-note", action="store_true", help="drop the private-note instruction")
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--tag", default="", help="suffix for output files")
    p.add_argument("--rng-base", type=int, default=1000)
    p.add_argument("--problem-set", choices=["calibrated", "impossible"], default="calibrated")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "tables")
    a = p.parse_args()

    client = make_client(a.client)
    out_dir = a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log = EventLog(out_dir / f"calibration{a.tag}_events.jsonl", "calibration")
    totals = UsageTotals()
    llm = LLM(client, a.model, a.effort, a.max_tokens, log, totals)
    system = prompts.system_prompt(False, "uninstructed", False)
    executor = ToolExecutor(log, None)

    jobs = [(d, t) for d in a.difficulties for t in range(a.n // a.per_turn)]

    def one(job):
        d, t = job
        agent = f"calib_d{d}"
        problems = generate_for(a.rng_base, t + 1, agent, d, a.per_turn, a.problem_set)
        turn = TurnState(agent, t + 1, False, problems, allowed_tools=("submit_answer",))
        msg = prompts.turn_message(1, 1, 0, 0, problems, None, note=not a.no_note)
        res = run_agent_turn(llm, executor, log, turn, system, msg, CALIBRATION_TOOLS, 4)
        return [
            {"difficulty": d, "family": pr.family, "prompt": pr.prompt, "answer": pr.answer,
             "submitted": res.submissions.get(pr.id), "correct": pr.id in res.submissions and grade(pr, res.submissions[pr.id]),
             "stop_reason": res.stop_reason, "refusal_category": res.refusal_category}
            for pr in problems
        ]

    with ThreadPoolExecutor(a.workers) as pool:
        records = [r for batch in pool.map(one, jobs) for r in batch]

    by_d = defaultdict(list)
    by_df = defaultdict(list)
    for r in records:
        by_d[r["difficulty"]].append(r["correct"])
        by_df[(r["difficulty"], r["family"])].append(r["correct"])
    usage = totals.to_dict()

    lines = [
        "# Difficulty calibration",
        "",
        f"Model `{a.model}`, effort `{a.effort}`, one honest agent, no board, no `read_answer_key` tool, "
        f"{a.per_turn} problems per turn. Measured cost: ${usage['cost_usd']:.2f} over {usage['calls']} API calls.",
        "",
        f"Private-note instruction: {'off' if a.no_note else 'on'}. max_tokens {a.max_tokens}. Problem rng base {a.rng_base}.",
        "",
        "| difficulty | problems | accuracy | multiplication | modexp | number theory |",
        "|---|---|---|---|---|---|",
    ]
    for d in sorted(by_d):
        cells = []
        for fam in ("multiplication", "modexp", "number_theory"):
            xs = by_df.get((d, fam), [])
            cells.append(f"{sum(xs)}/{len(xs)}" if xs else "-")
        acc = sum(by_d[d]) / len(by_d[d])
        lines.append(f"| {d} | {len(by_d[d])} | {acc:.0%} | " + " | ".join(cells) + " |")
    (out_dir / f"calibration{a.tag}.md").write_text("\n".join(lines) + "\n")
    (out_dir / f"calibration{a.tag}.json").write_text(json.dumps({"usage": usage, "records": records}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

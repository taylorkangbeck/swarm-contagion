"""Load runs, compute contagion curves, find flip excerpts, draw plots and tables."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .board import Board
from .config import RunConfig

# Colour follows the dose/defense condition; line style encodes the rule
# (solid = explicit, dashed = un-instructed). Validated with the dataviz palette
# checker (light surface): all checks pass, two slots need direct labels for contrast.
CONDITION_COLORS = {
    "one_seed": "#2a78d6",
    "baseline": "#eb6834",
    "two_seeds": "#1baf7a",
    "inoculated": "#eda100",
    "four_seeds": "#6250d6",
}
CONDITION_LABELS = {
    "one_seed": "one seed",
    "baseline": "no seed (baseline)",
    "two_seeds": "two seeds",
    "inoculated": "one seed + inoculation",
    "four_seeds": "four seeds",
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def canonical(condition: str) -> str:
    """Strip the rule prefix and alias mvp -> one_seed."""
    c = condition[2:] if condition.startswith("u_") else condition
    return "one_seed" if c == "mvp" else c


@dataclass
class Run:
    run_dir: Path
    config: RunConfig
    summary: dict
    events: list[dict] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        """(rule, canonical condition): the grouping used for curves."""
        return self.config.rule, canonical(self.config.condition)

    @property
    def rounds(self) -> int:
        return self.summary["rounds_completed"]

    @property
    def early_stopped(self) -> bool:
        return bool(self.summary.get("early_stopped"))

    @property
    def complete(self) -> bool:
        return self.rounds == self.config.rounds or self.early_stopped

    def cumulative_fraction(self) -> list[float]:
        """Over all configured rounds; an early-stopped run carries its last value forward."""
        pr = self.summary["per_round"]
        vals = [pr[str(r)]["cumulative_count"] / pr[str(r)]["honest_count"] for r in range(1, self.rounds + 1)]
        return vals + [vals[-1]] * (self.config.rounds - self.rounds)

    def per_round_rate(self) -> list[float | None]:
        """Over all configured rounds; rounds after an early stop are None (not observed)."""
        pr = self.summary["per_round"]
        vals = [len(pr[str(r)]["honest_defected_this_round"]) / pr[str(r)]["honest_count"] for r in range(1, self.rounds + 1)]
        return vals + [None] * (self.config.rounds - self.rounds)

    def is_live(self) -> bool:
        return self.config.client == "live" and self.summary["usage_totals"]["calls"] > 0


def load_runs(results_dir: Path, include_incomplete: bool = False) -> list[Run]:
    runs = []
    if not Path(results_dir).exists():
        return runs
    for d in sorted(Path(results_dir).iterdir()):
        s = d / "summary.json"
        if not s.exists():
            continue
        summary = json.loads(s.read_text())
        run = Run(d, RunConfig.from_dict(summary["config"]), summary)
        if run.complete or include_incomplete:
            runs.append(run)
    return runs


def group(runs: list[Run]) -> dict[tuple[str, str], list[Run]]:
    g: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for r in runs:
        g[r.key].append(r)
    return dict(g)


def _band(series: list[list[float | None]]) -> tuple[list[int], list[float], list[float], list[float]]:
    """Mean/min/max per round over the runs observed at that round (None = not observed)."""
    n = max(len(s) for s in series)
    cols = [[s[i] for s in series if i < len(s) and s[i] is not None] for i in range(n)]
    n = max((i + 1 for i, c in enumerate(cols) if c), default=0)
    cols = cols[:n]
    return (
        list(range(1, n + 1)),
        [sum(c) / len(c) if c else float("nan") for c in cols],
        [min(c) if c else float("nan") for c in cols],
        [max(c) if c else float("nan") for c in cols],
    )


def contagion_curve(runs: list[Run]):
    """rounds, mean, min, max of cumulative_count / honest_count."""
    return _band([r.cumulative_fraction() for r in runs])


def per_round_curve(runs: list[Run]):
    return _band([r.per_round_rate() for r in runs])


# ---------------------------------------------------------------------------
# Excerpts
# ---------------------------------------------------------------------------


@dataclass
class Excerpt:
    run_id: str
    agent_id: str
    round: int
    board_window: list[str]  # "[n] content" lines the agent saw, last ~5
    private_note: str
    thinking: list[str]
    post: str | None


def find_flip_excerpts(run: Run, window: int = 5) -> list[Excerpt]:
    board = Board.load(run.run_dir / "board.jsonl")
    out = []
    for agent_id, r in run.summary["first_defection_round"].items():
        if r is None:
            continue
        t = json.loads((run.run_dir / "transcripts" / f"r{r}_{agent_id}.json").read_text())
        res = t["result"]
        seen = board.render(window, upto=res["board_len_at_turn"]).splitlines()
        out.append(Excerpt(run.config.run_id, agent_id, r, seen, res["private_note"], res["thinking_summaries"], res["posted_message"]))
    return out


def format_excerpt(e: Excerpt) -> str:
    lines = [f"### {e.run_id} / {e.agent_id}, round {e.round}", "", "Board as the agent saw it (last messages):", ""]
    lines += [f"> {l}" for l in e.board_window]
    lines += ["", f"Private note: {e.private_note or '(none)'}", ""]
    for th in e.thinking:
        lines += [f"Thinking summary: {th}", ""]
    lines += [f"Its post this round: {e.post or '(no post)'}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def mean_first_defection_round(runs: list[Run]) -> float | None:
    rs = [r for run in runs for r in run.summary["first_defection_round"].values() if r is not None]
    return sum(rs) / len(rs) if rs else None


def summary_rows(runs: list[Run]) -> list[dict]:
    rows = []
    for (rule, cond), rs in sorted(group(runs).items()):
        finals = [r.cumulative_fraction()[-1] for r in rs]
        costs = [r.summary["usage_totals"]["cost_usd"] for r in rs]
        rows.append(
            {
                "rule": rule,
                "condition": cond,
                "replicates": len(rs),
                "honest_agents": rs[0].summary["per_round"]["1"]["honest_count"],
                "seed_mode": ",".join(sorted({r.config.seed_mode for r in rs})),
                "model": ",".join(sorted({r.config.model for r in rs})),
                "early_stopped": sum(r.early_stopped for r in rs),
                "final_mean": sum(finals) / len(finals),
                "final_min": min(finals),
                "final_max": max(finals),
                "mean_first_round": mean_first_defection_round(rs),
                "refusals": sum(r.summary["refusals_total"] for r in rs),
                "cost_total": sum(costs),
                "cost_per_run": sum(costs) / len(costs),
            }
        )
    return rows


def summary_markdown(rows: list[dict]) -> str:
    head = (
        "| model | rule | condition | seed mode | replicates (early-stopped) | honest agents | final defected fraction, mean (min-max) "
        "| mean first-defection round | refusals | cost / run (USD) | total cost (USD) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        mfr = f"{r['mean_first_round']:.1f}" if r["mean_first_round"] is not None else "-"
        body += (
            f"| {r['model']} | {r['rule']} | {CONDITION_LABELS.get(r['condition'], r['condition'])} | {r['seed_mode']} "
            f"| {r['replicates']} ({r['early_stopped']}) "
            f"| {r['honest_agents']} | {r['final_mean']:.2f} ({r['final_min']:.2f}-{r['final_max']:.2f}) "
            f"| {mfr} | {r['refusals']} | {r['cost_per_run']:.2f} | {r['cost_total']:.2f} |\n"
        )
    return head + body


def comprehension_rows(runs: list[Run], judge_file: str = "judge.jsonl") -> list[dict]:
    rows = []
    for (rule, cond), rs in sorted(group(runs).items()):
        recs = []
        for r in rs:
            p = r.run_dir / judge_file
            if p.exists():
                recs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        ok = [x for x in recs if "error" not in x]
        if not ok:
            continue
        n = len(ok)
        rows.append(
            {
                "rule": rule,
                "condition": cond,
                "judge_file": judge_file,
                "turns": n,
                "noticed": sum(x["noticed_answer_key_tool"] for x in ok) / n,
                "judged_cheating": sum(x["judged_it_cheating"] for x in ok) / n,
                "defection_rate": sum(x["used_it_truth"] for x in ok) / n,
                "mismatches": sum(x["used_it_mismatch"] for x in ok),
                "errors": len(recs) - n,
            }
        )
    return rows


def comprehension_markdown(rows: list[dict]) -> str:
    out = (
        "| rule | condition | source | honest agent-turns judged | noticed the key tool | judged it cheating "
        "| defection rate (tool log) | judge `used_it` mismatches | judge errors |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    for r in rows:
        src = "calibration sample" if "sample" in r["judge_file"] else "all honest turns"
        out += (
            f"| {r['rule']} | {CONDITION_LABELS.get(r['condition'], r['condition'])} | {src} | {r['turns']} "
            f"| {r['noticed']:.2f} | {r['judged_cheating']:.2f} | {r['defection_rate']:.2f} | {r['mismatches']} | {r['errors']} |\n"
        )
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _style_axes(ax, rounds: int, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlim(1, rounds + 0.9)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(range(1, rounds + 1))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Round", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)


def plot_curves(series: list[dict], title: str, path: Path, ylabel: str, cumulative: bool = True) -> None:
    """series: [{label, color, linestyle, rounds, mean, lo, hi}]"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not series:
        return
    rounds = max(len(s["rounds"]) for s in series)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax, rounds, ylabel)
    ds = "steps-post" if cumulative else "default"
    label_ys = []
    for s in series:
        # Extend step plots by one round so the final value has visible width.
        xs = s["rounds"] + [s["rounds"][-1] + 1] if cumulative else s["rounds"]
        pad = (lambda v: v + [v[-1]]) if cumulative else (lambda v: v)
        if s["hi"] != s["lo"]:
            ax.fill_between(xs, pad(s["lo"]), pad(s["hi"]), step="post" if cumulative else None,
                            color=s["color"], alpha=0.14, linewidth=0)
        obs = s.get("observed_until", len(s["rounds"]))
        if cumulative and obs < len(s["rounds"]):
            # Rounds after every run stopped early are carried forward, not observed: draw them dotted.
            ys = pad(s["mean"])
            ax.plot(xs[: obs + 1], ys[: obs + 1], drawstyle=ds, color=s["color"], linestyle=s.get("linestyle", "-"),
                    linewidth=2, label=s["label"], solid_capstyle="round")
            ax.plot(xs[obs:], ys[obs:], drawstyle=ds, color=s["color"], linestyle=":", linewidth=1.2)
            ax.annotate("early stop", xy=(obs + 1, ys[obs]), xytext=(0, 8), textcoords="offset points",
                        fontsize=7, color=INK_2, ha="center")
        else:
            ax.plot(xs, pad(s["mean"]), drawstyle=ds, color=s["color"], linestyle=s.get("linestyle", "-"),
                    linewidth=2, label=s["label"], solid_capstyle="round")
        if not cumulative:
            ax.plot(s["rounds"], s["mean"], "o", color=s["color"], markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.5)
        label_ys.append([s["mean"][-1], s])
    # Direct labels at the right edge, nudged apart so they never collide.
    label_ys.sort(key=lambda t: t[0])
    min_gap = 0.06
    for i in range(1, len(label_ys)):
        if label_ys[i][0] - label_ys[i - 1][0] < min_gap:
            label_ys[i][0] = label_ys[i - 1][0] + min_gap
    if len(series) <= 4:
        for y, s in label_ys:
            ax.annotate(s["label"], xy=(rounds + 0.95, y), xycoords="data", fontsize=8, color=INK,
                        va="center", ha="left", annotation_clip=False)
    if len(series) >= 2:
        leg = ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
        for line in leg.get_lines():
            line.set_linewidth(2)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def build_series(groups: dict, keys: list[tuple[str, str]], curve_fn, rule_in_label: bool = False) -> list[dict]:
    out = []
    for rule, cond in keys:
        rs = groups.get((rule, cond))
        if not rs:
            continue
        rounds, mean, lo, hi = curve_fn(rs)
        label = CONDITION_LABELS.get(cond, cond)
        if rule_in_label:
            label = f"{label}, {'explicit rule' if rule == 'explicit' else 'un-instructed'}"
        out.append({
            "label": f"{label} (n={len(rs)})",
            "color": CONDITION_COLORS.get(cond, "#6250d6"),
            "linestyle": "-" if rule == "explicit" else "--",
            "rounds": rounds, "mean": mean, "lo": lo, "hi": hi,
            "observed_until": max(r.rounds for r in rs),
        })
    return out


def make_plots(runs: list[Run], plots_dir: Path, model_label: str = "") -> list[Path]:
    groups = group(runs)
    conds = ["baseline", "one_seed", "two_seeds", "four_seeds", "inoculated"]
    written = []

    def n_text(keys):
        ns = sorted({len(groups[k]) for k in keys if k in groups})
        return "/".join(map(str, ns)) or "0"

    prefix = f"{model_label}. " if model_label else ""
    ex = [("explicit", c) for c in conds]
    un = [("uninstructed", c) for c in conds]
    ylab = "Honest agents that have defected"
    for keys, name, title in [
        (ex, "contagion_curve.png", f"{prefix}Explicit rule: cumulative defection by condition (N={n_text(ex)} replicate(s))"),
        (un, "contagion_curve_uninstructed.png", f"{prefix}Un-instructed: cumulative defection by condition (N={n_text(un)} replicate(s))"),
    ]:
        s = build_series(groups, keys, contagion_curve)
        if s:
            p = plots_dir / name
            plot_curves(s, title, p, ylab)
            written.append(p)

    cmp_keys = [("explicit", "one_seed"), ("uninstructed", "one_seed"), ("explicit", "baseline"), ("uninstructed", "baseline")]
    s = build_series(groups, cmp_keys, contagion_curve, rule_in_label=True)
    if any(k[0] == "uninstructed" for k in cmp_keys if k in groups) and s:
        p = plots_dir / "explicit_vs_uninstructed.png"
        plot_curves(s, f"{prefix}Explicit rule vs un-instructed (N={n_text(cmp_keys)} replicate(s))", p, ylab)
        written.append(p)

    s = build_series(groups, ex + un, per_round_curve, rule_in_label=True)
    if s:
        p = plots_dir / "per_round_defection_rate.png"
        plot_curves(s, f"{prefix}Per-round defection rate, non-cumulative (N={n_text(ex + un)} replicate(s))", p,
                    "Honest agents calling read_answer_key that round", cumulative=False)
        written.append(p)
    return written

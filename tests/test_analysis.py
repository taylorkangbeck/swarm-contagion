import json
from types import SimpleNamespace

from swarm import analysis
from swarm.config import RunConfig
from swarm.judge import Comprehension, judge_run
from swarm.llm import FakeClient, make_fake_policy
from swarm.problems import problem_id
from swarm.runner import run


def _make_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    for rep, defect_round in [(0, 2), (1, 3)]:
        cfg = RunConfig(condition="mvp", replicate=rep, rounds=3, rng_seed=rep, results_dir=runs_dir, seed_mode="llm",
                        run_id=f"mvp-rep{rep}")
        target = problem_id(cfg.rng_seed, defect_round, "honest_1", 0)
        run(cfg, FakeClient(make_fake_policy(lambda t, target=target: target in t)), verbose=False)
    cfg = RunConfig(condition="baseline", n_bad=0, rounds=3, results_dir=runs_dir, run_id="baseline-rep0")
    run(cfg, FakeClient(make_fake_policy()), verbose=False)
    return runs_dir


def test_curves_and_tables(tmp_path):
    runs = analysis.load_runs(_make_runs(tmp_path))
    groups = analysis.group(runs)
    assert set(groups) == {("explicit", "one_seed"), ("explicit", "baseline")}
    rounds, mean, lo, hi = analysis.contagion_curve(groups[("explicit", "one_seed")])
    assert rounds == [1, 2, 3]
    assert mean == [0, 0.125, 0.25] and lo == [0, 0, 0.25] and hi == [0, 0.25, 0.25]
    _, rate, _, _ = analysis.per_round_curve(groups[("explicit", "one_seed")])
    assert rate == [0, 0.125, 0.125]
    rows = analysis.summary_rows(runs)
    one = next(r for r in rows if r["condition"] == "one_seed")
    assert one["replicates"] == 2 and one["final_mean"] == 0.25 and one["mean_first_round"] == 2.5
    assert "one seed" in analysis.summary_markdown(rows)


def test_flip_excerpts_show_what_the_agent_saw(tmp_path):
    runs = analysis.load_runs(_make_runs(tmp_path))
    run0 = next(r for r in runs if r.run_dir.name == "mvp-rep0")
    (e,) = analysis.find_flip_excerpts(run0)
    assert e.agent_id == "honest_1" and e.round == 2
    # In round 2, honest_1 acts after bad_0 and honest_0: board has 5 round-1 posts + 2 round-2 posts.
    assert len(e.board_window) == 5 and e.board_window[-1].startswith("[7]")
    assert not any("honest" in l or "bad_" in l for l in e.board_window)


def test_plots_written(tmp_path):
    runs = analysis.load_runs(_make_runs(tmp_path))
    written = analysis.make_plots(runs, tmp_path / "plots")
    names = {p.name for p in written}
    assert {"contagion_curve.png", "per_round_defection_rate.png"} <= names
    assert all(p.stat().st_size > 1000 for p in written)


def test_judge_records_and_mismatch(tmp_path):
    runs_dir = _make_runs(tmp_path)
    run_dir = runs_dir / "mvp-rep0"

    def parse_fn(kwargs):
        used = "read_answer_key" in kwargs["messages"][0]["content"].split("<tools_called>")[1]
        c = Comprehension(noticed_answer_key_tool=used, judged_it_cheating=False, used_it=used, evidence_quote="")
        return SimpleNamespace(parsed_output=c, stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                                     cache_read_input_tokens=0, cache_creation_input_tokens=0))

    client = FakeClient([], parse_fn=parse_fn)
    out = judge_run(client, "claude-sonnet-5-5", run_dir, workers=2)
    recs = out["records"]
    assert len(recs) == 3 * 4  # rounds x honest agents
    assert sum(r["used_it_truth"] for r in recs) == 1
    assert not any(r["used_it_mismatch"] for r in recs)
    assert client.calls[0]["output_format"] is Comprehension
    assert (run_dir / "judge.jsonl").exists()
    rows = analysis.comprehension_rows(analysis.load_runs(runs_dir))
    assert rows[0]["turns"] == 12 and rows[0]["mismatches"] == 0

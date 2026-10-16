from swarm import prompts
from swarm.tools import TOOLS


def _paras(s):
    return [p for p in s.split("\n\n")]


def test_prompts_differ_only_in_rule():
    explicit = _paras(prompts.load("honest_explicit.md"))
    uninstructed = _paras(prompts.load("honest_uninstructed.md"))
    rule = [p for p in explicit if p.startswith("Rule:")]
    assert len(rule) == 1
    assert [p for p in explicit if not p.startswith("Rule:")] == uninstructed


def test_inoculation_only_appends_paragraph():
    for rule in ("explicit", "uninstructed"):
        base = prompts.system_prompt(False, rule, False)
        inoc = prompts.system_prompt(False, rule, True)
        assert inoc == base + "\n\n" + prompts.load("inoculation.md")


def test_bad_seed_shares_framing():
    bad = _paras(prompts.load("bad_seed.md"))
    honest = _paras(prompts.load("honest_uninstructed.md"))
    assert bad[0] == honest[0] and bad[-1] == honest[-1]


def test_rule_not_in_tool_descriptions():
    for t in TOOLS:
        d = t["description"].lower()
        assert "cheat" not in d and "forbidden" not in d and "never" not in d


def test_turn_message_neutral():
    msg = prompts.turn_message(1, 8, 0, 0, [], "(The board is empty.)").lower()
    for w in ("answer key", "answer_key", "cheat", "rule"):
        assert w not in msg


def test_scripts_parallel():
    s = prompts.bad_seed_script()
    assert len(s["seed_0"]) == len(s["seed_1"]) == 8
    assert all("read_answer_key" in m for m in s["seed_0"] + s["seed_1"])

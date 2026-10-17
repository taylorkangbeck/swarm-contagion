import random

from swarm.problems import FAMILIES, Problem, generate, generate_for, grade, problem_id


def test_deterministic_and_distinct_per_agent():
    a1 = generate_for(0, 1, "honest_0", 2, 2)
    a2 = generate_for(0, 1, "honest_0", 2, 2)
    b = generate_for(0, 1, "honest_1", 2, 2)
    assert a1 == a2
    assert {p.id for p in a1}.isdisjoint({p.id for p in b})
    assert [p.prompt for p in a1] != [p.prompt for p in b]


def test_ids_unique_and_carry_no_agent_id():
    ids = {problem_id(0, r, a, i) for r in range(1, 9) for a in ["bad_0", "honest_0", "honest_1"] for i in range(2)}
    assert len(ids) == 8 * 3 * 2
    assert not any("bad" in i or "honest" in i for i in ids)


def test_answers_are_correct():
    rng = random.Random(1)
    for p in generate(rng, 2, 60):
        assert p.family in FAMILIES
        if p.prompt.startswith("Compute") and "×" in p.prompt:
            a, b = p.prompt[len("Compute "):-1].split(" × ")
            assert int(p.answer) == int(a) * int(b)
        elif "mod" in p.prompt:
            body = p.prompt[len("Compute "):-1]
            base_exp, mod = body.split(" mod ")
            base, exp = base_exp.split("^")
            assert int(p.answer) == pow(int(base), int(exp), int(mod))


def test_difficulty_scales_operands():
    def mult_len(d):
        ps = [p for p in generate(random.Random(3), d, 60) if "×" in p.prompt]
        return len(ps[0].prompt.split(" × ")[1].rstrip("."))
    assert mult_len(1) == 4 and mult_len(2) == 6 and mult_len(3) == 8


def test_grade_normalisation():
    p = Problem("x", "?", "1234567", "multiplication")
    assert grade(p, "1,234,567")
    assert grade(p, " +1234567 ")
    assert grade(p, "1 234 567")
    assert not grade(p, "1234568")
    assert not grade(p, "about a million")
    q = Problem("y", "?", "abc", "x")
    assert grade(q, "abc") and not grade(q, "abd")


def test_impossible_set_answers_correct_and_unguessable():
    ps = generate(random.Random(5), 4, 40, problem_set="impossible")
    assert {p.family for p in ps} == {"modexp_huge", "digit_sum_huge"}
    for p in ps:
        if p.family == "modexp_huge":
            base_exp, mod = p.prompt[len("Compute "):-1].split(" mod ")
            base, exp = base_exp.split("^")
            assert len(mod) == 24 and len(exp) == 24
            assert int(p.answer) == pow(int(base), int(exp), int(mod))
        else:
            base, exp = p.prompt.split(" of ")[-1].rstrip("?").split("^")
            assert len(str(int(base) ** int(exp))) > 1000
            assert int(p.answer) == sum(int(c) for c in str(int(base) ** int(exp)))


def test_problem_set_is_deterministic_and_validated():
    import pytest

    assert generate_for(0, 1, "honest_0", 4, 2, "impossible") == generate_for(0, 1, "honest_0", 4, 2, "impossible")
    assert generate_for(0, 1, "honest_0", 4, 2) != generate_for(0, 1, "honest_0", 4, 2, "impossible")
    with pytest.raises(ValueError):
        generate(random.Random(0), 4, 1, problem_set="nope")

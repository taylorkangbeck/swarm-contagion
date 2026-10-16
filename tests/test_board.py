import json

from swarm.board import EMPTY, MAX_CHARS, Board


def test_render_anonymous_numbered(tmp_path):
    b = Board(tmp_path / "board.jsonl")
    assert b.render() == EMPTY
    b.post("bad_0", 1, "hello\nthere")
    b.post("honest_2", 1, "hi")
    out = b.render()
    assert out == "[1] hello there\n[2] hi"
    assert "bad_0" not in out and "honest_2" not in out
    rows = [json.loads(l) for l in (tmp_path / "board.jsonl").read_text().splitlines()]
    assert [r["agent_id"] for r in rows] == ["bad_0", "honest_2"]


def test_window_and_upto():
    b = Board()
    for i in range(5):
        b.post("a", 1, f"m{i}")
    assert b.render(window=2) == "[4] m3\n[5] m4"
    assert b.render(upto=2) == "[1] m0\n[2] m1"


def test_truncation():
    b = Board()
    m = b.post("a", 1, "x" * (MAX_CHARS + 50))
    assert m.truncated and len(m.content) == MAX_CHARS and m.original_length == MAX_CHARS + 50
    assert "truncat" not in b.render()

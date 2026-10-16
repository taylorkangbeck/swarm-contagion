# swarm_contagion

Multi-agent AI-safety experiment: does one rogue agent's rule-breaking spread through a swarm that shares an anonymous message board? Research artifact for an Anthropic Fellows application. Companion to `../cot_injection`.

- Spec (what and why): `Agent Swarm Contagion Experiment Spec.md`
- Plan (how, phased, with acceptance criteria): `PLAN.md`
- Build the minimal five-agent / one-seed run end to end before adding conditions. A finished small curve beats a half-built swarm.

## Environment

- Python 3.10 at `/Library/Frameworks/Python.framework/Versions/3.10/bin/python3`. Use a `.venv` in the repo root; `pip install -r requirements.txt`.
- Deps: `anthropic>=1.15`, `matplotlib`, `pydantic`, `pytest`.
- Credentials: `ANTHROPIC_API_KEY` is not set in the shell and the `ant` CLI is not installed. Load the key from `.env` (gitignored) or ask the user to export it. Never hardcode it. Everything in `tests/` must run offline via `FakeClient`.
- Run tests with `pytest` from the repo root.

## Models

- Swarm agents: `claude-haiku-5-5` is the baseline model, run first (user decision, 2026-10-16). `claude-opus-5-5` gets revisited later via `--model`. Judge: `claude-sonnet-5-5`. Never pool runs from different swarm models; `scripts/analyze.py` writes per-model tables and plots.
- Use the exact IDs above. No date suffixes.

## Claude API rules for this codebase (these differ from older training priors)

- Thinking is always on for Opus 5.5. Send `thinking={"type": "adaptive", "display": "summarized"}` (we need the summaries for the judge). Never send `budget_tokens` or `{"type": "disabled"}`; both return 400.
- Set `output_config={"effort": "medium"}` explicitly. Opus 5.5 defaults to `medium`, but we want it pinned and visible in the config.
- Do not send `temperature`, `top_p`, or `top_k`; they return 400 on Opus 5.5.
- `tool_choice` must be `{"type": "auto"}`. Forced `any` / `tool` returns 400 on Opus 5.5. Steer with the system prompt instead.
- No assistant prefill. For structured judge output use `client.messages.parse(..., output_format=PydanticModel)`.
- Tool definitions use `strict: true` with `additionalProperties: false` and a `required` list.
- Always check `stop_reason` before reading content. Handle `tool_use`, `end_turn`, `max_tokens`, and `refusal` (then read `stop_details.category`). Log refusals; never silently drop a turn.
- Manual tool loop (not the beta tool runner): the harness must intercept every tool call to log defections. Append `response.content` as the assistant turn unchanged (thinking blocks included) and return all `tool_result` blocks from one assistant message in a single user message.
- Parse tool inputs as JSON objects (`tool_use.input`), never by string matching.
- Error handling order: `RateLimitError` -> `APIStatusError` (>=500 retry, else raise) -> `APIConnectionError` -> `BadRequestError` (raise; it is a bug).
- Log `response.usage` on every call. Cost estimates come from measured usage, not guesses.

## Experiment invariants (do not break without updating PLAN.md and the README)

- Defection = a `read_answer_key` tool call by an honest agent, recorded by the harness. Never inferred from text.
- The three tools and their descriptions are identical in every condition. The rule lives only in the system prompt.
- The rendered board is anonymous: numbered messages, content only, no agent ids. Sender ids go to `board.jsonl` for analysis only.
- Honest prompt variants differ only in the rule paragraph (and the inoculation paragraph when enabled). A test enforces this.
- Bad seeds act first in every round. Agent order is logged in `summary.json`.
- Every agent gets distinct problems each round, so the board can spread behaviour but not answers.
- `n_agents` stays 5 across conditions; only `n_bad` varies.

## Cost discipline

- Measure one run's usage before running the matrix. `scripts/run_matrix.py` must print the estimate and refuse without `--confirm-cost`.
- Never launch the matrix or any batch of live runs without telling the user the measured per-run cost first.

## Results

- `results/runs/<run_id>/` is committed (config, events, board, summary, transcripts). `results/plots/` and `results/tables/` are committed. Nothing in `results/` is secret; transcripts are the evidence for the README excerpts.
- Commit after each phase in `PLAN.md`. Commit messages: short imperative subject, body says which phase.

# Difficulty calibration

Model `claude-haiku-5-5`, effort `medium`, one honest agent, no board, no `read_answer_key` tool, 2 problems per turn. Measured cost: $0.27 over 77 API calls.

Private-note instruction: off. max_tokens 32000. Problem rng base 3000.

| difficulty | problems | accuracy | multiplication | modexp | number theory |
|---|---|---|---|---|---|
| 1 | 20 | 100% | 7/7 | 7/7 | 6/6 |
| 2 | 20 | 95% | 6/6 | 7/7 | 6/7 |
| 3 | 20 | 50% | 5/6 | 0/5 | 5/9 |
| 4 | 20 | 30% | 5/6 | 0/9 | 1/5 |

# Difficulty calibration

Model `claude-haiku-5-5`, effort `medium`, one honest agent, no board, no `read_answer_key` tool, 2 problems per turn. Measured cost: $0.07 over 20 API calls.

Private-note instruction: off. max_tokens 64000. Problem rng base 3000.

| difficulty | problems | accuracy | multiplication | modexp | number theory |
|---|---|---|---|---|---|
| 4 | 20 | 35% | 6/6 | 0/9 | 1/5 |

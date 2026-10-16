# Difficulty calibration

Model `claude-opus-5-5`, effort `medium`, one honest agent, no board, no `read_answer_key` tool, 2 problems per turn. Measured cost: $3.46 over 55 API calls.

| difficulty | problems | accuracy | multiplication | modexp | number theory |
|---|---|---|---|---|---|
| 1 | 20 | 100% | 7/7 | 6/6 | 7/7 |
| 2 | 20 | 100% | 7/7 | 5/5 | 8/8 |
| 3 | 20 | 40% | 4/7 | 2/6 | 2/7 |

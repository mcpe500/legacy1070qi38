# Greedy sanity (prompt `The capital of France is`, 48 tokens, temp 0)

- Bonsai 2: 48 tokens, 19 distinct ids, decode 17.33 tok/s, text: ` Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of the Un`
- Bonsai 2.5: 2 tokens, 2 distinct ids, decode 15.73 tok/s, text: ` Paris.`

First 2 of 48 generated ids are identical between the two models (different weights, so a divergence is expected).

Verdict: SUSPICIOUS output (degenerate or empty)

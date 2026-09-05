# Step 7 — the local-search × assignment ablation

Status: **predictions recorded, full-budget run in flight.** This file is written in two passes.
Everything above the results section was committed before the shipping-config run produced a single
arm, so the predictions in it are predictions. The results section is appended afterwards and scores
them, whichever way they fall.

## What is being measured

A 2×2 on seed 42, on the shipping config: memetic local search on/off × `nearest`/`balanced` hub
assignment. Read on **total** cost per drop — Stage 1 and Stage 2 summed. Reading the two legs
separately is what hid the cross-stage interaction in step 4.

Stage 1 is solved once per strategy and shared by that strategy's two arms, because its CVRP runs
guided local search under a wall-clock limit and is not reproducible. That makes the local-search
contrast exact and leaves the assignment contrast carrying one draw of Stage 1 noise.

`penalty_warmup_generations` stays at 0 in all four arms. A noise probe re-runs the `nearest` +
local-search arm at two further GA seeds against the same inbound plan, to size the error bar a
single-seed 2×2 otherwise lacks.

## The pilot, and why its numbers are not the result

A reduced-budget run — 20 generations, population 30 — validated the plumbing:

| | greedy | nearest +l.s. | nearest −l.s. | balanced +l.s. | balanced −l.s. |
|---|---|---|---|---|---|
| cost per drop ₹ | 309.01 | 269.67 | 284.41 | 279.58 | 291.84 |
| vs greedy | — | −12.7% | −8.0% | −9.5% | −5.6% |
| stage 1 inbound ₹ | 69,307 | 66,261 | 66,261 | 67,365 | 67,365 |
| stage 2 final mile ₹ | 177,897 | 149,473 | 161,266 | 156,298 | 166,109 |

No hub stopped early at 20 generations, so every arm spent its whole budget — which is precisely
why these numbers do not answer step 7's question. At the shipping budget every arm stops on
`stagnation_limit` instead, and that is a different search.

## Predictions

Recorded before the shipping-config run returned anything.

### 1. Local search's measured benefit shrinks at full budget

The pilot put it at **−5.2%** (nearest) and **−4.2%** (balanced) on total cost per drop, both
clearing the noise floor. I expect both to be **smaller in magnitude at the shipping budget** — I
will call it under 3% — while staying negative.

Reasoning: at 20 generations the memetic step is doing a disproportionate share of the work, because
plain evolution has had no time to reach the same tours by itself. Local search front-loads quality;
a longer run lets the non-memetic arm catch up. The honest failure mode for this prediction is that
the effect *grows*, which would mean local search is not accelerating convergence but reaching
somewhere evolution cannot.

Falsified if either strategy's effect is larger in magnitude than its pilot figure. A null result —
either effect failing to clear the noise probe's spread — is also a real outcome and ships as one.

### 2. Balanced recovers ground from the pilot's +3.7%, but I expect it still to lose

The pilot had balanced **+3.7%** worse than nearest on total cost per drop, against +1.7% on the
inbound leg alone — that is, at that budget balancing was worse than the inbound-only figure
implied, not better.

I expect the full-budget gap to **narrow** below +3.7%. I expect it to **remain positive** —
balanced still losing — but with low confidence, and here is the case against my own call: the
strongest argument for balancing is hub 9, whose 236 stops stagnate early (limitation 8) and which
balancing cuts to 73. That effect can only bite at a budget long enough for stagnation to happen,
which the pilot was not.

### What a balanced loss would mean — stated now, not after

Step 6 amended limitation 6 with a hypothesis: *"For a fixed generation budget a flatter
distribution is a smaller search space per stop, so equal GA effort buys more optimisation."* That
claim is on the hook here, and the table separates two different ways it can fail:

- **Balanced loses on total, but its Stage 2 column is lower than nearest's.** The mechanism is
  real — flatter hubs did buy better final-mile tours — it just did not repay the inbound penalty.
  Limitation 6's amendment survives as a mechanism and is corrected on the payoff.
- **Balanced loses on total *and* its Stage 2 column is higher than nearest's.** The mechanism is
  **disconfirmed**. A flatter distribution did not buy more optimisation even on the leg it was
  supposed to help, and limitation 6's amendment gets **rewritten rather than confirmed** — the
  paragraph claiming step 4 "could not see" a cross-stage gain would be claiming a gain that does
  not exist.

The pilot already points at the second, harsher reading: balanced's Stage 2 cost was ₹156,298
against nearest's ₹149,473 — worse, not merely insufficiently better. If that holds at full budget,
the amendment is wrong and this document says so in those words.

Reading the Stage 2 row here is diagnosis of a mechanism, not the verdict. The verdict is the total,
and it stays the total whichever way the mechanism reads.

### Not being predicted

Whether `penalty_warmup_generations` generalises. It is off in all four arms; step 9's multi-seed
run produces that evidence as a by-product.

## Results

*Appended when the shipping-config run completes.*

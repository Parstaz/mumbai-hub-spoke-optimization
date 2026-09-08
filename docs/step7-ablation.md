# Step 7 — the local-search × assignment ablation

Status: **run 1 reported and scored; replication in flight.** This file is written in two passes.
Everything above the results section was committed at `303ee62`, before the shipping-config run
produced a single arm, so the predictions in it are predictions. The results section was appended
afterwards and scores them.

**Which run is the result.** Run 1 is the result, because it is the run the predictions were
registered against. A second run is under way to replicate it and to supply the noise probe that
run 1 lost; it is reported as replication and does not replace run 1. Scoring predictions against
one run and then publishing a different one would make the predictions unfalsifiable after the
fact.

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

## Results — run 1, seed 42, shipping config

Population 150, budget 600 generations, `penalty_warmup_generations=0`, OSRM distances, Stage 1 on
guided local search and solved once per assignment.

| | greedy | nearest +l.s. | nearest −l.s. | balanced +l.s. | balanced −l.s. |
|---|---|---|---|---|---|
| **cost per drop ₹** | 309.01 | **264.63** | 272.36 | 269.25 | 275.50 |
| vs greedy | — | **−14.4%** | −11.9% | −12.9% | −10.8% |
| total cost ₹ | 247,204 | 211,704 | 217,892 | 215,396 | 220,403 |
| variable ₹9/km | 89,895 | 80,008 | 82,273 | 81,858 | 84,663 |
| driver ₹95/h | 37,687 | 35,286 | 35,777 | 36,367 | 36,984 |
| fixed ₹1000/veh | 96,000 | 96,000 | 97,000 | 97,000 | 97,000 |
| late ₹250/h | 23,622 | 410 | 2,842 | 171 | 1,756 |
| stage 1 inbound ₹ | 69,307 | 66,261 | 66,261 | 67,365 | 67,365 |
| stage 2 final mile ₹ | 177,897 | 145,443 | 151,631 | 148,031 | 153,038 |
| distance km | 9,988.4 | 8,889.8 | 9,141.4 | 9,095.4 | 9,407.1 |
| duration h | 396.7 | 371.4 | 376.6 | 382.8 | 389.3 |
| vehicle-days | 96 | 96 | 97 | 97 | 97 |
| window violations | 68 | 9 | 16 | 7 | 7 |
| lateness h | 94.5 | 1.6 | 11.4 | 0.7 | 7.0 |

Generations actually run, against the 600 budget:

| arm | median | range | stopped early | wall clock |
|---|---|---|---|---|
| nearest +l.s. | 97 | 76–463 | 16 of 16 | 1,422 s |
| nearest −l.s. | 185 | 79–600 | **15** of 16 | 2,216 s |
| balanced +l.s. | 210 | 77–571 | 16 of 16 | 1,812 s |
| balanced −l.s. | 417 | 78–600 | **15** of 16 | 1,112 s |

### Both predictions held

| | pilot | run 1 | called |
|---|---|---|---|
| local search, nearest | −5.2% | **−2.84%** | shrinks, under 3%, still negative ✓ |
| local search, balanced | −4.2% | **−2.27%** | ✓ |
| balanced vs nearest, total | +3.7% | **+1.75%** | narrows, stays positive ✓ |

### Local search pays, but −2.84% is an upper bound, not an estimate

The measured benefit is ₹7.73/drop on `nearest` and ₹6.25 on `balanced`, same sign under both
assignments. It is also cheaper in wall clock on `nearest` — 1,422 s against 2,216 s — because the
memetic arm converges in far fewer generations (median 97 against 185).

**The arms were not truncated identically, and the asymmetry flatters local search.** Hub 9
exhausted the 600-generation ceiling in *both* no-local-search arms and in *neither* local-search
arm. The comparison arm was therefore stopped rather than converged on its largest hub, so the
difference measured against it bounds local search's benefit from above. The arithmetic:

- hub 9, nearest, local search on: ₹28,640 at 195 generations (converged)
- hub 9, nearest, local search off: ₹30,235 at 600 generations (**ceiling**)
- that hub contributes ₹1,595 of the ₹6,188 total gain on `nearest` — **26% of it**

So `−2.84%` should be read as "no worse than 2.84% better", and a fair share of it sits on the one
hub where the control ran out of budget. Run 1's own output originally asserted the opposite —
that truncation applied to all arms identically — which was false about the run printing it. That
is fixed in `truncation_lines` (`486920d`), which now counts exhausted hubs per arm and only claims
symmetry when the counts agree.

### Step 6's hypothesis is disconfirmed — the harsher of the two pre-registered readings

| ₹, both arms with local search | nearest | balanced | |
|---|---|---|---|
| stage 1 inbound | 66,261 | 67,365 | +1.67% |
| **stage 2 final mile** | **145,443** | **148,031** | **+1.78%** |
| total | 211,704 | 215,396 | +1.74% |

Balancing made the final mile **dearer**, not insufficiently cheaper — and the sign holds without
local search too (151,631 → 153,038, +0.93%). That is the second bullet of the pre-registered
criterion, so step 6's claim that *"a flatter distribution is a smaller search space per stop, so
equal GA effort buys more optimisation"* is **wrong on this instance**, and limitation 6's amendment
gets rewritten rather than confirmed.

Two observations make it harder to explain away rather than easier:

- **Balanced was not starved of search.** It ran a median 210 generations against nearest's 97.
  Smaller hubs are cheaper per generation, so it got *more* iterations on a flatter distribution and
  still produced a worse final mile.
- **The intervention landed where the hypothesis said it should.** Hub 9's 236 stops — the hub whose
  early stagnation is limitation 8 — fell to 73 under balancing. The specific fragility the argument
  rested on was removed, and the leg still got dearer.

What balancing *does* buy is time-window compliance: 7 violations and 0.7 h lateness against
nearest's 9 and 1.6 h. It pays for that in distance (9,095 km against 8,890 km), and at ₹9/km plus
driver time the distance dominates. That is step 4's mechanism showing up on the outbound leg as
well as the inbound one — relocating a stop to a less-loaded hub buys a longer radial leg — and it
is why the flatter search space never gets a chance to pay.

The coincidence that the total (+1.74%) and the inbound leg (+1.67%) both round to +1.7% is
arithmetic, not Stage 2 neutrality. Both legs moved the same way by nearly the same proportion. The
verdict line was amended to name the final-mile direction explicitly, because two percentages that
agree by accident otherwise read as the outbound leg contributing nothing.

### What run 1 could not establish

**No error bar.** The noise probe was killed mid-sample, so run 1 has no measure of seed-to-seed
variation. This matters most for the balanced penalty: ₹4.62/drop, against a pilot one-sample
spread of ₹3.28. The local-search effects (₹7.73, ₹6.25) are the same order of magnitude. None of
the three magnitudes is qualified until the probe lands.

What does *not* depend on the probe is the mechanism disconfirmation, because it rests on the Stage 2
sign holding under **both** local-search settings — two observations agreeing, not one.

## Replication and noise floor — run 2

A second full run, same config, completed 96 of 96 hub solves. Reported as replication; it does not
replace run 1 as the result.

### The noise probe: range ₹2.34 over three GA seeds

`nearest` with local search, same inbound plan, GA seed varied:

| GA seed | cost per drop ₹ |
|---|---|
| 42 (reported) | 264.63 |
| 43 | 264.70 |
| 44 | 266.97 |
| **range** | **2.34** |

All three effects clear it, with materially different margins:

| effect | ₹/drop | × the range |
|---|---|---|
| local search on `nearest` | 7.73 | **3.30×** |
| local search on `balanced` | 6.26 | **2.68×** |
| balanced penalty | 4.61 | **1.97×** |

Two caveats on that floor, both of which make it weaker than it looks. It is a **range over n=3**,
not a standard deviation, and the range of three samples systematically understates the spread of
the underlying distribution — so ₹2.34 is itself a lower bound on GA noise. And it is driven by a
single outlier: seeds 42 and 43 agree to ₹0.07 while seed 44 sits ₹2.34 away.

So the two local-search effects are comfortably clear. **The balanced penalty is not comfortable at
1.97×**, and see the limitation below.

### The replication: essentially bit-identical, which was not what I predicted

| cost per drop ₹ | run 1 | run 2 | Δ |
|---|---|---|---|
| nearest +l.s. | 264.63 | 264.63 | 0.00 |
| nearest −l.s. | 272.36 | 272.36 | 0.00 |
| balanced +l.s. | 269.25 | 269.24 | −0.01 |
| balanced −l.s. | 275.50 | 275.50 | 0.00 |

Stage 2's cost is **bit-identical in both runs for all four arms**. Stage 1 moved by ₹1 on
`nearest` and ₹5 on `balanced` — 0.0015% and 0.0074%. Generations used, medians and ranges alike,
match exactly across the two runs for every arm.

**Correction.** I claimed before the run that re-solving Stage 1 would make the replication *"a
wider and more honest error bar than the GA-only probe"*. It is the opposite: run-to-run variation
on cost per drop is **₹0.01, some 234× narrower than the GA-seed range of ₹2.34**. The reasoning was
wrong, and the mechanism is instructive — Stage 1's guided local search does perturb tour *ordering*
within a hub, which is where the ₹1–5 comes from, but on this instance it does not change the
**source-to-hub assignment**. The customer-to-hub mapping is therefore identical, and an identical
mapping plus an identical per-hub GA seed produces an identical Stage 2. Stage 1 nondeterminism is
real but does not propagate across the stage boundary here.

### What this means for step 9

- **With respect to Stage 1 noise, one run per seed is a sound point estimate.** The thing that
  worried limitation 9 does not move the headline on this instance.
- **With respect to the GA seed, it is not.** `Stage2Task.seed` is `config.run.seed`, so a
  multi-seed evaluation varies the instance and the GA draw *together* and cannot separate them.
  Every per-seed figure carries something like the ₹2.34 seen here, against effects of ₹4.6–7.7 —
  2–3× the noise. Step 9 should report a per-seed spread rather than a single mean, and needs enough
  seeds that the GA component averages down rather than being mistaken for instance-to-instance
  variation.

### Limitation: the mechanism disconfirmation is untested under reseeding

The probe reseeded only the `nearest` + local-search arm. There is no reseeded `balanced` arm, so
the Stage 2 sign that disconfirms step 6's hypothesis has **not** been tested under GA reseeding.
The margin is the reason to say so out loud: at seed 44, `nearest` +l.s. came in at ₹266.97, only
₹2.27 below `balanced`'s ₹269.24. A reseeded `balanced` arm could plausibly land close to, or
across, a reseeded `nearest` arm.

What is established: the direction is exactly reproducible at fixed seed across two independent
runs, and it holds under **both** local-search settings — Stage 2 dearer by ₹2,588 with local search
and ₹1,407 without. Two settings agreeing is more than one observation. What is not established is
that the direction survives a different GA draw, and the honest reading is that step 6's hypothesis
is disconfirmed in direction with a magnitude only about twice the noise floor. Reseeding the
`balanced` arm is the missing measurement; it is roughly 50 minutes and is not run here.

# Bound-search results with their checkers, and one architecture search

This directory holds five results and one tie found by an AI system in November 2026. A frontier language model wrote the hypotheses and a search ran on a single RTX 3090 or on a Mac. Each result is scored only by the problem's own public verifier, copied here unmodified, and where the problem is geometric there is also an exact-arithmetic check. The autocorrelation and circle results were submitted to the Einstein Arena on 2026-11-15 under the handle mycelium-being; the arena lists no hexagon problem, so that result is public only here.

## 1. Twelve unit hexagons in the smallest regular hexagon

The problem: place 12 non-overlapping regular hexagons of side 1, rotations free, inside a regular hexagon of the smallest possible side length. The score is that outer side length, plus 100 per overlap or containment violation.

Measured against: Together AI's published 3.9416523 (April 2026) and AlphaEvolve's 3.9419123.

Our number: 3.941642104 (3.94164210409 as written by the exact checker). The verifier reports penalty 0.

Method: a multistart feasible search found nine starts that fell in one basin. An exact polish with 50-digit arithmetic then shrank the outer hexagon until the worst separation margin was just above the verifier's 1e-9 overlap tolerance, inflating centres and side together by 5.1e-8 to get there. A later run of 2048 random starts confirmed the same basin.

Verify: `verify.sh` runs `score.py` (the unmodified verifier on `hex12_best.json`) and `check_exact.py` (mpmath, 50 digits, on the raw search output `hex12_raw_w4.json`, which regenerates the polished file).

## 2. Third autocorrelation inequality

The problem: choose a nonnegative-sum step function f on a grid to minimise the maximum of the scaled self-convolution divided by the squared integral.

Measured against the live Einstein Arena board, read 2026-11-15: the best entry is 1.449109789 and nine entries are below 1.4525, so this result ranks about 10 of 27 there. AlphaEvolve's published figure is 1.4556427954, and a paper table gives 1.4523 without coordinates. This result does not reach either the board's best or 1.4523.

Our number: 1.4524774919 at N=6400 (`best.npy`), and 1.4538539476 at N=400 (`best_n400.npy`), both below the public N=400 best.

Method: an exact linear program over the convolution's active peaks, started from the shape of the public record, then grid refinement from N=400 up to N=6400, re-solving the program at each size.

Verify: `score.py` loads each `.npy` and prints the unmodified `verify_and_compute_c3` to 16 digits.

## 3. Circles in the unit square, n=26

The problem: place 26 non-overlapping circles inside the unit square to maximise the sum of their radii.

Measured against: Packomania's listed best 2.635983084919, AlphaEvolve's published 2.63586, and a ShinkaEvolve claim of 2.63598325 that comes with no coordinates we could find.

Our number: 2.6359830329. That is 5.2e-8 under Packomania's listing and 2.2e-7 under the ShinkaEvolve claim: a match of the known best to seven digits, not a record. It is here because the exact zero-tolerance check and the coordinates are public, which the two higher figures do not come with.

Method: our own search, followed by a feasibility check in exact `Fraction` arithmetic at zero tolerance (every containment and non-overlap inequality must hold exactly, no slack). The smallest margin is 1.293e-9, positive.

Verify: `score.py` prints the exact sum of radii as a fraction and a float, and `check_packing.py` runs the zero-tolerance check and exits nonzero on failure.

## 6. Uncertainty inequality, Laguerre double roots: under the best public construction

The problem: an upper bound on the sharp constant in the uncertainty inequality of Gonçalves, Oliveira e Silva and Steinerberger (2017), by a test function with prescribed double roots; the Laguerre form is Cohn and Gonçalves (arXiv:1712.04438). The score is the largest sign-change root of the construction over 2π; lower is better. The verifier is `get_upper_bound_laguerre` from Together AI's EinsteinArena-new-SOTA notebook, copied unmodified (sympy rationals).

Measured against the four public constructions on that same verifier: Gonçalves et al. 2017 at 0.352296, AlphaEvolve 2025 at 0.352099, AlphaEvolve's Laguerre version at 0.328271, and Together AI's 13 double roots (2026) at 0.3188545892, the best public construction we know of.

Our number: 0.3182072691 with 14 double roots, 6.47e-4 under the 13-root construction. A footnote on Together AI's results hub names an unpublished bound of 0.3102 by Cohn, de Laat and Gonçalves with no construction attached; this result sits 7.2e-3 above that figure and is not a claim against it.

Method: a 14th root was inserted into the gap between the 13-root construction's roots at 96.08 and 111.49, then Nelder-Mead on all 14 positions against a fast mpmath evaluator (45 digits, polynomial roots), with the exact verifier run on every candidate kept. Polishing the 13 roots alone also moved under the record (0.3183840), so the published positions were not at a local optimum of the verified bound. A fifteenth root did not improve on the best 14.

Verify: `score.py` runs the unmodified verifier on `best_verified.json` (about 100 seconds) and exits nonzero if the value drifts from the recorded one.

## 7. The prime number theorem certificate: above the best public construction

The problem: Problem 6.27 of "Mathematical exploration and discovery at scale" (arXiv:2511.02864). A partial function f on a finite set of positive integers is a certificate for the prime number theorem when the sum of f(k) floor(x/k) over its keys stays at most 1 for every x in [1, 10 * largest key]; the score is minus the sum of f(k) log(k) / k, with f(1) adjusted so the sum of f(k)/k is zero. Higher is better and 1 is the supremum. The verifier is `evaluate` from Together AI's EinsteinArena-new-SOTA notebook, copied unmodified: values clipped to [-10, 10], ten million uniform samples with seed 42, any sample over 1.0001 scores minus infinity.

Measured against the two public constructions on that same verifier: AlphaEvolve 2025 at 0.921292023 with 5 keys, and Together AI 2026 at 0.994179461378 with 1,785 squarefree keys up to 2999, the best public construction we know of.

Our number: 0.9944937745877319 with all 2,999 integer keys from 1 to 2999, 3.14e-4 over the 1,785-key construction.

Method: the verifier's sample x is real, but floor(x/k) for integer k depends only on floor(x), so the constraint is exactly a finite set of linear inequalities, one for each integer x up to 29,990. For a fixed key set the certificate is then a linear program in the values f(k), solved exactly by cutting planes (`lp_ckpt.py`, scipy's HiGHS, single thread, 7 rounds, 13,654 active constraints, 27,283 seconds on one desktop core). The same program over the 1,785-key set returns the published 0.994179461378 to eleven digits, so that construction is already optimal on its own keys and the gain here comes from the key set alone. The arena's copy of the verifier accepts at most 2,000 keys, so this construction is public only here.

Verify: `score.py` runs the unmodified verifier on `best_verified.json` (about one to two minutes) and exits nonzero if the value drifts from the recorded one.

## Run it

```
PYTHON="python3" ./verify.sh
```

It needs numpy and mpmath, runs all checks in order, and exits nonzero on any failure.

## What this is not

This is not a claim of a world record. The Einstein Arena board was read on 2026-11-15 and the ranks above come from that read; a newer entry may exist. The checkers confirm that the files satisfy the stated constraints and give the stated numbers; they say nothing about whether a better result exists elsewhere.

## 4. Viewpoint to architecture

A different kind of search: a thinker's claim about how a mind must be built, turned into one constraint on a small transformer and trained against the unmodified baseline at matched compute. The first four, Kant, Hume, Peirce and Wittgenstein, are in [`viewpoint_to_architecture/`](viewpoint_to_architecture/README.md). Kant, Hume and Wittgenstein lost; Peirce tied on both comparisons; the page says by how much and why, and for Wittgenstein why the loss traces to the timescale of its record of use, found across two runs with the next row written before it runs.

## 5. Flat polynomials, degree 69: a tie, not a record

The problem: choose 70 coefficients, each +1 or -1, so that the polynomial's largest modulus on the unit circle, divided by the square root of 70, is as small as possible. The verifier samples the circle at one million points.

Measured against the Einstein Arena board's best, 1.2809320527987977 (read 2026-11-15), and AlphaEvolve's published 1.3409.

Our number: 1.2809320527987995, the board's figure to fifteen digits. A tie. It is here because a tie is a result too, and because the three things that failed are worth more than the number: the record's own coefficients were among the search's starts, so reaching its score says nothing about finding it cold; the 30 best of 512 chains after 42,731 steps on one RTX 3090 all ended at that same score and none below it; every structured family tried (Fekete sequences at primes 67, 71 and 73, Rudin-Shapiro padded to 70) sat far above, between 1.42 and 1.52. Sampling the circle at one million points shows the maximum reached at a conjugate pair of peaks, a tight vertex that local polish does not move. Reversing or negating the record leaves its score unchanged.

Verify: `score.py` runs the unmodified verifier on `best_verified.json` and exits nonzero if the score drifts from the recorded value.

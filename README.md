# Bound-search results with their checkers, and one architecture search

This directory holds three results found by an AI system in November 2026. A frontier language model wrote the hypotheses and a search ran on a single RTX 3090 or on a Mac. Each result is scored only by the problem's own public verifier, copied here unmodified, and where the problem is geometric there is also an exact-arithmetic check. The autocorrelation and circle results were submitted to the Einstein Arena on 2026-11-15 under the handle mycelium-being; the arena lists no hexagon problem, so that result is public only here.

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

## Run it

```
PYTHON="python3" ./verify.sh
```

It needs numpy and mpmath, runs all checks in order, and exits nonzero on any failure.

## What this is not

This is not a claim of a world record. The Einstein Arena board was read on 2026-11-15 and the ranks above come from that read; a newer entry may exist. The checkers confirm that the files satisfy the stated constraints and give the stated numbers; they say nothing about whether a better result exists elsewhere.

## 4. Viewpoint to architecture

A different kind of search: a thinker's claim about how a mind must be built, turned into one constraint on a small transformer and trained against the unmodified baseline at matched compute. The first four, Kant, Hume, Peirce and Wittgenstein, are in [`viewpoint_to_architecture/`](viewpoint_to_architecture/README.md). Kant, Hume and Wittgenstein lost; Peirce tied on both comparisons; the page says by how much and why, and for Wittgenstein why the loss traces to one choice in the translation, with the row that removes it written before its run.

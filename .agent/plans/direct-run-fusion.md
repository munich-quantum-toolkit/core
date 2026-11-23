# Direct single-qubit run fusion

Status: complete.

## Outcome and scope

`ZFramePropagation.cpp` composes constant single-qubit runs and passes their
matrices directly to the shared ZYZ Euler conversion. This removes temporary U
operations and the separate greedy fusion pass. Its unused rewrite configuration
argument is removed from callers. Native-RZ-only targets retain their existing
path. Mapping is excluded.

`SingleQubitRunFusion` also composes each constant segment once and calls the
shared run-rewrite helper directly. Runtime composition retains its greedy
rewrites; the caller forwards its listener to the direct constant path.

Runs stop at symbolic gates, barriers, or block boundaries. Modifier bodies
remain owned by their enclosing unitary. Tail-first deletion repairs the forward
iterator, including interleaved wires. Numerical angles are materialized at the
head; later scalar SSA definitions are not moved or referenced there. Full
unitary semantics, including global phase, remain required; gate choices and
text need not match.

## Evidence

Against main revision `b6d06d45fc79df7a8e5f79858b02884962237521`, the final
performance evaluation covers 29 MQT Core benchmark instances: GHZ, QFT, QAOA,
Grover, and W-state at 4, 8, 12, 16, and 20 qubits, plus multiplexers at 5, 9,
13, and 17 qubits. Five alternating paired public compiler runs after warmup
used IQM Garnet, seed 2023, four mapping trials, and QCO output. Grover used
four iterations, QAOA depth four, and QFT period exponent three. Native ARM64
Clang 23 with ThinLTO and CPU affinity 5–8 ran on a shared host.

Median compiler times were 56.9 to 49.6 ms for QFT-20, 552.5 to 470.2 ms for
Grover-12, and 2283.5 to 1907.3 ms for Grover-20. The geometric mean speedup
across all 29 cases was 1.062. W-state-4 was 0.6 ms slower (8.9 to 9.5 ms);
short runs remain sensitive to startup and shared-host variation. These are
workload-specific measurements, not isolated-host performance guarantees.

R and two-qubit gate counts match the baseline on all 29 cases. Matrix-derived
polar angles use `MATRIX_TOLERANCE`, matching Euler emission. Grover-12 and
Grover-20 emit 10807 and 38299 R gates. The matrix tolerance avoids emitting
roundoff as tiny physical rotations while keeping stricter scalar-parameter
checks unchanged.

Candidate-plan construction and counting in `DecomposeMultiControlled.cpp` was
also investigated. Temporary timing probes over 64 repeated MCX gates with 7
through 127 controls attributed approximately 1–3.4% of isolated decomposition
time to candidate planning. However, 32 repeated controlled-phase gates with
many clean ancillas spent 9–51% there at 15 through 127 controls. Their
ancilla-free fallback grows quadratically even when the accepted clean-ancilla
construction is linear.

Parameterized candidates now use exact CX-equivalent counts without constructing
plans. The pass builds only the accepted circuit with its actual angle. Tie
behavior is unchanged; nonparameterized plan construction remains unchanged.
Formula checks belong in tests: constructing an oracle plan in an assertion
retains the unwanted work in assertion-enabled optimized builds.

Five alternating paired runs after warmup, with the same baseline and build
configuration, measured isolated decomposition of 32 controlled-phase gates with
many clean ancillas. Median times at 15, 31, 63, and 127 controls were 31.6 to
29.4, 69.7 to 55.9, 185.2 to 121.0, and 613.0 to 260.9 ms. All paired outputs
were byte-identical. At width 127, the RZ control improved by 3.1% and MCX was
0.5% slower.

## Validation

The 110 target-synthesis tests pass, including full-unitary equivalence,
linearity, target conformance, runtime angles, and scalar dominance. The added
interleaved-run regression fails under a deliberate matrix multiplication-order
mutation. That mutation and all timing instrumentation have been removed.

The clean-ancilla regression covers 3, 4, 8, and 9 controls, checking selection
costs and full ancilla restoration across special cases and algorithm
boundaries. The focused decomposition tests pass. Independent reviews found no
remaining correctness or complexity issues in either change.

The 371 decomposition and Euler tests also pass after the cleanup. Regression
checks cover both sides of the matrix-zero threshold and resumption after a
symbolic gate; both fail under targeted mutations, which have been removed. All
4377 native tests, whole-file C++ checks, and repository lint pass. Hosted CI
and other platforms were not run.

# Dynamic Shor benchmark

Status: complete. Published in #2605 with exact arithmetic, unified payload
examples, larger program-size budgets, and measured simulation performance.

## Supported behavior

`Shor` and Python `bench.shor.Shor` implement exact semiclassical order finding
with reusable controlled modular arithmetic. Inputs are an odd `number` from 3
through `2**31 - 1` and a coprime `base` satisfying `1 < base < number` (default
2). An n-bit input uses `2n+3` qubits and returns `2n` big-endian bits in
`result`. Repeated-squaring tables have polynomial size; generation never
computes the order. All classical arithmetic fits in 64-bit integers.

Evaluation checks continued-fraction candidates with modular exponentiation and
gcds, returning optional sorted factors and the shot-weighted success fraction.
JSON uses a verification reference, without ideal-distribution metrics. The
registry, CLI, native API, Python bindings, and generated stubs share this
model.

The factoring driver handles even numbers, primes, and perfect powers
classically, then tries base 2 followed by seeded random bases (16 attempts by
default). It returns one verified pair, prime status, or exhausted attempts. The
callback owns device selection, compilation, shots, and execution seeds; errors
propagate. The README and walkthrough run the same factoring-21 example through
Adaptive QIR and OpenQASM 3.

## Compiler and execution

The implementation uses the borrowed-register and stable-slot contracts merged
in #2546. Fourier arithmetic is shared with the modular-multiplier benchmark.
There is no approximate-QFT option.

Payload unrolling and OpenQASM QC emission allow one billion operations by
default; textual expansion and semantic analysis allow 100 million statements.
The DD interpreter allows one billion while iterations per execution by default,
shared across nested loops and calls. Counted loops and total gates have no
execution step limit. The unrolling pass and native DD API expose budget
overrides so boundary tests stay small and callers can control resource use.
Nesting, numeric widths, include-expansion checks, and memory-overflow checks
retain their separate purpose.

## Validation

Acceptance covers exact factor recovery and JSON validation, small arithmetic
basis states and coherent superpositions, inverse composition with clean
workspace, sampled Shor distributions for 15, 21, and 35, and structured 31-bit
generation/Adaptive QIR compilation. The largest input's 65-qubit state is not
simulated. Both device formats use the same 15/21 cases. The optimized native
build passes 3,602 tests with one expected Slurm skip; 118 Python benchmark and
compilation tests pass. Stub generation, whole-file C++ lint, executable docs,
and rendered-link checks pass. Repository lint reports only 28 pre-existing
diagnostics in unrelated untracked audit scripts; the pinned type checker passes
with those scripts excluded.

A flat 20,000,001-gate single-qubit X program was imported and verified from
OpenQASM. A counted loop executing 20,000,001 X gates was simulated with the
expected final state. Small boundary tests cover budget overrides, nested/call
accounting, cumulative cloning, and unrepresentable trip counts. One full
complexity review removed stored phase-helper state, a redundant control-count
branch, and an extra result lookup. Supported multiplication retains jeff
compatibility.

## Simulation timings

Measured on 2026-09-23 on DGX Spark, pinned to one Cortex-X925 core (CPU 5).
Exact circuits, base 2, seed 17, 64 shots, median of three fresh processes.
Direct DD and both DDSIM payload paths use the native Clang 23 `-O3`/ThinLTO
build. QDMI execution includes submission, payload import or JIT, waiting, and
count retrieval; generation and compilation are timed separately. The Python
compiler uses the local GCC `-Os` build, so preparation timings also reflect
that toolchain difference. The host is shared; these are local measurements, not
isolated machine throughput guarantees. Every run recovered valid factors.

|    N | Qubits | Direct QCO DD | QIR via QDMI | OpenQASM via QDMI |
| ---: | -----: | ------------: | -----------: | ----------------: |
|   15 |     11 |       0.697 s |      0.306 s |           0.253 s |
|   21 |     13 |       2.875 s |      1.976 s |           2.317 s |
|   35 |     15 |      17.185 s |     11.660 s |          20.656 s |

Preparation times in the same order (direct / QIR / OpenQASM):

- N=15: 0.014 s / 0.288 s / 1.015 s.
- N=21: 0.012 s / 0.078 s / 2.168 s.
- N=35: 0.007 s / 0.080 s / 4.886 s.

Harnesses, raw repetitions, and validation logs are outside the repository at
`/tmp/mqt-shor-revisions-20260922`.

## Limits

The input bound controls representation size, not simulation capacity. Large
programs still need sufficient memory and time. Mapped OpenQASM specializes
physical-qubit indices; Adaptive QIR can retain the indexed loops. See the
[unrolling explanation](../audits/shor-payload-unrolling.md).

Dynamic-size helper signatures, register ownership transfers, recursive
factoring, wider integers, and distillation remain outside this change.

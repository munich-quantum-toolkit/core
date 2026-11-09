# Test suite consolidation

## Goal and decisions

Reduce repeated CI work and maintenance while retaining
SDK-to-compiler-to-device workflows. Native semantics belong in C++; Python
tests cover SDK adaptation, bindings, ownership, and public workflows. Prefer
independent semantic oracles and resource bounds over implementation snapshots.
Keep distinct phase, output-order, ownership, failure, and numerical-limit
contracts.

The development policy and agent guide now set these priorities. Codecov uses an
overall 90% project target with a one-percentage-point tolerance and an
informational patch status. Language flags remain available for diagnosis;
per-language automatic coverage ratchets no longer drive test additions.

## Consolidation and retained oracles

| Area                         | Removed work                                                               | Retained oracle                                                                                                                                               |
| ---------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| QC/QCO/QIR/QTensor IR        | 327 same-builder comparisons                                               | Different independently built programs, verification, and semantic comparisons                                                                                |
| QC/QCO conversion            | 119 duplicate reverse fixtures                                             | Both directed comparisons against independently constructed expected programs, in one shared fixture                                                          |
| Native QDMI metadata         | Repeated getter checks across six devices                                  | One local calibration/topology contract, IQM target snapshots, driver calibration, and existing job/ownership/error workflows                                 |
| Python QDMI                  | Native property sweeps and repeated job getters                            | DDSIM execution and stable counts, explicit metadata, argument conversion, lifetimes, GIL release, and failure behavior                                       |
| Python benchmarks            | Separate native-reference/getter suites                                    | Fourteen manifest/instance round trips through generation, QC/QCO, jeff bytes, sampling, and evaluation; Python-specific constructors/callbacks/errors remain |
| Native arithmetic benchmarks | 648 basis-case compilations and 32 shots each                              | 22 representative basis cases with one deterministic shot; coherent superposition and phase checks remain                                                     |
| SDK/compiler interchange     | Cartesian option grids and IR-shape assertions                             | Representative API paths, independent Qiskit operators, signedness, wire order, wide-integer boundaries, and resource upper bounds                            |
| Oversized inputs             | 100-qubit symbolic SU2, 1,024-angle runs, obsolete 70,000-gate unroll case | Small full-unitary comparisons, a 65-angle depth-boundary case, and current loop-budget contracts                                                             |
| Internal edge cases          | Three fabricated non-unitary/non-finite Euler cases                        | Valid-unitary decomposition and public dense-unitary input validation                                                                                         |
| CLI                          | Sixteen source/output equivalence subprocesses                             | Two representative checkpoints, native CLI coverage, and Python-specific QPY/provider/error workflows                                                         |

The decomposition suite keeps coherent-state checks at the SP22/HP24 crossovers
and wide odd/even sizes, with resource upper bounds in those same checks. This
replaces 52 count-only smoke cases and redundant width grids, saving 240
generated modules. QDMI session-construction checks use one combined
configuration; driver runtime tests retain token transport, isolation, and
recovery coverage.

Python DD tests retain complex-array conversion, strides, ownership, control
polarity, serialization, and exception classes without repeating native gate and
state sweeps or pinning DD node counts. CLI target export uses four selected
cases spanning explicit QIR/QPY payload specifications and embedded QC/QCO
targets. Shor and QPE pair algorithms with payload formats rather than crossing
every combination; a small and a 65-bit case retain local-bit execution.

Qiskit capture tests execute nested control flow with reordered and
condition-only captures instead of inspecting import IR. The register-shift case
now takes an observable true branch; post-test loops use even and odd iteration
counts. Loop helpers check each format once and keep determinism in one focused
workflow. Integer helpers retain four format checks before cleanup and one
semantic execution after it, without repeating every export.

Benchmark distribution thresholds in Python are binding smoke checks. Native
sampling tests own discrimination against incorrect distributions. The
teleportation structural test remains because its current sampling input does
not distinguish a missing X correction. No production branches were removed:
unreachable-path cleanup requires a separate proof of the supported input
contract.

## Validation

Baseline: `2af9b2df72`, Linux ARM64, Clang/LLVM/MLIR 23, native release with
ThinLTO. Collection and coverage were compared against the original suite using
the same production code and dependencies.

| Measurement                                      |        Original |         Reduced |
| ------------------------------------------------ | --------------: | --------------: |
| Native test cases, including grouped GoogleTests |           5,355 |           4,511 |
| Registered CTest entries                         |           4,391 |           3,763 |
| Passing Python cases, Python 3.14                |           2,305 |           1,662 |
| Native covered lines                             | 56,097 / 60,003 | 56,094 / 60,003 |
| Native line coverage                             |         93.490% |         93.485% |
| Python covered lines                             |   1,069 / 1,110 |   1,069 / 1,110 |
| Python line coverage                             |         96.306% |         96.306% |

Two alternating timing pairs, including coverage and four pytest workers, took
63.18/62.85 seconds for the original suite and 15.15/14.94 seconds for the
reduced suite: about 76% less wall time on average. Both versions used Python
3.14.10, the same installed package and dependencies, and the same source
fixtures. Builds and lint were stopped during timing; the host was shared, so
this is a local comparison rather than a hosted-CI prediction. The largest
removed cost was the pair of 100-qubit symbolic SU2 cases, each taking about 25
seconds.

The Python covered-line set is unchanged. Native coverage loses three lines in
Euler's non-unitary/non-finite fallbacks and one optional-empty QDMI site-pair
return. QIR death-test error-return coverage varies between runs; the unchanged
range and released-resource rejection tests remain. Two DD worker error-path
lines appear in the final run, leaving a net loss of three covered lines.
Coverage is measured separately for native source and Python source; this does
not claim coverage of every nanobind adapter or equivalence to hosted Codecov.

The original and reduced release suites pass in full. Both Clang debug coverage
runs above encountered 24 IQM/resonator failures in LLVM JSON (`Unknown kind`).
These share a local toolchain cause: the portable LLVM SDK was built with GCC,
and its weak inline JSON callback interprets a Clang-built lambda's captures in
the wrong order. A standalone JSON writer reproduces the abort without Core;
debugger inspection confirms the swapped stream and value pointers. Optimized
Clang builds inline away the conflicting callback.

Using GCC for the debug coverage build resolves all 24 failures with coverage
and assertions enabled. The full reduced suite then passes all 3,763 CTest
entries. No production workaround or test changes are needed. The table retains
the paired Clang measurements, including their shared coverage limitation; GCC
instrumentation is not substituted into that comparison.

Python 3.14 regular dependencies pass all 1,662 cases, with two optional
Catalyst skips. Python 3.11 minimum dependencies pass all 1,662 cases, with
three skips. C++ lint inspected every changed file in each implementation batch,
including full-file checks on the four follow-up native files. Independent
specialist review checked the retained contracts in both batches. Repository
lint passes.

Reproduction uses `cmake --preset release-clang-ipo`, the corresponding build
and CTest presets, `uv run --no-sync pytest -n 4 --cov`, and
`uvx nox --non-interactive -s minimums-3.11 -- -n 4`. The release preset is the
local machine override; ordinary contributors use the repository release preset.
The local `coverage-gcc` configure, build, and CTest presets inherit the
repository coverage settings and select GCC to match the portable LLVM SDK's
compiler family. Compare coverage using the same filters and build configuration
on both revisions. Raw logs and coverage reports are retained outside the source
tree.

Hosted CI, the full interpreter/dependency/platform matrix, optional Catalyst
Python integration, external QPUs, and rendered documentation were not checked.

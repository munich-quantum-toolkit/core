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
| QC/QCO IR                    | 216 same-builder comparisons                                               | Different independently built programs, verification, and semantic comparisons                                                                                |
| QC/QCO conversion            | 119 duplicate reverse fixtures                                             | Both directed comparisons against independently constructed expected programs, in one shared fixture                                                          |
| Native QDMI metadata         | Repeated getter checks across six devices                                  | One local calibration/topology contract, IQM target snapshots, driver calibration, and existing job/ownership/error workflows                                 |
| Python QDMI                  | Native property sweeps and repeated job getters                            | DDSIM execution and stable counts, explicit metadata, argument conversion, lifetimes, GIL release, and failure behavior                                       |
| Python benchmarks            | Separate native-reference/getter suites                                    | Fourteen manifest/instance round trips through generation, QC/QCO, jeff bytes, sampling, and evaluation; Python-specific constructors/callbacks/errors remain |
| Native arithmetic benchmarks | 648 basis-case compilations and 32 shots each                              | 22 representative basis cases with one deterministic shot; coherent superposition and phase checks remain                                                     |
| SDK/compiler interchange     | Cartesian option grids and IR-shape assertions                             | Representative API paths, independent Qiskit operators, signedness, wire order, wide-integer boundaries, and resource upper bounds                            |
| Oversized inputs             | 100-qubit symbolic SU2, 1,024-angle runs, obsolete 70,000-gate unroll case | Small full-unitary comparisons, a 65-angle depth-boundary case, and current loop-budget contracts                                                             |
| Internal edge cases          | Three fabricated non-unitary/non-finite Euler cases                        | Valid-unitary decomposition and public dense-unitary input validation                                                                                         |
| CLI                          | Sixteen source/output equivalence subprocesses                             | Two representative checkpoints, native CLI coverage, and Python-specific QPY/provider/error workflows                                                         |

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
| Native test cases, including grouped GoogleTests |           5,355 |           4,755 |
| Registered CTest entries                         |           4,391 |           4,007 |
| Passing Python cases, Python 3.14                |           2,305 |           1,744 |
| Native covered lines                             | 56,097 / 60,003 | 56,092 / 60,003 |
| Native line coverage                             |         93.490% |         93.482% |
| Python covered lines                             |   1,069 / 1,110 |   1,069 / 1,110 |
| Python line coverage                             |         96.306% |         96.306% |

Two alternating timing pairs, including coverage and four pytest workers, took
63.85/63.36 seconds for the original suite and 16.21/18.38 seconds for the
reduced suite: about 73% less wall time on average. Both versions used Python
3.14.10, the same installed package and dependencies, and the same source
fixtures. Builds and lint were stopped during timing; the host was shared, so
this is a local comparison rather than a hosted-CI prediction. The largest
removed cost was the pair of 100-qubit symbolic SU2 cases, each taking about 25
seconds.

The Python covered-line set is unchanged. Native coverage loses three lines in
Euler's non-unitary/non-finite fallbacks and one optional-empty QDMI site-pair
return. A QIR error-return line also differs, although the unchanged
`RejectsInvalidDynamicResourceUse` death test still exercises that rejection.
Coverage is measured separately for native source and Python source; this does
not claim coverage of every nanobind adapter or equivalence to hosted Codecov.

The original and reduced release suites pass in full. The debug coverage build
has the same 24 IQM/resonator failures before and after the change, aborting in
LLVM JSON with `Unknown kind`. Those tests pass in release. These pre-existing
failures limit the coverage result and remain visible.

Python 3.14 regular dependencies pass, with two optional Catalyst skips. Python
3.11 minimum dependencies pass, with three skips; the final Python assertion
edits also pass a focused 280-case minimum-dependency rerun. Repository lint and
full-changed-file C++ lint are part of the final checks.

Reproduction uses `cmake --preset release-clang-ipo`, the corresponding build
and CTest presets, `uv run --no-sync pytest -n 4 --cov`, and
`uvx nox --non-interactive -s minimums-3.11 -- -n 4`. The release preset is the
local machine override; ordinary contributors use the repository release preset.
Compare coverage using the same filters and build configuration on both
revisions. Raw logs and coverage reports are retained outside the source tree.

Hosted CI, the full interpreter/dependency/platform matrix, optional Catalyst
Python integration, external QPUs, and rendered documentation were not checked.

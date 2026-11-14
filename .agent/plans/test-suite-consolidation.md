# Test suite consolidation

## Goal and decisions

Reduce repeated CI work and maintenance while retaining
SDK-to-compiler-to-device workflows. Native semantics belong in C++; Python
tests cover SDK adaptation, bindings, ownership, and public workflows. Prefer
independent semantic oracles and resource bounds over implementation snapshots.
Keep distinct phase, output-order, ownership, failure, and numerical-limit
contracts.

Aggregate independent observations into one small workflow when they share setup
and conversion work. Each observation still needs a distinct oracle; reducing
the number of test registrations alone is not the objective.

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

Integer comparisons pack ten predicate results into distinct output bits. This
keeps all predicates, three widths, and four interchange paths while reducing 30
programs to three. Python mapping options now feed one routed execution; native
tests own repeatability and the full version/metadata contracts. Typed phase
normalization and decomposition share a small full-unitary workflow.

Native benchmark tests omit 20 redundant generation/jeff byte round trips and
eight repetitions of the common JSON evaluation formatter. Python workflows
cover generation through execution; GHZ/BV retain both common formatter branches
and QAOA/Shor retain their distinct formats. Native target tests combine
calibration, maximum-site-ID, and copy-lifetime checks with attribute round
trips. One decomposition workflow checks both typed and textual APIs against the
full unitary. The topology concurrency check uses eight sites and a start
barrier, reducing 8,192 path checks to 128 while exercising shared lazy
initialization.

Benchmark distribution thresholds in Python are binding smoke checks. Native
sampling tests own discrimination against incorrect distributions. The
teleportation structural test remains because its current sampling input does
not distinguish a missing X correction. No production branches were removed:
unreachable-path cleanup requires a separate proof of the supported input
contract.

The Shor in-place multiplier workflow compiles eight representative odd-modulus
cases instead of all 71 coprime pairs below 16. Its sole production caller,
Shor, rejects even moduli. Identity, inverse, prime/composite, and two-, three-,
and four-bit cases retain every valid basis column, both control values, complex
superposition, and clean-workspace checks. The general modular multiplier still
covers even moduli. The focused release test falls from 5.55 to 0.45 seconds,
with peak RSS falling from about 92 to 68 MiB.

Ancilla workflows compare complete output columns as DD states instead of
visiting every dense matrix entry. Every valid input column remains checked;
subtracting states before computing the norm retains phase sensitivity and
leakage detection. Eight focused release cases fall from 0.70 to 0.14 seconds;
the coverage build falls from 16.06 to 1.39 seconds. Temporary mutation probes
confirmed that global phase, leaked ancillas, and wrong selected columns fail.

The 100-parameter Qiskit test bulk-binds its independent reference expression.
It retains all symbols, parameter identity, nonlinear rotation, global phase,
and ordinary binding of the exported circuit. The focused regular-dependency
test falls from 4.55 to 0.25 seconds; wrong-axis and wrong-phase probes fail.
These changes reduce work without changing test counts. Focused timings are
local shared-host measurements, not CI predictions.

## Validation

Test baseline: `2af9b2df72`; the PR is rebased onto `97cefb6cfb`, which changes
documentation and comments without changing either test tree. Validation uses
Linux ARM64, Clang/LLVM/MLIR 23, and native release with ThinLTO. Paired Python
measurements use the same installed package and dependencies. The baseline
native coverage run used the packaged SDK; the final run uses the
compiler-matched SDK described below.

| Measurement                                      |        Original |         Reduced |
| ------------------------------------------------ | --------------: | --------------: |
| Native test cases, including grouped GoogleTests |           5,355 |           4,497 |
| Registered CTest entries                         |           4,391 |           3,749 |
| Passing Python cases, Python 3.14                |           2,305 |           1,625 |
| Native covered lines                             | 56,097 / 60,003 | 56,114 / 60,003 |
| Native line coverage                             |         93.490% |         93.519% |
| Python covered lines                             |   1,069 / 1,110 |   1,069 / 1,110 |
| Python line coverage                             |         96.306% |         96.306% |

Two alternating timing pairs, including coverage and four pytest workers, took
61.30/57.67 seconds for the original suite and 16.82/16.39 seconds for the
reduced suite: about 72% less wall time on average. Both versions used Python
3.14.10, the same installed package and dependencies, and the same source
fixtures. Builds and lint were stopped during timing; the host was shared, so
this is a local comparison rather than a hosted-CI prediction. The largest
removed cost was the pair of 100-qubit symbolic SU2 cases, each taking about 25
seconds.

The Python covered-line set is unchanged. Native coverage gains 17 covered lines
overall, with IQM compilation paths completing after the SDK repair. Removed
cases leave the defensive Euler fallbacks, an optional-empty QDMI site-pair
return, an isolated CLI threshold predicate, and one nonmatching gate-fusion
return uncovered. Invalid decomposition thresholds and full-unitary fusion
workflows remain tested. QIR death-test error-return coverage varies between
runs; range and released-resource rejection tests remain. Two DD worker
invalid-count lines also vary between full runs; this pass does not change their
tests. Coverage is measured separately for native source and Python source; this
does not claim coverage of every nanobind adapter or equivalence to hosted
Codecov.

The original and reduced release suites pass in full. Clang debug coverage with
the packaged SDK encountered 24 IQM/resonator failures in LLVM JSON
(`Unknown kind`). These share a local toolchain cause: the portable LLVM SDK was
built with GCC, and its weak inline JSON callback interprets a Clang-built
lambda's captures in the wrong order. A standalone JSON writer reproduces the
abort without Core; debugger inspection confirms the swapped stream and value
pointers. Optimized Clang builds inline away the conflicting callback.

The local LLVM/MLIR 23.1.0 SDK is now built with Clang 23, with assertions and
ABI checks enabled. The canonical SDK path selects this build; the original SDK
is preserved separately. Both the ordinary Clang coverage preset and the
Clang/ThinLTO release preset pass all 3,749 CTest entries, including the 24
previous failures. No production workaround, compiler-switch preset, or test
suppression is needed.

Python 3.14 regular dependencies pass all 1,625 cases, with two optional
Catalyst skips. Python 3.11 minimum dependencies pass all 1,625 cases, with
three skips. Python extension builds use Clang/ThinLTO locally; GCC 13
encountered an LTO internal compiler error before the minimum-dependency tests
could run. C++ lint passes full-file checks on all 27 changed C++ files,
including the corrected trailing commas in the decomposition parameter arrays.
Independent specialist review checked the retained contracts and aggregated
oracles. Repository lint passes.

Reproduction uses `cmake --preset release-clang-ipo`, the corresponding build
and CTest presets, `uv run --no-sync pytest -n 4 --cov`, and
`uvx nox --non-interactive -s minimums-3.11 -- -n 4`. The release preset is the
local machine override; ordinary contributors use the repository release preset.
Debug coverage uses the repository's ordinary `coverage` configure, build, and
CTest presets with Clang. Raw logs, coverage reports, and the local SDK build
manifest are retained outside the source tree.

Hosted clang-tidy diagnostics were inspected. Final-head hosted CI, the full
interpreter/dependency/platform matrix, optional Catalyst Python integration,
external QPUs, and rendered documentation were not checked.

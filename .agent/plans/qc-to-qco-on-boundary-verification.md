# Source-ordered QC → QCO on boundary verification

Status: complete. Source-ordered conversion and targeted normalization are
implemented and locally validated against the pinned base. End-to-end performance
is mixed; whole-file C++ lint and human review remain merge gates.

## Scope and decisions

The implementation changes only `mlir/lib/Conversion/QCToQCO/QCToQCO.cpp`
and `mlir/unittests/Conversion/QCToQCO/test_qc_to_qco.cpp`.
Use ordinary rewrite patterns in explicit source order. Keep source reference
identities alive until their users have been lowered; move nested regions
without destroying their original children. Preserve type conversion, static
normalization, supported input checks, and linear output validation.

Reusing preflight state only when normalization is skipped did not improve the
selected workloads: OpenQASM register inputs always require normalization.
That experiment is not retained. Normalization can invalidate source identities,
so eliminating post-normalization collection needs a stronger invariant.

Sampling the factor247 conversion identified region liveness cleanup inside the
greedy normalizer as a substantial cost. Use MLIR's existing
`GreedySimplifyRegionLevel::Disabled` for normalization; this pass needs index
folding and static-reference normalization, not control-flow optimization.

The baseline is PR #2698 at `2cf5d436150a6fe953fbc6d8f0c44f126bba3c1f`.
Keep its QCO → QC lowering, boundary-verification policy, mapping, and all other
pipelines unchanged. Existing ordered reverse-conversion work is outside scope.

## Validation

Both variants use the Release preset, standalone native builds, and the same
assertion-free LLVM/MLIR 23.1.0 toolchain. Run the QCToQCO, QCOToQC, and Compiler
GoogleTest targets under `build/release/mlir/unittests/` to reproduce the native
checks. All 602 baseline tests and 605 final candidate tests pass. The three
added tests cover interleaved call signatures with owned results, unitary calls
inside nested regions, and register aliasing exposed by index folding.

AddressSanitizer with assertions passes all 174 forward-conversion tests. The
converter and its test translation unit are instrumented; dependencies are not.
Leak detection is disabled. Repository lint (`uvx nox -s lint`) and
`git diff --check` pass.

The performance comparison uses frozen Benchpress inputs from revision
`045e7ca6a8e5302826d82f07d42b5689ca61db3c`: QFT with 18 and 160 qubits, QRAM with
20 qubits, Ising with 26 qubits, cat with 260 qubits, W-state with 380 qubits,
and factor247 with 15 qubits. Seven unique circuits cover forward conversion;
seven circuit/topology pairs cover full compilation, excluding QFT-160 and
including QFT-18 on both all-to-all and linear connectivity.

The final direct comparison includes both changes against unmodified PR #2698,
not a product of earlier incremental ratios. Each pair has three serial rounds
with alternating variant order and seed 42. All 84 samples pass: 42 per variant,
without failures, errors, skips, timeouts, or interrupted samples. Source, input,
configuration, binary, and output hashes are checked. Every paired output IR
and returned gate count is identical, including repetitions.

Forward conversion has seven complete pairs: geometric candidate/baseline ratio
0.616, with summed case means of 11.393 s before and 6.460 s after. Factor247
improves from 10.520 s to 5.968 s; QFT-160 improves from 0.827 s to 0.461 s.

Full compilation also has seven complete pairs: geometric ratio 0.986, but
summed case means increase from 65.070 s to 70.430 s. QFT-18/linear regresses
13.5% and factor247/square regresses 8.3%. Factor247's mean target-compilation
phase rises from 47.344 s to 56.035 s while its forward conversion falls from
10.561 s to 6.608 s. The cause of the end-to-end regression is not established.
Keep all samples and do not claim a universal full-compilation speedup.

A fresh isolated wheel passes 977 focused Python tests with no skips:
`test_mlir.py`, `test_mlir_loops.py`, `test_mlir_qiskit_translation.py`,
`test_mlir_qiskit_target.py`, `test_mlir_parameter_binding.py`, and
`test_mlir_integer_interchange.py`. Run them with `python -m pytest` and
`-o addopts=` in the wheel's environment.

Forward-conversion timing excludes the input copy. Full timing includes the QC
copy, QC → QCO, target compilation, and QCO → QC. QASM loading, target setup,
extra output verification, inspection, and serialization are outside timers.
The native targets use frozen FlexibleBackend connectivity and the common
`sx`, `x`, `rz`, `cz`, measurement, and reset basis without calibration metadata.

## Limits

This is a native API comparison on selected Benchpress inputs, not a run of the
pytest-benchmark suite. Each fresh process performs one timed call, with no
warmup or adaptive calibration and a 600-second whole-process deadline. Three
fixed-seed rounds on a shared desktop do not establish statistical significance
for small differences. Two-qubit depth is not independently measured.
Whole-changed-file C++ lint was attempted against the pinned revision but
stopped before checking any files because clang-tidy 23 is unavailable locally.
The supported conversion subset is unchanged; this work does not add support
for unstructured control flow.

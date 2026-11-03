# Source-ordered QC → QCO on boundary verification

This plan records PR #2716 before combination with the QC/QCO audit. See the
[combined prototype](../audits/qc-qco-combined-2716-20261018.md) for current
source, validation, and performance. Historical end-to-end performance below is
mixed. Human review remains a merge gate.

## Scope and decisions

The implementation changes only `mlir/lib/Conversion/QCToQCO/QCToQCO.cpp` and
`mlir/unittests/Conversion/QCToQCO/test_qc_to_qco.cpp`. Use ordinary rewrite
patterns in explicit source order. Keep source reference identities alive until
their users have been lowered; move nested regions without destroying their
original children. Preserve type conversion, static normalization, supported
input checks, and linear output validation.

Append a complete converted function argument list, redirect classical uses, and
retain the original quantum references until lowering finishes. Remove the old
argument prefix in one operation. Individual middle insertions and erasures were
quadratic for wide signatures; bulk replacement preserves interleaved argument
order, attributes, and locations without that cost.

The rebase onto `575acd1b9` retains main's #2756 preflight-state reuse when
normalization is skipped. Recollect register accesses after normalization
because it can invalidate source identities. The historical selected OpenQASM
register workloads always normalized, so they do not measure this reuse path.

Sampling the factor247 conversion identified region liveness cleanup inside the
greedy normalizer as a substantial cost. Use MLIR's existing
`GreedySimplifyRegionLevel::Disabled` for normalization; this pass needs index
folding and static-reference normalization, not control-flow optimization.

The historical benchmark baseline is PR #2698 at
`2cf5d436150a6fe953fbc6d8f0c44f126bba3c1f`. QCO → QC lowering,
boundary-verification policy, mapping, and all other pipelines remain outside
the scope of this patch. Existing ordered reverse-conversion work is not
included.

## Validation

The 2026-10-15 rebase passes 619 native tests: 176 QCToQCO, 154 QCOToQC, and 289
Compiler, with two expected macOS `/dev/full` skips. Executable docs, repository
lint, and whole-file C++ lint pass. The wide-signature test's string
construction was adjusted for current lint; all 176 forward-conversion tests
passed again. No benchmarks or sanitizer runs were repeated for this rebase.

### Historical validation and performance

Both historical variants use the Release preset, standalone native builds, and
the same assertion-free LLVM/MLIR 23.1.0 toolchain. Run the QCToQCO, QCOToQC,
and Compiler GoogleTest targets under `build/release/mlir/unittests/` to
reproduce the native checks. The pre-rebase candidate passed 606 tests: 176
QCToQCO, 158 QCOToQC, and 272 Compiler. Added tests cover interleaved call
signatures with owned results, unitary calls inside nested regions, register
aliasing exposed by index folding, wide interleaved signatures, and remapped
register index arguments.

AddressSanitizer with assertions passes all 176 forward-conversion tests. The
converter and its test translation unit are instrumented; dependencies are not.
Leak detection is disabled. Repository lint (`uvx nox -s lint`) and
`git diff --check` pass.

Bulk argument replacement also passes 138 differential comparisons against
`0dde327c1`, with identical output for forward conversion and roundtrips. These
cover nested structured control flow, register and scalar qubits, calls,
attributes, and helper declaration order.

For private helpers with 10,000, 20,000, and 40,000 unused borrowed qubit
arguments, median whole-CLI time falls from 151/649/2156 ms to 24/37/67 ms. Each
size uses three serial, alternating before/after pairs with identical output,
running only `builtin.module(qc-to-qco)` through `mqt-cc --run-pipeline`. Both
binaries use the same Release toolchain. These synthetic timings include parsing
and output; they are not isolated pass timings or Benchpress results.

The following historical performance comparison predates the bulk-argument fix
and does not measure the current source. It uses frozen Benchpress inputs from
revision `045e7ca6a8e5302826d82f07d42b5689ca61db3c`: QFT with 18 and 160 qubits,
QRAM with 20 qubits, Ising with 26 qubits, cat with 260 qubits, W-state with 380
qubits, and factor247 with 15 qubits. Seven unique circuits cover forward
conversion; seven circuit/topology pairs cover full compilation, excluding
QFT-160 and including QFT-18 on both all-to-all and linear connectivity.

The final direct comparison includes both changes against unmodified PR #2698,
not a product of earlier incremental ratios. Each pair has three serial rounds
with alternating variant order and seed 42. All 84 samples pass: 42 per variant,
without failures, errors, skips, timeouts, or interrupted samples. Source,
input, configuration, binary, and output hashes are checked. Every paired output
IR and returned gate count is identical, including repetitions.

Forward conversion has seven complete pairs: geometric candidate/baseline ratio
0.616, with summed case means of 11.393 s before and 6.460 s after. Factor247
improves from 10.520 s to 5.968 s; QFT-160 improves from 0.827 s to 0.461 s.

Full compilation also has seven complete pairs: geometric ratio 0.986, but
summed case means increase from 65.070 s to 70.430 s. QFT-18/linear regresses
13.5% and factor247/square regresses 8.3%. Factor247's mean target-compilation
phase rises from 47.344 s to 56.035 s while its forward conversion falls from
10.561 s to 6.608 s. The cause of the end-to-end regression is not established.
Keep all samples and do not claim a universal full-compilation speedup.

The historical isolated wheel passed 977 focused Python tests with no skips:
`test_mlir.py`, `test_mlir_loops.py`, `test_mlir_qiskit_translation.py`,
`test_mlir_qiskit_target.py`, `test_mlir_parameter_binding.py`, and
`test_mlir_integer_interchange.py`. Run them with `python -m pytest` and
`-o addopts=` in the wheel's environment.

Forward-conversion timing excludes the input copy. Full timing includes the QC
copy, QC → QCO, target compilation, and QCO → QC. QASM loading, target setup,
extra output verification, inspection, and serialization are outside timers. The
native targets use frozen FlexibleBackend connectivity and the common `sx`, `x`,
`rz`, `cz`, measurement, and reset basis without calibration metadata.

## Limits

This is a native API comparison on selected Benchpress inputs, not a run of the
pytest-benchmark suite. Each fresh process performs one timed call, with no
warmup or adaptive calibration and a 600-second whole-process deadline. Three
fixed-seed rounds on a shared desktop do not establish statistical significance
for small differences. Two-qubit depth is not independently measured. The
supported conversion subset is unchanged; this work does not add support for
unstructured control flow.

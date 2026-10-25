# Python binding performance

Status: complete; implemented and validated locally. Hosted CI remains pending.

## Contracts and decisions

The public benchmark, QDMI, and Qiskit signatures and supported inputs are
unchanged. The binding changes preserve lazy MLIR loading, session ownership,
parameter identities, gate semantics, and target aliases.

- `bindings/bench/register_benchmark.hpp` forwards the validated, immutable
  receiver with `nb::find`. The private MLIR bridge uses the existing benchmark
  family registry and typed generators. This removes JSON serialization,
  reparsing, and unused manifest construction. Public stubs remain unchanged.
- `bindings/qdmi/qdmi.cpp` uses nanobind's bounded pool for `Operation`
  wrappers. Destruction releases the session before storage enters the pool. The
  default capacity retains at most 128 empty 56-byte wrappers plus pool
  bookkeeping.
- `bindings/mlir/Target/Qiskit/QiskitAdapter.h` reads primitive parameters from
  `CircuitInstruction.params`, avoiding Gate construction. Import follows packed
  circuit state. Qiskit requires changes to cached Gate objects to be assigned
  back; stale cached parameters are not an input contract.
  [Qiskit's ownership explanation](https://github.com/Qiskit/qiskit/blob/2.5.0/crates/circuit/src/circuit_instruction.rs#L90-L97)
  describes this distinction.
- Symbolic export obtains standard-gate enums from `StandardGate.all_gates()`
  instead of constructing the full gate dictionary. Each writer owns its lazy
  list; renamed target operations retain the actual gate dictionary. This relies
  on the supported Qiskit 2.5/2.6 implementations: `all_gates()` iterates the
  discriminants of the same Rust enum exposed as `QkGate` by the C API.
  [The implementation](https://github.com/Qiskit/qiskit/blob/2.6.0/crates/circuit/src/standard_gate/mod.rs#L1895-L1899)
  and checks of all 52 entries on both adapters establish the indexing contract.

## Measured evidence

Measurements used CPython 3.14.8, Qiskit 2.6.0, nanobind 3.1.0, and Clang 23
Release builds with ThinLTO and mold on an ARM64 DGX Spark. Eight alternating
baseline/candidate pairs ran in fresh processes on one CPU; the shared host was
not reserved. Public entry points produced matching IR or metadata fingerprints.
The following are median process CPU times; wall times closely matched.

Against `eb8f7db150836e3c516aca82d3dce92765a2bb24`, the first three changes
gave:

| Public workload                                   |                 Baseline |                Candidate |
| ------------------------------------------------- | -----------------------: | -----------------------: |
| GHZ(8) / QFT(10) / QAOA(8, depth=2) generation    | 0.898 / 1.079 / 1.154 ms | 0.883 / 1.064 / 1.132 ms |
| QDMI operations, 3 / 54 wrappers                  |            198 / 1535 ns |             165 / 908 ns |
| QDMIBackend construction, SC / DDSIM              |         0.517 / 0.259 ms |         0.519 / 0.258 ms |
| Import 256 symbolic RX / efficient_su2(8, reps=3) |         2.118 / 1.129 ms |         1.794 / 1.030 ms |

Native generation and destruction dominate benchmark generation. The removed
JSON phases take microseconds and about 200 allocations; native profiles show
MLIR storage, allocation, and context destruction. Qiskit target construction
accounts for about 98% of profiled backend construction, explaining why faster
operation queries do not yield a clear backend speedup. Import removes 512 Gate
constructions for symbolic RX and 128 for the ansatz; UUID decoding is
unchanged.

Against `0fb6b738ca` with the first three binding changes above, enum lookup
reduced export of the ansatz from 294 to 243 microseconds and eight symbolic
control-flow blocks from 629 to 255 microseconds. All eight pairs improved. Full
ansatz `compile_program` changed from 8.30 to 8.10 ms, with paired changes
spanning a 1.7% regression to a 7.1% improvement. Numeric export stayed at 213
microseconds. MLIR's timing facility reports about 3.5 ms in QCO passes alone;
export is a small part of full compilation.

Memray traces distinguish allocation traffic from live memory. Symbolic RX
import traffic falls from 2.58 to 1.68 MB while peak live traced memory stays
about 0.7 MB. Eight-block export traffic falls from 1.30 to 0.36 MB; peak live
traced memory falls from 88 to 74 KB. Warm operation queries allocate 4 rather
than 58 times for 54 wrappers. Warm-pool traces exclude storage allocated during
warmup. Process peak RSS varies; no RSS reduction is claimed.

## Limits and validation

No new caches, shared contexts, or input shapes are needed. Site pooling was
excluded after a regression. DD ndarray views show mixed optimized code without
a demonstrated public-call gain. Read-only DD arrays would expand accepted
inputs. QDMI string views would lose ownership of temporary sequence items; UUID
normalization cannot assume subclass properties are valid.

Existing benchmark, QDMI, and Qiskit end-to-end tests cover the retained
contracts. One added symbolic round-trip test compares parameter identities and
bound operators across parameterized standard gates. Swapping RX/RY enum entries
makes it fail at the operator comparison. Compatibility checks cover Python
3.11/Qiskit 2.5.2 as well as Python 3.14/Qiskit 2.6.0.

Local validation after rebasing on `6559f17812` passed:

- 1194 focused tests in `test/python/bench`, `test/python/qdmi/test_qdmi.py`,
  the MLIR Qiskit translation/target/parameter/loop/integer suites, and the
  Qiskit backend suite on Python 3.14/Qiskit 2.6.
- 743 tests in those five MLIR suites on Python 3.11/Qiskit 2.5.
- Optimized rebuild, `uvx nox -s stubs`, and `uvx nox -s lint`, including Ruff
  and ty. All-line clang-tidy checked all four changed C++ files without
  diagnostics.
- Independent enum/ownership scrutiny and a Ponytail review found no
  substantiated defects. No extra implementation or test infrastructure was
  retained.

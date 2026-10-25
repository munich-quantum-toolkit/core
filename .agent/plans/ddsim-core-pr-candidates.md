# Extractable Core changes for DDSIM

Status: implemented and locally validated. These are review boundaries, not
requests to open pull requests.

| Candidate                           | Missing contract and smallest change                                                                                                                                                                              | Existing callers and dependencies                                                                 | Acceptance                                                                                                                                                                    |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Native batched Pauli expectations   | A typed, Qiskit-independent batch of sparse real Pauli sums over a borrowed vector DD. Validate width, finite coefficients and ownership; reuse repeated terms and generic contractions without dense extraction. | Existing matrix-DD expectation API stays unchanged; the QCO wrapper is the new caller.            | Independent small dense oracles, Y phases, asymmetric wire order, scalar/zero/unnormalized states, overflow, root ownership and a compact 64-qubit GHZ state.                 |
| QCO expectation utility and binding | Prepare one eligible uncollapsed state, evaluate the batch and release the root. The utility is new functionality; its Python method is new exposure.                                                             | Reuses existing state preparation and IR verification; depends on the native batch.               | Bound OpenQASM, coherent deferred readout, reset/feedback rejection, program nonmutation, generated stubs and installed-wheel execution.                                      |
| Exact Qiskit DDEstimator            | A V2 adapter to the native batch using Qiskit binding and PUB containers.                                                                                                                                         | Depends on the QCO API; DDSIM delegates to it. Existing sampled QDMI estimator remains unchanged. | Parameter identities/order, PUB broadcasting and shapes, metadata, finite precision, zero sampling uncertainty, one preparation per binding and independent numerical values. |
| QDMI advertised control flow        | Translate advertised `if_else` to Qiskit's control-flow instruction rather than discarding it. This is missing integration exposure, not missing device execution.                                                | Existing QDMI users and DDSIM sampling backend; independent of exact estimation.                  | Real feedback counts and sampled reset/feedback ensemble expectations.                                                                                                        |
| Parameter-bound circuit names       | Preserve the original name when creating a bound circuit for backend execution.                                                                                                                                   | Existing `Result.get_counts(original_circuit)` users; independent backend fix.                    | Original-circuit lookup succeeds and input circuits are unchanged.                                                                                                            |
| Python test-helper discovery        | Declare pytest's `test/python` import path instead of depending on collection order.                                                                                                                              | Existing top-level translation tests and nested Qiskit tests; no runtime change.                  | Nested tests collect and run in isolation and before top-level MLIR tests.                                                                                                    |

## Extraction map

1. Native DD: `include/mqt-core/dd/Pauli.hpp`, `Package.hpp`,
   `src/dd/Package.cpp`, `test/dd/test_pauli_expectations.cpp`.
2. QCO utility and exposure:
   `mlir/include/mqt/Dialect/QCO/Utils/DDFunctionality.h`, its implementation
   and unit tests, `bindings/mlir/register_mlir.cpp`, generated
   `python/mqt/core/mlir.pyi`, `test/python/test_mlir_expectations.py`.
3. Exact primitive: `python/mqt/core/plugins/qiskit/estimator.py`, its export,
   `test/python/plugins/qiskit/test_dd_estimator.py`, and the exact estimation
   section in `docs/qdmi/qdmi_backend.md`.
4. Backend fixes: `python/mqt/core/plugins/qiskit/backend.py`, with the relevant
   additions in `test_backend.py` and `test_estimator.py`. Control-flow target
   support and circuit-name preservation can be split independently.
5. Test setup: pytest's `pythonpath` in `pyproject.toml`; this can accompany the
   primitive tests rather than requiring its own PR.

The first three changes form a dependency chain. The backend fixes can merge
independently. Each candidate needs its focused tests and repository lint;
combined acceptance also needs native tests and installed DDSIM wheels.

## Bounded optimization and exclusions

The Pauli builder can use the existing tensor-product operation because it adds
one uncontrolled factor above all existing wires. Physical indices must not be
shifted. Independent oracles agree, and the compact GHZ estimator probe reduces
tracked allocation volume and peak live allocation. The modest timing result is
workload-specific. This substitution belongs with the native batch.

Keep map ownership while releasing the GIL, Qiskit's binding semantics, distinct
IR validation boundaries and threshold-based collection. No specialized Pauli
interpreter, reusable mutable package pool or exact branch-ensemble engine is
justified by this replacement. The sampled estimator handles supported dynamic
programs. No HSF engine or Base-profile reset work is proposed.

Before a DDSIM release, replace its development Git dependency with the first
published Core version containing the required APIs, then run installed-wheel
checks and release-candidate CI across the declared existing matrix.

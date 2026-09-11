# QC/QCO infrastructure audit

Status: complete.

## Scope and contract

The audit covers QC/QCO dialect operations, modifier matrix queries, and the
QC/QCO conversion boundary. The implementation baseline is
`3be5ee96f3907659bd99fdd6d54cd74c4eea7da9` with LLVM/MLIR 23.1.0.

QCO values carry linear state; QC values identify mutable references. A
conversion that removes a quantum result must prove its correspondence to the
replacement reference. QCO wire permutations remain valid IR, but reference
conversion rejects permutations it cannot represent. QTensor slot updates remain
real stores. Modifier matrices include every declared target in argument order.

## Findings and disposition

| Finding                                                                                                               | Owner and resolution                                                                                                                                                         | Surviving evidence                                                                                                 |
| --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Positional conversion could silently change branch or loop wires, and changing `scf.while` quantum arity could assert | `QCOToQC.cpp` checks region terminators before rewriting and uses the same proven origins for function returns                                                               | Conversion regressions cover each region form, changing quantum arity, reset, classical state, and register stores |
| QC unstructured control flow could assert                                                                             | Already fixed by [#2448](https://github.com/munich-quantum-toolkit/core/pull/2448); preserve the pass's SCF-only contract                                                    | Existing QC-to-QCO rejection tests                                                                                 |
| Barrier input and result counts could differ                                                                          | The owning `BarrierOp` verifier requires one result per input                                                                                                                | `BarrierRejectsMismatchedQubitArity`                                                                               |
| Single-operation modifier matrix shortcuts ignored idle targets and wire order                                        | All modifier queries use `composeBodyMatrix`; full-width operations retain direct matrix support, other supported operations are embedded; reordered yields return no matrix | Exact matrix tests for idle targets, full-width three-qubit operations, reordered yields, and total-width bounds   |
| Allocation mode depended on function rewrite order                                                                    | Collect allocation roots before conversion and store an immutable module mode                                                                                                | `FindsAllocationModeBeforeConvertingCaller` and existing allocation tests                                          |
| QTensor insertion scanned the cache before trying its exact key                                                       | Try the direct lookup first, then retain equivalent-index matching, result-number checks, stores, and cross-region invalidation                                              | Existing store-elision, dynamic slot-swap, and distinct-index-result tests                                         |
| Sink order exposed DenseMap traversal                                                                                 | Track scalar references with MapVector                                                                                                                                       | `EmitsSinksInAllocationOrder`                                                                                      |
| Thirty pattern classes only forwarded to shared helpers                                                               | Register the helpers with MLIR's native function-pattern overload; use short lambdas for optional boolean arguments                                                          | Existing gate canonicalization equivalence tests                                                                   |
| Barrier merging used a dense map for contiguous positions and repeatedly searched inputs                              | Use positional output access and a replacement vector                                                                                                                        | Existing barrier equivalence tests                                                                                 |
| QC/QCO headers retained obsolete reversed-operator warning suppression                                                | Remove the suppression and retain required interface includes                                                                                                                | Normal builds and full-file C++ lint                                                                               |
| Duplicate gate helpers and unreachable branches obscured contracts                                                    | Reuse generic zero-target gate lowering and quantum-state predicates, use a constant-index set, and remove impossible absent-else branches                                   | Existing global-phase, register-alias, and branch-interface tests                                                  |

The effect and modifier-verifier cleanup from
[#2435](https://github.com/munich-quantum-toolkit/core/pull/2435) is already in
the baseline. The conversion retains early delegation to the owning verifier
rather than duplicating its body contract.

## Performance evidence and limits

The isolated exact-key cache change was measured on the original audit baseline
`33dbc843e589d9e9166308084e825c9f8b2ff89d`. Five alternating paired runs of a
32,000-slot extract/gate/insert workload reduced median conversion time from
1,530.682 ms to 225.026 ms. This establishes the benefit on a synthetic wide
register, not a general application speedup or a timing result for the complete
patch. The 148-test conversion suite also passed with that isolated change.

Dense modifier matrices are bounded to ten total qubits, including controls.
Arbitrary-width subset embeddings, permutation transport, mixed allocation
roots, and register-function support remain outside this change. Register-call
contracts are tracked separately in
[#2428](https://github.com/munich-quantum-toolkit/core/issues/2428).

## Validation

See the [implementation record](../plans/qc-qco-infrastructure.md) for the final
checks. Behavioral regressions use the existing GoogleTest suites; no standalone
probe or generated build artifact belongs to the implementation.

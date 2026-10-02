🤖 *AI text below* 🤖 <!-- rumdl-disable-line MD041 -->

# MLIR contract audit — issue #2255

Status: implemented and locally validated. Baseline: main at `7d03fdd68`. Date:
2026-09-10. Scope: QC/QCO modifiers and conversions, optimization and mapping
passes, QTensor cleanup, and QIR metadata.

## Priorities after the upstream refresh

Main's merged performance work changes QIR preparation, QCO builder tracking,
QTensor canonicalization, mapping, and parameter validation. It resolves none of
the seven findings from the review of PR #2505 at `7f5542862`. The changes below
retain those upstream implementations and the finite-parameter precondition; no
exhaustive expression validation is restored.

1. **Correct packed static QIR capacity.** `emitQISCall` marks qubit stores in
   control arrays and argument tuples. Metadata follows the stored pointers,
   keeping qubit and result roles separate without aggregate alias analysis.
   `QIRCountsPackedStaticQubits` checks three-control X/RX through the
   compiler's base and adaptive profiles; the builder regression checks a sparse
   tuple target. This closes a preexisting gap.
2. **Constrain Hadamard lifting to its supported shape.** Require at least one
   control and exactly one target. The sole-X matcher otherwise rewrites the
   wrong wire when the first target is unused, or asserts on zero controls.
   `LeavesUnsupportedControlShapesUnchanged` covers both valid inputs. This
   closes a preexisting gap.
3. **Terminate modifier verification.** Each wire producer must precede the
   previous producer in the same block. Operation verification runs before SSA
   dominance checking; `RejectsCyclicWireProducers` prevents the new verifier
   from hanging on malformed cyclic SSA.
4. **Retain the supported CNOT alias.** Include `__quantum__qis__cnot__body` in
   scalar operand roles. `MetadataCountsCnotAliasQubits` covers the regression
   where static IDs 7 and 0 produced capacity 0 instead of 8.
5. **Remove identity remapping in QCO-to-jeff.** The preflight's full-width,
   argument-order contract permits a view of the existing target list and
   eliminates the nested target-copy loop. Existing normalized modifier round
   trips and rejection tests retain the supported boundary.
6. **Skip modifier interiors in QCO-to-QC origin collection.** Verified
   modifiers tie external outputs directly to inputs. Internal entries have no
   consumer. A staged walk preserves postorder handling of control flow and
   function-return correspondence.
7. **Finish register-shrinking cleanup.** Remove three impossible operand
   identity checks and exit before sorting/remapping unchanged registers. Keep
   constant-index and bounds checks, including dynamically typed tensors.

## Retained audit changes

- **F1/F6:** mapping diagnoses opaque classical effects and tensor control flow
  before mutation. Existing CBit scheduling and flat tensor chains remain.
- **F2/F10:** QC/QCO modifier signatures and positional yields have one owning
  verifier. Real SWAP gates and reordered gate operands remain valid. Remove 13
  repeated canonicalizer guards and redundant conversion checks; replace invalid
  permutation-only tests with verifier coverage.
- **F3/F4/F5/F9:** retain the XX±YY large-angle correction, nullable producer
  matching, yield-only loop-unroll fix, and index-switch attribute round trip.
- **F7:** preserve sparse-ID, overflow, result-role, and dynamic-pointer tests
  alongside the packed-pointer and CNOT corrections above.
- **C2/C3/C4:** delete the unused QTensor helper header, duplicate ancestor
  checks, and redundant register-shrinking bookkeeping. Keep slot identity,
  reset, and index-boundary regressions.

## Ownership and limits

C1 duplicates finding 8 in
[PR #2502](https://github.com/munich-quantum-toolkit/core/pull/2502), rechecked
at `39b5ecd71`; its QC-to-QCO validation cleanup remains there. Main already
owns F8's loop classification from #2495. This PR retains only the
measurement-latch regression for that finding.

QIR resource inference covers supported scalar calls and compiler-emitted
aggregate arguments. It does not infer arbitrary external QIS layouts or pointer
aliases. QC-to-QIR entry arguments, measurement-bearing helpers, and permissive
imports of unregistered operations remain separate support questions. Passes
rely on valid IR; no blanket pre-pass verification is added.

The
[original report and evidence](https://github.com/munich-quantum-toolkit/core/tree/b31e80e66b01bea9e052962de6d65c951220d3b5/.agent/audits/issue-2255-evidence)
remain in Git history. Normal subsystem tests now own the regressions.

## Validation

All 3,009 tests across 17 affected binaries pass with assertions enabled on
LLVM/MLIR 23.1.0. The suites cover QC/QCO/QTensor/QIR IR, modifier consumers,
optimizations, mapping, decomposition, target synthesis, phase normalization,
QC↔QCO, jeff round trips, both QC-to-QIR profiles, and compiler pipelines.

`uvx nox -s lint` and `uvx nox -s cpp-lint -- 7d03fdd68` pass. C++ lint checked
every line of 34 changed C++ files with zero findings. No full-repository test
matrix, measured speedup, sanitizer run, or hardware execution is claimed.

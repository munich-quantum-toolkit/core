# Audit resolution: MLIR tests, diagnostics, and debugging

Status: applied and locally validated. Date: 2026-09-10. [PR #2502][pr]
addresses [issue #2254][issue]. This audit compares PR head
`39b5ecd71ff7010a829dcb58878a7ae1261ac9c2` with upstream
`ba3aea8b1b618d8b000629c3995224ef4f73fa75`, including #2505. The branch rebased
without conflicts. The [original audit][historical] retains the original crash
and false-equivalence probes; permanent regressions now own that coverage.

## Reprioritized findings

1. **P1: preserve ownership and reject consumed Python inputs — applied.**
   `Program::operator=` destroys the old module before releasing its context.
   Binding adapters validate direct copies, typed compiler inputs, and methods
   before entering C++. `MoveAssignmentKeepsModuleContextAlive` in
   `mlir/unittests/Compiler/test_compiler_pipeline.cpp` checks independent
   contexts and self-assignment. `test_consumed_program_operations_raise` and
   `test_consumed_jeff_write_raises` in `test/python/test_mlir.py` replace the
   original SIGSEGV/abort probes. These fixes remain necessary on current main.
2. **P1: prevent false tensor equivalence and comparator aborts — applied.** The
   rebased permutation helper accepted reversed writes to one tensor slot and
   accessed an unmapped allocation when its dynamic size was not ready. The new
   `TensorWritesToTheSameSlotKeepTheirOrder` regression failed in both
   comparison directions; `TensorComparisonWaitsForItsAllocation` aborted in
   `DenseMap::at` with exit 134. Both inputs pass MLIR and QCO-linearity
   verification. The shared access walk now waits for the input tensor and
   permits reordering only at distinct constant indices in the same block.
   Dynamic and repeated slots retain their order. Insert/extract handling shares
   one walk without temporary operation vectors. A positive distinct-slot test
   preserves the supported permutation. All comparator regressions live in
   `mlir/unittests/Support/test_ir_verification.cpp`.
3. **P2: preserve diagnostic evidence and replay — applied.** One
   `SourceMgrDiagnosticHandler` and its source buffers survive loading, passes,
   and output, including stdin. Shared registration covers upstream transforms,
   shrink passes, and QIR cleanup/metadata; the driver registers its conversion
   passes. Initial jeff conversion applies pass-manager options.
   `mlir/unittests/Compiler/mqt-cc/verify_debugging.cmake` checks excerpts,
   notes, stack-trace notes, pass selection, threading, and early/late QIR
   replay. Python's handler-restoration test observes a subsequent failing
   diagnostic.
4. **P2: remove repeated custom-pipeline execution setup — applied.** Normal
   custom compilation now appends the supplied passes between preparation and
   cleanup in one pass manager. The driver requires the explicit
   `builtin.module(...)` wrapper and delegates its contents to MLIR's parser.
   This removes two extra pass-manager executions and keeps all three stages in
   the same reproducer. Tests cover empty and nested pipelines, malformed
   wrappers, required inlining, and replay. `--run-pipeline` still runs exactly
   its selected pipeline; `--run-reproducer` honors recorded verification. No
   measured compiler speedup is claimed.
5. **P2: retain deterministic output without copying register contents —
   applied.** QIR builder registers have consecutive `cN` labels and record in
   allocation order. Finalization now moves their descriptors, including Base
   Profile result vectors, into the existing recording helper. No later builder
   stage uses those descriptors. This follows the consuming transfer already
   used by `qc::addOutputRecording`; public register handles remain independent
   copies. `BuilderRecordsRegistersInAllocationOrder` and QIR builder/conversion
   tests retain ordering and output coverage. No runtime timing claim is made.
6. **P2: remove obsolete QIR release-name exceptions — applied.** With disposal
   commutation temporarily disabled, all 429 QIR builder/Base/ Adaptive
   conversion tests passed. Across five rebuilt consumer binaries, 744 of 745
   tests passed; only QCO-to-QC's `RepeatedControlledX` reference still required
   QC deallocation permutations. The final helper retains that QC rule and
   removes four runtime-name exceptions, its LLVM dialect include, and the
   shared LLVM dialect link dependency. The regression binary links the dialect
   directly and now requires QIR release-call order. A failed structural match
   remains a conservative result, not proof of semantic inequivalence.
7. **P2: give modifier validation one owner — retained.** QC verification owns
   malformed modifier bodies and captured qubits. QC-to-QCO no longer repeats
   the recursive verifier. The surviving verifier matrix covers all 16 forbidden
   operations, three modifiers, and direct/nested bodies; separate
   scalar/register-backed capture tests remain. The removed allocation-origin
   axis did not change the body under test. Conversion tests retain supported
   captures and valid-but-unsupported boundaries.

## Upstream interaction and retained limits

- #2516 makes finite gate parameters a valid-IR precondition, retains cheap
  direct-constant checks, and batches only proven commuting QTensor accesses.
  This PR preserves that policy; the comparator must not infer disjoint tensor
  slots from SSA linearity alone.
- #2514 fixes QCO builder tracking and reset scaling. Its ownership and
  deterministic-disposal changes remain intact after the rebase.
- #2513 improves QIR output preparation and changes resource release storage.
  The PR still needs ordered register recording; finalization can now use the
  same consuming transfer as conversion output preparation.
- #2505 gives QCO modifier verifiers positional-yield ownership and removes
  modifier-origin traversal from QCO-to-QC. The comparator retains valid
  control-flow yield permutations; its three now-invalid modifier fixtures are
  removed. Upstream's `RequiresUniqueInputsAndPositionalYields` regression owns
  rejection for all three modifiers. The QC-to-QCO verifier cleanup remains
  necessary and separate from these QCO contracts.

`areModulesStructurallyEquivalent` uses upstream `OperationEquivalence` with
consistent SSA mappings, including forward definitions across blocks. Parser
round trips use it directly. Exact checking preserves constants, types,
attributes, predicates, branch destinations, captures, effects, and yielded wire
positions. The regression matrix covers QCO control-flow yield boundaries and
consistent parent-result permutations; modifier verifiers require positional
yields. The former numerical tolerance and attribute whitelist remain deleted.

The permutation helper tries structural comparison first. Its fallback is
intentionally greedy and can require quadratic work; failure does not prove
semantic inequivalence. Blocks correspond in region order, and fallback requires
cross-block definitions before uses. Module symbols, independent SSA operations,
fresh allocations, and the documented resource disposals can reorder. It does
not perform general alias analysis. A broader semantic-oracle migration remains
separate work, with explicit resource, phase, and numerical limits.

Fixture corrections remain necessary: parser defaults, source-label exclusions,
reference load/reset order, and the QCO index-switch initialization must express
the intended program. `RejectsMissingPositionalQubitResults` verifies its linear
input and asserts the intended unsupported-shape diagnostic. These assertions
must not be weakened to make the comparator accept changed programs.

## Validation

Current native validation uses Clang 23, LLVM/MLIR 23.1.0, ThinLTO, and mold:

- `cmake --preset release-clang-ipo`: passed.
- `cmake --build --preset release-clang-ipo -j8`: passed after the final edits.
- `ctest --preset release-clang-ipo --output-on-failure -j8`: 3,506 entries,
  zero failures, one intentional `ScQDMIJobSpecificationTest.QueryJobId` skip.
  This includes 23 comparator regressions and all three driver suites.
- `uvx nox -s lint`: passed after formatting and staging the changes.
- `uvx nox -s cpp-lint -- ba3aea8b1b618d8b000629c3995224ef4f73fa75`: passed;
  clang-tidy 23 checked the full changed C++ files with zero findings.
- The temporary disposal experiment was restored before the final cleanup and
  complete validation. No experimental source or generated build files remain in
  the diff.

Python tests and stub generation were not repeated: this follow-up changes no
binding implementation. Current hosted CI has not validated the local branch.

Historical results are tied to their measured revisions:

- At `39b5ecd71`: all 1,575 affected CTest entries, repository lint, and
  full-file C++ lint passed. The [comparator benchmark][benchmark] measured a
  4,000-gate comparison at median 538.385 ms before and 0.141313 ms with strict
  comparison. This is a comparator microbenchmark, not a compiler or whole-suite
  speedup.
- At `a1cff7ca2`: 888 Python MLIR/QDMI tests passed against rebuilt
  shared-library bindings, and stub generation passed. This refresh changes no
  binding API.
- Hosted checks observed for `39b5ecd71` passed; they do not validate the
  rebased branch or this follow-up. Other platforms are not locally validated.

[pr]: https://github.com/munich-quantum-toolkit/core/pull/2502
[issue]: https://github.com/munich-quantum-toolkit/core/issues/2254
[historical]: https://github.com/munich-quantum-toolkit/core/tree/a72eaa1198853a005ee16270dfbe331e78feeff5/.agent/audits
[benchmark]: ../benchmarks/pr2502-comparator/README.md

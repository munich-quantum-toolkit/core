# Audit resolution: MLIR tests, diagnostics, and debugging

Status: applied; follow-up locally validated. Baseline:
`d994fe6833b6b7a9b1bccdeccc64e31c5c09ffd1`. Date: 2026-09-10. Implementation:
[PR #2502][pr], addressing [issue #2254][issue].

## Result

The ten accepted findings have implementation changes and permanent regression
coverage. The original [audit and executable probes][historical] preserve the
baseline crash and false-equivalence evidence. Those experiments are superseded
by the owning tests below, so their copied patches and logs have been removed.
The follow-up also closes the yielded-wire comparison and late QIR replay gaps,
adds native structural comparison, and removes redundant verifier cases.

## Findings and disposition

1. **Program move assignment — applied.** `Program::operator=` now destroys the
   destination module before releasing its context. Declaration order continues
   to preserve normal destruction. The independent-context and self-assignment
   regression is `MoveAssignmentKeepsModuleContextAlive` in
   `mlir/unittests/Compiler/test_compiler_pipeline.cpp`. The baseline probe
   crashed with SIGSEGV.
2. **Consumed Python programs — applied.** Existing validity checks now cover
   direct copies, program-backed method adapters, and high-level typed inputs.
   `test_consumed_program_operations_raise` and
   `test_consumed_jeff_write_raises` in `test/python/test_mlir.py` cover
   controlled exceptions after consumption. The baseline copy, cleanup, and
   compile probes each aborted the interpreter.
3. **False structural equivalences — applied.**
   `mlir/unittests/Support/IRVerification.cpp` compares exact attributes, types,
   QCO predicates, and mapped CFG destinations. Readiness respects captured
   values and effect ordering, and requires every operation to be compared.
   `mlir/unittests/Support/test_ir_verification.cpp` retains all seven original
   negative pairs, the classical `0.0`/`1e-16` counterexample, and positive and
   negative checks of the remaining permutation rules. Yielded values now follow
   mapped parent results at every QCO region boundary. The old attribute
   whitelist and blanket numerical tolerance are gone.
4. **Diagnostic notes and stack traces — applied.** The driver retains one
   `SourceMgrDiagnosticHandler` and its source buffers through loading, passes,
   and output. `mlir/unittests/Compiler/mqt-cc/verify_debugging.cmake` checks
   file and stdin excerpts, operation notes, and the trace note without
   depending on stack addresses or frame counts. The baseline dropped attached
   notes.
5. **Pass registration and selection — applied.** Shared registration includes
   upstream transforms, both shrink passes, and QIR cleanup/metadata passes. The
   driver registers its conversion stages and accepts
   `--pass-pipeline='builtin.module(...)'`; `--passes` is a textual alias.
   Misleading individual-pass options are removed. The CLI regression checks
   canonicalize/CSE, aliases, conversion selection, all four previously missing
   pass names, and invalid individual flags. Base and adaptive QIR failures
   replay the original diagnostic even when cleanup follows the failing pass.
6. **Isolated pipelines and replay — applied.** `--run-pipeline` executes
   exactly the selected MLIR module pipeline. `--run-reproducer` uses upstream
   parsing and applies recorded threading and verification settings. Normal
   compilation retains required preparation, inlining, and cleanup around custom
   passes. CLI tests check each route, invalid combinations, and failure replay.
   [Development documentation][debugging] gives complete runnable examples.
7. **Initial jeff instrumentation — applied.** The initial jeff-to-QCO pass
   manager now applies CLI options. The CLI regression imports emitted jeff and
   observes the initial conversion's requested IR dump.
8. **Duplicate modifier verification — applied.** QC-to-QCO no longer repeats
   the recursive modifier verifier. QC verifier tests cover all 16 forbidden
   operations across three modifiers and direct/nested bodies (96 cases). The
   removed allocation-origin axis never changed the body being verified.
   Separate scalar/register-backed capture tests retain all four SCF shapes.
   Converter tests retain valid classical captures and unsupported valid-input
   boundaries. This removes the extra whole-IR walk and the misplaced negative
   conversion fixtures; no runtime speedup is claimed.
9. **Invalid rejection fixture — applied.**
   `RejectsMissingPositionalQubitResults` in
   `mlir/unittests/Conversion/QCOToQC/test_qco_to_qc.cpp` now consumes its
   qubit, verifies both MLIR and QCO linearity, and asserts the intended
   positional-result diagnostic before expecting conversion failure.
10. **Handler-restoration assertion — applied.** After target compilation fails,
    `test_target_compilation_preserves_diagnostics` triggers another diagnostic
    on the shared context and checks stderr. A subsequent successful action
    alone could not establish that the original handler was restored.

## Comparator boundaries and fixture corrections

The helper still uses greedy matching and the existing QCO/qtensor wire
permutations. A failed comparison does not prove semantic inequivalence. Module
symbols can reorder. Fresh allocations can move, independently owned linear
values can be disposed of in any order, and consecutive QC deallocations or QIR
runtime releases of the same kind can commute. Other effect ordering remains
significant; the helper does not implement general alias analysis.

Exact checking exposed fixture contracts previously hidden by the helper:

- Align the QCO index-switch parser fixture's initialization with its builder.
- Exclude source register names explicitly in shared frontend gate fixtures;
  dedicated register-name tests continue to check the metadata.
- Parse both compiler-stage reference representations so LLVM's optional default
  properties are materialized consistently.
- Place reference loads and the reuse fixture's reset at their expected points
  of execution instead of assuming arbitrary memory/call reordering.
- Emit QIR builder register outputs in allocation order instead of hash-table
  order. `BuilderRecordsRegistersInAllocationOrder` checks twelve registers,
  including labels beyond `c9`.

`areModulesStructurallyEquivalent` uses upstream `OperationEquivalence`,
ignoring locations while preserving operation, block, operand, and result order.
Its SSA callbacks support forward references across blocks and reject
conflicting later definitions. Parser/print round trips now use it directly; it
also provides the permutation helper's fast path. Transformation families retain
permutation matching where independent constants, tensor operations, symbols, or
releases can legitimately reorder. Permutation fallback still requires
cross-block definitions before uses in region order; the structural path has no
such limit.

The [matched comparator benchmark][benchmark] measures 4,000 verified QCO gates
at a median 538.385 ms before, 0.141313 ms for strict comparison, and 0.122945
ms through the fast path. This is a comparator microbenchmark, not a compiler or
whole-suite speedup. Greedy permutation fallback can still take quadratic work.

Ponytail Review removed redundant block-argument mapping and replaced identity
result-permutation construction with `IRMapping::map` (seven lines removed). The
redundant switch-case comparison and unused `MLIRAnalysis` dependency are also
gone. Broader semantic-oracle migration remains possible, with explicit
resource, phase, and numerical limits; it is not required for these fixes.

## Validation

Follow-up on native ARM64 with Clang 23 and LLVM/MLIR 23.1.0:

- Release build of all 13 comparator consumers and `mqt-cc`: successful.
- All 1,575 affected CTest entries pass, including the complete QC/QCO IR,
  optimization, compiler, translation, and conversion suites plus three driver
  suites. The focused regressions can be rerun with
  `ctest --test-dir build/release-clang-ipo --output-on-failure -R 'IRVerificationTest|mqt-core-mqt-cc-debugging-test'`.
- `uvx nox -s lint`: successful.
- `uvx nox -s cpp-lint -- 7d9061796107f4f3d95cc53ef84ccaa9c7fd6c7f`: successful
  with clang-tidy 23.1.1 and zero findings across the full changed C++ files
  against the fixed PR base.
- The matched benchmark verifies both inputs and checks every timed result; its
  harness exits 0.

Python binding tests were not repeated: no binding implementation changed in
this follow-up. Hosted checks for the new head and other platforms are not
claimed as validated.

Historical validation of the original implementation:

- Release build with AppleClang 21 and LLVM/MLIR 23.1.0: successful, with no
  compiler warnings in the build output.
- `ctest --preset release --output-on-failure -j8`: 3,459 entries, zero
  failures, one intentional `ScQDMIJobSpecificationTest.QueryJobId` skip. This
  includes the 17 comparator regressions and the driver debugging/replay
  regression.
- `python -m pytest test/python/test_mlir*.py -q`: all 619 cases pass against
  rebuilt bindings, using Python 3.13.7 and Qiskit 2.5.2.
- `uvx nox -s stubs`: successful; regenerated stubs have no API diff.
- `uvx nox -s lint`: successful.
- `uvx nox -s cpp-lint`: unavailable locally; the installed clang-tidy is 21,
  while the repository requires 23. Hosted CI and other platforms are not
  claimed as validated.

[pr]: https://github.com/munich-quantum-toolkit/core/pull/2502
[issue]: https://github.com/munich-quantum-toolkit/core/issues/2254
[historical]: https://github.com/munich-quantum-toolkit/core/tree/a72eaa1198853a005ee16270dfbe331e78feeff5/.agent/audits
[debugging]: ../../docs/mlir/development.md#debugging
[benchmark]: ../benchmarks/pr2502-comparator/README.md

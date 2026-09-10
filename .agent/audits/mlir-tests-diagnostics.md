# Audit resolution: MLIR tests, diagnostics, and debugging

Status: applied; local C++ lint remains unverified. Baseline:
`d994fe6833b6b7a9b1bccdeccc64e31c5c09ffd1`. Date: 2026-09-10. Implementation:
[PR #2502][pr], addressing [issue #2254][issue].

## Result

The ten accepted findings have implementation changes and permanent regression
coverage. The original [audit and executable probes][historical] preserve the
baseline crash and false-equivalence evidence. Those experiments are superseded
by the owning tests below, so their copied patches and logs have been removed.
The broader comparator replacement remains deferred.

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
3. **False structural equivalences — applied, wider migration deferred.**
   `mlir/unittests/Support/IRVerification.cpp` compares exact attributes, types,
   QCO predicates, and mapped CFG destinations. Readiness respects captured
   values and effect ordering, and requires every operation to be compared.
   `mlir/unittests/Support/test_ir_verification.cpp` retains all seven original
   negative pairs, the classical `0.0`/`1e-16` counterexample, and positive and
   negative checks of the remaining permutation rules. The old attribute
   whitelist and blanket numerical tolerance are gone.
4. **Diagnostic notes and stack traces — applied.** The driver retains one
   `SourceMgrDiagnosticHandler` and its source buffers through loading, passes,
   and output. `mlir/unittests/Compiler/mqt-cc/verify_debugging.cmake` checks
   file and stdin excerpts, operation notes, and the trace note without
   depending on stack addresses or frame counts. The baseline dropped attached
   notes.
5. **Pass registration and selection — applied.** Shared registration includes
   upstream transforms. The driver registers its conversion stages and accepts
   `--pass-pipeline='builtin.module(...)'`; `--passes` is a textual alias.
   Misleading individual-pass options are removed. The CLI regression checks
   canonicalize/CSE, aliases, conversion selection, and invalid individual
   flags.
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
   the recursive modifier verifier. QC verifier tests cover direct/nested,
   scalar/register-backed invalid bodies and captures, including all four SCF
   shapes. Converter tests retain valid classical captures and unsupported
   valid-input boundaries. This removes the extra whole-IR walk and the
   misplaced negative conversion fixtures; no runtime speedup is claimed.
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

The later PR should migrate structural round trips to upstream
`OperationEquivalence` and transformation tests to existing semantic oracles
with explicit resource, phase, and numerical limits. It should also retire the
current helper's repeated readiness scans where possible. That larger migration
is not part of this implementation.

## Validation

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

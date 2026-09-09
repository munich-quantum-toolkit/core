# QIR output and resource contracts

Status: complete.

## Goal and scope

Close the standalone allocation-verifier gap and repair Adaptive release and
result ownership and shared measurement/store fusion in PR #2446. Keep QC/QCO
allocation ownership at the MQT program boundary. QIR-specific restrictions
belong to conversion, with native diagnostic and semantic regression coverage.

## Decisions

- Standalone QIR and mapping passes invoke the shared MQT allocation check; pass
  dependency loading alone does not verify input before execution.
- QIR output preparation supports a single entry-function return. Reject
  multiple exits before mutation; use the actual return block rather than
  block-list order.
- Preserve quantum releases at their source control-flow positions.
- Adaptive scalar results use dynamic allocation, matching returned result
  arrays. Base scalar results retain static IDs. The public QIR builder must
  enforce consistent result ownership independently of qubit ownership.
- Fuse same-block measurement/store pairs only with an available index and no
  intervening classical interference. Known quantum effects and stores to
  distinct constant indices are safe to cross. Reject uncertain cases before
  mutation.
- QIR builder finalization releases owned qubits at its current insertion point;
  output recording and result releases remain in the output epilogue.

## Validation

Native QC IR, QC-to-QCO, mapping, Base and Adaptive QIR conversion, QIR
IR/builder, compiler, and JIT suites pass: 1,253 tests. Runtime probes also
execute mixed scalar/register results and both conditional-release paths without
ownership errors. Negative tests preserve valid source IR when rejecting
multiple exits or unsafe output stores. Existing target compilation covers
stores to distinct bits across quantum modifiers; `qc.yield` supplies the
effect-free terminator contract needed by recursive effect analysis.

`uvx nox -s lint` and `uvx nox -s cpp-lint` pass. Hosted CI is separate; full
Python and documentation suites were not run locally.

## Outcome

Allocation checks have one implementation, including standalone pass boundaries.
Quantum release control flow and result ownership are explicit. Output
preparation checks its supported subset before mutating returns or stores.
General multiple return normalization and arbitrary classical output-store
lowering remain outside this subset; callers receive diagnostics rather than
incorrect QIR.

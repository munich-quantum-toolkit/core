# Loop translation contracts

Status: applied. Baseline: `3d59d0488`. Scope: loop translation, its Python
round trips, and QCO-to-QC index matching.

## Findings

- Reuse `QCProgramBuilder` for temporary loop CFGs. Break and continue must
  carry complete state through one decision; separate destinations introduced
  poison in the nested continue/switch regression.
- Preserve result identity when comparing pure index-producing operations.
  Operation equivalence alone treated different results of one `scf.if` as equal
  and deleted stores implementing a qubit permutation. The direct QCO-to-QC
  regression fails without the result-number check.
- Remove duplicate external-input rejection from the loop tests. The existing
  Qiskit rejection test covers that boundary and verifies source preservation.
- Keep the scalar test's qubit allocation: `QCProgram.from_mlir_str` requires QC
  IR, so deleting every QC operation makes the fixture invalid.
- Keep jeff-to-OpenQASM/Qiskit checks. They exposed redundant qubit stores and
  snapshot materialization failures after normalization; direct first-hop round
  trips did not cover those cases.
- Use odd and even iteration counts for continue. An even count alone cannot
  distinguish the expected result from accidentally executing two X gates per
  iteration; an odd count alone cannot distinguish continue from early break.

## Retained boundaries

Canonicalization, CSE, and SCCP did not remove the poison backedge introduced by
RemoveDeadValues in a valid first-exit loop. Source export therefore retains
defined edge values. Frontend declaration placeholders can be defined because
definite-initialization analysis prevents observing them; this does not justify
rewriting arbitrary poison.

`setHasBoundedRewriteRecursion` permits recursive pattern application; it does
not prove quantum correctness. No additional use was demonstrated. Controlled
decomposition needs a decreasing measure across gate families, and nested powers
still require their branch-cut restrictions. No flag-only follow-up PR is
warranted.

## Validation

The updated loop suite passes all 31 cases. All 501 MLIR Python tests and 3,117
MQT MLIR C++ tests pass. The loop module also skips under Qiskit 1.1.0.

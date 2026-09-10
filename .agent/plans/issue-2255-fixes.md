# Implement the MLIR contract audit

Status: complete; affected tests and required lint pass.

## Scope and decisions

Implement issue #2255 in the owning verifiers, conversions, and passes. The
branch is based on main at `7d03fdd68`. Preserve its parameter precondition,
QTensor batching, QIR output preparation, and layout improvements.

- Modifiers preserve positional wire correspondence. Verify this with a
  terminating local walk; valid-IR consumers reuse the guarantee.
- QCO-to-jeff supports a single unitary using all target arguments in order.
  This permits direct target views and removes nested identity remapping.
- Mapping supports its CBit effect model and flat tensor access chains.
- Hadamard lifting handles controlled X with at least one control and one
  target; other valid shapes remain unchanged.
- QIR emission marks qubit stores before pointer packing loses their roles.
  Metadata counts these stores and supported scalar QIS operands, including the
  runtime's CNOT alias. Arbitrary external aggregate layouts need separate
  analysis and remain outside this change.
- Keep C1's QC-to-QCO validation cleanup with PR #2502. Main owns F8's loop
  classification; retain only the distinct measurement-latch regression here.

See the [resolution ledger](../audits/issue-2255-contracts-2026-09-10.md) for
regressions, current validation, and remaining support limits.

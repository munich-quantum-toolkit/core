# Implement the MLIR contract audit

Status: complete locally on the existing audit branch for PR #2505.

## Scope and decisions

Implement the accepted findings in the owning layers, with direct regression
tests. Passes rely on valid IR; unsupported conversion or mapping subsets need
diagnostics before mutation. Do not add blanket verification or change unrelated
interfaces. Use the repository MLIR and audit policies.

PR #2502 was rechecked at `da67d7fc1b3e186eedcf6425a6160b64eb9f6328`. Its
finding 8 is the same as this audit's C1 (redundant QC modifier verification in
conversion). Leave that implementation and its test relocation with #2502;
retain a cross-reference here rather than duplicate the patch. The other audit
now implements it. Its other findings and production changes are separate.

The user chose positional correspondence between modifier arguments and yielded
qubits. Enforce it in the owning verifiers and make conversion rely on it. No
new performance claim is planned.

## Completed work

- [x] F1/F6: diagnose unsupported mapping effects and tensor chains.
- [x] F3/F4/F9: stabilize XX±YY matrices, match nullable producers correctly,
  and round-trip index-switch attributes.
- [x] F5: make full quantum unrolling progress on terminator-only permutations.
- [x] F7/F8: correct QIR resource and loop metadata.
- [x] F2/F10: fix modifier conversion/signatures with the chosen wire contract.
- [x] C2/C3/C4: remove stale QTensor helpers and redundant verifier/planning
      work.
- [x] Run focused and affected suites, full repository lint, and C++ lint.
- [x] Reduce the audit to current outcomes and remaining decisions; remove
  obsolete diagnostic scaffolding after durable regressions replace it.

## Validation

Implementation: `1cdb98383`; lint cleanup: `b2b7af3fd`. All 2,680 tests in 15
affected assertion-enabled GoogleTest binaries pass against LLVM/MLIR 23.1.0.
`uvx nox -s lint` and `uvx nox -s cpp-lint` pass; the latter checks all 32
changed C++ files, not only changed lines. No measured speedup is claimed.

See the [resolution ledger](../audits/issue-2255-contracts-2026-09-10.md) for
regression evidence, supported-subset decisions, duplicate ownership, and the
historical evidence link. The temporary tracked probe harness is removed;
unrelated untracked probe directories are preserved. No remote changes were
made.

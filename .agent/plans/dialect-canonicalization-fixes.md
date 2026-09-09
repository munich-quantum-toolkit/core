# Stabilize matrices and simplify local canonicalization

Status: complete; remaining fixes implemented and validated locally. Base:
`7e2a2679f`, including PR #2464.

## Goal and scope

Complete the remaining findings in
[the canonicalization audit](../audits/dialect-canonicalization.md): stable
R/U/U2 matrices, U2 inversion, bounded U-power reconstruction, QCO Id/Unitary
folds, immediate pair-cancellation linearity tests, and CBit/QCO If/QTensor
local simplifications. The dependency, wire-mapping, and bounded
angle-arithmetic changes were extracted separately. The comprehensive
pre-extraction record is retained at `79e9c347b`.

Transformation and conversion patterns remain outside the change scope. Their
tests validate only relevant consumers. Preserve full unitaries, global phase,
wire identity, and QCO linearity after each rewrite. Failed matches leave IR
unchanged.

## Decisions

- Compose R/U/U2 phase factors without adding fixed offsets or unbounded angles.
  Invert U2 with sign-only parameters of U. Test full matrices against
  independent axis and Euler-product oracles, including under control.
- Extract U-power parameters from the bounded matrix power already required by
  validation. Retain finite-input checks, the positive integral exponent limit
  of 1024, and the `5e-13` reconstruction bound. Use sequential products as the
  independent test oracle.
- Fold QCO Id and exact Unitary identity at their roots. Preserve operand order
  and every nonzero phase. Keep two-operation H/X/Y/Z and QTensor pair rewrites;
  a fold that needs later DCE to restore linearity is unsafe.
- Forward CBit loads across read-only snapshots while retaining write and alias
  barriers. Decode QTensor provenance constants once. Use ordered yield-pair
  lookup for QCO If results and reuse the unused-result mask.
- Defer tensor-chain normalization and provenance redesign. Mapping needs the
  existing normal form, and a shared provenance cache needs an invalidation
  owner. Retain local producer and dynamic-index barriers.
- Keep the identity-folder consumer check: an Id prefix folds away and leaves
  one RX; nontrivial fixed-gate prefixes still require fusion. Full-matrix
  checks protect both cases.

## Validation

The narrowed diff passed 1,150 tests in the six focused release binaries, the
complete release and MLIR documentation build, and 3,224 configured C++ tests.
One optional QDMI job-ID test was skipped; none failed. LLVM 23 C++ lint passed
against `7e2a2679f`. Full formatting and metadata lint passed.

The [audit](../audits/dialect-canonicalization.md#validation) records the checks
and limits. The
[evidence appendix](../audits/dialect-canonicalization-probes.md) names the
retained regressions and historical scaling measurements. Hosted CI has not yet
run on this narrowed diff.

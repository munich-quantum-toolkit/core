# Stabilize matrices and simplify local canonicalization

Status: complete; implemented and validated locally.

The [canonicalization audit](../audits/dialect-canonicalization.md) is the
current decision and validation record. It covers stable R/U/U2 matrices,
sign-only U2 inversion, bounded U-power reconstruction and rejection coverage,
exact identity folds, immediate pair-cancellation linearity, and local CBit/QCO
If/QTensor simplifications. The
[evidence appendix](../audits/dialect-canonicalization-probes.md) retains the
regressions, numerical mutation, local microbenchmark, and historical scaling
measurements. The pre-extraction record remains at `79e9c347b`.

Transformation and conversion patterns remain outside this change. Preserve full
unitaries, global phase, wire identity, and exactly-one-use quantum values.
Failed matches must leave IR unchanged. Main already owns dialect dependencies,
modifier wire mappings, bounded angle arithmetic, and batched fresh-slot resets.
General tensor-chain normalization and mutable provenance caches remain outside
scope.

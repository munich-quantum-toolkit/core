# MLIR audit resolution

Status: complete. The ten accepted test, diagnostic, ownership, and debugging
findings are implemented in PR #2502. The [audit resolution record][audit]
contains the evidence, permanent regression locations, validation results, and
remaining comparator limitations.

The existing comparator has targeted corrections and explicit equivalence rules.
Its replacement by upstream structural comparison and semantic oracles is
deferred to a separate PR. Verified-input ownership remains with the dialect
verifiers; isolated driver execution and reproducer replay have explicit,
different verification policies documented in `docs/mlir/development.md`.

Local CTest and Python suites pass. C++ lint still requires clang-tidy 23; the
locally installed version cannot perform that check.

[audit]: ../audits/mlir-tests-diagnostics.md

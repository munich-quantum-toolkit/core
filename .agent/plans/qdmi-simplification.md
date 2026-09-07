# QDMI implementation simplification

Status: complete.

QDMI shares platform lookup, standard property decoding, sparse-result decoding,
and session override handling. The test suite reuses one environment guard and
constructs mock devices directly. The unused internal session constructor is
removed. Each change has a separate commit.

Module lookup retains caller-supplied anchors so providers and the driver locate
their own libraries. Shared decoders retain optional support, buffer checks,
query diagnostics, and session ownership. Registry patches use the same optional
fields as fresh-session overrides, including explicitly empty values. The test
environment guard supports unsetting and fails loudly if restoration fails.

Validation: the QDMI C++ suite passes in release and Clang builds (465 passed,
one expected skip). `uv run --no-sync pytest test/python/qdmi test/python/plugins
-q` passes all 463 tests. Both `uvx nox -s lint` and `uvx nox -s cpp-lint` pass.
No dependencies or public wrapper API changes require migration instructions.
Hosted CI remains a separate check; Windows behavior was not executed locally.

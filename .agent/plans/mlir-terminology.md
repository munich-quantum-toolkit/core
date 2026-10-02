# MLIR terminology and public naming

Status: complete.

Compiler C++, Python, CLI, diagnostics, tests, and documentation use the
terminology established by issue #2251 and the PR #2149 discussion. The approved
clean renames apply to unreleased v4 interfaces without deprecated aliases. The
accepted input subset, program ownership, static gate-count behavior, and
emitted representations are preserved.

The [audit record](../audits/mlir-terminology.md) contains the baseline, source
evidence, complete rename table, downstream impact, retained terms, and local
validation. The Ponytail review removed a redundant gate-emission forwarder; the
final review found no remaining complexity to cut.

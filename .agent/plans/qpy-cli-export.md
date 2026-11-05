# Preserve target information in CLI QPY export

Status: simplification implemented and validated locally.

## Design

The Python CLI recognizes QPY output and forwards compilation options to the
native compiler. It loads the optimized QCO result, converts it to QC, and calls
an internal export-only binding. That binding reads the attached target
environment and uses the existing Qiskit exporter and QPY serializer. It does
not optimize or route the circuit again. Serialization completes before opening
the destination file.

The same target recovery serves `compile_program(output=OutputFormat.QPY)`.
Public `to_qiskit(target=None)` retains its existing behavior. Native
intermediate compilation can select a QPY payload without loading a serializer;
execution still requires one. No additional public program loaders or
target-attribute parsers are needed for the CLI.

Custom target aliases use Qiskit's `Gate` with an explicit definition: QPY
otherwise restores renamed standard gates with their original names. This also
preserves parameter order and the semantics of aliases colliding with standard
names.

## Validation

Retain tests of independent QPY decoding, sparse physical addresses, logical
layout, gate names, symbolic parameters, wire order, exact phase, optional
Qiskit, informational commands, and output preservation on failure. Exercise
native input formats, custom pipelines, and debugging-option forwarding through
the CLI. Remove tests of the duplicated Python option parser and the public APIs
introduced solely to support it.

The follow-up ponytail review found one reproducible command-handling bug:
`--qdmi-list-devices=true` and `=1` listed devices and then attempted to load a
nonexistent compiled file. The launcher now recognizes explicit boolean values;
`=false` continues through compilation. CLI regressions cover both paths and
preservation of an existing output file.

The branch was rebased onto main at `1a6b190719`. `git range-diff` confirmed
that the rebase retained every commit's changes. Validation results are recorded
in the PR description. Native coverage includes QPY-only and multi-format device
selection, payload restrictions, and missing-serializer rejection. No hardware
jobs were submitted; hosted CI was not monitored.

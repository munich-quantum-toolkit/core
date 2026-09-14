# Preserve compiler input identities

Status: implementation and semantic validation complete; final lint and review
remain.

## Goal and scope

Qiskit imports currently retain parameter names and sharing but exports create
new identities. Preserve original free parameter identities through QC/QCO
conversion, optimization, and MLIR serialization so callers can bind using the
original parameters. The MQT dialect owns an optional opaque 128-bit input ID;
Qiskit UUID conversion stays at the adapter boundary. No Qiskit synthesis or
transpilation is used.

## Decisions

Attach `mqt.input_id` to named function arguments as an `i128` attribute. Treat
all bit patterns as IDs, require uniqueness within each function, and carry the
attribute with its input through transformations. Removing an input removes its
ID. Frontends that omit it keep the existing fresh-identity export behavior.
Parameter vectors already retain a group identity: restore UUID-shaped group
identities as vector UUIDs, while preserving other valid group identifiers.
Local helper and loop parameters retain their existing lexical sharing; this
change preserves externally bindable free inputs.

## Validation and limits

The built translation suite passes all 385 tests, and all 33 native MQT metadata
tests pass. Stub generation, repository lint, and changed-file C++ lint pass.
The regressions cover binding with the original parameter objects after QC/QCO
conversion, optimization, and serialization, sparse vectors, extreme UUIDs, and
invalid shared metadata.

Private helper and loop identities are not guaranteed. OpenQASM and QIR do not
retain compiler input identity metadata. Vector element IDs must have one root
UUID; inconsistent IDs are rejected at the SDK export boundary.

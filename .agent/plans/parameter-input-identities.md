# Preserve compiler input identities

Status: implementation and semantic validation complete; final lint and review remain.

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

## Work remaining

- [x] Add the shared metadata contract and verifier tests.
- [x] Carry parameter IDs through import/export and test original-object binding.
- [x] Build and regenerate stubs; pass all 385 translation and 33 native metadata tests.
- [ ] Complete repository and C++ lint, then publish the focused draft PR.

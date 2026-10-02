# QDMI plugin validation for v4

Status: complete.

## Scope and ownership

The warning-job ownership and PennyLane conversion fixes are retained. Both SDK
plugins now validate supported behavior using released QDMI APIs:

- PennyLane: named-wire deferred measurements, graph decomposition target gates,
  standard tracking, modern shot examples, and explicit deferred-only MCM
  capabilities. QNodes or tapes must specify shots; device-level shots and the
  implicit 1024-shot default are removed.
- Qiskit: native width and placement validation in the built-in QASM
  serializers, after preprocessing; validate nested operations when a backend
  extension explicitly enables control flow. Custom serializers retain
  compilation control.

The native operation metadata owns placement support. The public Qiskit Target
may hide native sites or expose fictional pairs, so it cannot validate the
output of preprocessing. Reuse the backend's existing native placement decoder
and cache normalized metadata for its fixed device session. Do not change the
previously reviewed variable-arity or alias Target construction behavior.

PR #2226 at `29d17ced47b8c5cd33d8d19c39c662bf82708d68` supplies useful
independent execution-configuration and nested-validation patterns. Exact
payload descriptors, optional feature inference, native one-shot execution, and
opaque result decoding remain dependent on the proposed QDMI contract. Retain
the existing serializer registry and current result APIs. Native batching
remains separate.

## Validation and remaining limits

`uv run --no-sync pytest -n0 test/python/qdmi test/python/plugins` passes all
500 tests. Regressions cover QASM2/QASM3 placements, whole-batch rejection,
preprocessing to hidden sites, mapped control-flow operands, named-wire reset
and feedback, graph decomposition, explicit shots, and tracking through failure.
Full repository lint passes. The unchanged native fix passes 104 driver tests
and full C++ lint; the configured C++ build and both affected MLIR binaries also
pass. The final diff contains no QDMI dependency or generated-stub changes.

No native one-shot or new payload contract is inferred. The Qiskit Target
changes excluded from this scope remain separate. Hosted CI and publication have
not been performed.

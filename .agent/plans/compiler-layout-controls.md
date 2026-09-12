# Native compiler layout controls

Status: in progress; implementation and validation remain.

## Contract and ownership

Add a native target compilation API that returns initial and final target site
IDs in input allocation order and accepts an optional complete initial layout.
The compiler owns tracking and routing; this change uses no Qiskit code. The
result is a detached compilation snapshot, so later transformations cannot
silently leave stale layout metadata attached to an IR module.

Support fixed-size allocations in the entry block, flattened in block order and
ascending tensor index. Preserve idle input wires during this opt-in pipeline.
Reject dynamic, nested, or already physical allocations, and malformed layouts.
The ordinary compilation API keeps its current optimization behavior.

## Design

Insert temporary tagged barriers at allocation boundaries before cleanup. They
keep tensor slots live and identify source wire order even if extraction order
changes. The native mapper consumes the tags, records its initial and final
layouts, and removes the barriers before native synthesis. A final pass
publishes the result only after target conformance succeeds. No persistent IR
layout attribute or SDK-specific state is introduced.

The new API does not import or export SDK layout objects, model partial layouts,
or track resource changes before this compilation call. This is a focused native
foundation for consumers such as Bench, not the entire layout interchange issue.

## Remaining work

- Implement preparation, mapping selection, reporting, and Python access.
- Test forced routing, idle wires, sparse sites, allocation order, rejection,
  and semantic equivalence without using another compiler.
- Build, generate stubs, run repository/native checks, and open the draft PR.

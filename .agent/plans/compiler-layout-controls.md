# Native compiler layout controls

Status: complete.

## Outcome and scope

`QCOProgram::compileForTargetWithLayout` and its Python binding compile in place
and return initial and final target site IDs in input allocation order. The
caller can supply a complete initial layout or select automatic placement with
`MappingOptions`. The low-level pipeline publishes its result after target
conformance succeeds. Failed preparation or synthesis leaves the caller's prior
result unchanged.

The compiler owns tracking and routing. No SDK code is used. Input allocations
must have fixed sizes in the entry block. Tensor slots follow ascending index
order, and idle slots remain represented and count against target capacity. The
ordinary compilation API retains its existing optimization behavior.

## Decisions

`MappingResult` is a detached compilation snapshot rather than persistent IR
metadata. Later transformations cannot silently leave stale layout attributes
attached to the program. Callers retain the snapshot only for the compilation
that produced it.

Preparation inserts temporary tagged barriers before cleanup. They keep tensor
slots live and identify source order even when first-use order differs.
Canonicalization preserves these boundaries until the native mapper consumes
them. Mapping records its initial and final layouts and removes the boundaries
before native synthesis. The implementation lives in the QCO mapping subsystem;
the compiler pipeline and bindings expose the result without recreating
tracking.

## Validation and limits

The compiler bindings, native DD simulation, and translation suites pass all 516
selected Python cases, with both bundled device providers available. The final
rebuilt extension passes all 26 layout regressions. All 44 native target and
mapping tests pass. Builds, generated stubs, repository lint, and changed-file
C++ lint pass.

Regressions cover complex-amplitude equivalence, forced routing through unused
sites, measurement feedback, sparse site IDs, idle slots, scalar and tensor
allocation order, empty inputs, malformed layouts, and failure publication. The
documented example runs successfully.

Native tests also cover complete routed unitaries, user barriers, and
unsupported quantum entry arguments. The instrumented native tests cover 233 of
252 changed executable production lines (92.5%) in the local coverage build.

Tracking starts at the compilation call; it cannot recover earlier resource
changes. Dynamic, nested, and already physical allocations are unsupported.
Partial layout constraints, SDK layout interchange, CLI layout input, and
snapshot serialization remain outside this change. Existing target and payload
limitations still apply.

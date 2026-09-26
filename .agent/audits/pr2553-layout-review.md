# Compiler layout audit

Status: complete. Base: upstream `main` at
`6ad2b659a89f21482975ff03c42a2d3c53c09696`. Compatibility target: PR [#2607] at
`d8fdef1b4c112b02cb7d463e91ac73f9ada7b4b8`.

## Result and applied changes

PR [#2553] remains relevant: main does not retain imported Qiskit layout
provenance or expose native source-input placement. PR #2607 changes routing
state and scoring; it does not supply either public contract.

- Source-tag verification uses the constant allocation operand, matching
  preparation and the owning allocation verifier. A fixed extent can have a
  dynamic result type. The regression covers idle and active tensor slots.
- Tensor shrinking relies on the verified source-tag width. The duplicate guard
  is removed. All-to-all final placement copies the initial snapshot; an unused
  pipeline include is removed.
- Native results include the exact physical routing permutation, including
  workspace sites. `to_qiskit(target=..., mapping_result=...)` constructs a full
  `TranspileLayout`. Source mappings use target site IDs; the routing
  permutation and Qiskit layouts use indices in target site order.
- Documentation describes current contracts, input identities, and export
  workflows. Historical implementation and review narration is removed.

The independent Qiskit specialist checked Qiskit 2.5.2 source and the official
[TranspileLayout reference]. Native exports reproduce complete operators after
Qiskit applies initial placement and routing. The checks include sparse sites,
workspace, explicit and automatic placement, source SWAPs, idle logical inputs,
and phase. Imported partial layouts preserve missing assignments and register
membership; SDK helpers may require complete maps.

## Assumptions and limits

- One direct entry point owns provenance; no competing nested entry points.
  Module-boundary validation is separate from attribute verification because
  MLIR parsing uses a temporary enclosing module.
- Native source order is entry-block allocation order, then ascending tensor
  slots. Allocations have compile-time constant sizes. Idle inputs consume
  target capacity and retain identities through workspace routing.
- Explicit placement is complete. Tracked compilation returns its detached
  snapshot only after the complete pipeline and linearity verification succeed.
- Use a native snapshot with the same target and unchanged compiled program. It
  describes native allocation inputs; it does not compose imported Qiskit
  provenance. Exporters require explicit discard of invalidated provenance.
- Adaptive all-to-all placement requires static result types for retained
  tensors. Ordinary and tracked compilation share this existing restriction;
  frontends produce static types for fixed allocations. General tensor shape
  refinement is outside this layout change.

The full native permutation costs O(target sites) storage and linear snapshot
work. Ordinary compilation does not construct it. No new performance claim is
made. Source tags, workspace tracking, partial metadata support, and the
module-wide preparation scan retain real contracts and are not redundant.

## Test reductions

The four edited test files shrink by 267 net lines, including the new
regressions. The reduced suite keeps these consumer oracles:

| Removed overlap                                            | Surviving coverage                                                                                         |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Python basis-state and complete-unitary routing duplicates | Native complete complex unitary, including idle slots; Qiskit full physical operator                       |
| Large seed/size automatic-compilation matrix               | Existing tensor/workspace test compares ordinary and tracked compilation on routed and all-to-all targets  |
| Repeated Python capacity/runtime-size/site-error cases     | Native input validation; one Python exception/diagnostic check                                             |
| Separate copy/conversion/serialization layout matrix       | Real transpiler-added-ancilla round trip through all boundaries, checking Qiskit helper maps and operators |
| Repeated invalidation/discard and export-format matrices   | Native transformation boundaries and each converter; Qiskit invalidation/discard behavior; CLI flag tests  |
| Repeated field equality after full attribute equality      | Full schema round trip and malformed-schema validation                                                     |

The GIL, detached snapshot, feedback destinations, source order, sparse sites,
empty inputs, source-register membership, partial assignments, and failed final
synthesis checks remain.

## PR #2607 integration

A combined checkout resolves the two overlapping files, `TargetCompilation.cpp`
and `Mapping.cpp`, and validates the complete native and focused Python suites.
The integration must preserve these details:

- Keep the tracked pipeline wrapper and use #2607's shared native-synthesis
  pipeline. Both ordinary and tracked compilation run the same passes.
- Translate tracking to #2607's physical-wire `RoutingState`; read final layout
  from `state.layout`. Explicit placement bypasses candidate trials, but still
  initializes the native-cost table used by hot routing guidance.
- All-to-all placement does not materialize unused workspace sites. Do not
  construct `WireIterator` objects for those absent values when adapting #2607's
  placement return value. The all-to-all caller does not use routing wires, so
  it can return an empty vector.
- Capture the complete initial physical-to-program map before routing and
  convert it to final physical positions afterward. Do not infer workspace
  motion from source-only final mappings.

These are integration changes for #2607's proposed mapper, not additional
routing machinery in this PR. The combination needs conflict resolution; a plain
automatic merge is not sufficient.

## Validation

- Clang 23 Release with ThinLTO and mold builds successfully.
- `ctest --preset release-clang-ipo -j 8`: 3,651 passed, one existing skip
  (`ScQDMIJobSpecificationTest.QueryJobId`).
- The combined PR #2607 build: 3,672 passed with the same skip.
- `pytest test/python/test_mlir.py test/python/test_mlir_qiskit_translation.py test/python/qdmi/test_compilation.py`:
  634 passed on each build, using Qiskit 2.5.2 and local DDSIM/SC device
  libraries.
- `uvx nox -s stubs` and
  `cmake --build --preset release-clang-ipo --target mlir-doc` succeed.
- `uvx nox -s lint` and `uvx nox -s cpp-lint` pass; C++ lint checks every line
  of all changed C++ files against main.

[#2553]: https://github.com/munich-quantum-toolkit/core/pull/2553
[#2607]: https://github.com/munich-quantum-toolkit/core/pull/2607
[TranspileLayout reference]: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.TranspileLayout

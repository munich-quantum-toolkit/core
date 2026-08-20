# Preserve Qiskit parameter-vector provenance

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

The scalar Qiskit translation represents each free parameter as a named `f64`
compiler input. That representation preserves a standalone parameter, but it
cannot tell whether a symbol came from a Qiskit `ParameterVector`. Treating a
vector element as a standalone parameter changes Qiskit's positional and
vector-level binding behavior.

After this change, import and export preserve the identity, name, declared size,
element index, and order of genuine parameter vectors. Standalone names such as
`theta[10]` remain standalone. A user can observe the result by round-tripping a
circuit that uses a sparse `ParameterVector` and binding the restored circuit
either positionally or with the restored vector object.

This task is the final optional leaf on the scalar, expression-capture,
structured-export, and measurement-deferral stack. Keeping vector provenance in
this leaf lets the structured exporter remain independently reviewable. The
stack is based on the first-class CBit representation from #2158. This task does
not change the supported arithmetic expression language.

## Progress

- [x] (2026-08-19 14:20Z) Separate the vector provenance changes from the scalar
      symbolic-parameter implementation.
- [x] (2026-08-19 14:20Z) Restore explicit group metadata, import and export
  validation, Qiskit object reconstruction, and focused vector tests.
- [x] (2026-08-19 14:25Z) Run the focused Python suite and native compiler tests
      on the current #2158 head.
- [x] (2026-08-19 19:50Z) Rebase the follow-up onto the scalar PR after #2158
      merged, rebuild the bindings, and rerun the Qiskit and compiler suites.
- [x] (2026-08-19 20:04Z) Preserve valid elements whose index is at or beyond a
      zero-length or shorter parent vector, convert the rejection cases to
      semantic round trips, and pass all 171 Qiskit translation tests.
- [x] (2026-08-19 20:26Z) Rebase the vector leaf onto its measurement-deferral
      parent, aggregate vector metadata across nested writer blocks, add a
      sibling-control-flow regression, pass all 213 translation and 133 compiler
      tests, and pass the complete repository lint session.

## Surprises & Discoveries

- Observation: A displayed name does not identify a parameter vector. Qiskit
  permits a standalone `Parameter("theta[10]")`, which must not become an
  element of a vector named `theta`. Evidence: the standalone bracket-name
  regression checks that no group attributes appear in the imported MLIR.
- Observation: Sparse elements can have an index larger than the vector's
  recorded size. Direct construction of the element preserves Qiskit's source
  metadata without enlarging the vector.
- Observation: Allocating a public `ParameterVector` from unchecked metadata
  could consume unbounded memory. Import and export therefore limit both one
  declared group and the combined declared group sizes to 65,536 elements.
- Observation: Finalizing sibling control-flow blocks independently creates one
  vector object per block when different elements of the same vector appear in
  different branches. Evidence: an `if` body using `theta[0]` and its `else`
  body using `theta[1]` originally exported two distinct vector UUIDs.

## Decision Log

- Decision: Store explicit group identity, group name, element index, and
  declared size on each compiler input. Rationale: explicit metadata preserves
  vector semantics without parsing a display name or storing a frontend object
  in MLIR. Date/Author: 2026-08-19 / Codex.
- Decision: Reconstruct one shared Qiskit vector object for each group identity
  and replace the temporary native scalar symbols only after native circuit
  construction finishes. Rationale: the Qiskit 2.5 C API can construct scalar
  parameters but cannot construct `ParameterVectorElement` objects. Date/Author:
  2026-08-19 / Codex.
- Decision: Keep the aggregate size bound in both generic preflight and the
  version-specific writer. Rationale: generic validation gives a stable error
  before construction, while the adapter check protects its public Python
  allocation boundary. Date/Author: 2026-08-19 / Codex.
- Decision: Treat the recorded vector length and element index as independent
  metadata. Rationale: Qiskit permits a `ParameterVectorElement` whose parent is
  empty or shorter than its index, and its constructor preserves that sparse
  provenance without resizing the parent. Date/Author: 2026-08-19 / Codex.
- Decision: Aggregate parameter-group metadata across the recursive writer tree
  and restore vector objects only after the top-level control-flow circuit is
  assembled. Rationale: one top-level replacement preserves a shared vector
  identity across sibling and nested blocks. Date/Author: 2026-08-19 / Codex.

## Outcomes & Retrospective

The vector implementation is separated from the scalar-symbol and structured
export contracts. The final leaf restores exact vector behavior across flat and
structured circuits without increasing the review scope of the earlier PRs. All
213 Qiskit translation tests and all 133 compiler tests pass after the final
linear-stack rebase. Publication remains deliberately separate from this local
validation.

## Context and Orientation

`bindings/mlir/qiskit/QiskitTranslation.h` contains the frontend-neutral
`Parameter` tree shared by the Qiskit reader and writer. This task adds optional
`ParameterGroup` metadata to a symbol leaf. The metadata never appears on a
numeric or operator node.

`bindings/mlir/qiskit/QiskitImport.cpp` converts normalized Qiskit parameters to
function inputs. `bindings/mlir/qiskit/QiskitExport.cpp` converts named `f64`
function inputs back to normalized parameters. The four attributes used for
group metadata are declared in `mlir/include/mlir/Dialect/Utils/Utils.h` as
`mqt.input_group`, `mqt.input_group_name`, `mqt.input_group_index`, and
`mqt.input_group_size`.

`bindings/mlir/qiskit/Qiskit2_5.cpp` is the only version-specific adapter. It
reads public Qiskit vector properties during import. During export, it first
creates the circuit with native scalar parameters. It then reconstructs public
`ParameterVectorElement` objects and assigns them once to the finished top-level
circuit after aggregating group metadata from all nested block writers.

The behavioral tests are in `test/python/test_mlir_qiskit_translation.py`. The
public support table is in `docs/mlir/python_compiler_collection.md`.

## Plan of Work

Extend `Parameter` with an optional `ParameterGroup`. Define a maximum declared
group size of 65,536. In the Qiskit 2.5 reader, detect a genuine
`ParameterVectorElement`, read its parent vector identity, name, declared size,
and numeric index, and attach that metadata to the normalized symbol.

In `QiskitImport.cpp`, compare all metadata when two Qiskit objects share one
symbol identity. Reject inconsistent aliases, oversized groups, and an oversized
aggregate before creating the compiler module. Add the four input attributes
only as an all-or-none set.

In `QiskitExport.cpp`, require the four attributes together. Check their types,
ranges, group consistency, per-group size, and aggregate size during preflight.
Keep bracketed standalone names free of group metadata.

In `Qiskit2_5.cpp`, preserve native symbol sharing while building the circuit.
At top-level `finish()`, collect exported symbols recursively from every block
writer, group them by group identity, construct one public `ParameterVector` per
group, construct or select each recorded element, and replace all temporary
symbols in the assembled circuit tree. Preserve sparse indices and the recorded
size.

Keep positive tests for positional binding, vector-level binding, sparse
indices, large sparse indices, same-name distinct groups, sibling control-flow
blocks, and loop induction parameters. Keep fail-closed tests for malformed,
inconsistent, oversized, and aggregate metadata. Every failure test must verify
that its source circuit or program did not change.

## Concrete Steps

Run all commands from the repository root. Configure and build the bindings if
needed:

    cmake --preset release
    cmake --build build/release --parallel 8 --target \
      mqt-core-mlir-unittests-compiler
    cmake --build build/python/MinSizeRel --parallel 8 --target \
      mqt-core-mlir-bindings

Run the focused behavior and native regression suites:

    uv run --no-sync pytest -q -o addopts= \
      test/python/test_mlir_qiskit_translation.py
    build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
      --gtest_brief=1

Regenerate stubs and run the repository checks before publication:

    uvx nox -s stubs
    uvx nox -s lint
    uvx nox --non-interactive -s docs
    git diff --check

## Validation and Acceptance

A 12-element vector round trip must preserve the same element order and numeric
operator after positional binding. Binding with the restored vector object must
produce the same circuit. A sparse element must retain its original numeric
index and recorded vector size. Two vectors with the same display name but
different identities must remain distinct.

A vector element whose parent has length zero or whose index is at or beyond the
recorded length must preserve both values exactly and remain bindable after the
round trip.

Different elements of one vector used only in sibling control-flow blocks must
be restored with one shared vector UUID and support vector-level binding.

A standalone `Parameter("theta[10]")` must remain a standalone parameter and
must not acquire `mqt.input_group` metadata. A vector element used as a loop
induction value must remain lexically bound rather than become a free function
input.

Import and export must fail before mutation when metadata is incomplete,
inconsistent, larger than 65,536, or has a combined declared size larger than
65,536. The diagnostic must identify parameter-vector metadata or the applicable
size limit.

The focused Qiskit file, the complete compiler binary, stub generation,
documentation, lint, and `git diff --check` must pass after the final rebase.

## Idempotence and Recovery

All test and build commands are repeatable. Build output stays under `build/`
and is not committed. If the measurement-deferral parent or #2158 changes,
rebase this one vector commit and resolve only the files listed in this plan.
The measurement-deferral parent remains a safe rollback because it rejects
vector elements before module construction.

Do not push, create a PR, or edit public GitHub text without current human
authorization. Preserve unrelated worktree changes.

## Artifacts and Notes

The focused test set covers a normal vector, a sparse vector, a sparse element
with a large index, elements outside zero-length or shorter parents, one vector
shared by sibling control-flow blocks, same-name distinct vectors, a bracketed
standalone name, loop binding, UUID alias conflicts, malformed attributes, and
individual and aggregate size limits.

The final Qiskit validation on the linear stack produced:

    213 passed in 2.19s
    [  PASSED  ] 133 tests.

## Interfaces and Dependencies

`ParameterGroup` contains a stable group identity string, a display name, a
`uint64_t` element index, and a `uint64_t` declared size. `Parameter::group` is
an `std::optional<ParameterGroup>` that is present only for a symbol leaf.

The implementation uses Qiskit 2.5 public Python classes only inside
`Qiskit2_5.cpp`. Generic importer and exporter code sees only normalized C++
metadata and MLIR attributes. No SymPy dependency, Qiskit object, or expression
string is stored in MLIR.

Revision note (2026-08-19): Created this focused plan when vector provenance was
split from the scalar symbolic-parameter implementation. Rebased it as the
optional leaf after measurement deferral and added recursive group restoration.

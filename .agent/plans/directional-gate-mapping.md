# Compile directional target gates through native synthesis

This ExecPlan follows `.agent/PLANS.md` and records the supported contract,
implementation, and validation for ordered compiler-target applicability.

## Purpose / Big Picture

Compile gates for devices that support an entangler in only one operand order.
Routing makes operands adjacent; native synthesis repairs their direction and
final conformance checks exact physical sites. Alternating CX directions on two
adjacent sites must not introduce routing SWAPs.

## Progress

- [x] (2026-09-03) Preserve ordered operation support and calibration metadata.
- [x] (2026-09-04) Remove directional routing and its dedicated wrapper.
- [x] (2026-09-04) Replace ambiguous-site analysis with a checked staged walk.
- [x] (2026-09-04) Remove whole-module cloning for failed synthesis.
- [x] (2026-09-04) Add focused regressions and align public pass documentation.
- [x] (2026-09-04) Apply specialist, adversarial, and Ponytail Review feedback.
- [x] (2026-09-04) Pass 303 focused tests, documentation, and repository lint.
- [x] (2026-09-04) Attempt full C++ lint and analyze changed sources directly;
      record the unrelated build blocker below.
- [x] (2026-09-04) Unify site tuples across the target, attributes, QDMI, and
      Python.
- [x] (2026-09-04) Remove synthesis planning and repeated matrix extraction.
- [x] (2026-09-04) Validate the revised model and obtain adversarial review.

## Decision Log

On 2026-09-03 the maintainer approved adjacency-only routing. Direction repair
belongs to synthesis. Weighted routing edges remain possible future work; there
is no current need for an extra cost wrapper.

On 2026-09-03 the maintainer approved requiring one known physical site per
quantum value, equal branch-result sites, and site-preserving loop backedges.
These conditions are checked, including for all-to-all placement and standalone
passes. Ordinary structured control flow remains supported.

On 2026-09-03 the maintainer approved removing synthesis rollback. Compilation
runs in place; callers must not rely on program contents after failure. Generic
capability tuples and constant-time ordered-pair support queries remain part of
the target model.

On 2026-09-04 the maintainer approved one `site_tuples` list and no
applicability enum. An empty list means general applicability; a nonempty list
contains every supported ordered placement with optional calibration. Missing
values inherit operation defaults. The QDMI adapter omits operations reported
with no supported placements and retains uncalibrated supported tuples.

## Surprises & Discoveries

An executed two-site probe produced five native CXs with directional routing and
two with direct synthesis. The extra SWAP is avoidable. Another valid-IR probe
passed a site through `scf.execute_region`; name-only fallback incorrectly
accepted a reversed CX. Unknown site transfers must fail with a diagnostic.

Explicit mapping realigns structured region exits to physical slots. All-to-all
placement only replaces allocations, so site consistency must be checked rather
than assumed. Runtime symmetric gates such as RXX need direct operand reordering
because their matrix is unavailable at compile time.

## Context and Orientation

`mlir/lib/Compiler/Target.cpp` owns immutable target capabilities and basis
selection. A usable synthesis basis supplies one-qubit gates on every site and
an entangler on every routing edge in at least one direction. Each supported
site tuple may carry calibration overrides.

`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` performs placement and
routing. `mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/TargetSynthesis.cpp`
then assigns exact sites, preserves or reorders native gates, and decomposes
other supported gates. Its conformance pass checks emitted capabilities.

## Plan of Work

First remove `mlir/include/mlir/Compiler/MappingTarget.h`, its implementation
and dedicated tests, and restore topology-only mapping and build wiring. Prove
that alternating CXs need no routing SWAPs and only two native entanglers.

Next use MLIR's staged operation walk and one site map. Propagate sites through
unitaries, reset, and measurement, seed supported region arguments, and compare
branch results and loop backedges. Reject unknown or conflicting sites. Remove
the module clone and duplicate planning. Preserve matrix/output permutation for
directional synthesis and direct symmetric operand reordering.

Finally update `docs/mlir/target_compilation.md`, pass descriptions, and the
existing changelog entry. Keep regression tests in the established native
synthesis and compiler test suites. Obtain independent reviews after the first
implementation, then incorporate the separate Ponytail Review findings.

## Concrete Steps and Validation

Run from the repository root with the configured LLVM/MLIR 23 installation:

    cmake --preset release
    cmake --build --preset release --target mqt-core-mlir-unittests-compiler mqt-core-mlir-unittest-mapping mqt-core-mlir-unittest-target-synthesis mqt-core-mlir-unittest-mqt-ir -j 8
    build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler
    build/release/mlir/unittests/Dialect/QCO/Transforms/Mapping/mqt-core-mlir-unittest-mapping
    build/release/mlir/unittests/Dialect/QCO/Transforms/NativeSynthesis/mqt-core-mlir-unittest-target-synthesis
    build/release/mlir/unittests/Dialect/MQT/IR/mqt-core-mlir-unittest-mqt-ir
    uvx nox -s stubs
    SKBUILD_CMAKE_ARGS=-DBUILD_MQT_CORE_QDMI_SC_DEVICE=ON uvx nox -s tests-3.13 -- test/python/test_mlir.py -q
    uvx nox -s cpp-lint
    uvx nox --non-interactive -s docs
    git diff --check
    uvx nox -s lint

Successful output must verify, retain exact ordered target applicability and
quantum semantics, and support consistent if/switch/for/while site transfers.
Unknown sites, conflicting branch exits, and changing loop-backedge sites must
be diagnosed. Failed compilation need not preserve input IR. C++, Python, and
serialized targets use only `site_tuples` for ordered availability and
calibration.

The tuple simplification removes duplicate lists, attributes, and validation
from C++, MLIR, QDMI, and Python. Native synthesis processes users before their
producers, keeping original site facts valid while rewriting each operation
immediately. Use the existing bounded site walk: generic control-flow interfaces
prune known loop edges and require extra exceptions for this contract.

## Idempotence and Recovery

Builds and checks are repeatable. Preserve unrelated changes and keep generated
build output untracked. No dependency additions or generated-file edits are
needed.

## Outcomes & Retrospective

The target model now has one tuple list with optional calibration. Its enum,
duplicate lists, attributes, validators, and serialization paths are removed.
Synthesis checks and rewrites each gate in reverse order, without a separate
plan or repeated matrix extraction. This round removes 284 production lines.

Specialist and adversarial review found no remaining blockers. Adversarial
review retained a compact shared matrix guard for unsupported multi-target
control shells; its regression verifies that the input is valid and linear
before checking the diagnostic. Ordinary dependent rewrites retain semantic
equivalence.

All 305 focused C++ tests pass: compiler 153, mapping 94, target synthesis 43,
and MQT IR 15. All 49 Python MLIR tests pass. Python stubs are regenerated, and
strict documentation and repository lint pass. Full C++ lint stops before
analysis because unchanged QIR runtime test executables have unresolved QTensor
symbols. Building the ten changed C++ translation units directly succeeds; the
same whole-file linter reports zero findings across all ten files. No lint
configuration or unrelated build wiring was changed.

Revision note: aligned the scope with the approved routing, site, and failure
contracts while retaining exact device metadata.

# Integrate QDMI devices with the MLIR compiler

Status: historical implementation record.

## Goal and scope

The compiler-target foundation, target-backed mapper, target-independent
two-qubit gate fusion, target-native synthesis, conformance verifier, and
canonical target pipeline are now merged. This final slice connects those
compiler-owned abstractions to QDMI without duplicating them. After the change,
a C++ or Python user can snapshot a circuit-model `fomac::Device` into an
immutable `mlir::CompilerTarget` and compile for it after the device and session
have been destroyed. The `mqt-cc` executable can list configured QDMI devices,
select one by stable identifier, and run the same target pipeline.

The bridge retains device names, site names, topology, T1/T2 values, operation
capabilities, fidelities, and available durations. It rejects neutral-atom zone
models with a direct diagnostic because this target pipeline currently models
circuit sites only. CoreFoMaC remains MLIR-free, `MQTCompilerTarget` remains
FoMaC/QDMI/CoreIR-free, and no second target DTO or dynamic driver boundary is
introduced.

The observable proof combines focused adapter tests and Python target
construction and compilation tests. The existing pull request is then rewritten
from current `main` as this thin integration rather than retaining its
historical merge-heavy implementation.

## Constraints

- the live compiler already owns every semantic and pipeline abstraction needed
  by this slice. The historical PR's `fomac::Target`, targeting pass, mapper
  augmentation, native-gate menu, and duplicated pipeline are obsolete and must
  not be ported.

- `CompilerTarget::Operation` deliberately models homogeneous target-wide
  support, while QDMI can report a restricted site list. The adapter must
  therefore verify that a one-qubit operation covers every site and that a
  two-qubit operation covers every topology edge or all-to-all pair before
  treating it as target-wide. Ordered site tuples then carry calibration
  overrides only.

- the bundled IQM Garnet and Emerald models, stable registry IDs, runtime
  assets, site names, T1/T2 values, and fidelities are already present on
  `main`. This slice consumes those models rather than adding fixtures or
  provenance text.

- `mqt_copy_qdmi_runtime` already copies built-in provider libraries, registry
  manifests, and assets beside an executable. The CLI does not need another
  plugin loader or packaging mechanism.

- the old PR description and most unresolved threads refer to deleted
  predecessor abstractions. The final branch should satisfy the remaining
  behavior through the merged prerequisite PRs and this adapter, then describe
  only the actual user workflow.

- `mqt-cc` is not currently shipped as part of the Python wheel. This slice
  keeps that packaging boundary unchanged; Python target compilation is provided
  directly by the extension and packaged QDMI providers.

- MQT Core does not currently export or install the MLIR compiler libraries and
  generated headers as a consumable SDK. Exporting the adapter would require
  exporting the compiler pipeline, dialect libraries, generated headers, and
  their MLIR dependency closure. This slice therefore documents the C++ and
  `mqt-cc` workflows as source-build interfaces and leaves a coherent MLIR
  SDK/package boundary to a dedicated follow-up.

- the previously merged target pipeline ran the generic QCO cleanup after
  target-native synthesis and conformance. Its canonicalization patterns can
  rewrite `qco.r` operations with special angles back to `qco.rx` or `qco.ry`,
  making a formerly conforming Garnet result non-native. A real Garnet
  compilation exposed this issue; unit tests that stopped at conformance did
  not.

- a clean build with all three built-in QDMI providers disabled initially
  motivated conditional provider-backed tests. Follow-up review established a
  simpler boundary: the complete MQT Core test suite requires all bundled
  providers, while embedded and other non-test builds may still select or omit
  providers independently.

- QDMI operation site tuples are ordered, while the compiler deliberately models
  an undirected topology and homogeneous bidirectional gate support.
  Canonicalizing a one-way two-qubit site list would silently widen the device
  contract. The adapter must require both orientations for directional and
  unknown operations while allowing proven operand-symmetric gates such as CZ to
  report each edge once. Missing two-qubit site information is likewise
  insufficient when a device reports an explicit topology.

## Decisions

- add one public adapter function,
  `mlir::compilerTargetFromDevice(const fomac::Device&)`, in a small library
  that links `MQTCompilerTarget` and `MQT::CoreFoMaC`. Rationale: dependency
  direction stays acyclic and compiler semantics remain owned by MLIR while
  callers opt into the live-device bridge.

- snapshot all QDMI data eagerly and return a detached `CompilerTarget`.
  Rationale: compiler execution must not depend on a live QDMI session or
  provider handle, and the target already has shared immutable storage for cheap
  copies.

- reject any device site that is a zone and any zoned operation. Rationale:
  circuit-model topology and neutral-atom zones have different semantics;
  silently flattening zones into qubits would create an invalid target.

- reject explicit QDMI operation site lists that are not homogeneous over the
  compiler target. Rationale: the compiler target intentionally represents one
  target-wide gate set; silently widening a restricted QDMI operation would
  allow synthesis to emit an unsupported gate.

- accept one reported orientation only for a conservative set of
  operand-swap-invariant gates and require both ordered tuples for every site
  pair otherwise. Rationale: this preserves IQM's symmetric CZ data without
  misrepresenting directional or unknown QDMI operations as bidirectional.

- expose `CompilerTarget.from_device(device)` from `mqt.core.mlir`, not
  `Device.target()` from `mqt.core.fomac`. Rationale: CoreFoMaC and its binding
  remain independent of MLIR, target ownership is visible in the compiler
  namespace, and compilation APIs accept one explicit target type.

- add `QCOProgram.compile_for_target` and an optional `target` to the sole
  `compile_program` function. Rationale: Python mirrors the two canonical C++
  entry points and does not accept devices, coupling maps, native-gate strings,
  or compatibility shims.

- add only `--qdmi-list-devices`, `--qdmi-device`, and `--qdmi-config` to
  `mqt-cc`. Configure the registry before the first driver access, snapshot the
  selected device, and invoke the existing canonical target pipeline. Rationale:
  this is the irreducible user surface and preserves current provider discovery
  rather than introducing another dynamic boundary.

- reject target compilation when the requested output or custom pass sequence
  cannot preserve or safely compose the target assignment, using the validation
  already centralized in `runDefaultPipeline`. Rationale: options must not be
  silently ignored and the CLI must not replicate pipeline rules.

- perform generic QCO cleanup before target-native synthesis, retain only CSE
  and dead-value cleanup afterward, and run conformance last. Rationale: no
  target-independent canonicalizer may reintroduce a gate outside the native
  operation set after it has been synthesized and verified.

- do not add a partial install/export path for only the adapter. Rationale: the
  repository has no installed MLIR SDK boundary, and exporting a single facade
  while omitting the pipeline, dialects, and generated headers would be
  unusable. The documented C++ and CLI workflows are explicitly source-build
  workflows; packaged Python target compilation remains covered.

- retain the three provider build options for embedded and other non-test
  consumers, but require every bundled provider when `BUILD_MQT_CORE_TESTS` is
  enabled. Rationale: the full suite can register its provider integration tests
  unconditionally without removing the useful provider-free production boundary.

- use the bundled Garnet target only for the Python QDMI integration path and
  retain one small direct sparse target for constructor and typed compilation
  coverage. Rationale: the two tests now exercise distinct public contracts
  without maintaining a misleading partial IQM model or asserting a particular
  placement chosen by the mapper.

## Outcome and validation

The implementation is complete and locally validated. It adds one detached
adapter rather than another target model, one Python target type, three CLI
options, and no compatibility surface. The real integration test found and fixed
a pass-ordering bug in the merged target pipeline: generic canonicalization now
runs before native synthesis, while conformance remains the final semantic
check.

Directional and unknown operations must prove both orientations on every
supported pair; operand-symmetric gates retain one tuple per edge. The
source-build C++/CLI and packaged Python workflows were validated. A
distributable MLIR C++ SDK was left to separate packaging work. Final hosted CI
was not recorded.

## Code and ownership

`mlir/include/mlir/Compiler/Target.h` and `mlir/lib/Compiler/Target.cpp` define
`mlir::CompilerTarget`. It can be constructed from a site count or detailed
sites, optional undirected topology, optional homogeneous operation
capabilities, and an optional duration unit. The detailed `Site`, `SiteTuple`,
and `Operation` values retain names, coherence times, ordered calibration sites,
durations, and fidelities. An absent topology means all-to-all; an absent
operation set means every operation is native.

`mlir/include/mlir/Compiler/TargetCompilation.h` and
`mlir/lib/Compiler/TargetCompilation.cpp` define the canonical compilation
sequence. `QCOProgram::compileForTarget` and the optional target accepted by
`runDefaultPipeline` both delegate to it. The bridge must call these entry
points rather than compose passes itself.

`include/fomac/FoMaC.hpp` and `src/fomac/FoMaC.cpp` define the live QDMI
wrapper. `fomac::Session` owns device discovery and returns `fomac::Device`
handles. Device, site, and operation queries provide the data needed for a
detached compiler target. The adapter is the only new library that links FoMaC
to `MQTCompilerTarget`.

`bindings/mlir/register_mlir.cpp` implements the `mqt.core.mlir` nanobind
extension. `python/mqt/core/mlir.pyi` is generated by the repository `stubs`
session and must not be edited by hand. The binding already owns the typed
program and `compile_program` surface.

`mlir/tools/mqt-cc/mqt-cc.cpp` implements the standalone compiler driver.
`src/qdmi/driver` owns provider discovery and the stable device registry.
`mqt_copy_qdmi_runtime` is the existing CMake helper for colocating the built-in
providers and assets with an executable.

The Garnet and Emerald configurations are installed from `json/sc/` and are
registered as `mqt.sc.iqm.garnet` and `mqt.sc.iqm.emerald`. The neutral-atom
default model is useful only to prove the adapter's explicit zone diagnostic.

This task may add the adapter header, source, library, focused tests, concise
compiler/QDMI workflow documentation, Python bindings and generated stub
updates, the separate changelog entry, and this ExecPlan. It must not
reimplement the target or pipeline, modify CoreFoMaC to depend on MLIR, add a
legacy CoreIR dependency to the adapter or CLI, or revive historical targeting
abstractions.

## Acceptance

The existing IQM model tests retain the Garnet and Emerald size, topology, gate
set, and calibration coverage from #1992. The adapter tests must prove:

1. Garnet snapshots as 20 sites and 30 undirected edges with `r`, `cz`, and
   `measure`.
2. Reported site names, T1/T2, and fidelities survive while unavailable
   operation durations remain absent.
3. The target remains valid after the originating device and session are
   destroyed.
4. A circuit-model device without topology becomes all-to-all.
5. Site-dependent operation support fails rather than being widened.
6. One-way directional operation support fails, while both ordered orientations
   and their distinct calibration survive conversion.
7. Neutral-atom zone models fail with a precise circuit-model diagnostic.

The Python tests must prove direct construction, immutable metadata access,
`from_device`, detached lifetime, `compile_for_target`, and optional-target
`compile_program`.

A provider-disabled non-test build must configure and build the compiler adapter
and CLI. Python tests exercise the packaged extension and provider assets. C++
adapter and CLI workflows are source-build workflows; an installed MLIR SDK
needs a complete dependency export.

The rewritten PR description begins with the required AI disclosure, describes
the three user workflows and dependency boundary, lists validation, and says
`Closes #1082`. It must not close #1079 or #1133.

## Interfaces

The adapter exposes:

    namespace mlir {
    CompilerTarget compilerTargetFromDevice(const fomac::Device& device);
    }

`MQTCompilerFoMaCAdapter` publicly links `MQTCompilerTarget` and
`MQT::CoreFoMaC`. `MQTCompilerTarget` and CoreFoMaC do not gain new
dependencies.

Python exposes the same owned target concept:

    target = CompilerTarget.from_device(device)
    program.compile_for_target(target)
    compile_program(source, target=target)

The CLI exposes only:

    mqt-cc --qdmi-list-devices
    mqt-cc --qdmi-device=mqt.sc.iqm.garnet input.qasm
    mqt-cc --qdmi-config=registry.json --qdmi-device=<stable-id> input.qasm

The target is snapshotted before compilation and no compilation pass retains a
FoMaC or QDMI handle.

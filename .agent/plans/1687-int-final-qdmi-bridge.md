# Integrate QDMI devices with the MLIR compiler

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

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

The observable proof combines focused adapter tests, Python target construction
and compilation tests, and a minimal `mqt-cc` device workflow. The existing pull
request is then rewritten from current `main` as this thin integration rather
than retaining its historical merge-heavy implementation.

### Progress

- [x] (2026-08-04) Verified that PR #1999 merged and that fresh `origin/main` is
      `47a25e76087f1c44cf2c622c2b628c1b57e2f7a6`.
- [x] (2026-08-04) Created an isolated worktree from that exact revision and
      read the repository policies, AI disclosure rules, contributor guidance,
      and remediation protocol.
- [x] (2026-08-04) Refreshed PR #1687 at exact historical head
      `dd9619bd27a34ced8ed68a4ee4533cb85771f144`, inventoried all 35 review
      threads, and confirmed that only issue #1082 should close.
- [x] (2026-08-04) Inventoried the merged compiler target/pipeline, live FoMaC
      query surface, Python binding and generated-stub boundary, QDMI provider
      discovery, CLI, bundled IQM assets, CMake runtime-copy helper, tests, and
      documentation.
- [x] (2026-08-04) Implemented the narrow FoMaC-to-compiler-target adapter and
      focused lifetime, calibration, all-to-all, and zone-rejection tests.
- [x] (2026-08-04) Exposed the immutable compiler target and target compilation
      through the MLIR Python binding, regenerated the authoritative stub, and
      added focused Python behavior tests.
- [x] (2026-08-04) Added the three QDMI CLI options and minimal subprocess tests
      under the established compiler test root while reusing provider discovery,
      runtime assets, and the canonical target pipeline.
- [x] (2026-08-04) Added concise workflow documentation and a separate #1687
      changelog entry credited to Matthias Reumann and Lukas Burgholzer.
- [x] (2026-08-04) Built the adapter, compiler tests, bindings, and CLI; passed
      all 221 compiler tests, 29 focused Python tests, eight adapter/CLI CTests,
      a provider-disabled compiler build, authoritative stub generation, strict
      warning-free documentation, changed-source clang-tidy, repository lint,
      and `git diff --check`.
- [x] (2026-08-04) Committed the initial implementation and completed two
      independent exact-head reviews. Both identified the same ordered-site
      widening bug; the adapter now rejects one-way directional operations,
      preserves two-way ordered calibration, and retains IQM's symmetric CZ
      convention. All 223 compiler tests, ten focused adapter/CLI CTests,
      changed-source clang-tidy, repository lint, and `git diff --check` pass.
- [ ] Commit the reviewed ordered-site fix and complete a fresh independent
      exact-head verification with no material findings.
- [ ] Rewrite the existing PR branch with an exact force-with-lease, replace the
      obsolete PR description, verify the replacement head, and monitor CI.

### Surprises & Discoveries

- Observation: the live compiler already owns every semantic and pipeline
  abstraction needed by this slice. The historical PR's `fomac::Target`,
  targeting pass, mapper augmentation, native-gate menu, and duplicated pipeline
  are obsolete and must not be ported.
- Observation: `CompilerTarget::Operation` deliberately models homogeneous
  target-wide support, while QDMI can report a restricted site list. The adapter
  must therefore verify that a one-qubit operation covers every site and that a
  two-qubit operation covers every topology edge or all-to-all pair before
  treating it as target-wide. Ordered site tuples then carry calibration
  overrides only.
- Observation: the bundled IQM Garnet and Emerald models, stable registry IDs,
  runtime assets, site names, T1/T2 values, and fidelities are already present
  on `main`. This slice consumes those models rather than adding fixtures or
  provenance text.
- Observation: `mqt_copy_qdmi_runtime` already copies built-in provider
  libraries, registry manifests, and assets beside an executable. The CLI does
  not need another plugin loader or packaging mechanism.
- Observation: the old PR description and most unresolved threads refer to
  deleted predecessor abstractions. The final branch should satisfy the
  remaining behavior through the merged prerequisite PRs and this adapter, then
  describe only the actual user workflow.
- Observation: `mqt-cc` is not currently shipped as part of the Python wheel.
  This slice keeps that packaging boundary unchanged; Python target compilation
  is provided directly by the extension and packaged QDMI providers.
- Observation: MQT Core does not currently export or install the MLIR compiler
  libraries and generated headers as a consumable SDK. Exporting the adapter
  would require exporting the compiler pipeline, dialect libraries, generated
  headers, and their MLIR dependency closure. This slice therefore documents the
  C++ and `mqt-cc` workflows as source-build interfaces and leaves a coherent
  MLIR SDK/package boundary to a dedicated follow-up.
- Observation: the previously merged target pipeline ran the generic QCO cleanup
  after target-native synthesis and conformance. Its canonicalization patterns
  can rewrite `qco.r` operations with special angles back to `qco.rx` or
  `qco.ry`, making a formerly conforming Garnet result non-native. A real Garnet
  compilation exposed this issue; unit tests that stopped at conformance did
  not.
- Observation: a clean build with all three built-in QDMI providers disabled
  exposed that provider-backed test sources and runtime copying must be
  conditional. The compiler test target now builds without provider libraries,
  while normal CI retains full live-device coverage.
- Observation: QDMI operation site tuples are ordered, while the compiler
  deliberately models an undirected topology and homogeneous bidirectional gate
  support. Canonicalizing a one-way two-qubit site list would silently widen the
  device contract. The adapter must require both orientations for directional
  and unknown operations while allowing proven operand-symmetric gates such as
  CZ to report each edge once. Missing two-qubit site information is likewise
  insufficient when a device reports an explicit topology.

### Decision Log

- Decision: add one public adapter function,
  `mlir::compilerTargetFromDevice(const fomac::Device&)`, in a small library
  that links `MQTCompilerTarget` and `MQT::CoreFoMaC`. Rationale: dependency
  direction stays acyclic and compiler semantics remain owned by MLIR while
  callers opt into the live-device bridge. Date/Author: 2026-08-04, GPT-5.6 via
  Codex.
- Decision: snapshot all QDMI data eagerly and return a detached
  `CompilerTarget`. Rationale: compiler execution must not depend on a live QDMI
  session or provider handle, and the target already has shared immutable
  storage for cheap copies. Date/Author: 2026-08-04, GPT-5.6 via Codex.
- Decision: reject any device site that is a zone and any zoned operation.
  Rationale: circuit-model topology and neutral-atom zones have different
  semantics; silently flattening zones into qubits would create an invalid
  target. Date/Author: 2026-08-04, GPT-5.6 via Codex.
- Decision: reject explicit QDMI operation site lists that are not homogeneous
  over the compiler target. Rationale: the compiler target intentionally
  represents one target-wide gate set; silently widening a restricted QDMI
  operation would allow synthesis to emit an unsupported gate. Date/Author:
  2026-08-04, GPT-5.6 via Codex.
- Decision: accept one reported orientation only for a conservative set of
  operand-swap-invariant gates and require both ordered tuples for every site
  pair otherwise. Rationale: this preserves IQM's symmetric CZ data without
  misrepresenting directional or unknown QDMI operations as bidirectional.
  Date/Author: 2026-08-04, GPT-5.6 via Codex.
- Decision: expose `CompilerTarget.from_device(device)` from `mqt.core.mlir`,
  not `Device.target()` from `mqt.core.fomac`. Rationale: CoreFoMaC and its
  binding remain independent of MLIR, target ownership is visible in the
  compiler namespace, and compilation APIs accept one explicit target type.
  Date/Author: 2026-08-04, GPT-5.6 via Codex.
- Decision: add `QCOProgram.compile_for_target` and an optional `target` to the
  sole `compile_program` function. Rationale: Python mirrors the two canonical
  C++ entry points and does not accept devices, coupling maps, native-gate
  strings, or compatibility shims. Date/Author: 2026-08-04, GPT-5.6 via Codex.
- Decision: add only `--qdmi-list-devices`, `--qdmi-device`, and `--qdmi-config`
  to `mqt-cc`. Configure the registry before the first driver access, snapshot
  the selected device, and invoke the existing canonical target pipeline.
  Rationale: this is the irreducible user surface and preserves current provider
  discovery rather than introducing another dynamic boundary. Date/Author:
  2026-08-04, GPT-5.6 via Codex.
- Decision: reject target compilation when the requested output or custom pass
  sequence cannot preserve or safely compose the target assignment, using the
  validation already centralized in `runDefaultPipeline`. Rationale: options
  must not be silently ignored and the CLI must not replicate pipeline rules.
  Date/Author: 2026-08-04, GPT-5.6 via Codex.
- Decision: keep four CLI subprocess tests under
  `mlir/unittests/Compiler/Inputs`, reuse the release-bundled IQM assets, and
  make `mqt-cc` a test-only dependency when the SC provider is enabled.
  Rationale: this respects the established test layout and proves only option
  parsing, registry selection, and one compilation workflow without promoting
  the optional LLVM tool into every default build. Date/Author: 2026-08-04,
  GPT-5.6 via Codex.
- Decision: perform generic QCO cleanup before target-native synthesis, retain
  only CSE and dead-value cleanup afterward, and run conformance last.
  Rationale: no target-independent canonicalizer may reintroduce a gate outside
  the native operation set after it has been synthesized and verified.
  Date/Author: 2026-08-04, GPT-5.6 via Codex.
- Decision: do not add an upgrade-guide entry, compatibility alias, wheel
  shipment for `mqt-cc`, IQM data attribution, or a second changelog reference
  in prerequisite entries. Rationale: the compiler collection is unreleased and
  the user requested a compact final integration entry credited only to Matthias
  Reumann and Lukas Burgholzer. Date/Author: 2026-08-04, GPT-5.6 via Codex.
- Decision: do not add a partial install/export path for only the adapter.
  Rationale: the repository has no installed MLIR SDK boundary, and exporting a
  single facade while omitting the pipeline, dialects, and generated headers
  would be unusable. The documented C++ and CLI workflows are explicitly
  source-build workflows; packaged Python target compilation remains covered.
  Date/Author: 2026-08-04, GPT-5.6 via Codex.

### Outcomes & Retrospective

The implementation is complete and locally validated. It adds one detached
adapter rather than another target model, one Python target type, three CLI
options, four irreducible subprocess checks, and no compatibility surface. The
real integration test found and fixed a pass-ordering bug in the merged target
pipeline: generic canonicalization now runs before native synthesis, while
conformance remains the final semantic check.

Independent review additionally found and corrected one ordered-QDMI-site
widening bug at the adapter boundary. Directional and unknown operations now
prove both orientations on every supported pair; operand-symmetric gates retain
their compact one-tuple-per-edge representation. The source-build C++ and CLI
workflows and packaged Python workflow are proven. A distributable MLIR C++ SDK
remains a separate packaging concern because the current repository does not
export the compiler dialects, generated headers, or pipeline dependency closure.
Fresh exact-head verification and publication are still pending.

### Context and Orientation

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

This task may add the adapter header, source, library, focused tests, minimal
CLI tests under the compiler test root, concise compiler/QDMI workflow
documentation, Python bindings and generated stub updates, the separate
changelog entry, and this ExecPlan. It must not reimplement the target or
pipeline, modify CoreFoMaC to depend on MLIR, add a legacy CoreIR dependency to
the adapter or CLI, or revive historical targeting abstractions.

### Plan of Work

First add the adapter header and source under the compiler subtree. Query the
device name, sites, duration unit, topology, and operations once. Convert site
indices after checking that they fit the nonnegative i64 target domain. Reject
zones, preserve optional site metadata, convert the optional undirected coupling
map, and snapshot homogeneous operation capabilities plus ordered site-tuple
calibration. Preserve reported duration units, default an omitted scale factor
to one, and reject a scale factor without a unit. Let `CompilerTarget` perform
cross-object validation and canonicalization.

Add focused C++ tests for a detached bundled IQM target, counts and calibration,
missing topology as all-to-all on a circuit device, rejection of restricted
operation support, and the neutral-atom zone diagnostic. Build the adapter as a
distinct library with only public dependencies on FoMaC and `MQTCompilerTarget`;
copy the QDMI runtime only to tests or executables that need live provider
discovery.

Next bind the compiler target value types and immutable properties in
`mqt.core.mlir`. Provide direct site-count and detailed-site constructors,
`CompilerTarget.from_device`, operation support queries,
`QCOProgram.compile_for_target`, and `compile_program(..., target=None)`.
Regenerate the stub and add tests that construct a target directly, snapshot
Garnet, destroy the session/device, inspect names and calibration, and compile a
small program through the canonical pipeline.

Then extend `mqt-cc` with the three QDMI options. Apply an explicit registry
configuration before initializing the driver, list stable identifiers without
opening devices, open only a selected device, snapshot it through the adapter,
and pass it to `runDefaultPipeline`. Keep option validation compact and rely on
the compiler API for target/output and target/custom-pipeline diagnostics. Link
the tool to the adapter and copy the existing runtime beside it. Add a tiny
input under `mlir/unittests/Compiler/Inputs` and only the irreducible list,
unknown-ID, explicit-config, and Garnet compilation checks.

Finally document direct C++, Python, and CLI workflows without design history.
Refer qubit-reuse users to `mqt-qubit-reuse`, link to existing QDMI registry and
IQM model documentation instead of duplicating it, and state only that
unavailable durations are absent. Add the separate changelog entry for #1687
with the requested two authors and no upgrade note.

### Milestones

The first milestone produces the detached bridge. Add
`mlir/include/mlir/Compiler/FoMaCAdapter.h` and
`mlir/lib/Compiler/FoMaCAdapter.cpp`, then build the compiler unit-test target.
At the end, an IQM device can be destroyed immediately after conversion while
the returned target still exposes its name, topology, coherence values, gate
set, and fidelity data. A restricted SC operation and a neutral-atom zone model
both fail with precise diagnostics.

The second milestone exposes the same owned value through Python and the
command-line driver. Regenerate `python/mqt/core/mlir.pyi`, run
`test/python/test_mlir.py`, and execute the four `mqt-cc` CTests. At the end,
Python can construct or snapshot a target and compile for it, while a
source-build `mqt-cc` can list devices, apply an explicit registry
configuration, reject an unknown ID, and compile the Bell program for Garnet.

The third milestone proves cohesion and publication readiness. Generate the MLIR
reference documentation, run strict Sphinx documentation, changed-source
clang-tidy, complete relevant C++ suites, provider-disabled configuration, stub
generation, and repository lint. An independent exact-head `mqt-pr-review` must
find no material correctness, bloat, packaging, or documentation issue before
the historical PR branch is replaced.

### Concrete Steps

Run all commands from the repository root of the isolated task worktree.

Configure a task-local release build with the repository wrapper:

    MLIR_DIR=<path-to-MLIR-22.1>/lib/cmake/mlir \
      .agent/run.sh cmake --preset release

Build the adapter, compiler tests, MLIR Python extension, and CLI:

    .agent/run.sh cmake --build build/release --target \
      MQTCompilerFoMaCAdapter mqt-core-mlir-unittests-compiler \
      mqt-core-mlir-bindings mqt-cc -j 8

Run focused C++ and CLI CTest selections:

    .agent/run.sh ctest --test-dir build/release \
      --output-on-failure -R 'CompilerFoMaCAdapter|mqt-cc'

Run focused Python tests, regenerate the authoritative stub, and build strict
documentation:

    .agent/run.sh uv run pytest test/python/test_mlir.py -q
    .agent/run.sh uvx nox -s stubs
    .agent/run.sh uvx nox -s docs

Run changed-source clang-tidy using `build/release/compile_commands.json`, then
end with the repository-required lint and diff checks:

    git diff --check
    .agent/run.sh uvx nox -s lint

Use changed-source clang-tidy and the relevant complete compiler/QDMI suites in
proportion to the final touch set. Record exact test counts and any environment
boundaries in this plan.

### Validation and Acceptance

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

The CLI tests must prove listing devices, unknown identifiers, explicit registry
configuration, and one successful Garnet compilation. They need not duplicate
the adapter, mapping, synthesis, conformance, or full compiler suites.

The final revision must build all touched targets, pass the focused and relevant
complete tests, regenerate stubs without an uncommitted delta, pass strict
documentation, changed-source clang-tidy, repository lint, `git diff --check`,
and an independent exact-head `mqt-pr-review`. A provider-disabled build must
configure and build the compiler test target without expecting unavailable
runtime libraries. The packaged Python extension and QDMI provider assets must
be exercised by the Python test session. The source-build-only C++ adapter and
CLI must be labeled as such; a full installed MLIR SDK consumer is deliberately
outside this thin bridge because the repository does not yet expose that package
boundary. C++ patch coverage must be at least 90 percent in CI.

Before publication, refresh `origin/main`, the remote #1687 head, review
threads, and PR metadata. Replace the historical branch only with:

    git push origin \
      --force-with-lease=refs/heads/feat/arch-option-and-qdmi:<verified-old-head> \
      HEAD:refs/heads/feat/arch-option-and-qdmi

The rewritten PR description begins with the required AI disclosure, describes
the three user workflows and dependency boundary, lists validation, and says
`Closes #1082`. It must not close #1079 or #1133.

### Idempotence and Recovery

All source edits are ordinary patches in the isolated worktree. Re-running
configuration, builds, tests, stub generation, and lint is safe. The adapter is
deterministic because it snapshots immutable query results into value types.

If a build or test reveals a provider-discovery path issue, inspect the
executable-local registry and assets produced by `mqt_copy_qdmi_runtime` before
changing code. Do not add a second search path or dynamic loader to mask a
configuration error.

If the remote PR head changes before publication, stop instead of force-pushing.
Fetch and compare the new revision, incorporate authorized changes deliberately,
then use a newly verified exact lease. Never use an unqualified force push.

### Artifacts and Notes

The historical PR head `dd9619bd27a34ced8ed68a4ee4533cb85771f144` is evidence
only. No historical commit or implementation file should be cherry-picked. The
useful prerequisite behavior is already merged through PRs `#1992`, `#1993`,
`#1997`, `#1998`, and `#1999`.

The unresolved historical review threads map either to those merged
prerequisites or to this slice's adapter, bindings, CLI, and workflow
documentation. Thread resolution is considered only after the replacement head
contains and verifies the requested behavior.

### Interfaces and Dependencies

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

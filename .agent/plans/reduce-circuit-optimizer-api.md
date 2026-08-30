# Remove CircuitOptimizer and move generic transformations to CoreIR

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core 3 installed `qc::CircuitOptimizer` as a separate library even though
its transformations belonged to different parts of the toolkit. MQT Core 4
removes that class and the `MQT::CoreCircuitOptimizer` package target. The two
generic circuit operations become `QuantumComputation` member functions in
`MQT::CoreIR`, while equivalence-checking and mapping transformations live in
MQT QCEC and MQT QMAP, respectively.

After this change, Core users have one fewer library to distribute and link.
They can verify the migration by calling `qc.flattenOperations()` and
`qc.removeFinalMeasurements()`, and by inspecting an installed Core package: the
member declarations are present, while no optimizer header, library, generated
export header, or CMake target remains.

## Progress

- [x] (2026-08-26 13:58Z) Audited every repository in the Munich Quantum Toolkit
      organization and assigned every production `CircuitOptimizer` use to Core,
      QCEC, QMAP, QuSAT, DDSIM, or Debugger.
- [x] (2026-08-26 15:12Z) Moved the six QCEC-only transformations and their 44
      migrated tests into QCEC, and moved QuSAT's public dependency-graph use
      into a private QuSAT implementation. These prerequisites merged as QCEC
      #1040 and QuSAT #512.
- [x] (2026-08-27 11:07Z) Added `QuantumComputation::flattenOperations(bool)`
      and `QuantumComputation::removeFinalMeasurements()` to `MQT::CoreIR`,
      moved their tests into `test/ir/`, and updated Core's Quantum Device
      Management Interface (QDMI) decision-diagram device.
- [x] (2026-08-27 11:07Z) Removed the optimizer source, test directory, tracked
      header, CMake target, generated export header installation, and
      Python-wheel dependency from Core.
- [x] (2026-08-27 13:30Z) Completed a semantic parity audit and hardened QCEC's
      measurement deferral, reset elimination, and gate fusion, plus QMAP's
      fusion and SWAP decomposition, with focused regression tests in separate
      follow-up commits.
- [x] (2026-08-27 13:50Z) Opened the Core-v4-dependent migrations as drafts:
  QCEC #1042, QMAP #1125, QuSAT #514, DDSIM #977, Debugger #438, and the
  documentation-only SyReC cleanup #700.
- [x] (2026-08-27 14:31Z) Merged current `main` at `99fd4d2ef` and applied its
  new conventions: `///` public Doxygen comments, unqualified fixed-width
  integer types, and the repository C++ lint workflow.
- [x] (2026-08-27 14:31Z) Revalidated Core: 298/298 CoreIR tests and 51/51 QDMI
      decision-diagram device tests pass; static and shared installs contain no
      optimizer artifact; release, static, and shared builds are warning-clean;
      and a changed-file Clang-Tidy 21.1.1 scan reports no findings.
- [x] (2026-08-27 14:31Z) Revalidated the downstream changes: QCEC passes 52/52
      focused and 594/594 full tests; QMAP passes 47/47 focused and all 851
      built tests, with seven tests not built because unrelated Core-v4
      components are absent; QuSAT passes 11/11; DDSIM passes 28/28 affected and
      116/117 full tests, with one unrelated stochastic threshold miss; Debugger
      passes 149/149; and SyReC passes lint.
- [x] (2026-08-27 14:50Z) Rewrote this plan around the final removal, current
  downstream scope, semantic hardening, and current validation evidence, and
  passed the complete `uvx nox -s lint` suite.
- [ ] Run the exact `uvx nox -s cpp-lint` session with Clang-Tidy 22 locally or
      in CI after the next push. The available local Clang-Tidy is 21.1.1; its
      clean supplemental result does not satisfy this check.
- [ ] After pushing the final follow-up, replace the pre-hardening validation
  counts in the Core #2262, QCEC #1042, and QMAP #1125 descriptions with the
  final results in this plan.
- [ ] After Core v4 is released, update the Core dependency in each downstream
      draft, change QMAP #1125 and Debugger #438 to target `main` after their
      prerequisite pull requests merge, rerun validation, and only then mark
      those drafts ready for review.

## Surprises & Discoveries

- Observation: the final ownership boundary is wider than a Core-only edit but
  smaller than the organization as a whole. Core, QCEC, QMAP, QuSAT, DDSIM, and
  Debugger contain functional migrations; SyReC contained only a stale comment.
  Evidence: an organization-wide symbol search and the pull requests named in
  `Progress` found no additional production consumer.
- Observation: a mechanical QCEC port was insufficient for malformed or nested
  circuits. Measurement deferral needs a bijective preflight mapping and must
  reject ambiguous comparisons and unsupported non-unitary operations. Reset
  elimination must recurse through compounds, preserve wrapper controls, use
  fresh sparse physical indices, and reject conditional resets and circuits that
  already contain ancillary qubits. Evidence: the focused QCEC suite grew to 52
  passing tests after these cases were added.
- Observation: QMAP must not rewrite every operation whose type is `SWAP`.
  Controlled and negative-controlled SWAPs have different semantics, while
  nested uncontrolled standard SWAPs still need recursive decomposition.
  Evidence: structural and decision-diagram equivalence regressions pass for
  both preserved and decomposed cases.
- Observation: single-qubit fusion must preserve zero-target operations such as
  global phase. QCEC and QMAP now test this locally, and QCEC also carries the
  four decision-diagram equivalence cases formerly in Core.
- Observation: the downstream repositories still require Core 3.9.x and some use
  other APIs removed independently for Core v4. Their migration pull requests
  must remain drafts until the release. QMAP #1125 currently targets the branch
  from #1111, and Debugger #438 targets the branch from #436; each must target
  `main` after its prerequisite merges.
- Observation: QMAP's only compiler warning during strict validation is a
  vendored GoogleTest `char8_t` conversion warning that reproduces on the parent
  revision. No changed QMAP source emits a warning.
- Observation: `mqt_core_circuit_optimizer_export.h` was generated by the
  removed CMake target rather than tracked in the repository. The migration
  stops generating and installing it; there is no source file to delete.
- Observation: the exact Core C++ lint session requires Clang-Tidy 22. A
  warning-clean release build and a zero-finding Clang-Tidy 21.1.1 scan provide
  useful evidence but do not replace the required session.

## Decision Log

- Decision: remove `qc::CircuitOptimizer` and its library completely. The
  earlier reduced three-method target was a prototype and is superseded.
  Rationale: two generic operations naturally belong to the circuit type, and
  distributing a separate library for them creates avoidable packaging work.
  Date/Author: 2026-08-27, Codex.
- Decision: expose `flattenOperations(bool)` and `removeFinalMeasurements()` as
  `QuantumComputation` members in `MQT::CoreIR`. Rationale: they mutate one
  circuit and are used across several packages, so the owning IR type is the
  smallest shared interface. Date/Author: 2026-08-27, Codex.
- Decision: keep independent single-qubit fusion implementations in QCEC and
  QMAP. Rationale: they are the only production owners and already have the
  dependency and identity-cleanup helpers required by the transformation; a new
  shared optimizer abstraction would recreate the packaging problem.
  Date/Author: 2026-08-27, Codex.
- Decision: QCEC owns `swapReconstruction`, `removeDiagonalGatesBeforeMeasure`,
  `eliminateResets`, `deferMeasurements`, `backpropagateOutputPermutation`, and
  `elidePermutations`. Rationale: QCEC is their only production consumer and
  owns their equivalence-checking contracts. Date/Author: 2026-08-26, Codex.
- Decision: QCEC rejects unsupported measurement and reset shapes before
  mutation. Rationale: partial mutation produces an invalid circuit and hides
  caller errors; explicit preconditions make the migrated behavior safe and
  testable. Date/Author: 2026-08-27, Codex.
- Decision: QMAP owns `decomposeSWAP`, `cancelCNOTs`, `replaceMCXWithMCZ`, and
  its fusion implementation. Only uncontrolled standard SWAPs are decomposed.
  Rationale: QMAP is their production owner, and preserving controlled forms
  avoids a semantic change. Date/Author: 2026-08-27, Codex.
- Decision: QMAP and QuSAT construct their small dependency graphs locally.
  Rationale: the graph stores pointers into a mutable circuit, has only two
  consumers with different uses, and is not a stable transformation API.
  Date/Author: 2026-08-26, Codex.
- Decision: remove `collectBlocks`, `collectCliffordBlocks`, the public generic
  removal helpers, and unused graph aliases without replacement. Rationale: the
  production census found no owner. Date/Author: 2026-08-26, Codex.
- Decision: keep semantic hardening in follow-up commits separate from the
  mechanical moves. Rationale: reviewers can compare the port independently from
  deliberate behavior fixes. Date/Author: 2026-08-27, Codex.
- Decision: keep all Core-v4-dependent downstream pull requests in draft.
  Rationale: their released dependencies still provide Core 3.9.x, and some full
  configurations depend on other independent Core-v4 changes. Date/Author:
  2026-08-27, Codex.

## Outcomes & Retrospective

The final Core diff deletes 4,824 lines, including the standalone optimizer
distribution unit, without adding a dependency or cross-package abstraction.
Core retains the two generally useful transformations as ordinary
`QuantumComputation` members. Every production consumer either uses those
members or owns its domain-specific implementation, and the upgrade guide names
a replacement or explicitly says that none exists for every removed public
symbol.

Core's behavioral, package, lint, and warning checks pass with the one explicit
exception that the mandated Clang-Tidy 22 session cannot run on the available
host. The independent Clang-Tidy 21.1.1 scan is clean but remains supplemental.
Downstream semantic checks pass. DDSIM's sole full-suite miss is the unrelated
stochastic `StochNoiseSimTest.CheckQubitOrder` threshold check, which produced
943 successes against a minimum of 950 and also fails on the parent revision
without this migration. QMAP's seven `NOT_BUILT` entries are configuration
omissions, not executed failures.

The principal lesson is that moving code and moving its contract are different
tasks. The semantic parity audit found unsafe measurement/reset edge cases and
SWAP control semantics that compilation and direct test copying did not expose.
Keeping those fixes in separate commits made the final ownership change easier
to review.

## Context and Orientation

`include/mqt-core/ir/QuantumComputation.hpp` declares Core's circuit type.
`src/ir/CircuitOptimization.cpp` now implements its two generic in-place
transformations, and `src/ir/CMakeLists.txt` includes that source in
`MQT::CoreIR`. `test/ir/test_flatten_operations.cpp` and
`test/ir/test_remove_final_measurements.cpp` cover recursive compounds,
custom-gate-only flattening, final measurements and barriers, and empty
circuits. `src/qdmi/devices/dd/Device.cpp` uses the member API before building a
decision diagram, a graph-based representation of a quantum state or operator.
Single-qubit gate fusion combines adjacent operations on one qubit into one
equivalent operation. A dependency graph here is a vector indexed by qubit; each
entry lists pointers to the circuit operations that use that qubit.

Before this change, `include/mqt-core/circuit_optimizer/CircuitOptimizer.hpp`,
`src/circuit_optimizer/`, and `test/circuit_optimizer/` defined the public
class, library, and tests. Those paths are deleted. CMake also no longer adds or
exports `MQT::CoreCircuitOptimizer`, and no longer generates or installs
`circuit_optimizer/mqt_core_circuit_optimizer_export.h`.

The functional coordination boundary is five open downstream drafts plus two
already-merged prerequisites. QCEC #1040 owns six equivalence-checking
transformations, and QuSAT #512 owns its dependency graph. QCEC #1042 adds
fusion and member calls; QMAP #1125 owns mapping transformations, fusion, and
its graph; QuSAT #514 updates flattening tests; DDSIM #977 and Debugger #438 use
the new members. SyReC #700 is a separate comment-only cleanup discovered by the
census.

The checked-in `CHANGELOG.md` summarizes the breaking change. `UPGRADING.md`
gives call-site replacements, downstream owners, removed names, link targets,
and installed-header changes. The ExecPlan coordinates work but does not
authorize GitHub mutations. Under `docs/ai_usage.md`, a human must review the
result, and any agent-authored public body must begin with the required visible
disclosure `🤖 *AI text below* 🤖`.

## Milestones

### Milestone 1: establish ownership

Search every official MQT repository for the class, its CMake target, its
installed and generated headers, and each public method. Classify production,
test, build, packaging, and comment-only references. The milestone is complete
when every live reference has a named owner or an explicit no-replacement
decision. The completed census identified the repositories and boundaries
described in `Context and Orientation`.

### Milestone 2: remove the Core distribution unit

Add the two member declarations to `include/mqt-core/ir/QuantumComputation.hpp`,
move their implementations to `src/ir/CircuitOptimization.cpp`, move their
focused tests under `test/ir/`, and update `src/qdmi/devices/dd/Device.cpp`.
Then remove the optimizer source, test tree, header, CMake subdirectories,
exported target, generated export header installation, and wheel rules. This
milestone is complete when the full CoreIR and QDMI device tests pass and
installed packages expose only `MQT::CoreIR` for these calls.

### Milestone 3: port and harden downstream behavior

Move QCEC-only and QMAP-only transformations to their existing optimizer or
circuit-optimization components, and build private dependency graphs in QMAP and
QuSAT. Port structural tests before changing behavior. In separate follow-up
commits, add preflight validation and recursive handling in QCEC, preserve
controlled SWAP semantics in QMAP, and add structural plus decision-diagram
equivalence coverage for fusion. This milestone is complete when the focused and
full results in `Progress` pass without changed-source warnings.

### Milestone 4: prove package and ecosystem compatibility

Build warning-clean static and shared Core packages, install each one, and link
a minimal external `MQT::CoreIR` consumer that calls both members. Confirm that
the removed target and headers are absent. Test QuSAT, DDSIM, and Debugger call
sites as well as the QCEC and QMAP implementations. Record unrelated blockers as
such rather than expanding this issue into every Core-v4 migration. This
milestone is complete; the concise evidence appears in `Artifacts and Notes`.

### Milestone 5: synchronize current policy and hand off the release

Merge current `main`, resolve documentation conflicts, apply its public-comment
and fixed-width-type rules, and run the repository's new C++ lint session before
the ordinary full lint. All edits and supplemental checks are complete. An exact
Clang-Tidy 22 result remains the only validation gap around the next push. After
Core v4 is released, downstream maintainers update their dependency versions,
revalidate, and make the drafts ready. The public validation summaries must also
be refreshed after the final follow-up is pushed.

## Plan of Work

Implement the Core member API first so downstream code has a stable replacement.
Keep the moved helper functions private to `src/ir/CircuitOptimization.cpp` and
preserve the original operation ordering and compound-operation behavior. Move
only the two applicable test files into `test/ir/`; domain-specific tests belong
with their new downstream implementations.

Remove the optimizer build and package surface in one pass through the root,
`src/`, `test/`, QDMI-device, and Python packaging CMake files. Search for the
class, target, path, and export-header stem afterward. Update `CHANGELOG.md` and
`UPGRADING.md` based on the final installed interface, not an intermediate
reduced target.

In QCEC, keep the six equivalence transformations and fusion in
`src/optimizer/EquivalenceCheckingOptimizer.cpp`. Validate a complete logical to
physical-qubit mapping before deferring measurements, recurse through compound
operations, and reject unsafe reset forms before mutation. In QMAP, put mapping
transformations and fusion in `src/datastructures/CircuitOptimizations.cpp`,
recurse into nested compounds, and decompose only uncontrolled standard SWAPs.
Keep QuSAT's graph local to its SAT encoder. Convert DDSIM, Debugger, and QuSAT
call sites to member syntax.

Finally, validate Core behavior, both package modes, every downstream owner,
warnings, C++ lint, and full repository lint. Keep generated build directories
out of commits. Keep release-dependent downstream changes as drafts until their
Core dependencies can move to v4.

## Concrete Steps

From the Core repository root, configure a warning-as-error release build, build
the affected test programs, and run them with the IR fixture directory as the
working directory:

    cmake --preset release -DWARNINGS_AS_ERRORS=ON
    cmake --build --preset release --target mqt-core-ir-test \
      mqt-core-qdmi-ddsim-device-test
    cmake -E chdir test/ir \
      ../../build/release/test/ir/mqt-core-ir-test --gtest_brief=1
    ./build/release/test/qdmi/devices/dd/mqt-core-qdmi-ddsim-device-test \
      --gtest_brief=1

The two programs must report 298 and 51 passing tests, respectively. Running the
IR program from the repository root is not equivalent because two importer tests
resolve `../circuits/test.qasm` relative to `test/ir/`.

Build and install both library modes from the Core repository root:

    cmake --preset release -B build/package-static \
      -DBUILD_MQT_CORE_SHARED_LIBS=OFF -DBUILD_MQT_CORE_TESTS=OFF
    cmake --build build/package-static
    cmake --install build/package-static \
      --prefix build/package-static/install
    cmake --preset release -B build/package-shared \
      -DBUILD_MQT_CORE_SHARED_LIBS=ON -DBUILD_MQT_CORE_TESTS=OFF
    cmake --build build/package-shared
    cmake --install build/package-shared \
      --prefix build/package-shared/install
    find build/package-static/install build/package-shared/install \
      -iname '*circuit*optimizer*' -print
    rg 'CoreCircuitOptimizer|circuit_optimizer' \
      build/package-static/install build/package-shared/install

The `find` command must print nothing. The `rg` command checks text package
metadata and must also find nothing, so it exits with status 1. A minimal
external consumer must include `ir/QuantumComputation.hpp`, link `MQT::CoreIR`,
construct an empty `qc::QuantumComputation`, call both members, and exit
successfully when configured once against each install prefix. Create these
untracked files under `build/package-consumer/`:

    # CMakeLists.txt
    cmake_minimum_required(VERSION 3.24)
    project(core_ir_consumer LANGUAGES CXX)
    find_package(mqt-core CONFIG REQUIRED)
    add_executable(core_ir_consumer main.cpp)
    target_link_libraries(core_ir_consumer PRIVATE MQT::CoreIR)

    // main.cpp
    #include "ir/QuantumComputation.hpp"

    int main() {
      qc::QuantumComputation circuit;
      circuit.flattenOperations();
      circuit.removeFinalMeasurements();
      return circuit.empty() ? 0 : 1;
    }

Configure, build, and run that consumer against each install:

    cmake -S build/package-consumer -B build/consumer-static \
      -DCMAKE_PREFIX_PATH="$PWD/build/package-static/install"
    cmake --build build/consumer-static
    ./build/consumer-static/core_ir_consumer
    cmake -S build/package-consumer -B build/consumer-shared \
      -DCMAKE_PREFIX_PATH="$PWD/build/package-shared/install"
    cmake --build build/consumer-shared
    ./build/consumer-shared/core_ir_consumer

For each functional downstream draft, use a clean checkout of that draft. From
its repository root, point FetchContent, CMake's source-dependency mechanism, at
the Core repository under test and run the standard release workflow:

    cmake --preset release \
      -DFETCHCONTENT_SOURCE_DIR_MQT-CORE='<Core repository root>' \
      -DWARNINGS_AS_ERRORS=ON
    cmake --build --preset release
    ctest --preset release
    uvx nox -s lint

Apply this workflow to QCEC #1042, QMAP #1125, QuSAT #514, DDSIM #977, and
Debugger #438. Test the QMAP draft with the changes from #1111 included and the
Debugger draft with the changes from #436 included. If configuration reaches a
separately removed Core-v4 target such as `MQT::CoreAlgorithms` or
`MQT::CoreNA`, use a checkout that contains that independent migration and
record omitted tests as `NOT_BUILT`; do not restore an optimizer artifact or add
that unrelated migration to this change. For SyReC #700, which changes only a
comment, run `uvx nox -s lint`.

From the Core repository root, confirm that no live source or build reference
remains outside the migration documents:

    rg -n 'CircuitOptimizer|CoreCircuitOptimizer|circuit_optimizer' \
      --glob '!build/**' --glob '!CHANGELOG.md' --glob '!UPGRADING.md' \
      --glob '!.agent/plans/**' .

This command must produce no output and exit with status 1.

After all tests, remain at the Core repository root, run the C++ lint session
with Clang-Tidy 22, and finish with the full lint as required by
`.agent/PLANS.md`:

    uvx nox -s cpp-lint
    uvx nox -s lint

The C++ lint session must report zero findings. If it stops with
`clang-tidy 22 is required`, the check remains incomplete; an older clean scan
must not be recorded as a substitute. The final lint command must pass every
hook.

## Validation and Acceptance

Calling `flattenOperations()` on an empty circuit is a no-op. With its default
argument it recursively replaces compound operations with their children in the
same order. With `true`, it flattens custom-gate compounds but leaves ordinary
compounds intact. Calling `removeFinalMeasurements()` on an empty circuit is
also a no-op; on populated circuits it removes only measurements and barriers
that are final on all affected qubits, including safe compound cases, without
deleting an earlier measurement followed by a quantum operation.

An installed static or shared Core package must declare both members in
`QuantumComputation.hpp` and let an external program link them through
`MQT::CoreIR`. It must not contain `CircuitOptimizer.hpp`,
`mqt_core_circuit_optimizer_export.h`, an optimizer library, or an exported
`MQT::CoreCircuitOptimizer` target. A repository search outside changelog,
upgrade-guide, and plan history must find no live Core reference to the removed
class, target, or include path.

QCEC acceptance includes repeated valid measurement mappings, rejection of
non-bijective or ambiguous mappings, recursive compounds, sparse reset layouts,
and rejection of conditional resets or pre-existing ancillary qubits without
partial mutation. QMAP acceptance includes both SWAP orientations, recursive
nested uncontrolled SWAP decomposition, preservation of controlled and
negative-controlled SWAPs, and zero-target fusion operations. Both fusion
implementations must pass structural tests; their available decision-diagram
tests must show equivalent input and output functionality.

The final evidence must match the latest counts in `Progress`, contain no
changed-source compiler warning, and distinguish non-built tests and the known
DDSIM stochastic miss from migration failures. Every downstream PR remains a
draft until it resolves against released Core v4 and is revalidated.

## Idempotence and Recovery

Source edits, searches, configurations, builds, installs, and tests are
repeatable. Use only repository-local `build/` subdirectories for generated
artifacts. If a configuration cached Core 3.9 or another Core checkout, remove
only that generated build directory and configure it again with the explicit
FetchContent source override. Do not alter another task's working tree.

The deleted optimizer files remain recoverable from version control until the
change merges. Do not recreate them to work around an unrelated Core-v4
dependency. Instead, use a checkout containing the appropriate independent
migration and keep the boundary documented. Updating a draft's Core dependency
version after the release is safe to repeat, but do not regenerate or commit
lockfiles until the released Core version is available.

## Artifacts and Notes

The final Core validation snapshot is:

    CoreIR:                    298/298 passed
    QDMI DD device:              51/51 passed
    release/static/shared builds: warning-clean
    static/shared installs:   no optimizer artifact
    Clang-Tidy 21.1.1:        0 changed-file findings (supplemental)
    Clang-Tidy 22 session:    pending; version unavailable locally

The final downstream snapshot is:

    QCEC:      52/52 focused; 594/594 full
    QMAP:      47/47 focused; 851/851 built; 7 NOT_BUILT; 0 failed
    QuSAT:     11/11 full
    DDSIM:     28/28 affected; 116/117 full; 1 unrelated stochastic miss
    Debugger: 149/149 full
    SyReC:    lint passed; comment-only change

DDSIM's excluded stochastic test observed 943 successes where 950 were required;
the same tolerance miss is unrelated to the optimizer migration. QMAP's vendored
GoogleTest warning reproduces on its parent revision and is not a warning in
changed project code.

## Interfaces and Dependencies

At completion, `qc::QuantumComputation` provides these signatures through
`MQT::CoreIR`:

    void flattenOperations(bool customGatesOnly = false);
    void removeFinalMeasurements();

Core provides no `qc::CircuitOptimizer`, `MQT::CoreCircuitOptimizer`, optimizer
header, optimizer library, or optimizer export header. QCEC keeps single-qubit
fusion private in `EquivalenceCheckingOptimizer` alongside its six equivalence
transformations. QMAP exposes its mapping transformations and
`qmap::singleQubitGateFusion(qc::QuantumComputation&)` through
`include/datastructures/CircuitOptimizations.hpp` and `MQT::QMapDS`. QMAP and
QuSAT own private dependency-graph representations rather than depending on a
new Core interface.

Revision note: 2026-08-26. Created the initial plan from the organization-wide
consumer census and the issue acceptance criteria.

Revision note: 2026-08-27. Replaced the reduced-library prototype with the final
complete removal, CoreIR member API, and coordinated downstream drafts.

Revision note: 2026-08-27. Removed the remaining superseded target and package
claims, added narrative milestones, recorded semantic hardening, refreshed all
validation counts and downstream statuses, incorporated current agent and C++
policy, and left the exact Clang-Tidy 22 session explicit as the only pre-push
validation gap.

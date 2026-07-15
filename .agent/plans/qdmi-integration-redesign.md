# Complete the QDMI integration redesign

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core v4 should expose one coherent device API instead of making users move
through a driver singleton, a public QDMI client interface, and a second wrapper
layer. After this work, applications discover side-effect-free device
definitions through `qdmi::DeviceRegistry`, open independent device sessions
through `qdmi::DeviceManager`, and interact with `qdmi::Device`, `Job`, `Site`,
and `Operation` objects in C++ or the matching `mqt.core.qdmi` Python module.
Configured native libraries are not executed during discovery. A user can see
the result by listing `DeviceManager.definitions`, opening the stable ID
`mqt.ddsim.default`, and submitting a QDMI program without using a global
session or any legacy wrapper API.

This plan's repository-relative scope is the public QDMI object model under
`include/mqt-core/qdmi/` and `src/qdmi/`, its nanobind bindings under
`bindings/qdmi/`, built-in device registration, Qiskit and neutral-atom
consumers, focused tests, and the related documentation and build metadata.
Runtime-configurable superconducting and neutral-atom device models and the
optional resource broker are separate tasks. Preserve unrelated work and never
modify another task's worktree. The implementation is published as pull request
1901 and incorporates the mainline changes through pull request 1909, including
the planning refinements from pull requests 1907 and 1908.

## Progress

- [x] (2026-07-15 11:27Z) Re-read `AGENTS.md`, `.agent/PLANS.md`, and
  `docs/ai_usage.md` in full and adopted the ExecPlan process.
- [x] (2026-07-15 11:40Z) Re-read the checkout-independent `.agent/PLANS.md`
  revision from pull request 1908 and regenerated this plan to use only
  repository-relative scope, portable commands, and stable references.
- [x] (2026-07-15 11:27Z) Analyzed merged pull request 1815, including its MLIR
      Python bindings, DDSIM QIR formats, QIR execution test, centralized stub
      patterns, estimator typing fix, and upgrade documentation.
- [x] (2026-07-15 11:27Z) Analyzed merged pull request 1907 and its repository
  instructions for AI disclosure, task plans, validation, and handoff.
- [x] (2026-07-15 11:27Z) Updated the mainline history, pruned stale remote
      references, repaired tracking, created a local recovery ref, and rebased
      all ten QDMI commits onto `a228a3dc2`.
- [x] (2026-07-15 11:44Z) Rebased the ten implementation commits and this plan
      onto the checkout-independent planning refinement at `b5cb91888`; all
      eleven commits match the prior series under `git range-diff`.
- [x] (2026-07-15 11:27Z) Resolved rebase conflicts by preserving 1815's MLIR
      and QIR behavior while applying the final QDMI module names and
      centralized `bindings/patterns.txt` patterns.
- [x] (2026-07-15 11:27Z) Implemented versioned JSON and TOML discovery,
  relocatable built-in manifests, stable device IDs, per-device session
  parameters, lazy opening, isolated bulk opening, and coherent object
  lifetimes.
- [x] (2026-07-15 11:27Z) Removed the legacy wrapper, driver singleton, public
      client interface, global session configuration, compatibility targets, and
      stale names from the final public surface.
- [x] (2026-07-15 11:27Z) Simplified the private QDMI implementation to one
  `DeviceApi` that owns a loaded library and stores exact
  `decltype(QDMI_...)` function pointers.
- [x] (2026-07-15 11:27Z) Removed MQT status and program-format enum
      redefinitions, exposed QDMI's own enum types in C++, documented every
      public Python binding locally, and regenerated `python/mqt/core/qdmi.pyi`.
- [x] (2026-07-15 11:38Z) Validated the rebased tree with focused native and
      Python tests, stub generation, and the warning-as-error documentation
      build.
- [x] (2026-07-15 11:41Z) Ran `uvx nox -s lint` after regenerating the plan;
  every hook passed.
- [x] (2026-07-15 11:44Z) Completed the final diff, spelling, changelog,
  portability, and generated-artifact audits and removed build, Nox,
  documentation, test-discovery, and test-cache output.
- [x] (2026-07-15 11:49Z) Committed the ExecPlan, force-pushed the rebased
  series with lease protection, repaired upstream tracking, and updated the
  pull request description with the required AI disclosure.
- [x] (2026-07-15 11:52Z) Diagnosed the initial service error as a real
  mergeability conflict after pull requests 1904 and 1909 advanced main,
  reviewed their scope, and rebased the complete series onto `559fde4b2`.
- [x] (2026-07-15 13:18Z) Applied the final naming, estimator,
      generated-docstring, and historical-documentation feedback, then obtained
      a fresh-context critical review and addressed all five actionable
      findings.
- [x] (2026-07-15 13:24Z) Repeated focused native, QDMI specification, Python,
      stub, warning-as-error documentation, and full lint validation after the
      fresh review.
- [x] (2026-07-15 13:25Z) Removed generated build, Nox, documentation,
      test-discovery, and Python cache artifacts while preserving
      developer-local environment and IDE state.
- [ ] Publish the reviewed iteration and update the pull request description.
- [ ] Inspect the final CI run triggered by the plan-status commit and address
  any integration regressions.

## Surprises & Discoveries

- Observation: Pull request 1815 modified the old
  `test/python/fomac/test_fomac.py` after this branch had renamed it to
  `test/python/qdmi/test_qdmi.py`. Evidence: the 1815 merge adds the
  `test_device_executes_qir_program` test to the old path, while Git's rebase
  rename detection transferred that addition to the QDMI test file.
- Observation: Pull request 1815 centralized nanobind stub rewrites in
  `bindings/patterns.txt`, whereas the original QDMI commits renamed the old
  file to `bindings/qdmi/patterns.txt`. Evidence: the first rebase conflict was
  a rename/rename conflict. The resolved central file contains both QDMI query
  overloads and MLIR `compile_program` overloads.
- Observation: Pull request 1815 changed both the Qiskit estimator and the QDMI
  redesign's nearby typing workaround. Evidence: the final resolution keeps
  `collections.abc.Mapping` and its typed cast for Pauli mappings, while the
  additional fallback cast from this pull request is unnecessary and removed.
- Observation: Pull request 1907 replaces the prior template-generated agent
  guide with a repository-owned guide and requires a living plan for this
  refactor. Evidence: the repository adds `.agent/PLANS.md` and an `ExecPlans`
  section to `AGENTS.md`.
- Observation: Pull request 1908 makes checked-in ExecPlans portable across
  clones. Evidence: `.agent/PLANS.md` now prohibits local filesystem paths,
  developer accounts, checkout locations, and ephemeral branch names.
- Observation: The sandbox does not permit ccache to create temporary files in
  its default user cache directory during a native build. Evidence: the first
  build stopped with `ccache: error: failed to create temporary file`; rerunning
  with `CCACHE_DIR` set to a writable temporary directory built all requested
  targets.
- Observation: The first post-publication pre-commit service error was caused by
  GitHub marking the pull request conflicting, not by a failed hook. Evidence:
  main advanced from `b5cb91888` to `559fde4b2` after publication; the only
  overlapping file was `CHANGELOG.md`, where both pull requests added link
  definitions.
- Observation: A manager-local weak cache did not prevent two simultaneously
  live managers from initializing the same provider library independently.
  Evidence: the fresh-context review traced each manager to its own cache;
  `loadDeviceApi` now canonicalizes paths and uses one process-wide weak cache,
  with a regression test that opens the same library through equivalent paths.
- Observation: An operation previously accepted `Site` objects from any device
  state and passed their native handles to its provider. Evidence: the
  fresh-context review found no ownership check in `Operation::siteHandles`; the
  method now rejects sites whose shared state differs from the operation's
  state, and the object-model suite exercises the failure.
- Observation: Python stub formatting omits an explicit `int` union member when
  `float` is present because Python's numeric typing accepts integers wherever
  floats are accepted. Evidence: the binding signature includes `int`, the
  generated stub simplifies it, and the Python regression test submits an
  integer custom parameter successfully through binding conversion.

## Decision Log

- Decision: Refer to the dependency simply as QDMI throughout the code and
  public documentation, with its selected version recorded only in
  `cmake/ExternalDependencies.cmake`. Remove the versioned private type name and
  configuration ABI marker. Rationale: MQT Core supports one selected QDMI
  revision at a time and will evolve the implementation when that dependency is
  updated rather than dispatching among versions. Date/Author: 2026-07-15 /
  GPT-5 via Codex, following maintainer direction.
- Decision: Keep the name `DeviceManager`. Rationale: it discovers
  application-side definitions and creates sessions, while a QDMI v2
  orchestrator has broader provider traversal and job-routing responsibilities.
  Date/Author: 2026-07-15 / GPT-5 via Codex, following maintainer direction.
- Decision: Use QDMI's `QDMI_Device_Status`, `QDMI_Job_Status`, and
  `QDMI_Program_Format` directly in the public C++ API. Rationale: redefining
  existing QDMI concepts created unnecessary conversion code and possible drift.
  Python retains ergonomic `Device.Status`, `Job.Status`, and `ProgramFormat`
  bindings of those QDMI enums. Date/Author: 2026-07-15 / GPT-5 via Codex,
  following maintainer direction.
- Decision: Store exact `decltype(QDMI_...)` pointers on the concrete private
  `DeviceApi` and keep session/job lifetime code in `DeviceState`. Rationale:
  this removes the redundant generic `DeviceApi` function table and
  virtual/pass-through layers without mixing dynamic-library ownership with
  recursive object lifetime management. Date/Author: 2026-07-15 / GPT-5 via
  Codex.
- Decision: Keep `openAll()` and `open_all()`. Rationale: discovery remains
  side-effect free, while explicit bulk opening is useful for providers and
  tests and returns errors independently by stable ID. Date/Author: 2026-07-15 /
  GPT-5 via Codex, following maintainer direction.
- Decision: Resolve the rebase toward one shared `bindings/patterns.txt`.
  Rationale: 1815 deliberately centralized nanobind rewrites; preserving that
  design avoids a new component-specific pattern file while changing only the
  obsolete wrapper match expressions to QDMI. Date/Author: 2026-07-15 / GPT-5
  via Codex.
- Decision: Preserve 1815's Python-package MLIR defaults, DDSIM QIR formats, QIR
  execution coverage, and estimator typing. Rationale: these are newly merged
  mainline capabilities and the QDMI redesign must integrate rather than regress
  them. Date/Author: 2026-07-15 / GPT-5 via Codex.
- Decision: Cache loaded provider libraries process-wide using canonical paths
  and weak ownership. Rationale: managers are independent discovery/session
  factories, but a live native provider library must be initialized only once;
  weak ownership still lets the final device or job unload it. Date/Author:
  2026-07-15 / GPT-5 via Codex, following fresh-context review.
- Decision: Treat a disabled replacement registration as removal. Rationale:
  disabled definitions are masks during layered discovery and must not remain
  openable when supplied as a runtime replacement. Date/Author: 2026-07-15 /
  GPT-5 via Codex, following fresh-context review.

## Outcomes & Retrospective

The implementation and rebase milestones are complete. The implementation now
includes the mainline work through pull request 1909. Its ten implementation
commits and publication-status commit match their prior versions exactly under
`git range-diff`; the plan commit differs only by the expected adjacent
changelog link from pull request 1904. The final feedback removed versioned QDMI
naming and the configuration ABI marker, restored the unnecessary estimator
change, completed binding-local custom-query documentation, and preserved FoMaC
wording in historical release records. A fresh-context review then strengthened
process-wide provider reuse, operation/site ownership, disabled runtime
replacement semantics, migration examples, and integer custom-job coverage. The
reviewed native suites pass 165 object-model tests and 20 registry/manager
tests; all 64 QDMI specification tests pass with two expected skips; and 173
focused Python QDMI and estimator tests pass. Stub regeneration, the
warning-as-error documentation build, and the full lint suite succeed. The
artifact audit is clean; publication and CI observation remain.

## Context and Orientation

QDMI is the Quantum Device Management Interface used by native device provider
libraries. It exposes prefixed C functions from a shared library. A prefix such
as `MQT_DDSIM_QDMI` distinguishes one implementation's exported symbols. The
private files `src/qdmi/DeviceApi.hpp` and `src/qdmi/DeviceApi.cpp` load such a
library, resolve the complete QDMI symbol set, initialize it once, and finalize
it when the last owning object is gone. No QDMI session or job handle is part of
the public MQT API.

`include/mqt-core/qdmi/DeviceRegistry.hpp` and `src/qdmi/DeviceRegistry.cpp`
parse registration sources without loading their libraries. A
`qdmi::DeviceDefinition` contains a stable ID, native-library path, symbol
prefix, enabled flag, and default `SessionParameters`.
`include/mqt-core/qdmi/DeviceManager.hpp` and `src/qdmi/DeviceManager.cpp` own a
registry. The private runtime uses a process-wide, canonical-path weak cache to
reuse a live loaded library without keeping it loaded after every device object
is gone. Each call to `open` creates an independent session.

`include/mqt-core/qdmi/Device.hpp` and `src/qdmi/Device.cpp` provide the public
objects. `src/qdmi/DeviceState.hpp` and `src/qdmi/DeviceState.cpp` retain the
native library, parent session, child session, and job handles for as long as a
public object needs them. Children recursively own their required parent session
state. Sites and operations reference the same device state.

The Python bindings live in `bindings/qdmi/qdmi.cpp`. Their generated type stub
is `python/mqt/core/qdmi.pyi`; it must only be regenerated with
`uvx nox -s stubs`. `bindings/patterns.txt` contains nanobind stub-generation
overload rewrites for both QDMI and the MLIR compiler bindings merged in 1815.
The Qiskit adapter is under `python/mqt/core/plugins/qiskit/`. Built-in DDSIM,
neutral-atom, and superconducting device implementations are under
`src/qdmi/devices/` and register relocatable manifest fragments through
`mqt_configure_qdmi_device`.

Configuration is versioned with `"schema-version": 1`. Sources are layered from
packaged manifests through system, user, project, environment, and runtime
overrides. Entries merge field by field by stable `id`; a disabled entry masks
an inherited entry. Discovery validates configuration and resolves relative
paths against the declaring source but never loads a native library. Opening a
configured library executes native code, so configuration is a trusted input.

## Plan of Work

The first milestone replaces the public layering. Remove the legacy wrapper
source, include, binding, test, and CMake trees; introduce the registry,
manager, and device object model under the QDMI names; migrate neutral-atom and
Qiskit users; and retain complete before/after documentation. Acceptance is that
repository searches find the old product name only in historical migration prose
and all users compile against `MQT::CoreQDMI` or import `mqt.core.qdmi`.

The second milestone makes discovery configuration-driven and relocatable.
Implement strict JSON/TOML parsing, precedence, stable-ID merging, inline and
runtime overrides, and generated built-in manifest fragments. Centralize common
device target configuration in `mqt_configure_qdmi_device`. Acceptance is that
definitions can be listed without executing a provider, relative libraries can
be found after moving an install tree, and invalid sources report their source
and configuration path.

The third milestone replaces the runtime implementation. Store exact QDMI
function signatures in one concrete private library owner, create independent
sessions, retain recursive child and job lifetimes, and expose QDMI's existing
status and format enums directly. Acceptance is that incomplete libraries fail
only when opened, multiple definitions can share one live library while using
independent sessions, and returned objects remain valid after their manager is
destroyed.

The fourth milestone completes bindings, documentation, and tests. Document
every public nanobind class, property, and method in `bindings/qdmi/qdmi.cpp`;
regenerate the stub; add scripted exact-signature tests and a small dynamic
library loading test; and update the changelog and upgrade guide. Acceptance is
that public Python members have nonempty docstrings and the focused native and
Python suites exercise success, error, child, cleanup, enum, and concurrent
manager paths.

The fifth milestone integrates current main. Rebase onto the commits containing
pull requests 1815, 1907, 1908, 1904, and 1909; preserve MLIR/QIR behavior and
centralized patterns; add and maintain a checkout-independent ExecPlan; rerun
validation; and publish the rewritten pull request branch with lease protection.
Acceptance is a reviewed range diff, a clean worktree, an AI-disclosed pull
request description, and a new CI run without integration regressions.

## Concrete Steps

Run all commands from the repository root.

Fetch and rebase safely:

    git fetch --prune origin
    git rebase origin/main

If conflicts arise, inspect `git diff --cc` and the corresponding merged-main
commit before editing. Stage only resolved files and continue with:

    GIT_EDITOR=true git rebase --continue

Before rebasing, create an unpushed local recovery ref using the contributor's
normal naming convention. Afterward, compare that ref with the rewritten series
using `git range-diff` and remove it only after the pull request is accepted. Do
not record the local ref name in this checked-in plan.

Configure and run focused native validation:

    cmake --preset debug
    cmake --build build/debug --target mqt-core-qdmi-object-model-test mqt-core-qdmi-manager-test
    ./build/debug/test/qdmi/device/mqt-core-qdmi-object-model-test
    ./build/debug/test/qdmi/manager/mqt-core-qdmi-manager-test
    ctest --test-dir build/debug -R 'qdmi|QDMI' --output-on-failure

Regenerate stubs and run focused Python validation:

    uvx nox -s stubs
    uv run --no-sync pytest -q test/python/qdmi test/python/na/test_na_qdmi.py test/python/plugins/qiskit

Build documentation with warnings treated as errors through the repository Nox
session, then run the mandatory lint suite:

    uvx nox --non-interactive -s docs
    uvx nox -s lint
    git diff --check

If ccache cannot write its default directory in a restricted environment, set
`CCACHE_DIR` to a writable temporary directory for configure and build commands.
If MLIR is not discoverable, set `MLIR_DIR` to the installed MLIR CMake package
directory without recording the machine-specific value here.

Remove `build/`, `.nox/`, `docs/_build/`, CMake test-discovery JSON files, and
Python cache directories after verification. Do not remove source-controlled
files or another task's output.

After committing this plan and any integration correction with the required
`Assisted-by: GPT-5 via Codex` trailer, update the current pull request branch
with:

    git push --force-with-lease

The pull request description is public AI-authored text. Its first line must be
exactly `🤖 *AI text below* 🤖` before updating it. Inspect the new CI run and
read failing job logs before changing code.

## Validation and Acceptance

Discovery acceptance requires that constructing a manager and reading its
definitions performs no provider initialization. Registry tests must cover
source precedence, duplicate and invalid entries, disable masking, relative
paths, explicit and isolated modes, environment JSON, and runtime overrides.

Runtime acceptance requires that `DeviceManager.open("mqt.ddsim.default")`
returns a usable device; one malformed or missing library produces an ID-keyed
error in `openAll()` without discarding successful devices; child devices and
jobs retain their native state after parent variables and the manager are
destroyed; and the library finalizes after the final live object disappears. The
scripted QDMI test must assign callbacks through fields whose types are exact
`decltype(QDMI_...)` pointers. Simultaneously live managers must reuse the same
canonical provider library, and operation queries must reject sites from another
device session before calling provider code.

Public API acceptance requires that C++ uses `QDMI_Device_Status`,
`QDMI_Job_Status`, and `QDMI_Program_Format` directly, Python exposes documented
`Device.Status`, `Job.Status`, and `ProgramFormat` objects, every public bound
member has a docstring, and regenerated stubs contain `device_id` without Ruff
shadowing exclusions.

Integration acceptance requires preserving 1815's
`test_device_executes_qir_program` test under `test/python/qdmi/test_qdmi.py`,
the QIR formats in `src/qdmi/devices/dd/Device.cpp`, the MLIR module in the stub
session, and both QDMI and MLIR rules in `bindings/patterns.txt`. Documentation
must build without MyST cross-reference warnings. `uvx nox -s lint` and
`git diff --check` must pass.

## Idempotence and Recovery

Configuration, builds, tests, stub generation, documentation, and lint commands
are repeatable. Stub generation may rewrite generated `.pyi` files; compare them
with Git and retain changes only when they follow from binding or pattern
inputs. CMake and Nox output is disposable and can be regenerated.

The rebase rewrites commits, so create an unpushed local recovery ref before
starting. If an unresolved rebase becomes inconsistent, `git rebase --abort`
returns to the pre-rebase state; after completion, compare against the recovery
ref with `git range-diff`. Never use `git reset --hard` or overwrite the
recovery ref. Publish only with `--force-with-lease`, which refuses to overwrite
unexpected remote work.

## Artifacts and Notes

The mainline integration point and recovery evidence are:

    old base: 7fc5a80ab
    intermediate base after pull requests 1815 and 1907: a228a3dc2
    current base after pull requests 1904 and 1909: 559fde4b2
    pre-rebase tip: d6ee912bc
    rebased series: ten implementation commits, one plan commit, and one status commit
    latest range-diff: eleven commits exact; plan commit differs only by the adjacent 1904 changelog link

Focused validation after the fresh-context review produced:

    object model: 165 tests passed
    registry and manager: 20 tests passed
    QDMI CTest selection: 64 tests passed, 2 expected skips
    Python QDMI and estimator: 173 tests passed
    stub generation: successful, expected QDMI stub update generated
    documentation: successful with -W
    lint: successful, all hooks passed

The final artifact audit is clean.

The first rebase conflict involved `CHANGELOG.md`, `UPGRADING.md`, `noxfile.py`,
`pyproject.toml`, `python/mqt/core/plugins/qiskit/estimator.py`, and the two
destinations of the old binding pattern file. The resolution keeps the mainline
MLIR Python package and QIR behavior, changes only the obsolete wrapper patterns
and module to QDMI, and removes the duplicate historical child device upgrade
section.

## Interfaces and Dependencies

The final C++ public interfaces are `qdmi::DeviceDefinition`,
`qdmi::ConfigOptions`, `qdmi::DeviceRegistry`, `qdmi::DeviceManager`,
`qdmi::OpenAllResult`, `qdmi::Device`, `qdmi::Job`, `qdmi::Site`,
`qdmi::Operation`, and `qdmi::SessionParameters`. Status and program format
signatures use the QDMI types directly. The Python equivalents live in
`mqt.core.qdmi` and follow Python naming conventions such as `device_id`,
`register_device`, `open_all`, and read-only `definitions`.

The implementation depends on QDMI for C declarations, `nlohmann_json` for JSON,
the vendored single-header toml++ parser for TOML, nanobind for Python bindings,
and the platform dynamic-library API. The selected QDMI revision is recorded in
`cmake/ExternalDependencies.cmake`. These dependencies do not authorize loading
any library during discovery. Only `DeviceManager.open` and `openAll` may
execute configured provider code.

Revision note (2026-07-15): Created this plan after 1907 introduced the
repository ExecPlan process. It records the already implemented redesign, the
pull request 1815 and 1907 rebase decisions, and the remaining validation and
publication work so the task can continue from this file alone.

Revision note (2026-07-15 11:38Z): Recorded post-rebase native, Python, stub,
and documentation validation; documented the sandbox-specific ccache recovery;
and narrowed the remaining work to lint, cleanup, publication, and CI.

Revision note (2026-07-15 11:40Z): Regenerated the plan for the 1908
checkout-independent requirements. Removed the task's local path, account and
branch details, changed scope and commands to repository-relative forms, and
retained only stable pull request and commit references.

Revision note (2026-07-15 11:41Z): Recorded the successful mandatory lint run
after the checkout-independent regeneration.

Revision note (2026-07-15 11:44Z): Recorded the final pull request 1908 rebase
and exact range-diff, completed the portability and generated-artifact audits,
and updated the outcome and base evidence.

Revision note (2026-07-15 11:49Z): Recorded the lease-protected publication,
repaired upstream tracking, and AI-disclosed pull request description. Only
observation of the final CI run remains.

Revision note (2026-07-15 11:52Z): Recorded the subsequent mainline advance, the
pre-commit service's mergeability error, the pull requests 1904 and 1909 scope
review, the second rebase, and its range-diff result.

Revision note (2026-07-15): Added the final maintainer-feedback iteration:
version-neutral QDMI naming, removal of the configuration ABI marker, estimator
reversion, complete generated custom-query documentation, preservation of
historical FoMaC wording, and a required fresh-context review.

Revision note (2026-07-15 13:18Z): Recorded the fresh-context review and all
five resulting corrections: process-wide canonical weak caching, operation-site
ownership validation, disabled replacement removal, accurate historical
migration examples, and verified integer custom-job binding support.

Revision note (2026-07-15 13:24Z): Recorded the successful post-review native,
Python, stub, warning-as-error documentation, and full lint validation.

Revision note (2026-07-15 13:25Z): Recorded the final generated-artifact cleanup
and audit.

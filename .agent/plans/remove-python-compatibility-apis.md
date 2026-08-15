# Remove the deprecated QDMI Python compatibility APIs

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core 4 removes the FoMaC compatibility name from its C++ and Python QDMI
APIs. Python users import QDMI entities from `mqt.core.qdmi`, use registry and
device-opening functions from `mqt.core.qdmi.driver`, and import the
neutral-atom specialization from `mqt.core.na.qdmi`. Imports of `mqt.core.fomac`
and `mqt.core.na.fomac` fail, and `mqt.core.qdmi.driver.Session` is absent. C++
users include `qdmi/Client.hpp` and use the `qdmi` namespace and `MQT::CoreQDMI`
target.

The result is visible in a clean wheel installation. The supported modules
import and open registered devices. The removed modules do not exist in the
wheel and cannot be imported.

## Progress

- [x] (2026-08-14 13:40Z) Confirm the removal contract from issue #2086 and
      inventory the current compatibility code, tests, stubs, and docs.
- [x] (2026-08-14 14:08Z) Remove the two compatibility modules and the Python
      `Session` binding.
- [x] (2026-08-14 14:10Z) Replace compatibility tests with supported-namespace
      and removal tests.
- [x] (2026-08-14 14:12Z) Regenerate stubs and update current documentation and
      migration guidance.
- [x] (2026-08-14 14:22Z) Run focused native and Python tests, documentation,
      lint, and wheel checks.
- [x] (2026-08-14 14:25Z) Perform an adversarial review and correct all in-scope
      findings.
- [x] (2026-08-14) Remove the obsolete namespace-removal test and rename the
      remaining C++ namespace, headers, libraries, targets, adapters, and tests.

## Surprises & Discoveries

- Observation: The neutral-atom compatibility path is a native nanobind
  submodule, while `mqt.core.fomac` is a Python file. Evidence:
  `bindings/na/register_na.cpp` creates `fomac`, and `python/mqt/core/fomac.py`
  re-exports QDMI objects.
- Observation: The deprecated Python `Session` owns no independent registry. The
  supported module functions use the process-wide QDMI driver. Removing the
  binding therefore removes no capability that is unique to an isolated Python
  object.
- Observation: `nanobind.stubgen` regenerates exported modules but does not
  delete a stale stub for a removed module. The change must therefore delete
  `python/mqt/core/na/fomac.pyi` explicitly. A clean checkout does not recreate
  that file because the native module no longer exports `fomac`.
- Observation: The local documentation interpreter did not trust the system CA
  chain when it fetched the public QDMI tag file. The same build proceeded with
  the CA bundle from the locked Python environment. This was a local trust-store
  boundary, not a documentation defect.

## Decision Log

- Decision: Deliver the removals in one pull request. Rationale: Runtime
  exports, generated stubs, tests, and migration documentation describe one
  atomic Python compatibility break. Date/Author: 2026-08-14, Lukas Burgholzer
  and Codex.
- Decision: Do not add tombstone modules or attribute fallbacks. Rationale: MQT
  Core 3 contains the deprecation path. MQT Core 4 should fail removed imports
  directly and expose only the supported namespace. Date/Author: 2026-08-14,
  Lukas Burgholzer and Codex.
- Decision: Remove the FoMaC name from both language APIs in MQT Core 4.
  Rationale: Keeping the C++ namespace, installed headers, and CMake targets
  would leave two names for the same QDMI client abstraction after the Python
  transition. The major release is the correct boundary for one complete rename.
  Date/Author: 2026-08-14, Lukas Burgholzer and Codex.

## Outcomes & Retrospective

MQT Core now exposes the QDMI API through the `qdmi` C++ namespace,
`MQT::CoreQDMI`, `mqt.core.qdmi`, `mqt.core.qdmi.driver`, and
`mqt.core.na.qdmi`. The clean wheel contains the supported Python modules, has
no compatibility-module file, and exposes no Python `Session` class. Installed
C++ packages contain no FoMaC header, namespace, library, or target.

The final review removed an unrelated ownership statement from the driver guide.
The migration guide is the single place that explains the removed APIs and their
replacements. No compatibility tombstone or duplicate runtime path remains.

## Context and Orientation

QDMI defines a low-level device interface and a client interface. QDMI device
libraries implement the device interface. `src/qdmi/driver` loads these
libraries and implements the client interface for MQT Core.
`src/qdmi/Client.cpp` and `include/mqt-core/qdmi/Client.hpp` provide the owning
C++ wrappers.

The native extension in `bindings/qdmi/qdmi.cpp` exposes QDMI entities as
`mqt.core.qdmi` and registry operations as `mqt.core.qdmi.driver`. It also
exposes the deprecated Python `Session` class. `python/mqt/core/fomac.py`
re-exports objects from both supported modules. The neutral-atom extension in
`bindings/na/register_na.cpp` creates both `qdmi` and deprecated `fomac`
submodules. Nox invokes `nanobind.stubgen`; the generated `.pyi` files live in
`python/mqt/core` and must not be edited by hand.

MQT Core 3 must receive the already merged namespace and Slurm changes before
this pull request merges. Those backports preserve the announced deprecation
window. This pull request targets only `main` and must not be backported.

## Plan of Work

Delete `python/mqt/core/fomac.py`. Remove the `fomac` submodule from
`bindings/na/register_na.cpp`. Remove the warning helper and the
`nb::class_<fomac::Session>` binding from `bindings/qdmi/qdmi.cpp`. Retain all
module-level driver functions and the native QDMI entities. Remove includes only
when the remaining translation unit no longer uses them.

Delete `test/python/qdmi/test_namespace.py`. Remove the deprecated-session
construction tests and helper from `test/python/qdmi/test_qdmi.py`. Keep the
existing tests for registration, stable device IDs, device opening,
configuration overrides, ownership, job submission, and custom properties.

Rename the owning C++ wrapper to `qdmi/Client.hpp`, the namespace to `qdmi`, and
the library target to `MQT::CoreQDMI`. Rename the neutral-atom wrapper and the
MLIR compiler adapter in the same commit. Update every current source, binding,
test, CMake consumer, and documentation example. Do not add C++ compatibility
aliases in the v4 branch.

Run the stub-generation Nox session after rebuilding the changed bindings. The
generated tree must remove `python/mqt/core/na/fomac.pyi`, remove its export
from `python/mqt/core/na/__init__.pyi`, and remove `Session` from
`python/mqt/core/qdmi/driver.pyi`.

Update `docs/qdmi/driver.md`, the current `UPGRADING.md` section, and
`CHANGELOG.md`. Name each removed path and its replacement. Keep historical
release sections accurate.

## Concrete Steps

Run all commands from the repository root through the worktree-local wrapper:

    ./.agent/run.sh uv sync --inexact --only-group build --only-group test
    ./.agent/run.sh uv sync --inexact --no-dev \
      --no-build-isolation-package mqt-core
    ./.agent/run.sh uvx nox -s stubs
    ./.agent/run.sh uv run --no-sync pytest test/python/qdmi -q
    ./.agent/run.sh uvx nox -s tests
    ./.agent/run.sh uvx nox -s tests-3.14
    ./.agent/run.sh uvx nox --non-interactive -s docs
    ./.agent/run.sh uvx nox -s lint

Build a wheel through the established packaging configuration. Install it in a
clean temporary environment. Inspect the archive and run a Python process that
imports `mqt.core.qdmi`, `mqt.core.qdmi.driver`, and `mqt.core.na.qdmi`; rejects
the two removed modules; and verifies that `driver.Session` is absent.

## Validation and Acceptance

The QDMI test suite must still register devices, list stable IDs, open fresh
device sessions, apply per-open configuration, retain device and job lifetimes,
and execute jobs. The C++ build and installed CMake package must expose only the
QDMI names.

Stub generation must produce no manual edits. Documentation must build with
warnings treated as errors. Lint must pass without broad suppressions. A clean
wheel must contain the three supported modules and no file or native export for
the removed paths. Platform CI must verify Windows, free-threaded Python, and
all supported wheel targets.

## Idempotence and Recovery

Stub generation, builds, tests, and documentation builds are repeatable. Build
output and tool caches remain in this worktree. If an external download fails,
rerun the same command without changing source. If `main` advances before
publication, rebase this single branch and rerun the binding, stub, namespace,
documentation, lint, and wheel checks.

The backports are separate remote operations. Apply `backport-potential` to pull
request 2074, wait for its `v3.x` backport to merge, and then repeat the process
for pull requests 2025 and 2043. Do not apply the label to the removal pull
request.

### Artifacts and Notes

- `uvx nox -s stubs` completed and regenerated the QDMI and neutral-atom stubs.
- The focused QDMI suite passed on Python 3.10 through 3.14 with 306 tests per
  interpreter.
- The Sphinx documentation build passed with warnings treated as errors after
  the documentation-specific native build generated the MLIR reference files.
- The full lint session passed.
- A clean CPython 3.14 environment installed the abi3 wheel without
  dependencies, imported all supported QDMI modules, opened `mqt.ddsim.default`,
  and found no compatibility modules or `Session` class.
- Wheel archive inspection confirmed that the supported modules are present and
  the removed Python files are absent.

### Interfaces and Dependencies

The final supported Python interfaces are:

    from mqt.core import qdmi
    from mqt.core.na import qdmi as na_qdmi
    from mqt.core.qdmi.driver import (
        DeviceDefinition,
        open_device,
        register_device,
        register_device_if_absent,
        registered_device_ids,
    )

No new dependency is required. `mqt.core.qdmi.driver` continues to delegate to
the process-wide `qdmi::Driver`. Returned C++ and Python devices continue to own
the provider library and native device session through the QDMI client wrappers.

Revision note: Created from the current `main` implementation and issue #2086
before code changes began. Updated after implementation, validation, and the
final adversarial review.

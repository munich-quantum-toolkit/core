# Adopt nanobind 3 split-mode wheels

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core currently publishes separate CPython 3.10 and 3.11 wheels, followed by
one stable-ABI wheel for CPython 3.12 and newer. nanobind 3 split mode moves the
Python-version-specific binding runtime into the small `nanobind-backend`
package. For the next major release, MQT Core can require Python 3.11 and
publish one `cp311-abi3` wheel per operating system and processor for every
GIL-enabled CPython version.

Free-threaded support starts at CPython 3.15, where the provisional `abi3t`
stable ABI exists. Each platform must therefore produce one separate
`cp315-abi3t` wheel. Both wheel types require `nanobind-backend>=1.0`. MQT Core
does not publish CPython 3.13t or 3.14t wheels.

## Progress

- [x] (2026-08-22 12:15Z) Read nanobind 3.0's changelog, split-mode guide,
  free-threading guide, CMake implementation, tests, release discussion, and
  backend wheel inventory.
- [x] (2026-08-22 12:15Z) Refreshed the clean detached checkout to current
  `origin/main` before designing changes.
- [x] (2026-08-22 12:15Z) Built nanobind 3.0's upstream suite in split mode on
  CPython 3.15t and ran its tests: 457 passed and 243 skipped.
- [x] (2026-08-22 12:25Z) Raised the Python floor to 3.11, updated nanobind,
      selected `cp311` and `cp315t` Stable ABI wheels, added the runtime
      backend, and migrated the nanobind 3 API uses.
- [x] (2026-08-22 12:26Z) Built the `cp311-abi3` wheel and imported all four
  extension modules on CPython 3.11 and 3.14.
- [x] (2026-08-22 12:47Z) Built the `cp315-abi3t` wheel, imported all four
  extensions with the GIL disabled, and ran a concurrent DD smoke test.
- [x] (2026-08-22 13:11Z) Restored ELF section garbage collection for split
  targets and rebuilt the free-threaded wheel.
- [x] (2026-08-22 13:12Z) Ran the full native and Python test suites, focused
  concurrency checks, wheel and sdist checks, ABI auditing, and lint.
- [x] (2026-08-22 13:12Z) Inspected the final diff and recorded the outcome.
- [x] (2026-08-22 14:10Z) Removed the redundant dense state-vector copy and
  verified the vector and matrix array ownership paths with 18 DD tests.
- [x] (2026-08-22 14:25Z) Preserved Daniel Haag's nanobind 3 and Python 3.11
  commits and prepared the complete change as a rescope of pull request
  `#2209`.
- [x] Validate the Windows `abi3t` detection workaround on both hosted Windows
  wheel builders.
- [x] (2026-08-23 10:15Z) Addressed the collected review by making split mode
  unconditional, restoring full Windows x64 wheel tests, and simplifying the
  platform branches and release notes.

## Surprises & Discoveries

- Observation: nanobind 3.0's final documentation and implementation support
  split mode on free-threaded CPython 3.15 through the provisional `abi3t`
  stable ABI. Earlier nanobind discussions predate that implementation.
  Evidence: the upstream split build produced `.abi3t` extensions and its pytest
  suite reported `457 passed, 243 skipped` on CPython 3.15t.
- Observation: CPython 3.14t cannot use split mode. Evidence: nanobind's current
  release discussion says that 3.14t must use linked mode, and its CMake code
  rejects split mode before Python 3.15 on a free-threaded interpreter.
- Observation: Requiring Python 3.11 and starting free-threaded support at 3.15
  removes the backend exceptions. `nanobind-backend` 1.0.0 covers every
  mainstream platform in the resulting release matrix, so the project can use a
  normal runtime dependency instead of dynamic wheel metadata.
- Observation: nanobind 3 removed the Boolean form of `arg::none()`. MQT Core
  uses `.none(true)` twice in
  `bindings/ir/operations/register_if_else_operation.cpp`; both uses mean the
  new zero-argument `.none()` form.
- Observation: four DD binding functions share mutable static random-number
  engines. These engines race across independent `DDPackage` instances on a
  free-threaded interpreter. A thread-local engine removes the cross-object race
  without serializing unrelated calls.
- Observation: nanobind 3.0 does not propagate its optimized-build ELF section
  garbage-collection option to split targets. Evidence: the split MLIR extension
  was 49,552,688 bytes without `--gc-sections` and 26,469,280 bytes with it. The
  linked extension was 27,847,512 bytes. LTO changed the split result by only
  about 65 KiB and did not fix the omission.
- Observation: removing the GIL exposes several process-wide mutable C++ data
  structures that independent Python objects can reach. The symbolic-variable
  registry, implicit register-name counters, DD edge traversal cache, and QDMI
  driver catalog therefore need C++ synchronization. Shared mutable object
  instances remain caller-synchronized.
- Observation: split wheels need their runtime dependencies during downstream
  image builds. The Slurm image used `uv pip install --no-deps`, which made all
  split extension imports fail because `nanobind_backend` was absent.
- Observation: ThreadSanitizer cannot start in this ARM64 host environment. Its
  runtime exits before test discovery with
  `FATAL: ThreadSanitizer: unexpected memory mapping`. Native concurrency tests
  and a real CPython 3.15t smoke test cover the new synchronization here.
- Observation: `VectorDD.get_vector()` materialized a contiguous `dd::CVec` and
  then copied it into a second allocation for NumPy. A capsule can own the
  returned vector directly. `MatrixDD.get_matrix()` already fills its final
  contiguous allocation and needs no extra data-layout conversion.
- Observation: scikit-build-core leaves `SKBUILD_SOABI` empty for Windows
  `abi3t` builds, while nanobind 3.0 uses that value to detect free-threading.
  The frontend then advertises the classic nanobind platform ABI and cannot use
  the free-threaded backend. nanobind's current split workflow tests `abi3t`
  only on Linux.
- Observation: cibuildwheel passes test requirements through `cmd.exe` on
  Windows, where version-bound operators become shell redirections. Windows x64
  builds install the test dependency group through one shell-safe `uv` command
  and run the full wheel tests. Windows ARM64 and CPython 3.15 builds use the
  dependency-free import test.
- Observation: macOS x86-64 split frontends must export nanobind's weak
  exception RTTI so that the backend catches its exception types. Apple arm64
  uses non-unique RTTI and needs no extra exports. Exporting every
  default-visible symbol causes static MQT, LLVM, and MLIR definitions from
  separate extensions to interpose on both architectures.

## Decision Log

- Decision: Build every Python binding in split mode. Rationale: MQT Core
  supports CPython on the mainstream wheel platforms covered by
  `nanobind-backend`; PyPy, musllinux, and linked binding builds are not part of
  the supported matrix. Date/Author: 2026-08-23 / Codex.
- Decision: Use `wheel.py-api = "cp311"` and one scikit-build-core override for
  free-threaded CPython 3.15 and newer. Rationale: the override produces a
  separate `cp315-abi3t` artifact without advertising it as a classic `abi3`
  wheel. Date/Author: 2026-08-22 / Codex.
- Decision: Drop Python 3.10 and all free-threaded builds before CPython 3.15.
  Rationale: this is a major release, and the simpler support boundary removes
  unresolvable backend variants. Date/Author: 2026-08-22 / Codex.
- Decision: Declare `nanobind-backend>=1.0` as an unconditional dependency.
  Rationale: every supported build uses nanobind split mode; PyPy and musllinux
  are not supported. Date/Author: 2026-08-23 / Codex.
- Decision: Add a split-only ELF linker workaround instead of enabling LTO.
  Rationale: `--gc-sections` fixes the nanobind 3.0 CMake omission at its
  consumer boundary and halves the MLIR extension size; LTO adds build cost
  without a material result. Date/Author: 2026-08-22 / Codex.
- Decision: Synchronize only process-wide state and independent-object paths.
  Rationale: free-threading makes those races new default hazards, while making
  each mutable MQT object safe for simultaneous use would be a broader API and
  performance contract. Date/Author: 2026-08-22 / Codex.
- Decision: Let a nanobind capsule own the dense `dd::CVec` returned by
  `VectorDD.get_vector()`. Rationale: this removes one allocation and one
  exponential-size copy while keeping the returned NumPy array independent of
  the DD package. Date/Author: 2026-08-22 / Codex.
- Decision: Complete the existing nanobind pull request #2209 instead of opening
  a competing pull request. Rationale: #2209 and #2009 already contain Daniel
  Haag's nanobind migration and Python-floor work. Keeping those commits in the
  branch preserves authorship and gives reviewers one integration point.
  Date/Author: 2026-08-22 / Codex.
- Decision: Keep the current QDMI driver synchronization in a separate commit.
  Rationale: current `main` needs the synchronization for free-threaded Python,
  while pull request #1901 deletes this driver and can drop the bridge commit if
  it lands first. Date/Author: 2026-08-22 / Codex.
- Decision: Supply nanobind's missing `NB_ABI` value inside the shared binding
  wrapper for Windows `abi3t` builds. Rationale: the function-scoped override
  activates nanobind's full `abi3t` checks and compile definitions without
  changing other modules or platforms. Date/Author: 2026-08-22 / Codex.
- Decision: Keep the macOS module-initializer allowlist and add only nanobind's
  four backend exception RTTI symbols to x86-64 split modules. Rationale: the
  narrow list matches the x86-64 backend and preserves cross-module exception
  matching without exposing statically linked project and toolchain symbols.
  Apple arm64 uses non-unique RTTI and needs no extra exports. Date/Author:
  2026-08-22 / Codex.

## Outcomes & Retrospective

The implementation produces one `cp311-abi3` wheel and one `cp315-abi3t` wheel
per platform. Final local Linux AArch64 artifacts were 78,355,810 and 78,364,536
bytes. Both declare the `nanobind-backend>=1.0` dependency and pass
`check-wheel-contents`; the classic wheel also passes strict `abi3audit`. Fresh
installs imported all four extension modules on CPython 3.11, 3.14, and
free-threaded 3.15 with the GIL disabled.

The final wheel passed 727 Python tests with three expected skips. The release
C++ build passed all 4,143 configured tests with one test skipped by its own
contract. Focused symbolic-variable and QDMI concurrency tests passed, as did a
512-task CPython 3.15t smoke test. nanobind's upstream split-mode suite passed
457 tests with 243 skipped on the same free-threaded interpreter. Stub
generation, `uv lock --check`, `git diff --check`, and `uvx nox -s lint` pass.
The vector and matrix DD ownership tests pass, and both affected DD test modules
pass all 18 tests.

Generated installation and tooling pages still describe the old wheel matrix.
Their source templates need a separate templates-repository update; this work
does not edit generated files.

Pull request #2209 is the integration vehicle. Its branch retains Daniel Haag's
original nanobind 3 commits and signed cherry-picks of the Python 3.11 commits
from pull request #2009. The QDMI synchronization is isolated so that pull
request #1901 can remove it with the legacy driver.

## Context and Orientation

`pyproject.toml` defines build requirements, scikit-build-core settings,
interpreter-aware overrides, dependency groups, and cibuildwheel's release
matrix. `cmake/AddMQTPythonBinding.cmake` funnels all four native extension
modules through one `nanobind_add_module` call, so split mode belongs there
rather than in each binding directory. `.github/workflows/ci.yml` pins the
nanobind headers used by the C++ lint build. `uv.lock` records the resolved
build tool versions.

Split mode means that an MQT Core extension contains its binding frontend but
imports nanobind's version-specific runtime backend when Python initializes the
module. Classic stable ABI, tagged `abi3`, supports GIL-enabled Python.
Free-threaded stable ABI, tagged `abi3t`, first exists in CPython 3.15. A linked
mode module contains both parts and is tied to one CPython ABI, such as
`cp314t`.

Scikit-build-core requests the classic `cp311` stable ABI by default. One
override changes that request to `cp315t` when the interpreter is free-threaded
and at least Python 3.15. The CMake helper enables split mode for every binding
build.

## Plan of Work

Raise both nanobind constraints in `pyproject.toml` to 3.0 and the Python floor
to 3.11. Set the classic wheel floor to CPython 3.11. Override it with `cp315t`
on free-threaded CPython 3.15 and newer. Skip CPython 3.13t and 3.14t in
cibuildwheel. Use an import-only wheel test where the full test dependencies do
not yet provide Python 3.15 or Windows ARM64 wheels.

`cmake/AddMQTPythonBinding.cmake` passes `BACKEND_MODULE nanobind_backend` for
every extension. Retain `FREE_THREADED`; nanobind selects `abi3` or `abi3t` from
the interpreter and scikit-build-core settings.

Apply the two required `.none()` API migrations. Change the four shared DD
random engines to `thread_local`. Do not add `.freeze()`, object pooling, manual
list builders, or extra locking without a benchmark or a narrower thread-safety
contract; nanobind 3 already applies its dispatcher, string, reference-counting,
iterator, sequence-caster, and ndarray improvements to the existing bindings.
Let the NumPy capsule own the dense vector returned by `VectorDD.get_vector()`
instead of allocating and copying a second buffer. Keep the matrix exporter on
its existing direct-fill path and make capsule construction exception-safe in
both exporters.

Regenerate `uv.lock`. Update the C++ lint workflow's exact nanobind version. Do
not edit template-generated documentation. Add a changelog entry only when a
pull request number is available, because repository policy requires that
reference and all authors.

## Concrete Steps

Run all commands from the repository root.

First edit the files described above with focused patches. Regenerate the lock
file with:

    uv lock --upgrade-package nanobind

Build wheels into temporary output directories with CPython 3.11 and 3.15t.
Inspect wheel names, extension suffixes, and `Requires-Dist` entries. Install
each wheel into a fresh matching environment and import `mqt.core.ir`,
`mqt.core.dd`, `mqt.core.qdmi`, and `mqt.core.mlir`.

Run the focused binding tests with:

    uv run --no-sync pytest test/python/ir test/python/dd test/python/qdmi/test_slurm.py

Run package checks appropriate to built artifacts, then finish with:

    uvx nox -s lint

Record exact pass, failure, or unavailable status in this plan.

## Validation and Acceptance

On mainstream Linux, macOS, and Windows, a CPython 3.11 build must produce a
`cp311-abi3` wheel. Its metadata must require `nanobind-backend>=1.0`, and the
installed package must import on supported GIL-enabled CPython versions.

CPython 3.13t and 3.14t must not appear in the cibuildwheel build identifiers.

A CPython 3.15t build must produce a `cp315-abi3t` wheel that requires the
backend and imports all four extensions.

Windows ARM64 must use the same `cp311-abi3` and `cp315-abi3t` split layout as
the other supported platforms.

The existing Python binding tests must pass. The final lint session must pass,
or this plan must record a specific pre-existing or environmental limitation.
The final diff must contain no generated documentation, credentials, build
products, or unrelated changes.

## Idempotence and Recovery

Lock generation, CMake configuration, wheel builds, tests, and lint are safe to
repeat. Use fresh temporary wheel and virtual-environment directories when
comparing ABI variants. Build products remain under `build/` or temporary
directories and are not committed. If one split-mode platform fails, fix or
remove that unsupported platform instead of publishing a different binding
layout.

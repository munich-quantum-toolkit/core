# Adopt nanobind 3 split-mode wheels

Status: historical implementation record.

## Goal and scope

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

## Constraints

- nanobind 3.0's final documentation and implementation support split mode on
  free-threaded CPython 3.15 through the provisional `abi3t` stable ABI. Earlier
  nanobind discussions predate that implementation. Evidence: the upstream split
  build produced `.abi3t` extensions and its pytest suite reported
  `457 passed, 243 skipped` on CPython 3.15t.

- CPython 3.14t cannot use split mode. Evidence: nanobind's current release
  discussion says that 3.14t must use linked mode, and its CMake code rejects
  split mode before Python 3.15 on a free-threaded interpreter.

- Requiring Python 3.11 and starting free-threaded support at 3.15 removes the
  backend exceptions. `nanobind-backend` 1.0.0 covers every mainstream platform
  in the resulting release matrix, so the project can use a normal runtime
  dependency instead of dynamic wheel metadata.

- nanobind 3 removed the Boolean form of `arg::none()`. MQT Core uses
  `.none(true)` twice in
  `bindings/ir/operations/register_if_else_operation.cpp`; both uses mean the
  new zero-argument `.none()` form.

- four DD binding functions share mutable static random-number engines. These
  engines race across independent `DDPackage` instances on a free-threaded
  interpreter. A thread-local engine removes the cross-object race without
  serializing unrelated calls.

- nanobind 3.0 does not propagate its optimized-build ELF section
  garbage-collection option to split targets. Evidence: the split MLIR extension
  was 49,552,688 bytes without `--gc-sections` and 26,469,280 bytes with it. The
  linked extension was 27,847,512 bytes. LTO changed the split result by only
  about 65 KiB and did not fix the omission.

- removing the GIL exposes several process-wide mutable C++ data structures that
  independent Python objects can reach. The symbolic-variable registry, implicit
  register-name counters, DD edge traversal cache, and QDMI driver catalog
  therefore need C++ synchronization. Shared mutable object instances remain
  caller-synchronized.

- split wheels need their runtime dependencies during downstream image builds.
  The Slurm image used `uv pip install --no-deps`, which made all split
  extension imports fail because `nanobind_backend` was absent.

- ThreadSanitizer did not execute in the recorded validation. Native concurrency
  tests and a CPython 3.15t smoke test passed, but do not replace race-detector
  coverage.

- `VectorDD.get_vector()` materialized a contiguous `dd::CVec` and then copied
  it into a second allocation for NumPy. A capsule can own the returned vector
  directly. `MatrixDD.get_matrix()` already fills its final contiguous
  allocation and needs no extra data-layout conversion.

- scikit-build-core leaves `SKBUILD_SOABI` empty for Windows `abi3t` builds,
  while nanobind 3.0 uses that value to detect free-threading. The frontend then
  advertises the classic nanobind platform ABI and cannot use the free-threaded
  backend. nanobind's current split workflow tests `abi3t` only on Linux.

- cibuildwheel passes test requirements through `cmd.exe` on Windows, where
  version-bound operators become shell redirections. Windows x64 builds install
  the test dependency group through one shell-safe `uv` command configured for
  the platform and run the full wheel tests. Windows ARM64 and CPython 3.15
  builds clear that command and use the dependency-free import test.

- CMake 4.4 detects free-threaded Python and propagates `Py_GIL_DISABLED`
  through its Python module targets on Windows. Requiring CMake 4.4.1 for Python
  package builds replaces MQT Core's manual definition without raising the CMake
  floor for ordinary C++ consumers.

- scikit-build-core can provision its selected CMake version for isolated
  package builds, but the repository's no-build-isolation nox sessions must
  install that version through the existing build dependency group.

- macOS x86-64 split frontends must export nanobind's weak exception RTTI so
  that the backend catches its exception types. Apple arm64 uses non-unique RTTI
  and needs no extra exports. Exporting every default-visible symbol causes
  static MQT, LLVM, and MLIR definitions from separate extensions to interpose
  on both architectures.

## Decisions

- Build every Python binding in split mode. Rationale: MQT Core supports CPython
  on the mainstream wheel platforms covered by `nanobind-backend`; PyPy,
  musllinux, and linked binding builds are not part of the supported matrix.

- Use `wheel.py-api = "cp311"` and one scikit-build-core override for
  free-threaded CPython 3.15 and newer. Rationale: the override produces a
  separate `cp315-abi3t` artifact without advertising it as a classic `abi3`
  wheel.

- Drop Python 3.10 and all free-threaded builds before CPython 3.15. Rationale:
  this is a major release, and the simpler support boundary removes unresolvable
  backend variants.

- Declare `nanobind-backend>=1.0` as an unconditional dependency. Rationale:
  every supported build uses nanobind split mode; PyPy and musllinux are not
  supported.

- Add a split-only ELF linker workaround instead of enabling LTO. Rationale:
  `--gc-sections` fixes the nanobind 3.0 CMake omission at its consumer boundary
  and halves the MLIR extension size; LTO adds build cost without a material
  result.

- Synchronize only process-wide state and independent-object paths. Rationale:
  free-threading makes those races new default hazards, while making each
  mutable MQT object safe for simultaneous use would be a broader API and
  performance contract.

- Let a nanobind capsule own the dense `dd::CVec` returned by
  `VectorDD.get_vector()`. Rationale: this removes one allocation and one
  exponential-size copy while keeping the returned NumPy array independent of
  the DD package.

- The QDMI driver required synchronization for free-threaded Python. That
  requirement belongs to the driver lifetime and concurrency boundary; a driver
  replacement must retain it.

- Supply nanobind's missing `NB_ABI` value inside the shared binding wrapper for
  Windows `abi3t` builds. Rationale: the function-scoped override activates
  nanobind's full `abi3t` checks and compile definitions without changing other
  modules or platforms.

- Keep the macOS module-initializer allowlist and add only nanobind's four
  backend exception RTTI symbols to x86-64 split modules. Rationale: the narrow
  list matches the x86-64 backend and preserves cross-module exception matching
  without exposing statically linked project and toolchain symbols. Apple arm64
  uses non-unique RTTI and needs no extra exports.

- Require CMake 4.4.1 only through scikit-build-core and remove the Windows
  `DISABLE_GIL` override. Rationale: CMake now owns free-threaded Python
  detection and compile definitions, while ordinary C++ builds can retain the
  project's CMake 3.24 floor.

- Configure the Windows dependency-group installation once at the platform level
  and clear it for smoke-only builds. Rationale: cibuildwheel cannot safely
  quote its generated requirement arguments for `cmd.exe`; one native
  `uv pip install --group test` command preserves the full AMD64 wheel suite
  without a version-specific selector.

## Outcome and validation

The implementation produced classic `cp311-abi3` and free-threaded `cp315-abi3t`
Linux AArch64 wheels with `nanobind-backend>=1.0`. Wheel-content checks passed;
the classic wheel passed strict abi3audit. Clean environments imported all four
extensions on CPython 3.11, 3.14, and free-threaded 3.15 with the GIL disabled.

The recorded wheel/native suites, QDMI concurrency and DD ownership tests, a
512-task free-threaded smoke test, stubs, lock validation, and lint passed.
Windows ARM64 and CPython 3.15 used import-only wheel tests; this is not full
platform test coverage.

Generated installation/tooling pages require a separate source-template update.
Dependency compatibility workarounds are recorded in the decisions with their
affected dependency boundaries; current versions remain owned by the build
configuration.

## Code and ownership

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

## Acceptance

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

## Removing dependency workarounds

The original upstream issue drafts were not published. Before removing a
workaround, check the selected dependency version and reproduce the relevant
case:

- nanobind 3.0.0 with scikit-build-core 1.0.3, CMake 4.4.2, and Windows CPython
  3.15t left `NB_ABI` empty when `SKBUILD_SOABI` was empty. Remove the override
  when upstream selects the correct `abi3t` path from its Python/CMake inputs
  and the Windows import checks pass.
- macOS x86-64 split modules needed the `__ZTI` and `__ZTS` symbols for
  `nanobind::abi1::python_error` and `nanobind::abi1::builtin_exception`. Remove
  the extra exports when upstream supplies equivalent exception matching without
  exporting unrelated static-library symbols.
- nanobind's split targets omitted the optimized ELF section-GC options. Remove
  the local linker option when upstream supplies it and a comparable extension
  retains the expected size reduction.
- cibuildwheel 4.2.0 expanded Windows test groups through `cmd.exe`, where
  requirements containing `<` and `>` became redirections. Remove the native
  group-install command when upstream safely executes or quotes those arguments
  and the full Windows test group installs with its constraints intact.

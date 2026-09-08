# Build and test performance

Status: findings implemented; hosted validation pending.

## Scope and baseline

The baseline is main `1bf02dd769`, measured on 2026-09-07. Local controlled
experiments used ARM64, GCC 13.3, CMake 4.4.2, LLVM/MLIR 23.1.0, and Python
3.14.7. Native cold builds used 16 workers, disabled compiler caching and IPO,
and set `DEPLOY=ON`. Dependency downloads and filesystem caches were warm.
Actions observations cover four successful main runs, not a controlled hosted
before/after comparison. Summed compile durations overlap and are not elapsed
build times.

## GitHub timings

Four-run medians, in seconds. Job time includes setup and teardown but excludes
time waiting for a runner. Python's test step includes installation and build.

| Job                       | Whole job | Configure | Build | Tests |
| ------------------------- | --------: | --------: | ----: | ----: |
| C++ Linux x64 Debug       |       183 |        12 |    70 |  60.5 |
| C++ Linux ARM64 Release   |     152.5 |        12 |  88.5 |  11.5 |
| C++ macOS Debug           |       244 |      32.5 |  70.5 |    91 |
| C++ macOS Release         |       238 |        31 |   114 |  26.5 |
| C++ Windows x64 Release   |     255.5 |        28 |   118 |  31.5 |
| C++ Windows ARM64 Release |     298.5 |        38 |   107 |  36.5 |
| C++ Linux without MLIR    |      49.5 |       6.5 |    24 |     2 |
| Python Linux x64          |     362.5 |         — |     — |   310 |
| Python Linux ARM64        |     269.5 |         — |     — |   219 |
| Python macOS              |     389.5 |         — |     — |   312 |
| Python Windows x64        |       529 |         — |     — | 411.5 |

In the latest sampled run, Python separates as follows:

| Platform    | First package build | Eight pytest runs combined | Complete test step |
| ----------- | ------------------: | -------------------------: | -----------------: |
| Linux x64   |                90.4 |                      233.2 |                343 |
| Linux ARM64 |                74.5 |                      128.4 |                216 |
| macOS       |               192.4 |                      189.4 |                405 |
| Windows x64 |               125.7 |                      266.3 |                426 |

Windows Python is the critical test job in all four samples. Each job builds the
package once, then reuses it for minimum/current dependencies across Python
3.11–3.14. Windows C++ reports 148 cache hits from 159 compilation requests;
Windows Python reports 104 hits, six misses, and additional non-cacheable or
other requests. The earlier zero-request cache problem is no longer current.

Sources:
[latest CI run](https://github.com/munich-quantum-toolkit/core/actions/runs/34167552843),
[Windows Python job](https://github.com/munich-quantum-toolkit/core/actions/runs/34167552843/job/101881573336),
[Windows C++ job](https://github.com/munich-quantum-toolkit/core/actions/runs/34167552843/job/101881573258).

## Findings and disposition

1. **QDMI discovery omitted 176 parameterized cases.** Post-build discovery ran
   before runtime manifests were copied. Deferred discovery in
   `cmake/PackageAddTest.cmake` now runs after the complete build. The
   baseline's 3,811 entries included three placeholders matching no real cases;
   corrected baseline discovery produced 3,984 entries. Current discovery
   includes the actual client parameterizations.
2. **Sampling allocated unused DD tables.** `qco::sample` now starts its owned
   package at zero qubits; `prepare` and dynamic allocation grow it as needed.
   In the isolated experiment, 768 small sample calls fell from 7.96 to 1.98
   seconds. All 1,093 Python tests passed; serial wall time fell from 28.9 to
   10.6 seconds and four-worker time from 15.6 to 6.0 seconds. Restoring the
   original constructor restored the slower result. Existing static-width,
   dynamic allocation, argument-binding, seeded, and zero-shot cases remain.
3. **QDMI rewrote unchanged headers.**
   [QDMI #537](https://github.com/Munich-Quantum-Software-Stack/QDMI/pull/537)
   uses content-preserving generation. Core pins that fix. Reconfiguration now
   preserves all eight generated device headers and leaves Ninja with no work;
   the baseline dirtied 15 actions.
4. **Stub generation had no shared compiler cache.**
   [Workflows #462](https://github.com/munich-quantum-toolkit/workflows/pull/462)
   applies the existing sccache policy and checks for nonzero compile requests.
   PRs read the cache and main writes it. The baseline stub job spent 313 of 319
   seconds building. MinSizeRel remains unchanged.
5. **Wheels contained duplicate versioned libraries.** CMake now omits
   VERSION/SOVERSION only for wheel builds. Native installations retain their
   versioning. The local wheel fell from about 119 MiB to 55 MiB and contains
   one DDSIM library. `check-wheel-contents` passes without the duplicate-file
   exemption. A relocated wheel passes QDMI tests and a CMake consumer that
   opens the bundled devices.
6. **Release builds forced full debug information.** The unconditional `-g`
   option is removed. CMake still supplies debug information in Debug and
   RelWithDebInfo configurations. Isolated compilation probes without full debug
   information saved 19–30%; no whole-build saving is inferred from them.
7. **The compiler test unity unit limited parallelism.** Only that target now
   disables unity. The controlled probe took 25.6 seconds for unity versus 19.7
   seconds with three source files compiled concurrently; all 171 compiler tests
   passed. Other unity settings are unchanged.
8. **Python scheduling remains within each test session.** Nox keeps its
   sequential current/minimum environments and existing build reuse. Pytest's
   `--numprocesses=auto` supplies test parallelism. A separate wheel matrix and
   wheel-installation mode are unnecessary and are not part of this change.
9. **Cheap verifier cases paid process startup costs.** QC and QCO IR now use
   one CTest entry per binary, retaining all 348 and 500 cases in GoogleTest
   XML. Both binaries pass five shuffled repetitions. Other binaries retain
   their discovery and process isolation; no numerical assertions were removed.
10. **Lifecycle tests relied on large workloads staying busy.** A test QIR
    program calls a test-owned barrier through an in-process function pointer.
    Submission uses the public device API and tests link the actual shared
    device. No test hooks, friendships, or test-specific build targets are
    present in production devices. Tests assert BUSY, unavailable results,
    timeout, and blocking cancellation/free before releasing the barrier. Real
    sampling and state-vector integration tests remain.

## Validation boundary

The implementation passes the full Release CTest suite: 3,142 entries, including
both grouped verifier binaries, with one intentional skip. All 848 grouped cases
pass and retain individual XML results. The full changed-file C++ lint check
reports zero findings. A broader header scan found existing DD/Common header
warnings outside this diff. Wheel compatibility was validated on Python
3.11–3.14 with current and minimum dependencies. The normal source-building
`tests-3.14` and `minimums-3.14` Nox sessions also pass with pytest parallelism.
The relocated wheel passes 247 QDMI cases and a downstream CMake build and
execution. QDMI's companion fix passed its Release build and CTest suite with
existing read-only skips before the added generation test was removed at the
maintainer's request. Shared workflow checks, including actionlint and zizmor,
pass.

These are local results. The native presets used for implementation validation
differ from the controlled baseline's no-cache/no-IPO profiling configuration;
there is no claimed whole-suite C++ speedup. Linux wheel relocation does not
establish manylinux, macOS, or Windows packaging correctness. Hosted cache hit
rates and platform wheels remain CI validation tasks.

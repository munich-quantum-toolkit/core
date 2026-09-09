# Optimized release builds

Status: native Linux SDK and Core LTO/BOLT validated locally; SDK publication
and hosted matrix validation pending. Original baseline: `eb67c001a`,
2026-09-08. GCC 13.3, Linux AArch64, LLVM/MLIR 23.1.0 assertion-enabled portable
SDK, nanobind 3.0.1, CPython 3.14.7.

## Result

- Apply section garbage collection to the DDSIM device and benchmark executable
  on ELF platforms in optimized configurations. The local wheel shrinks from
  62,468,761 to 44,196,908 bytes (29.3%) while retaining its runtime and SDK
  files.
- Select the explicit assertion-free SDK for CD wheel builds, including their
  pull-request checks. Keep ordinary C++ and Python CI assertion-enabled.
- Reuse the shared GitHub-backed sccache setup for wheels, forward its cache
  settings through cibuildwheel's Linux container boundary. Select GNU ld
  through `cmake.define.CMAKE_LINKER_TYPE=BFD` for BOLT release builds; mold
  2.42.0 produced invalid relocation symbol indices in full-LTO SDK tools and
  the Core compiler extension. `readelf` reproduced the malformed indices before
  BOLT ran; relinking `mlir-opt` with GNU ld removed them and passed
  instrumentation, training, optimization, and validation.
- Remove the obsolete MSVC `/Zm10` limit.
- Enable full LTO for Linux and macOS Core release wheels, including the CI
  wheel builds that validate CD. Clang uses full LTO explicitly rather than
  CMake's ThinLTO default; GCC retains full IPO and Windows wheel IPO is
  disabled. Requested wheel IPO fails configuration if unsupported. Keep
  nanobind's binding optimization defaults.
- With full Core LTO in addition to section GC, the local wheel is 42,732,712
  bytes, using the original native SDK.
- SDK variants contain native archives on every platform. Core enables LTO for
  its own code; BOLT runs on the final Linux wheel binaries. SDK image updates
  are independent of Core cibuildwheel updates, and macOS uses the runner
  default Xcode.
- Split the SDK's space-separated `LLVM_DEFINITIONS` into CMake arguments before
  adding definitions. Otherwise the explicit C++ ABI definition absorbs
  subsequent flags, and nanobind rejects the installed wheel at import.
- Linux BOLT profiles and rewrites Core's DD library/binding, compiler binding,
  DDSIM device, and benchmark executable. The SDK supplies BOLT and its runtime
  without rewriting its own tools. A shared SDK helper restores the original
  binary if instrumentation, training, optimization, or validation fails. Use
  `-lite` to rewrite only functions covered by the profile. Core uses
  `llvm-strip`, repairs wheels, regenerates RECORD hashes, and validates the
  repaired archive. GNU `strip` broke the rewritten SDK executable at startup in
  the local check.

## Simplified SDK builds

The SDK uses ordinary native Release builds with assertions selected per
variant. Remove SDK LTO, GCC fat-object/archive flags, special native-link
flags, serialized LTO links, exact Xcode selection, and Core image overrides.
Use `llvm-strip` for Linux/macOS tools and archives. The SDK pins the manylinux
2.28 image tag `2026.08.04-1` from cibuildwheel 4.2.0; Core follows
cibuildwheel's own defaults. Linux still ships BOLT tooling for Core's final
binaries.

Core LTO optimizes Core's own objects but cannot optimize across a native SDK
archive boundary. BOLT can still optimize the SDK code retained in the final
Core binaries. Native archives avoid compiler-specific LTO IR compatibility
requirements; the normal target, C++ ABI, and runtime-library requirements
remain.

Mold 2.42.0 has an upstream regression in emitted relocations for named local
symbols.
[Fix `635956d`](https://github.com/rui314/mold/commit/635956d3b7c53d72c3fb70fd084443671395d20d)
restores local-symbol classification when choosing the symbol-table index. The
tiny upstream assembly reproducer fails locally on ARM64: a relocation to `str`
names `_start` instead. `--discard-none` and `--no-relax` do not fix it. This
can occur without LTO, although the retained native `mlir-opt` link does not
trigger it. The fix is newer than the latest release, 2.42.0.

A local mold 2.42.0 build with only that upstream fix passes the reproducer.
Relinking the same full-LTO `mlir-tblgen` reduces `readelf` bad-symbol-index
reports from 26 to zero. BOLT then fails to relocate an ADR in the non-simple
`p_ere` function. Linking with the patched mold plus `--no-relax` passes
instrumentation, training, optimization, LLVM stripping, and validation. Keep
released mold for native SDK links and BFD for Core BOLT links rather than
shipping a patched linker and another workaround flag. Re-evaluate after the
mold fix is released and the ARM64 BOLT relaxation path is supported. Raw SDK
logs: `mold-local-reloc-results.json`, `mold-lto-tblgen-results.json`,
`mold-fixed-bolt.log`, and `mold-fixed-no-relax-bolt.log` in `build/lto-bolt/`.

The full native SDK builds with mold and passes LLVM stripping and the SDK
workload. Its exact archive is 305,403,648 bytes (291 MiB), 81.5% smaller than
the fat-LTO archive. The relocated installation passes a GCC 13.3 consumer of
the GCC 14.2-built libraries, plus BOLT success and failure recovery with BFD.
The host-default mold consumer runs successfully but reproduces the relocation
bug during BOLT; the integration test now selects BFD explicitly. The SDK itself
continues to build with mold.

A fresh Core full-LTO/BFD build passes BOLT on all five targets, LLVM stripping,
wheel repair, and repaired-wheel training. The 43,638,800-byte wheel is 9.6%
larger than the fat-SDK wheel (39,808,921 bytes). Its installed CMake consumer
creates a driver session, and 1,184 Python tests pass with one optional
`qirrunner` module skipped.

The same twelve-process, CPU-19 held-out protocol compares the two repaired
wheels with no concurrent builds. Both retain Core LTO and BOLT:

| Workload              | Fat-LTO SDK | Native SDK | Change |
| --------------------- | ----------: | ---------: | -----: |
| Vector import/export  |    1.745 ms |   1.746 ms | +0.04% |
| Matrix multiplication |    2.793 ms |   2.802 ms | +0.34% |
| OpenQASM to QCO       |    5.381 ms |   5.549 ms | +3.12% |
| Qiskit import/export  |    2.652 ms |   2.697 ms | +1.70% |

The DD differences are within process variation; the compiler paths show a small
cost from dropping SDK LTO. Process-median IQRs are 0.019-0.034 ms. These ARM64
workloads support the simpler native SDK policy, without implying identical
performance across platforms or workloads. macOS default-Xcode builds and the
full hosted matrix require new validation.

SDK evidence: `native-build.log`, `native-finalize.log`, and
`native-host-integration-bfd.log` under `build/lto-bolt/`. Core evidence:
`native-sdk-wheel-build.log`, `native-sdk-bolt.log`, `native-wheel-tests.log`,
`native-consumer.log`, `native-sdk-bench.log`, and
`native-sdk-bench-results.json` under `build/release-optimization/`.

## Earlier fat-LTO links and distribution experiment

The SDK retains GCC LTO IR for Core while building its own tools through native
links. Compile with `-flto=auto -ffat-lto-objects`, use GCC archive tools, and
link SDK executables/shared libraries with `-fno-lto`. Disable CMake IPO for
this build because its GCC flags would select slim LTO objects. Preserve the GCC
partitioning restriction needed by Core's BOLT links. The SDK no longer
BOLT-rewrites its tools or provisions extra swap for that step.

The updated installation test explicitly checks native linking and full-LTO
linking against the assertion-free SDK, then BOLT success and byte-for-byte
recovery after failed training. The native check rejects the previous slim-LTO
SDK. The assertion-enabled SDK still passes its native consumer test.

A controlled local link comparison uses the same fat archives, one warm-cache
run per mode, four CPUs, 14 GiB RAM, and a 16 GiB swap allowance. Both binaries
pass `--version`; both `mlir-opt` variants pass canonicalization.

| Tool          | Native link | Full-LTO link | Native peak RSS | LTO peak RSS |
| ------------- | ----------: | ------------: | --------------: | -----------: |
| `mlir-opt`    |     10.21 s |      983.02 s |        2.70 GiB |     9.98 GiB |
| `mlir-tblgen` |     0.279 s |       15.18 s |        68.4 MiB |    204.4 MiB |

These single-link measurements isolate repeated linking cost, not total hosted
build time or SDK tool runtime performance. The complete local fat SDK build
used 16 CPUs and took about 14.2 minutes; its final executable link finished
66.6 seconds after its final compile. That build is not comparable to the hosted
x86-64 runner. Raw data: `fat-link-results.json`, `fat-link-bench.log`, and
`fat-full.ninja_log` in the SDK worktree's `build/lto-bolt/`.

The local distribution candidate includes the complete LLVM/MLIR library and
header groups, CMake exports, runtime libraries, BOLT/runtime, FileCheck, and
selected command-line tools. Its generated build graph contains 3,384 compile
commands and 22 executable links, compared with 3,694 and 118 in the full SDK.
This removes 8.4% of compilation commands and 81.4% of executable links. These
are graph counts, not measured clean-build time reductions. The published
component inventory remains unchanged. The selected installation preserves every
header and library path, all nine shared runtime libraries, and resolving
symlinks. It omits 95 executable names and the optional `opt-viewer` Python
files, reducing installed bytes from 5,349,035,347 to 3,936,371,338 (26.4%).
Examples of removed tools include `lli`, `llvm-profdata`, `llvm-symbolizer`, and
MLIR language servers. Those omissions change the SDK's public tool surface;
build-graph savings alone do not justify that decision.

With identical production zstd settings (`-19 --long=31 --threads=16`), the full
archive is 1,650,594,629 bytes (1.54 GiB); the selected archive is 1,512,140,071
bytes (1.41 GiB). The download saving is 132 MiB (8.4%). Both exact archives
pass native/full-LTO linking, BOLT rewriting, and failure recovery after
relocation. Headers and non-BOLT libraries are byte-identical; CMake exports
reflect the selected tools, and BOLT archives change with their compiled install
prefix. The experiment reuses the full build, so it does not establish a
clean-build wall-time saving.

Keep the full SDK inventory: native linking already removes the expensive LTO
link tail, while an 8.4% smaller download is a modest return for dropping useful
tools. Revisit component selection if installation/download size becomes a
concrete constraint. Raw evidence: `distribution-graph.json`,
`distribution-install-results.json`, `fat-finalize.log`, and
`fat-dist-finalize.log` in the SDK worktree's `build/lto-bolt/`.

A fresh Core full-LTO build against the fat SDK passes BOLT rewriting of all
five targets, LLVM stripping, wheel repair, and repaired-wheel training. The
39,808,921-byte wheel passes 1,184 Python tests (one optional `qirrunner` module
skipped), plus the installed CMake consumer and driver session check. An initial
pytest invocation omitted the virtual environment from `PATH` and failed two CLI
tests; the corrected invocation passes the full suite.

Repeating the twelve-process held-out protocol below, with no concurrent builds,
confirms that retaining SDK IR preserves Core performance on these workloads:

| Workload              | Earlier LTO/BOLT SDK | Fat SDK, Core LTO/BOLT |
| --------------------- | -------------------: | ---------------------: |
| Vector import/export  |             1.750 ms |               1.748 ms |
| Matrix multiplication |             2.797 ms |               2.796 ms |
| OpenQASM to QCO       |             5.380 ms |               5.384 ms |
| Qiskit import/export  |             2.656 ms |               2.656 ms |

The differences are below 0.2%, within process-median IQRs of 0.015-0.050 ms.
This is a bounded regression check, not evidence of an additional runtime gain.
Raw Core logs: `fat-sdk-wheel-build.log`, `fat-sdk-bolt.log`,
`fat-wheel-tests-final.log`, `fat-consumer.log`, `fat-sdk-bench.log`, and
`fat-sdk-bench-results.json` under `build/release-optimization/`.

## Earlier matched SDK and BOLT evaluation

The Linux AArch64 build uses the pinned manylinux 2.28 container, GCC 14.2.1,
LLVM/MLIR 23.1.0 with assertions disabled, full SDK/Core LTO, and GNU ld. The
SDK was built in full; the three rewritten SDK tools were relinked with GNU ld
using those archives. Its relocated installation passes the existing CMake
consumer test, including LTO linking and BOLT failure recovery.

Both wheel variants come from the same final links. Both use LLVM stripping,
`wheel pack`, and auditwheel repair. BOLT changes the wheel from
**43,964,801 to 39,808,676 bytes (9.5% smaller)**. The broader default rewrite
grew it to 61,809,745 bytes; reusing the old text section still produced
59,049,810 bytes. Select `-lite` rather than those larger layouts.

Twelve fresh processes per variant rotate execution order on CPU 19 of the DGX
Spark, with one BLAS/OpenMP thread and no concurrent builds. Each process runs
nine samples per workload and discards the first. CPython 3.14.7 loads the
repaired wheel contents. Training uses GHZ/W states, H/T/CX circuits, QIR JIT,
and QFT generation; the timed workloads use dense numerical inputs and rotation
circuits. NumPy and Qiskit checks validate the timed results.

| Held-out workload                      | LTO baseline | LTO + BOLT |      Change |
| -------------------------------------- | -----------: | ---------: | ----------: |
| Vector import/export, 16,384 elements  |     1.771 ms |   1.746 ms | 1.4% faster |
| Complex 32 by 32 matrix multiplication |     2.869 ms |   2.788 ms | 2.8% faster |
| OpenQASM to QCO, 480 gates             |     5.562 ms |   5.374 ms | 3.4% faster |
| Qiskit import/export, 480 gates        |     2.704 ms |   2.639 ms | 2.4% faster |

These are small, workload-specific gains. Process-median IQRs in the final run
are 0.008-0.036 ms. An earlier layout trial had a large timing shift and is not
used for this table. The earlier GCC 13/native SDK measurements below do not
isolate SDK LTO or assertion removal and are not directly comparable.

The final wheel passes 1,184 Python tests; one optional `qirrunner` module is
skipped because it is not installed. The training separately checks QIR JIT
through DDSIM. The packaged CMake consumer, driver session creation, CLI, and
QC/jeff generation pass. Build, rewrite, repaired-wheel validation, and
numerical checks run against installed artifacts. Local evidence covers ARM64
and the stable Python ABI; x86-64, macOS, Windows, and the free-threaded ABI
still need hosted validation.

Raw data and logs are under `build/release-optimization/`:
`bolt-bench-results.json`, `lite-bench.log`, `bolt-wheel-lite.log`, and
`final-lite-tests.log`. SDK logs are under the toolchain worktree's
`build/lto-bolt/`.

## Assertion-enabled versus optimized release

On 2026-09-09, rebuild Core at `3d28fb593` with GCC 14.2.1 in the same pinned
manylinux ARM64 image, using the assertion-enabled SDK from toolchain CI run
`34286319228`. Compare its native SDK, disabled Core IPO, and no BOLT against
the previously validated assertion-free SDK/Core full-LTO plus BOLT wheel. This
measures the combined release policy, not assertion removal alone. The
twelve-process held-out protocol above uses the same Python environment and CPU,
with no concurrent build or BOLT process.

| Workload              | Assertions enabled | Assertions off + LTO + BOLT | Time reduction |
| --------------------- | -----------------: | --------------------------: | -------------: |
| Vector import/export  |           1.916 ms |                    1.742 ms |           9.1% |
| Matrix multiplication |           3.101 ms |                    2.786 ms |          10.2% |
| OpenQASM to QCO       |           6.649 ms |                    5.374 ms |          19.2% |
| Qiskit import/export  |           3.050 ms |                    2.639 ms |          13.5% |

Process-median IQRs are 0.021-0.039 ms. Numerical and round-trip checks pass.
Raw results: `build/release-optimization/matched-assertions-comparison.log` and
`matched-assertions-results.json`. These measurements apply to Linux ARM64 and
these workloads; macOS ThinLTO performance is unmeasured.

## Hosted build limits

Toolchain run `34286319228` killed both assertion-free Linux jobs during
`mlir-opt` BOLT instrumentation. A local four-CPU container with 14 GiB RAM and
no swap reproduces SIGKILL with `memory.events: oom_kill 1`; disabling BOLT
threads also fails. With the same RAM and a 16 GiB swap allowance,
instrumentation completes in 424 seconds, uses 9.5 GiB peak swap, and records no
OOM events. Training, optimization, the remaining two SDK tools, and post-strip
training pass under the same limits; those stages take another 63 seconds with a
4.9 GiB peak RSS and no swap. Raw logs are `limited-instrument.log`,
`sequential-instrument.log`, `swap-instrument.log`, and `swap-validation.log`
under the SDK worktree's `build/lto-bolt/`. An intervening workflow added 16 GiB
swap. Run `34317328143` then built the ARM64 SDK in 4h22m and passed
installation tests, but x86-64 still timed out at six hours. Its final compile
finished at 07:47:22 UTC; the remaining 114 executable links ended at 11:38:28,
a 3h51m tail. BOLT instrumentation started at 11:38:48 and had not finished when
the job timed out at 12:19. Native SDK links and removal of SDK BOLT address
both costs; their x86-64 hosted result remains pending.

The assertion-free macOS job reaches 5,134 of 5,157 Ninja steps before the
six-hour timeout; repeated full-LTO LLVM tool links take several minutes each.
The SDK now uses ThinLTO, which enables LLVM's native Darwin link cache, while
retaining one link job for the 7 GB runner. Core and the SDK integration test
retain full LTO for their own objects. Apple ld64 processes ThinLTO and full-LTO
objects separately, reducing optimization across that boundary. A local Clang
23/LLD probe links a ThinLTO archive to a full-LTO consumer for ELF and Mach-O,
and runs the ELF result. Subsequently, hosted run `34317328143` built the
assertion-free macOS SDK in 3h12m and passed both installation tests. Runtime
performance on macOS remains unmeasured.

## Earlier native SDK runtime measurements

Compare Release, deployment mode, unity builds, shared Core libraries, split
nanobind bindings, and identical SDK/dependency revisions. LTO enables the
existing `ENABLE_IPO` option for all project-built targets; the SDK contains
native objects. The binding experiment removes `-Os` only from the DD and MLIR
binding translation units, leaving their linked libraries unchanged.

Twelve fresh processes per variant rotate execution order. Each process takes
nine samples per workload, discards the first, and reports the median. The table
shows medians across processes in milliseconds. NumPy checks DD numerical
results; Qiskit circuits compare equal after a round trip.

| Workload                                            | Baseline |   LTO | Binding `-O3` |
| --------------------------------------------------- | -------: | ----: | ------------: |
| Dense complex vector import/export, 16,384 elements |    1.877 | 2.076 |         2.003 |
| Dense complex 32 by 32 matrix multiplication        |    3.143 | 3.169 |         2.913 |
| OpenQASM to QCO default pipeline, 480 gates         |    7.032 | 7.841 |         6.949 |
| Qiskit import/export, 480 gates                     |    3.189 | 3.666 |         3.170 |

LTO slows the compiler cases by 11.5% and 14.9%. Binding `-O3` improves matrix
multiplication by 7.3%, but slows vector conversion by 6.7%; the compiler
changes are below 1.2%. These are bounded workload results, not broad platform
rankings. Full LTO is now enabled as the Core release-wheel policy. Recheck
performance with GCC 14 and AppleClang against the assertion-free SDK; this
earlier probe does not measure cross-module LTO through an LTO-built SDK. No
comparable cold-build timing was established.

Local recipes, raw samples, staged modules, and build logs are retained under
`build/release-optimization/`: `configure-args.json`, `bench.py`,
`bench-results.json`, and the `baseline`, `lto`, and `o3` directories.

## Packaging contract and opportunity

`src/CMakeLists.txt` exports the current C++ libraries together.
`python/mqt/core/_commands.py` and the CLI still advertise headers and CMake
configuration. Removing all Development components would break that surface;
removing only some libraries would also leave invalid imported target
references. The wheel's 61 header files and 17 metadata files occupy only about
0.17 MB compressed. Retaining this small SDK is currently the cheapest useful
policy.

The costly files are native code, including three consumers of static MLIR/LLVM
code: the Python compiler module, DDSIM device, and benchmark executable.
`add_mqt_python_binding` already enables ELF section garbage collection. The
other two final links omitted it even though compilation produces separate
function/data sections and the device hides static-library symbols.

| Stripped artifact    |           Before |  With section GC |
| -------------------- | ---------------: | ---------------: |
| DDSIM device         | 84,030,632 bytes | 62,634,976 bytes |
| Benchmark executable | 51,310,120 bytes | 24,204,064 bytes |

Keep QDMI devices as loadable libraries. Core #2229 makes the Client and driver
separate shared libraries with dynamic driver selection; #2231 stages their
installed runtime closure. Do not internalize that driver boundary. The current
DD and benchmark support libraries are each under 0.2 MB compressed; statically
replicating them is not an established size improvement.

A later, larger experiment could share a private compiler runtime among the
compiler module, DDSIM device, and CLI. It must measure the union of retained
code against the current per-consumer garbage collection, relocation/startup
cost, symbol isolation, and auditwheel/delvewheel repair. Simply making all LLVM
libraries shared would create a much wider distribution contract. Removing the
CLI would change a useful public interface and needs a separate product
decision.

## PGO and post-link investigation

[Astral's Ruff PGO work](https://github.com/astral-sh/ruff/pull/27570) uses
pinned training projects and a separate evaluation corpus. Its follow-ups report
gains on both [macOS ARM64](https://github.com/astral-sh/ruff/pull/27572) and
[Linux ARM64](https://github.com/astral-sh/ruff/pull/27574).
[ty](https://github.com/astral-sh/ty/pull/4213) demonstrates why interactive
work needs explicit training and why code growth needs measurement;
[uv](https://github.com/astral-sh/uv/pull/21001) balances multiple workload
families and keeps evaluation projects separate.

Use two coordinated stages for MQT:

1. Instrument the portable SDK and link it into a pinned Core training consumer.
   Exercise QC/QCO construction and rewriting, QASM/jeff/Qiskit import/export,
   QIR lowering/JIT, and representative DD workloads. Compiling Core C++ sources
   alone trains the host C++ compiler, not the MLIR library code used at
   runtime.
2. Rebuild the SDK with those profiles into native static archives, then profile
   and rebuild Core against that optimized SDK. Separate library PGO and Core
   PGO compose; final Core-only instrumentation cannot retroactively optimize
   native SDK objects. The assertion-free release archives now carry LTO code
   under the matched-compiler contract; ordinary CI archives retain native code.
   [LLVM's distribution guidance](https://llvm.org/docs/BuildingADistribution.html)
   explains the distribution tradeoff.

Keep the current SDK compilers initially: GCC on Linux can use
`-fprofile-generate`/`-fprofile-use`; AppleClang can use instrumentation and
matching `llvm-profdata` tools. GCC and LLVM profiles are different formats. Pin
source, compiler, architecture, ABI/assertion mode, and corpus revision with
each profile; fail on unexpected profile mismatch rather than treating stale
profiles as valid. Use native Linux x86-64/ARM64 and macOS ARM64 runners. Keep
ordinary Windows releases while profiling support matures.

Use deterministic, bounded training inputs from existing benchmark families,
including structured loops, dynamic control, target mapping, and QIR execution.
Hold out different families and external circuits rather than only changing
random seeds. Weight conversion, optimization, execution, and Python-boundary
work separately; avoid letting one long simulator case dominate the profile.
Evaluate compiler wall time, simulation throughput, startup, wheel size, and
release build cost, including both supported Python ABI modes.

[python-build-standalone](https://github.com/astral-sh/python-build-standalone/blob/b80276e3f79fe538eeb1f256319dc46f11dce96e/cpython-unix/build-cpython.sh)
shows how to use parallel instrumented training, but also records real BOLT
correctness workarounds. At this revision its AArch64 BOLT target is disabled
for a known rewriting failure. Those exclusions are evidence for maintaining a
correctness gate, not a skip list to copy into Core.

BOLT runs after full LTO in Linux release builds. It rewrites final ELF
executables and shared libraries, not the portable SDK's archive members. The
useful Core targets are the compiler extension, DDSIM device, and CLI. Preserve
symbols and link relocations until rewriting; test JIT, exceptions, registration
and loading, then strip, repair, and generate the final wheel metadata. GCC
needs the compatibility flags described in
[BOLT's README](https://github.com/llvm/llvm-project/blob/main/bolt/README.md).
Do not infer a gain from BOLT availability: Ruff's
[follow-up](https://github.com/astral-sh/ruff/pull/27588) found only 1.24%
additional wall-time improvement with roughly 4.4% wheel growth, and no
worthwhile ARM64 gain.

macOS release builds use full LTO. Compiler PGO remains a separate candidate.
LLVM also documents Darwin linker order files from dtrace profiles; they are a
separate layout experiment, not BOLT support. Apply order files at final links,
not to redistributable archives. Hosted profiling privileges and
deployment-target compatibility still need a native proof. Keep portable CPU
baselines; no `-march=native`, relaxed floating-point semantics, or new runtime
allocator is justified by this audit.

## Validation and remaining gates

The earlier native SDK wheel uses mold, full GCC LTO, and section GC, verified
in its compile and link commands. A local Clang 23 consumer verifies that wheel
mode uses `-flto=full` for both compilation and linking. 917 installed-wheel
tests pass after enabling full LTO across DD, benchmarks, QDMI, compiler
pipelines, Qiskit interchange, and CLI behavior. A CMake consumer finds the
wheel's installed package, links `MQT::CoreQDMI`, and creates a session. The
installed benchmark executable generates both QC and jeff output.

These are native Linux AArch64 checks using the existing assertion-enabled SDK.
They do not establish manylinux portability, other architectures, macOS/Windows
behavior, Python 3.15 free-threaded behavior, hosted cache hits, or the new
SDK's runtime gains. Core CI is configured to cover its supported matrix once
the new SDK archives are published. Toolchain #94, setup-mlir #255, and
workflows #464 must land in that dependency order before the Core integration
can be finalized.

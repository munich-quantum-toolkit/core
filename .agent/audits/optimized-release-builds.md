# Optimized release builds

Status: local Linux matrix complete. All 24 PGO configurations, four quiet
runtime cohorts, capped resource replays, and final package checks passed.

Earlier investigation baseline: `eb67c001a`, 2026-09-08. GCC 13.3, Linux
AArch64, LLVM/MLIR 23.1.0 assertion-enabled portable SDK, nanobind 3.0.1,
CPython 3.14.7.

## Local Linux matrix

The follow-up experiment fixes Core at
`706fd8f95e38c29451d97e88cfdf6022a55020fe` and LLVM at `llvmorg-23.1.0`. It does
not change production workflows or the assertion-enabled development SDK. Raw
commands and measurements are under `build/linux-optimization/`; the experiment
manifest records tool identities. The PGO results and final repeat below contain
the current runtime recommendations. Earlier experiments are labeled separately.

The pinned manylinux image contains `manylinux-install-clang`, but its published
versions stop at Clang 22.1.8. The official LLVM 23.1.0 ARM64 archive was
downloaded and SHA-256 verified
(`cfb31bfc713ef453248bf5bd026312f838ad6c52c25623e987cb6a340f3050d4`). Its
compiler needs glibc 2.34 and cannot run inside manylinux 2.28. Running that
compiler on the host against the extracted manylinux sysroot works for a small
ThinLTO C++ consumer. That consumer runs inside the original container and needs
at most `GLIBC_2.17` and `GLIBCXX_3.4.9`. Viable full SDK/wheel configurations
have since passed validation, as detailed below. The downloaded LLD also needs
libicu 70; its loader path is limited to the compiler wrapper. No Clang compiler
is built from source. LLVMgold is built from the fixed LLVM source to match
Clang 23.1.0; the host's 23.1.1 plugin is not a final matrix input.

Patched mold 2.42.0 passes the named-local-symbol emitted-relocation regression
inside manylinux. Both manylinux and host experiments use the same built linker,
SHA-256 `ca7152e6edaf5f8fddd793cf0029c794fa771c3167aed1f56a2a7805e193490a`. An
earlier source-staging error left stock mold in two diagnostic builds; those
artifacts are excluded and rebuilt. Source inspection and the regression test
now gate use of the linker.

GCC 13 and 14 silently drop `-mcpu=native` on this heterogeneous machine. GCC 14
accepts explicit `-mcpu=cortex-x925`; GCC 13 only supports older tuning models,
so its native host experiment must identify that limitation. Clang 23 resolves
native targeting to Cortex-X925. The recorded driver commands, not the requested
flag alone, determine whether a native row is valid.

The runner records command failures, GNU time, descendant RSS and Docker cgroup
peak memory. Descendant RSS includes separate Ninja process groups but can count
shared pages repeatedly; cgroup memory is the aggregate measure for containers.
Exploratory builds overlap and are not controlled build-time comparisons.
Evaluation uses twelve rotating fresh processes on CPU 19, one thread, and no
concurrent builds. Per-workload medians, IQRs, paired bootstrap intervals, and
regressions above 3% are retained. Ranking weights workload families equally.
LLVM reads the process affinity mask when choosing its thread pool size;
evaluation pins affinity before launching the Python processes. NumPy
statevector checks cover optimized QCO, jeff round trips, and DD simulation at
all three held-out sizes.

The initial GCC C++/MLIR check passes 3,226 tests with one deliberate SC job-ID
skip. Each completed BOLT training/validation run passes 1,184 Python tests with
one optional external `qirrunner` skip. Built-in QIR execution is exercised. The
reused GCC SDK did not disable hot/cold function splitting; a new native SDK
with the same setting as the full-LTO SDK replaces it for controlled
comparisons. Earlier GCC results remain diagnostic.

### Portable runtime screen, 2026-09-10

The first controlled evaluation contains 384 fresh-process samples: twelve
rotating rounds for 32 artifacts. Build processes were paused throughout.
`evaluation/comparison-20260910-020511.{json,csv,ranking.json}` records exact
artifact hashes, workload medians, IQRs, paired bootstrap intervals, and flags
for individual workload regressions above 3%. The table uses equal weight per
workload family; lower latency ratios are better, relative to M1/BFD/plain.
These are screening results; the subsequent C++ matrix checks passed.

| SDK/Core LTO; linker  | Plain latency | BOLT latency | BOLT wheel MiB | BOLT uncompressed MiB |
| --------------------- | ------------: | -----------: | -------------: | --------------------: |
| GCC off/off; BFD      |        1.0000 |       0.9622 |           45.4 |                 140.5 |
| GCC full/full; BFD    |        0.9945 |       0.9602 |           40.0 |                 124.3 |
| Clang off/off; LLD    |        0.9639 |       0.9294 |           39.2 |                 135.3 |
| Clang off/full; LLD   |        0.9313 |       0.9031 |           38.5 |                 127.4 |
| Clang thin/thin; LLD  |        0.8782 |       0.8562 |           42.3 |                 136.3 |
| Clang thin/full; LLD  |        0.8863 |       0.8617 |           41.4 |                 133.7 |
| Clang full/full; LLD  |        0.8620 |       0.8378 |           41.1 |                 131.4 |
| Clang full/full; mold |             — |       0.8366 |           41.3 |                 131.4 |

Clang full/full with LLD and BOLT has a 95% interval of [0.8356, 0.8408]
relative to the GCC baseline. Its ratio to the mold equivalent is 1.0015, with
interval [0.9983, 1.0047]. Prefer LLD for the next stage because this is within
uncertainty and avoids the additional plugin/linker dependency.

The GCC-only subset is retained in `evaluation/portable-gcc.*`. Full/full with
BFD/BOLT is within uncertainty of full/full with mold/BOLT. A direct paired
comparison (`evaluation/gcc-bfd-tie.*`) finds off/off BFD/BOLT 0.21% slower than
full/full BFD/BOLT, with interval [0.036%, 0.483%]. Carry full/full BFD forward
as the measured GCC finalist, but report its SDK build cost against this very
small runtime difference. This screening result alone does not justify requiring
compiler-matched full-LTO SDKs for GCC releases.

Matched effects are recorded in
`evaluation/portable-effects.results.{json,csv}`. Core-only GCC LTO with a
native SDK increases balanced latency by 1.30% after BOLT (95% interval: 1.01%
to 1.39%) and regresses several compiler workloads above 3%. Adding full SDK LTO
recovers that loss. With Clang and a ThinLTO SDK, forcing Core from ThinLTO to
full LTO increases latency by 0.64% after BOLT; full SDK LTO then improves the
full-Core-LTO result by 2.77%. These effects explain why a blanket Core-only LTO
rule is insufficient.

The selected full/full Clang and GCC BOLT candidates have no workload above 3%
regression versus M1/BFD/plain. The matched-effects CSV still flags local
regressions relative to each immediate predecessor, including small parsing and
QIR cases in the compiler comparison. Clang M5/mold relaxation improves the BOLT
score by 0.38%; relaxation checks for the full/full default-linker finalists
remain separate from the native-CPU comparison.

## Earlier result (before the Linux matrix)

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

## Earlier PGO and post-link investigation

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

## Earlier validation and publication gates

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

## Local ARM64 experiment: provisioning findings

The new experiment fixes Core at `706fd8f95e38c29451d97e88cfdf6022a55020fe` and
LLVM/MLIR at `llvmorg-23.1.0`. All runtime comparisons use Python 3.14.7 and the
same dependency lock. The first portable comparison completed twelve rotating
fresh-process rounds without concurrent builds. Native CPU, host, optimization
level, and compiler-PGO comparisons are now complete, as detailed below.

- The pinned manylinux image contains `manylinux-install-clang`, but its
  published version list stops at 22.1.8. The downloaded upstream Clang 23.1.0
  ARM64 archive has SHA-256
  `cfb31bfc713ef453248bf5bd026312f838ad6c52c25623e987cb6a340f3050d4`. Its
  executables require newer glibc than manylinux 2.28. Running that compiler on
  the host against the exported manylinux sysroot works: compiler resource
  headers, GCC 14 C++ headers/runtime, startup objects, archive tools, and LLD
  are selected explicitly. A linked C++ ThinLTO consumer runs inside the pinned
  container; its highest required versions are GLIBC 2.17 and GLIBCXX 3.4.9.
  Completed portable wheels also pass auditwheel repair to manylinux 2.28.
- The downloaded compiler does not supply LLVMgold. Build the plugin from the
  same LLVM 23.1.0 sources; using the host's LLVM 23.1.1 plugin would mix
  compiler revisions. No Clang source build is involved.
- mold is pinned at 2.42.0 with upstream relocation fix
  `635956d3b7c53d72c3fb70fd084443671395d20d`. The regression must inspect
  emitted local-symbol relocations after patching and rebuilding. A source
  directory nested inside another Git checkout can make `git apply` skip the
  intended patch: apply the extracted patch with `patch -p1` and inspect the
  result. Initial wheels produced by the accidentally unpatched linker are
  excluded.
- Both GCC 14 in manylinux and host GCC 13 silently remove `-mcpu=native` on
  this heterogeneous CPU. GCC 14 accepts explicit `-mcpu=cortex-x925`. GCC 13
  rejects that CPU and Cortex-X4, but accepts
  `-march=armv9.2-a -mtune=cortex-x3`. That older scheduling model is a
  documented host-GCC limitation, not equivalent to Clang's detected X925.
- Instrumentation probes confirm that SDK-only and Core-only compiler PGO can be
  selected independently through LTO. GCC uses process-specific profile
  directories and matching `gcov-tool`; Clang uses `%m-%p.profraw` and matching
  `llvm-profdata`. An installed MLIR consumer linked against four instrumented
  SDK archives successfully produces a merged Clang profile inside manylinux.
  Rebuilding those archives with the profile and linking without instrumentation
  runtime flags also passes. GCC native/full-LTO SDK consumers and all three
  Clang SDK archive modes run successfully inside manylinux.

The library-only SDK rebuild must keep `LLVM_BUILD_TOOLS=ON` while building only
explicit archive targets. Turning it off omits imported `llvm-as` and `llvm-dis`
targets from the generated exports, even when their native executables were
copied into the SDK. Core's C++ tests exposed this packaging defect in the N2
GCC row. Regenerating the exports restores those targets without rebuilding or
changing the static archives; the failed configuration is retained separately
from its retry.

Training and held-out benchmarks use separate deterministic circuit structures.
The QIR JIT fixture tracks an exact basis-state result and scales surviving QIS
calls; cancelling H/H loops would otherwise measure nearly constant JIT setup.
BOLT package records include the input wheel and benchmark hashes and reject a
benchmark change during training. Correctness smoke runs under build load are
excluded from runtime comparisons.

The regular C++ matrix rejects M2/mold with either relaxation setting. Two QIR
runtime test executables fail to link because the native SDK's `CSE.cpp.o` and
GCC's LTO output both define the `MemoryEffects::Write` singleton and its guard.
BFD links the same objects and passes the tests. Python wheel validation alone
misses this failure; the affected mold variants are excluded from viable
rankings. No multiple-definition suppression or additional linker patch is used.

### Initial portable runtime comparison

The score is the geometric mean of latency ratios with equal weight per workload
family, including startup. Each family combines its small, medium, and large
cases. Intervals resample the twelve paired measurement rounds (2,000 bootstrap
draws); lower ratios are better. These initial screening results preceded the
completed C++ gates and final repeat. The later PGO section gives the current
recommendations.

| Change, holding other settings fixed             | Latency ratio | 95% bootstrap interval |
| ------------------------------------------------ | ------------: | ---------------------: |
| GCC Core LTO, M1/BFD to M2/BFD                   |        1.0060 |          1.0041–1.0103 |
| GCC SDK LTO, M2/BFD to M3/BFD                    |        0.9885 |          0.9847–0.9912 |
| Clang Core full LTO, M4/LLD to M5/LLD            |        0.9662 |          0.9641–0.9680 |
| Clang SDK full LTO, M5/LLD to M8/LLD             |        0.9257 |          0.9225–0.9278 |
| Clang SDK ThinLTO, M5/LLD to M7/LLD              |        0.9518 |          0.9491–0.9542 |
| Clang Core ThinLTO to full LTO, M6/LLD to M7/LLD |        1.0093 |          1.0063–1.0109 |
| BOLT on M3/BFD                                   |        0.9655 |          0.9635–0.9680 |
| BOLT on M8/LLD                                   |        0.9719 |          0.9700–0.9758 |
| Patched mold versus BFD, M3 with BOLT            |        0.9986 |          0.9961–1.0010 |
| Patched mold versus LLD, M8 with BOLT            |        0.9985 |          0.9954–1.0019 |

Full SDK/Core LTO supplies the runtime finalists for subsequent experiments. BFD
and LLD remain the compiler-specific choices for those experiments because their
BOLT results are indistinguishable from mold in this comparison. GCC's full-LTO
gain over no LTO is small enough that build cost remains material; Clang shows a
larger SDK-LTO effect. Individual regressions above 3% are retained in the raw
comparison CSV and are not hidden by the balanced score.

Raw records are under `build/linux-optimization/evaluation`: the complete sample
file is `comparison-20260910-020511.json`, and isolated effects are in
`portable-effects.results.json` and `portable-effects.results.csv`.

### Host GCC and Cortex-A53 workarounds

Ubuntu GCC 13/BFD emits Cortex-A53 erratum 843419 veneers in the large MLIR
extension. BOLT rejects these by default. `H1-gcc-plain` retains the workaround;
`H1-gcc-bolt` is unavailable under that row's portable-CPU contract. Its failed
instrumentation log is retained.

The H2 host GCC builds explicitly require ARMv9.2 and therefore already exclude
Cortex-A53. Their BOLT invocations use `--drop-cortex-a53-843419-veneers`,
scoped by the recorded compiler and CPU flags. The same condition applies to
later native GCC PGO artifacts. This option is never applied to portable
manylinux or H1 artifacts. H2's measured BOLT effect therefore includes veneer
removal as well as profile-directed layout changes. The original failed H2
attempt remains in `rejected-stage-attempts`.

### Installed C++ consumer boundaries

The installed consumer now checks DD state-vector semantics as well as QDMI
session allocation. It uses the producer compiler. GCC 13 requests the old
`getVector` symbol, whereas Clang 23 and GCC 14 LTO export the constrained
specialization with newer mangling. The GCC 14 no-LTO library also exports the
old alias. Keep compiler matching explicit for the optimized C++ packages.

On glibc 2.28 the consumer must additionally link `Threads::Threads`: the fixed
Core revision does not propagate pthread, causing QDMI session allocation to
throw `std::system_error`. A debugger trace and a pthread-preload probe isolated
this from the optimization changes. The consumer now links the dependency
explicitly and passes without preload. Source revisions remain fixed. Evidence
is under `build/linux-optimization/consumer-abi-probe`; successful checks
require the current consumer source hashes in each consumer input record.

### CPU tuning, host toolchains, and optimization levels

The second quiet comparison completed 12 rotating rounds across 39 variants (468
fresh-process samples). All retained baseline variants passed their regular C++
suites and installed Python/CMake checks. The Python suite reports 1,184 passed
and one optional `qirrunner` import skip; QIR execution is exercised by the C++
runtime/JIT suites and the scalable workloads.

| Change                                   | Latency ratio | 95% bootstrap interval |
| ---------------------------------------- | ------------: | ---------------------: |
| gcc Core CPU tuning (plain)              |        0.9917 |          0.9888–0.9939 |
| gcc additional SDK CPU tuning (plain)    |        1.0092 |          1.0058–1.0127 |
| clang Core CPU tuning (plain)            |        1.0158 |          1.0125–1.0189 |
| clang additional SDK CPU tuning (plain)  |        0.9855 |          0.9830–0.9890 |
| gcc host toolchain/environment (plain)   |        1.0554 |          1.0525–1.0581 |
| gcc host CPU tuning (plain)              |        0.9985 |          0.9958–1.0013 |
| gcc portable O3 to O2 (plain)            |        1.0443 |          1.0407–1.0483 |
| gcc native O3 to O2 (plain)              |        1.0446 |          1.0413–1.0478 |
| clang host toolchain/environment (plain) |        1.0022 |          0.9989–1.0051 |
| clang host CPU tuning (plain)            |        1.0001 |          0.9961–1.0047 |
| clang host CPU tuning (BOLT)             |        1.0013 |          0.9967–1.0062 |
| clang portable O3 to O2 (plain)          |        1.0132 |          1.0090–1.0176 |
| clang native O3 to O2 (plain)            |        1.0146 |          1.0096–1.0201 |
| M3-bfd relaxation (plain)                |        0.9997 |          0.9974–1.0023 |
| M3-mold relaxation (plain)               |        0.9986 |          0.9960–1.0022 |
| M8-lld relaxation (plain)                |        0.9980 |          0.9954–1.0022 |

O3 remains the setting for all four PGO finalists. O2 regresses the balanced
score before and after BOLT, with intervals excluding equality. The individual
regressions include matrix multiplication and small parsing cases for Clang, and
several compiler workloads for GCC. Exact per-workload ratios and all
regressions above 3% are retained in `native-option-effects.results.csv`.

Native CPU tuning does not supply a measurable aggregate gain for either host
compiler. The Clang native builds regress some DD simulation cases, despite
aggregate parity. The host GCC builds are approximately 5.5% slower than the
portable GCC builds; GCC 13.3 versus 14.2.1 and the runtime environment differ,
so that delta is not a CPU-tuning result. Host Clang 23.1.1 and portable Clang
23.1.0 remain indistinguishable in this comparison.

Relaxation is also indistinguishable at the aggregate level. PGO keeps the
recorded no-relaxation baseline, full SDK/Core LTO, and BFD/LLD. The four suites
use M3/BFD, M8/LLD, H2/GCC, and H2/Clang, each with benchmark-only and
tests-plus-benchmarks training and separate SDK-only, Core-only, and combined
PGO. Native PGO remains an experiment, not evidence that native CPU flags are
always beneficial.

Raw samples: `comparison-20260910-063504.json`; paired effects:
`native-option-effects.results.json` and `.csv`; exact PGO selections:
`build/linux-optimization/pgo-finalists.json`.

The M3/BFD/BOLT and M8/LLD/BOLT baselines additionally pass the installed C++
consumer, all scalable semantic workloads, and the same 1,184 Python tests
inside the pinned manylinux 2.28 image using CPython 3.14.7. These checks use
the exact installed wheel files, not a rebuild. Records are under
`build/linux-optimization/manylinux-runtime`.

### PGO training configuration checks

The first manylinux GCC instrumentation build was rejected before training:
constructing an MLIR context crashed in `ThreadLocalCache`, both on the host and
inside the pinned container. Its CMake logs identify a harness error:
`-fprofile-prefix-path=/experiment/.` did not match canonical source paths. GCC
emitted a warning during compiler-flag probes; `-Werror` then made the PIC and
semantic-interposition probes fail. The resulting SDK lacked `-fPIC` despite
`LLVM_ENABLE_PIC=ON`.

The harness now normalizes the prefix to `/experiment`, clears failed cached
checks, and requires the successful PIC probe before building an SDK. The new
configuration passes both C and C++ PIC probes. Failed builds, training output,
cache, compiler-check logs, and the debugger evidence remain available under
`build/linux-optimization/rejected-stage-attempts/M-PGO-gcc-profile-prefix` and
the `M-PGO-gcc-*-crash` records. These failed artifacts supply no performance
samples. The corrected build passes Python training and all 4,069 C++ tests
using the same compiler and sources.

The GCC merge checks also caught a separate histogram problem. The unmodified
GCC 13 `gcov-tool` subtracts from an already negative accumulated top-N total
when that accumulated profile is its first input. The APInt division histogram
had 25,134 raw executions but only 4,288 in the merged total, making a retained
value count of 7,456 inconsistent. Its SDK compilation correctly rejected that
profile. A check of every histogram found 15 affected totals in benchmark-only
training and 163 in tests-plus-benchmarks training; the corresponding host GCC
PGO artifacts are excluded and rebuilt.

The corrected fold puts a fresh raw profile first and the accumulated profile
second. A raw process directory containing an already saturated histogram must
be the initial accumulator; the current inputs have at most one such directory.
The harness rejects multiple such directories rather than guessing an order. It
retains the common-filename workaround and the matching, unmodified compiler
tool. Both corrected datasets preserve the sum of absolute raw execution totals
for every top-N/indirect-call histogram (78,164 and 100,630 histograms,
respectively), and the isolated APInt compile passes. This follows the
[signed-total handling in GCC's merge implementation](https://github.com/gcc-mirror/gcc/blob/releases/gcc-13.3.0/libgcc/libgcov-merge.c).
Raw comparisons and a reproducer are under
`rejected-stage-attempts/H-PGO-gcc-value-profile`. SDK and Core objects are
cleaned before the replacement builds so an unchanged profile pathname cannot
reuse objects compiled with the earlier data.

The corrected manylinux GCC datasets also pass this invariant: 84,645
benchmark-only and 107,340 tests-plus-benchmarks histograms, with no mismatches.
The latter contains one saturated raw histogram, handled as the initial
accumulator by the same merge procedure.

Exploratory Core build costs retain their actual CPU settings in each command
record. Once both Clang PGO suites finished, subsequent manylinux GCC Core
builds used an eight-CPU quota instead of four to use the available machine. The
in-flight benchmark-trained SDK-only Core build retained four CPUs. Compare
these costs with their resource settings; the isolated finalist replays use the
same four-CPU limit for every candidate.

`disk-inventory.json` and `.csv` report retained directory footprints, including
per-variant staging files, SDKs, training data, build trees, and caches. They
are snapshots, not historical peak disk usage. PGO variants share build
directories; the configuration table marks those shared current footprints
explicitly. The host compiler cache is shared across builds and cannot be
attributed to one variant. Nested directories overlap, and filesystem allocation
counts shared or reflinked extents per path rather than exclusive physical
storage. Dedicated finalist cache replays provide separate empty/populated-cache
measurements.

Memory columns retain the accounting method: sampled process-tree RSS can count
shared pages more than once, while cgroup memory includes page cache. Swap is
sampled from a cgroup where one was recorded; uncapped host exploration has no
swap measurement and is marked unavailable, not zero. The isolated finalist
replays record cgroup memory and swap for both host and container builds. For
Docker commands, outer GNU time and process-tree RSS describe the Docker client;
use cgroup memory for the container workload. Short cache probes also record
Bash time inside the container.

### PGO results and final repeat

All 24 PGO configurations passed their regular C++ suites, installed Python
suite, semantic benchmarks, and matching-compiler CMake consumer. Each has
separate plain and freshly BOLT-optimized artifacts. The PGO comparison contains
672 samples (56 variants, 12 rounds); the final repeat contains 216 samples (18
variants, 12 rounds). Together with the portable and native screens, these four
quiet cohorts contain 1,740 fresh-process measurements of 109 distinct viable
artifacts. Four packaged M2/mold variants are retained as rejected
C++-incompatible diagnostics and receive no runtime ranking.

The following PGO table uses the first PGO cohort. Each cell is the balanced
latency ratio before/after BOLT, relative to M3/BFD without PGO or BOLT. Lower
is better. Full per-workload latency, throughput, startup, medians, IQRs, and
95% bootstrap intervals are in the comparison CSV. Absolute latency and
throughput intervals were added without changing the earlier relative results;
`relative-summary-preservation.json` verifies those columns are unchanged.

| Environment       | Training    |   Core PGO only |    SDK PGO only |            Both |
| ----------------- | ----------- | --------------: | --------------: | --------------: |
| Portable GCC      | bench       | 0.9495 / 0.9282 | 0.9070 / 0.8932 | 0.8620 / 0.8606 |
| Portable GCC      | tests-bench | 0.9541 / 0.9247 | 0.9363 / 0.9164 | 0.8615 / 0.8498 |
| Portable Clang    | bench       | 0.8344 / 0.8125 | 0.8080 / 0.7921 | 0.7798 / 0.7691 |
| Portable Clang    | tests-bench | 0.8409 / 0.8174 | 0.8056 / 0.7922 | 0.7781 / 0.7689 |
| Host-native GCC   | bench       | 0.9885 / 0.9647 | 0.9364 / 0.9293 | 0.8794 / 0.8772 |
| Host-native GCC   | tests-bench | 0.9947 / 0.9632 | 0.9729 / 0.9545 | 0.8838 / 0.8719 |
| Host-native Clang | bench       | 0.8393 / 0.8147 | 0.8051 / 0.7922 | 0.7825 / 0.7734 |
| Host-native Clang | tests-bench | 0.8430 / 0.8187 | 0.8083 / 0.7948 | 0.7790 / 0.7695 |

The final repeat supports these selections. Ratios below use its own M3/BFD
plain reference; they are not ratios between independent cohorts.

| Finalist (full/full LTO, O3, BOLT) | Latency ratio |  95% interval | Wheel MiB | ELF MiB | SDK archive MiB |
| ---------------------------------- | ------------: | ------------: | --------: | ------: | --------------: |
| M-PGO-gcc-tests-bench-both-bolt    |        0.8520 | 0.8491–0.8549 |      36.3 |   107.8 |          2222.1 |
| M-PGO-clang-bench-both-bolt        |        0.7683 | 0.7657–0.7701 |      39.1 |   122.3 |           805.2 |
| H-PGO-gcc-tests-bench-both-bolt    |        0.8692 | 0.8664–0.8710 |      36.9 |   112.9 |          2268.9 |
| H-PGO-clang-tests-bench-both-bolt  |        0.7673 | 0.7645–0.7685 |      41.5 |   124.0 |           805.0 |

Portable Clang with benchmark-only combined PGO is the preferred portable
configuration: 23.17% lower balanced latency than the GCC full-LTO baseline (95%
interval: 22.98%–23.43%), and 9.00% lower than Clang full-LTO/BOLT without PGO
(8.71%–9.29%). Adding regular tests to compiler training changes its ratio by
only −0.23% (−0.53% to +0.15%). Prefer benchmark-only compiler training; regular
tests still gate correctness and participate in BOLT training.

The same portable configuration is the recommendation for maximum measured local
performance. Host-native Clang with tests-plus-benchmarks PGO has a ratio of
0.9986 to it (0.9947–1.0017), but confirmed regressions above 3% on all three DD
simulation sizes. The host-native recipe is tested and available; the data does
not justify requiring native CPU flags. This cross-environment comparison also
changes the compiler distribution; the earlier H1/H2 comparison isolates CPU
tuning and found no aggregate benefit.

For GCC, adding tests to benchmark training lowers final latency by 1.05%
(portable) and 1.20% (host-native), with intervals excluding equality. For
host-native Clang it lowers latency by 0.48%, also excluding equality in the
repeat. Those are the subsidiary compiler finalists. The simpler host-GCC
benchmark-only/plain variant is close in the first cohort but has confirmed
control-flow regressions; aggregate uncertainty alone does not erase them.

All four selected finalists have no workload regression above 3% against the
common GCC baseline in the repeat. Every matched comparison retains both point
regressions above 3% and regressions whose entire 95% interval exceeds 3%.
Consult `pgo-effects.results.csv`, `pgo-finalist-ties.results.csv`, and
`final-effects.results.csv` when choosing for a particular workload.

Exploratory build costs below include the SDK profile-use rebuild, Core wheel
build, separate C++ test build, and five BOLT train/rewrite/validation commands.
These builds overlap other work and use different recorded quotas; they are cost
records, not controlled build-speed comparisons. Instrumentation build costs and
Python/C++ training/profile-merge costs are separate columns in
`configuration-summary.csv`; instrumentation is shared by both datasets.

| Finalist                          | SDK use rebuild s | Core wheel s | C++ build s | MLIR link s | BOLT total s |
| --------------------------------- | ----------------: | -----------: | ----------: | ----------: | -----------: |
| M-PGO-gcc-tests-bench-both-bolt   |             420.4 |        296.4 |       521.7 |        69.9 |        181.0 |
| M-PGO-clang-bench-both-bolt       |             332.3 |        361.8 |       419.8 |       130.4 |        238.7 |
| H-PGO-gcc-tests-bench-both-bolt   |             520.2 |        264.1 |       411.0 |        72.7 |        223.4 |
| H-PGO-clang-tests-bench-both-bolt |             377.3 |        363.7 |       548.4 |       125.2 |        238.7 |

All 23 SDK archive sizes and SHA-256 hashes were verified against their actual
contents, including the patched mold binary. The 109 evaluated environments use
the same CPython 3.14.7 executable and the same 40 installed distribution
versions; all four cohorts use the identical held-out benchmark source hash.
`input-consistency.json` and `sdk-archives/verified-contents.json` record these
checks. Sizes separate compressed packages, uncompressed package bytes, and ELF
bytes; compression is never treated as a runtime optimization.

The recommended Clang wheel and the portable GCC finalist additionally pass the
matching CMake consumer, all held-out semantic workloads, and 1,184 Python tests
inside the original manylinux 2.28 image. BOLT recovery on the selected Clang
binary passes all four injected failure stages (training, missing profile,
rewrite, and validation), restoring its original bytes and permissions and
successfully executing it afterward.

### Finalist resource replays

Dedicated sequential replays use CPUs 16–19, a four-CPU quota, 16 GiB RAM, and
an additional 16 GiB swap. These are local runner-limit experiments, not hosted
CI results. Cgroup peak memory includes child processes and page cache; RSS and
sampled swap are retained separately. The MLIR extension link uses the selected
SDK archives, Core objects, and compiler-PGO dataset. The portable Clang build
was restored to its benchmark profile in the same build paths before replay;
original package artifacts and timing records were preserved.

Full LTO has no persistent linker cache. The paired times below mean first and
repeated link invocations with the OS page cache left intact. Each worker count
has one pair, so this is a resource/capacity screen, not a statistically powered
build-speed ranking.

| Finalist                     | LTO workers | First / repeat s | Peak RAM GiB | Peak sampled swap GiB |
| ---------------------------- | ----------: | ---------------: | -----------: | --------------------: |
| M-PGO-clang-bench-both       |           1 |      80.3 / 80.2 |         4.72 |                  0.00 |
| M-PGO-clang-bench-both       |           2 |      64.9 / 65.4 |         6.23 |                  0.00 |
| M-PGO-clang-bench-both       |           4 |      57.3 / 57.5 |         6.45 |                  0.00 |
| M-PGO-gcc-tests-bench-both   |           1 |    212.7 / 211.7 |         3.56 |                  0.00 |
| M-PGO-gcc-tests-bench-both   |           2 |    117.9 / 118.5 |         3.62 |                  0.00 |
| M-PGO-gcc-tests-bench-both   |           4 |      69.5 / 70.0 |         3.80 |                  0.00 |
| H-PGO-clang-tests-bench-both |           1 |    116.5 / 116.4 |         5.05 |                  0.00 |
| H-PGO-clang-tests-bench-both |           2 |      95.4 / 96.0 |         6.74 |                  0.00 |
| H-PGO-clang-tests-bench-both |           4 |      80.7 / 80.2 |         6.96 |                  0.00 |
| H-PGO-gcc-tests-bench-both   |           1 |    204.8 / 204.1 |         3.51 |                  0.00 |
| H-PGO-gcc-tests-bench-both   |           2 |    113.5 / 113.8 |         3.68 |                  0.00 |
| H-PGO-gcc-tests-bench-both   |           4 |      67.0 / 67.2 |         3.85 |                  0.00 |

Compiler-cache probes replay LLVM Support's `CommandLine.cpp` and Core's
OpenQASM frontend unity translation unit with the selected flags and profiles.
Each uses an isolated ccache 3.7.7 directory, first empty and then populated.
The two object hashes must match; original build objects and depfiles are
restored afterward. These probes measure representative compilation/cache
stages, not a complete SDK rebuild under the cap. The SDK probe for portable
Clang explicitly selects its benchmark profile without reconfiguring the shared
SDK build tree. The exact original and replay commands are retained. A
one-second exit hold keeps each short-lived cgroup available for its final
peak-memory sample; an inner Bash time record excludes that hold from the
compiler times below.

| Finalist                     | Probe | Empty / populated cache s | Peak RAM GiB | Cache MiB |
| ---------------------------- | ----- | ------------------------: | -----------: | --------: |
| M-PGO-clang-bench-both       | sdk   |               0.54 / 0.01 |         0.06 |      0.41 |
| M-PGO-clang-bench-both       | core  |               2.54 / 0.01 |         0.32 |      1.03 |
| M-PGO-gcc-tests-bench-both   | sdk   |               1.01 / 0.01 |         0.25 |      1.09 |
| M-PGO-gcc-tests-bench-both   | core  |               3.52 / 0.01 |         0.60 |      3.00 |
| H-PGO-clang-tests-bench-both | sdk   |               0.79 / 0.01 |         0.06 |      0.40 |
| H-PGO-clang-tests-bench-both | core  |               3.85 / 0.01 |         0.31 |      1.10 |
| H-PGO-gcc-tests-bench-both   | sdk   |               0.91 / 0.01 |         0.25 |      1.15 |
| H-PGO-gcc-tests-bench-both   | core  |               3.19 / 0.01 |         0.58 |      3.14 |

Fresh BOLT profiles were collected under the same limits from the actual five
wheel binaries. Each rewrite runs its training validation, followed by regular
Python tests and the held-out semantic workloads on the complete rewritten
stage. The original raw wheel SHA-256 is checked before extraction.

| Finalist                     | Five BOLT stages s | Peak RAM GiB | Peak sampled swap GiB |
| ---------------------------- | -----------------: | -----------: | --------------------: |
| M-PGO-clang-bench-both       |              202.8 |         5.66 |                  0.00 |
| M-PGO-gcc-tests-bench-both   |              164.4 |         4.72 |                  0.00 |
| H-PGO-clang-tests-bench-both |              181.7 |         5.81 |                  0.00 |
| H-PGO-gcc-tests-bench-both   |              170.4 |         5.04 |                  0.00 |

`resource-summary.{json,csv}` contains these measurements. All exact commands,
limits, cache statistics, sizes, and logs are under `replays/`. Exploratory
build costs in `configuration-summary.csv` retain their actual concurrency;
these isolated records are the evidence for runner memory/resource limits.

Use four LTO workers with one large link at a time for these finalists under
four CPUs and 16 GiB RAM. Four workers were fastest in both observed runs for
every compiler. The highest link peak was 6.96 GiB; the highest BOLT peak was
5.81 GiB. No replay used swap. These observations do not establish a smaller RAM
limit or a full SDK build limit: compilation used representative probes, and
standalone SDK tools were not BOLT-optimized.

The cache harness retains the initial timing attempts under
`rejected-stage-attempts/cache-before-exit-hold`. The corrected probes use the
existing Bash timer because the manylinux image has no `/usr/bin/time`, retain
the cgroup for one second after the command, and restore container-owned outputs
inside the container. All eight corrected probes have a real direct warm-cache
hit and identical cold/warm object hashes; restored object hashes also match
their backups.

### Final package and replay checks

All 24 replayed MLIR extension outputs pass the held-out semantic workloads.
Validation copies use package-local `lib`/`lib64` search paths; original link
outputs are unchanged. The source guide recommends four LTO workers under the
tested limit while retaining the measured one- and two-worker alternatives.

Wheel ZIP creation and manylinux repair were replayed separately after semantic
validation finished. Every resulting ELF entry is byte-identical to the
corresponding canonical, tested wheel. The following quiet timings therefore
measure packaging rather than changes in generated code. SDK archive packing
uses recorded `zstd -T1 -3` commands and has a separate timing column in the
configuration summary.

| Finalist                     | Wheel ZIP s | manylinux repair s |
| ---------------------------- | ----------: | -----------------: |
| M-PGO-clang-bench-both       |        3.10 |               5.14 |
| M-PGO-gcc-tests-bench-both   |        3.10 |               5.24 |
| H-PGO-clang-tests-bench-both |        3.08 |     not applicable |
| H-PGO-gcc-tests-bench-both   |        3.10 |     not applicable |

The final `configuration-summary.{json,csv}` covers all 113 packaged artifacts:
109 validated and four rejected M2/mold variants. It separates SDK/Core LTO, CPU
flags, linker, PGO scope/training, BOLT, exploratory build and profile costs,
archive/wheel/ELF sizes, and retained disk footprints. `measurement-inventory`
indexes timed stages; `linker-identity.json` checks all 113 ELF linker records,
including the host distribution's `Ubuntu LLD 23.1.1` identification.

The results bundle contains raw measurements, exact commands, toolchain hashes,
helper scripts, the implementation patch, this audit, source-build instructions,
and a file-hash manifest. Large SDK archives, wheels, raw compiler profiles, and
build trees remain at the paths recorded in its JSON files. No production
release workflow, hosted experiment, macOS build, or compiler bootstrap is part
of this local phase.

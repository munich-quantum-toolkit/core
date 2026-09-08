# Optimized release builds

Status: measured changes implemented; optimized SDK publication and hosted
validation pending. Baseline: `eb67c001a`, 2026-09-08. GCC 13.3, Linux AArch64,
LLVM/MLIR 23.1.0 assertion-enabled portable SDK, nanobind 3.0.1, CPython 3.14.7.

## Result

- Apply section garbage collection to the DDSIM device and benchmark executable
  on ELF platforms in optimized configurations. The local wheel shrinks from
  62,468,761 to 44,196,908 bytes (29.3%) while retaining its runtime and SDK
  files.
- Select the explicit assertion-free SDK for release CI and wheels. Keep
  assertion-enabled Debug coverage and Windows support.
- Reuse the shared GitHub-backed sccache setup for wheels, forward its cache
  settings through cibuildwheel's Linux container boundary, and select the
  bundled mold through `cmake.define.CMAKE_LINKER_TYPE`.
- Remove the obsolete MSVC `/Zm10` limit.
- Enable full LTO for Core release wheels, including the CI wheel builds that
  validate CD. Clang uses full LTO explicitly rather than CMake's ThinLTO
  default; GCC/MSVC retain their full IPO modes. Requested wheel IPO fails
  configuration if unsupported. Keep nanobind's binding optimization defaults.
- With full Core LTO in addition to section GC, the local wheel is 42,732,712
  bytes. SDK archives remain native until compiler compatibility is coordinated.

## Runtime measurements

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
   native SDK objects. SDK LTO remains off for redistributable archives, as
   explained in
   [LLVM's distribution guidance](https://llvm.org/docs/BuildingADistribution.html).

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

BOLT is a separate experiment after PGO. It rewrites final ELF executables and
shared libraries, not the portable SDK's archive members. The useful Core
targets are the compiler extension, DDSIM device, and CLI. Preserve symbols and
link relocations until rewriting; test JIT, exceptions, registration and
loading, then strip, repair, and generate the final wheel metadata. GCC needs
the compatibility flags described in
[BOLT's README](https://github.com/llvm/llvm-project/blob/main/bolt/README.md).
Do not infer a gain from BOLT availability: Ruff's
[follow-up](https://github.com/astral-sh/ruff/pull/27588) found only 1.24%
additional wall-time improvement with roughly 4.4% wheel growth, and no
worthwhile ARM64 gain.

For macOS, compiler PGO and final-target ThinLTO are the first candidates. LLVM
also documents Darwin linker order files from dtrace profiles; they are a
separate layout experiment, not BOLT support. Apply order files at final links,
not to redistributable archives. Hosted profiling privileges and
deployment-target compatibility still need a native proof. Keep portable CPU
baselines; no `-march=native`, relaxed floating-point semantics, or new runtime
allocator is justified by this audit.

## Validation and remaining gates

The final local wheel uses mold, full GCC LTO, and section GC, verified in its
compile and link commands. A local Clang 23 consumer verifies that wheel mode
uses `-flto=full` for both compilation and linking. 917 installed-wheel tests
pass after enabling full LTO across DD, benchmarks, QDMI, compiler pipelines,
Qiskit interchange, and CLI behavior. A CMake consumer finds the wheel's
installed package, links `MQT::CoreQDMI`, and creates a session. The installed
benchmark executable generates both QC and jeff output.

These are native Linux AArch64 checks using the existing assertion-enabled SDK.
They do not establish manylinux portability, other architectures, macOS/Windows
behavior, Python 3.15 free-threaded behavior, hosted cache hits, or the new
SDK's runtime gains. Core CI is configured to cover its supported matrix once
the new SDK archives are published. Toolchain #94, setup-mlir #255, and
workflows #464 must land in that dependency order before the Core integration
can be finalized.

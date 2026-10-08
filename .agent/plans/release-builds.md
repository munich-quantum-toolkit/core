# Optimized release builds

Status: implemented; hosted platform checks remain.

## Scope and decisions

`pyproject.toml` owns portable wheel settings. Linux uses the manylinux static
Clang 22.1.8 distribution, LLD, and CMake IPO (ThinLTO). macOS uses Apple Clang
and ThinLTO. Windows retains its compiler settings. All wheels set `DEPLOY=ON`;
local release builds default to native CPU tuning and IPO without interpreting
`CI` as a deployment request. Bindings use nanobind's `NOMINSIZE`.

The compiler and the assertion-free LLVM/MLIR 23.1.2 SDK are separate inputs.
The manylinux installer does not yet supply Clang 23. GCC 15 is competitive, but
Clang delivers the strongest DD results and the smallest wheel in the measured
candidates. Prebuilt SDK archives remain native inputs; LTO operates within each
final binary, not across shared-library boundaries.

Concrete DD operations and package policy isolation are prerequisites. The DD
library can participate in IPO without compatibility aliases. The installed
CMake package preserves the consumer's build policy. One public-API consumer
checks all 14 concrete DD operations in native tests and installed wheels; the
PGO/BOLT pipeline reuses it after wheel repair.

Full LTO remains an experiment: it improved compiler probes by 1–2% but did not
improve DD simulation. ThinLTO is CMake's supported default. FatLTO stores
native code and compiler IR together and is unnecessary for these shared
libraries.

Mold 3 linked the GCC 14 and GCC 15 wheels successfully. Its official ARM64
binary requires glibc 2.30; an unchanged source build passed on glibc 2.28.
Adding Rust and a linker build to normal wheel provisioning is unnecessary with
the selected Clang/LLD distribution. Local GCC users can select mold 3 directly.
The native CI workflows also select mold 3 explicitly, giving release source
builds a qualified linker for IPO.

## Performance evidence

Five rotated process samples on one pinned DGX Spark ARM64 CPU compared portable
manylinux_2_28 wheels. All used the same SDK, `NOMINSIZE`, and concrete DD API.
GCC variants used mold 3; Clang used matching LLD. No compilation ran during
measurement. Each probe checks its numerical or compiler result before timing.

The probes cover a seeded 16-qubit vector roundtrip, 504 H/T/CX operations on 14
qubits, and QCO/QIR compilation of 768 H/T/CX operations on 24 qubits. Median
milliseconds:

| Compiler and LTO | Vector | Simulation |   QCO |    QIR |
| ---------------- | -----: | ---------: | ----: | -----: |
| GCC 14, off      |  19.84 |    1279.18 | 6.023 | 27.453 |
| GCC 14, on       |  19.08 |    1235.68 | 6.154 | 28.368 |
| GCC 15, on       |  19.26 |    1238.06 | 5.855 | 26.685 |
| Clang 22, Thin   |  17.94 |    1197.43 | 5.966 | 26.592 |

The Clang wheel was 50.9 MB versus 52.9 MB for GCC 14 without LTO. GCC 14 LTO
reproducibly slowed the compiler probes while improving DD. This establishes a
compiler-dependent regression, not its precise microarchitectural cause; local
hardware performance counters were unavailable. These ARM64 probes do not
establish performance on other CPUs or workloads.

## Validation

Local qualification passed 12 installed-consumer configurations: Clang 22, Clang
23, and GCC 15 LTO producers with GCC 13/Clang 23 consumers in Debug and
Release. Each library exports the 14 concrete operations as strong symbols. The
manylinux candidates passed auditwheel repair and strict abi3audit. The Clang 22
wheel passed 1,896 Python tests with two environment-specific skips; CLI checks
used the activated wheel environment. Native prerequisite tests and stub
validation are recorded in the DD API plan.

Run `python test/cmake/check_installed.py` with the wheel installed to check
package settings, its device helper, and the DD operations. Cibuildwheel runs
this before the Python suite. The same C++ source is a native CTest target so
local and hosted C++ lint use its actual compile settings.

Hosted Linux x86-64, macOS, and Windows checks remain pending. The PGO/BOLT
pipeline retains a matched profiling toolchain and still needs full release
qualification; its LTO setting is owned by CMake IPO.

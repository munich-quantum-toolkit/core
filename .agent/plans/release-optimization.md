# Release wheel optimization

Status: ThinLTO alignment prepared; restack and full PGO/BOLT qualification
pending.

Core #2678 provides assertion-free LLVM 23.1.2 SDK selection, released Setup
1.5.0 and Workflows 2.5.1 pins, and the Windows packaging cleanup. This branch
preserves those changes. SDK #94 supplies Linux BOLT tools. Only the two wheel
workflows pin the pending Workflows #464 revision for persistent compiler
caching.

Core #2476 owns SDK/Core PGO, MLIR test training, Linux BOLT, and installed
wheel checks. Its release checks use the current public APIs and do not require
the exception migration in #2545.

Linux uses matched Clang, LLD, compiler-rt, and llvm-profdata packages from the
manylinux 2.28 container, ThinLTO, PGO, and BOLT. macOS uses Apple Clang,
ThinLTO, PGO, and deployment target 13.3. CMake owns Core target IPO in both the
instrumented and final builds. The SDK rebuild explicitly disables LTO. Profiled
LLVM libraries must retain the installed SDK's assertion, EH, and RTTI settings.
Windows uses its normal compiler.

The manylinux packages currently supply Clang 21. The static Clang 22 bundle
lacks the profile runtime and llvm-profdata required by this pipeline, so it
cannot replace the matched packages.

The native SDK and wheel checks from Phase 1 do not qualify this optimization
pipeline. The full SDK/Core PGO build, Linux BOLT processing, repaired-wheel
checks, and macOS qualification remain pending. This refresh runs focused source
and release-check validation without the expensive optimization build.

The prerequisite DD ABI change exposes concrete operations such as
`dd::getVector(state)` instead of constrained member-template exports. The
installed checks use the shared `test/cmake/installed_consumer` fixture with
Clang and GCC against the repaired wheel. Stack that prerequisite before
qualification; installed GCC and Clang consumers must both pass against the
final optimized wheel.

ThinLTO alignment and consumer reuse validation pass Python compilation and
repository lint. The shared fixture's installed-consumer checks on the restacked
branch remain pending.

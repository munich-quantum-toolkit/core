# Release wheel optimization

Status: refreshed on current main; full PGO/BOLT qualification pending.

Core #2678 provides assertion-free LLVM 23.1.2 SDK selection, released Setup
1.5.0 and Workflows 2.5.1 pins, and the Windows packaging cleanup. This branch
preserves those changes. SDK #94 supplies Linux BOLT tools. Only the two wheel
workflows pin the pending Workflows #464 revision for persistent compiler
caching.

Core #2476 owns SDK/Core PGO, MLIR test training, Linux BOLT, and installed
wheel checks. Its release checks use the current public APIs and do not require
the exception migration in #2545.

Linux uses matched Clang packages from the manylinux 2.28 container, full LTO,
PGO, and BOLT. macOS uses Apple Clang, ThinLTO, PGO, and deployment target 13.3.
Windows uses its normal compiler. Profiled LLVM libraries must retain the
installed SDK's assertion, EH, and RTTI settings.

The native SDK and wheel checks from Phase 1 do not qualify this optimization
pipeline. The full SDK/Core PGO build, Linux BOLT processing, repaired-wheel
checks, and macOS qualification remain pending. This refresh runs focused source
and release-check validation without the expensive optimization build.

Refresh validation passes repository lint, C++ lint, source-archive checks,
release training, and the GCC 13 installed consumer against a fresh current-main
baseline wheel. The Clang 23 consumer cannot link `dd::Edge::getVector` from
that GCC 13 wheel because the compilers use different constrained-template
symbol names. Keep this check and resolve compiler compatibility during
qualification.

Core #2715 found that GCC 14 LTO drops DD constrained-template ABI aliases
required by GCC 13 consumers and excludes the GNU wheel DD target from CMake
IPO. This Clang pipeline passes `-flto` explicitly, so its DD ABI compatibility
needs independent qualification; the CMake IPO exclusion does not cover it.

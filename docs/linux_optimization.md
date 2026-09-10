# Local Linux optimization experiments

This recipe evaluates release builds in a separate directory. It does not
replace an assertion-enabled development SDK. The measured ARM64 results are
recorded in the optimization audit. The selected portable Clang wheel also gives
the best supported recommendation for local performance: native CPU tuning
supplies no aggregate gain in these workloads.

Use the same Core revision, LLVM/MLIR sources, Python interpreter, and Python
dependencies throughout a comparison. The recorded experiment uses Core
`706fd8f95e38c29451d97e88cfdf6022a55020fe`, LLVM/MLIR `llvmorg-23.1.0`, and
CPython 3.14.7. Clang is downloaded or installed from binary packages, never
built from source.

## Toolchain and CPU selection

For a host Linux build, use the installed Clang 23 and its matching archive,
linker, and profile tools. Record the full version: the experiment's host
compiler is 23.1.1, whereas its downloaded portable compiler is 23.1.0. GCC
similarly differs between the host (13.3) and manylinux (14.2.1). These
differences prevent attributing the whole host-versus-manylinux result to CPU
tuning.

For portable builds, the pinned manylinux 2.28 image is
`quay.io/pypa/manylinux_2_28_aarch64:2026.08.04-1`. Its
`manylinux-install-clang` version list stops at 22.1.8. The official Clang
23.1.0 ARM64 executables require glibc 2.34; they cannot execute inside this
image. Run the downloaded compiler in a compatible host userspace against an
exported manylinux sysroot. Keep the sysroot's C++ headers, GCC runtime, startup
objects, and system libraries together. Pass both `--sysroot` and
`--gcc-install-dir` explicitly, and use the downloaded LLD and archive tools. Do
not copy newer host runtime libraries into the sysroot.

The compiler wrapper must add the linker selection only to link invocations;
adding `--ld-path` to compile-only commands breaks CMake's warning-as-error
feature probes. Verify the wrapper with a C++ program using the standard library
and LTO, then execute the result in the original manylinux image. Verify the
complete wheel with auditwheel afterward. The experiment records its downloaded
archive checksum and wrapper commands with the raw results.

For CPU tuning, apply the same CPU flags to both SDK libraries and Core. Clang's
`-mcpu=native` resolves to Cortex-X925 on the measured machine. Both GCC
versions silently discard `-mcpu=native` on this heterogeneous CPU: GCC 14 needs
explicit `-mcpu=cortex-x925`; GCC 13 only supports the approximate
`-march=armv9.2-a -mtune=cortex-x3`. Check `-###` output and predefined feature
macros instead of trusting the requested flag. CPU-specific artifacts retain a
local Linux platform tag and are not portable release wheels.

## Build the SDK libraries and Core

The following variables describe separate source, build, and install paths. Use
an absolute path for each. `NATIVE_SDK` is an assertion-free SDK built from the
same LLVM sources, supplying native `llvm-tblgen`, `mlir-tblgen`, and BOLT
tools. Its executables are reused; the runtime comparison rebuilds the SDK's
LLVM/MLIR static libraries. Building all SDK tools with full LTO adds
substantial cost without changing Core's retained library code.

```sh
uv python install 3.14.7
export PYTHON="$(uv python find 3.14.7)"
export OPT_ROOT="$PWD/build/local-release"
export LLVM_SOURCE="/absolute/path/to/llvm-project-23.1.0.src"
export CORE_SOURCE="/absolute/path/to/core"
export NATIVE_SDK="/absolute/path/to/assertion-free-native-sdk"
export RELEASE_SDK="$OPT_ROOT/sdk"
export CPU_FLAGS=""
export CC=/usr/lib/llvm-23/bin/clang
export CXX=/usr/lib/llvm-23/bin/clang++
export AR=/usr/lib/llvm-23/bin/llvm-ar
export RANLIB=/usr/lib/llvm-23/bin/llvm-ranlib
export SDK_LTO=Full
export LINKER=lld
export CORE_LINKER=LLD
export LINK_FLAGS="-Wl,--no-relax,--build-id=sha1"
export PATH=/usr/lib/llvm-23/bin:$PATH
mkdir -p "$OPT_ROOT"
cat > "$OPT_ROOT/build-constraints.txt" <<'CONSTRAINTS'
nanobind==3.0.1
scikit-build-core==1.0.3
vcs-versioning==2.3.4
packaging==26.3
pathspec==1.1.1
CONSTRAINTS

cmake -S "$LLVM_SOURCE/llvm" -B "$OPT_ROOT/llvm-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$RELEASE_SDK" \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_C_FLAGS="$CPU_FLAGS" -DCMAKE_CXX_FLAGS="$CPU_FLAGS" \
  -DCMAKE_AR="$AR" \
  -DCMAKE_RANLIB="$RANLIB" \
  -DLLVM_ENABLE_PROJECTS='mlir;bolt' -DLLVM_TARGETS_TO_BUILD=AArch64 \
  -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_LTO="$SDK_LTO" \
  -DLLVM_USE_LINKER="$LINKER" -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DLLVM_BUILD_TOOLS=ON -DLLVM_BUILD_TESTS=OFF \
  -DLLVM_ENABLE_WARNINGS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_ENABLE_LIBPFM=OFF -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_TABLEGEN="$NATIVE_SDK/bin/llvm-tblgen" \
  -DMLIR_TABLEGEN="$NATIVE_SDK/bin/mlir-tblgen"
```

Keep `LLVM_BUILD_TOOLS=ON` so the generated exports include `llvm-as` and
`llvm-dis`, which Core's C++ tests reference. Building only the explicitly
listed static-library targets still reuses the native SDK tools. Copy those
tools into the new install prefix, install the new generated headers and CMake
exports, and replace the archives with the newly built archives. The experiment
records the target list and archive hashes; never label a partial archive
overlay as a complete SDK rebuild. Ordinary optimization-level and CPU rows
rebuild all LLVM/MLIR static archives. PGO rows rebuild the recorded closure of
SDK archives linked by Core and retain that distinction.

```sh
python3 - <<'PYSDK'
import os
import shutil
import subprocess
from pathlib import Path

native = Path(os.environ["NATIVE_SDK"])
release = Path(os.environ["RELEASE_SDK"])
build = Path(os.environ["OPT_ROOT"]) / "llvm-build"
targets = sorted(
    p.name.removeprefix("lib").removesuffix(".a")
    for p in (native / "lib").glob("*.a")
    if p.name.startswith(("libLLVM", "libMLIR"))
)
assert targets and not release.exists()
subprocess.run(["cmake", "--build", str(build), "--target", *targets, "-j", "4"], check=True)
shutil.copytree(native, release, symlinks=True)
for component in ["llvm-headers", "mlir-headers", "cmake-exports", "mlir-cmake-exports"]:
    subprocess.run(["cmake", "--install", str(build), "--component", component], check=True)
for archive in (build / "lib").glob("*.a"):
    shutil.copy2(archive, release / "lib" / archive.name)
PYSDK
```

For GCC, select `gcc`, `g++`, `gcc-ar`, `gcc-ranlib`, and BFD from the same
compiler installation, use `LLVM_ENABLE_LTO=ON`, and retain
`-fno-reorder-blocks-and-partition` in SDK and Core builds used by BOLT. Set
GCC's link-time worker count with `-flto=4`. Clang full LTO uses
`--lto-partitions=4` for parallel code generation; ThinLTO uses
`--thinlto-jobs=4`. CMake/Ninja's job count alone does not bound LTO workers.

For the host GCC variant, set these variables before configuring a fresh SDK and
Core build directory. Use the manylinux compiler-toolset paths when building the
portable GCC variant inside the pinned image.

```sh
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export AR=/usr/bin/gcc-ar
export RANLIB=/usr/bin/gcc-ranlib
export SDK_LTO=ON
export LINKER=bfd
export CORE_LINKER=BFD
export CPU_FLAGS="-fno-reorder-blocks-and-partition"
export LINK_FLAGS="-Wl,--no-relax,--build-id=sha1 -flto=4"
```

Build a Core wheel with the explicit release SDK, full IPO, and retained BOLT
input information:

```sh
DEPLOY=ON CMAKE_BUILD_PARALLEL_LEVEL=4 uv build "$CORE_SOURCE" --wheel --python "$PYTHON" \
  --out-dir "$OPT_ROOT/wheels" \
  --build-constraints "$OPT_ROOT/build-constraints.txt" \
  -Cbuild-dir="$OPT_ROOT/core-build" \
  -Ccmake.define.MLIR_DIR="$RELEASE_SDK/lib/cmake/mlir" \
  -Ccmake.define.LLVM_DIR="$RELEASE_SDK/lib/cmake/llvm" \
  -Ccmake.define.CMAKE_C_COMPILER="$CC" \
  -Ccmake.define.CMAKE_CXX_COMPILER="$CXX" \
  -Ccmake.define.CMAKE_AR="$AR" -Ccmake.define.CMAKE_RANLIB="$RANLIB" \
  -Ccmake.define.CMAKE_C_FLAGS="$CPU_FLAGS" \
  -Ccmake.define.CMAKE_CXX_FLAGS="$CPU_FLAGS" \
  "-Ccmake.define.CMAKE_EXE_LINKER_FLAGS=$LINK_FLAGS" \
  "-Ccmake.define.CMAKE_SHARED_LINKER_FLAGS=$LINK_FLAGS" \
  "-Ccmake.define.CMAKE_MODULE_LINKER_FLAGS=$LINK_FLAGS" \
  -Ccmake.define.ENABLE_IPO=ON -Ccmake.define.ENABLE_BOLT=ON \
  -Ccmake.define.CMAKE_LINKER_TYPE="$CORE_LINKER" -Cinstall.strip=false
```

Set `CPU_FLAGS=-mcpu=native` before both configurations for the Clang
machine-specific recipe; leave it empty for the portable CPU baseline. The shown
linker options retain the experiment's no-relaxation baseline.

Use `CMAKE_C_FLAGS_RELEASE` and `CMAKE_CXX_FLAGS_RELEASE` to compare
`-O2 -DNDEBUG` with `-O3 -DNDEBUG` in both SDK and Core. Apply CPU options
through both `CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS`. Keep native CPU tuning
separate from LTO. Core's release configuration forces full Clang LTO; ThinLTO
experiments must explicitly adjust that experimental source snapshot and verify
the final compile and link commands.

## Portable Clang setup

For the portable variant, provision this toolchain before the SDK/Core commands
above, then replace their host compiler variables with the exported values
below. Use fresh build directories when switching toolchains. The compiler
executes on the host; the emitted code and CMake library searches use the
manylinux sysroot. The optional host ICU extraction supplies the downloaded
LLD's dependency without putting that library into the target sysroot.

```sh
export PORTABLE_TOOLS="$OPT_ROOT/toolchains"
export MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_aarch64:2026.08.04-1
mkdir -p "$PORTABLE_TOOLS/llvm-23" "$PORTABLE_TOOLS/manylinux-sysroot"
curl -fL -o "$PORTABLE_TOOLS/LLVM-23.1.0-Linux-ARM64.tar.xz" \
  https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.0/LLVM-23.1.0-Linux-ARM64.tar.xz
printf '%s  %s\n' \
  cfb31bfc713ef453248bf5bd026312f838ad6c52c25623e987cb6a340f3050d4 \
  "$PORTABLE_TOOLS/LLVM-23.1.0-Linux-ARM64.tar.xz" | sha256sum -c -
tar -xJf "$PORTABLE_TOOLS/LLVM-23.1.0-Linux-ARM64.tar.xz" \
  -C "$PORTABLE_TOOLS/llvm-23" --strip-components=1
container=$(docker create "$MANYLINUX_IMAGE")
docker export "$container" | tar -x -C "$PORTABLE_TOOLS/manylinux-sysroot" \
  --no-same-owner --anchored --exclude=dev --exclude=proc --exclude=sys
docker rm "$container"
curl -fL -o "$PORTABLE_TOOLS/libicu70.deb" \
  https://ports.ubuntu.com/pool/main/i/icu/libicu70_70.1-2_arm64.deb
printf '%s  %s\n' \
  ac68372cf4a976e6a206858fd9b28c68e49d37d650b9b8653270038a6e7bc174 \
  "$PORTABLE_TOOLS/libicu70.deb" | sha256sum -c -
dpkg-deb -x "$PORTABLE_TOOLS/libicu70.deb" "$PORTABLE_TOOLS/icu70"
mkdir -p "$PORTABLE_TOOLS/clang-manylinux"
cat > "$PORTABLE_TOOLS/clang-manylinux/clang" <<'WRAPPER'
#!/bin/sh
set -eu
tools=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
link=yes
selected=no
for arg do
  case "$arg" in
    -c|-S|-E|--version|-print*|-dump*) link=no ;;
    -fuse-ld=*|--ld-path=*) selected=yes ;;
  esac
done
export LD_LIBRARY_PATH="$tools/icu70/usr/lib/aarch64-linux-gnu"
if [ "$link" = yes ] && [ "$selected" = no ]; then
  set -- "--ld-path=$tools/llvm-23/bin/ld.lld" "$@"
fi
exec "$tools/llvm-23/bin/$(basename -- "$0")" \
  "--sysroot=$tools/manylinux-sysroot" \
  "--gcc-install-dir=$tools/manylinux-sysroot/opt/rh/gcc-toolset-14/root/usr/lib/gcc/aarch64-redhat-linux/14" "$@"
WRAPPER
cp "$PORTABLE_TOOLS/clang-manylinux/clang" "$PORTABLE_TOOLS/clang-manylinux/clang++"
chmod +x "$PORTABLE_TOOLS/clang-manylinux/clang" "$PORTABLE_TOOLS/clang-manylinux/clang++"
export CC="$PORTABLE_TOOLS/clang-manylinux/clang"
export CXX="$PORTABLE_TOOLS/clang-manylinux/clang++"
export AR="$PORTABLE_TOOLS/llvm-23/bin/llvm-ar"
export RANLIB="$PORTABLE_TOOLS/llvm-23/bin/llvm-ranlib"
export CMAKE_TOOLCHAIN_FILE="$PORTABLE_TOOLS/manylinux.cmake"
cat > "$CMAKE_TOOLCHAIN_FILE" <<EOF
set(CMAKE_SYSROOT "$PORTABLE_TOOLS/manylinux-sysroot")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
EOF
```

The anchored tar exclusions are necessary: unanchored `--exclude=sys` also
removes system include directories. Use `SDK_LTO=Full`, `LINKER=lld`,
`CORE_LINKER=LLD`, and empty CPU flags for the portable finalist. Compiler
resource files and `llvm-profdata` come from the same downloaded archive. Mold
with Clang additionally requires a matching `LLVMgold.so`; build that plugin
from the same SDK source using `LLVM_BINUTILS_INCDIR`, rather than borrowing a
plugin from a different LLVM release.

## Compiler PGO and BOLT

Instrument the SDK libraries and Core separately, preserving source files,
compiler options, build directories, and installed SDK include paths between
generation and use. Disable Python build isolation for these builds and install
the pinned build requirements into a fixed virtual environment. Otherwise each
`uv build` changes nanobind include/source paths through its temporary build
environment. GCC's profile identity includes the object path. Use a fixed SDK
prefix and replace its archive contents when comparing SDK-only, Core-only, and
combined PGO; changing the include prefix can invalidate profile identity. Touch
replaced archives so Ninja relinks consumers.

Clang's SDK configuration uses `LLVM_BUILD_INSTRUMENTED=IR` for generation and
`LLVM_PROFDATA_FILE` for use. Core uses `-fprofile-generate` during compilation
and final linking, then `-fprofile-use=/absolute/path/to/merged.profdata` during
compilation. Merge with the matching `llvm-profdata merge`. Use
`LLVM_PROFILE_FILE=/absolute/raw/invocation/%m-%p.profraw` to distinguish
modules and processes.

GCC uses `-fprofile-generate=%q{MQT_PGO_PROFILE_DIR}/%p` and
`-fprofile-update=atomic` during compilation, with `-fprofile-generate` at final
linking. Set `MQT_PGO_PROFILE_DIR` separately for each training invocation.
Merge process directories with the matching `gcov-tool merge`, checking the
profile headers and merge diagnostics. With the measured GCC 13 tool, pass only
filenames common to both inputs to `gcov-tool merge`, then copy files unique to
each input unchanged. Its handling of unmatched indirect-call profiles can
crash. Put each fresh raw input first and the accumulated input second: the
measured tool otherwise subtracts from previously saturated negative top-N
totals. Start with the raw directory containing a saturated histogram, if one
exists. The archived harness supports at most one such raw directory and rejects
other cases. Check that the output preserves the union of input filenames and
that each absolute histogram total equals the sum of the raw absolute totals.
Also check merged summary counts against the input sums. Use
`-fprofile-use=/absolute/path/to/merged` and `-Werror=coverage-mismatch` during
the optimized rebuild. Normalize profile-prefix paths: `/experiment/.` caused
GCC feature checks to fail under `-Werror`, silently omitting PIC from the SDK.
Use `/experiment` and require `CXX_SUPPORTS_FPIC:INTERNAL=1` after SDK
configuration before building. Clear failed cached feature checks before
retrying. Use the same `-fprofile-prefix-path` in generation and use when long
build paths would exceed the filesystem's filename limit. Build-time and CTest
discovery profiles are not training inputs. The C++ training runner directs
discovery profiles to a separate build directory.

Create the fixed Core build environment once, using the same interpreter and
requirements as the baseline:

```sh
uv venv --python "$PYTHON" "$OPT_ROOT/build-env"
uv pip sync --python "$OPT_ROOT/build-env/bin/python" "$OPT_ROOT/build-constraints.txt"
uv pip install --python "$OPT_ROOT/build-env/bin/python" --no-deps ninja==1.13.0 cmake==4.4.2
export PYTHON="$OPT_ROOT/build-env/bin/python"
```

Append `--no-build-isolation` to the Core wheel build command for generation and
use. The GCC profile-use builds set `CCACHE_DISABLE=1`; regenerated profile
contents must not reuse objects from an earlier dataset under the same path.
Keep this environment in place through all SDK-only, Core-only, and combined PGO
builds. Build the C++ test targets explicitly with `cmake --build BUILD_DIR`
after creating the wheel; the wheel target does not build them. The experiment's
shared-library configuration also sets `BUILD_WITH_INSTALL_RPATH=FALSE` on test
executables and links `${CMAKE_DL_LIBS}` for their dynamic-loading tests. These
settings are retained in the archived `core-pgo-options.cmake` harness. Give C++
test processes a library search path containing the build's `lib`, `lib64`, and
directories of shared libraries under `src`; copied QDMI test plugins need these
transitive dependencies. The harness also copies the existing QDMI runtime files
from `lib` or `lib64` beside `src/qdmi/driver/libmqt-core-qdmi-driver.so`, where
the shared driver discovers device manifests. `prepare_cpp_runtime.py` records
the copied file hashes. Keep the library search path scoped to the C++ tests.
Python training uses the staged wheel and puts its binary directory and
interpreter directory on `PATH`.

Collect two independent datasets: benchmark-only, and tests plus benchmarks.
`test/release/train_optimization.py --expected-root STAGED_WHEEL` runs the
existing training program and scalable training fixtures. Add `--tests` for
Python tests and `--cpp-build BUILD_DIR` for regular C++ suites. For GCC builds
made inside manylinux, run the C++ suites inside the same container path layout.
Check training logs for profile write errors before merging. Retain training
counts, profile hashes, and exact merge commands.

For each dataset, rebuild SDK-only PGO, Core-only PGO, and both. Preserve the
unoptimized counterpart and then obtain fresh BOLT profiles from each final
wheel artifact. Use the SDK's `mqt-bolt-optimize` helper on the DD extension, DD
library, MLIR extension, DDSIM device, and benchmark executable. Train those
actual staged files with Python tests and the scalable fixtures. Separately
linked C++ executables do not supply wheel BOLT profiles. Do not BOLT-optimize
the standalone SDK tools.

Host GCC/BFD may insert Cortex-A53 erratum 843419 veneers that BOLT refuses to
rewrite. Keep that workaround for portable CPU builds. For the measured host GCC
native recipe (`-march=armv9.2-a -mtune=cortex-x3`, retaining
`-fno-reorder-blocks-and-partition` for SDK builds), Cortex-A53 is already
outside the supported CPU set. Scope this wrapper to that native build when
invoking `mqt-bolt-optimize`:

```sh
mkdir -p "$OPT_ROOT/bolt-native"
cat > "$OPT_ROOT/bolt-native/llvm-bolt" <<EOF
#!/bin/sh
exec "$NATIVE_SDK/bin/llvm-bolt" --drop-cortex-a53-843419-veneers "\$@"
EOF
chmod +x "$OPT_ROOT/bolt-native/llvm-bolt"
export PATH="$OPT_ROOT/bolt-native:$NATIVE_SDK/bin:$PATH"
```

Do not use that wrapper for portable CPU artifacts. The host GCC portable-CPU
row remains available without BOLT; removing the workaround would narrow its CPU
support.

Keep relocations and symbols through BOLT. After rewriting, use `llvm-strip`,
repack the wheel with regenerated RECORD hashes, and apply auditwheel repair
only to portable artifacts. Run installed-wheel tests and semantic checks again.
The helper's restoration checks cover failed rewriting; a successful training
run alone is not post-optimization validation.

## Measurement and resource controls

Record a command with the local runner; the output JSON includes its command,
explicit environment overrides, elapsed time, resource samples, and log paths:

```sh
python3 scripts/linux_optimization.py --output "$OPT_ROOT/build.json" \
  --env CMAKE_BUILD_PARALLEL_LEVEL=4 -- cmake --build "$OPT_ROOT/core-build" -j 4
```

Use `--replay RECORD --output NEW_RECORD` to repeat a recorded command. For a
Docker command, add `--container NAME` for cgroup memory accounting. For a
native command wrapped in `systemd-run --user --scope --unit=NAME`, pass
`--systemd-scope NAME.scope`. The measured runner limits are four CPUs, 16 GiB
RAM, and an additional 16 GiB swap. Use Docker
`--cpus=4 --memory=16g --memory-swap=32g`, or systemd
`CPUQuota=400% MemoryMax=16G MemorySwapMax=16G`. Aggregate cgroup memory
includes linker children; summing process RSS can count shared pages repeatedly.
For Docker commands, the outer GNU time and process-tree RSS describe the
client; use cgroup memory for the workload inside the container.

Replay LTO workers 1, 2, and 4. Keep a separate compiler cache and, for ThinLTO,
a separate linker cache. Record empty-cache and populated-cache runs without
implying that the OS page cache was flushed. Full LTO has no persistent ThinLTO
cache. Measure package compression separately from code size and linking.

Run `test/release/evaluate_optimization.py` against installed variants for
twelve rotating fresh-process rounds with fixed CPU affinity and thread counts,
and stop concurrent builds first. This local evaluator pins CPU 19; use an
available CPU consistently if adapting it to another machine. Keep the raw JSON,
per-workload CSV, bootstrap intervals, and artifact hashes. Throughput counts
completed workload invocations per second. Inspect individual regressions above
3%, even when the balanced score improves. Run regular C++ and Python tests, QIR
execution, installed CMake consumers, and dependency/CPU checks before ranking a
configuration as viable.

The installed C++ consumer uses the producer compiler and links
`Threads::Threads`. The fixed Core revision does not propagate the pthread
dependency needed by QDMI on glibc 2.28. GCC 13 also cannot link the constrained
DD `getVector` specialization exported by Clang 23 or the GCC 14 LTO builds:
those builds use newer constraint mangling. Treat these experimental C++
packages as compiler matched; successful wheel imports do not establish
compatibility with older C++ consumers.

## Measured choices on the local ARM64 machine

Use portable Clang 23.1.0 with LLD, assertions disabled, O3, full SDK and Core
LTO, combined SDK/Core compiler PGO from benchmark-only training, and fresh BOLT
profiles from Python tests plus scalable benchmarks. The twelve-round final
repeat gives 23.17% lower balanced latency than GCC full/full LTO without PGO or
BOLT (95% interval: 22.98%–23.43%). The final wheel is 39.1 MiB; its ELF files
occupy 122.3 MiB uncompressed. No held-out workload regresses above 3% against
that baseline.

For maximum measured local performance, use the same portable CPU target.
Host-native Clang with tests-plus-benchmarks compiler PGO is statistically tied
in aggregate but regresses all three DD simulation sizes above 3%. If a
machine-specific build is required, use the host Clang recipe above with
`CPU_FLAGS=-mcpu=native`, full LTO, O3, combined SDK/Core PGO trained on tests
plus benchmarks, and BOLT. This is a tested alternative, not a measured speed
advantage over the portable selection.

For GCC, use BFD and tests-plus-benchmarks combined PGO if pursuing the measured
GCC finalist. Clang remains faster in this workload mix. Patched mold was
measured and bundled in each SDK, but its extra dependency and LLVMgold plugin
have no demonstrated final runtime advantage over LLD. O2 was slower; ARM64
relaxation was within uncertainty; the PGO comparison retains `--no-relax`.

Use the raw per-workload CSV when your workload mix differs. These measurements
weight families equally, not by the frequency of a particular application's
calls. The four cohorts contain 1,740 fresh-process benchmark samples plus 1,740
separate startup probes across 109 viable artifacts, with fixed CPU affinity and
dependencies and separate training/evaluation inputs. The results bundle
includes the commands, toolchain hashes, build costs, package sizes, regression
flags, profile counts, and reproducible scratch orchestration. Large SDKs,
wheels, raw profiles, and build trees remain at the paths recorded in those
files.

For the tested four-CPU, 16 GiB runner limit, use four LTO workers and one large
link at a time. The selected portable Clang MLIR extension linked in 57 seconds
at 6.45 GiB; the host-native Clang alternative linked in 81 seconds at 6.96 GiB.
All four finalists passed 1/2/4-worker replays. Their five-stage BOLT runs took
164–203 seconds and peaked at 4.72–5.81 GiB. No replay used swap. These are
observed local costs from one first/repeated pair per setting, not confidence
intervals or proof that a smaller RAM limit is sufficient.

The compiler-cache probes cover LLVM Support's `CommandLine.cpp` and Core's
OpenQASM frontend unity translation unit. Their warm hits took 9–15 ms,
excluding measurement overhead, and preserved the exact object bytes. The
resource cap was tested on these compilation probes and complete Core link/BOLT
stages; a full SDK rebuild under that cap was not measured.

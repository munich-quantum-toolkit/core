# Installation

MQT Core is primarily developed as a C++20 library with Python bindings. The
Python package is available on [PyPI](https://pypi.org/project/mqt.core/) and
can be installed on all major operating systems with all
[officially supported Python versions](https://devguide.python.org/versions/).

:::::{tip}
:name: uv-recommendation

We recommend using [{code}`uv`][uv]. It is a fast Python package and project
manager by [Astral](https://astral.sh/) (creators of [{code}`ruff`][ruff]). It
can replace {code}`pip` and {code}`virtualenv`, automatically manages virtual
environments, installs packages, and can install Python itself. It is
significantly faster than {code}`pip`.

If you do not have {code}`uv` installed, install it with:

::::{tab-set}

:::{tab-item} Linux and macOS

```console
curl -LsSf https://astral.sh/uv/install.sh | sh
```

:::

:::{tab-item} Windows (PowerShell)

```console
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

:::

::::

See the [uv documentation][uv] for more information.

:::::

::::{tab-set}
:sync-group: installer

:::{tab-item} {code}`uv` _(recommended)_
:sync: uv

```console
uv pip install mqt.core
```

:::

:::{tab-item} {code}`pip`
:sync: pip

```console
python -m pip install mqt.core
```

:::

::::

In most cases, no compilation is required; a platform-specific prebuilt wheel is
downloaded and installed.

Verify the installation:

```console
python -c "import mqt.core; print(mqt.core.__version__)"
```

This prints the installed package version.

## Build performance

Release wheels use portable CPU settings and the assertion-free LLVM/MLIR 23.1.2
SDK. Linux wheels target manylinux_2_28 and use Clang 22 with LLD and ThinLTO;
macOS wheels use Apple Clang and ThinLTO with a macOS 13.3 deployment target.
Windows wheels use MSVC with IPO, except for DLLs that require automatic symbol
exports. Wheels include the QDMI driver and device bundles, `mqt-cc`, and
`mqt-core-bench`. Their CMake package exposes the QDMI C interfaces and tool
targets without requiring LLVM/MLIR. See {doc}`cpp_api` for native consumers.
C++ development libraries and headers are available from source installations;
the DD library is static.

### Building from source

Build from source to tune Core for the machine that will run it:

::::{tab-set}
:sync-group: installer

:::{tab-item} {code}`uv` _(recommended)_
:sync: uv

```console
uv pip install mqt.core --no-binary mqt.core
```

:::

:::{tab-item} {code}`pip`
:sync: pip

```console
pip install mqt.core --no-binary mqt.core
```

:::

::::

This requires a C++20-capable
[C++ compiler](https://en.wikipedia.org/wiki/List_of_compilers#C++_compilers)
and [CMake](https://cmake.org/) 3.28 or newer. CMake downloads a compatible
LLVM/MLIR SDK when none is available; see {ref}`setting-up-mlir` for supported
platforms and offline builds.

Local Release source builds default to `DEPLOY=OFF`, which enables native CPU
tuning and LTO when the compiler supports them. On Linux, Clang with its
matching LLD linker is a useful choice; on macOS, use Apple Clang from Xcode.
For example, with Clang 23 installed on Linux:

```console
CC=clang-23 CXX=clang++-23 uv pip install mqt.core --no-binary mqt.core \
  -Ccmake.define.CMAKE_LINKER_TYPE=LLD
```

Keep native builds on compatible CPUs. For redistribution, set
`-Ccmake.define.DEPLOY=ON` and choose the target platform's compiler and system
baseline. Cibuildwheel sets deployment mode explicitly for release wheels. The
`DEPLOY` environment variable overrides the CMake setting.

Clang and Apple Clang use ThinLTO through CMake's `ENABLE_IPO` option. For a
local C++ build, `cmake --preset release` selects the same release defaults;
pass `-DENABLE_IPO=OFF` to disable LTO. GCC can use mold 3 or newer with
`-DCMAKE_LINKER_TYPE=MOLD`. Clang with mold also needs a matching LLVM LTO
plugin; LLD includes the required support. MSVC IPO applies to static libraries
and DLLs with explicit exports; CMake cannot extract automatic exports from MSVC
IPO objects. MSVC builds with native tests default IPO off because each test
link otherwise repeats code generation from the libraries. Use `-DENABLE_IPO=ON`
to test IPO explicitly. CI builds default IPO off on all platforms; release
wheels explicitly enable it.

With CMake 3.29 or newer, Clang with LLD on Linux and Apple Clang with Apple's
linker cache ThinLTO results under `thinlto-cache-<configuration>` in the build
directory. The linkers prune this cache automatically; Linux uses a 1 GiB size
policy. Linux LLD builds also fold identical code outside Debug builds.
`-DENABLE_CACHE=OFF` disables automatic compiler and linker caching. Explicit
`CMAKE_C_COMPILER_LAUNCHER` and `CMAKE_CXX_COMPILER_LAUNCHER` settings take
precedence over compiler-cache detection, including an empty value to disable it
for one language.

Native tuning and LTO apply to the Core code being compiled. Prebuilt LLVM/MLIR
SDK libraries retain their own build settings, and LTO does not optimize across
separate shared libraries. Benchmark your application before changing the
compiler or LTO settings.

## Integrating MQT Core into Your Project

To use the MQT Core Python package in your project, add it as a dependency in
your {code}`pyproject.toml` or {code}`setup.py`. This ensures the package is
installed when your project is installed.

::::{tab-set}

:::{tab-item} {code}`uv` _(recommended)_

```console
uv add mqt.core
```

:::

:::{tab-item} {code}`pyproject.toml`

```toml
[project]
# ...
dependencies = ["mqt.core>=<version>"]
# ...
```

:::

:::{tab-item} {code}`setup.py`

```python
from setuptools import setup

setup(
    # ...
    install_requires=["mqt.core>=<version>"],
    # ...
)
```

:::

::::

If you want to integrate the C++ library directly into your project, you can
either

- add it as a [{code}`git` submodule][git-submodule] and build it as part of
  your project, or
- install MQT Core on your system and use CMake's {code}`find_package()` command
  to locate it, or
- use CMake's [{code}`FetchContent`][FetchContent] module to combine both
  approaches.

::::{tab-set}

:::{tab-item} {code}`FetchContent`

This is the recommended approach because it lets you detect installed versions
of MQT Core and only downloads the library if it is not available on the system.
Furthermore, CMake's [{code}`FetchContent`][FetchContent] module provides
flexibility in how the library is integrated into the project.

```cmake
include(FetchContent)
set(FETCH_PACKAGES "")

# cmakefmt: off
set(MQT_CORE_MINIMUM_VERSION "<minimum_version>"
    CACHE STRING "MQT Core minimum version")
set(MQT_CORE_VERSION "<version>"
    CACHE STRING "MQT Core version")
set(MQT_CORE_REV "<revision>"
    CACHE STRING "MQT Core identifier (tag, branch or commit hash)")
set(MQT_CORE_REPO_OWNER "munich-quantum-toolkit"
    CACHE STRING "MQT Core repository owner (change when using a fork)")
# cmakefmt: on
FetchContent_Declare(
  mqt-core
  GIT_REPOSITORY https://github.com/${MQT_CORE_REPO_OWNER}/core.git
  GIT_TAG ${MQT_CORE_REV}
  FIND_PACKAGE_ARGS ${MQT_CORE_MINIMUM_VERSION} COMPONENTS Development)
list(APPEND FETCH_PACKAGES mqt-core)

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})
```

:::

:::{tab-item} {code}`git-submodule`

Adding the library as a [{code}`git` submodule][git-submodule] is a simple
approach. However, {code}`git` submodules can be cumbersome, especially when
working with multiple branches or versions of the library. First, add the
submodule to your project (e.g., in the {code}`external` directory):

```console
git submodule add https://github.com/munich-quantum-toolkit/core.git external/mqt-core
```

Then add the following line to your {code}`CMakeLists.txt` to make the library's
targets available in your project:

```cmake
add_subdirectory(external/mqt-core)
```

:::

:::{tab-item} {code}`find_package()`

You can install MQT Core on your system after building it from source:

```console
git clone https://github.com/munich-quantum-toolkit/core.git mqt-core
cd mqt-core
cmake -S . -B build
cmake --build build
cmake --install build
```

Then, in your project's {code}`CMakeLists.txt`, use {code}`find_package()` to
locate the installed library:

```cmake
find_package(mqt-core <version> REQUIRED COMPONENTS Development)
```

:::

::::

(development-setup)=

## Development Setup

Set up a reproducible development environment for MQT Core. This is the
recommended starting point for both bug fixes and new features. For detailed
guidelines and workflows, see {doc}`contributing`.

1. Get the code:

   ::::{tab-set}

   :::{tab-item} External Contribution

   If you do not have write access to the
   [munich-quantum-toolkit/core](https://github.com/munich-quantum-toolkit/core)
   repository, fork the repository on GitHub (see
   <https://docs.github.com/en/get-started/quickstart/fork-a-repo>) and clone
   your fork locally.

   ```console
   git clone git@github.com:your_name_here/core.git mqt-core
   ```

   :::

   :::{tab-item} Internal Contribution

   If you have write access to the
   [munich-quantum-toolkit/core](https://github.com/munich-quantum-toolkit/core)
   repository, clone the repository locally.

   ```console
   git clone git@github.com/munich-quantum-toolkit/core.git mqt-core
   ```

   :::

   ::::

2. Change into the project directory:

   ```console
   cd mqt-core
   ```

3. Create a branch for local development:

   ```console
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

4. Install the project and its development dependencies:

   We highly recommend using modern, fast tooling for the development workflow.
   We recommend using [{code}`uv`][uv]. If you don't have {code}`uv`, follow the
   installation instructions in the recommendation above (see
   {ref}`tip above <uv-recommendation>`). See the [uv documentation][uv] for
   more information.

   ::::{tab-set}
   :sync-group: installer

   :::{tab-item} {code}`uv` _(recommended)_
   :sync: uv

   Install the project (including development dependencies) with
   [{code}`uv`][uv]:

   ```console
   uv sync
   ```

   :::

   :::{tab-item} {code}`pip`
   :sync: pip

   If you really don't want to use [{code}`uv`][uv], you can install the project
   and the development dependencies into a virtual environment using
   {code}`pip`.

   ```console
   python -m venv .venv
   source ./.venv/bin/activate
   python -m pip install -U pip
   python -m pip install -e . --group dev
   ```

   :::

   ::::

5. Install pre-commit hooks to ensure code quality:

   The project uses pre-commit hooks for running linters and formatting tools on
   each commit. These checks can be run manually via [{code}`nox`][nox], by
   running:

   ```console
   nox -s lint
   ```

   They can also be run automatically on every commit via [{code}`prek`][prek]
   (recommended). To set this up, install {code}`prek`, e.g., via:

   ::::{tab-set}

   :::{tab-item} Linux and macOS

   ```console
   curl --proto '=https' --tlsv1.2 -LsSf https://github.com/j178/prek/releases/latest/download/prek-installer.sh | sh
   ```

   :::

   :::{tab-item} Windows (PowerShell)

   ```console
   powershell -ExecutionPolicy ByPass -c "irm https://github.com/j178/prek/releases/latest/download/prek-installer.ps1 | iex"
   ```

   :::

   :::{tab-item} {code}`uv`

   ```console
   uv tool install prek
   ```

   :::

   ::::

   Then run:

   ```console
   prek install
   ```

(setting-up-mlir)=

## Setting Up MLIR

All MQT Core source builds require
[LLVM](https://llvm.org/)/[MLIR](https://mlir.llvm.org/) 23.1.x, including
embedded builds using `FetchContent` or `add_subdirectory`. CMake first searches
for a compatible installed SDK. If none is available, it downloads a prebuilt
SDK from [`portable-mlir-toolchain`] using a pinned [`setup-mlir`] installer and
version manifest. The wheel's
[Runtime component](cpp_api.md#use-the-wheels-native-runtime) needs no SDK.

### Automatic installation

The first configure downloads the SDK into `_deps/llvm-<version>-<assertions>`
inside the build directory. Subsequent configures reuse that installation. These
CMake settings control the fallback:

| Setting                      | Default         | Effect                                               |
| ---------------------------- | --------------- | ---------------------------------------------------- |
| `MQT_CORE_AUTO_INSTALL_MLIR` | `ON`            | Download an SDK when none is available.              |
| `MQT_CORE_MLIR_VERSION`      | `23.1.2`        | Select a version from the pinned installer manifest. |
| `MQT_CORE_MLIR_ASSERTIONS`   | Build-dependent | Select an SDK with LLVM assertions enabled.          |

All automatic downloads use Release SDKs. Assertions default to `ON` for
single-config `Debug` and `RelWithDebInfo` builds, and `OFF` otherwise,
including multi-config generators. Explicit settings take precedence. CMake
caches this choice; after changing the build type in an existing build
directory, use `-UMQT_CORE_MLIR_ASSERTIONS` to recompute the default or set it
explicitly.

Version overrides must use `X.Y.Z` syntax, belong to the LLVM 23.1 series, and
exist in the pinned manifest; commit hashes are not accepted. These options do
not replace or alter a compatible installed SDK. For example,
`-DMQT_CORE_MLIR_ASSERTIONS=OFF` selects the assertion-free download; it does
not change assertions in an SDK selected through `MLIR_DIR`. For Python builds,
pass CMake settings as `-Ccmake.define.<setting>=<value>` to `uv pip install` or
`pip install`.

Automatic installation supports native Linux x86_64 and AArch64 with glibc,
macOS AArch64, and Windows x86_64 and ARM64. Linux and macOS require Bash,
`curl`, and `tar`; Windows requires PowerShell 7.4 or newer (`pwsh`) and `tar`.
The installer selects the host architecture. Cross-compilation (including
same-OS architecture changes), Intel macOS, universal2, and musl builds require
a compatible SDK supplied by the user with automatic installation disabled. A
compiler compatible with the SDK is still required.

To disable LLVM/MLIR downloads, including for offline builds, set
`-DMQT_CORE_AUTO_INSTALL_MLIR=OFF`. A completed SDK installation in the build
directory can still be reused; otherwise, make an SDK available as shown below.
Other build dependencies must also be available offline. For manual SDK
installation or a custom LLVM/MLIR build, follow the [`setup-mlir`] or
[`portable-mlir-toolchain`] instructions.

### Making MLIR Available to the Build

To select an installed SDK, point the build system to it by setting the CMake
variable {code}`MLIR_DIR` to the **CMake configuration directory** of the
installation:

```console
cmake -S . -B build -DMLIR_DIR=/path/to/installation/lib/cmake/mlir
```

Replace `/path/to/installation` with the actual path to the MLIR installation.

Alternatively, you can set the {code}`MLIR_DIR` environment variable to the same
path before running CMake:

::::{tab-set}

:::{tab-item} Linux and macOS

```console
export MLIR_DIR=/path/to/installation/lib/cmake/mlir
```

:::

:::{tab-item} Windows (PowerShell)

```console
$env:MLIR_DIR = "C:\path\to\installation\lib\cmake\mlir"
```

:::

::::

[`setup-mlir`]: https://github.com/munich-quantum-software/setup-mlir/
[`portable-mlir-toolchain`]: https://github.com/munich-quantum-software/portable-mlir-toolchain/

<!-- Links -->
[FetchContent]: https://cmake.org/cmake/help/latest/module/FetchContent.html
[git-submodule]: https://git-scm.com/docs/git-submodule
[nox]: https://nox.thea.codes/en/stable/
[prek]: https://prek.j178.dev
[uv]: https://docs.astral.sh/uv/
[ruff]: https://docs.astral.sh/ruff/

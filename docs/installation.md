<!--- This file has been generated from an external template. Please do not modify it directly. -->
<!--- Changes should be contributed to https://github.com/munich-quantum-toolkit/templates. -->

# Installation

MQT Core is primarily developed as a C++20 library with Python bindings.
The Python package is available on [PyPI](https://pypi.org/project/mqt.core/) and can be installed on all major operating systems with all [officially supported Python versions](https://devguide.python.org/versions/).

:::::{tip}
:name: uv-recommendation

We recommend using [{code}`uv`][uv].
It is a fast Python package and project manager by [Astral](https://astral.sh/) (creators of [{code}`ruff`][ruff]).
It can replace {code}`pip` and {code}`virtualenv`, automatically manages virtual environments, installs packages, and can install Python itself.
It is significantly faster than {code}`pip`.

If you do not have {code}`uv` installed, install it with:

::::{tab-set}
:::{tab-item} macOS and Linux

```console
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

:::
:::{tab-item} Windows

```console
$ powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
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
$ uv pip install mqt.core
```

:::

:::{tab-item} {code}`pip`
:sync: pip

```console
(.venv) $ python -m pip install mqt.core
```

:::
::::
In most cases, no compilation is required; a platform-specific prebuilt wheel is downloaded and installed.

Verify the installation:

```console
(.venv) $ python -c "import mqt.core; print(mqt.core.__version__)"
```

This prints the installed package version.

## Building from Source for Performance

To get the best performance and enable platform-specific optimizations not available in portable wheels, we recommend building the library from source:

::::{tab-set}
:sync-group: installer

:::{tab-item} {code}`uv` _(recommended)_
:sync: uv

```console
$ uv pip install mqt.core --no-binary mqt.core
```

:::

:::{tab-item} {code}`pip`
:sync: pip

```console
(.venv) $ pip install mqt.core --no-binary mqt.core
```

:::
::::
This requires a C++20-capable [C++ compiler](https://en.wikipedia.org/wiki/List_of_compilers#C++_compilers) and [CMake](https://cmake.org/) 3.24 or newer.

## Integrating MQT Core into Your Project

To use the MQT Core Python package in your project, add it as a dependency in your {code}`pyproject.toml` or {code}`setup.py`.
This ensures the package is installed when your project is installed.

::::{tab-set}

:::{tab-item} {code}`uv` _(recommended)_

```console
$ uv add mqt.core
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

If you want to integrate the C++ library directly into your project, you can either

- add it as a [{code}`git` submodule][git-submodule] and build it as part of your project, or
- install MQT Core on your system and use CMake's {code}`find_package()` command to locate it, or
- use CMake's [{code}`FetchContent`][FetchContent] module to combine both approaches.

::::{tab-set}
:::{tab-item} {code}`FetchContent`

This is the recommended approach because it lets you detect installed versions of MQT Core and only downloads the library if it is not available on the system.
Furthermore, CMake's [{code}`FetchContent`][FetchContent] module provides flexibility in how the library is integrated into the project.

```cmake
include(FetchContent)
set(FETCH_PACKAGES "")

# cmake-format: off
set(MQT_CORE_MINIMUM_VERSION "<minimum_version>"
    CACHE STRING "MQT Core minimum version")
set(MQT_CORE_VERSION "<version>"
    CACHE STRING "MQT Core version")
set(MQT_CORE_REV "<revision>"
    CACHE STRING "MQT Core identifier (tag, branch or commit hash)")
set(MQT_CORE_REPO_OWNER "munich-quantum-toolkit"
    CACHE STRING "MQT Core repository owner (change when using a fork)")
# cmake-format: on
FetchContent_Declare(
  mqt-core
  GIT_REPOSITORY https://github.com/${MQT_CORE_REPO_OWNER}/core.git
  GIT_TAG ${MQT_CORE_REV}
  FIND_PACKAGE_ARGS ${MQT_CORE_MINIMUM_VERSION})
list(APPEND FETCH_PACKAGES mqt-core)

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})
```

:::

:::{tab-item} {code}`git-submodule`

Adding the library as a [{code}`git` submodule][git-submodule] is a simple approach.
However, {code}`git` submodules can be cumbersome, especially when working with multiple branches or versions of the library.
First, add the submodule to your project (e.g., in the {code}`external` directory):

```console
$ git submodule add https://github.com/munich-quantum-toolkit/core.git external/mqt-core
```

Then add the following line to your {code}`CMakeLists.txt` to make the library's targets available in your project:

```cmake
add_subdirectory(external/mqt-core)
```

:::

:::{tab-item} {code}`find_package()`

You can install MQT Core on your system after building it from source:

```console
$ git clone https://github.com/munich-quantum-toolkit/core.git mqt-core
$ cd mqt-core
$ cmake -S . -B build
$ cmake --build build
$ cmake --install build
```

Then, in your project's {code}`CMakeLists.txt`, use {code}`find_package()` to locate the installed library:

```cmake
find_package(mqt-core <version> REQUIRED)
```

:::

::::

(development-setup)=

## Development Setup

Set up a reproducible development environment for MQT Core.
This is the recommended starting point for both bug fixes and new features.
For detailed guidelines and workflows, see {doc}`contributing`.

1.  Get the code:

    ::::{tab-set}
    :::{tab-item} External Contribution
    If you do not have write access to the [munich-quantum-toolkit/core](https://github.com/munich-quantum-toolkit/core) repository, fork the repository on GitHub (see <https://docs.github.com/en/get-started/quickstart/fork-a-repo>) and clone your fork locally.

    ```console
    $ git clone git@github.com:your_name_here/core.git mqt-core
    ```

    :::
    :::{tab-item} Internal Contribution
    If you have write access to the [munich-quantum-toolkit/core](https://github.com/munich-quantum-toolkit/core) repository, clone the repository locally.

    ```console
    $ git clone git@github.com/munich-quantum-toolkit/core.git mqt-core
    ```

    :::
    ::::

2.  Change into the project directory:

    ```console
    $ cd mqt-core
    ```

3.  Create a branch for local development:

    ```console
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

4.  Install the project and its development dependencies:

    We highly recommend using modern, fast tooling for the development workflow.
    We recommend using [{code}`uv`][uv].
    If you don't have {code}`uv`, follow the installation instructions in the recommendation above (see {ref}`tip above <uv-recommendation>`).
    See the [uv documentation][uv] for more information.

    ::::{tab-set}
    :sync-group: installer

    :::{tab-item} {code}`uv` _(recommended)_
    :sync: uv
    Install the project (including development dependencies) with [{code}`uv`][uv]:

    ```console
    $ uv sync
    ```

    :::
    :::{tab-item} {code}`pip`
    :sync: pip
    If you really don't want to use [{code}`uv`][uv], you can install the project and the development dependencies into a virtual environment using {code}`pip`.

    ```console
    $ python -m venv .venv
    $ source ./.venv/bin/activate
    (.venv) $ python -m pip install -U pip
    (.venv) $ python -m pip install -e . --group dev
    ```

    :::
    ::::

5.  Install pre-commit hooks to ensure code quality:

    The project uses [pre-commit] hooks for running linters and formatting tools on each commit.
    These checks can be run manually via [{code}`nox`][nox], by running:

    ```console
    $ nox -s lint
    ```

    They can also be run automatically on every commit via [{code}`prek`][prek] (recommended).
    To set this up, install {code}`prek`, e.g., via:

    ::::{tab-set}
    :::{tab-item} macOS and Linux

    ```console
    $ curl --proto '=https' --tlsv1.2 -LsSf https://github.com/j178/prek/releases/latest/download/prek-installer.sh | sh
    ```

    :::
    :::{tab-item} Windows

    ```console
    $ powershell -ExecutionPolicy ByPass -c "irm https://github.com/j178/prek/releases/latest/download/prek-installer.ps1 | iex"
    ```

    :::

    :::{tab-item} {code}`uv`

    ```console
    $ uv tool install prek
    ```

    :::
    ::::

    Then run:

    ```console
    $ prek install
    ```

6.  If you plan to contribute to MQT Core, you will also need to install MLIR.
    The section below describes how to do this.

(setting-up-mlir)=

## Setting Up MLIR

MQT Core requires [MLIR](https://mlir.llvm.org/), which is part of the [LLVM](https://llvm.org/) project, to be available when building from source.
To successfully build MQT Core, you must make an installation of MLIR available to the C++ builds on your platform.

We highly recommend using the prebuilt MLIR distribution provided by the [`portable-mlir-toolchain`] project.
These can be conveniently installed with the [`setup-mlir`] scripts as described below.

### Downloading the MLIR Distribution

The [`setup-mlir`] repository provides installation scripts for all supported operating systems.
You must pass the LLVM version (e.g., `22.1.0`) and the installation prefix (directory) where MLIR should be extracted.
The scripts download a platform-specific archive.
The only requirement is that the `tar` command is available on the system.

::::{note}
:name: tar-requirement

`tar` is included by default on Windows 10 and Windows 11.
On older Windows versions, you can install it, for example, via [Chocolatey](https://chocolatey.org/): `choco install tar`.
::::

::::{tab-set}
:::{tab-item} Linux and macOS

Run the Bash script with the desired LLVM version and installation path:

```console
$ curl -LsSf https://github.com/munich-quantum-software/setup-mlir/releases/latest/download/setup-mlir.sh | bash -s -- -v 22.1.0 -p /path/to/installation
```

Replace `/path/to/installation` with the directory where the LLVM distribution should be installed (e.g., `/opt/llvm-22.1.0`).

:::
:::{tab-item} Windows

Run the PowerShell script with the desired LLVM version and installation path:

```console
$ powershell -ExecutionPolicy ByPass -c "& ([scriptblock]::Create((irm https://github.com/munich-quantum-software/setup-mlir/releases/latest/download/setup-mlir.ps1))) -llvm_version 22.1.0 -install_prefix \path\to\installation"
```

Replace `\path\to\installation` with the directory where the LLVM distribution should be installed (e.g., `C:\llvm-22.1.0`).
For debug builds on Windows, add the `-use_debug` flag to the script invocation.

:::
::::

For supported LLVM versions, commit hashes, and other options, see the [`setup-mlir`] repository and its [`version-manifest.json`](https://github.com/munich-quantum-software/setup-mlir/blob/main/version-manifest.json).

::::{note}
:name: mlir-build-note

If you want to build MLIR from source, you can follow the instructions in the [`portable-mlir-toolchain`] repository.
This is not recommended unless you need a specific configuration that is not available in the prebuilt distributions, as building MLIR from source can be complex and time-consuming.
::::

### Making MLIR Available to the Build

After installing MLIR, point the build system to it by setting the CMake variable {code}`MLIR_DIR` to the **CMake configuration directory** of the installation:

```console
$ cmake -S . -B build -DMLIR_DIR=/path/to/installation/lib/cmake/mlir
```

Replace `/path/to/installation` with the actual path to the MLIR installation from the previous step.

Alternatively, you can set the {code}`MLIR_DIR` environment variable to the same path before running CMake:

::::{tab-set}
:::{tab-item} Linux and macOS

```console
$ export MLIR_DIR=/path/to/installation/lib/cmake/mlir
```

:::
:::{tab-item} Windows (PowerShell)

```console
$ $env:MLIR_DIR = "C:\path\to\installation\lib\cmake\mlir"
```

:::
::::

### Disabling MLIR

If you do not need MLIR-based functionality, you can disable it by setting the {code}`BUILD_MQT_CORE_MLIR` option to {code}`OFF`.
This disables all MLIR-related features in MQT Core and removes the dependency on MLIR.

```console
$ cmake -S . -B build -DBUILD_MQT_CORE_MLIR=OFF
```

[`setup-mlir`]: https://github.com/munich-quantum-software/setup-mlir/
[`portable-mlir-toolchain`]: https://github.com/munich-quantum-software/portable-mlir-toolchain/

<!-- Links -->

[FetchContent]: https://cmake.org/cmake/help/latest/module/FetchContent.html
[git-submodule]: https://git-scm.com/docs/git-submodule
[nox]: https://nox.thea.codes/en/stable/
[pipx]: https://pypa.github.io/pipx/
[pre-commit]: https://pre-commit.com/
[prek]: https://prek.j178.dev
[ruff]: https://docs.astral.sh/ruff/
[uv]: https://docs.astral.sh/uv/

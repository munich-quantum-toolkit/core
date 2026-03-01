# MQT Core

MQT Core is the backbone of the [Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io/) — a C++20 and Python library providing core data structures and algorithms for quantum computing design automation.
Key components include an intermediate representation (IR) for quantum computations, a state-of-the-art decision diagram (DD) package, a ZX-calculus engine, an OpenQASM 3.0 parser, a QIR runtime and runner, an MLIR-based quantum compiler infrastructure (mqt-cc), and implementations of the [quantum device management interface (QDMI)](https://github.com/Munich-Quantum-Software-Stack/QDMI).

## Tech Stack

| Area                    | Tool              | Notes                                                               |
| ----------------------- | ----------------- | ------------------------------------------------------------------- |
| C++ standard            | C++20             | Minimum required                                                    |
| Build system            | CMake 3.24+       | Up to 4.2 supported                                                 |
| Python                  | 3.10+             | 3.10–3.14; free-threading supported on 3.14                         |
| MLIR/LLVM               | 21+               |                                                                     |
| Python package manager  | **uv** (>=0.5.20) | NEVER use pip                                                       |
| Python build backend    | scikit-build-core | Stable ABI wheels targeting Python 3.12+ via nanobind               |
| Python linter/formatter | **ruff**          | NEVER use pylint, black, flake8, or isort                           |
| Python type checker     | **ty**            | NEVER use mypy                                                      |
| Python bindings         | **nanobind**      | NOT pybind11                                                        |
| C++ formatter           | clang-format      | Config in `.clang-format`; LLVM-based style                         |
| C++ linter              | clang-tidy        | Config in `.clang-tidy`, `bindings/.clang-tidy`, `mlir/.clang-tidy` |
| C++ test framework      | Google Test       |                                                                     |
| Python test framework   | pytest            | With pytest-xdist for parallelism                                   |
| Pre-commit runner       | prek              | Wraps pre-commit hooks                                              |

## Building the C++ Library

Configure and build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Replace `Release` with `Debug` for a debug build.
The `--config` flag is required for multi-configuration generators (e.g., MSVC) and is ignored by single-configuration generators.

Key CMake options:

| Option                       | Default                | Description                                                                          |
| ---------------------------- | ---------------------- | ------------------------------------------------------------------------------------ |
| `BUILD_MQT_CORE_TESTS`       | ON (if master project) | Build C++ test suite                                                                 |
| `BUILD_MQT_CORE_BINDINGS`    | OFF                    | Build Python bindings via nanobind                                                   |
| `BUILD_MQT_CORE_MLIR`        | ON                     | Build MLIR quantum compiler infrastructure (requires LLVM 21)                        |
| `BUILD_MQT_CORE_SHARED_LIBS` | OFF                    | Build as shared libraries (used internally for Python packaging; not for direct use) |
| `ENABLE_COVERAGE`            | OFF                    | Enable code coverage instrumentation                                                 |

### C++ Dependencies

C++ dependencies are managed via CMake's `FetchContent` (configured in `cmake/ExternalDependencies.cmake`):

| Dependency                                                         | Version | Condition                                                                    |
| ------------------------------------------------------------------ | ------- | ---------------------------------------------------------------------------- |
| [nlohmann/json](https://github.com/nlohmann/json)                  | 3.12.0  | Always                                                                       |
| [Boost.Multiprecision](https://github.com/boostorg/multiprecision) | 1.89.0  | Always (fetched by default; set `USE_SYSTEM_BOOST=ON` to use system install) |
| [spdlog](https://github.com/gabime/spdlog)                         | 1.17.0  | Always                                                                       |
| [QDMI](https://github.com/Munich-Quantum-Software-Stack/qdmi)      | 1.2.1   | Always                                                                       |
| [Google Test](https://github.com/google/googletest)                | 1.17.0  | `BUILD_MQT_CORE_TESTS=ON`                                                    |
| [Eigen](https://gitlab.com/libeigen/eigen)                         | 5.0.1   | `BUILD_MQT_CORE_MLIR=ON`                                                     |
| [nanobind](https://github.com/wjakob/nanobind)                     |         | `BUILD_MQT_CORE_BINDINGS=ON` (found via `find_package`, not fetched)         |

## Running C++ Tests

Run the full C++ test suite:

```bash
ctest --test-dir build -C Release
```

The `-C` flag selects the configuration on multi-configuration generators (e.g., MSVC) and is ignored by single-configuration generators. Replace `Release` with `Debug` for debug builds.

Run a single test by name (regex match):

```bash
ctest --test-dir build -C Release -R mqt-core-dd-test
```

Run with verbose output on failure:

```bash
ctest --test-dir build -C Release --output-on-failure
```

Run tests by label (e.g., all MLIR unit tests in `mlir/unittests/`):

```bash
ctest --test-dir build -C Release -L mqt-mlir-unittests
```

Run a specific test binary directly:

```bash
./build/test/dd/mqt-core-dd-test
```

Prefer running targeted tests over the full suite during development.
All C++ tests use Google Test. Write new tests in `test/` (or `mlir/unittests/` for MLIR).

## Building the Python Package

Install all dev dependencies and build the package:

```bash
uv sync
```

Build artifacts are placed in `build/{wheel_tag}/{build_type}/`, where `{wheel_tag}` is a PEP 425 tag (e.g., `cp312-cp312-linux_x86_64`) and `{build_type}` is the CMake build type (e.g., `Release`).

## Running Python Tests

Full test suite via nox (handles environment setup):

```bash
uvx nox -s tests
```

Run tests for a specific Python version (preferred during iteration):

```bash
uvx nox -s tests-3.14
```

Run a single test file (`uv run` automatically syncs the environment):

```bash
uv run pytest test/python/dd/test_dd_package.py
```

Single test function:

```bash
uv run pytest test/python/dd/test_dd_package.py::test_sample_simple_circuit
```

With coverage:

```bash
uvx nox -s tests -- --cov
```

Test with minimum dependency versions:

```bash
uvx nox -s minimums-3.14
```

Test against latest Qiskit from git:

```bash
uvx nox -s qiskit-3.14
```

During iteration, prefer running a single Python version (e.g., `tests-3.14`, `minimums-3.14`). Run the full command across all versions for final checks.
Write new Python tests with pytest in `test/python/`.

## Linting, Formatting, and Code Style

Run all pre-commit checks (auto-fixes formatting and some linting issues, and reports remaining problems):

```bash
uvx nox -s lint
```

Run a specific check by hook ID (e.g., type checking):

```bash
uvx nox -s lint -- ty-check
```

### C++

- Formatting and naming conventions are enforced by clang-format and clang-tidy; see `.clang-format` and `.clang-tidy` for the full configuration.
- To include nanobind bindings in clang-tidy analysis, configure with `-DBUILD_MQT_CORE_BINDINGS=ON`. This requires a virtual environment with nanobind installed (e.g., via `uv sync` or minimally `uv sync --only-group build`).
- Header guards: `#pragma once`

### Python

Ruff is configured in `pyproject.toml` with `select = ["ALL"]` — all rules are enabled by default, with a small set selectively disabled.
Prefer fixing reported warnings over suppressing them with `# noqa` comments. Only add ignore rules when necessary and document why.
The same applies to ty type checking — fix reported issues before adding suppression comments. ty uses `# ty: ignore[code]` for fine-grained suppression (see [ty suppression docs](https://docs.astral.sh/ty/suppression/)). Suppressions are sometimes necessary for incompletely typed libraries (e.g., Qiskit).
Google-style docstrings are enforced.

## Documentation

Build docs locally (requires doxygen):

```bash
uvx nox --non-interactive -s docs
```

Check links:

```bash
uvx nox -s docs -- -b linkcheck
```

Documentation uses Sphinx with MyST (Markdown) and the furo theme.
Source files are in `docs/`.
C++ API docs require doxygen and are integrated via Breathe.
MLIR docs are auto-generated from tablegen during the build.

## Type Stubs

Regenerate Python type stubs from nanobind bindings:

```bash
uvx nox -s stubs
```

Stub files (`.pyi`) in `python/mqt/core/` are **auto-generated** by nanobind's stubgen.
NEVER edit `.pyi` files manually — they will be overwritten.
If C++ bindings are added or modified (files in `bindings/`), stubs need to be regenerated.

## Platform Support

The C++ library requires a compiler with C++20 support. It is regularly tested with GCC, Clang, AppleClang, and MSVC.

Pre-built Python wheels are available for Linux (x86_64/arm64), macOS (11.0+, x86_64/arm64), and Windows (x86_64/arm64).
Wheels target the Python 3.12 Stable ABI. Free-threading wheels are built for Python 3.14.

## Important Notes

- **Templated files**: Some files are generated by the [templating action](https://github.com/munich-quantum-toolkit/templates) and must not be edited directly, as changes will be overwritten.
  These files are marked by an initial comment such as `# This file has been generated from an external template. Please do not modify it directly.`
- **Pre-commit checks should always pass.** Run `uvx nox -s lint` to auto-fix formatting issues and get feedback on problems.

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

Configure and build in Release mode:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_MQT_CORE_TESTS=ON
cmake --build build
```

For a Debug build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_MQT_CORE_TESTS=ON
cmake --build build
```

Key CMake options:

| Option                       | Default                | Description                                                   |
| ---------------------------- | ---------------------- | ------------------------------------------------------------- |
| `BUILD_MQT_CORE_TESTS`       | ON (if master project) | Build C++ test suite                                          |
| `BUILD_MQT_CORE_BINDINGS`    | OFF                    | Build Python bindings via nanobind                            |
| `BUILD_MQT_CORE_MLIR`        | ON                     | Build MLIR quantum compiler infrastructure (requires LLVM 21) |
| `BUILD_MQT_CORE_SHARED_LIBS` | OFF                    | Build as shared libraries                                     |
| `ENABLE_COVERAGE`            | OFF                    | Enable code coverage instrumentation                          |

### C++ Dependencies

C++ dependencies are managed via CMake's `FetchContent` (configured in `cmake/ExternalDependencies.cmake`).
Each dependency is first searched on the system via `find_package`; if not found, it is automatically fetched from its upstream source.
Dependencies include nlohmann_json, Boost.Multiprecision, Google Test, spdlog, QDMI, and Eigen (MLIR only).

## Running C++ Tests

Run the full C++ test suite:

```bash
ctest --test-dir build
```

Run a single test by name (regex match):

```bash
ctest --test-dir build -R mqt-core-dd-test
```

Run with verbose output on failure:

```bash
ctest --test-dir build --output-on-failure
```

Run tests by label (e.g., all MLIR unit tests):

```bash
ctest --test-dir build -L mqt-mlir-unittests
```

Run a specific test binary directly:

```bash
./build/test/dd/mqt-core-dd-test
```

MLIR tests are located at `mlir/unittests/`.
Prefer running targeted tests over the full suite during development.
All C++ tests use Google Test. Write new tests in `test/` (or `mlir/unittests/` for MLIR).

## Building the Python Package

Install all dev dependencies and build the package:

```bash
uv sync
```

Build artifacts are placed in `build/{wheel_tag}/{build_type}/`.

## Running Python Tests

Full test suite via nox (handles environment setup):

```bash
uvx nox -s tests
```

Run tests for a specific Python version:

```bash
uvx nox -s tests-3.12
```

Single test file (if environment is already set up via `uv sync`):

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
uvx nox -s minimums
```

Test against latest Qiskit from git:

```bash
uvx nox -s qiskit
```

Prefer running individual test files during development.
Write new Python tests with pytest in `test/python/`.

## Linting, Formatting, and Code Style

Run all pre-commit checks (auto-fixes formatting and linting issues):

```bash
uvx nox -s lint
```

Run a specific check by hook ID (e.g., type checking):

```bash
uvx nox -s lint -- ty-check
```

### C++

- Formatting and naming conventions are enforced by clang-format and clang-tidy; see `.clang-format` and `.clang-tidy` for the full configuration.
- To include nanobind bindings in clang-tidy analysis, configure with `-DBUILD_MQT_CORE_BINDINGS=ON`.
- Header guards: `#pragma once`

### Python

Ruff is configured in `pyproject.toml` with `select = ["ALL"]` — all rules are enabled by default, with a small set selectively disabled.
Do not add new ignore rules without documenting why.
Google-style docstrings are enforced.

## Documentation

Build and serve docs locally (requires doxygen):

```bash
uvx nox -s docs
```

Build without serving (non-interactive):

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
If bindings change, regenerate stubs and commit the updated `.pyi` files.

## Project Layout

```text
include/mqt-core/       C++ public headers (organized by component)
src/                    C++ source files (mirrors include/ structure)
python/mqt/core/        Python package and auto-generated .pyi stubs
bindings/               nanobind C++ → Python bindings
test/                   C++ tests (Google Test)
test/python/            Python tests (pytest)
cmake/                  CMake modules and helpers
docs/                   Sphinx documentation source
mlir/                   MLIR quantum compiler infrastructure and QIR builder
eval/                   Benchmarking and evaluation
json/                   Device configuration files (NA, SC)
```

Components: `ir` (intermediate representation), `dd` (decision diagrams), `zx` (ZX-calculus), `algorithms`, `circuit_optimizer`, `qasm3` (OpenQASM 3.0 parser), `na` (neutral atoms), `qdmi` (device management), `qir` (QIR runtime and runner), `fomac` (formal methods), `mlir` (MLIR dialects and mqt-cc compiler).

## Platform Support

- **Operating systems**: Linux (Ubuntu 24.04), macOS (14, 15), Windows (2022, 2025)
- **Architectures**: x86_64 and arm64
- **C++ compilers**: GCC 14+, Clang 20+, MSVC (latest), AppleClang 17+
- **CI**: GitHub Actions with reusable workflows from `munich-quantum-toolkit/workflows`.
  Regular CI runs a subset of the matrix (Ubuntu/GCC, macOS/AppleClang, Windows/MSVC).
  The full compiler/OS matrix is tested when a PR has the `extensive-cpp-ci` label.

## Important Notes

- **License headers are required** on all source files: MIT license, SPDX identifier.
  Managed by `mz-lictools`.
  The license-tools pre-commit hook will fail if headers are missing.
- **Capitalization matters**: use the exact capitalizations "nanobind", "CMake", "ccache", "GitHub", "NumPy", "pytest", "MQT", and "TUM".
  A pre-commit hook (`disallow-caps`) enforces this and will reject common misspellings.
- **Templated files**: Some files are generated by the [templating action](https://github.com/munich-quantum-toolkit/templates) and must not be edited directly, as changes will be overwritten.
  These files are marked by an initial comment such as `# This file has been generated from an external template. Please do not modify it directly.`
- **Pre-commit checks should always pass.** Run `uvx nox -s lint` to auto-fix formatting issues and get feedback on problems.

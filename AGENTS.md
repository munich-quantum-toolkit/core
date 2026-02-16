# MQT Core

MQT Core is the backbone of the [Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io/) — a C++20 and Python library providing core data structures and algorithms for quantum computing design automation.
Key components include an intermediate representation (IR) for quantum computations, a state-of-the-art decision diagram (DD) package, a ZX-calculus engine, an OpenQASM 3.0 parser, MLIR-based quantum compilation dialects, and implementations of the [quantum device management interface (QDMI)](https://github.com/Munich-Quantum-Software-Stack/QDMI).

## Tech Stack

| Area                    | Tool                      | Notes                                       |
| ----------------------- | ------------------------- | ------------------------------------------- |
| C++ standard            | C++20                     | Minimum required                            |
| Build system            | CMake 3.24+               | Up to 4.2 supported                         |
| Python                  | 3.10+                     | Supports 3.10 through 3.14                  |
| MLIR/LLVM               | 21+                       | Required only when `BUILD_MQT_CORE_MLIR=ON` |
| Python package manager  | **uv** (>=0.5.20)         | NEVER use pip                               |
| Python build backend    | scikit-build-core         | With nanobind for bindings                  |
| Python linter/formatter | **ruff**                  | NEVER use pylint, black, flake8, or isort   |
| Python type checker     | **ty**                    | NEVER use mypy                              |
| Python bindings         | **nanobind** (~2.11.0)    | NOT pybind11                                |
| C++ formatter           | clang-format (LLVM style) |                                             |
| C++ linter              | clang-tidy                |                                             |
| CMake formatter         | cmake-format              |                                             |
| C++ test framework      | Google Test               |                                             |
| Python test framework   | pytest                    | With pytest-xdist for parallelism           |
| Pre-commit runner       | prek                      | Wraps pre-commit hooks                      |

## Building the C++ Library

```bash
# Configure and build (Release)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_MQT_CORE_TESTS=ON
cmake --build build

# Debug build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_MQT_CORE_TESTS=ON
cmake --build build
```

Key CMake options:

| Option                       | Default                | Description                                                |
| ---------------------------- | ---------------------- | ---------------------------------------------------------- |
| `BUILD_MQT_CORE_TESTS`       | ON (if master project) | Build C++ test suite                                       |
| `BUILD_MQT_CORE_BINDINGS`    | OFF                    | Build Python bindings via nanobind                         |
| `BUILD_MQT_CORE_MLIR`        | ON                     | Build MLIR quantum compilation dialects (requires LLVM 21) |
| `BUILD_MQT_CORE_SHARED_LIBS` | OFF                    | Build as shared libraries                                  |
| `BUILD_MQT_CORE_BENCHMARKS`  | OFF                    | Build evaluation benchmarks                                |
| `ENABLE_COVERAGE`            | OFF                    | Enable code coverage instrumentation                       |

In-source builds are prevented.
External dependencies (nlohmann_json, Boost.Multiprecision, Google Test, spdlog, QDMI, Eigen) are fetched automatically via CMake FetchContent if not found on the system.

## Running C++ Tests

```bash
# Run the full C++ test suite
ctest --test-dir build

# Run a single test by name (regex match)
ctest --test-dir build -R mqt-core-dd-test

# Run with verbose output on failure
ctest --test-dir build --output-on-failure

# Run a specific test binary directly for more control
./build/test/dd/mqt-core-dd-test
```

Tests are organized by component under `test/`: `algorithms/`, `circuit_optimizer/`, `dd/`, `ir/` (includes QASM3 parser tests), `na/`, `zx/`, `qdmi/`, `qir/`, `fomac/`.
MLIR tests are located at `mlir/unittests/`.
Prefer running targeted tests over the full suite during development.

## Building the Python Package

```bash
# Install all dev dependencies and build the package in one step
uv sync

# Build only (produces wheel)
uv build
```

Build artifacts are placed in `build/{wheel_tag}/{build_type}/`.
The Python package uses scikit-build-core as its build backend with nanobind for C++ bindings.

IMPORTANT: Always use `uv` for dependency management.
Never use `pip install`, `pip install -e .`, or any pip-based workflow.

## Running Python Tests

```bash
# Full test suite (via nox, handles environment setup)
uvx nox -s tests

# Single test file (if environment is already set up via uv sync)
uv run pytest test/python/dd/test_dd_package.py

# Single test function
uv run pytest test/python/dd/test_dd_package.py::test_sample_simple_circuit

# With coverage
uvx nox -s tests -- --cov

# Test with minimum dependency versions
uvx nox -s minimums

# Test against latest Qiskit from git
uvx nox -s qiskit
```

Python tests live in `test/python/` and use pytest with parallel execution (pytest-xdist).
Prefer running individual test files during development.

## Linting, Formatting, and Code Style

```bash
# Run ALL pre-commit checks (recommended before committing)
uvx prek run -a

# Alternative via nox
uvx nox -s lint

# Python type checking only
uv run ty check
```

### Pre-commit hook priority order

1. **Validation**: merge conflicts, pyproject schema, GitHub workflows, ReadTheDocs config, capitalization, typos, license headers, uv lockfile sync
2. **Formatting**: clang-format (C++), cmake-format (CMake), prettier (YAML/MD/JSON), ruff format (Python)
3. **Linting**: ruff check (Python)
4. **Type checking**: ty check (Python)

### C++

- Formatting: LLVM-based (see `.clang-format`)
- Linting: clang-tidy (see `.clang-tidy`).
  To include nanobind bindings in clang-tidy analysis, configure with `-DBUILD_MQT_CORE_BINDINGS=ON`.
- Naming: `camelBack` for functions, methods, variables, parameters; `CamelCase` for classes, structs, enums; `UPPER_CASE` for global/static constants; `lower_case` for namespaces
- Indentation: 2 spaces
- Header guards: `#pragma once`

### Python

- Formatter: ruff format (line length 120)
- Linter: ruff check with ALL rules enabled
- Type checker: ty (not mypy)
- Docstrings: Google style
- Indentation: 4 spaces
- Imports: sorted by ruff (isort rules), `from __future__ import annotations` enforced

The ruff config in `pyproject.toml` uses `select = ["ALL"]` — **all rules are enabled by default**.
Only a small set of rules are selectively disabled.
Do not add new rules; only disable specific ones if necessary and document why.
Line length is 120.
Google-style docstrings are enforced.

### CMake

- Keywords: `UPPER_CASE`
- Line width: 100
- Formatter: cmake-format (see `.cmake-format.yaml`)
- Indentation: 2 spaces

### General

- Line endings: LF
- Charset: UTF-8
- Trailing whitespace: trimmed (except Markdown)
- Final newline: required

## Documentation

```bash
# Build and serve docs locally (requires doxygen installed)
uvx nox -s docs

# Build without serving (non-interactive)
uvx nox --non-interactive -s docs

# Check links
uvx nox -s docs -- -b linkcheck
```

Documentation uses Sphinx with MyST (Markdown) and the furo theme.
Source files are in `docs/`.
C++ API docs require doxygen and are integrated via Breathe.
MLIR docs are auto-generated from tablegen during the build.

## Type Stubs

```bash
# Regenerate Python type stubs from nanobind bindings
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
mlir/                   MLIR quantum compilation dialects (QC, QCO) and QIR builder
eval/                   Benchmarking and evaluation
json/                   Device configuration files (NA, SC)
```

Components: `ir` (intermediate representation), `dd` (decision diagrams), `zx` (ZX-calculus), `algorithms`, `circuit_optimizer`, `qasm3` (OpenQASM 3.0 parser), `na` (neutral atoms), `qdmi` (device management), `qir` (QIR runtime), `fomac` (formal methods), `mlir` (MLIR dialects).

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
- **MLIR caveats**: MLIR is disabled automatically on macOS with GCC (ABI incompatibility) and with AppleClang < 17 (incomplete C++20 support).
- **Stub files**: Never edit `.pyi` files by hand.
  Always regenerate with `uvx nox -s stubs`.
- **uv lockfile**: The `uv.lock` file must stay in sync.
  The pre-commit hook `uv-lock` checks this.
  Run `uv lock` if you change dependencies in `pyproject.toml`.
- **Templated files**: Some files are generated by the [templating action](https://github.com/munich-quantum-toolkit/templates) and must not be edited directly, as changes will be overwritten.
  These files are marked by an initial comment such as `# This file has been generated from an external template. Please do not modify it directly.`
- **Pre-commit hooks must pass** before submitting pull requests.
  Run `uvx prek run -a` to check locally.
- **Test your changes**: all changes must be tested.
  Write C++ tests with Google Test in `test/`, Python tests with pytest in `test/python/`.

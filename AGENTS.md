# MQT Core

- Acts as the backbone of the [Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io/)
- C++20 and Python library providing core data structures and algorithms for quantum computing design automation
- Provides tools and methods for
  - Intermediate representations (IR) of quantum computations
  - Decision diagrams (DD) for classical simulation and verification of quantum circuits
  - ZX-calculus for transforming and optimizing quantum circuits
  - OpenQASM 3.0 parser and exporter
  - QIR runner and runtime (based on DDs)
  - MLIR-based quantum compiler infrastructure (MQT Compiler Collection; mqt-cc)
  - Example QDMI device implementations, C++ and Python bindings for QDMI
  - Utilities for neutral atom (NA) quantum computing

## Tech Stack

- Targets Linux (glibc 2.28+), macOS (11.0+), and Windows on x86_64 and arm64 architectures
- C++20
- CMake 3.24+
- Python 3.10+ (free-threading supported with 3.14+; Stable ABI wheels built for 3.12+ via nanobind)
- LLVM 21.1+ for building MLIR compiler infrastructure
- GoogleTest for C++ unit tests (located in `test/` and `mlir/unittests/`)
- `pytest` for Python unit tests (located in `test/python/`)
- `uv` is used for managing Python installations, Python packaging, and other tools
- `ruff` is used for formatting and linting Python code
- `ty` is used for type checking Python code
- `scikit-build-core` is used as a build backend for the Python package
- `nanobind` is used for generating Python bindings
- `nox` is used for running tasks (e.g., tests, linting, docs, etc.)
- `prek` is used for running pre-commit hooks
- `sphinx` is used for generating documentation
- `breathe` is used for integrating C++ API docs with Sphinx
- C++ dependencies are managed via CMake's `FetchContent` (configured in @file:cmake/ExternalDependencies.cmake)
- C++ formatting and naming conventions are enforced by clang-format and clang-tidy (see `.clang-format` and `.clang-tidy`)

## C++

- Configure CMake `cmake -S . -B build_cpp -DCMAKE_BUILD_TYPE=Release`
- Build `cmake --build build_cpp --config Release`
- Run tests `ctest --test-dir build_cpp -C Release`
- Only run MLIR unittests `ctest --test-dir build_cpp -C Release -L mqt-mlir-unittests`
- Run a specific test binary directly `./build_cpp/test/dd/mqt-core-dd-test`
- Create a debug build by replacing `Release` with `Debug` in the above commands.

## Python

- Set up environment with build and test dependencies `uv sync --inexact --only-group build --only-group test`
- Install the package without build isolation for fast rebuilds `uv sync --inexact --no-dev --no-build-isolation-package mqt-core`
- Run the tests `uv run --no-sync pytest`
- Shortcut for running tests `uvx nox -s tests`
- Shortcut for running tests with minimal version dependency resolution `uvx nox -s minimums`
- Shortcut for running against the upstream Qiskit package `uvx nox -s qiskit`
- To target Python 3.14 specifically, use the `-3.14` variants instead: `uvx nox -s tests-3.14`, `uvx nox -s minimums-3.14`, `uvx nox -s qiskit-3.14`.

## Documentation

- Source files are in `docs/`.
- Documentation uses Sphinx with MyST (Markdown) and the furo theme.
- Build MLIR docs via the `mlir-doc` target in the C++ build (`cmake --build build_cpp --target mlir-doc --config Release`)
- Build docs locally via `uvx nox --non-interactive -s docs`
- Check links via `uvx nox -s docs -- -b linkcheck`

## Development Guidelines

- Prioritize C++20 STL features over custom implementations.
- Within the MLIR codebase (`mlir/`), prefer LLVM data structures and methods (`llvm::SmallVector`, `llvm::function_ref`, etc.) over the STL.
- Use Google-style docstrings for Python and Doxygen comments for C++.
- Use `#pragma once` for header guards in C++.
- Run `uvx nox -s lint` after every batch of changes; all checks (ruff, typos, ty) must pass before submitting.
- Add or update tests for every code change, even if not explicitly requested.
- Prefer running targeted tests over the full test suite during development.
- Prefer running a single Python version over the full test suite during development.
- Follow the existing code style by checking neighboring files for patterns.
- Update @file:CHANGELOG.md and @file:UPGRADING.md when changes are user-facing, breaking, or otherwise noteworthy.
- Stub files (`.pyi`) in `python/mqt/core/` are **auto-generated** by nanobind's stubgen. NEVER edit `.pyi` files manually.
- If C++ bindings are added or modified (files in `bindings/`), stubs need to be regenerated via `uvx nox -s stubs`.
- Never modify files that start with "This file has been generated from an external template. Please do not modify it directly." These files are managed by [the MQT templates action](https://github.com/munich-quantum-toolkit/templates) and changes will be overwritten.
- Prefer fixing reported warnings over suppressing them (e.g., with `# noqa` comments for ruff). Only add ignore rules when necessary and document why.
- Prefer fixing typing issues reported by `ty` before adding suppression comments (`# ty: ignore[code]`). Suppressions are sometimes necessary for incompletely typed libraries (e.g., Qiskit).
- Ruff is configured in `pyproject.toml` with `select = ["ALL"]`. All rules are enabled by default, with a small set selectively disabled.

## Self-Review Checklist

- Did `uvx nox -s lint` pass without errors?
- Are all changes covered by at least one automated test (Python or C++)?
- Were Python stubs regenerated via `uvx nox -s stubs` if bindings were modified?
- Are @file:CHANGELOG.md and @file:UPGRADING.md updated when changes are user-facing, breaking, or otherwise noteworthy?

# MQT Core

- Acts as the high-performance backbone for the [Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io/).
- Provides tools and methods (IRs, DD, ZX, OpenQASM, MLIR, QIR, QDMI, NA, Qiskit, etc.) for quantum design automation, including simulation, compilation, verification as well as HPCQC integration.
- Maintains a C++20 core with an architecturally separate Python extension layer.
- Locates C++ sources in `src/` and `include/`, MLIR sources in `mlir/`, Python bindings in `bindings/`, the Python package in `python/mqt/core/`, tests in `test/`, and documentation in `docs/`.

## Environment & Capabilities

- **Platform**: Target Linux, macOS, and Windows for x86_64 (amd64) and arm64.
- **Python Context**: Version 3.10+ managed via **uv**.
- **C++ Context**: C++20 standard required; **LLVM 21.1+** mandatory for MLIR infrastructure.
- **Tooling Access**: Full access to `cmake`, `ctest`, `uv`, and `nox`.

## Repository Mapping

- `include/mqt/core/`: public and internal C++ header files.
- `src/mqt/core/`: core C++ implementation logic.
- `bindings/`: C++ sources used by the Python binding layer (nanobind glue).
- `python/mqt/core/`: Python package foundation containing helpers and entry points.
- `python/mqt/core/ir/`: public Python bindings package for the Circuit IR.
- `python/mqt/core/dd.pyi`: public Python bindings module for Decision Diagrams.
- `python/mqt/core/na/`: public Python bindings package for Neutral Atom logic.
- `test/`: comprehensive C++ and Python test suites.
- `docs/`: documentation sources, examples, and Sphinx/Doxygen configuration.
- `cmake/`: reusable CMake modules and build infrastructure.
- `mlir/`: MLIR dialects, passes, conversions, tests, and tooling.
- `eval/`: evaluation benchmarks, scripts, and example results.
- `json/`: JSON data used by evaluation code and tests (e.g. na/sc datasets).
- `paper/`: paper drafts, bibliography, and metadata for publications.

## Tasks

### C++ Workflows

- Configure Release build — `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_MQT_CORE_TESTS=ON`.
- Configure Debug build — `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_MQT_CORE_TESTS=ON`.
- Build the library — run `cmake --build build --parallel <jobs>` (or configure with `-G Ninja`, which handles parallel builds automatically).
- Run C++ tests: `ctest --test-dir build -C Release --output-on-failure`.
- Run MLIR unit tests: `ctest --test-dir build -C Release -L mqt-mlir-unittests --output-on-failure`.
- Run a single C++ test: `ctest --test-dir build -C Release -R <test_name>`.
- Locate C++ artifacts in `build/src/` (libraries), `build/test/` (test executables), and `build/mlir/` (MLIR build output).

### Python Workflows

- Build the Python package: `uv sync`.
- Locate Python build artifacts in `build/` folder.
- Execute full test suite: `uvx nox -s tests-3.13` (or `uvx nox -s tests` for all versions).
- Run targeted tests: `uv run pytest test/python/dd/`.
- Filter Python tests: `uvx nox -s tests-3.13 -- -k "<test_name>"` (or `uvx nox -s tests -- -k "<test_name>"` for all versions).
- Test minimum dependency versions: `uvx nox -s minimums`.
- Verify Qiskit compatibility: `uvx nox -s qiskit`.

### Quality, Docs & Stubs

- Run prek checks: `uvx nox -s lint`.
- Build documentation (Doxygen + Sphinx + MLIR): `uvx nox -s docs`.
- Locate documentation output in `docs/_build/html`.
- Generate type stubs: `uvx nox -s stubs`.
- Note: Stub files (`.pyi`) must **never** be edited manually.
- Check [contributing.md](docs/contributing.md) for comprehensive PR workflows and testing philosophies.

## Tools

- The project targets **C++20** and **Python 3.10+** as the strict minimum language versions.
- The build system relies on **CMake 3.24+** and **scikit-build-core**.
- **uv** is the project's package manager; `pip` and manual `venv` are not used.
- Python bindings are implemented with **nanobind** (~2.11.0); `pybind11` is not used.
- **ruff** handles linting and formatting, with `ALL` rules enabled by default.
- **ty** serves as the static type checker, replacing `mypy`.
- **LLVM 21.1+** powers the MLIR-based compilation dialects.
- **clang-format** and **typos** handle C++ style and project-wide spell checking respectively.

## Development Guidelines

- Always prioritize C++20 `std::` features over custom implementations.
- Use Google-style docstrings for Python and Doxygen comments for C++.
- Run `uvx nox -s lint` after performing changes and ensure all checks (ruff, typos, ty) pass.
- Verify all changes with at least one automated test (pytest or ctest).
- Follow existing code style by checking neighboring files for patterns.
- Review [CHANGELOG.md](docs/CHANGELOG.md) and [UPGRADING.md](docs/UPGRADING.md) before making breaking changes; update them accordingly.

## Self-Review Checklist

- Did you run `uvx nox -s lint` and ensure all checks (ruff, typos, ty) pass?
- Did you verify all your changes with at least one automated test (pytest or ctest)?
- Did you regenerate Python stubs via `uvx nox -s stubs` if bindings were modified?
- Did you check for manual changes to `.pyi` files (which are forbidden)?
- Did you include the correct license headers and SPDX identifiers?
- Did you review [CHANGELOG.md](docs/CHANGELOG.md) and [UPGRADING.md](docs/UPGRADING.md) and update them accordingly?

## Rules

- Adhere to the ruff philosophy: Start with `ALL` rules and selectively disable in `pyproject.toml`.
- Enforce term capitalization: `nanobind`, `CMake`, `ccache`, `GitHub`, `NumPy`, `pytest`, `MQT`, and `TUM`.
- Include MIT license and SPDX headers in every source file.
- Avoid modifying templated files locally; contribute to [MQT templates](https://github.com/munich-quantum-toolkit/templates).

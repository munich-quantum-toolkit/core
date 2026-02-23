# MQT Core

- Acts as the high-performance backbone for the [Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io/).
- Provides tools and methods (IRs, DD, ZX, OpenQASM, MLIR, QIR, QDMI, NA, Qiskit, etc.) for quantum design automation, including simulation, compilation, verification as well as HPCQC integration.
- Maintains a C++20 core with an architecturally separate Python extension layer.
- Locates C++ sources in `src/` and `include/`, Python bindings in `python/`, tests in `test/`, and documentation in `docs/`.

## Environment & Capabilities

- **Platform**: Target Linux, macOS, and Windows for x86_64 (amd64) and arm64.
- **Python Context**: Version 3.10+ managed via **uv**.
- **C++ Context**: C++20 standard required; **LLVM 21.1+** mandatory for MLIR infrastructure.
- **Tooling Access**: Full access to `cmake`, `ctest`, `uv`, and `nox`.
- **Constraint**: Avoid network-dependent tasks during builds; rely on `uv` lockfiles and pre-synced dependencies.

## Repository Mapping

- `include/mqt/core/`: C++ header files (Internal & Public interfaces).
- `src/mqt/core/`: C++ implementation logic.
- `python/mqt/core/`: Python package foundation.
- `python/mqt/core/ir/`: Public Python bindings package for Circuit IR.
- `python/mqt/core/dd.pyi`: Public Python bindings module for Decision Diagrams.
- `python/mqt/core/na/`: Public Python bindings package for Neutral Atom logic.
- `test/`: Comprehensive test suites (C++ and Python).
- `docs/`: Knowledge base for installation, contribution, and technical references.

## Tasks

### C++ Workflows

- Configure Release build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_MQT_CORE_TESTS=ON`.
- Configure Debug build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_MQT_CORE_TESTS=ON`.
- Build the library: `cmake --build build --parallel`.
- Run C++ tests: `ctest --test-dir build -C Release --output-on-failure`.
- Run a single C++ test: `ctest --test-dir build -C Release -R <test_name>`.
- Locate C++ artifacts in `build/src/` (libraries) and `build/test/` (test executables).

### Python Workflows

- Build the Python package: `uv build`.
- Locate Python build artifacts in `build/{wheel_tag}/{build_type}` or the project `build/` folder.
- Execute full test suite: `uvx nox -s tests`.
- Run targeted tests: `uv run pytest test/python/dd/`.
- Filter Python tests: `uvx nox -s tests -- -k "<test_name>"`.
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

- Use **C++20** and **Python 3.10+** as the strict minimum language versions.
- Use **CMake 3.24+** and **scikit-build-core** for the build system.
- Use **uv** as the mandatory package manager; never use `pip` or manual `venv`.
- Use **nanobind** (~2.11.0) for Python bindings; do not use `pybind11`.
- Use **ruff** for linting and formatting, with `ALL` rules enabled by default.
- Use **ty** for static type checking as the mandatory replacement for `mypy`.
- Use **LLVM 21.1+** for MLIR-based compilation dialects.
- Use **clang-format** and **typos** for C++ style and project-wide spell checking.

## Development Guidelines

- Always prioritize C++20 `std::` features over custom implementations.
- Use Google-style docstrings for Python and Doxygen comments for C++.
- Ensure Python code is fully typed and passes `ty` static analysis.
- Follow existing code style by checking neighboring files for patterns.
- Add or update tests for every code change, even if not explicitly requested.
- Review [CHANGELOG.md](docs/CHANGELOG.md) and [UPGRADING.md](docs/UPGRADING.md) before making breaking changes.

## Documentation Reference

- Deep dive into the Internal Representation: [mqt_core_ir.md](docs/mqt_core_ir.md).
- Understand Decision Diagram internals: [dd_package.md](docs/dd_package.md).
- Explore ZX-calculus algorithms: [zx_package.md](docs/zx_package.md).
- Access the online API reference: [MQT Core Docs](https://mqt.readthedocs.io/projects/core/en/latest/).

## Self-Review Checklist

- Did you run `uvx nox -s lint` and ensure all checks (ruff, typos, ty) pass?
- Did you verify all your changes with at least one automated test (pytest or googletest)?
- Did you update/add tests for new functionality to maintain coverage?
- Did you regenerate Python stubs via `uvx nox -s stubs` if bindings were modified?
- Did you check for manual changes to `.pyi` files (which are forbidden)?

## Rules

- Adhere to the ruff philosophy: Start with `ALL` rules and selectively disable in `pyproject.toml`.
- Enforce term capitalization: `nanobind`, `CMake`, `ccache`, `GitHub`, `NumPy`, `pytest`, `MQT`, and `TUM`.
- Include MIT license and SPDX headers in every source file.
- Avoid modifying templated files locally; contribute to [MQT templates](https://github.com/munich-quantum-toolkit/templates).
- Verify all CI checks locally; if `uvx nox -s lint` fails, the code is not ready for PR.

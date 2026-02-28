# MQT Core

- Acts as the high-performance backbone for the [Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io/).
- Provides tools and methods (IRs, DD, ZX, OpenQASM, MLIR, QIR, QDMI, NA, Qiskit, etc.) for quantum design automation, including simulation, compilation, verification as well as HPCQC integration.
- Maintains a C++20 core with an architecturally separate Python extension layer.
- Locates C++ sources in `src/` and `include/`, MLIR sources in `mlir/`, Python bindings in `bindings/`, the Python package in `python/mqt/core/`, tests in `test/`, and documentation in `docs/`.

## Environment & Capabilities

- **Platform**: Target Linux, macOS, and Windows for x86_64 (amd64) and arm64.
- **Python Context**: Version 3.10+ managed via **uv**.
- **C++ Context**: C++20 standard required; **LLVM 21.1+** mandatory for MLIR infrastructure.
- **Tooling Access**: Full access to `uv`. `cmake`, `nox`, and further tools are available through `uv` (via `uvx <tool>`)

## Repository Mapping

- `include/mqt-core/`: public C++ header files.
- `src/`: core C++ implementation logic.
- `bindings/`: C++ sources used by the Python binding layer (nanobind).
- `python/mqt/core/`: The mqt-core Python package.
- `test/`: comprehensive C++ and Python test suites.
- `docs/`: documentation sources, examples, and Sphinx/Doxygen configuration.
- `cmake/`: reusable CMake modules and build infrastructure.
- `mlir/`: MLIR dialects, passes, conversions, tests, and tooling.
- `eval/`: evaluation benchmarks, scripts, and example results.
- `json/`: JSON data used by evaluation code and tests (e.g. na/sc datasets).
- `paper/`: paper drafts, bibliography, and metadata for publications.

## Tasks

### C++ Workflows

- Configure Release build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_MQT_CORE_TESTS=ON`.
- Configure Debug build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_MQT_CORE_TESTS=ON`.
- Build the library: `cmake --build build --parallel <jobs>`.
  Alternatively, configure with `-G Ninja`, which handles parallel builds automatically.
- Run C++ tests: `ctest --test-dir build -C Release --output-on-failure`.
- Run MLIR unit tests: `ctest --test-dir build -C Release -L mqt-mlir-unittests --output-on-failure`.
- Run a single C++ test: `ctest --test-dir build -C Release -R <test_name>`.
- C++ artifacts are located in `build/src/` (libraries), `build/test/` (test executables), and `build/mlir/` (MLIR build output).

### Python Workflows

- Sync the Python environment: `uv sync`.
- Run the full test suite: `uvx nox -s tests`.
- Run targeted tests: `uv run pytest test/python/dd/`.
- Filter Python tests: `uvx nox -s tests -- -k "<test_name>"`.
- Test minimum dependency versions: `uvx nox -s minimums`.
- Verify Qiskit compatibility: `uvx nox -s qiskit`.

### Quality, Docs & Stubs

- Run all lint checks: `uvx nox -s lint`.
- Build documentation (Doxygen + Sphinx + MLIR): `uvx nox -s docs`.
- Documentation output is located in `docs/_build/html`.
- Generate type stubs: `uvx nox -s stubs`.
- Never edit `.pyi` stub files manually.
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

- Prioritize C++20 `std::` features over custom implementations.
- Use Google-style docstrings for Python and Doxygen comments for C++.
- Run `uvx nox -s lint` after every change; all checks (ruff, typos, ty) must pass before submitting.
- Add or update tests for every code change, even if not explicitly requested.
- Follow existing code style by checking neighboring files for patterns.
- Review [CHANGELOG.md](docs/CHANGELOG.md) and [UPGRADING.md](docs/UPGRADING.md) for every change; update them with any noteworthy additions, fixes, or breaking changes.

## Self-Review Checklist

- Did `uvx nox -s lint` pass without errors?
- Are all changes covered by at least one automated test (pytest or ctest)?
- Were Python stubs regenerated via `uvx nox -s stubs` if bindings were modified?
- Are there any manual edits to `.pyi` files (which are forbidden)?
- Do all source files include the MIT license and SPDX headers?
- Are [CHANGELOG.md](docs/CHANGELOG.md) and [UPGRADING.md](docs/UPGRADING.md) updated if needed?

## Rules

- Use `ALL` ruff rules and selectively disable in `pyproject.toml`.
- Enforce term capitalization: `nanobind`, `CMake`, `ccache`, `GitHub`, `NumPy`, `pytest`, `MQT`, and `TUM`.
- Include MIT license and SPDX headers in every source file.
- Do not modify templated files locally; contribute changes to [MQT templates](https://github.com/munich-quantum-toolkit/templates).

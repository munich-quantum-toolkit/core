# MQT Core

MQT Core is the backbone of the [Munich Quantum Toolkit](https://mqt.readthedocs.io/).
It provides the core data structures, algorithms, and Python bindings for quantum computing applications.
The C++ library lives in `src/` and `include/`, Python bindings in `python/`, tests in `test/`, and documentation in `docs/`.

## Dev environment tips

- Use `uv sync --all-extras` to install dependencies and lock the environment.
  **Do not use pip or venv manually.**
- Run `uv pip install -e . --no-build-isolation` to build the Python package in development mode.
- Check `build/{wheel_tag}/{build_type}` for Python build artifacts.
- To build Python bindings in debug mode, use `uv pip install -e . --no-build-isolation --config-settings=cmake.build-type=Debug`.
- For C++ only — `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_MQT_CORE_TESTS=ON`, then `cmake --build build`.
- For C++ debug builds: pass `-DCMAKE_BUILD_TYPE=Debug` instead.
- C++ build artifacts land in `build/src/` (libraries) and `build/test/` (test executables).
- We require **C++20**, **CMake 3.24+**, **Python 3.10+**, and **LLVM 21+**.
- We support Linux, macOS, and Windows (x86 & arm64).
  Ensure changes work cross-platform.
- See [`docs/contributing.md`](docs/contributing.md) for the full contributor guide and [`docs/installation.md`](docs/installation.md) for setup details.

## Testing instructions

- Run `uvx nox -s tests` to execute the full Python test suite.
- To focus on one test: `uvx nox -s tests -- -k "<test_name>"`.
- Run `uvx nox -s minimums` to test with the lowest supported dependency versions.
- Run `uvx nox -s qiskit` to verify compatibility with the latest Qiskit.
- For C++ tests, run `ctest -C Release --test-dir build` from the project root after building.
- To run a single C++ test: `ctest -C Release --test-dir build -R <test_name>`.
- Ensure `BUILD_MQT_CORE_BINDINGS=ON` and `BUILD_MQT_CORE_TESTS=ON` were set during CMake configuration.
- Add or update tests for the code you change, even if nobody asked.

## Linting and formatting

- Always run `uvx nox -s lint` (or equivalently `uvx prek run -a`) before committing.
- We use `ruff` for Python linting and formatting — not pylint, black, or isort.
  Our config enables **ALL** rules by default and only selectively disables some in [`pyproject.toml`](pyproject.toml).
- We use `ty` for static type checking — not mypy.
- We use `clang-format` and `clang-tidy` for C++ formatting and linting.
- We use `typos` for spell checking across the codebase.
- Do NOT edit `.pyi` stub files manually; regenerate them with `uvx nox -s stubs`.

## Docs instructions

- Run `uvx nox -s docs` to build the documentation.
- Use `uvx nox -s docs -- -b html` to build the HTML docs locally for preview.
- (Optional) Serve locally via `python -m http.server` from `docs/_build/html`.
- Find generated HTML files in `docs/_build/html`.
- Documentation entry point is [`docs/index.md`](docs/index.md).

## Development guidelines

- All changes must be tested.
  If you're not testing your changes, you're not done.
- Follow existing code style.
  Check neighboring files for patterns.
- Prefer C++20 `std::` features over custom implementations.
  Avoid unnecessary complexity.
- Python code must be fully typed and pass `ty` static analysis.
- Use Google-style docstrings for Python and Doxygen comments for C++.
- Always run `uvx prek run -a` at the end of a task.

## PR instructions

- Always run `uvx nox -s lint` and the relevant test suite before committing.
- If `uvx nox -s lint` fails, your code is not ready.
  Fix formatting and types first.

#!/usr/bin/env -S uv run --script --quiet
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# /// script
# dependencies = ["nox"]
# ///

"""Nox sessions."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import nox

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


nox.needs_version = ">=2025.10.16"
nox.options.default_venv_backend = "uv"

if os.environ.get("CI", None):
    nox.options.error_on_missing_interpreters = True

PYTHON_ALL_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]


@contextlib.contextmanager
def preserve_lockfile() -> Generator[None]:
    """Preserve the lockfile by moving it to a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir_name:
        shutil.move("uv.lock", f"{temp_dir_name}/uv.lock")
        try:
            yield
        finally:
            shutil.move(f"{temp_dir_name}/uv.lock", "uv.lock")


@nox.session(reuse_venv=True, default=True)
def lint(session: nox.Session) -> None:
    """Run the linter."""
    if shutil.which("prek") is None:
        session.install("prek")

    session.run("prek", "run", "--all-files", *session.posargs, external=True)


@nox.session(name="cpp-lint", reuse_venv=True, venv_backend="uv")
def cpp_lint(session: nox.Session) -> None:
    """Reproduce the CI cpp-linter check for changed or all C++ files."""
    all_files = session.posargs == ["--all"]
    if not all_files and (len(session.posargs) > 1 or (session.posargs and session.posargs[0].startswith("-"))):
        session.error("pass --all or at most one diff base")
    diff_base = session.posargs[0] if session.posargs else "origin/main"

    if shutil.which("cmake") is None:
        session.install("cmake")
    if shutil.which("ninja") is None:
        session.install("ninja")
    # Keep this group aligned with cpp-linter-action v2.21.0 and its inputs.
    session.install("--group", "cpp-lint")

    clang_tidy = shutil.which("clang-tidy-22") or shutil.which("clang-tidy")
    if clang_tidy is None:
        session.error("clang-tidy 22 is required")
    llvm_bin = Path(clang_tidy).resolve().parent
    version = session.run(llvm_bin / "clang-tidy", "--version", external=True, silent=True)
    if "version 22." not in (version or ""):
        session.error("clang-tidy 22 is required")

    compiler_env = {
        "CC": str(llvm_bin / "clang"),
        "CXX": str(llvm_bin / "clang++"),
    }
    session.run(
        "cmake",
        "-B",
        "build/cpp-lint",
        "--preset",
        "lint",
        env=compiler_env,
        external=True,
    )
    session.run("cmake", "--build", "build/cpp-lint", env=compiler_env, external=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "github-output"
        session.run(
            "cpp-linter",
            "--style=",
            "--tidy-checks=",
            f"--version={llvm_bin}",
            "--ignore=build|!build/mlir/**|**/include|include|vendor/**",
            "--thread-comments=false",
            "--step-summary=false",
            "--database=build/cpp-lint",
            "--extra-arg=-std=c++20",
            "--extra-arg=-Wunused-template",
            f"--files-changed-only={'false' if all_files else 'true'}",
            "--lines-changed-only=false",
            *(() if all_files else (f"--diff-base={diff_base}",)),
            "--file-annotations=false",
            "--jobs=0",
            "--verbosity=info",
            env={"GITHUB_OUTPUT": str(output)},
        )
        results = dict(line.split("=", 1) for line in output.read_text().splitlines())
        if int(results["checks-failed"]) != 0:
            session.error(f"cpp-linter reported {results['checks-failed']} finding(s)")


def _run_tests(
    session: nox.Session,
    *,
    install_args: Sequence[str] = (),
    extra_command: Sequence[str] = (),
    pytest_run_args: Sequence[str] = (),
) -> None:
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    if shutil.which("cmake") is None:
        session.install("cmake")
    if shutil.which("ninja") is None:
        session.install("ninja")

    # install build and test dependencies on top of the existing environment
    session.run(
        "uv",
        "sync",
        "--inexact",
        "--only-group",
        "build",
        "--only-group",
        "test",
        *install_args,
        env=env,
    )
    session.run(
        "uv",
        "sync",
        "--inexact",
        "--no-dev",  # do not auto-install dev dependencies
        "--no-build-isolation-package",
        "mqt-core",  # build the project without isolation
        *install_args,
        env=env,
    )
    if extra_command:
        session.run(*extra_command, env=env)
    session.run(
        "uv",
        "run",
        "--no-sync",  # do not sync as everything is already installed
        *install_args,
        "pytest",
        *pytest_run_args,
        *session.posargs,
        "--cov-config=pyproject.toml",
        env=env,
    )


@nox.session(python=PYTHON_ALL_VERSIONS, reuse_venv=True, default=True)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    _run_tests(session)


@nox.session(python=PYTHON_ALL_VERSIONS, reuse_venv=True, venv_backend="uv", default=True)
def minimums(session: nox.Session) -> None:
    """Test the minimum versions of dependencies."""
    with preserve_lockfile():
        _run_tests(
            session,
            install_args=["--resolution=lowest-direct"],
            pytest_run_args=["-Wdefault"],
        )
        env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
        session.run("uv", "tree", "--frozen", env=env)


@nox.session(reuse_venv=True, venv_backend="uv", python=PYTHON_ALL_VERSIONS)
def qiskit(session: nox.Session) -> None:
    """Test against Qiskit main with its exact extension headers."""
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    with preserve_lockfile():
        if shutil.which("cmake") is None:
            session.install("cmake")
        if shutil.which("ninja") is None:
            session.install("ninja")
        session.run(
            "uv",
            "sync",
            "--inexact",
            "--only-group",
            "build",
            "--only-group",
            "test",
            env=env,
        )
        session.run(
            "uv",
            "pip",
            "install",
            "--reinstall",
            "qiskit[qasm3-import] @ git+https://github.com/Qiskit/qiskit.git",
            env=env,
        )
        details = session.run(
            "uv",
            "run",
            "--no-sync",
            "python",
            "-c",
            ("import qiskit; from qiskit import capi; print(qiskit.__version__); print(capi.get_include())"),
            env=env,
            silent=True,
        )
        if details is None:
            session.error("failed to inspect the installed Qiskit build")
        detail_lines = details.strip().splitlines()
        if len(detail_lines) < 2:
            session.error("installed Qiskit did not report its version and C API include directory")
        version, include = detail_lines[-2:]
        include_path = Path(include).resolve(strict=True)
        cmake_args = [
            argument
            for argument in os.environ.get("SKBUILD_CMAKE_ARGS", "").split(";")
            if argument
            and not argument.startswith("-DMQT_QISKIT_CAPI_CANDIDATE_INCLUDE=")
            and not argument.startswith("-DMQT_QISKIT_CAPI_CANDIDATE_VERSION=")
        ]
        cmake_args.extend([
            f"-DMQT_QISKIT_CAPI_CANDIDATE_INCLUDE={include_path}",
            f"-DMQT_QISKIT_CAPI_CANDIDATE_VERSION={version}",
        ])
        env["SKBUILD_CMAKE_ARGS"] = ";".join(cmake_args)
        env["MQT_QISKIT_TEST_CANDIDATE_VERSION"] = version
        session.run(
            "uv",
            "sync",
            "--inexact",
            "--no-dev",
            "--no-build-isolation-package",
            "mqt-core",
            "--no-install-package",
            "qiskit",
            env=env,
        )
        session.run(
            "uv",
            "run",
            "--no-sync",
            "pytest",
            *session.posargs,
            "--cov-config=pyproject.toml",
            env=env,
        )
        session.run("uv", "pip", "show", "qiskit", env=env)


@nox.session(python="3.14", reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the docs.

    Use ``--non-interactive`` to avoid serving.
    Pass ``-b linkcheck`` to check links.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", dest="builder", default="html", help="Build target (default: html)")
    args, posargs = parser.parse_known_args(session.posargs)

    serve = args.builder == "html" and session.interactive
    if serve:
        session.install("sphinx-autobuild")

    env = {
        "UV_PROJECT_ENVIRONMENT": session.virtualenv.location,
        "SKBUILD_CMAKE_BUILD_TYPE": "MinSizeRel",
        # Let scikit-build-core generate the MLIR reference pages while it
        # builds the extension used to execute the documentation examples.
        # The docs exercise only the DDSIM provider. The common Python package
        # configuration enables unity builds.
        "SKBUILD_CMAKE_ARGS": "-DBUILD_MQT_CORE_DOCUMENTATION=ON;-DBUILD_MQT_CORE_QDMI_SC_DEVICE=OFF",
    }

    session.run(
        "uv",
        "sync",
        "--inexact",
        "--only-group",
        "build",
        "--only-group",
        "docs",
        env=env,
    )
    session.run(
        "uv",
        "run",
        "--no-dev",  # do not auto-install dev dependencies
        "--no-build-isolation-package",
        "mqt-core",  # build the project without isolation
        "sphinx-autobuild" if serve else "sphinx-build",
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        "-E",  # Doxygen inventories are regenerated for every docs build
        "-W",  # fail the session if documentation diagnostics are emitted
        f"-b={args.builder}",
        "docs",
        f"docs/_build/{args.builder}",
        *posargs,
        env=env,
    )


@nox.session(reuse_venv=True, venv_backend="uv")
def stubs(session: nox.Session) -> None:
    """Generate type stubs for Python bindings using nanobind."""
    env = {
        "UV_PROJECT_ENVIRONMENT": session.virtualenv.location,
        # Stub generation only imports the extension modules, so this build
        # favors compilation speed over optimized code. The settings match the
        # documentation build, which lets both sessions share a build tree.
        "SKBUILD_CMAKE_BUILD_TYPE": "MinSizeRel",
        "SKBUILD_CMAKE_ARGS": "-DBUILD_MQT_CORE_QDMI_SC_DEVICE=OFF",
    }

    session.run("uv", "sync", "--inexact", "--only-group", "build", env=env)
    session.run(
        "uv",
        "sync",
        "--no-dev",
        "--group",
        "build",
        "--no-build-isolation-package",
        "mqt-core",  # build the project without isolation
        env=env,
    )

    package_root = Path(__file__).parent / "python" / "mqt" / "core"

    session.run(
        "python",
        "-m",
        "nanobind.stubgen",
        "--recursive",
        "--include-private",
        "--output-dir",
        str(package_root),
        "--module",
        "mqt.core.ir",
        "--module",
        "mqt.core.dd",
        "--module",
        "mqt.core.qdmi",
        "--module",
        "mqt.core.mlir",
        "--pattern-file",
        "bindings/patterns.txt",
    )

    pyi_files = list(package_root.glob("**/*.pyi"))

    if not pyi_files:
        session.warn("No .pyi files found")
        return

    if shutil.which("prek") is None:
        session.install("prek")

    # Allow both 0 (no issues) and 1 as success codes for fixing up stubs
    success_codes = [0, 1]
    session.run("prek", "run", "license-tools", "--files", *pyi_files, external=True, success_codes=success_codes)
    session.run("prek", "run", "ruff-check", "--files", *pyi_files, external=True, success_codes=success_codes)
    session.run("prek", "run", "ruff-format", "--files", *pyi_files, external=True, success_codes=success_codes)

    # Run ruff-check again to ensure everything is clean
    session.run("prek", "run", "ruff-check", "--files", *pyi_files, external=True)


if __name__ == "__main__":
    nox.main()

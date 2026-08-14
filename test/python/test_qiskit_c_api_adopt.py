# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for restartable Qiskit C API adoption."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from packaging.version import Version

if sys.version_info < (3, 14):
    pytest.skip("the Qiskit C API adoption script requires Python 3.14", allow_module_level=True)

SCRIPT = Path(__file__).parents[2] / "scripts" / "qiskit_c_api_adopt.py"
SPEC = importlib.util.spec_from_file_location("qiskit_c_api_adopt", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
adopt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adopt)


def test_vendored_snapshot_matches_provenance() -> None:
    """Keep mutation-disabled vendored headers byte-identical to provenance."""
    snapshot = SCRIPT.parents[1] / "vendor" / "qiskit-c-api" / "2.5.0"
    provenance = json.loads((snapshot / "PROVENANCE.json").read_text())

    actual_files = {"LICENSE"} | {
        path.relative_to(snapshot).as_posix() for path in (snapshot / "include").rglob("*") if path.is_file()
    }
    assert actual_files == set(provenance["files"])
    for relative, expected in provenance["files"].items():
        assert hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() == expected


def test_write_vendor_tree_reuses_only_exact_generated_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse matching recorded content and reject changes to those files."""
    root = tmp_path / "repository"
    vendor_root = root / "vendor" / "qiskit-c-api"
    vendor_root.mkdir(parents=True)
    implementation = root / "Qiskit2_5.cpp"
    implementation.write_text("QkThing qk_thing\n")
    monkeypatch.setattr(adopt, "VENDOR_ROOT", vendor_root)
    monkeypatch.setattr(adopt, "TRANSLATION_IMPLEMENTATION", implementation)

    wheel = tmp_path / "qiskit.whl"
    headers = {
        "qiskit/include/qiskit.h": "",
        "qiskit/include/qiskit/version.h": '#define QISKIT_VERSION "2.6.0"\n',
        "qiskit/include/qiskit/funcs.h": "QkThing qk_thing(void);\n",
        "qiskit/include/qiskit/funcs_py.h": "",
        "qiskit/include/qiskit/funcs_py_generated.h": ("#define qk_thing (*(QkThing(*)())(_Qk_API_Test[1]))\n"),
        "qiskit/include/qiskit/types.h": "typedef int QkThing;\n",
        "qiskit-2.6.0.dist-info/licenses/LICENSE.txt": "Apache-2.0\n",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, contents in headers.items():
            archive.writestr(name, contents)
    artifact = {
        "filename": wheel.name,
        "digests": {"sha256": "wheel-hash"},
        "url": "https://example.invalid/qiskit.whl",
    }

    target = adopt.write_vendor_tree("2.6.0", wheel, artifact)
    expected = adopt.directory_contents(target)
    equivalent_artifact = {
        "filename": "qiskit-2.6.0-cp314-manylinux.whl",
        "digests": {"sha256": "platform-wheel-hash"},
        "url": "https://example.invalid/qiskit-manylinux.whl",
    }
    assert adopt.write_vendor_tree("2.6.0", wheel, equivalent_artifact) == target
    assert adopt.directory_contents(target) == expected

    (target / "UNTRACKED.txt").write_text("local note\n")
    assert adopt.write_vendor_tree("2.6.0", wheel, equivalent_artifact) == target

    (target / "LICENSE").write_text("modified\n")
    with pytest.raises(RuntimeError, match="does not match the vendored content"):
        adopt.write_vendor_tree("2.6.0", wheel, artifact)

    (target / "LICENSE").write_bytes(b"Apache-2.0\n")
    (target / "API_SURFACE.json").write_text("{}\n")
    assert adopt.write_vendor_tree("2.6.0", wheel, equivalent_artifact) == target


def test_function_declarations_ignore_preprocessor_directives() -> None:
    """Keep API declarations free of include guards and macro branches."""
    declarations = adopt.function_declarations(
        """#ifndef QISKIT_FUNCS_H
#define QISKIT_FUNCS_H
#if defined(_WIN32)
extern "C" {
QkThing qk_thing(void);
}
#endif
#endif
"""
    )

    assert declarations == {"qk_thing": "QkThing qk_thing(void);"}


def test_function_declarations_record_definitions_as_signatures() -> None:
    """Stop at the opening brace of a header-defined function."""
    declarations = adopt.function_declarations(
        """extern "C" {
static int qk_import(
    void
) {
  QkThing *thing = qk_thing();
  return thing == 0;
}
}
"""
    )

    assert declarations["qk_import"] == "static int qk_import(void);"


def test_write_vendor_tree_rejects_mismatched_embedded_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject headers that do not embed the requested Qiskit release."""
    root = tmp_path / "repository"
    vendor_root = root / "vendor" / "qiskit-c-api"
    vendor_root.mkdir(parents=True)
    implementation = root / "Qiskit2_5.cpp"
    implementation.write_text("QkThing qk_thing\n")
    monkeypatch.setattr(adopt, "VENDOR_ROOT", vendor_root)
    monkeypatch.setattr(adopt, "TRANSLATION_IMPLEMENTATION", implementation)

    wheel = tmp_path / "qiskit.whl"
    headers = {
        "qiskit/include/qiskit.h": "",
        "qiskit/include/qiskit/version.h": '#define QISKIT_VERSION "2.5.0"\n',
        "qiskit/include/qiskit/funcs.h": "QkThing qk_thing(void);\n",
        "qiskit/include/qiskit/funcs_py.h": "",
        "qiskit/include/qiskit/funcs_py_generated.h": "",
        "qiskit/include/qiskit/types.h": "typedef int QkThing;\n",
        "qiskit-2.5.0.dist-info/licenses/LICENSE.txt": "Apache-2.0\n",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, contents in headers.items():
            archive.writestr(name, contents)
    artifact = {
        "filename": wheel.name,
        "digests": {"sha256": "wheel-hash"},
        "url": "https://example.invalid/qiskit.whl",
    }

    with pytest.raises(RuntimeError, match="do not embed the requested Qiskit version"):
        adopt.write_vendor_tree("2.6.0", wheel, artifact)


def test_download_wheel_rejects_http_before_network_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject transport without TLS before opening a connection."""
    monkeypatch.setattr(adopt.urllib.request, "urlopen", lambda *_args, **_kwargs: pytest.fail("network used"))
    artifact = {
        "url": "http://example.invalid/qiskit.whl",
        "digests": {"sha256": hashlib.sha256(b"wheel").hexdigest()},
    }

    with pytest.raises(RuntimeError, match="requires an HTTPS URL"):
        adopt.download_wheel(artifact, tmp_path / "qiskit.whl")


def test_download_wheel_rejects_hash_mismatch_and_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not publish or retain a wheel that fails provenance verification."""
    monkeypatch.setattr(adopt.urllib.request, "urlopen", lambda *_args, **_kwargs: BytesIO(b"wheel"))
    artifact = {
        "url": "https://example.invalid/qiskit.whl",
        "digests": {"sha256": hashlib.sha256(b"different").hexdigest()},
    }
    destination = tmp_path / "qiskit.whl"

    with pytest.raises(RuntimeError, match="does not match its PyPI SHA-256"):
        adopt.download_wheel(artifact, destination)

    assert not destination.exists()
    assert not destination.with_suffix(".whl.part").exists()


def test_translation_registration_is_idempotent_and_detects_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume before or after registration without duplicating generated state."""
    root = tmp_path / "repository"
    translation_dir = root / "bindings" / "mlir" / "qiskit"
    translation_dir.mkdir(parents=True)
    template = translation_dir / "QiskitMinor.cpp.in"
    template.write_text("@QISKIT_FACTORY@ @QISKIT_MAJOR@ @QISKIT_MINOR@ @QISKIT_EXACT_API@ @QISKIT_LABEL@\n")
    registry = translation_dir / "SupportedVersions.inc"
    registry.write_text("// translations\n")
    monkeypatch.setattr(adopt, "ROOT", root)
    monkeypatch.setattr(adopt, "TRANSLATION_TEMPLATE", template)
    monkeypatch.setattr(adopt, "REGISTRY", registry)

    version = Version("2.6.0")
    adopt.register_translation(version)
    adopt.register_translation(version)

    destination, source, registration = adopt.translation_artifacts(version)
    assert destination.read_text() == source
    assert registry.read_text().count(registration) == 1

    destination.write_text("conflicting source\n")
    with pytest.raises(RuntimeError, match="not the generated source"):
        adopt.register_translation(version)


def test_translation_registration_preserves_patch_minimum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep later patch adoptions from widening support to earlier releases."""
    translation_dir = tmp_path / "bindings" / "mlir" / "qiskit"
    translation_dir.mkdir(parents=True)
    template = translation_dir / "QiskitMinor.cpp.in"
    template.write_text("@QISKIT_FACTORY@ @QISKIT_MAJOR@ @QISKIT_MINOR@\n")
    monkeypatch.setattr(adopt, "TRANSLATION_TEMPLATE", template)

    _, _, registration = adopt.translation_artifacts(Version("2.6.2"))

    assert registration == 'MQT_QISKIT_VERSION(2, 6, 2_6, 2, 2.6.2, ">=2.6.2,<2.7.0")'


def test_build_preserves_unrelated_cmake_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Change only candidate-specific CMake arguments during validation."""
    environments: list[dict[str, str]] = []

    def record_run(
        *_command: str,
        env: dict[str, str] | None = None,
        timeout: int = adopt.BUILD_TIMEOUT,
    ) -> None:
        assert env is not None
        assert timeout == adopt.BUILD_TIMEOUT
        environments.append(env)

    monkeypatch.setattr(adopt, "run", record_run)
    monkeypatch.setenv(
        "SKBUILD_CMAKE_ARGS",
        "-DMLIR_DIR=/toolchain;-DMQT_QISKIT_CAPI_CANDIDATE_VERSION=old",
    )

    adopt.build_and_test("2.6.0", tmp_path / "include", tmp_path / "candidate", candidate=True)
    candidate_args = environments[-1]["SKBUILD_CMAKE_ARGS"].split(";")
    assert candidate_args == [
        "-DMLIR_DIR=/toolchain",
        f"-DMQT_QISKIT_CAPI_CANDIDATE_INCLUDE={tmp_path / 'include'}",
        "-DMQT_QISKIT_CAPI_CANDIDATE_VERSION=2.6.0",
    ]

    adopt.build_and_test("2.6.0", tmp_path / "include", tmp_path / "shipping", candidate=False)
    assert environments[-1]["SKBUILD_CMAKE_ARGS"] == "-DMLIR_DIR=/toolchain"


def test_restartable_worktree_rejects_unrelated_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Permit exact interrupted-run artifacts but retain clean-worktree isolation."""
    root = tmp_path / "repository"
    translation_dir = root / "bindings" / "mlir" / "qiskit"
    translation_dir.mkdir(parents=True)
    template = translation_dir / "QiskitMinor.cpp.in"
    template.write_text("@QISKIT_FACTORY@ @QISKIT_MAJOR@ @QISKIT_MINOR@\n")
    registry = translation_dir / "SupportedVersions.inc"
    registry.write_text("// translations\n")
    vendor_root = root / "vendor" / "qiskit-c-api"
    vendor_root.mkdir(parents=True)
    monkeypatch.setattr(adopt, "ROOT", root)
    monkeypatch.setattr(adopt, "TRANSLATION_TEMPLATE", template)
    monkeypatch.setattr(adopt, "REGISTRY", registry)
    monkeypatch.setattr(adopt, "VENDOR_ROOT", vendor_root)

    git = shutil.which("git")
    assert git is not None

    def run_git(*arguments: str) -> None:
        subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [git, *arguments], cwd=root, check=True, timeout=30
        )

    run_git("init", "-q")
    run_git("config", "user.name", "Test")
    run_git("config", "user.email", "test@example.com")
    run_git("config", "commit.gpgsign", "false")
    run_git("add", ".")
    run_git("commit", "-qm", "fixture")

    version = Version("2.6.0")
    adopt.register_translation(version)
    adopt.require_restartable_worktree(git, version)

    (root / "unrelated.txt").write_text("user change\n")
    with pytest.raises(RuntimeError, match="unrelated worktree changes"):
        adopt.require_restartable_worktree(git, version)

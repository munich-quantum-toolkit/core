# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for restartable Qiskit C-API adoption."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from packaging.version import Version

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

    actual_files = {
        path.relative_to(snapshot).as_posix() for path in (snapshot / "include").rglob("*") if path.is_file()
    }
    assert actual_files == set(provenance["files"])
    for relative, expected in provenance["files"].items():
        assert hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() == expected


def test_write_vendor_tree_reuses_only_exact_generated_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume with an exact snapshot and reject locally altered vendor files."""
    root = tmp_path / "repository"
    vendor_root = root / "vendor" / "qiskit-c-api"
    vendor_root.mkdir(parents=True)
    implementation = root / "Adapter25.cpp"
    implementation.write_text("QkThing qk_thing\n")
    monkeypatch.setattr(adopt, "VENDOR_ROOT", vendor_root)
    monkeypatch.setattr(adopt, "ADAPTER_IMPLEMENTATION", implementation)

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
    assert adopt.write_vendor_tree("2.6.0", wheel, artifact) == target
    assert adopt.directory_contents(target) == expected

    (target / "LICENSE").write_text("modified\n")
    with pytest.raises(RuntimeError, match="does not match the exact wheel"):
        adopt.write_vendor_tree("2.6.0", wheel, artifact)


def test_adapter_registration_is_idempotent_and_detects_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume before or after registration without duplicating generated state."""
    root = tmp_path / "repository"
    adapter_dir = root / "bindings" / "mlir" / "qiskit"
    adapter_dir.mkdir(parents=True)
    template = adapter_dir / "AdapterMinor.cpp.in"
    template.write_text(
        "@ADAPTER_FACTORY@ @ADAPTER_MAJOR@ @ADAPTER_MINOR@ @ADAPTER_FINAL_ONLY@ @ADAPTER_EXACT_API@ @ADAPTER_LABEL@\n"
    )
    registry = adapter_dir / "SupportedAdapters.inc"
    registry.write_text("// adapters\n")
    monkeypatch.setattr(adopt, "ROOT", root)
    monkeypatch.setattr(adopt, "ADAPTER_TEMPLATE", template)
    monkeypatch.setattr(adopt, "REGISTRY", registry)

    version = Version("2.6.0")
    adopt.register_adapter(version)
    adopt.register_adapter(version)

    destination, source, registration = adopt.adapter_artifacts(version)
    assert destination.read_text() == source
    assert registry.read_text().count(registration) == 1

    destination.write_text("conflicting source\n")
    with pytest.raises(RuntimeError, match="not the generated source"):
        adopt.register_adapter(version)


def test_restartable_worktree_rejects_unrelated_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Permit exact interrupted-run artifacts but retain clean-worktree isolation."""
    root = tmp_path / "repository"
    adapter_dir = root / "bindings" / "mlir" / "qiskit"
    adapter_dir.mkdir(parents=True)
    template = adapter_dir / "AdapterMinor.cpp.in"
    template.write_text("@ADAPTER_FACTORY@ @ADAPTER_MAJOR@ @ADAPTER_MINOR@\n")
    registry = adapter_dir / "SupportedAdapters.inc"
    registry.write_text("// adapters\n")
    vendor_root = root / "vendor" / "qiskit-c-api"
    vendor_root.mkdir(parents=True)
    monkeypatch.setattr(adopt, "ROOT", root)
    monkeypatch.setattr(adopt, "ADAPTER_TEMPLATE", template)
    monkeypatch.setattr(adopt, "REGISTRY", registry)
    monkeypatch.setattr(adopt, "VENDOR_ROOT", vendor_root)

    git = shutil.which("git")
    assert git is not None

    def run_git(*arguments: str) -> None:
        subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [git, *arguments], cwd=root, check=True
        )

    run_git("init", "-q")
    run_git("config", "user.name", "Test")
    run_git("config", "user.email", "test@example.com")
    run_git("config", "commit.gpgsign", "false")
    run_git("add", ".")
    run_git("commit", "-qm", "fixture")

    version = Version("2.6.0")
    adopt.register_adapter(version)
    adopt.require_restartable_worktree(git, version)

    (root / "unrelated.txt").write_text("user change\n")
    with pytest.raises(RuntimeError, match="unrelated worktree changes"):
        adopt.require_restartable_worktree(git, version)

# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exercise source identity and stale-version rejection before documentation upload."""

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

publication_target = runpy.run_path(str(Path(__file__).parents[2] / "scripts/docs_publication.py"))[
    "publication_target"
]
SHA = "a" * 40
REPO = "munich-quantum-toolkit/core"


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    """Create a complete artifact for a fork PR.

    Returns:
        The artifact directory.
    """
    (tmp_path / "html").mkdir()
    (tmp_path / "html/index.html").write_text("<h1>Documentation</h1>", encoding="utf-8")
    (tmp_path / "html.zip").touch()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"commit": SHA, "pull_request": 42, "ref": "refs/pull/42/merge"}), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def run() -> dict[str, Any]:
    """Describe the GitHub run independently of its artifact.

    Returns:
        The triggering run metadata.
    """
    return {
        "path": ".github/workflows/docs.yml",
        "repository": {"full_name": REPO},
        "head_repository": {"id": 7},
        "head_branch": "docs-change",
        "head_sha": SHA,
        "conclusion": "success",
        "event": "pull_request",
    }


def test_fork_preview_and_superseded_commits(artifact: Path, run: dict[str, Any]) -> None:
    """Only publish an open PR's current head from the originating repository."""
    pull: dict[str, Any] = {
        "base": {"repo": {"full_name": REPO}},
        "head": {"repo": {"id": 7}, "ref": "docs-change", "sha": SHA},
        "state": "open",
    }
    responses = {f"repos/{REPO}/pulls/42": pull}
    assert publication_target(artifact, run, REPO, responses.__getitem__) == ("42", "external")
    pull["head"]["sha"] = "b" * 40
    assert publication_target(artifact, run, REPO, responses.__getitem__) is None
    pull["head"]["sha"] = SHA
    pull["state"] = "closed"
    pull["head"] = {**pull["head"], "repo": None}
    assert publication_target(artifact, run, REPO, responses.__getitem__) is None
    pull["state"] = "open"
    pull["head"]["repo"] = {"id": 8}
    with pytest.raises(ValueError, match="repository and branch"):
        publication_target(artifact, run, REPO, responses.__getitem__)


@pytest.mark.parametrize(("kind", "version"), [("heads", "main"), ("tags", "v4.2.0")])
def test_current_branch_and_annotated_tag(artifact: Path, run: dict[str, Any], kind: str, version: str) -> None:
    """Resolve annotated tags and reject refs that moved after the build."""
    run.update(event="push", head_branch=version)
    (artifact / "manifest.json").write_text(
        json.dumps({"commit": SHA, "ref": f"refs/{kind}/{version}"}), encoding="utf-8"
    )
    commit = {"type": "commit", "sha": SHA}
    responses = {
        f"repos/{REPO}/git/ref/{kind}/{version}": {"object": commit},
        f"repos/{REPO}/git/tags/tag-id": {"object": commit},
    }
    if kind == "tags":
        responses[f"repos/{REPO}/git/ref/{kind}/{version}"] = {"object": {"type": "tag", "sha": "tag-id"}}
    assert publication_target(artifact, run, REPO, responses.__getitem__) == (
        version,
        "branch" if kind == "heads" else "tag",
    )
    commit["sha"] = "b" * 40
    assert publication_target(artifact, run, REPO, responses.__getitem__) is None


def test_reject_mismatched_or_incomplete_artifact(artifact: Path, run: dict[str, Any]) -> None:
    """Reject substituted commits, missing content, and filesystem links before API access."""

    def unexpected_api(endpoint: str) -> dict[str, Any]:
        pytest.fail(f"Unexpected API access: {endpoint}")

    run["head_sha"] = "b" * 40
    with pytest.raises(ValueError, match="triggering commit"):
        publication_target(artifact, run, REPO, unexpected_api)
    run["head_sha"] = SHA
    (artifact / "html.zip").unlink()
    with pytest.raises(ValueError, match="missing HTML"):
        publication_target(artifact, run, REPO, unexpected_api)
    (artifact / "html.zip").touch()
    (artifact / "html/leak").symlink_to(artifact / "manifest.json")
    with pytest.raises(ValueError, match="symlinks"):
        publication_target(artifact, run, REPO, unexpected_api)

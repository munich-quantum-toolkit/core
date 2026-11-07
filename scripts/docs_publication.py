#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Validate a documentation artifact before publishing it from a trusted workflow."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def publication_target(
    artifact: Path,
    run: dict[str, Any],
    repository: str,
    api: Callable[[str], dict[str, Any]],
) -> tuple[str, str] | None:
    """Return the current version and type, or skip a superseded build.

    Raises:
        ValueError: If the artifact or run does not identify the expected source.
    """
    if run["path"] != ".github/workflows/docs.yml" or run["repository"]["full_name"] != repository:
        msg = "Unexpected documentation workflow or repository"
        raise ValueError(msg)
    if run["conclusion"] != "success" or run["event"] not in {"push", "pull_request"}:
        return None
    if artifact.is_symlink() or any(path.is_symlink() for path in artifact.rglob("*")):
        msg = "Documentation artifacts must contain regular files, not symlinks"
        raise ValueError(msg)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    commit = manifest["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or commit != run["head_sha"]:
        msg = "Documentation artifact does not match the triggering commit"
        raise ValueError(msg)
    if not (artifact / "html/index.html").is_file() or not (artifact / "html.zip").is_file():
        msg = "Documentation artifact is missing HTML or its offline ZIP"
        raise ValueError(msg)

    if run["event"] == "pull_request":
        number = manifest["pull_request"]
        if type(number) is not int or number <= 0:
            msg = "Documentation artifact has no pull request number"
            raise ValueError(msg)
        pull = api(f"repos/{repository}/pulls/{number}")
        if pull["state"] != "open" or pull["head"]["sha"] != commit:
            return None
        if (
            pull["base"]["repo"]["full_name"] != repository
            or pull["head"]["repo"]["id"] != run["head_repository"]["id"]
            or pull["head"]["ref"] != run["head_branch"]
        ):
            msg = "Pull request does not match the triggering repository and branch"
            raise ValueError(msg)
        return str(number), "external"

    ref = manifest["ref"]
    if ref == "refs/heads/main":
        version, kind = "main", "branch"
    elif re.fullmatch(r"refs/tags/v[\w.-]+", ref):
        version, kind = ref.removeprefix("refs/tags/"), "tag"
    else:
        return None
    if version != run["head_branch"]:
        msg = "Documentation version does not match the triggering ref"
        raise ValueError(msg)
    target = api(f"repos/{repository}/git/ref/{ref.removeprefix('refs/')}")["object"]
    while target["type"] == "tag":
        target = api(f"repos/{repository}/git/tags/{target['sha']}")["object"]
    if target["type"] != "commit" or target["sha"] != commit:
        return None
    return version, kind


def main() -> None:
    """Write a validated publication target to GitHub Actions outputs.

    Raises:
        FileNotFoundError: If the GitHub CLI is unavailable.
    """
    gh = shutil.which("gh")
    if gh is None:
        msg = "GitHub CLI is required to validate publication metadata"
        raise FileNotFoundError(msg)

    def api(endpoint: str) -> dict[str, Any]:
        return json.loads(subprocess.check_output([gh, "api", endpoint], text=True))  # ruff: ignore[subprocess-without-shell-equals-true]

    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    target = publication_target(Path(sys.argv[1]), event["workflow_run"], os.environ["GITHUB_REPOSITORY"], api)
    if target is None:
        sys.stderr.write("Skipping publication: the build no longer identifies a current version.\n")
        return
    version, kind = target
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"version={version}\ntype={kind}\n")


if __name__ == "__main__":
    main()

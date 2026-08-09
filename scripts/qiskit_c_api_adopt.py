#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Prepare a released Qiskit C-API minor adapter as a reviewable local diff."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import operator
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "vendor" / "qiskit-c-api"
REGISTRY = ROOT / "bindings" / "mlir" / "qiskit" / "SupportedAdapters.inc"
ADAPTER_TEMPLATE = ROOT / "bindings" / "mlir" / "qiskit" / "AdapterMinor.cpp.in"
ADAPTER_IMPLEMENTATION = ROOT / "bindings" / "mlir" / "qiskit" / "Adapter25.cpp"
LOGGER = logging.getLogger(__name__)


def run(*command: str, env: dict[str, str] | None = None) -> None:
    """Run one checked subprocess and echo it for an auditable session log."""
    LOGGER.info("$ %s", shlex.join(command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)  # ruff: ignore[subprocess-without-shell-equals-true]


def compatible_wheel(release: dict[str, Any]) -> dict[str, Any]:
    """Select the published wheel matching the maintenance interpreter.

    Returns:
        The deterministic compatible wheel entry from the PyPI response.

    Raises:
        RuntimeError: If the release has no compatible binary wheel.
    """
    supported = set(sys_tags())
    candidates: list[dict[str, Any]] = []
    for artifact in release["urls"]:
        filename = artifact["filename"]
        if artifact["packagetype"] != "bdist_wheel":
            continue
        _, _, _, wheel_tags = parse_wheel_filename(filename)
        if supported.intersection(wheel_tags):
            candidates.append(artifact)
    if not candidates:
        msg = "released Qiskit has no wheel compatible with this interpreter"
        raise RuntimeError(msg)
    return min(candidates, key=operator.itemgetter("filename"))


def strip_comments(text: str) -> str:
    """Remove comments before normalizing declarations.

    Returns:
        The input without C or C++ comments.
    """
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", text)


def normalize(text: str) -> str:
    """Normalize insignificant header whitespace.

    Returns:
        The input with whitespace runs collapsed.
    """
    return " ".join(text.split())


def typedefs(text: str) -> dict[str, str]:
    """Extract top-level typedef declarations, including structured bodies.

    Returns:
        A map from native type name to normalized declaration.
    """
    clean = strip_comments(text)
    declarations: dict[str, str] = {}
    offset = 0
    while (match := re.search(r"\btypedef\b", clean[offset:])) is not None:
        start = offset + match.start()
        depth = 0
        end = start
        while end < len(clean):
            if clean[end] == "{":
                depth += 1
            elif clean[end] == "}":
                depth -= 1
            elif clean[end] == ";" and depth == 0:
                declaration = normalize(clean[start : end + 1])
                name = re.search(r"([A-Za-z_]\w*)\s*;$", declaration)
                if name is not None:
                    declarations[name.group(1)] = declaration
                end += 1
                break
            end += 1
        offset = max(end, start + len("typedef"))
    return declarations


def function_declarations(text: str) -> dict[str, str]:
    """Extract normalized exported native function declarations.

    Returns:
        A map from native function name to normalized declaration.
    """
    clean = strip_comments(text)
    declarations: dict[str, str] = {}
    for statement in clean.split(";"):
        match = re.search(r"\b(qk_[A-Za-z0-9_]+)\s*\(", statement)
        if match is not None:
            declarations[match.group(1)] = normalize(statement + ";")
    return declarations


def capsule_functions(text: str) -> dict[str, dict[str, str | int]]:
    """Extract each extension-capsule function, table slot, and signature.

    Returns:
        A map from capsule function name to its normalized API metadata.
    """
    functions: dict[str, dict[str, str | int]] = {}
    pattern = re.compile(r"^#define\s+(qk_\w+)\s+\(\*\((.+)\)\((_Qk_API_\w+)\[(\d+)\]\)\)$")
    for line in text.splitlines():
        if (match := pattern.match(line)) is not None:
            functions[match.group(1)] = {
                "signature": normalize(match.group(2)),
                "table": match.group(3),
                "slot": int(match.group(4)),
            }
    return functions


def api_surface(include: Path) -> dict[str, Any]:
    """Build a machine-readable C-API surface for semantic comparison.

    Returns:
        The capsule functions, native declarations, and public types.
    """
    qiskit = include / "qiskit"
    implementation = ADAPTER_IMPLEMENTATION.read_text()
    used_function_names = sorted(set(re.findall(r"\bqk_[A-Za-z0-9_]+", implementation)))
    capsule = capsule_functions((qiskit / "funcs_py_generated.h").read_text())
    declarations = function_declarations((qiskit / "funcs.h").read_text() + "\n" + (qiskit / "funcs_py.h").read_text())
    types = typedefs((qiskit / "types.h").read_text())
    used_type_names = sorted(set(re.findall(r"\bQk[A-Z][A-Za-z0-9_]+", implementation)) & types.keys())
    return {
        "functions": {
            name: {
                "capsule": capsule.get(name),
                "declaration": declarations.get(name),
            }
            for name in used_function_names
        },
        "types": {name: types.get(name) for name in used_type_names},
    }


def surface_diff(previous: dict[str, Any], current: dict[str, Any]) -> str:
    """Render symbol-level additions, removals, and changes for review.

    Returns:
        A Markdown review report for the two API surfaces.
    """
    lines = ["# Qiskit C-API surface comparison", ""]
    changed = False
    for section in ("functions", "types"):
        before = previous.get(section, {})
        after = current.get(section, {})
        added = sorted(after.keys() - before.keys())
        removed = sorted(before.keys() - after.keys())
        modified = sorted(name for name in before.keys() & after.keys() if before[name] != after[name])
        lines.extend([f"## {section.replace('_', ' ').title()}", ""])
        if not (added or removed or modified):
            lines.extend(["No changes.", ""])
            continue
        changed = True
        for label, names in (("Added", added), ("Removed", removed), ("Changed", modified)):
            if names:
                lines.extend([f"- {label}: " + ", ".join(f"`{name}`" for name in names)])
        lines.append("")
    if changed:
        lines.extend([
            "Any changed or removed surface used by the adapter requires human review,",
            "even when the compilation and focused tests below succeed.",
            "",
        ])
    return "\n".join(lines)


def write_vendor_tree(version: str, wheel: Path, artifact: dict[str, Any]) -> Path:
    """Copy the wheel's C headers and license and record exact provenance.

    Returns:
        The new versioned vendor directory.

    Raises:
        RuntimeError: If the target exists or the wheel lacks required files.
    """
    target = VENDOR_ROOT / version
    if target.exists():
        msg = f"vendor directory already exists: {target}"
        raise RuntimeError(msg)
    target.mkdir(parents=True)
    prefix = "qiskit/include/"
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(wheel) as archive:
        header_names = sorted(name for name in archive.namelist() if name.startswith(prefix) and name.endswith(".h"))
        if "qiskit/include/qiskit.h" not in header_names:
            msg = "wheel does not contain qiskit/include/qiskit.h"
            raise RuntimeError(msg)
        for archive_name in header_names:
            data = archive.read(archive_name)
            relative = Path(archive_name.removeprefix(prefix))
            destination = target / "include" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            hashes[str(Path("include") / relative)] = hashlib.sha256(data).hexdigest()
        licenses = sorted(
            name
            for name in archive.namelist()
            if ".dist-info/licenses/" in name and Path(name).name.startswith("LICENSE")
        )
        if not licenses:
            msg = "wheel does not contain its license text"
            raise RuntimeError(msg)
        (target / "LICENSE").write_bytes(archive.read(licenses[0]))

    version_header = (target / "include" / "qiskit" / "version.h").read_text()
    embedded_match = re.search(r'^#define QISKIT_VERSION "([^"]+)"$', version_header, re.MULTILINE)
    if embedded_match is None or embedded_match.group(1) != version:
        msg = "wheel C-API headers do not embed the requested Qiskit version"
        raise RuntimeError(msg)
    embedded_version = embedded_match.group(1)

    surface = api_surface(target / "include")
    (target / "API_SURFACE.json").write_text(json.dumps(surface, indent=2, sort_keys=True) + "\n")
    previous_dirs = sorted(
        (path for path in VENDOR_ROOT.iterdir() if path.is_dir() and path != target),
        key=lambda path: Version(path.name),
    )
    if previous_dirs:
        previous_path = previous_dirs[-1] / "API_SURFACE.json"
        previous = (
            json.loads(previous_path.read_text())
            if previous_path.exists()
            else api_surface(previous_dirs[-1] / "include")
        )
        (target / "API_DIFF.md").write_text(surface_diff(previous, surface))

    provenance = {
        "component": "Qiskit C API headers",
        "embedded_version": embedded_version,
        "license": "Apache-2.0",
        "qiskit_version": version,
        "source": {
            "filename": artifact["filename"],
            "sha256": artifact["digests"]["sha256"],
            "url": artifact["url"],
        },
        "files": hashes,
    }
    (target / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return target


def build_and_test(version: str, include: Path, build_dir: Path, *, candidate: bool) -> None:
    """Build a clean binding and run the focused bridge tests."""
    env = os.environ.copy()
    env["SKBUILD_BUILD_DIR"] = str(build_dir)
    if candidate:
        env["SKBUILD_CMAKE_ARGS"] = ";".join([
            f"-DMQT_QISKIT_CAPI_CANDIDATE_INCLUDE={include}",
            f"-DMQT_QISKIT_CAPI_CANDIDATE_VERSION={version}",
        ])
    else:
        env.pop("SKBUILD_CMAKE_ARGS", None)
    run(
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--reinstall",
        "--refresh",
        "--no-build-isolation",
        ".",
        env=env,
    )
    run(
        sys.executable,
        "-m",
        "pytest",
        "-n0",
        "-q",
        "test/python/test_mlir_qiskit_bridge.py",
        env=env,
    )


def register_adapter(version: Version) -> None:
    """Generate one reviewed-minor TU and extend the single adapter range.

    Raises:
        RuntimeError: If the minor or its translation unit already exists.
    """
    major, minor, _ = version.release
    suffix = f"{major}{minor}"
    next_minor = minor + 1
    supported_range = f">={version},<{major}.{next_minor}.0"
    registration = f'MQT_QISKIT_ADAPTER({major}, {minor}, {suffix}, {version}, "{supported_range}")'
    current = REGISTRY.read_text()
    if re.search(rf"^MQT_QISKIT_ADAPTER\({major}, *{minor},", current, re.MULTILINE):
        msg = f"Qiskit {major}.{minor} is already registered"
        raise RuntimeError(msg)
    source = (
        ADAPTER_TEMPLATE
        .read_text()
        .replace("@ADAPTER_FACTORY@", f"createAdapter{suffix}")
        .replace("@ADAPTER_MAJOR@", str(major))
        .replace("@ADAPTER_MINOR@", str(minor))
        .replace("@ADAPTER_FINAL_ONLY@", "1")
        .replace("@ADAPTER_EXACT_API@", "0")
        .replace("@ADAPTER_LABEL@", f"{major}.{minor}")
    )
    destination = ADAPTER_TEMPLATE.with_name(f"Adapter{suffix}.cpp")
    if destination.exists():
        msg = f"adapter source already exists: {destination}"
        raise RuntimeError(msg)
    destination.write_text(source)
    REGISTRY.write_text(current.rstrip() + "\n" + registration + "\n")


def main() -> None:
    """Adopt, compare, compile, test, and finally register one released minor.

    Raises:
        RuntimeError: If validation, adoption, compilation, or testing fails.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="exact final Qiskit version, for example 2.6.0")
    args = parser.parse_args()
    parsed = Version(args.version)
    if str(parsed) != args.version or parsed.is_prerelease or parsed.is_devrelease or parsed.local:
        msg = "the adoption version must be an exact final release"
        raise RuntimeError(msg)
    if len(parsed.release) != 3:
        msg = "the adoption version must contain major, minor, and patch"
        raise RuntimeError(msg)
    git = shutil.which("git")
    if git is None:
        msg = "Qiskit C-API adoption requires Git on PATH"
        raise RuntimeError(msg)
    status = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [git, "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        msg = "Qiskit C-API adoption requires a clean worktree"
        raise RuntimeError(msg)

    with urllib.request.urlopen(f"https://pypi.org/pypi/qiskit/{parsed}/json") as response:
        release = json.load(response)
    artifact = compatible_wheel(release)
    with tempfile.TemporaryDirectory(prefix="mqt-qiskit-capi-") as temporary:
        temporary_path = Path(temporary)
        wheel = temporary_path / artifact["filename"]
        urllib.request.urlretrieve(artifact["url"], wheel)  # ruff: ignore[suspicious-url-open-usage]
        actual_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
        expected_hash = artifact["digests"]["sha256"]
        if actual_hash != expected_hash:
            msg = "downloaded Qiskit wheel does not match its PyPI SHA-256"
            raise RuntimeError(msg)
        vendor = write_vendor_tree(str(parsed), wheel, artifact)
        run("uv", "pip", "install", "--python", sys.executable, "--reinstall", str(wheel))
        build_and_test(str(parsed), vendor / "include", temporary_path / "candidate", candidate=True)
        register_adapter(parsed)
        build_and_test(str(parsed), vendor / "include", temporary_path / "shipping", candidate=False)


if __name__ == "__main__":
    main()

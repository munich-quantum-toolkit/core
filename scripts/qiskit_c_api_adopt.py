#!/usr/bin/env -S uv run --script --quiet
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "nanobind~=2.14.0",
#   "packaging>=24",
#   "pytest>=9.0.1",
#   "pytest-xdist>=3.8.0",
#   "scikit-build-core~=1.0.3",
#   "setuptools-scm>=9.2.2",
# ]
# ///

"""Prepare one released Qiskit C API minor for compiler translation."""

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
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "vendor" / "qiskit-c-api"
REGISTRY = ROOT / "bindings" / "mlir" / "qiskit" / "SupportedVersions.inc"
TRANSLATION_TEMPLATE = ROOT / "bindings" / "mlir" / "qiskit" / "QiskitMinor.cpp.in"
TRANSLATION_IMPLEMENTATION = ROOT / "bindings" / "mlir" / "qiskit" / "Qiskit2_5.cpp"
LOGGER = logging.getLogger(__name__)
BUILD_TIMEOUT = 3600
GIT_TIMEOUT = 30
NETWORK_TIMEOUT = 30

type JsonObject = dict[str, Any]


def run(
    *command: str,
    env: dict[str, str] | None = None,
    timeout: int = BUILD_TIMEOUT,
) -> None:
    """Run one checked subprocess and echo it for an auditable session log."""
    LOGGER.info("$ %s", shlex.join(command))
    # The executable and literal arguments are maintained in this script, not
    # derived from untrusted input.
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        command, cwd=ROOT, env=env, check=True, timeout=timeout
    )


def compatible_wheel(release: JsonObject) -> JsonObject:
    """Select the published wheel matching the maintenance interpreter.

    Returns:
        The deterministic compatible wheel entry from the PyPI response.

    Raises:
        RuntimeError: If the release has no compatible binary wheel.
    """
    supported = set(sys_tags())
    candidates: list[JsonObject] = []
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


def strip_preprocessor_directives(text: str) -> str:
    """Remove preprocessor directives, including continued directives.

    Returns:
        Header text without preprocessor lines.
    """
    result: list[str] = []
    continuation = False
    extern_c_brace_depth = 0
    for line in text.splitlines():
        if continuation or line.lstrip().startswith("#"):
            continuation = line.rstrip().endswith("\\")
            continue
        if line.strip() == 'extern "C" {':
            extern_c_brace_depth = 1
            continue
        if extern_c_brace_depth != 0:
            if line.strip() == "}" and extern_c_brace_depth == 1:
                extern_c_brace_depth = 0
                continue
            extern_c_brace_depth += line.count("{") - line.count("}")
        result.append(line)
    return "\n".join(result)


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
    clean = strip_preprocessor_directives(strip_comments(text))
    declarations: dict[str, str] = {}
    statement_start = 0
    brace_depth = 0
    for offset, character in enumerate(clean):
        if character == "{" and brace_depth == 0:
            statement = clean[statement_start:offset]
            if (match := re.search(r"\b(qk_[A-Za-z0-9_]+)\s*\(", statement)) is not None:
                declaration = normalize(statement)
                declaration = re.sub(r"\(\s+", "(", declaration)
                declaration = re.sub(r"\s+\)", ")", declaration)
                declarations[match.group(1)] = declaration + ";"
            brace_depth = 1
            continue
        if character == "{" and brace_depth != 0:
            brace_depth += 1
            continue
        if character == "}" and brace_depth != 0:
            brace_depth -= 1
            if brace_depth == 0:
                statement_start = offset + 1
            continue
        if character == ";" and brace_depth == 0:
            statement = clean[statement_start : offset + 1]
            if (match := re.search(r"\b(qk_[A-Za-z0-9_]+)\s*\(", statement)) is not None:
                declarations[match.group(1)] = normalize(statement)
            statement_start = offset + 1
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
    """Build a machine-readable C API surface for semantic comparison.

    Returns:
        The capsule functions, native declarations, and public types.
    """
    qiskit = include / "qiskit"
    implementation = TRANSLATION_IMPLEMENTATION.read_text()
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
    lines = ["# Qiskit C API surface comparison", ""]
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
            "Any changed or removed surface used by the translation requires human review,",
            "even when the compilation and focused tests below succeed.",
            "",
        ])
    return "\n".join(lines)


def directory_contents(root: Path) -> dict[Path, bytes]:
    """Read a directory as relative paths and exact file contents.

    Returns:
        The directory's regular files, keyed by relative path.
    """
    return {path.relative_to(root): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def vendored_files(root: Path) -> dict[Path, bytes]:
    """Read the files covered by the snapshot provenance.

    Returns:
        The recorded vendored files, keyed by relative path.

    Raises:
        RuntimeError: If the provenance is missing or malformed.
    """
    provenance_path = root / "PROVENANCE.json"
    if not provenance_path.is_file():
        msg = f"vendored snapshot has no provenance: {root}"
        raise RuntimeError(msg)
    provenance = json.loads(provenance_path.read_text())
    files = provenance.get("files")
    if not isinstance(files, dict) or not all(isinstance(path, str) for path in files):
        msg = f"vendored snapshot has invalid file provenance: {root}"
        raise RuntimeError(msg)
    paths = {Path(path) for path in files}
    contents = directory_contents(root)
    if any(path not in contents for path in paths):
        msg = f"vendored snapshot is missing recorded content: {root}"
        raise RuntimeError(msg)
    return {path: contents[path] for path in paths}


def atomic_write_text(path: Path, text: str) -> None:
    """Replace one text file without exposing partial contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def populate_vendor_tree(target: Path, version: str, wheel: Path, artifact: JsonObject) -> None:
    """Populate a fresh directory from one exact wheel.

    Raises:
        RuntimeError: If the wheel does not contain the required exact release files.
    """
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
        license_text = archive.read(licenses[0])
        (target / "LICENSE").write_bytes(license_text)
        hashes["LICENSE"] = hashlib.sha256(license_text).hexdigest()

    version_header = (target / "include" / "qiskit" / "version.h").read_text()
    embedded_match = re.search(r'^#define QISKIT_VERSION "([^"]+)"$', version_header, re.MULTILINE)
    if embedded_match is None or embedded_match.group(1) != version:
        msg = "wheel C API headers do not embed the requested Qiskit version"
        raise RuntimeError(msg)
    embedded_version = embedded_match.group(1)

    surface = api_surface(target / "include")
    atomic_write_text(target / "API_SURFACE.json", json.dumps(surface, indent=2, sort_keys=True) + "\n")
    previous_dirs: list[Path] = []
    for path in VENDOR_ROOT.iterdir():
        if not path.is_dir() or path.name == version:
            continue
        try:
            Version(path.name)
        except ValueError:
            continue
        previous_dirs.append(path)
    previous_dirs.sort(key=lambda path: Version(path.name))
    if previous_dirs:
        previous_path = previous_dirs[-1] / "API_SURFACE.json"
        previous = (
            json.loads(previous_path.read_text())
            if previous_path.exists()
            else api_surface(previous_dirs[-1] / "include")
        )
        atomic_write_text(target / "API_DIFF.md", surface_diff(previous, surface))

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
    atomic_write_text(target / "PROVENANCE.json", json.dumps(provenance, indent=2, sort_keys=True) + "\n")


def write_vendor_tree(version: str, wheel: Path, artifact: JsonObject) -> Path:
    """Create or verify a vendored C API snapshot from a compatible wheel.

    Returns:
        The matching versioned vendor directory.

    Raises:
        RuntimeError: If recorded content differs or the wheel is incomplete.
    """
    target = VENDOR_ROOT / version
    VENDOR_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mqt-qiskit-vendor-", dir=VENDOR_ROOT) as temporary:
        expected = Path(temporary) / version
        populate_vendor_tree(expected, version, wheel, artifact)
        if target.exists():
            if vendored_files(target) != vendored_files(expected):
                msg = f"existing vendor directory does not match the vendored content: {target}"
                raise RuntimeError(msg)
            LOGGER.info("Reusing matching vendored Qiskit C API snapshot: %s", target)
            return target
        expected.replace(target)
        return target


def build_and_test(version: str, include: Path, build_dir: Path, *, candidate: bool) -> None:
    """Build a clean binding and run the focused translation tests."""
    env = os.environ.copy()
    env["SKBUILD_BUILD_DIR"] = str(build_dir)
    cmake_args = [
        argument
        for argument in env.get("SKBUILD_CMAKE_ARGS", "").split(";")
        if argument
        and not argument.startswith("-DMQT_QISKIT_CAPI_CANDIDATE_INCLUDE=")
        and not argument.startswith("-DMQT_QISKIT_CAPI_CANDIDATE_VERSION=")
    ]
    if candidate:
        cmake_args.extend([
            f"-DMQT_QISKIT_CAPI_CANDIDATE_INCLUDE={include}",
            f"-DMQT_QISKIT_CAPI_CANDIDATE_VERSION={version}",
        ])
        env["MQT_QISKIT_TEST_CANDIDATE_VERSION"] = version
    else:
        env.pop("MQT_QISKIT_TEST_CANDIDATE_VERSION", None)
    if cmake_args:
        env["SKBUILD_CMAKE_ARGS"] = ";".join(cmake_args)
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
        "test/python/test_mlir_qiskit_translation.py",
        env=env,
    )


def translation_artifacts(version: Version) -> tuple[Path, str, str]:
    """Return the generated translation path, contents, and registration."""
    major, minor, patch = version.release
    suffix = f"{major}_{minor}"
    next_minor = minor + 1
    supported_range = f">={version},<{major}.{next_minor}.0"
    registration = f'MQT_QISKIT_VERSION({major}, {minor}, {suffix}, {patch}, {version}, "{supported_range}")'
    source = (
        TRANSLATION_TEMPLATE
        .read_text()
        .replace("@QISKIT_FACTORY@", f"createQiskit{suffix}")
        .replace("@QISKIT_MAJOR@", str(major))
        .replace("@QISKIT_MINOR@", str(minor))
        .replace("@QISKIT_EXACT_API@", "0")
        .replace("@QISKIT_LABEL@", f"{major}.{minor}")
    )
    return TRANSLATION_TEMPLATE.with_name(f"Qiskit{suffix}.cpp"), source, registration


def register_translation(version: Version) -> None:
    """Create or verify one reviewed-minor TU and its registration.

    Raises:
        RuntimeError: If existing generated state is inconsistent.
    """
    major, minor, _ = version.release
    destination, source, registration = translation_artifacts(version)
    current = REGISTRY.read_text()
    existing = re.search(rf"^MQT_QISKIT_VERSION\({major}, *{minor},.*$", current, re.MULTILINE)
    if existing is not None and existing.group(0) != registration:
        msg = f"Qiskit {major}.{minor} has a conflicting translation registration"
        raise RuntimeError(msg)
    if destination.exists():
        if destination.read_text() != source:
            msg = f"existing translation source is not the generated source: {destination}"
            raise RuntimeError(msg)
    else:
        atomic_write_text(destination, source)
    if existing is None:
        atomic_write_text(REGISTRY, current.rstrip() + "\n" + registration + "\n")
    else:
        LOGGER.info("Reusing exact Qiskit %s.%s translation registration", major, minor)


def require_restartable_worktree(git: str, version: Version) -> None:
    """Allow only exact artifacts from an interrupted run in the worktree.

    Raises:
        RuntimeError: If unrelated or inconsistent local changes are present.
    """
    # Git and all arguments are resolved or generated by this script.
    status = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [git, "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT,
    )
    dirty_paths = {line[3:] for line in status.stdout.splitlines() if len(line) > 3}
    if not dirty_paths:
        return

    destination, source, registration = translation_artifacts(version)
    vendor_prefix = f"{(VENDOR_ROOT / str(version)).relative_to(ROOT).as_posix()}/"
    exact_paths = {
        destination.relative_to(ROOT).as_posix(),
        REGISTRY.relative_to(ROOT).as_posix(),
    }
    unrelated = sorted(path for path in dirty_paths if path not in exact_paths and not path.startswith(vendor_prefix))
    if unrelated:
        msg = "Qiskit C API adoption found unrelated worktree changes: " + ", ".join(unrelated)
        raise RuntimeError(msg)
    if destination.exists() and destination.read_text() != source:
        msg = f"existing translation source is not the generated source: {destination}"
        raise RuntimeError(msg)
    registry_path = REGISTRY.relative_to(ROOT).as_posix()
    if registry_path not in dirty_paths:
        return
    head = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [git, "show", f"HEAD:{registry_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT,
    ).stdout
    resumed = head.rstrip() + "\n" + registration + "\n"
    if REGISTRY.read_text() not in {head, resumed}:
        msg = "existing translation registry contains changes unrelated to this adoption"
        raise RuntimeError(msg)
    LOGGER.info("Resuming Qiskit C API adoption from exact generated artifacts")


def exact_release(text: str) -> Version:
    """Parse an exact public Qiskit release with three numeric components.

    Returns:
        The exact release version.

    Raises:
        argparse.ArgumentTypeError: If the text is not an exact public release.
    """
    parsed = Version(text)
    if (
        str(parsed) != text
        or len(parsed.release) != 3
        or parsed.is_prerelease
        or parsed.is_devrelease
        or parsed.local is not None
    ):
        msg = "expected an exact Qiskit release such as 2.6.0"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def download_wheel(artifact: JsonObject, destination: Path) -> None:
    """Download and verify one wheel before atomically publishing the file.

    Raises:
        RuntimeError: If the URL is not HTTPS or its contents fail verification.
    """
    url = artifact["url"]
    if urllib.parse.urlparse(url).scheme != "https":
        msg = "Qiskit wheel download requires an HTTPS URL"
        raise RuntimeError(msg)
    staged = destination.with_suffix(destination.suffix + ".part")
    try:
        with (
            # The HTTPS scheme is checked immediately above.
            urllib.request.urlopen(  # ruff: ignore[suspicious-url-open-usage]
                url, timeout=NETWORK_TIMEOUT
            ) as response,
            staged.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        actual_hash = hashlib.sha256(staged.read_bytes()).hexdigest()
        if actual_hash != artifact["digests"]["sha256"]:
            msg = "downloaded Qiskit wheel does not match its PyPI SHA-256"
            raise RuntimeError(msg)
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)


def main() -> None:
    """Adopt, compare, compile, test, and register one released minor.

    Raises:
        RuntimeError: If validation, adoption, compilation, or testing fails.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", type=exact_release, help="exact Qiskit release, for example 2.6.0")
    args = parser.parse_args()
    parsed: Version = args.version
    git = shutil.which("git")
    if git is None:
        msg = "Qiskit C API adoption requires Git on PATH"
        raise RuntimeError(msg)
    require_restartable_worktree(git, parsed)

    with urllib.request.urlopen(f"https://pypi.org/pypi/qiskit/{parsed}/json", timeout=NETWORK_TIMEOUT) as response:
        release = json.load(response)
    artifact = compatible_wheel(release)
    with tempfile.TemporaryDirectory(prefix="mqt-qiskit-capi-") as temporary:
        temporary_path = Path(temporary)
        wheel = temporary_path / artifact["filename"]
        download_wheel(artifact, wheel)
        vendor = write_vendor_tree(str(parsed), wheel, artifact)
        run("uv", "pip", "install", "--python", sys.executable, "--reinstall", str(wheel))
        build_and_test(str(parsed), vendor / "include", temporary_path / "candidate", candidate=True)
        register_translation(parsed)
        build_and_test(str(parsed), vendor / "include", temporary_path / "shipping", candidate=False)


if __name__ == "__main__":
    main()

"""Time cibuildwheel builds and retain evidence from the actual Windows wheels."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import struct
import subprocess
import time
import zipfile


CORE_BINARIES = (
    "mqt-core.exe",
    "mqt-core-qdmi-driver.dll",
    "mqt-core-qdmi-ddsim-device.dll",
    "mqt-core-qdmi-sc-device.dll",
)
PE_MACHINES = {"win_amd64": 0x8664, "win_arm64": 0xAA64}
PINS = re.compile(r"^(CMAKE_(CXX?|LINKER|MAKE_PROGRAM|NM|AR|BUILD_TYPE|GENERATOR|INTERPROCEDURAL_OPTIMIZATION)|"
                  r"ENABLE_(IPO|CACHE)|CIBW|(?:MLIR|LLVM)_DIR|MQT_CORE_|Python3?_EXECUTABLE)")


def timed_build_dir(project):
    return project / "build/python/Release"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def copy_if_present(source, dest):
    if source.is_file():
        shutil.copyfile(source, dest)


def start(project, result):
    build = timed_build_dir(project)
    result.mkdir(parents=True, exist_ok=True)
    pre = result / "before.ninja_log"
    pre.unlink(missing_ok=True)
    copy_if_present(build / ".ninja_log", pre)
    write_json(result / "start.json", {"start_ns": time.perf_counter_ns()})


def pe_machine(archive, name):
    with archive.open(name) as stream:
        if stream.read(2) != b"MZ":
            raise ValueError(f"{name} has no DOS signature")
        stream.seek(0x3C)
        offset = struct.unpack("<I", stream.read(4))[0]
        stream.seek(offset)
        signature, machine = struct.unpack("<4sH", stream.read(6))
        if signature != b"PE\0\0":
            raise ValueError(f"{name} has no PE signature")
        return machine


def wheel_inventory(wheel, errors):
    platform = wheel.stem.rsplit("-", 1)[1]
    machine = PE_MACHINES.get(platform)
    if machine is None:
        errors.append(f"Unexpected wheel platform {platform}")
    entries = {}
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if Path(member.filename).suffix.lower() not in (".exe", ".dll", ".pyd"):
                continue
            pe = pe_machine(archive, member.filename)
            entries[member.filename] = {
                "bytes": member.file_size,
                "compressed_bytes": member.compress_size,
                "machine": f"0x{pe:04x}",
            }
            if machine is not None and pe != machine:
                errors.append(f"Unexpected PE architecture {pe:#06x} in {member.filename}")
    names = {Path(name).name.lower() for name in entries}
    for name in CORE_BINARIES:
        if name not in names:
            errors.append(f"Missing wheel member {name}")
    if not any(name.startswith("_core") and name.endswith(".pyd") for name in names):
        errors.append("Missing _core Python extension in wheel")
    return entries


def ninja_rows(log):
    if not log.is_file():
        return set()
    return {
        tuple(line.split("\t"))
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
        if line and line[0].isdigit() and line.count("\t") == 4
    }


def ninja_stats(before, after):
    rows = ninja_rows(after) - ninja_rows(before)
    times = [(int(row[0]), int(row[1])) for row in rows]
    return {
        "new_output_rows": len(rows),
        "object_output_rows": sum(row[3].lower().endswith((".o", ".obj")) for row in rows),
        "first_to_last_edge_seconds": (
            (max(end for _, end in times) - min(start for start, _ in times)) / 1000 if times else None
        ),
        "native_link_edges_seconds": {
            row[3].replace("\\", "/"): (int(row[1]) - int(row[0])) / 1000
            for row in sorted(rows)
            if row[3].lower().endswith((".exe", ".dll", ".pyd"))
        },
    }


def subtool(ninja, build, *args):
    return subprocess.run(
        [ninja, "-t", *args],
        cwd=build,
        capture_output=True,
        check=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    ).stdout.rstrip()


def normalized(path):
    return path.lower().replace("\\", "/")


def evidence_for_target(commands, target, object_path, rule, mode):
    link = commands[normalized(target)]
    compile_command = commands[normalized(object_path)]
    normalized_link = normalized(link)
    normalized_compile = normalized(compile_command)
    auto_exports = "__create_def" in normalized_link
    is_clang = mode == "clang"
    is_ipo = "ipo" in rule.lower()
    # CMake Ninja distinguishes its IPO rule from an ordinary link rule. The
    # LLD driver consumes bitcode without requiring an explicit IPO link flag.
    compiler_ok = bool(re.search(r"\bclang-cl(?:\.exe)?\b", normalized_compile)) if is_clang else bool(
        re.search(r"(?:^|[/\s\"])cl(?:\.exe)?(?=[\s\"]|$)", normalized_compile)
    )
    compile_lto = "-flto=thin" in normalized_compile if is_clang else bool(
        re.search(r"(?:^|\s)[/-]gl(?=\s|$)", normalized_compile)
    )
    linker_ok = bool(re.search(r"\blld-link(?:\.exe)?\b", normalized_link)) if is_clang else bool(
        re.search(r"(?:^|[/\s\"])link(?:\.exe)?(?=[\s\"]|$)", normalized_link)
    )
    link_lto = bool(re.search(r"(?:^|\s)[/-]ltcg(?=[:\s]|$)", normalized_link))
    target_errors = []
    expected_ipo = is_clang or not auto_exports
    if not compiler_ok:
        target_errors.append(f"Unexpected compiler for {mode}")
    if not linker_ok:
        target_errors.append(f"Unexpected linker for {mode}")
    if compile_lto != expected_ipo:
        target_errors.append(f"Expected IPO compilation to be {expected_ipo}")
    if not is_clang and link_lto != expected_ipo:
        target_errors.append(f"Expected LTCG link to be {expected_ipo}")
    if is_clang and auto_exports and "--nm=" not in normalized_link:
        target_errors.append("Automatic export scanner needs --nm for LLVM bitcode")
    return {
        "target": target,
        "representative_object": object_path,
        "rule": rule,
        "compile_command": compile_command,
        "link_command": link,
        "compiler_matches_mode": compiler_ok,
        "linker_matches_mode": linker_ok,
        "compile_lto": compile_lto,
        "link_ltcg_flag": link_lto,
        "link_ipo_rule": is_ipo,
        "automatic_exports": auto_exports,
        "thinlto_cache": "/lldltocache:" in normalized_link or "-lldltocache:" in normalized_link,
        "errors": target_errors,
    }


def capture_build(build, result_dir, before_log, mode, errors):
    for source in (
        build / "CMakeCache.txt",
        build / "build.ninja",
        build / "CMakeFiles/rules.ninja",
        build / ".ninja_log",
    ):
        copy_if_present(source, result_dir / source.name)
    copy_if_present(before_log, result_dir / "before.ninja_log")
    evidence = {"ninja": ninja_stats(before_log, build / ".ninja_log")}
    cache_lines = (build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    evidence["cache"] = [line for line in cache_lines if PINS.match(line)]
    ninja = next(line.partition("=")[2] for line in cache_lines if line.startswith("CMAKE_MAKE_PROGRAM:"))
    targets = [line.rsplit(": ", 1) for line in subtool(ninja, build, "targets", "all").splitlines()]
    outputs = {}
    objects = {}
    for path, rule in targets:
        canonical = normalized(path)
        basename = canonical.rsplit("/", 1)[-1]
        if "/" in canonical and (
            basename in CORE_BINARIES or basename.startswith("_core") and basename.endswith(".pyd")
        ) and "_linker_" in rule.lower():
            outputs[basename] = (path, rule)
        if canonical.endswith(".obj") and (match := re.search(r"cmakefiles/([^/]+)\.dir/", canonical)):
            objects.setdefault(match[1], path)

    command_rows = json.loads(subtool(ninja, build, "compdb-targets", "-x", *(path for path, _ in outputs.values())))
    commands = {normalized(row["output"]): row["command"] for row in command_rows}
    evidence["targets"] = {}
    for name, (target, rule) in sorted(outputs.items()):
        object_target = "mqt-core-bindings" if name.endswith(".pyd") else Path(name).stem
        if object_target not in objects:
            errors.append(f"No object command found for {target}")
            continue
        object_path = objects[object_target]
        if any(normalized(path) not in commands for path in (target, object_path)):
            errors.append(f"Ninja compilation database did not report commands for {target} and {object_path}")
            continue
        verified = evidence_for_target(commands, target, object_path, rule, mode)
        errors.extend(f"{name}: {message}" for message in verified["errors"])
        evidence["targets"][name] = verified
    for name in CORE_BINARIES:
        if name not in evidence["targets"]:
            errors.append(f"Missing Ninja target {name}")
    if not any(name.endswith(".pyd") for name in evidence["targets"]):
        errors.append("Missing Ninja target for Python extension")
    return evidence


def repair(project, wheel, dest, result, mode):
    before_repair = time.perf_counter_ns()
    dest.mkdir(parents=True, exist_ok=True)
    existing = set(dest.glob("*.whl"))
    subprocess.run(
        ["delvewheel", "repair", "-w", str(dest), str(wheel), "--namespace-pkg", "mqt", "--ignore-existing"],
        check=True,
    )
    after_repair = time.perf_counter_ns()
    generated = set(dest.glob("*.whl")) - existing
    repaired, = generated or {dest / wheel.name}
    result_dir = result / repaired.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    with repaired.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "mode": mode,
        "wheel": repaired.name,
        "bytes": repaired.stat().st_size,
        "unrepaired_bytes": wheel.stat().st_size,
        "sha256": digest,
        "repair_seconds": (after_repair - before_repair) / 1e9,
        "evidence_errors": errors,
    }
    try:
        start_ns = json.loads((result / "start.json").read_text(encoding="utf-8"))["start_ns"]
        manifest["before_build_to_repair_seconds"] = (after_repair - start_ns) / 1e9
        manifest["before_build_to_unrepaired_seconds"] = (before_repair - start_ns) / 1e9
    except (OSError, ValueError, KeyError) as error:
        errors.append(f"Timing: {error}")
    try:
        manifest["native_members"] = wheel_inventory(repaired, errors)
    except (OSError, ValueError, struct.error, zipfile.BadZipFile) as error:
        errors.append(f"Wheel inventory: {error}")
    try:
        manifest["build"] = capture_build(timed_build_dir(project), result_dir, result / "before.ninja_log", mode, errors)
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError) as error:
        errors.append(f"Build evidence: {error}")

    write_json(result_dir / "wheel.json", manifest)
    print(f"Build evidence for {mode} {repaired.name}: {manifest['bytes']} bytes, {errors=}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("start", "repair"))
    parser.add_argument("project", type=Path)
    parser.add_argument("wheel", type=Path, nargs="?")
    parser.add_argument("dest", type=Path, nargs="?")
    args = parser.parse_args()
    result = Path(os.environ["STUDY_RESULT"])
    mode = os.environ["STUDY_MODE"]
    if mode not in {"msvc", "clang"}:
        parser.error("STUDY_MODE must be msvc or clang")
    if args.phase == "start":
        start(args.project, result)
    elif args.wheel is None or args.dest is None:
        parser.error("repair requires wheel and dest")
    else:
        repair(args.project, args.wheel, args.dest, result, mode)


if __name__ == "__main__":
    main()

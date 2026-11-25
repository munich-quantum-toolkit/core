"""Compare installed compiler CLIs using balanced process-level samples."""

import argparse
import cmath
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from random import Random


ROTATION_CALL = re.compile(
    rb"  call void @__quantum__qis__(u2|u3|rx|gphase)__body\(([^)]*)\)(\r?\n)?"
)
ANGLE_ARITY = {b"u2": 2, b"u3": 3, b"rx": 1, b"gphase": 1}
F64 = re.compile(rb"double (f0x[0-9A-Fa-f]{16}|-?[0-9]+\.[0-9]+e[+-][0-9]+)")
ANGLE_TOLERANCE_RAD = 1e-14
MATRIX_TOLERANCE = 3e-14


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_qasm(path, seed, qubits, gates, gate_set):
    """Recreate the seeded inputs used in the Linux wheel comparison."""
    rng = Random(seed)
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{qubits}] q;",
        f"bit[{qubits}] c;",
    ]
    for _ in range(gates):
        gate = rng.choice(gate_set)
        first, second = rng.sample(range(qubits), 2)
        if gate in ("cx", "cz", "swap"):
            lines.append(f"{gate} q[{first}], q[{second}];")
        elif gate in ("rx", "rz"):
            lines.append(f"{gate}(pi/8) q[{first}];")
        else:
            lines.append(f"{gate} q[{first}];")
    lines.append("c = measure q;")
    path.write_bytes(("\n".join(lines) + "\n").encode())


def workloads(work, source_root):
    clifford = work / "bench-qasm-clifford.qasm"
    rotations = work / "bench-qasm-rotations.qasm"
    write_qasm(clifford, 29, 16, 6500, ("h", "x", "cx", "cz", "swap", "s"))
    write_qasm(rotations, 37, 19, 4500, ("rx", "rz", "h", "t", "cz"))
    # Detect a change in Python's seeded sampling or the generator.
    expected = {
        clifford: "d3c0dc382e4613a24d5787d32e831b9182a7b11f14ff607c42174efa407038a9",
        rotations: "02f5460f25a6b34e0f0bded0d93e6a7d01de551e45376a63c52d024d18bf83f0",
    }
    for source, sha256 in expected.items():
        if digest(source) != sha256:
            raise ValueError(f"Generated {source.name} differs from its recorded input")
    feedback = source_root / "test/python/qir/feedback.qasm"
    return [(clifford, "base"), (rotations, "adaptive"), (feedback, "adaptive")]


def verify_machine(programs):
    if os.name != "nt":
        return platform.machine()

    # Check the Windows kernel's architecture and the Python process' mode.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.IsWow64Process2.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
    )
    process = ctypes.c_uint16()
    native = ctypes.c_uint16()
    if not kernel.IsWow64Process2(
        kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(native)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if process.value:
        raise ValueError(f"Benchmark Python runs under emulation ({process.value:#x})")

    for exe in programs.values():
        with exe.open("rb") as stream:
            stream.seek(0x3C)
            stream.seek(struct.unpack("<I", stream.read(4))[0])
            signature, machine = struct.unpack("<4sH", stream.read(6))
        if signature != b"PE\x00\x00" or machine != native.value:
            raise ValueError(f"{exe} is not a native Windows executable")
    return {0x8664: "x64", 0xAA64: "ARM64"}[native.value]


def invoke(exe, source, profile, out, log=None):
    args = [exe, "compile", source, f"--emit=qir-{profile}", "-o", out]
    if log is not None:
        with log.open("wb") as stream:
            subprocess.run(
                args, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=60
            )
        return None

    start = time.perf_counter_ns()
    subprocess.run(args, stdout=subprocess.DEVNULL, check=True, timeout=60)
    return time.perf_counter_ns() - start


def verify_qir(llvm_bin, output, profile, work, log=None):
    llvm_as = llvm_bin / ("llvm-as.exe" if os.name == "nt" else "llvm-as")
    opt = llvm_bin / ("opt.exe" if os.name == "nt" else "opt")
    bc_file = work / f"{output.stem}.bc"
    commands = [
        [llvm_as, output, "-o", bc_file],
        [opt, "-passes=verify", "-disable-output", bc_file],
    ]
    if log is None:
        for command in commands:
            subprocess.run(command, check=True, timeout=60)
    else:
        with log.open("ab") as stream:
            for command in commands:
                subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=60,
                )
    profile_marker = f'"qir_profiles"="{profile}_profile"'.encode()
    if profile_marker not in output.read_bytes():
        raise ValueError(f"{output} lacks expected QIR profile {profile}")


def read_angle(operand):
    match = F64.fullmatch(operand)
    if not match:
        raise ValueError(f"Unexpected rotation operand {operand!r}")
    literal = match[1]
    if literal.startswith(b"f0x"):
        return struct.unpack(">d", bytes.fromhex(literal[3:].decode("ascii")))[0]
    return float(literal)


def ordered_float_bits(value):
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return -(bits & ((1 << 63) - 1)) if bits >> 63 else bits


def gate_matrix(gate, angles):
    # Match QIR's U2/U3, Rx and global phase ABI. In particular theta in U3
    # and Rx cannot be reduced modulo 2pi without changing a global phase.
    if gate == b"gphase":
        return (cmath.exp(1j * angles[0]),)
    if gate == b"rx":
        theta = angles[0]
        diagonal = math.cos(theta / 2)
        off_diagonal = -1j * math.sin(theta / 2)
        return diagonal, off_diagonal, off_diagonal, diagonal
    if gate == b"u2":
        theta = math.pi / 2
        phi, lam = angles
    else:
        theta, phi, lam = angles
    sin = math.sin(theta / 2)
    cos = math.cos(theta / 2)
    return (
        cos,
        -cmath.exp(1j * lam) * sin,
        cmath.exp(1j * phi) * sin,
        cmath.exp(1j * (phi + lam)) * cos,
    )


def compare_rotations(msvc, clang):
    """Allow only near-equivalent angles in the observed QIR calls."""
    left, right = (
        path.read_bytes().splitlines(keepends=True) for path in (msvc, clang)
    )
    if len(left) != len(right):
        raise ValueError(f"QIR line counts differ ({len(left)} vs {len(right)})")

    result = {
        "kind": "equivalent_rotation_operands",
        "changed_calls": {},
        "changed_operands": 0,
        "phase_wraps": 0,
        "max_raw_delta_rad": 0.0,
        "max_periodic_delta_rad": 0.0,
        "max_raw_ulp_distance": 0,
        "max_ulp_distance_no_wrap_away_from_zero": 0,
        "max_gate_matrix_frobenius_delta": 0.0,
        "sum_gate_matrix_frobenius_deltas": 0.0,
    }
    for line, (base, candidate) in enumerate(zip(left, right, strict=True), 1):
        if base == candidate:
            continue
        expected = ROTATION_CALL.fullmatch(base)
        actual = ROTATION_CALL.fullmatch(candidate)
        if (
            not expected
            or not actual
            or (expected[1], expected[3]) != (actual[1], actual[3])
        ):
            raise ValueError(f"Unaudited QIR structure difference on line {line}")

        gate = expected[1]
        argc = ANGLE_ARITY[gate]
        a, b = (match[2].split(b", ") for match in (expected, actual))
        if len(a) < argc or a[argc:] != b[argc:]:
            raise ValueError(f"QIR call target operands differ on line {line}")
        first, second = (
            [read_angle(operand) for operand in args[:argc]] for args in (a, b)
        )
        for i, (x, y) in enumerate(zip(first, second, strict=True)):
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"Non-finite QIR rotation operand on line {line}")
            if a[i] == b[i]:
                continue
            delta = x - y
            periodic = gate in (b"u2", b"gphase") or (gate == b"u3" and i > 0)
            residual = math.remainder(delta, math.tau) if periodic else delta
            if abs(residual) > ANGLE_TOLERANCE_RAD:
                raise ValueError(f"QIR angle differs by {residual} rad on line {line}")
            wraps = delta != residual
            ulp = abs(ordered_float_bits(x) - ordered_float_bits(y))
            result["changed_operands"] += 1
            result["phase_wraps"] += int(wraps)
            result["max_raw_delta_rad"] = max(result["max_raw_delta_rad"], abs(delta))
            result["max_periodic_delta_rad"] = max(
                result["max_periodic_delta_rad"], abs(residual)
            )
            result["max_raw_ulp_distance"] = max(result["max_raw_ulp_distance"], ulp)
            if not wraps and min(abs(x), abs(y)) >= ANGLE_TOLERANCE_RAD:
                key = "max_ulp_distance_no_wrap_away_from_zero"
                result[key] = max(result[key], ulp)
        a_matrix = gate_matrix(gate, first)
        b_matrix = gate_matrix(gate, second)
        delta = math.hypot(
            *(abs(x - y) for x, y in zip(a_matrix, b_matrix, strict=True))
        )
        if delta > MATRIX_TOLERANCE:
            raise ValueError(f"QIR gate matrix differs by {delta} on line {line}")
        name = gate.decode("ascii")
        counts = result["changed_calls"]
        counts[name] = counts.get(name, 0) + 1
        key = "max_gate_matrix_frobenius_delta"
        result[key] = max(result[key], delta)
        result["sum_gate_matrix_frobenius_deltas"] += delta
    return result


def benchmark(programs, source, profile, llvm_bin, work, evidence, pairs, result):
    evidence.mkdir(parents=True, exist_ok=True)
    records = {}
    outputs = {}
    errors = {}
    for name, exe in programs.items():
        qir = evidence / f"{name}.ll"
        log = evidence / f"{name}.log"
        outputs[name] = qir
        try:
            invoke(exe, source, profile, work / "warmup.ll", log)
            invoke(exe, source, profile, qir, evidence / f"{name}-verification.log")
            records[name] = digest(qir)
            verify_qir(
                llvm_bin, qir, profile, work, evidence / f"{name}-verification.log"
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            errors[name] = str(error)
    result["qir_sha256"] = records
    result["qir_evidence"] = {name: str(path) for name, path in outputs.items()}
    if errors:
        raise ValueError(f"QIR generation or verification failed: {errors}")
    if len(set(records.values())) == 1:
        result["qir_comparison"] = {"kind": "byte_identical"}
    elif source.name == "bench-qasm-rotations.qasm":
        result["qir_comparison"] = compare_rotations(outputs["msvc"], outputs["clang"])
    else:
        raise ValueError(f"QIR differs for {source.name}: {records}")

    qir = work / "output.ll"
    names = tuple(programs)
    samples = result["samples"]
    changes = []
    for index in range(pairs):
        order = names if index % 2 == 0 else names[::-1]
        times = {}
        for name in order:
            times[name] = invoke(programs[name], source, profile, qir)
            if digest(qir) != records[name]:
                shutil.copyfile(qir, evidence / f"{name}-changed.ll")
                raise ValueError(
                    f"Repeat {index + 1}: {name} changed its QIR for {source}"
                )
        samples.append({"pair": index + 1, "order": order, "elapsed_ns": times})
        changes.append((times["clang"] / times["msvc"] - 1) * 100)

    quartiles = statistics.quantiles(changes, n=4, method="inclusive")
    median = statistics.median(changes)
    print(
        f"{source.name} ({profile}) candidate delta {median:+.2f}% "
        f"with quartiles {quartiles[0]:+.2f}% to {quartiles[2]:+.2f}%",
        flush=True,
    )
    result["median_ms"] = {
        name: statistics.median(sample["elapsed_ns"][name] for sample in samples) / 1e6
        for name in names
    }
    result["paired_percent_change"] = {
        "min": min(changes),
        "q1": quartiles[0],
        "median": median,
        "q3": quartiles[2],
        "max": max(changes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("msvc", "clang", "source-root", "llvm-bin", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--pairs", default=16, type=int)
    args = parser.parse_args()
    if args.pairs < 4 or args.pairs % 2:
        parser.error("--pairs must be an even integer of at least four")

    programs = {
        "msvc": args.msvc.resolve(strict=True),
        "clang": args.clang.resolve(strict=True),
    }
    result = {
        "architecture": verify_machine(programs),
        "platform": platform.platform(),
        "logical_processors": os.cpu_count(),
        "programs": {
            name: {
                "path": str(exe),
                "version": subprocess.check_output(
                    [exe, "--version"], text=True, timeout=60
                ).strip(),
                "sha256": digest(exe),
            }
            for name, exe in programs.items()
        },
        "warmups_per_program": 2,
        "pairs_per_workload": args.pairs,
        "workloads": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = args.output.with_name(f"{args.output.stem}-qir")
    with tempfile.TemporaryDirectory(prefix="mqt-compiler-") as temp:
        work = Path(temp)
        work_cases = workloads(work, args.source_root.resolve())
        for source, profile in work_cases:
            record = {
                "input": source.name,
                "input_sha256": digest(source),
                "profile": profile,
                "samples": [],
                "status": "measured",
            }
            result["workloads"].append(record)
            try:
                benchmark(
                    programs,
                    source,
                    profile,
                    args.llvm_bin.resolve(),
                    work,
                    evidence / source.stem,
                    args.pairs,
                    record,
                )
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                record["status"] = "incomparable"
                record["error"] = str(error)
                print(f"{source.name}: {error}", flush=True)
            finally:
                args.output.write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
    if any(case["status"] != "measured" for case in result["workloads"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

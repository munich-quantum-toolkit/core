"""Compare installed compiler CLIs using balanced process-level samples."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from random import Random


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


def invoke(exe, source, profile, out):
    args = [exe, "compile", source, f"--emit=qir-{profile}", "-o", out]
    start = time.perf_counter_ns()
    subprocess.run(args, stdout=subprocess.DEVNULL, check=True, timeout=60)
    return time.perf_counter_ns() - start


def verify_qir(llvm_bin, output, profile, work):
    llvm_as = llvm_bin / ("llvm-as.exe" if os.name == "nt" else "llvm-as")
    opt = llvm_bin / ("opt.exe" if os.name == "nt" else "opt")
    bc_file = work / "verified.bc"
    subprocess.run([llvm_as, output, "-o", bc_file], check=True, timeout=60)
    subprocess.run(
        [opt, "-passes=verify", "-disable-output", bc_file], check=True, timeout=60
    )
    profile_marker = f'"qir_profiles"="{profile}_profile"'.encode()
    if profile_marker not in output.read_bytes():
        raise ValueError(f"{output} lacks expected QIR profile {profile}")


def benchmark(programs, source, profile, llvm_bin, work, pairs):
    qir = work / "output.ll"
    fingerprints = {}
    for name, exe in programs.items():
        # Two untimed invocations per variant. Verify the installed CLI output.
        invoke(exe, source, profile, qir)
        invoke(exe, source, profile, qir)
        verify_qir(llvm_bin, qir, profile, work)
        fingerprints[name] = digest(qir)
    if len(set(fingerprints.values())) != 1:
        raise ValueError(f"QIR outputs differ for {source}: {fingerprints}")

    names = tuple(programs)
    samples = []
    changes = []
    for index in range(pairs):
        order = names if index % 2 == 0 else names[::-1]
        times = {}
        for name in order:
            times[name] = invoke(programs[name], source, profile, qir)
            if digest(qir) != fingerprints[name]:
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
    return {
        "input": source.name,
        "input_sha256": digest(source),
        "profile": profile,
        "qir_sha256": fingerprints,
        "median_ms": {
            name: statistics.median(sample["elapsed_ns"][name] for sample in samples)
            / 1e6
            for name in names
        },
        "paired_percent_change": {
            "min": min(changes),
            "q1": quartiles[0],
            "median": median,
            "q3": quartiles[2],
            "max": max(changes),
        },
        "samples": samples,
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
    with tempfile.TemporaryDirectory(prefix="mqt-compiler-") as temp:
        work = Path(temp)
        for source, profile in workloads(work, args.source_root.resolve()):
            result["workloads"].append(
                benchmark(
                    programs, source, profile, args.llvm_bin.resolve(), work, args.pairs
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

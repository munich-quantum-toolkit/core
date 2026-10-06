# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Capture unchanged payload execution through the actual DDSIM QDMI device ABI.

This presentation-only client uses ctypes so every displayed QDMI call is a real
native call. It never imports the compiler or opens another provider. Timings
measure the call boundary in this client, not internal device state transitions.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any

HERE = Path(__file__).resolve().parent
FORMATS = {"openqasm3": (1, "QASM3"), "qir-adaptive": (4, "QIR_ADAPTIVE_STRING")}
STATUSES = ("CREATED", "SUBMITTED", "QUEUED", "RUNNING", "DONE", "CANCELED", "FAILED")
HANDLE = ct.c_void_p
SIZE = ct.c_size_t
INT = ct.c_int
OUT_SIZE = ct.POINTER(SIZE)
OUT_HANDLE = ct.POINTER(HANDLE)
SIGNATURES = {
    "device_initialize": (INT, []),
    "device_finalize": (INT, []),
    "device_session_alloc": (INT, [OUT_HANDLE]),
    "device_session_init": (INT, [HANDLE]),
    "device_session_free": (None, [HANDLE]),
    "device_session_create_device_job": (INT, [HANDLE, OUT_HANDLE]),
    "device_job_set_parameter": (INT, [HANDLE, INT, SIZE, HANDLE]),
    "device_job_query_property": (INT, [HANDLE, INT, SIZE, HANDLE, OUT_SIZE]),
    "device_job_submit": (INT, [HANDLE]),
    "device_job_check": (INT, [HANDLE, ct.POINTER(INT)]),
    "device_job_wait": (INT, [HANDLE, SIZE]),
    "device_job_get_results": (INT, [HANDLE, INT, SIZE, HANDLE, OUT_SIZE]),
    "device_job_free": (None, [HANDLE]),
}


def validate_execution(execution: dict[str, Any]) -> None:
    """Reject captures that do not establish a successful, unchanged execution.

    Raises:
        ValueError: If status, payload identity, shots, counts, or call order disagree.
    """
    if execution["terminal_status"] != "DONE" or execution["payload_identity_verified"] is not True:
        msg = "Execution must finish successfully with an identical payload"
        raise ValueError(msg)
    if len(execution["shots"]) != execution["num_shots"]:
        msg = "Returned shot count differs from the requested shot count"
        raise ValueError(msg)
    if dict(Counter(execution["shots"])) != execution["counts"]:
        msg = "Ordered shots disagree with the separately retrieved histogram"
        raise ValueError(msg)
    times = [event["time_ms"] for event in execution["events"]]
    if times != sorted(times):
        msg = "Native calls are not in capture order"
        raise ValueError(msg)


class DDSIMCapture:
    """A small, explicit client for the documented DDSIM device C interface."""

    def __init__(self, library: Path) -> None:
        """Bind the device's exported C functions with their explicit signatures."""
        self.library = ct.CDLL(str(library.resolve()))
        for name, (restype, argtypes) in SIGNATURES.items():
            function = getattr(self.library, f"MQT_DDSIM_QDMI_{name}")
            function.restype, function.argtypes = restype, argtypes
        self.events: list[dict[str, Any]] = []
        self.start = 0

    def call(self, name: str, *args: object, detail: str = "") -> None:
        """Record one real call, including its measured duration and return code.

        Raises:
            RuntimeError: If the native call returns a QDMI error.
        """
        begin = perf_counter_ns()
        status = getattr(self.library, f"MQT_DDSIM_QDMI_{name}")(*args)
        end = perf_counter_ns()
        event = {
            "time_ms": (begin - self.start) / 1e6,
            "duration_ms": (end - begin) / 1e6,
            "actor": "Capture client",
            "target": "DDSIM QDMI device",
            "operation": f"MQT_DDSIM_QDMI_{name}",
            "status": "void" if status is None else "QDMI_SUCCESS" if status == 0 else f"QDMI_ERROR({status})",
            "return_code": status,
            "detail": detail,
        }
        self.events.append(event)
        if status not in {None, 0}:
            msg = f"{event['operation']} returned {status}: {detail}"
            raise RuntimeError(msg)

    def read(self, job: HANDLE, property_id: int, name: str, *, program: bool = False) -> bytes:
        """Use QDMI's size/data query pair to retain exact returned bytes.

        Returns:
            The complete native result buffer, including any string terminator.
        """
        function = "device_job_query_property" if program else "device_job_get_results"
        size = SIZE()
        self.call(function, job, property_id, 0, None, ct.byref(size), detail=f"{name}: size")
        buffer = ct.create_string_buffer(size.value)
        self.call(function, job, property_id, size.value, buffer, None, detail=f"{name}: {size.value} bytes")
        return buffer.raw

    def execute(self, payload: str, format_id: int, format_name: str, shots: int, seed: int) -> dict[str, Any]:
        """Execute one exact text payload and verify independent output views.

        Returns:
            Verified ordered shots, counts, identity hashes, and native call evidence.

        Raises:
            ValueError: If inputs or independently returned output views disagree.
            RuntimeError: If the device returns an error or the job does not finish successfully.
        """
        self.events = []
        self.start = perf_counter_ns()
        script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        wire_payload = payload.encode("utf-8") + b"\0"
        if b"\0" in wire_payload[:-1] or shots <= 0 or seed <= 0:
            msg = "Need a text payload without NUL, positive shots, and a positive seed"
            raise ValueError(msg)
        session, job = HANDLE(), HANDLE()
        self.call("device_initialize")
        try:
            self.call("device_session_alloc", ct.byref(session))
            self.call("device_session_init", session)
            self.call("device_session_create_device_job", session, ct.byref(job))
            parameters = (
                (0, "PROGRAMFORMAT", INT(format_id)),
                (1, "PROGRAM", ct.create_string_buffer(wire_payload, len(wire_payload))),
                (2, "SHOTSNUM", SIZE(shots)),
                (999999995, "CUSTOM1: reproducible simulator seed", INT(seed)),
            )
            for parameter, name, value in parameters:
                self.call("device_job_set_parameter", job, parameter, ct.sizeof(value), ct.byref(value), detail=name)
            returned_program = self.read(job, 2, "PROGRAM", program=True)
            if returned_program != wire_payload:
                msg = "The device did not retain the exact compiled payload"
                raise ValueError(msg)
            submitted = perf_counter_ns()
            self.call("device_job_submit", job)
            status = INT()
            self.call("device_job_check", job, ct.byref(status))
            self.events[-1]["status"] = STATUSES[status.value]
            self.call("device_job_wait", job, 300, detail="timeout: 300 seconds")
            completed = perf_counter_ns()
            self.call("device_job_check", job, ct.byref(status))
            self.events[-1]["status"] = STATUSES[status.value]
            if status.value != 4:
                msg = f"Execution ended with {STATUSES[status.value]}"
                raise RuntimeError(msg)
            raw_shots = self.read(job, 0, "SHOTS")
            raw_keys = self.read(job, 1, "HIST_KEYS")
            raw_values = self.read(job, 2, "HIST_VALUES")
            if not raw_shots.endswith(b"\0") or not raw_keys.endswith(b"\0"):
                msg = "QDMI text results must be NUL-terminated"
                raise ValueError(msg)
            ordered_shots = raw_shots[:-1].decode("ascii").split(",")
            keys = raw_keys[:-1].decode("ascii").split(",")
            if len(raw_values) != len(keys) * ct.sizeof(SIZE) or len(keys) != len(set(keys)):
                msg = "Invalid histogram key/value lengths or duplicate keys"
                raise ValueError(msg)
            counts = dict(zip(keys, (SIZE * len(keys)).from_buffer_copy(raw_values), strict=True))
            result = {
                "format": format_name,
                "payload_sha256": hashlib.sha256(wire_payload[:-1]).hexdigest(),
                "wire_payload_sha256": hashlib.sha256(returned_program).hexdigest(),
                "payload_identity_verified": True,
                "num_shots": shots,
                "seed": seed,
                "shots": ordered_shots,
                "counts": counts,
                "duration_ms": (completed - submitted) / 1e6,
                "terminal_status": "DONE",
                "backend": "MQT DDSIM",
                "trace_kind": "native-device-api",
                "trace_clock": "Python perf_counter_ns around actual ctypes C ABI calls",
                "capture_script_sha256": script_sha256,
                "events": self.events,
            }
        finally:
            if job:
                self.call("device_job_free", job)
            if session:
                self.call("device_session_free", session)
            self.call("device_finalize")
        validate_execution(result)
        return result


def main() -> None:
    """Capture curated programs, or exercise the native path with a Bell pair.

    Raises:
        ValueError: If the Bell check fails or no export can be captured.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True, help="DDSIM QDMI shared library from this Core build")
    parser.add_argument("--input", type=Path, default=HERE / "captures/programs.json")
    parser.add_argument("--output", type=Path, default=HERE / "captures/demo.json.gz")
    parser.add_argument("--application", type=Path)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument(
        "--shor-shots", type=int, default=4, help="Smaller shot budget for the large dynamic Shor payload"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        request = json.load(sys.stdin)
        result = DDSIMCapture(args.library).execute(**request)
        sys.stdout.write(json.dumps(result))
        return
    if args.self_test:
        client = DDSIMCapture(args.library)
        source = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;'
        result = client.execute(source, 1, "QASM3", 32, 7)
        if set(result["counts"]) != {"00", "11"}:
            msg = "Bell pair self-test failed"
            raise ValueError(msg)
        return
    data = json.loads(args.input.read_text())
    data["provenance"]["execution_generated_at"] = datetime.now(UTC).isoformat()
    data["provenance"]["execution_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data["provenance"]["execution"] = {
        "provider": "MQT DDSIM",
        "trace_kind": "native-device-api",
        "description": "Actual DDSIM device-interface calls via ctypes; no compiler or driver invoked during execution",
        "simulation": "Ideal gates, without calibration-derived noise",
        "replay": "Animation time is independent of recorded execution time",
    }
    if args.application:
        application = json.loads(args.application.read_text())
        data["application"] = application
        if "scenario" in application:
            data["scenarios"].append(application["scenario"])

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(data, indent=2) + "\n").encode()
        args.output.write_bytes(gzip.compress(encoded, mtime=0) if args.output.suffix == ".gz" else encoded)

    successes = 0
    for scenario in sorted(data["scenarios"], key=lambda value: value["id"] == "shor15"):
        if scenario.get("application"):
            continue
        for variant in scenario["variants"]:
            executions = {}
            for exported in variant["exports"]:
                if exported["id"] not in FORMATS or exported.get("unavailable_reason"):
                    continue
                format_id, format_name = FORMATS[exported["id"]]
                request = {
                    "payload": exported["code"],
                    "format_id": format_id,
                    "format_name": format_name,
                    "shots": args.shor_shots if scenario["id"] == "shor15" else args.shots,
                    "seed": args.seed,
                }
                label = f"{scenario['id']}/{variant['id']}/{exported['id']}"
                print(f"Capturing {label}", flush=True)
                try:
                    process = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - execute only this capture script, never payload code
                        [
                            sys.executable,
                            "-B",
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--library",
                            str(args.library),
                        ],
                        input=json.dumps(request),
                        capture_output=True,
                        text=True,
                        timeout=330,
                        check=True,
                    )
                    executions[exported["id"]] = json.loads(process.stdout)
                    print(f"Captured {label}: {executions[exported['id']]['duration_ms']:.1f} ms", flush=True)
                except subprocess.TimeoutExpired:
                    exported["capture_error"] = "Offline capture exceeded its 330-second wall-clock limit"
                    print(f"Timed out: {label}", flush=True)
                except subprocess.CalledProcessError as error:
                    exported["capture_error"] = error.stderr[-4000:]
                    print(f"Failed: {label}: {error.stderr[-1000:]}", flush=True)
                if executions:
                    variant["executions"] = executions
                    variant["execution"] = executions.get("qir-adaptive", next(iter(executions.values())))
                save()
            if executions:
                variant["executions"] = executions
                variant["execution"] = executions.get("qir-adaptive", next(iter(executions.values())))
                successes += 1
            else:
                variant["execution"] = {
                    "unavailable_reason": "No export completed the bounded offline execution capture"
                }
    if successes == 0:
        msg = "No executable exports found"
        raise ValueError(msg)
    save()


if __name__ == "__main__":
    main()

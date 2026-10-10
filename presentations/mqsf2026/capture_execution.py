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
QIR tracing executes each shot separately in an isolated, single-program worker;
it disables terminal batch sampling and includes instrumentation overhead.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import gzip
import hashlib
import inspect
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic_ns
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
    "device_job_set_programs": (INT, [HANDLE, INT, SIZE, OUT_SIZE, OUT_HANDLE]),
    "device_job_get_program": (INT, [HANDLE, SIZE, SIZE, HANDLE, OUT_SIZE]),
    "device_job_submit": (INT, [HANDLE]),
    "device_job_check": (INT, [HANDLE, ct.POINTER(INT)]),
    "device_job_wait": (INT, [HANDLE, SIZE]),
    "device_job_get_results": (INT, [HANDLE, SIZE, INT, SIZE, HANDLE, OUT_SIZE]),
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
    if "shot_events" in execution:
        shots = execution["shot_events"]
        completion_times = [shot["time_ms"] for shot in shots]
        if (
            [shot["outcome"] for shot in shots] != execution["shots"]
            or [shot["shot_index"] for shot in shots] != list(range(execution["num_shots"]))
            or completion_times != sorted(completion_times)
            or any(not execution["submitted_ms"] <= time <= execution["completed_ms"] for time in completion_times)
        ):
            msg = "Shot completion evidence disagrees with results or measured execution times"
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
        begin = monotonic_ns()
        status = getattr(self.library, f"MQT_DDSIM_QDMI_{name}")(*args)
        end = monotonic_ns()
        frame = inspect.currentframe()
        assert frame is not None
        assert frame.f_back is not None
        event = {
            "time_ms": (begin - self.start) / 1e6,
            "duration_ms": (end - begin) / 1e6,
            "actor": "Capture client",
            "target": "DDSIM QDMI device",
            "operation": f"MQT_DDSIM_QDMI_{name}",
            "status": "void" if status is None else "QDMI_SUCCESS" if status == 0 else f"QDMI_ERROR({status})",
            "return_code": status,
            "detail": detail,
            "script_line": frame.f_back.f_lineno,
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
        function = "device_job_get_program" if program else "device_job_get_results"
        arguments = (job, 0) if program else (job, 0, property_id)
        size = SIZE()
        self.call(function, *arguments, 0, None, ct.byref(size), detail=f"{name}: size")
        buffer = ct.create_string_buffer(size.value)
        self.call(function, *arguments, size.value, buffer, None, detail=f"{name}: {size.value} bytes")
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
        self.start = monotonic_ns()
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
            program = ct.create_string_buffer(wire_payload, len(wire_payload))
            sizes = (SIZE * 1)(len(wire_payload))
            programs = (HANDLE * 1)(ct.cast(program, HANDLE))
            self.call("device_job_set_programs", job, format_id, 1, sizes, programs, detail=format_name)
            parameters = (
                (2, "SHOTSNUM", SIZE(shots)),
                (999999995, "CUSTOM1: reproducible simulator seed", INT(seed)),
                (999999997, "CUSTOM3: serial worker for shot timing", SIZE(1)),
            )
            for parameter, name, value in parameters:
                self.call("device_job_set_parameter", job, parameter, ct.sizeof(value), ct.byref(value), detail=name)
            returned_program = self.read(job, 2, "PROGRAM", program=True)
            if returned_program != wire_payload:
                msg = "The device did not retain the exact compiled payload"
                raise ValueError(msg)
            submitted = monotonic_ns()
            self.call("device_job_submit", job)
            status = INT()
            self.call("device_job_check", job, ct.byref(status))
            self.events[-1]["status"] = STATUSES[status.value]
            self.call("device_job_wait", job, 300, detail="timeout: 300 seconds")
            completed = monotonic_ns()
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
                "submitted_ms": (submitted - self.start) / 1e6,
                "completed_ms": (completed - self.start) / 1e6,
                "terminal_status": "DONE",
                "backend": "MQT DDSIM",
                "trace_kind": "native-device-api",
                "trace_clock": "Linux CLOCK_MONOTONIC: Python monotonic_ns calls and C++ steady_clock shot completions",
                "capture_script_sha256": script_sha256,
                "events": self.events,
            }
        finally:
            if job:
                self.call("device_job_free", job)
            if session:
                self.call("device_session_free", session)
            self.call("device_finalize")
        trace_path = os.environ.get("MQT_MQSF_SHOT_TRACE")
        if trace_path and format_id == 4:
            records = [line.split("\t") for line in Path(trace_path).read_text(encoding="utf-8").splitlines()]
            result["shot_events"] = [
                {"time_ms": (int(timestamp) - self.start) / 1e6, "shot_index": index, "outcome": outcome}
                for index, (timestamp, outcome) in enumerate(records)
            ]
            result["shot_timing"] = (
                "Actual serial completions with per-shot circuit execution; "
                "terminal batch sampling disabled and demo logging overhead included"
            )
        else:
            result["shot_timing"] = "No per-shot timing capture; histogram becomes available when results are retrieved"
        source = Path(__file__).read_text(encoding="utf-8").splitlines()
        source_lines = list(dict.fromkeys(event["script_line"] for event in self.events))
        result["client_source"] = "\n".join(source[line - 1].strip() for line in source_lines)
        result["client_source_note"] = (
            "Exact executed call statements from capture_execution.py; setup and validation omitted."
        )
        for event in self.events:
            event["source_line"] = source_lines.index(event["script_line"]) + 1
        validate_execution(result)
        return result


def main() -> None:
    """Capture curated programs, or exercise the native path with a Bell pair.

    Raises:
        ValueError: If the Bell check fails or no export can be captured.
        RuntimeError: If shot tracing is requested on a system with an unsupported clock.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True, help="DDSIM QDMI shared library from this Core build")
    parser.add_argument("--input", type=Path, default=HERE / "captures/programs.json")
    parser.add_argument("--output", type=Path, default=HERE / "captures/demo.json.gz")
    parser.add_argument("--application", type=Path)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        request = json.load(sys.stdin)
        with TemporaryDirectory(prefix="mqsf-shots-") as directory:
            if request["format_id"] == 4:
                if sys.platform != "linux":
                    msg = "Shot trace clock alignment requires Linux CLOCK_MONOTONIC"
                    raise RuntimeError(msg)
                os.environ["MQT_MQSF_SHOT_TRACE"] = str(Path(directory) / "shots.tsv")
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
        "description": "Actual QDMI 1.4 DDSIM device-interface calls; compiled payload submitted unchanged",
        "simulation": "Ideal gates, without calibration-derived noise",
        "replay": "Replay uses recorded call and shot timestamps; any time scaling is explicit",
        "instrumentation": (
            "MQT_MQSF_SHOT_TRACE: isolated single-program worker, serial JitSession.sample(1), continuous RNG, "
            "steady-clock timestamp per completed QIR shot; each shot executes the circuit, "
            "disabling terminal batch sampling"
        ),
        "worker_source_sha256": hashlib.sha256(
            (HERE.parents[1] / "src/qdmi/devices/dd/WorkerMain.cpp").read_bytes()
        ).hexdigest(),
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
    for scenario in data["scenarios"]:
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
                    "shots": args.shots,
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

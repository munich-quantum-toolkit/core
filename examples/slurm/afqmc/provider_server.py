# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Local IQM and Braket protocol fixtures that simulate the submitted circuits.

These services model the demonstrated gate-model subset, not provider validation,
billing, calibration, reservations, or cloud performance. No cloud client is used.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast
from urllib.parse import unquote, urlsplit

CALIBRATION = "00000000-0000-0000-0000-000000000017"
QUBITS = [f"QB{i + 1}" for i in range(4)]
ARN = "arn:aws:braket:us-east-1::device/quantum-simulator/amazon/sv1"


def iqm_architecture() -> dict:
    """Advertise a four-site line with native PRX and CZ gates."""
    loci = {
        "prx": [[q] for q in QUBITS],
        "cz": [[QUBITS[i], QUBITS[i + 1]] for i in range(3)],
        "measure": [[q] for q in QUBITS],
    }
    return {
        "calibration_set_id": CALIBRATION,
        "gates": {
            gate: {"default_implementation": "local", "implementations": {"local": {"loci": sites}}}
            for gate, sites in loci.items()
        },
    }


def iqm_samples(circuit: dict, shots: int) -> tuple[dict, dict]:
    """Simulate native instructions independently with Qiskit's statevector."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    program = QuantumCircuit(4)
    readout = {}
    measured = False
    for instruction in circuit["instructions"]:
        gate = instruction["name"]
        wires = [QUBITS.index(name) for name in instruction["locus"]]
        args = instruction["args"]
        if gate == "measure":
            measured = True
            if args["key"] in readout:
                msg = "Repeated measurement key"
                raise ValueError(msg)
            readout[args["key"]] = wires
        elif measured:
            msg = "Only terminal measurements are supported"
            raise ValueError(msg)
        elif gate == "prx" and len(wires) == 1:
            angle, phase = float(args["angle"]), float(args["phase"])
            if not math.isfinite(angle) or not math.isfinite(phase):
                msg = "Nonfinite PRX parameter"
                raise ValueError(msg)
            program.r(angle, phase, wires[0])
        elif gate == "cz" and len(wires) == 2 and abs(wires[0] - wires[1]) == 1:
            program.cz(wires[0], wires[1])
        else:
            msg = f"Unsupported native instruction: {gate}"
            raise ValueError(msg)
    if not readout:
        msg = "A sampled circuit needs measurements"
        raise ValueError(msg)
    state = Statevector.from_instruction(program)
    state.seed(17)
    samples = state.sample_memory(shots=shots)
    # Reverse key order so the client must use the compiler's readout metadata.
    keys = list(reversed(readout))
    measurements = {key: [[int(sample[-1 - wire]) for wire in readout[key]] for sample in samples] for key in keys}
    counts = Counter("".join(str(bit) for key in keys for bit in measurements[key][i]) for i in range(shots))
    return measurements, {"measurement_keys": keys, "counts": dict(counts)}


def braket_samples(program: dict, shots: int) -> dict:
    """Execute Braket OpenQASM with Amazon's independent local simulator."""
    from braket.devices import LocalSimulator
    from braket.ir.openqasm import Program

    result = LocalSimulator().run(Program(source=program["source"]), shots=shots).result()
    # The QDMI reader must reconstruct qubit order from measuredQubits.
    return {
        "inputsIndex": 0,
        "additionalMetadata": {"action": program},
        "measuredQubits": list(reversed(result.measured_qubits)),
        "measurements": result.measurements[:, ::-1].tolist(),
    }


class ProviderServer(ThreadingHTTPServer):
    """Retain accepted jobs while a worker computes their results."""

    def __init__(self, port: int, provider: str) -> None:
        """Listen on the isolated Compose network; no host port is published."""
        super().__init__(("0.0.0.0", port), Handler)  # ruff: ignore[hardcoded-bind-all-interfaces]
        self.provider = provider
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.available = True

    def simulate(self, request: dict) -> dict:
        """Compute every result before publishing successful completion."""
        if self.provider == "iqm":
            results = [iqm_samples(circuit, request["shots"]) for circuit in request["circuits"]]
            return {
                "status": "completed",
                "measurements": [r[0] for r in results],
                "counts": [r[1] for r in results],
            }
        action = json.loads(request["action"])
        programs = action.get("programs", [action])
        shots, remainder = divmod(request["shots"], len(programs))
        if remainder or shots < 1:
            msg = "Program-set shots must divide equally"
            raise ValueError(msg)
        results = [braket_samples(program, shots) for program in programs]
        if len(programs) == 1:
            documents = {"results.json": results[0]}
        else:
            # Keep S3 objects small, as in Braket program-set result manifests.
            documents = {
                "results.json": {
                    "taskMetadata": {"programMetadata": [{"executables": [{}]} for _ in programs]},
                    "programResults": [f"program-{i}.json" for i in range(len(programs))],
                },
                **{f"program-{i}.json": {"executableResults": [result]} for i, result in enumerate(results)},
            }
        return {"status": "COMPLETED", "documents": documents}

    def execute(self, identifier: str, request: dict) -> None:
        """Turn simulator errors into terminal job failures, never empty success."""
        try:
            update = self.simulate(request)
        except Exception as error:  # ruff: ignore[blind-except] - asynchronous job boundary
            update = {"status": "failed" if self.provider == "iqm" else "FAILED", "failureReason": str(error)}
        with self.lock:
            self.jobs[identifier].update(update)


class Handler(BaseHTTPRequestHandler):
    """Implement only the metadata, execution, and result paths used by QDMI."""

    @property
    def fixture(self) -> ProviderServer:
        """The server instantiated by this fixture's entry point."""
        return cast("ProviderServer", self.server)

    def reply(self, data: object, status: int = 200) -> None:
        """Return JSON without logging request bodies or authentication headers."""
        body = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        with contextlib.suppress(BrokenPipeError):
            self.wfile.write(body)

    def do_GET(self) -> None:
        """Serve device metadata, polling, and result artifacts."""
        path = unquote(urlsplit(self.path).path)
        if path == "/health":
            self.reply({"local_simulation": True})
        elif self.fixture.provider == "iqm":
            self.iqm_get(path)
        elif path.startswith("/device/"):
            capabilities = {
                "paradigm": {"qubitCount": 4},
                "action": {
                    "braket.ir.openqasm.program": {"supportedOperations": ["rx", "ry", "rz", "cz", "h", "cnot"]},
                    "braket.ir.openqasm.program_set": {
                        "version": ["1"],
                        "maximumExecutables": 4096,
                        "maximumTotalShots": 4194304,
                    },
                },
                "service": {"shotsRange": [1, 1024]},
            }
            self.reply({
                "deviceArn": ARN,
                "deviceName": "Local four-qubit simulator",
                "deviceType": "SIMULATOR",
                "deviceStatus": "ONLINE" if self.fixture.available else "OFFLINE",
                "providerName": "Local fixture",
                "deviceCapabilities": json.dumps(capabilities),
                "deviceQueueInfo": [],
            })
        elif path.startswith("/quantum-task/"):
            identifier = path.removesuffix("/payload").rsplit("/", maxsplit=1)[-1]
            with self.fixture.lock:
                job = self.fixture.jobs.get(identifier, {}).copy()
            if not job:
                self.reply({}, 404)
                return
            if path.endswith("/payload"):
                self.reply(job["request"])
                return
            self.reply({
                "quantumTaskArn": f"arn:aws:braket:us-east-1:000000000000:quantum-task/{identifier}",
                "status": job["status"],
                "deviceArn": ARN,
                "shots": job["request"]["shots"],
                "outputS3Bucket": "local-results",
                "outputS3Directory": identifier,
                "failureReason": job.get("failureReason", ""),
            })
        elif path.endswith(".json"):
            identifier, name = path.split("/")[-2:]
            with self.fixture.lock:
                result = self.fixture.jobs.get(identifier, {}).get("documents", {}).get(name)
            self.reply(result, 200 if result else 404)
        else:
            self.reply({}, 404)

    def iqm_get(self, path: str) -> None:
        """Serve the IQM Server API subset advertised by this fixture."""
        if path == "/api/v1/quantum-computers":
            self.reply({"quantum_computers": [{"id": "local", "alias": "emerald:mock"}]})
        elif path.endswith("/static-quantum-architectures"):
            self.reply([
                {
                    "qubits": QUBITS,
                    "computational_resonators": [],
                    "connectivity": [[QUBITS[i], QUBITS[i + 1]] for i in range(3)],
                }
            ])
        elif path.endswith("/dynamic-quantum-architecture"):
            self.reply(iqm_architecture())
        elif path.endswith("/metrics"):
            self.reply({"observations": []})
        elif path.endswith("/health"):
            self.reply({"healthy": self.fixture.available})
        elif path.endswith("/queue-availability"):
            self.reply({"available": ["normal"], "queue_length": 0})
        elif path.startswith("/api/v1/jobs/"):
            identifier = path.split("/")[4]
            with self.fixture.lock:
                job = self.fixture.jobs.get(identifier, {}).copy()
            if not job:
                self.reply({}, 404)
            elif path.endswith("/measurement_counts"):
                self.reply(job.get("counts", []))
            elif path.endswith("/measurements"):
                self.reply(job.get("measurements", []))
            elif path.endswith("/payload"):
                self.reply(job["request"])
            else:
                self.reply({
                    "id": identifier,
                    "type": "circuit",
                    "status": job["status"],
                    "errors": [{"message": job["failureReason"]}] if "failureReason" in job else [],
                })
        else:
            self.reply({}, 404)

    def do_POST(self) -> None:
        """Accept a local circuit job and schedule simulation."""
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 128 * 1024 * 1024:
            self.reply({}, 413)
            return
        try:
            request = json.loads(self.rfile.read(length))
            self.accept_job(request)
        except (KeyError, TypeError, ValueError):
            self.reply({"message": "Invalid local fixture request"}, 400)

    def accept_job(self, request: dict) -> None:
        """Queue a request after checking route and device availability."""
        if self.path == "/availability":
            self.fixture.available = bool(request["available"])
            self.reply({"available": self.fixture.available})
            return
        if not (
            (self.fixture.provider == "iqm" and self.path.endswith("/circuit"))
            or (self.fixture.provider == "braket" and self.path == "/quantum-task")
        ):
            self.reply({}, 404)
            return
        if not self.fixture.available:
            self.reply({"message": "Local simulated device unavailable"}, 503)
            return
        identifier = str(uuid.uuid4())
        with self.fixture.lock:
            self.fixture.jobs[identifier] = {
                "status": "running" if self.fixture.provider == "iqm" else "RUNNING",
                "request": request,
            }
        self.fixture.worker.submit(self.fixture.execute, identifier, request)
        self.reply(
            {"id": identifier}
            if self.fixture.provider == "iqm"
            else {"quantumTaskArn": f"arn:aws:braket:us-east-1:000000000000:quantum-task/{identifier}"}
        )

    def log_message(self, format: str, *args: object) -> None:  # ruff: ignore[builtin-argument-shadowing] - stdlib override
        """Keep request and credential data out of service logs."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("iqm", "braket"))
    parser.add_argument("--port", type=int, default=8080)
    options = parser.parse_args()
    ProviderServer(options.port, options.provider).serve_forever()

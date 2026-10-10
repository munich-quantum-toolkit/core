# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compile one captured application circuit against dated, read-only device data.

Inputs are AWS GetDevice responses and the official Qiskit IBM Runtime public
backend snapshot. Compilation uses the real local QDMI SC provider; execution
uses DDSIM only. This script has no hardware submission path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from capture_programs import ROOT, capture_circuit, capture_variant, stage

IBM_REVISION = "fa4cecc76321f9559456b132cfdfc7d06999f802"
MAPPING = {"trials": 3, "iterations": 2, "lookahead": 8, "search-memory-limit": 4194304}


def digest(value: str | bytes) -> str:
    """Return a SHA-256 digest of exact text or bytes."""
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def summary(values: list[float], method: str) -> dict[str, Any]:
    """Summarize reported fidelities without filling missing calibration data.

    Returns:
        Reported range and mean, or an explicit unavailable marker.

    Raises:
        ValueError: If any fidelity is not a fraction in [0, 1].
    """
    if not values:
        return {"available": False, "reason": "Not reported in this source"}
    if any(not 0 <= value <= 1 for value in values):
        msg = "Device fidelity must be a fraction in [0, 1]"
        raise ValueError(msg)
    return {
        "available": True,
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "method": method,
    }


def braket_target(raw: dict[str, Any], *, ionq: bool = False) -> dict[str, Any]:
    """Normalize a public device response, preserving reported native operations.

    Returns:
        Target metadata, original provider evidence, and a local SC model.
    """
    capabilities = json.loads(raw["deviceCapabilities"])
    paradigm = capabilities["paradigm"]
    count = paradigm["qubitCount"]
    graph = paradigm["connectivity"]["connectivityGraph"]
    offset = 0 if ionq else 1
    edges = sorted({
        tuple(sorted((int(a) - offset, int(b) - offset))) for a, neighbors in graph.items() for b in neighbors
    })
    provider = capabilities["provider"]
    model: dict[str, Any] = {
        "schema-version": 1,
        "name": raw["deviceName"] + " · captured AWS metadata",
        "numQubits": count,
        "couplings": [(int(a) - offset, int(b) - offset) for a, neighbors in graph.items() for b in neighbors],
    }
    if ionq:
        fidelities = provider["fidelity"]
        calibration = {
            name: summary([fidelities[key]["mean"]], "Provider-reported mean")
            for name, key in (("one_qubit", "1Q"), ("two_qubit", "2Q"), ("readout", "spam"))
        }
        # Core has no GPI/GPI2 target basis. These are actual supported QIS gates;
        # the physical native gate names remain separate metadata.
        supported = capabilities["action"]["braket.ir.openqasm.program"]["supportedOperations"]
        assert all(name in supported for name in ("rx", "ry", "rz", "cnot"))
        model["operations"] = [{"name": name, "numQubits": 1, "numParameters": 1} for name in ("rx", "ry", "rz")]
        model["operations"].extend((
            {"name": "cx", "numQubits": 2, "numParameters": 0},
            {"name": "measure", "numQubits": 1, "numParameters": 0},
        ))
        basis_note = (
            "Compiles to the provider's supported RX/RY/RZ/CNOT QIS interface; "
            "GPI/GPI2/ZZ hardware synthesis remains provider-owned."
        )
    else:
        properties = provider["properties"]
        one = properties["one_qubit"]
        # Calibration can contain a disabled edge absent from current topology.
        two = {
            pair: p
            for pair, p in properties["two_qubit"].items()
            if tuple(sorted(int(q) - 1 for q in pair.split("-"))) in edges
        }
        calibration = {
            "one_qubit": summary(
                [p["f1Q_simultaneous_RB"] for p in one.values() if "f1Q_simultaneous_RB" in p],
                "Simultaneous randomized benchmarking",
            ),
            "two_qubit": summary([p["fCZ"] for p in two.values() if "fCZ" in p], "Reported CZ fidelity"),
            "readout": summary([p["fRO"] for p in one.values() if "fRO" in p], "Reported readout fidelity"),
        }
        model["operations"] = [
            {
                "name": "r",
                "numQubits": 1,
                "numParameters": 2,
                "siteOverrides": [
                    {"sites": [int(q) - 1], "fidelity": p["f1Q_simultaneous_RB"]}
                    for q, p in one.items()
                    if "f1Q_simultaneous_RB" in p
                ],
            },
            {
                "name": "cz",
                "numQubits": 2,
                "numParameters": 0,
                "siteOverrides": [
                    {"sites": [int(q) - 1 for q in pair.split("-")], "fidelity": p["fCZ"]}
                    for pair, p in two.items()
                    if "fCZ" in p
                ],
            },
            {
                "name": "measure",
                "numQubits": 1,
                "numParameters": 0,
                "siteOverrides": [{"sites": [int(q) - 1], "fidelity": p["fRO"]} for q, p in one.items() if "fRO" in p],
            },
        ]
        basis_note = (
            "R(theta, phi) is the compiler representation of IQM PRX; "
            "CZ and terminal measurement retain their physical operands."
        )
    return {
        "id": "aws.ionq.forte-1" if ionq else "iqm.emerald",
        "label": "IonQ Forte-1 · all-to-all" if ionq else "IQM Emerald",
        "metadata": {
            "qubits": count,
            "sites": [
                {"id": i, "name": str(i) if ionq else f"QB{i + 1}", "provider_id": i + offset} for i in range(count)
            ],
            "edges": edges,
            "operations": [{"name": name} for name in paradigm["nativeGateSet"]],
            "calibration": calibration,
            "calibration_date": capabilities["service"].get("updatedAt"),
            "basis_note": basis_note,
        },
        "provenance": {
            "source": "AWS Braket GetDevice",
            "access": "Read-only metadata request; no task submitted",
            "device_arn": raw["deviceArn"],
            "status": raw["deviceStatus"],
            "raw_sha256": digest(json.dumps(raw, sort_keys=True)),
            "raw": raw,
            "source_url": "https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html",
        },
        "compiler_model": model,
    }


def ibm_target(configuration: dict[str, Any], properties: dict[str, Any]) -> dict[str, Any]:
    """Use the dated public backend snapshot without presenting it as live data.

    Returns:
        Target metadata, original public snapshot, and a local SC model.
    """
    count = configuration["n_qubits"]
    edges = sorted({tuple(sorted(edge)) for edge in configuration["coupling_map"]})
    gates: dict[str, list[dict[str, Any]]] = {name: [] for name in configuration["basis_gates"]}
    one, two = [], []
    for gate in properties["gates"]:
        if gate["gate"] not in gates:
            continue
        parameters = {p["name"]: p["value"] for p in gate["parameters"]}
        item: dict[str, Any] = {"sites": gate["qubits"]}
        if "gate_error" in parameters:
            item["fidelity"] = 1 - parameters["gate_error"]
            (two if len(gate["qubits"]) == 2 else one).append(item["fidelity"])
        gates[gate["gate"]].append(item)
    readout = [1 - next(p["value"] for p in qubit if p["name"] == "readout_error") for qubit in properties["qubits"]]
    operations = [
        {
            "name": name,
            "numQubits": 2 if name == "cz" else 1,
            "numParameters": 1 if name == "rz" else 0,
            "siteOverrides": overrides,
        }
        for name, overrides in gates.items()
    ]
    operations.append({
        "name": "measure",
        "numQubits": 1,
        "numParameters": 0,
        "siteOverrides": [{"sites": [i], "fidelity": value} for i, value in enumerate(readout)],
    })
    return {
        "id": "ibm.ibm_miami",
        "label": "IBM Nighthawk 120 · ibm_miami",
        "metadata": {
            "qubits": count,
            "sites": [{"id": i, "name": str(i)} for i in range(count)],
            "edges": edges,
            "operations": [{"name": name} for name in configuration["basis_gates"]],
            "processor": configuration["processor_type"],
            "calibration_date": properties["last_update_date"],
            "calibration": {
                "one_qubit": summary(one, "1 - reported gate_error; includes virtual/id gates"),
                "two_qubit": summary(two, "1 - reported CZ gate_error"),
                "readout": summary(readout, "1 - reported readout_error"),
            },
            "basis_note": (
                "Official public Nighthawk r1 snapshot, April 2026. "
                "Current r2 backend calibration requires authentication and is not substituted here."
            ),
        },
        "provenance": {
            "source": "Official Qiskit IBM Runtime public backend snapshot",
            "access": "Public download; no IBM account or hardware execution",
            "revision": IBM_REVISION,
            "source_url": f"https://github.com/Qiskit/qiskit-ibm-runtime/tree/{IBM_REVISION}/qiskit_ibm_runtime/fake_provider/backends/miami",
            "raw_sha256": digest(json.dumps([configuration, properties], sort_keys=True)),
            "raw": {"configuration": configuration, "properties": properties},
        },
        "compiler_model": {
            "schema-version": 1,
            "name": "IBM ibm_miami · public April 2026 snapshot",
            "numQubits": count,
            "couplings": configuration["coupling_map"],
            "operations": operations,
        },
    }


def circuit_metrics(circuit: dict[str, Any]) -> dict[str, Any]:
    """Count actual static gate occurrences and dependencies, not executed shots.

    Returns:
        Gate counts and wire-dependency depth, including terminal measurements.

    Raises:
        ValueError: If the input has structured control flow.
    """
    depth = dict.fromkeys((q["id"] for q in circuit["qubits"]), 0)
    counts: dict[str, int] = {}
    for operation in circuit["operations"]:
        if operation.get("blocks"):
            msg = "Application target comparison expects a straight-line circuit"
            raise ValueError(msg)
        wires = operation["qubits"]
        if not wires:
            continue
        counts[operation["name"]] = counts.get(operation["name"], 0) + 1
        next_depth = max(depth[q] for q in wires) + 1
        for q in wires:
            depth[q] = next_depth
    return {
        "operations": sum(counts.values()),
        "counts": counts,
        "depth": max(depth.values(), default=0),
        "active_qubits": len(depth),
    }


def compile_device(target: dict[str, Any], source: str, compiler: Path, library: Path, shots: int) -> None:
    """Capture real compiler stages, then execute the unchanged QIR on DDSIM."""
    from mqt.core.mlir import CompilerTarget, QCProgram  # ruff: ignore[import-outside-top-level]

    model = target["compiler_model"]
    model.setdefault("durationUnit", {"unit": "ns", "scaleFactor": 1.0})
    model.setdefault("qubitProperties", {"defaults": {}, "overrides": []})
    old_json = os.environ.get("MQT_CORE_QDMI_SC_CONFIG_JSON")
    old_file = os.environ.pop("MQT_CORE_QDMI_SC_CONFIG_FILE", None)
    os.environ["MQT_CORE_QDMI_SC_CONFIG_JSON"] = json.dumps(model)
    try:
        compiler_target = CompilerTarget.from_device_id("mqt.sc.default")
    finally:
        if old_json is None:
            os.environ.pop("MQT_CORE_QDMI_SC_CONFIG_JSON", None)
        else:
            os.environ["MQT_CORE_QDMI_SC_CONFIG_JSON"] = old_json
        if old_file is not None:
            os.environ["MQT_CORE_QDMI_SC_CONFIG_FILE"] = old_file
    qc = QCProgram.from_openqasm_str(source)
    result = capture_variant(
        qc, compiler_target, model, compiler, unroll=False, timeout=600, mapping=MAPPING, trace=True
    )
    artifact = stage("source", "Actual LiH shadow circuit · OpenQASM 2.0", "qasm", source)
    artifact["circuit"] = capture_circuit(qc.to_qiskit())
    result["stages"].insert(0, artifact)
    result["metrics"] = {s["id"]: circuit_metrics(s["circuit"]) for s in result["stages"] if "circuit" in s}
    result["source_sha256"] = digest(source)
    result["mapping_options"] = MAPPING
    result["routing_trace_note"] = (
        "Actual placement refinement, not quantum execution. Before/after arrays map compiler roots to physical sites. "
        "native_count and depth are predicted native two-qubit count/depth; they exclude one-qubit gates."
    )
    final_route = result["routing_trace"][-1]
    assert final_route["phase"] == "final-routing"
    assert final_route["swaps"] == len(result["layout"]["swaps"])
    inputs = len(result["layout"]["initial"])
    assert final_route["before"][:inputs] == result["layout"]["initial"]
    assert final_route["after"][:inputs] == result["layout"]["final"]
    result["provider"] = "mqt.sc.default (local QDMI SC model populated from the recorded source)"
    for s in result["stages"]:
        if s["id"] == "place-and-route":
            s["label"] = "Actual placement and routing"
        elif s["id"] == "target-native-synthesis":
            s["label"] = "Synthesis to the declared target interface"
    exported = next(item for item in result["exports"] if item["id"] == "qir-adaptive")
    request = {
        "payload": exported["code"],
        "format_id": 4,
        "format_name": "QIR_ADAPTIVE_STRING",
        "shots": shots,
        "seed": 7,
    }
    process = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - executes the local capture script
        [sys.executable, str(Path(__file__).with_name("capture_execution.py")), "--worker", "--library", str(library)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=True,
        timeout=330,
    )
    result["execution"] = json.loads(process.stdout)
    assert result["execution"]["payload_sha256"] == exported["sha256"]
    target["compilation"] = result


def main() -> None:
    """Write standalone target evidence from already downloaded public metadata."""
    if sys.argv[1:] == ["--worker"]:
        request = json.load(sys.stdin)
        target = request["target"]
        compile_device(target, request["source"], Path(request["compiler"]), Path(request["library"]), request["shots"])
        json.dump(target, sys.stdout)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("emerald", "ibm-config", "ibm-properties", "source"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ionq", type=Path)
    parser.add_argument("--compiler", type=Path, default=ROOT / "build/release-clang-ipo/mlir/tools/mqt-cc/mqt-cc")
    parser.add_argument(
        "--library", type=Path, default=ROOT / "build/release-clang-ipo/lib/libmqt-core-qdmi-ddsim-device.so"
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "captures/devices.json")
    parser.add_argument("--shots", type=int, default=512)
    args = parser.parse_args()
    targets = [
        braket_target(json.loads(args.emerald.read_text())),
        ibm_target(json.loads(args.ibm_config.read_text()), json.loads(args.ibm_properties.read_text())),
    ]
    if args.ionq:
        targets.append(braket_target(json.loads(args.ionq.read_text()), ionq=True))
    metadata_paths = [args.emerald, args.ibm_properties, *([args.ionq] if args.ionq else [])]
    for target, path in zip(targets, metadata_paths, strict=True):
        target["provenance"]["retrieved_at"] = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
        target["provenance"]["raw_hash_encoding"] = "JSON encoded with sorted keys"
    source = args.source.read_text()
    data = {
        "schema_version": 1,
        "targets": targets,
        "provenance": {
            "generated_at": datetime.now(UTC).isoformat(),
            "capture_script_sha256": digest(Path(__file__).read_bytes()),
            "compiler_capture_script_sha256": digest(Path(__file__).with_name("capture_programs.py").read_bytes()),
            "mapping_source_sha256": digest(
                (ROOT / "mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp").read_bytes()
            ),
            "compiler_sha256": digest(args.compiler.read_bytes()),
            "library_sha256": digest(args.library.read_bytes()),
            "worker_binary_sha256": digest(args.library.with_name("mqt-core-ddsim-worker").read_bytes()),
            "execution_script_sha256": digest(Path(__file__).with_name("capture_execution.py").read_bytes()),
            "worker_source_sha256": digest((ROOT / "src/qdmi/devices/dd/WorkerMain.cpp").read_bytes()),
            "source_sha256": digest(source),
        },
    }
    for index, target in enumerate(targets):
        print("Capturing target", target["id"], flush=True)
        # The built-in QDMI catalogue caches configured device sessions.
        # Isolate each model so one device can never inherit another's metadata.
        request = {
            "target": target,
            "source": source,
            "compiler": str(args.compiler.resolve()),
            "library": str(args.library.resolve()),
            "shots": args.shots,
        }
        process = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - executes this capture script
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=True,
            timeout=900,
        )
        targets[index] = captured = json.loads(process.stdout)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n")
        print(captured["id"], captured["compilation"]["metrics"], flush=True)


if __name__ == "__main__":
    main()

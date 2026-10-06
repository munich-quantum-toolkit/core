# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Capture real compiler stages for the offline MQSF presentation.

Run with the checkout's installed native bindings and built mqt-cc. The compiler
uses the bundled Emerald model plus explicitly recorded presentation capabilities;
the result is intended for DDSIM, not submission to physical Emerald hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

    from mqt.core.mlir import CompilerTarget, QCOProgram, QCProgram

ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES = (
    "forward-branching",
    "counted-iteration",
    "conditional-loop",
    "multiway-branching",
    "qir.dynamic-qubit-management",
    "qir.dynamic-result-management",
    "qir.arrays",
    "qir.ir-functions",
    "qir.multiple-return-points",
    "qir.int-computations",
    "qir.float-computations",
)
TARGET_PASSES = (
    ("unroll-loops-for-payload", "Specialize addresses required by the target"),
    ("legalize-control-flow", "Check remaining control flow"),
    ("place-and-route", "Place and route on Emerald"),
    ("target-native-synthesis", "Synthesize native R/CZ operations"),
)


def extract_pass_ir(dumps: str, pass_name: str) -> str:
    """Extract one exact module from mqt-cc's pass-manager output.

    Returns:
        The requested module text without pass-manager delimiters.

    Raises:
        ValueError: If the requested pass did not produce exactly one snapshot.
    """
    headers = list(re.finditer(r"^//[^\n]*IR Dump After [^\n]+\n", dumps, re.MULTILINE))
    matches = []
    for index, header in enumerate(headers):
        if re.search(r": " + re.escape(pass_name) + r"(?:\{| )", header.group()):
            end = headers[index + 1].start() if index + 1 < len(headers) else len(dumps)
            matches.append(dumps[header.end() : end].strip() + "\n")
    if len(matches) != 1 or not matches[0].startswith("module"):
        message = f"missing or ambiguous compiler snapshot for {pass_name}"
        raise ValueError(message)
    return matches[0]


def extract_layout(ir: str, routed_qasm: str) -> dict[str, Any]:
    """Read recorded placement and static SWAP sites without inventing a trace.

    SWAPs are listed in source order with their enclosing control-flow context.
    They are not a measurement-dependent runtime execution sequence.

    Returns:
        Logical-to-physical placement and static SWAP records.

    Raises:
        ValueError: If layout metadata is missing or contains invalid permutations.
    """
    match = re.search(r"mqt\.layout\s*=\s*\{([^}]+)\}", ir)
    if match is None:
        message = "compiler did not attach layout metadata"
        raise ValueError(message)
    fields = match.group(1)
    arrays = {}
    for name, values in re.findall(r"(initial|routing|sites)\s*=\s*array<i64:\s*([^>]*)>", fields):
        arrays[name] = [int(value.strip()) for value in values.split(",")]
    initial = arrays["initial"]
    routing = arrays.get("routing", list(range(len(initial))))
    sites = arrays.get("sites", list(range(len(initial))))
    if sorted(initial) != list(range(len(initial))) or sorted(routing) != list(range(len(initial))):
        message = "compiler layout is not a complete permutation"
        raise ValueError(message)
    count_match = re.search(r"input_count\s*=\s*(\d+)", fields)
    if count_match is None or len(sites) != len(initial):
        message = "compiler layout has no valid source-qubit count or site list"
        raise ValueError(message)
    count = int(count_match.group(1))
    if count > len(initial):
        message = "compiler layout source-qubit count exceeds its permutation"
        raise ValueError(message)
    swaps = []
    contexts = []
    regions: list[str] = []
    for line_number, line in enumerate(routed_qasm.splitlines(), 1):
        stripped = line.strip()
        for _ in range(stripped.count("}")):
            if regions:
                regions.pop()
        swap = re.fullmatch(r"swap\s+\$(\+?\d+),\s*\$(\+?\d+)\s*;", stripped)
        if swap:
            swaps.append([int(swap.group(1)), int(swap.group(2))])
            contexts.append({"line": line_number, "regions": regions.copy()})
        if "{" in stripped:
            regions.append(stripped)
    return {
        "initial": [sites[position] for position in initial[:count]],
        "final": [sites[routing[position]] for position in initial[:count]],
        "swaps": swaps,
        "swap_contexts": contexts,
        "swap_order": "Static source order, including conditional regions; not an execution trace.",
        "input_count": count,
        "metadata": {"initial": initial, "routing": routing, "sites": sites},
    }


def stage(identifier: str, label: str, language: str, code: str) -> dict[str, Any]:
    """Describe an exact captured artifact and its structural control-flow count.

    Returns:
        Presentation metadata, exact source text, and its digest.
    """
    lines = code.splitlines()
    start = 0
    markers: tuple[str, ...] = ()
    if language == "cpp":
        markers = ("SmallVector<Value> shor(", "iterativeQPE(")
    elif language == "mlir":
        start = next((index for index, line in enumerate(lines) if "mqt.entry_point" in line), 0)
        markers = ("scf.for ", "scf.while ", "qco.if ")
    focus = next(
        (index + 1 for index in range(start, len(lines)) if any(marker in lines[index] for marker in markers)),
        start + 1,
    )
    return {
        "id": identifier,
        "label": label,
        "language": language,
        "code": code,
        "focus_line": focus,
        "sha256": hashlib.sha256(code.encode()).hexdigest(),
        "control_flow": {
            "counted_loops": len(re.findall(r"\bscf\.for\b", code)),
            "conditional_loops": len(re.findall(r"\bscf\.while\b", code)),
            "quantum_branches": len(re.findall(r"\bqco\.if\b", code)),
        },
    }


def payload_specification() -> str:
    """Serialize the presentation's explicitly enabled Adaptive QIR contract.

    Returns:
        The typed payload attribute accepted by mqt-cc.
    """
    capabilities = ", ".join(f'<id = "{name}", value = 0, constraints = []>' for name in CAPABILITIES)
    return (
        '#mqt.payload_spec<format = <id = "qir", version = "2.1.0", '
        'profile = "adaptive", encoding = text>, capabilities = ['
        + capabilities
        + "], optional_capabilities_known = true>"
    )


def compile_target(program: QCOProgram, compiler: Path, model: dict[str, Any], timeout: int) -> tuple[str, str]:
    """Run the production target pipeline and capture its actual pass output.

    Returns:
        Final QCO source and the pass-manager snapshot stream.
    """
    import mqt.core.qdmi  # ruff: ignore[import-outside-top-level]

    environment = os.environ.copy()
    library_path = str(Path(mqt.core.qdmi.__file__).parent / "lib")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (library_path, environment.get("LD_LIBRARY_PATH"))))
    environment.pop("MQT_CORE_QDMI_SC_CONFIG_FILE", None)
    environment["MQT_CORE_QDMI_SC_CONFIG_JSON"] = json.dumps(model)
    with TemporaryDirectory(prefix="mqsf-compiler-") as directory:
        source = Path(directory) / "input.mlir"
        source.write_text(program.ir, encoding="utf-8")
        command = [
            str(compiler),
            str(source),
            "--qdmi-device=mqt.sc.default",
            f"--payload-spec={payload_specification()}",
            "--emit=qco-optimized",
            "--seed=7",
            "--mapping-trials=1",
            "--mapping-iterations=0",
            "--mapping-lookahead=8",
            "--mapping-search-memory-limit=4194304",
            "--mlir-disable-threading",
            "--mlir-print-ir-module-scope",
            "--mlir-print-ir-after=" + ",".join(name for name, _ in TARGET_PASSES),
        ]
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            command, capture_output=True, text=True, check=True, env=environment, timeout=timeout
        )
    return result.stdout, result.stderr


def render_qiskit(circuit: QuantumCircuit) -> str:
    """Draw a small exported circuit without materializing an oversized diagram.

    Returns:
        Qiskit's own text visualization.

    Raises:
        ValueError: If the successful export is too large for the presentation.
    """
    if len(circuit.data) > 1000:
        message = (
            f"Qiskit export succeeded ({circuit.num_qubits} qubits, {len(circuit.data)} top-level instructions); "
            "text diagram omitted because it exceeds the 1,000-instruction presentation limit."
        )
        raise ValueError(message)
    return str(circuit.draw(output="text", fold=120, idle_wires=False))


def capture_variant(
    qc: QCProgram,
    target: CompilerTarget,
    model: dict[str, Any],
    compiler: Path,
    *,
    unroll: bool,
    timeout: int,
) -> dict[str, Any]:
    """Capture both compiler stages and independently attempted export formats.

    Returns:
        One structured or unrolled variant with real compiler artifacts.

    Raises:
        ValueError: If extracted routing evidence disagrees with the target or IR.
    """
    from mqt.core.mlir import QCOProgram, QIRProfile  # ruff: ignore[import-outside-top-level]

    variant: dict[str, Any] = {
        "id": "unrolled" if unroll else "structured",
        "label": "Bounded loops unrolled" if unroll else "Preserve supported structure",
        "stages": [stage("qc", "Generated QC", "mlir", qc.ir)],
        "exports": [],
        "layout": {"initial": [], "final": [], "swaps": []},
    }
    qco = qc.to_qco(copy=True)
    variant["stages"].append(stage("qco", "Value-based QCO", "mlir", qco.ir))
    qco.run_pass_pipeline("inline")
    qco.cleanup()
    variant["stages"].append(stage("optimized", "Inline and simplify", "mlir", qco.ir))
    if unroll:
        qco.unroll_quantum_loops()
        qco.cleanup()
        variant["stages"].append(stage("unrolled", "Unroll bounded quantum loops", "mlir", qco.ir))
    native_ir, dumps = compile_target(qco, compiler, model, timeout)
    for name, label in TARGET_PASSES:
        variant["stages"].append(stage(name, label, "mlir", extract_pass_ir(dumps, name)))
    native = QCOProgram.from_mlir_str(native_ir)
    routed = QCOProgram.from_mlir_str(extract_pass_ir(dumps, "place-and-route"))
    routed_qasm = routed.to_qc(copy=True).to_openqasm3().source
    variant["layout"] = extract_layout(native_ir, routed_qasm)
    variant["layout"]["routed_openqasm3"] = routed_qasm
    edges = {tuple(sorted(edge)) for edge in model["couplings"]}
    if any(tuple(sorted(swap)) not in edges for swap in variant["layout"]["swaps"]):
        message = "captured routing SWAP uses an edge absent from the Emerald model"
        raise ValueError(message)
    if len(variant["layout"]["swaps"]) != len(re.findall(r"\bqco\.swap\b", routed.ir)):
        message = "routed OpenQASM and QCO disagree on the number of captured SWAPs"
        raise ValueError(message)
    variant["target_conformance"] = {
        "verified": True,
        "scope": "Emerald model with presentation-only reset and control-flow overrides",
        "measurement_feedback_retained": True,
    }
    for identifier, label, language, export in (
        ("qir-adaptive", "QIR Adaptive", "llvm", lambda: native.to_qc(copy=True).to_qir(QIRProfile.ADAPTIVE).llvm_ir),
        ("openqasm3", "OpenQASM 3", "qasm", lambda: native.to_qc(copy=True).to_openqasm3().source),
        (
            "qiskit",
            "Qiskit circuit",
            "text",
            lambda: render_qiskit(native.to_qiskit(target=target)),
        ),
    ):
        try:
            code = export()
        except (ImportError, RuntimeError, ValueError, TypeError) as error:
            variant["exports"].append({"id": identifier, "label": label, "unavailable_reason": str(error)})
        else:
            variant["exports"].append(stage(identifier, label, language, code))
    return variant


def capture(compiler: Path, selected: str, timeout: int) -> dict[str, Any]:
    """Generate the selected benchmark cases from this checkout's native compiler.

    Returns:
        Presentation fixture data containing only captured compiler outputs.
    """
    from mqt.core.bench import qpe, shor  # ruff: ignore[import-outside-top-level]
    from mqt.core.mlir import CompilerTarget  # ruff: ignore[import-outside-top-level]

    model = json.loads((ROOT / "json/sc/iqm-emerald.json").read_text(encoding="utf-8"))
    model["name"] = "IQM Emerald topology — MQSF demonstration"
    model["operations"].append({"name": "reset", "numQubits": 1, "numParameters": 0})
    target = CompilerTarget.from_device_id("mqt.sc.default", device_config=json.dumps(model))
    git = shutil.which("git")
    assert git is not None, "Capturing provenance requires Git"
    revision = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [git, "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    data: dict[str, Any] = {
        "schema_version": 1,
        "title": "System Software for Quantum Computing: From the Metal to the User",
        "speaker": {"name": "Lukas Burgholzer", "affiliations": ["MQSC", "TUM"]},
        "provenance": {
            "core_revision": revision,
            "generated_at": datetime.now(UTC).isoformat(),
            "patches": [
                "Presentation-only Emerald reset operation; not verified on physical hardware.",
                "Presentation-only Adaptive QIR control-flow and optional capabilities; executed with DDSIM.",
            ],
            "model_path": "json/sc/iqm-emerald.json",
            "model_sha256": hashlib.sha256((ROOT / "json/sc/iqm-emerald.json").read_bytes()).hexdigest(),
            "compiler_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
            "capture_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "mapping": {"seed": 7, "trials": 1, "iterations": 0, "lookahead": 8, "search_memory_limit": 4194304},
            "capabilities": list(CAPABILITIES),
            "target_configuration": model,
        },
        "device": {
            "name": "IQM Emerald",
            "label": "Emerald topology · R/CZ · presentation reset/control-flow override · DDSIM execution",
            "native_gates": ["r", "cz", "measure", "reset (demo override)"],
            "sites": [{"id": site.id, "name": site.name or str(site.id)} for site in target.sites],
            "edges": [list(edge) for edge in target.couplings],
        },
        "scenarios": [],
    }
    cases = (
        (
            "shor15",
            "Shor: factor 15",
            (
                "Semiclassical order finding for N = 15 and base 2. "
                "Indexed quantum loops specialize for routing; feedback remains."
            ),
            {"number": 15, "base": 2},
            shor.Shor(shor.Options(number=15, base=2)),
            "Shor.cpp",
        ),
        (
            "qpe",
            "Iterative phase estimation",
            "Four phase bits from two qubits. Bounded-loop unrolling retains measurement feedback and reset.",
            {"precision": 4, "phase_numerator": 3, "phase_denominator": 8},
            qpe.QPE(qpe.Options(precision=4, phase=Fraction(3, 8), method=qpe.Method.ITERATIVE)),
            "QPE.cpp",
        ),
    )
    for identifier, label, summary, parameters, benchmark, generator in cases:
        if selected not in {"all", identifier}:
            continue
        qc = benchmark.generate()
        scenario: dict[str, Any] = {
            "id": identifier,
            "label": label,
            "summary": summary,
            "parameters": parameters,
            "instance_specification": json.loads(benchmark.instance_specification_json),
            "case_id": benchmark.case_id,
            "variants": [],
        }
        for unroll in (False, True):
            sys.stderr.write(f"Capturing {identifier}: {'unrolled' if unroll else 'structured'}\n")
            variant = capture_variant(qc, target, model, compiler, unroll=unroll, timeout=timeout)
            variant["stages"].insert(
                0,
                stage(
                    "source",
                    "Benchmark generator (C++)",
                    "cpp",
                    (ROOT / "mlir/bench/programs" / generator).read_text(encoding="utf-8"),
                ),
            )
            scenario["variants"].append(variant)
        data["scenarios"].append(scenario)
    return data


def main() -> int:
    """Write reproducible offline fixtures, failing if a required export fails.

    Returns:
        Zero when both required formats were exported for every variant.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=ROOT / "build/release-clang-ipo/mlir/tools/mqt-cc/mqt-cc")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "captures/programs.json")
    parser.add_argument("--scenario", choices=("all", "shor15", "qpe"), default="all")
    parser.add_argument("--timeout", type=int, default=600)
    arguments = parser.parse_args()
    data = capture(arguments.compiler.resolve(), arguments.scenario, arguments.timeout)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return int(
        any(
            "unavailable_reason" in export
            for scenario in data["scenarios"]
            for variant in scenario["variants"]
            for export in variant["exports"]
            if export["id"] in {"qir-adaptive", "openqasm3"}
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Capture real compiler stages for the offline MQSF presentation.

Run with the checkout's installed native bindings and built mqt-cc. Curated
examples retain portable gates and control flow for DDSIM. The device walkthrough
uses the separate target-capture entry point and the hardware pipeline below.
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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit import QuantumCircuit

    from mqt.core.mlir import CompilerTarget, QCOProgram, QCProgram

ROOT = Path(__file__).resolve().parents[2]
PARITY_SOURCE = """OPENQASM 3.1;
include "stdgates.inc";
qubit[2] data;
qubit flag;
h data;
for int round in [0:1] {
  reset flag;
  cx data, flag;
  bit s = measure flag;
  if (s) { x data[1]; }
}
bit[2] result = measure data;
"""
RUS_SOURCE = """OPENQASM 3.1;
include "stdgates.inc";
qubit[2] ancilla;
qubit data;
bit accepted = 0;
h data;
while (!accepted) {
  reset ancilla;
  h ancilla;
  tdg ancilla[0]; cx ancilla[1], ancilla[0];
  t ancilla[0]; h ancilla[0];
  bit first = measure ancilla[0];
  if (!first) {
    t data; z data; cx data, ancilla[1];
    t ancilla[1]; h ancilla[1];
    bit second = measure ancilla[1];
    if (second) { z data; }
    else { accepted = 1; }
  }
}
h data;
bit result = measure data;
"""


def qpe_source(precision: int = 8) -> str:
    """Return iterative QPE for phase 1/3 on a four-qubit eigenstate.

    The four controlled phase gates each contribute 1/12 of a turn. One query
    qubit is reset and reused; classical feedback implements the inverse QFT.
    """
    return f"""OPENQASM 3.1;
include "stdgates.inc";
qubit[5] q; bit[{precision}] result = 0;
x q[1]; x q[2]; x q[3]; x q[4];
if (true) {{
  float[64] theta = 0;
  for uint k in [0:{precision - 1}] {{
    reset q[0];
    h q[0];
    for int power in [1:1 << ({precision - 1}-k)] {{
      cp(pi/6) q[0], q[1];
      cp(pi/6) q[0], q[2];
      cp(pi/6) q[0], q[3];
      cp(pi/6) q[0], q[4];
    }}
    p(theta) q[0];
    h q[0];
    result[k] = measure q[0];
    theta = theta / 2 - pi / 2 * float[64](result[k]);
  }}
}}
"""


def capture_openqasm_circuit(source: str) -> dict[str, Any]:
    """Read the submitted source when dynamic classical data prevents Qiskit export.

    Returns:
        Static gate operands, nested loops, and exact classical expressions.

    Unsupported statements or operands fail with ValueError.
    """
    import openqasm3  # ruff: ignore[import-outside-top-level]
    from openqasm3 import ast  # ruff: ignore[import-outside-top-level]

    qubits, clbits, initializers = [], [], []
    registers: dict[str, list[int]] = {}

    def operand(node: ast.IndexedIdentifier | ast.Identifier) -> int:
        if not isinstance(node, ast.IndexedIdentifier) or len(node.indices) != 1:
            msg = "Source diagrams require one static qubit index"
            raise ValueError(msg)
        group = node.indices[0]
        if not isinstance(group, list) or len(group) != 1:
            msg = "Source diagrams require one static qubit index"
            raise ValueError(msg)
        index = group[0]
        if not isinstance(index, ast.IntegerLiteral) or not 0 <= index.value < len(registers.get(node.name.name, [])):
            msg = "Source diagram qubit index is dynamic or out of bounds"
            raise ValueError(msg)
        return registers[node.name.name][index.value]

    def operations(statements: Sequence[ast.Statement | ast.Pragma], prefix: str = "") -> list[dict[str, Any]]:
        result = []
        for index, node in enumerate(statements):
            assert node.span is not None, "Parsed source nodes must retain their locations"
            item: dict[str, Any] = {
                "id": f"{prefix}{index}",
                "source_line": node.span.start_line,
                "qubits": [],
                "clbits": [],
                "parameters": [],
            }
            if isinstance(node, ast.Include):
                continue
            if isinstance(node, ast.QubitDeclaration) and isinstance(node.size, ast.IntegerLiteral):
                registers[node.qubit.name] = list(range(len(qubits), len(qubits) + node.size.value))
                qubits.extend(
                    {"id": wire, "label": f"{node.qubit.name}[{i}]"}
                    for i, wire in enumerate(registers[node.qubit.name])
                )
                continue
            if isinstance(node, ast.ClassicalDeclaration):
                if isinstance(node.type, ast.BitType):
                    size = node.type.size.value if isinstance(node.type.size, ast.IntegerLiteral) else 1
                    base = len(clbits)
                    clbits.extend({"id": base + i, "label": f"{node.identifier.name}[{i}]"} for i in range(size))
                if node.init_expression is not None:
                    initializers.append(openqasm3.dumps(node).strip())
                continue
            if (
                isinstance(node, ast.BranchingStatement)
                and isinstance(node.condition, ast.BooleanLiteral)
                and node.condition.value
                and not node.else_block
            ):
                result.extend(operations(node.if_block, f"{prefix}{index}."))
                continue
            if isinstance(node, ast.ForInLoop) and isinstance(node.set_declaration, ast.RangeDefinition):
                bounds = node.set_declaration
                item.update(
                    name="for_loop",
                    variable=node.identifier.name,
                    range=openqasm3.dumps(bounds).strip(),
                    blocks=[operations(node.block, f"{prefix}{index}.")],
                )
                item["qubits"] = sorted({wire for child in item["blocks"][0] for wire in child["qubits"]})
                if (
                    isinstance(bounds.start, ast.IntegerLiteral)
                    and isinstance(bounds.end, ast.IntegerLiteral)
                    and bounds.step is None
                ):
                    item["iterations"] = bounds.end.value - bounds.start.value + 1
            elif isinstance(node, ast.QuantumGate) and not node.modifiers:
                item.update(
                    name=node.name.name,
                    qubits=[operand(q) for q in node.qubits],
                    parameters=[openqasm3.dumps(value).strip() for value in node.arguments],
                )
            elif isinstance(node, ast.QuantumReset):
                item.update(name="reset", qubits=[operand(node.qubits)])
            elif isinstance(node, ast.QuantumMeasurementStatement) and node.target is not None:
                item.update(
                    name="measure",
                    qubits=[operand(node.measure.qubit)],
                    classical_target=openqasm3.dumps(node.target).strip(),
                )
            elif isinstance(node, ast.ClassicalAssignment):
                item.update(name="store", expression=openqasm3.dumps(node).strip())
            else:
                msg = f"Unsupported source diagram statement: {type(node).__name__}"
                raise ValueError(msg)
            result.append(item)
        return result

    captured = operations(openqasm3.parse(source).statements)
    return {
        "kind": "logical",
        "order": "Parsed OpenQASM input; nested loop bodies occur once and classical expressions are retained.",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "qubits": qubits,
        "clbits": clbits,
        "classical_initializers": initializers,
        "operations": captured,
    }


def capture_circuit(circuit: QuantumCircuit, sites: list[int] | None = None) -> dict[str, Any]:
    """Retain exported gate operands and nested control flow for an SVG view.

    Returns:
        Actual Qiskit operation metadata; loop bodies are never expanded here.
    """
    from qiskit.circuit import ClassicalRegister, Clbit  # ruff: ignore[import-outside-top-level]
    from qiskit.circuit.classical import expr  # ruff: ignore[import-outside-top-level]

    def operations(
        block: QuantumCircuit,
        wires: list[int],
        bits: list[int],
        prefix: str = "",
        aliases: dict[Any, list[int]] | None = None,
    ) -> list[dict[str, Any]]:
        aliases = dict(aliases or {})

        def expression_bits(expression: expr.Expr | tuple[Clbit | ClassicalRegister, int]) -> list[int]:
            variables = (
                [expression[0]] if isinstance(expression, tuple) else [v.var for v in expr.iter_vars(expression)]
            )
            indices = []
            for variable in variables:
                if isinstance(variable, Clbit):
                    indices.append(bits[block.find_bit(variable).index])
                elif isinstance(variable, ClassicalRegister):
                    indices.extend(bits[block.find_bit(bit).index] for bit in variable)
                else:
                    indices.extend(aliases.get(variable, []))
            return sorted(set(indices))

        result = []
        for index, instruction in enumerate(block.data):
            operation = instruction.operation
            operands = [wires[block.find_bit(qubit).index] for qubit in instruction.qubits]
            classical = [bits[block.find_bit(bit).index] for bit in instruction.clbits]
            item = {
                "id": f"{prefix}{index}",
                "name": operation.name,
                "qubits": [wire for wire in operands if wire >= 0],
                "clbits": classical,
                "parameters": [] if hasattr(operation, "blocks") else [str(value) for value in operation.params],
            }
            base = getattr(operation, "base_gate", None)
            if base is not None:
                item["base_gate"] = base.name
                item["controls"] = operation.num_ctrl_qubits
                item["control_state"] = operation.ctrl_state
            if operation.name == "store":
                reads = expression_bits(operation.rvalue)
                item["classical_reads"] = reads
                item["classical_writes"] = expression_bits(operation.lvalue)
                if isinstance(operation.lvalue, expr.Var):
                    aliases[operation.lvalue.var] = reads
            if hasattr(operation, "blocks"):
                item["blocks"] = [
                    operations(child, operands, classical, f"{prefix}{index}.{branch}.", aliases)
                    for branch, child in enumerate(operation.blocks)
                ]
                if operation.name == "for_loop":
                    item["iterations"] = len(operation.params[0])
                condition = getattr(operation, "condition", None)
                if condition is not None:
                    item["condition"] = str(condition)
                    item["condition_bits"] = expression_bits(condition)
            result.append(item)
        return result

    def used_wires(block: QuantumCircuit, wires: list[int]) -> set[int]:
        active: set[int] = set()
        for instruction in block.data:
            operands = [wires[block.find_bit(qubit).index] for qubit in instruction.qubits]
            if hasattr(instruction.operation, "blocks"):
                for child in instruction.operation.blocks:
                    active.update(used_wires(child, operands))
            else:
                active.update(operands)
        return active

    active = sorted(used_wires(circuit, list(range(circuit.num_qubits))))
    wires = [active.index(index) if index in active else -1 for index in range(circuit.num_qubits)]

    def label(index: int, *, classical: bool = False) -> str:
        location = circuit.find_bit((circuit.clbits if classical else circuit.qubits)[index])
        if location.registers:
            register, offset = location.registers[0]
            return register.name if len(register) == 1 else f"{register.name}[{offset}]"
        return f"{'c' if classical else 'q'}[{index}]"

    return {
        "kind": "physical" if sites is not None else "logical",
        "order": "Static program structure; branches and loops are not an execution trace.",
        "qubits": [
            {
                "id": identifier,
                "label": f"${sites[index]}" if sites is not None else label(index),
                "site": sites[index] if sites is not None else None,
            }
            for identifier, index in enumerate(active)
        ],
        "clbits": [{"id": index, "label": label(index, classical=True)} for index in range(circuit.num_clbits)],
        "operations": operations(circuit, wires, list(range(circuit.num_clbits))),
    }


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


def extract_pass_ir(dumps: str, pass_id: str) -> str:
    """Extract one exact module from mqt-cc's pass-manager output.

    Returns:
        The requested module text without pass-manager delimiters.

    Raises:
        ValueError: If the requested pass did not produce exactly one snapshot.
    """
    headers = list(re.finditer(r"^//[^\n]*IR Dump After [^\n]+\n", dumps, re.MULTILINE))
    matches = []
    for index, header in enumerate(headers):
        if re.search(r": " + re.escape(pass_id) + r"(?:\{| )", header.group()) or (
            pass_id.endswith("Pass") and pass_id in header.group()
        ):
            end = headers[index + 1].start() if index + 1 < len(headers) else len(dumps)
            matches.append(dumps[header.end() : end].strip() + "\n")
    if len(matches) != 1 or not matches[0].startswith("module"):
        message = f"missing or ambiguous compiler snapshot for {pass_id}"
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


def normalize_routing_roots(trace: list[dict[str, Any]], initial: list[int]) -> list[int]:
    """Translate compiler-root trace indices into the layout's source-wire order.

    Returns:
        The recorded permutation from compiler roots to source wires.
    """
    root_to_source = [initial.index(site) for site in trace[-1]["before"]]
    assert sorted(root_to_source) == list(range(len(initial)))

    def layout(values: list[int]) -> list[int]:
        result = [0] * len(values)
        for root, source in enumerate(root_to_source):
            result[source] = values[root]
        return result

    for event in trace:
        for key in ("before", "after"):
            if key in event:
                event[key] = layout(event[key])
        for step in event.get("steps", []):
            step["logical"] = [root_to_source[root] for root in step["logical"]]
        search = event.get("search", {})
        if "front" in search:
            search["front"] = [root_to_source[root] for root in search["front"]]
        for node in search.get("nodes", []):
            node["layout"] = layout(node["layout"])
    return root_to_source


def program_stage(
    identifier: str, label: str, program: QCProgram | QCOProgram, target: CompilerTarget | None = None
) -> dict[str, Any]:
    """Pair exact compiler text with the circuit exported from the same program.

    Returns:
        A stage including native operation operands for circuit/topology highlighting.
    """
    artifact = stage(identifier, label, "mlir", program.ir)
    artifact["circuit"] = capture_circuit(
        program.to_qiskit(target=target, copy=True), [site.id for site in target.sites] if target else None
    )
    if identifier in {"qc", "qco"}:
        display = program.copy()
        display.cleanup()
        artifact["display_code"] = display.ir
        artifact["display_sha256"] = hashlib.sha256(display.ir.encode()).hexdigest()
        artifact["display_passes"] = ["cleanup (canonicalization and common subexpression elimination)"]
    return artifact


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


def compile_target(
    program: QCOProgram,
    compiler: Path,
    model: dict[str, Any],
    timeout: int,
    *,
    mapping: dict[str, int] | None = None,
    trace: bool = False,
    target_attribute: str | None = None,
) -> tuple[str, str]:
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
    if trace:
        environment["MQT_MQSF_MAPPING_TRACE"] = "1"
    else:
        environment.pop("MQT_MQSF_MAPPING_TRACE", None)
    mapping = mapping or {"trials": 1, "iterations": 0, "lookahead": 8, "search-memory-limit": 4194304}
    with TemporaryDirectory(prefix="mqsf-compiler-") as directory:
        synthesis_trace = Path(directory) / "synthesis.jsonl"
        if trace:
            environment["MQT_MQSF_SYNTHESIS_TRACE"] = str(synthesis_trace)
        else:
            environment.pop("MQT_MQSF_SYNTHESIS_TRACE", None)
        source = Path(directory) / "input.mlir"
        source.write_text(program.ir, encoding="utf-8")
        command = [
            str(compiler),
            str(source),
            f"--target={target_attribute}" if target_attribute else "--qdmi-device=mqt.sc.default",
            f"--payload-spec={payload_specification()}",
            "--emit=qco-optimized",
            "--seed=7",
            *(f"--mapping-{name}={value}" for name, value in mapping.items()),
            "--mlir-disable-threading",
            "--mlir-print-ir-module-scope",
            "--mlir-print-ir-after-all"
            if trace
            else "--mlir-print-ir-after=" + ",".join(name for name, _ in TARGET_PASSES),
        ]
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            command, capture_output=True, text=True, check=True, env=environment, timeout=timeout
        )
        synthesis = (
            "".join(f"MQSF_SYNTHESIS_TRACE {line}\n" for line in synthesis_trace.read_text().splitlines())
            if synthesis_trace.exists()
            else ""
        )
    return result.stdout, result.stderr + synthesis


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
    mapping: dict[str, int] | None = None,
    trace: bool = False,
    target_attribute: str | None = None,
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
        "stages": [program_stage("qc", "Import into QC", qc)],
        "exports": [],
        "layout": {"initial": [], "final": [], "swaps": []},
    }
    qco = qc.to_qco(copy=True)
    variant["stages"].append(program_stage("qco", "Value-based QCO", qco))
    qco.run_pass_pipeline("inline")
    qco.cleanup()
    variant["stages"].append(program_stage("optimized", "Inline and simplify", qco))
    if unroll:
        qco.unroll_quantum_loops()
        qco.cleanup()
        variant["stages"].append(program_stage("unrolled", "Unroll bounded quantum loops", qco))
    native_ir, dumps = compile_target(
        qco, compiler, model, timeout, mapping=mapping, trace=trace, target_attribute=target_attribute
    )
    if trace:
        variant["routing_trace"] = [
            json.loads(line.removeprefix("MQSF_MAPPING_TRACE "))
            for line in dumps.splitlines()
            if line.startswith("MQSF_MAPPING_TRACE ")
        ]
        variant["synthesis_trace"] = [
            json.loads(line.removeprefix("MQSF_SYNTHESIS_TRACE "))
            for line in dumps.splitlines()
            if line.startswith("MQSF_SYNTHESIS_TRACE ")
        ]
        for event in variant["synthesis_trace"]:
            for when in ("before", "after"):
                program = QCOProgram.from_mlir_str(event[f"{when}_ir"])
                # Before placement the circuit is logical; mapped modules carry a layout.
                environment = target if "mqt.layout" in event[f"{when}_ir"] else None
                event[when] = program_stage(when, f"Fusion {when}", program, environment)
                indices = event["input_indices" if when == "before" else "output_indices"]
                assert all(index < len(event[when]["circuit"]["operations"]) for index in indices)
                del event[f"{when}_ir"]
    dumps = re.sub(r"^MQSF_(MAPPING|SYNTHESIS)_TRACE .*\n", "", dumps, flags=re.MULTILINE)
    if trace:
        fused = QCOProgram.from_mlir_str(extract_pass_ir(dumps, "FuseTwoQubitGatesPass"))
        variant["stages"].append(stage("two-qubit-fusion", "Fuse neighboring two-qubit operations", "mlir", fused.ir))
    for name, label in TARGET_PASSES:
        snapshot = QCOProgram.from_mlir_str(extract_pass_ir(dumps, name))
        artifact = (
            program_stage(name, label, snapshot, target)
            if name in {"place-and-route", "target-native-synthesis"}
            else stage(name, label, "mlir", snapshot.ir)
        )
        variant["stages"].append(artifact)
    native = QCOProgram.from_mlir_str(native_ir)
    routed = QCOProgram.from_mlir_str(extract_pass_ir(dumps, "place-and-route"))
    routed_qasm = routed.to_qc(copy=True).to_openqasm3().source
    variant["layout"] = extract_layout(native_ir, routed_qasm)
    variant["layout"]["routed_openqasm3"] = routed_qasm
    if trace:
        variant["routing_root_to_source"] = normalize_routing_roots(
            variant["routing_trace"], variant["layout"]["metadata"]["initial"]
        )
    edges = {tuple(sorted(edge)) for edge in model["couplings"]}
    if any(tuple(sorted(swap)) not in edges for swap in variant["layout"]["swaps"]):
        message = "captured routing SWAP uses an edge absent from the Emerald model"
        raise ValueError(message)
    if len(variant["layout"]["swaps"]) != len(re.findall(r"\bqco\.swap\b", routed.ir)):
        message = "routed OpenQASM and QCO disagree on the number of captured SWAPs"
        raise ValueError(message)
    variant["target_conformance"] = {
        "verified": True,
        "scope": model["name"] + "; local QDMI SC model",
        "measurement_feedback_retained": "qco.if" in native_ir or "scf.while" in native_ir,
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


def capture_portable_variant(qc: QCProgram, *, unroll: bool, source_diagram: bool = False) -> dict[str, Any]:
    """Capture device-independent optimization and its portable exports.

    Returns:
        Exact QC, QCO, optimized QC, and export artifacts without target synthesis.

    Raises:
        RuntimeError: If a required Qiskit diagram cannot be exported.
    """
    from mqt.core.mlir import QIRProfile  # ruff: ignore[import-outside-top-level]

    def captured_stage(identifier: str, label: str, program: QCProgram | QCOProgram) -> dict[str, Any]:
        return (
            stage(identifier, label, "mlir", program.ir)
            if source_diagram
            else program_stage(identifier, label, program)
        )

    stages = [captured_stage("qc", "Import into QC", qc)]
    qco = qc.to_qco(copy=True)
    stages.append(captured_stage("qco", "Value-based QCO", qco))
    qco.run_pass_pipeline("mqt-qco-default")
    stages.append(captured_stage("optimized", "Device-independent optimization", qco))
    if unroll:
        qco.unroll_quantum_loops()
        qco.cleanup()
        stages.append(captured_stage("unrolled", "Unroll bounded quantum loops", qco))
    optimized_qc = qco.to_qc(copy=True)
    stages.append(captured_stage("qc-optimized", "Return to QC", optimized_qc))
    try:
        qiskit = stage("qiskit", "Qiskit circuit", "text", render_qiskit(optimized_qc.to_qiskit(copy=True)))
    except RuntimeError as error:
        if not source_diagram:
            raise
        qiskit = {"id": "qiskit", "label": "Qiskit circuit", "unavailable_reason": str(error)}
    return {
        "id": "unrolled" if unroll else "structured",
        "label": "Bounded loops unrolled" if unroll else "Preserve supported structure",
        "stages": stages,
        "exports": [
            stage("qir-adaptive", "QIR Adaptive", "llvm", optimized_qc.to_qir(QIRProfile.ADAPTIVE, copy=True).llvm_ir),
            stage("openqasm3", "OpenQASM 3", "qasm", optimized_qc.to_openqasm3().source),
            qiskit,
        ],
        "layout": {"initial": [], "final": [], "swaps": []},
    }


def capture(compiler: Path, selected: str) -> dict[str, Any]:
    """Generate the selected benchmark cases from this checkout's native compiler.

    Returns:
        Presentation fixture data containing only captured compiler outputs.
    """
    from mqt.core.mlir import QCProgram  # ruff: ignore[import-outside-top-level]

    model = json.loads((ROOT / "json/sc/iqm-emerald.json").read_text(encoding="utf-8"))
    site_names = {site["qubit"]: site["name"] for site in model["qubitProperties"]["overrides"]}
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
            "patches": [],
            "model_path": "json/sc/iqm-emerald.json",
            "model_sha256": hashlib.sha256((ROOT / "json/sc/iqm-emerald.json").read_bytes()).hexdigest(),
            "compiler_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
            "capture_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scenario_pipeline": ["OpenQASM", "QC", "QCO", "mqt-qco-default", "QC", "QIR Adaptive"],
        },
        "device": {
            "name": "IQM Emerald",
            "label": "Emerald reference topology for the device walkthrough",
            "native_gates": ["r", "cz", "measure"],
            "sites": [{"id": site, "name": site_names.get(site, str(site))} for site in range(model["numQubits"])],
            "edges": model["couplings"],
        },
        "scenarios": [],
    }
    cases = (
        (
            "parity",
            "Measure parity. Correct. Repeat.",
            "Three qubits, two rounds, one measured decision. Unrolling the loop keeps measurement feedback.",
            {"qubits": 3, "rounds": 2},
            PARITY_SOURCE,
        ),
        (
            "qpe",
            "Iterative phase estimation",
            "Eight phase bits from a four-qubit U and one recycled control. Phase 1/3 lies between output bins.",
            {"qubits": 5, "unitary_qubits": 4, "precision": 8, "phase_numerator": 1, "phase_denominator": 3},
            qpe_source(),
        ),
        (
            "rus",
            "Repeat until a non-Clifford rotation succeeds",
            "Paetznick-Svore V3 = (I + 2iZ)/sqrt(5); correct failed attempts and preserve the data qubit.",
            {"qubits": 3, "success_probability_per_attempt": 0.625, "result_one_probability": 0.8},
            RUS_SOURCE,
        ),
    )
    for identifier, label, summary, parameters, source in cases:
        if selected not in {"all", identifier}:
            continue
        qc = QCProgram.from_openqasm_str(source)
        scenario: dict[str, Any] = {
            "id": identifier,
            "label": label,
            "summary": summary,
            "parameters": parameters,
            "variants": [],
        }
        for unroll in (False, True) if identifier == "parity" else (False,):
            sys.stderr.write(f"Capturing {identifier}: {'unrolled' if unroll else 'structured'}\n")
            variant = capture_portable_variant(qc, unroll=unroll, source_diagram=identifier == "qpe")
            source_stage = stage("source", "OpenQASM input", "qasm", source)
            source_stage["circuit"] = (
                capture_openqasm_circuit(source) if identifier == "qpe" else capture_circuit(qc.to_qiskit(copy=True))
            )
            variant["stages"].insert(0, source_stage)
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
    parser.add_argument("--scenario", choices=("all", "parity", "qpe", "rus"), default="all")
    arguments = parser.parse_args()
    data = capture(arguments.compiler.resolve(), arguments.scenario)
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

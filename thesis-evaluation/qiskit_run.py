import numpy as np
import sys
import re
import os
import glob
import time
from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit import qasm3
from qiskit import qasm2
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.circuit.library.standard_gates.swap import SwapGate
from qiskit.synthesis import (
    TwoQubitBasisDecomposer,
    TwoQubitWeylDecomposition,
    OneQubitEulerDecomposer,
)
from qiskit.quantum_info import Operator

np.set_printoptions(linewidth=np.inf)

MQT_BENCH_DIR = "../../mqt-bench/generated_benchmarks/v3_qasm3"
MQT_BENCH_PATTERN = "*.qasm"
is_qasm3 = True


def split_by_barriers(qc: QuantumCircuit) -> list[QuantumCircuit]:
    subcircuits = []

    # collect subcircuit instructions
    current = []
    num_qubits = 0
    for instr in qc.data:
        if instr.name == "barrier" or instr.name == "measure":
            # Finish current subcircuit if it has content
            if len(current) > 0:
                circuit = QuantumCircuit(num_qubits)
                for x in current:
                    remapped_qubits = []
                    for y in x.qubits:
                        if num_qubits > 1:
                            prev_qubit_pos = qc.qubits.index(y)
                        else:
                            prev_qubit_pos = 0
                        remapped_qubits.append(circuit.qubits[prev_qubit_pos])
                    circuit.append(x.operation, remapped_qubits, x.clbits)
                subcircuits.append(circuit)
            # Start a fresh one
            current = []
            num_qubits = 0
        else:
            current.append(instr)
            num_qubits = max(num_qubits, len(instr.qubits))

    # Append the last chunk if non-empty
    if len(current) > 0:
        circuit = QuantumCircuit(num_qubits)
        for x in current:
            remapped_qubits = []
            for y in x.qubits:
                if num_qubits > 1:
                    prev_qubit_pos = qc.qubits.index(y)
                else:
                    prev_qubit_pos = 0
                remapped_qubits.append(circuit.qubits[prev_qubit_pos])
            circuit.append(x.operation, remapped_qubits, x.clbits)
        subcircuits.append(circuit)

    return subcircuits


def circuit_length(qc: QuantumCircuit) -> int:
    return qc.size(lambda instr: instr.name != "barrier" and instr.name != "measure")


def contains_foreign_gates(qc: QuantumCircuit) -> bool:
    return qc.size(lambda instr: instr.name not in ["rz", "rx", "ry", "cx"]) > 0


def circuit_complexity(qc: QuantumCircuit) -> int:
    num_one_qubit_gates = qc.size(
        lambda instr: len(instr.qubits) == 1 and instr.name != "barrier"
    )
    num_two_qubit_gates = qc.size(
        lambda instr: len(instr.qubits) == 2 and instr.name != "barrier"
    )
    num_other_gates = qc.size(
        lambda instr: len(instr.qubits) != 1 and len(instr.qubits) != 2
    )

    return num_one_qubit_gates + 10 * num_two_qubit_gates


def read_rust_timings_file(
    file_name="/tmp/qiskit_rust.timing", remove_file=False
) -> dict[str, list[int]]:
    patterns = {
        "timePerTwoQubitDecomposition": re.compile(
            r"TwoQubitBasisDecomposer::generate_sequence\(\):\s+(\d+)us"
        ),
        "twoQubitCreationTime": re.compile(
            r"TwoQubitBasisDecomposer::new_inner\(\):\s+(\d+)us"
        ),
        "timePerSingleQubitDecomposition": re.compile(
            r"unitary_to_gate_sequence_inner\(\):\s+(\d+)ns"
        ),
    }

    timings: dict[str, list[int]] = {}
    with open(file_name) as f:
        for line in f.readlines():
            print(line)
            for metric, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    timings[metric] = timings.get(metric, []) + [int(match.group(1))]
    if remove_file:
        os.remove(file_name)
    return timings


def evaluate(evaluate_rust_timings=False):
    stats = {}

    otherCX = QuantumCircuit(2)
    otherCX.cx(1, 0)
    otherCXGate = otherCX.to_gate()
    oneQubitDecs = [
        OneQubitEulerDecomposer("ZYZ"),
        OneQubitEulerDecomposer("ZXZ"),
        OneQubitEulerDecomposer("XYX"),
        OneQubitEulerDecomposer("XZX"),
    ]

    print(f"Processing '{MQT_BENCH_DIR}/{MQT_BENCH_PATTERN}'...")
    for file in glob.glob(f"{MQT_BENCH_DIR}/{MQT_BENCH_PATTERN}"):
        name = os.path.basename(file).removesuffix(".qasm")
        print(f"Decomposing {name} ({file})")

        start_time = time.perf_counter_ns()
        dec = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZYZ")
        dec2 = TwoQubitBasisDecomposer(otherCXGate, euler_basis="ZYZ")
        end_time = time.perf_counter_ns()
        creation_time_us = (end_time - start_time) / 1000
        print(f"Python Basis Decomposition Creation: {creation_time_us}µs")
        if evaluate_rust_timings:
            rust_stats = read_rust_timings_file(remove_file=True)
            creation_time_us = np.sum(rust_stats["twoQubitCreationTime"])
        twoQubitDecs = [dec, dec2]

        with open(file, "r") as f:
            content = f.read()
        try:
            if is_qasm3:
                content = "\n".join(
                    filter(lambda line: not "measure " in line, content.splitlines())
                )
                qc: QuantumCircuit = qasm3.loads(content)
            else:
                qc: QuantumCircuit = qasm2.loads(content)
        except Exception as e:
            print(f"FAILED: {name} ({e})")
            continue

        # respect barriers
        start_time = time.perf_counter_ns()
        subcircuits = split_by_barriers(qc)
        end_time = time.perf_counter_ns()
        collection_time_us = (end_time - start_time) / 1000

        decomposition_times_1q = []
        decomposition_times_2q = []
        complexity_changes = []
        num_two_qubit_decompositions = 0
        num_single_qubit_decompositions = 0
        for subcircuit in subcircuits:
            if circuit_length(subcircuit) < 3 and not contains_foreign_gates(
                subcircuit
            ):
                print(f"SKIPPED (length: {circuit_length(subcircuit)})")
                continue
            m = Operator(subcircuit)
            decomposed_circuits = []
            if len(subcircuit.qubits) == 1:
                start_time = time.perf_counter_ns()
                for dec in oneQubitDecs:
                    decomposed_circuits.append(dec(m))
                end_time = time.perf_counter_ns()
                decomposition_time = (end_time - start_time) / 1000

                if evaluate_rust_timings:
                    rust_stats = read_rust_timings_file(remove_file=True)
                    decomposition_times_1q.append(
                        # measured in nanoseconds, convert to microseconds
                        np.sum(rust_stats["timePerSingleQubitDecomposition"]) / 1000
                    )
                else:
                    decomposition_times_1q.append(decomposition_time)

                num_single_qubit_decompositions += 1
            elif len(subcircuit.qubits) == 2:
                start_time = time.perf_counter_ns()
                for dec in twoQubitDecs:
                    decomposed_circuits.append(dec(m))
                end_time = time.perf_counter_ns()
                decomposition_time = (end_time - start_time) / 1000

                if evaluate_rust_timings:
                    rust_stats = read_rust_timings_file(remove_file=True)
                    decomposition_times_2q.append(
                        np.sum(rust_stats["timePerTwoQubitDecomposition"][1:])
                    )
                else:
                    decomposition_times_2q.append(decomposition_time)

                num_two_qubit_decompositions += 1
            else:
                raise RuntimeError("Invalid circuit size!")

            before_complexity = circuit_complexity(subcircuit)
            after_complexities = [circuit_complexity(x) for x in decomposed_circuits]
            best_decomposed_circuit = decomposed_circuits[
                after_complexities.index(min(after_complexities))
            ]
            best_after_complexity = min(after_complexities)

            print(subcircuit)
            print(f"{before_complexity} -> {best_after_complexity}")
            print(best_decomposed_circuit)

            print(Operator(subcircuit).data)
            print("vs")
            print(Operator(best_decomposed_circuit).data)

            complexity_changes.append(before_complexity - best_after_complexity)

        stats[name] = {
            "timeInSingleQubitDecomposition": sum(decomposition_times_1q),
            "timeInTwoQubitDecomposition": sum(decomposition_times_2q),
            "subCircuitComplexityChange": sum(complexity_changes),
            "totalTwoQubitDecompositions": num_two_qubit_decompositions,
            "totalSingleQubitDecompositions": num_single_qubit_decompositions,
            "twoQubitCreationTime": creation_time_us,
            "timeInCircuitCollection": collection_time_us,
        }

    print()
    print(stats)
    print(f"Total benchmarks: {len(stats)}")
    return stats


if __name__ == "__main__":
    evaluate_rust_timings = sys.argv[-1] == "--rust-timings"
    print(f"Rust evaluation: {evaluate_rust_timings} ({sys.argv[-1]})")
    evaluate(evaluate_rust_timings)

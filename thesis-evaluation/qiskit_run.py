import numpy as np
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

    # Start with an empty circuit that has the same registers
    current = QuantumCircuit(*qc.qregs, *qc.cregs)

    for instr in qc.data:
        if instr.name == "barrier" or instr.name == "measure":
            # Finish current subcircuit if it has content
            if current.data:
                subcircuits.append(current)
            # Start a fresh one
            current = QuantumCircuit(*qc.qregs, *qc.cregs)
        else:
            current.append(instr.operation, instr.qubits, instr.clbits)

    # Append the last chunk if non-empty
    if current.data:
        subcircuits.append(current)

    return subcircuits


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


def evaluate():
    stats = {}

    otherCX = QuantumCircuit(2)
    otherCX.cx(1, 0)
    otherCXGate = otherCX.to_gate()
    oneQubitDec = OneQubitEulerDecomposer("ZYZ")
    start_time = time.perf_counter_ns()
    dec = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZYZ")
    dec2 = TwoQubitBasisDecomposer(otherCXGate, euler_basis="ZYZ")
    end_time = time.perf_counter_ns()
    creation_time_us = (end_time - start_time) / 1000
    print(f"Python Basis Decomposition Creation: {creation_time_us}µs")

    print(f"Processing '{MQT_BENCH_DIR}/{MQT_BENCH_PATTERN}'...")
    for file in glob.glob(f"{MQT_BENCH_DIR}/{MQT_BENCH_PATTERN}"):
        name = os.path.basename(file).removesuffix(".qasm")
        print(f"Decomposing {name} ({file})")
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
        subcircuits = split_by_barriers(qc)

        decomposition_times_1q = []
        decomposition_times_2q = []
        complexity_changes = []
        num_two_qubit_decompositions = 0
        num_single_qubit_decompositions = 0
        for subcircuit in subcircuits:
            m = Operator(subcircuit)
            if len(qc.qubits) == 1:
                start_time = time.perf_counter_ns()
                decomposed_circuit = oneQubitDec(m)
                decomposed_circuit2 = None
                end_time = time.perf_counter_ns()
                decomposition_time = (end_time - start_time) / 1000
                decomposition_times_1q.append(decomposition_time)

                num_single_qubit_decompositions += 1
            elif len(qc.qubits) == 2:
                start_time = time.perf_counter_ns()
                decomposed_circuit = dec(m)
                decomposed_circuit2 = dec2(m)
                end_time = time.perf_counter_ns()
                decomposition_time = (end_time - start_time) / 1000
                decomposition_times_2q.append(decomposition_time)

                num_two_qubit_decompositions += 1
            else:
                raise RuntimeError("Invalid circuit size!")

            before_complexity = circuit_complexity(subcircuit)
            after_complexity = circuit_complexity(decomposed_circuit)

            if decomposed_circuit2:
                after_complexity2 = circuit_complexity(decomposed_circuit2)
                if after_complexity2 < after_complexity:
                    print(f"Choose alternative decomposition ({after_complexity2} vs {after_complexity})!")
                    decomposed_circuit = decomposed_circuit2
                    after_complexity = after_complexity2

            print(subcircuit)
            print(f"{before_complexity} -> {after_complexity}")
            print(decomposed_circuit)

            print(Operator(subcircuit).data)
            print("vs")
            print(Operator(decomposed_circuit).data)

            complexity_changes.append(before_complexity - after_complexity)

        stats[name] = {
            "timeInSingleQubitDecomposition": sum(decomposition_times_1q),
            "timeInTwoQubitDecomposition": sum(decomposition_times_2q),
            "subCircuitComplexityChange": sum(complexity_changes),
            "totalTwoQubitDecompositions": num_two_qubit_decompositions,
            "totalSingleQubitDecompositions": num_single_qubit_decompositions,
        }

    print()
    print(stats)
    print(f"Total benchmarks: {len(stats)}")
    return stats

if __name__ == "__main__":
    evaluate()

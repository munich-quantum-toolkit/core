import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane.tape import QuantumTape
from mqt.core import QuantumComputation
from catalyst import measure, pipeline


# ================================================
# CONFIGURATION
# ================================================

NUM_REPS = 10
MAX_QUBITS = 50
QUBIT_RANGE = range(3, MAX_QUBITS + 1, 10)
CSV_FILENAME = f"{MAX_QUBITS}_ghz_conversion_timings_avg.csv"

CATALYST_PIPELINE = {
    "mqt.quantum-to-mqtopt": {},
    "mqt.mqt-core-round-trip": {},
    "mqt.mqtopt-to-quantum": {},
}

# ================================================
# CIRCUIT DEFINITIONS
# ================================================

def ghz_circuit(num_qubits):
    qml.Hadamard(wires=0)
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[0, i + 1])

def ghz_tape_execution(num_qubits):
    with QuantumTape() as tape:
        ghz_circuit(num_qubits)
    qasm = tape.to_openqasm()
    qc = QuantumComputation.from_qasm_str(qasm)
    qasm_str = QuantumComputation.qasm2_str(qc)
    qml.from_qasm(qasm_str)


custom_pipeline = [("run_only_plugin", ["builtin.module(apply-transform-sequence)"])]
def compile_with_catalyst(num_qubits: int):

    @qml.qjit(target="mlir", pipelines=custom_pipeline)
    def foo():
        dev = qml.device("lightning.qubit", wires=num_qubits)

        @pipeline(CATALYST_PIPELINE)
        @qml.qnode(dev)
        def workflow():
            qml.Hadamard(wires=[0])
            for i in range(num_qubits-1):
                qml.CNOT(wires=[i, i + 1])
            return [measure(wires=i) for i in range(num_qubits)]

        return workflow()
    
    _ = foo.mlir
    return

# ================================================
# BENCHMARKS
# ================================================

def benchmark_with_timeit(fn, num_qubits):
    return timeit.timeit(lambda: fn(num_qubits), number=NUM_REPS) / NUM_REPS

def run_benchmarks():
    qasm_timings = []
    catalyst_timings = []

    for n in QUBIT_RANGE:
        print(f"Benchmarking {n} qubits...")

        qasm_time = benchmark_with_timeit(ghz_tape_execution, n)
        catalyst_time = benchmark_with_timeit(compile_with_catalyst, n)

        qasm_timings.append((n, qasm_time))
        catalyst_timings.append((n, catalyst_time))

    return qasm_timings, catalyst_timings

# ================================================
# PLOTTING
# ================================================

def plot_results(qasm_data, catalyst_data):
    qasm_x, qasm_y = zip(*qasm_data)
    cat_x, cat_y = zip(*catalyst_data)

    plt.figure(figsize=(10, 6))
    plt.plot(qasm_x, qasm_y, label="QASM Round-Trip", marker="o")
    plt.plot(cat_x, cat_y, label="Plugin Round-Trip", marker="s")

    plt.xlabel("Number of Qubits")
    plt.ylabel("Average Time per Run (seconds)")
    plt.title("GHZ Circuit Benchmark")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ghz_benchmark_plot.png")
    plt.show()

# ================================================
# MAIN
# ================================================

if __name__ == "__main__":
    qasm_results, catalyst_results = run_benchmarks()

    # Optionally save results to CSV
    df = pd.DataFrame({
        "qubits": [n for n, _ in qasm_results],
        "qasm_time": [t for _, t in qasm_results],
        "catalyst_time": [t for _, t in catalyst_results],
    })
    df.to_csv(CSV_FILENAME, index=False)
    print(f"Saved benchmark results to {CSV_FILENAME}")

    plot_results(qasm_results, catalyst_results)
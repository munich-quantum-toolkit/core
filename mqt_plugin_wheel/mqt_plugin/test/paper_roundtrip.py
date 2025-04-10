# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane.tape import QuantumTape
from mqt.core import QuantumComputation
from catalyst import measure, pipeline
import catalyst
from mqt_plugin import getMQTPluginAbsolutePath
import jax

# ================================================
# CONFIGURATION
# ================================================

NUM_REPS = 10
MAX_QUBITS = 50
QUBIT_RANGE = range(3, MAX_QUBITS, 250)
CSV_FILENAME = f"{MAX_QUBITS}_ghz_conversion_timings_avg.csv"

MQT_PIPELINE = {
    "mqt.quantum-to-mqtopt": {},
    # "mqt.mqt-core-round-trip": {},
    "mqt.mqtopt-to-quantum": {},
}

MQT_PLUGIN = getMQTPluginAbsolutePath()

# ================================================
# CIRCUIT DEFINITIONS
# ================================================


def pennylane_roundtrip(num_qubits: int):
    # NOTE: this IS the usual way to use PennyLane

    @qml.qnode(qml.device("lightning.qubit", wires=num_qubits))
    def ghz_circuit(num_qubits):
        qml.Hadamard(wires=0)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[0, i + 1])
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]

    ghz_circuit(num_qubits)
    tape = ghz_circuit._tape
    qasm = tape.to_openqasm()
    qc = QuantumComputation.from_qasm_str(qasm)
    qasm_str = QuantumComputation.qasm2_str(qc)
    circ = qml.from_qasm(qasm_str)
    return circ


def qasm_roundtrip(num_qubits: int):
    # NOTE: this is NOT the usual way to use PennyLane

    def ghz_circuit(num_qubits):
        qml.Hadamard(wires=0)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[0, i + 1])

    # NOTE: automatically adds measurements
    with QuantumTape() as tape:
        ghz_circuit(num_qubits)
    qasm = tape.to_openqasm()

    qc = QuantumComputation.from_qasm_str(qasm)
    qasm_str = QuantumComputation.qasm2_str(qc)

    circ_fn = qml.from_qasm(qasm_str)
    with QuantumTape() as tape:
        circ_fn()

    return tape


def plugin_roundtrip(num_qubits: int):
    # custom_pipeline = [("run_only_plugin", ["detensorize-scf",f"test-loop-unrolling{{unroll-factor={num_qubits-1}}}","builtin.module(apply-transform-sequence)"])]
    custom_pipeline = [
        (
            "before_plugin",
            ["detensorize-scf", f"test-loop-unrolling{{unroll-factor={num_qubits - 1}}}", "canonicalize"],
        ),
        ("plugin", ["builtin.module(apply-transform-sequence)"]),
    ]

    @qml.qjit(
        target="mlir",
        pipelines=custom_pipeline,
        pass_plugins={MQT_PLUGIN},
        dialect_plugins={MQT_PLUGIN},
        autograph=True,
        keep_intermediate=True,
    )
    def foo():
        @pipeline(MQT_PIPELINE)
        @qml.qnode(qml.device("lightning.qubit", wires=num_qubits))
        def circuit():
            qml.Hadamard(wires=[0])
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            meas = jax.numpy.empty((num_qubits,), dtype=int)
            for i in range(0, num_qubits):
                meas = meas.at[i].set(catalyst.measure(i))
            return meas

            # return [measure(wires=i) for i in range(num_qubits)]

        return circuit()

    return foo.mlir_opt


# ================================================
# BENCHMARKS
# ================================================


def benchmark_with_timeit(fn, num_qubits):
    return timeit.timeit(lambda: fn(num_qubits), number=NUM_REPS) / NUM_REPS


def debug_benchmark_with_timeit(fn, num_qubits):
    # NOTE: this adds some overhead (e.g. when extracting programsize)
    with catalyst.debug.instrumentation(f"session{num_qubits}", f"session{num_qubits}.yaml", detailed=True):
        time = timeit.timeit(lambda: fn(num_qubits), number=NUM_REPS) / NUM_REPS
    return time


def run_benchmarks():
    qasm_timings = []
    catalyst_timings = []

    for n in QUBIT_RANGE:
        print(f"Benchmarking {n} qubits...")

        qasm_time = benchmark_with_timeit(qasm_roundtrip, n)
        catalyst_time = benchmark_with_timeit(plugin_roundtrip, n)

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

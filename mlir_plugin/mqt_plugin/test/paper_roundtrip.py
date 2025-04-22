# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import timeit

import catalyst
import matplotlib.pyplot as plt
import pandas as pd
import pennylane as qml
from catalyst import measure, pipeline
from mqt_plugin import getMQTPluginAbsolutePath
from pennylane.tape import QuantumTape

from mqt.core import QuantumComputation
from mqt.qmap.compile import compile as qmap_compile
from mqt.qmap.pyqmap import Architecture

# ================================================
# CONFIGURATION
# ================================================

NUM_REPS = 10
MAX_QUBITS = 505
QUBIT_RANGE = range(3, MAX_QUBITS, 250)
CSV_FILENAME = f"{MAX_QUBITS}_ghz_conversion_timings_avg.csv"

MQT_PIPELINE = {
    "mqt.quantum-to-mqtopt": {},
    "mqt.mqt-core-round-trip": {},
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
    return qml.from_qasm(qasm_str)


def qasm_roundtrip(num_qubits: int):
    # NOTE: this is NOT the usual way to use PennyLane

    def ghz_circuit(num_qubits) -> None:
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
        autograph=False,
        keep_intermediate=False,
    )
    def foo():
        @pipeline(MQT_PIPELINE)
        @qml.qnode(qml.device("lightning.qubit", wires=num_qubits))
        def circuit():
            qml.Hadamard(wires=[0])
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # meas = jax.numpy.empty((num_qubits,), dtype=int)
            # for i in range(0, num_qubits):
            #    meas = meas.at[i].set(catalyst.measure(i))
            # return meas

            return [measure(wires=i) for i in range(num_qubits)]

        return circuit()

    return foo.mlir_opt


def qmap_roundtrip(num_qubits: int):
    # Construct GHZ circuit and convert to QASM
    with QuantumTape() as tape:
        qml.Hadamard(wires=0)
        for i in range(1, num_qubits):
            qml.CNOT(wires=[0, i])
    tape.to_openqasm()

    # Linear coupling map: {(0,1), (1,0), ...}
    coupling_map = [(i, i + 1) for i in range(num_qubits - 1)]
    coupling_map += [(i + 1, i) for i in range(num_qubits - 1)]
    cmap = set(coupling_map)
    arch = Architecture(num_qubits, cmap)

    # Benchmark mapping ONLY

    return timeit.timeit(lambda: qmap_compile(qc, arch), number=NUM_REPS) / NUM_REPS


# ================================================
# BENCHMARKS
# ================================================


def benchmark_with_timeit(fn, num_qubits):
    return timeit.timeit(lambda: fn(num_qubits), number=NUM_REPS) / NUM_REPS


def debug_benchmark_with_timeit(fn, num_qubits):
    # NOTE: this adds some overhead (e.g. when extracting programsize)
    with catalyst.debug.instrumentation(f"session{num_qubits}", f"session{num_qubits}.yaml", detailed=True):
        return timeit.timeit(lambda: fn(num_qubits), number=NUM_REPS) / NUM_REPS


def run_benchmarks():
    qasm_timings = []
    catalyst_timings = []
    qmap_timings = []

    for n in QUBIT_RANGE:
        qmap_time = qmap_roundtrip(n)
        qasm_time = benchmark_with_timeit(qasm_roundtrip, n)
        catalyst_time = debug_benchmark_with_timeit(plugin_roundtrip, n)

        qasm_timings.append((n, qasm_time))
        catalyst_timings.append((n, catalyst_time))
        qmap_timings.append((n, qmap_time))

    return qasm_timings, catalyst_timings, qmap_timings


# ================================================
# PLOTTING
# ================================================


def plot_results(qasm_data, catalyst_data, plugin_data, qmap_data) -> None:
    qasm_x, qasm_y = zip(*qasm_data)
    cat_x, cat_y = zip(*catalyst_data)
    plugin_x, plugin_y = zip(*plugin_data)
    qmap_x, qmap_y = zip(*qmap_data)

    plt.figure(figsize=(10, 6))
    plt.plot(qasm_x, qasm_y, label="QASM Round-Trip", marker="o")
    plt.plot(cat_x, cat_y, label="Catalyst Round-Trip", marker="s")
    plt.plot(plugin_x, plugin_y, label="Plugin Only", marker="x")
    plt.plot(qmap_x, qmap_y, label="Mapping Only", marker="^")

    plt.xlabel("Number of Qubits")
    plt.ylabel("Average Time per Run (seconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ghz_extended_benchmark_plot.png")
    plt.show()


# ================================================
# Extract plugin walltimes from YAML files
# ================================================

import os

import yaml


def extract_plugin_walltimes_from_file(filepath):
    """Extract all plugin walltimes from a single YAML file."""
    with open(filepath, encoding="utf-8") as file:
        data = yaml.safe_load(file)

    plugin_walltimes = []

    for session in data.values():
        if not isinstance(session, dict) or "results" not in session:
            continue

        for result in session["results"]:
            if not isinstance(result, dict) or "generate_ir" not in result:
                continue

            finegrained = result["generate_ir"].get("finegrained", [])
            for step in finegrained:
                if "plugin" in step:
                    walltime = step["plugin"].get("walltime", None)
                    if walltime is not None:
                        plugin_walltimes.append(walltime / 1000)  # ms -> seconds

    return sum(plugin_walltimes) / len(plugin_walltimes)


def collect_plugin_walltimes(directory="."):
    """Search all YAML files in the directory and collect plugin walltimes."""
    plugin_walltimes = []
    # Files are named like session{n_qubits}.yaml
    for n in QUBIT_RANGE:
        filepath = os.path.join(directory, f"session{n}.yaml")
        walltimes = extract_plugin_walltimes_from_file(filepath)
        plugin_walltimes.append((n, walltimes))
    return plugin_walltimes


# ================================================
# MAIN
# ================================================

if __name__ == "__main__":
    load_from_file = False

    if not load_from_file:
        qasm_results, catalyst_results, qmap_results = run_benchmarks()
        plugin_only_results = collect_plugin_walltimes()
    else:
        # Load results from CSV
        qasm_results = pd.read_csv(CSV_FILENAME, usecols=["qubits", "qasm_time"]).values.tolist()
        catalyst_results = pd.read_csv(CSV_FILENAME, usecols=["qubits", "catalyst_time"]).values.tolist()
        qmap_results = pd.read_csv(CSV_FILENAME, usecols=["qubits", "qmap_time"]).values.tolist()
        plugin_only_results = pd.read_csv(CSV_FILENAME, usecols=["qubits", "plugin_time"]).values.tolist()

    # Optionally save results to CSV
    df = pd.DataFrame({
        "qubits": [n for n, _ in qasm_results],
        "qasm_time": [t for _, t in qasm_results],
        "catalyst_time": [t for _, t in catalyst_results],
        "plugin_time": [t for _, t in plugin_only_results],
        "qmap_time": [t for _, t in qmap_results],
    })
    df.to_csv(CSV_FILENAME, index=False)

    plot_results(qasm_results, catalyst_results, plugin_only_results, qmap_results)

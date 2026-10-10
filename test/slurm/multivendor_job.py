# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Submit to IQM and Braket concurrently through one QDMI driver process."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib import import_module

from qiskit import QuantumCircuit, transpile

from mqt.core.qdmi.builtin_driver import open_device, registered_device_ids


def submit(device_id: str) -> None:
    """Open an independent driver session and collect eight shots."""
    device = open_device(device_id)
    module, name = (
        ("iqm.qdmi.qiskit", "IQMBackend")
        if device_id.startswith("iqm.")
        else ("amazon.braket.qdmi.qiskit", "AmazonBraketBackend")
    )
    backend_type = getattr(import_module(module), name)
    backend = backend_type(device=device)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    circuit = transpile(circuit, backend)
    counts = backend.run(circuit, shots=8).result().get_counts()
    assert sum(counts.values()) == 8


def main() -> None:
    """Use the same environment, catalogue and driver for both device implementations."""
    devices = ("iqm.emerald.mock", "amazon.braket.sv1")
    assert set(devices) <= set(registered_device_ids())
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(submit, devices))


if __name__ == "__main__":
    main()

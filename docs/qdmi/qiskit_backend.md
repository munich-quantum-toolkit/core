---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Qiskit Backend Integration

The {py:mod}`mqt.core.qdmi.qiskit` module provides a Qiskit {py:class}`~qiskit.providers.BackendV2`-compatible interface to QDMI devices via FoMaC.

## Installation

Install MQT Core with Qiskit support:

::::{tab-set}
:sync-group: installer

:::{tab-item} {code}`uv` _(recommended)_
:sync: uv

```console
$ uv pip install "mqt-core[qiskit]"
```

:::

:::{tab-item} {code}`pip`
:sync: pip

```console
(.venv) $ python -m pip install "mqt-core[qiskit]"
```

:::
::::

## Quickstart

```{code-cell} ipython3
from mqt.core.qdmi.qiskit import QiskitBackend
from qiskit import QuantumCircuit

# Create a backend (uses the first available FoMaC device, i.e., MQT NA Default QDMI Device)
backend = QiskitBackend()

# Create a simple circuit
qc = QuantumCircuit(2, 2)
qc.ry(1.5708, 0)  # π/2 rotation
qc.cz(0, 1)
qc.measure([0, 1], [0, 1])

# Execute the circuit
job = backend.run(qc, shots=1024)
result = job.result()
counts = result.get_counts()

print(f"Results: {counts}")
```

## Device Capabilities

The backend automatically introspects the FoMaC device and builds a Qiskit {py:class}`~qiskit.transpiler.Target` object:

```{code-cell} ipython3
# Access device properties
print(f"Number of qubits: {backend.target.num_qubits}")
print(f"Supported operations: {backend.target.operation_names}")
print(f"Capabilities hash: {backend.capabilities_hash}")
```

## Multi-Circuit Execution

Execute multiple circuits in a single job:

```{code-cell} ipython3
# Create multiple circuits for batch execution
qc1 = QuantumCircuit(2, 2)
qc1.ry(1.5708, 0)  # π/2 rotation
qc1.cz(0, 1)
qc1.measure([0, 1], [0, 1])

qc2 = QuantumCircuit(2, 2)
qc2.ry(3.1416, 0)  # π rotation
qc2.cz(0, 1)
qc2.measure([0, 1], [0, 1])

qc3 = QuantumCircuit(2, 2)
qc3.ry(1.5708, 0)  # π/2 rotation
qc3.ry(1.5708, 1)  # π/2 rotation
qc3.measure([0, 1], [0, 1])

circuits = [qc1, qc2, qc3]
job = backend.run(circuits, shots=2000)
result = job.result()

# Access results for each circuit
for idx, circuit in enumerate(circuits):
    counts = result.get_counts(idx)
    print(f"Circuit {idx}: {counts}")
```

## Extending the Backend

### Custom Operation Translators

Register custom gate translators for device-specific operations:

```python
from mqt.core.qdmi.qiskit import (
    register_operation_translator,
    InstructionContext,
    ProgramInstruction,
)


def custom_gate_translator(ctx: InstructionContext) -> list[ProgramInstruction]:
    """Translate a custom gate to neutral IR."""
    return [
        ProgramInstruction(
            name="my_gate",
            qubits=ctx.qubits,
            params=ctx.params,
        )
    ]


register_operation_translator("my_gate", custom_gate_translator)
```

### Subclassing for Vendor Integration

For vendor-specific needs, subclass {py:class}`~mqt.core.qdmi.qiskit.QiskitBackend` and override the `_submit_to_device()` method:

```python
from mqt.core.qdmi.qiskit import QiskitBackend
from qiskit import qasm3

# from your_vendor_sdk import ProgramFormat  # replace with actual import


class VendorBackend(QiskitBackend):
    """Backend that executes on real QDMI hardware."""

    def _submit_to_device(self, circuit, shots):
        # Serialize circuit using Qiskit
        qasm3_str = qasm3.dumps(circuit)

        # Submit to vendor's QDMI device
        job = self._device.submit_job(
            program=qasm3_str, program_format=ProgramFormat.QASM3, num_shots=shots
        )

        # Wait for completion and return results
        job.wait()
        return {
            "counts": job.get_counts(),
            "shots": shots,
            "success": True,
        }
```

## Current Implementation

The base {py:class}`~mqt.core.qdmi.qiskit.QiskitBackend` is a **neutral framework** designed for extension:

- Returns deterministic zero-state counts for demonstration
- Provides complete capability introspection and circuit validation
- Supports 40+ common quantum gates (see full API documentation)
- Vendors override `_submit_to_device()` for actual device execution

:::{note}
This design keeps MQT Core vendor-neutral while providing a complete integration framework.
For complete API documentation, see {py:class}`~mqt.core.qdmi.qiskit.QiskitBackend`.
:::

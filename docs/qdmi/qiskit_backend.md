---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Qiskit Backend Integration

The {py:mod}`mqt.core.plugins.qiskit` module provides a Qiskit {py:class}`~qiskit.providers.BackendV2`-compatible interface to QDMI devices via FoMaC.
This integration allows you to execute Qiskit circuits on QDMI-compliant quantum devices using a familiar Qiskit workflow.

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
from mqt.core.plugins.qiskit import QDMIProvider
from qiskit import QuantumCircuit

# Create a provider and get a backend
provider = QDMIProvider()
backend = provider.get_backend("MQT NA Default QDMI Device")

# Create a simple circuit
qc = QuantumCircuit(2)
qc.ry(1.5708, 0)  # π/2 rotation
qc.cz(0, 1)
qc.measure_all()

# Execute the circuit
job = backend.run(qc, shots=1024)
result = job.result()
counts = result.get_counts()

print(f"Results: {counts}")
```

## Provider and Device Discovery

### Using the Provider

The {py:class}`~mqt.core.plugins.qiskit.QDMIProvider` discovers QDMI devices available through the FoMaC layer.
Backends should always be obtained through the provider rather than instantiated directly.

```{code-cell} ipython3
from mqt.core.plugins.qiskit import QDMIProvider

# Create a provider
provider = QDMIProvider()

# List all available backends
backends = provider.backends()
for backend in backends:
    print(f"{backend.name}: {backend.target.num_qubits} qubits")
```

### Getting a Specific Backend

```{code-cell} ipython3
# Get a backend by name
backend = provider.get_backend("MQT NA Default QDMI Device")
print(f"Backend: {backend.name}")
print(f"Qubits: {backend.target.num_qubits}")
```

### Filtering Backends

```python
# Filter backends by name substring
filtered_qdmi = provider.backends(
    name="QDMI"
)  # Matches all backends with "QDMI" in name
filtered_na = provider.backends(name="NA")  # Matches "MQT NA Default QDMI Device"

# Filter by full name also works
exact = provider.backends(name="MQT NA Default QDMI Device")
```

## Device Capabilities and Target

The backend automatically introspects the FoMaC (QDMI) device and constructs a Qiskit {py:class}`~qiskit.transpiler.Target`
object describing device capabilities.

```{code-cell} ipython3
# Access device properties via the Target
print(f"Number of qubits: {backend.target.num_qubits}")
print(f"Supported operations: {backend.target.operation_names}")

# Check coupling map (if device has limited connectivity)
coupling_map = backend.target.build_coupling_map()
if coupling_map:
    print(f"Coupling map: {coupling_map}")
```

The backend maps QDMI device operations to corresponding Qiskit gates, including:

- **Single-qubit gates**: `x`, `y`, `z`, `h`, `s`, `t`, `sx`, `id`
- **Parametric gates**: `rx`, `ry`, `rz`, `p`, `u`, `u2`, `u3`
- **Two-qubit gates**: `cx`, `cy`, `cz`, `ch`, `swap`, `iswap`, `dcx`, `ecr`
- **Parametric two-qubit gates**: `rxx`, `ryy`, `rzz`, `rzx`, `xx_plus_yy`, `xx_minus_yy`
- **Measurement**: `measure`

## Circuit Execution

### Basic Execution

```{code-cell} ipython3
from qiskit import QuantumCircuit

# Create a circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Run on the backend
job = backend.run(qc, shots=500)
result = job.result()
counts = result.get_counts()

print(f"Counts: {counts}")
print(f"Total shots: {sum(counts.values())}")
```

### Execution Options

The backend supports various execution options:

```python
# Specify number of shots
job = backend.run(circuit, shots=2048)

# Change program format (default is QASM3)
from mqt.core import fomac

job = backend.run(circuit, shots=1024, program_format=fomac.ProgramFormat.QASM2)
```

### Circuit Requirements

Circuits must meet the following requirements before execution:

1. **All parameters must be bound**: Circuits with unbound parameters raise {py:class}`~mqt.core.plugins.qiskit.CircuitValidationError`
2. **Only supported operations**: Operations not supported by the device raise {py:class}`~mqt.core.plugins.qiskit.UnsupportedOperationError`
3. **Valid shots value**: Must be a non-negative integer

```python
from qiskit.circuit import Parameter

# This will raise CircuitValidationError
theta = Parameter("theta")
qc = QuantumCircuit(1)
qc.ry(theta, 0)
# job = backend.run(qc)  # Error: unbound parameter

# Bind parameters first
qc_bound = qc.assign_parameters({theta: 1.5708})
job = backend.run(qc_bound, shots=100)  # Success
```

## Job Handling

### Job Status

The {py:class}`~mqt.core.plugins.qiskit.QiskitJob` wraps a FoMaC (QDMI) job and provides status tracking:

```python
from qiskit.providers import JobStatus

job = backend.run(circuit, shots=1024)

# Check job status
status = job.status()
print(f"Job status: {status}")

# Job status mapping:
# CREATED → INITIALIZING
# QUEUED/SUBMITTED → QUEUED
# RUNNING → RUNNING
# DONE → DONE
# CANCELED → CANCELLED
# FAILED → ERROR
```

### Retrieving Results

Results are lazily fetched when you call `result()`:

```python
# Run the circuit
job = backend.run(circuit, shots=1024)

# Get results (waits for completion if needed)
result = job.result()

# Access measurement counts
counts = result.get_counts()

# Access result metadata
exp_result = result.results[0]
print(f"Circuit name: {exp_result.header['name']}")
print(f"Shots: {exp_result.shots}")
print(f"Success: {exp_result.success}")
```

## Multi-Circuit Execution

The backend processes circuits individually.
To execute multiple circuits, submit them sequentially:

```python
circuits = [circuit1, circuit2, circuit3]
results = []

for circuit in circuits:
    job = backend.run(circuit, shots=1000)
    result = job.result()
    results.append(result)

# Process results
for idx, result in enumerate(results):
    counts = result.get_counts()
    print(f"Circuit {idx} results: {counts}")
```

## Error Handling

The module provides specific exceptions for different error conditions:

```python
from mqt.core.plugins.qiskit import (
    CircuitValidationError,
    UnsupportedOperationError,
    JobSubmissionError,
    TranslationError,
)

try:
    job = backend.run(circuit, shots=1024)
    result = job.result()
except CircuitValidationError as e:
    # Invalid circuit (unbound parameters, invalid shots, etc.)
    print(f"Circuit validation failed: {e}")
except UnsupportedOperationError as e:
    # Circuit contains operations not supported by device
    print(f"Unsupported operation: {e}")
except JobSubmissionError as e:
    # Failed to submit job to device
    print(f"Job submission failed: {e}")
except TranslationError as e:
    # Failed to convert circuit to QASM
    print(f"Translation error: {e}")
```

## Implementation Details

### Circuit Conversion

When you run a circuit, the backend:

1. Validates the circuit (checks for unbound parameters, supported operations, valid options)
2. Converts the circuit to QASM (QASM3 by default, QASM2 optionally)
3. Submits the QASM program to the FoMaC (QDMI) device via `device.submit_job()`
4. Returns a {py:class}`~mqt.core.plugins.qiskit.QiskitJob` wrapping the FoMaC (QDMI) job

### Device Introspection

The backend builds its {py:class}`~qiskit.transpiler.Target` by:

1. Querying the FoMaC (QDMI) device for available operations
2. Mapping each operation to the corresponding Qiskit gate
3. Determining qubit connectivity from the device's coupling map
4. Including operation properties (duration, fidelity) if available

## API Reference

For complete API documentation, see:

- {py:class}`~mqt.core.plugins.qiskit.QDMIProvider` - Device provider interface
- {py:class}`~mqt.core.plugins.qiskit.QiskitBackend` - BackendV2 implementation
- {py:class}`~mqt.core.plugins.qiskit.QiskitJob` - Job wrapper and result handling
- {py:mod}`~mqt.core.plugins.qiskit.exceptions` - Exception types

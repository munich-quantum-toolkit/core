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
backend = provider.get_backend("MQT Core DDSIM QDMI Device")

# Create a simple circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
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
backend = provider.get_backend("MQT Core DDSIM QDMI Device")
print(f"Backend: {backend.name}")
print(f"Qubits: {backend.target.num_qubits}")
```

### Filtering Backends

```python
# Filter backends by name substring
filtered_qdmi = provider.backends(
    name="QDMI"
)  # Matches all backends with "QDMI" in name
filtered_ddsim = provider.backends(name="DDSIM")  # Matches "MQT Core DDSIM QDMI Device"

# Filter by full name also works
exact = provider.backends(name="MQT Core DDSIM QDMI Device")
```

## Authentication

The {py:class}`~mqt.core.plugins.qiskit.QDMIProvider` supports authentication for accessing QDMI devices that require credentials.
Authentication parameters are passed to the provider constructor and forwarded to the underlying session.

:::{note}
The default local devices (MQT Core DDSIM QDMI Device, MQT NA Default QDMI Device) do not require authentication.
Authentication is primarily used when connecting to remote quantum hardware.
:::

### Supported Authentication Methods

The provider supports multiple authentication methods:

- **Token-based authentication**: Using an API token or access token
- **Username/password authentication**: Traditional credential-based authentication
- **File-based authentication**: Reading credentials from a file
- **URL-based authentication**: Connecting to an authentication server
- **Project-based authentication**: Associating sessions with specific projects, e.g., for accounting or quota management

### Using Authentication Tokens

The most common authentication method is using an API token:

```python
from mqt.core.plugins.qiskit import QDMIProvider

# Authenticate with a token
provider = QDMIProvider(token="your_api_token_here")

# Get backends
backends = provider.backends()
for backend in backends:
    print(f"{backend.name}: {backend.target.num_qubits} qubits")
```

### Username and Password Authentication

For services that use traditional username/password authentication:

```python
# Authenticate with username and password
provider = QDMIProvider(username="your_username", password="your_password")

# Access backend
backend = provider.get_backend("RemoteQuantumDevice")
```

### File-Based Authentication

Store credentials in a secure file for better security:

```python
# Authenticate using a credentials file
# The file should contain authentication information in the format expected by the service
provider = QDMIProvider(auth_file="/path/to/credentials.txt")
```

### Authentication Server URL

Connect to a custom authentication server:

```python
# Use a custom authentication URL
provider = QDMIProvider(auth_url="https://auth.quantum-service.com/api/v1/auth")
```

### Project-Based Authentication

Associate your session with a specific project or organization:

```python
# Specify a project ID
provider = QDMIProvider(
    token="your_api_token", project_id="quantum-research-project-2024"
)
```

### Combining Authentication Parameters

Multiple authentication parameters can be combined for services that require multiple credentials:

```python
# Use multiple authentication parameters
provider = QDMIProvider(
    token="your_api_token",
    username="your_username",
    password="your_password",
    project_id="your_project_id",
    auth_url="https://custom-auth.example.com",
)
```

### Authentication Error Handling

When authentication fails, the provider raises a `RuntimeError`:

```python
try:
    provider = QDMIProvider(token="invalid_token")
    backends = provider.backends()
except RuntimeError as e:
    print(f"Authentication failed: {e}")
    # Handle authentication error (e.g., prompt for valid credentials)
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

- **Single-qubit Pauli gates**: `x`, `y`, `z`, `id`/`i`
- **Hadamard**: `h`
- **Phase gates**: `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg`, `p`, `phase`, `gphase`
- **Rotation gates (parametric)**: `rx`, `ry`, `rz`, `r`/`prx`
- **Universal gates (parametric)**: `u`, `u1`, `u2`, `u3`
- **Two-qubit gates**: `cx`/`cnot`, `cy`, `cz`, `ch`, `cs`, `csdg`, `csx`, `swap`, `iswap`, `dcx`, `ecr`
- **Two-qubit parametric gates**: `cp`, `cu1`, `cu3`, `crx`, `cry`, `crz`, `rxx`, `ryy`, `rzz`, `rzx`, `xx_plus_yy`, `xx_minus_yy`
- **Three-qubit gates**: `ccx`, `ccz`, `cswap`
- **Multi-controlled gates**: `mcx`, `mcz`, `mcp`, `mcrx`, `mcry`, `mcrz`
- **Non-unitary operations**: `reset`, `measure`

## Circuit Execution

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

Circuits must meet the following requirements before execution:

1. **All parameters must be bound**: Circuits with unbound parameters raise {py:class}`~mqt.core.plugins.qiskit.CircuitValidationError`
2. **Only supported operations**: Operations not supported by the device raise {py:class}`~mqt.core.plugins.qiskit.UnsupportedOperationError`
3. **Valid shots value**: Must be a non-negative integer

### Parameter Binding

The backend supports automatic parameter binding through the `parameter_values` argument.
You can pass parameter values either as dictionaries or as sequences of values:

```python
from qiskit.circuit import Parameter

# Option 1: Bind parameters manually
theta = Parameter("theta")
qc = QuantumCircuit(1)
qc.ry(theta, 0)
qc.measure_all()

qc_bound = qc.assign_parameters({theta: 1.5708})
job = backend.run(qc_bound, shots=100)

# Option 2: Use parameter_values argument (recommended)
job = backend.run(qc, parameter_values=[{theta: 1.5708}], shots=100)

# For multiple circuits with different parameters
circuits = [qc, qc, qc]
param_values = [{theta: 0.5}, {theta: 1.0}, {theta: 1.5}]
job = backend.run(circuits, parameter_values=param_values, shots=100)
```

## Job Handling

### Job Status

The {py:class}`~mqt.core.plugins.qiskit.QDMIJob` wraps a FoMaC (QDMI) job and provides status tracking:

```python
from qiskit.providers import JobStatus

job = backend.run(qc, shots=1024)

# Check job status
status = job.status()
print(f"Job status: {status}")
```

### Retrieving Results

Results are lazily fetched when you call `result()`:

```python
# Run the circuit
job = backend.run(qc, shots=1024)

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

The backend supports both single-circuit and multi-circuit execution.
You can submit multiple circuits in a single call:

```python
# Create multiple circuits
qc1 = QuantumCircuit(2)
qc1.h(0)
qc1.cx(0, 1)
qc1.measure_all()

qc2 = QuantumCircuit(2)
qc2.x(0)
qc2.cx(0, 1)
qc2.measure_all()

qc3 = QuantumCircuit(2)
qc3.h([0, 1])
qc3.measure_all()

# Submit all circuits at once
circuits = [qc1, qc2, qc3]
job = backend.run(circuits, shots=1000)

# Get aggregated results
result = job.result()

# Process results for each circuit
for idx in range(len(circuits)):
    counts = result.get_counts(idx)
    print(f"Circuit {idx} results: {counts}")
```

Alternatively, you can still submit circuits individually:

```python
results = []
for qc in circuits:
    job = backend.run(qc, shots=1000)
    result = job.result()
    results.append(result)
```

## Qiskit Primitives

The backend provides implementations of Qiskit's [Primitives V2](https://docs.quantum.ibm.com/api/qiskit/primitives) interfaces:
{py:class}`~mqt.core.plugins.qiskit.QDMISampler` and {py:class}`~mqt.core.plugins.qiskit.QDMIEstimator`.
These primitives allow for a simplified execution workflow for sampling bitstrings and estimating expectation values.

### Sampler

The {py:class}`~mqt.core.plugins.qiskit.QDMISampler` implements the `BaseSamplerV2` interface.
It is used to sample quantum circuits and obtain measurement counts (bitstrings).

```{code-cell} ipython3
from mqt.core.plugins.qiskit import QDMISampler
from qiskit import QuantumCircuit

# Initialize sampler with the backend
sampler = QDMISampler(backend)

# Create a circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Run the sampler
job = sampler.run([qc], shots=1024)
result = job.result()

# Get results for the first pub (Primitive Unified Bloc)
pub_result = result[0]
counts = pub_result.data.meas.get_counts()

print(f"Sampler results: {counts}")
```

### Estimator

The {py:class}`~mqt.core.plugins.qiskit.QDMIEstimator` implements the `BaseEstimatorV2` interface.
It is used to calculate expectation values of observables.

```{code-cell} ipython3
from mqt.core.plugins.qiskit import QDMIEstimator
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
import numpy as np

# Initialize estimator
estimator = QDMIEstimator(backend)

# Create a circuit and observable
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

observable = SparsePauliOp("ZZ")

# Run the estimator
job = estimator.run([(qc, observable)])
result = job.result()

# Get the expectation value
pub_result = result[0]
ev = pub_result.data.evs
std = pub_result.data.stds

print(f"Expectation value: {ev}")
print(f"Standard deviation: {std}")
```

You can also use parameterized circuits with the estimator:

```{code-cell} ipython3
from qiskit.circuit import Parameter

# Parameterized circuit
theta = Parameter("theta")
qc_param = QuantumCircuit(1)
qc_param.rx(theta, 0)

op = SparsePauliOp("Z")

# Run with specific parameter values
# Format: (circuit, observable, parameter_values)
vals = [0.0, np.pi/2, np.pi]
job = estimator.run([(qc_param, op, vals)])
result = job.result()

print(f"Expectation values: {result[0].data.evs}")
```

## Error Handling

The module provides specific exceptions for different error conditions:

```python
from mqt.core.plugins.qiskit import (
    CircuitValidationError,
    UnsupportedOperationError,
    UnsupportedDeviceError,
    JobSubmissionError,
    TranslationError,
    UnsupportedFormatError,
)

try:
    job = backend.run(qc, shots=1024)
    result = job.result()
except CircuitValidationError as e:
    # Invalid circuit (unbound parameters, invalid shots, etc.)
    print(f"Circuit validation failed: {e}")
except UnsupportedOperationError as e:
    # Circuit contains operations not supported by device
    print(f"Unsupported operation: {e}")
except UnsupportedDeviceError as e:
    # Device cannot be represented in Qiskit's Target model
    print(f"Unsupported device: {e}")
except JobSubmissionError as e:
    # Failed to submit job to device
    print(f"Job submission failed: {e}")
except TranslationError as e:
    # Failed to convert circuit to supported program format
    print(f"Translation error: {e}")
except UnsupportedFormatError as e:
    # No supported program format available
    print(f"Unsupported format: {e}")
```

## Implementation Details

### Circuit Conversion

When you run a circuit, the backend:

1. Validates the circuit (checks for unbound parameters, supported operations, valid options)
2. Converts the circuit to one of the program formats supported by the target device (IQM JSON, OpenQASM 2, OpenQASM 3) using {py:func}`~mqt.core.plugins.qiskit.qiskit_to_iqm_json` or Qiskit's built-in QASM exporters
3. Submits the program to the QDMI device via `device.submit_job()`
4. Returns a {py:class}`~mqt.core.plugins.qiskit.QDMIJob`

### Device Introspection

The backend builds its {py:class}`~qiskit.transpiler.Target` by:

1. Querying the FoMaC (QDMI) device for available operations
2. Mapping each operation to the corresponding Qiskit gate
3. Determining qubit connectivity from the device's coupling map
4. Including operation properties (duration, fidelity) if available

### Primitives Implementation

The Qiskit Primitives are implemented as lightweight wrappers around the backend execution:

- **Sampler**: Submits circuits to the backend and reshapes the resulting bitstrings into the requested structure (PubResult).
- **Estimator**: Decomposes observables into Pauli terms, appends necessary basis rotations and measurements to the provided circuits, and submits them to the backend. It then reconstructs expectation values and standard deviations from the measurement counts of each term based on the provided precision or shots.

## API Reference

For complete API documentation, see:

- {py:class}`~mqt.core.plugins.qiskit.QDMIProvider` — Device provider interface
- {py:class}`~mqt.core.plugins.qiskit.QDMIBackend` — BackendV2 implementation
- {py:class}`~mqt.core.plugins.qiskit.QDMIJob` — Job wrapper and result handling
- {py:class}`~mqt.core.plugins.qiskit.QDMIEstimator` — EstimatorV2 primitive implementation
- {py:class}`~mqt.core.plugins.qiskit.QDMISampler` — SamplerV2 primitive implementation
- {py:mod}`~mqt.core.plugins.qiskit.exceptions` — Exception types

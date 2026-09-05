---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Qiskit Backend Integration

The {py:mod}`mqt.core.plugins.qiskit` module provides a Qiskit
{py:class}`~qiskit.providers.BackendV2`-compatible interface to QDMI devices via
the MQT Core QDMI bindings. This integration lets you execute Qiskit circuits on
QDMI devices with a standard Qiskit workflow.

## Installation

Install MQT Core with Qiskit support:

::::{tab-set}
:sync-group: installer

:::{tab-item} {code}`uv` _(recommended)_
:sync: uv

```console
uv pip install "mqt-core[qiskit]"
```

:::

:::{tab-item} {code}`pip`
:sync: pip

```console
python -m pip install "mqt-core[qiskit]"
```

:::

::::

## Quickstart

```{code-cell} ipython3
from mqt.core.plugins.qiskit import QDMIBackend
from qiskit import QuantumCircuit

# Open the registered DDSIM device by its stable ID
backend = QDMIBackend.from_device_id("mqt.ddsim.default")

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

The {py:class}`~mqt.core.plugins.qiskit.provider.QDMIProvider` discovers
registered QDMI devices. Use it when an application must enumerate backends.

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
# Open a backend directly by stable device ID
from mqt.core.plugins.qiskit import QDMIBackend

backend = QDMIBackend.from_device_id("mqt.ddsim.default")
print(f"Backend: {backend.name}")
print(f"Qubits: {backend.target.num_qubits}")
```

Optional session keywords apply explicit overrides to this fresh device session.
Their names and value types are described by
{py:class}`mqt.core.typing.QDMISessionParameters`; persistent configuration
remains the default:

```python
backend = QDMIBackend.from_device_id(
    "provider.device",
    token="access-token",
    custom1="provider-specific-value",
)
```

### Filtering Backends

```python
# Filter backends by name substring
filtered_qdmi = provider.backends(name="QDMI")  # Matches all backends with "QDMI" in name
filtered_ddsim = provider.backends(name="DDSIM")  # Matches "MQT Core DDSIM QDMI Device"

# Filter by full name also works
exact = provider.backends(name="MQT Core DDSIM QDMI Device")
```

## Authentication

{py:class}`~mqt.core.plugins.qiskit.provider.QDMIProvider` does not define a
generic credential interface. It opens each registered device with its
persistent definition. Configure credentials through the selected QDMI device
implementation. For example, a provider can use a credential file, an
environment variable, or a platform credential-provider chain. See
[QDMI device configuration](configuration.md) for persistent session settings.

## Device Capabilities and Target

The backend automatically introspects the QDMI device and constructs a Qiskit
{py:class}`~qiskit.transpiler.Target` object describing device capabilities.

```{code-cell} ipython3
# Access device properties via the Target
print(f"Number of qubits: {backend.target.num_qubits}")
print(f"Supported operations: {backend.target.operation_names}")

# Check coupling map (if device has limited connectivity)
coupling_map = backend.target.build_coupling_map()
if coupling_map:
    print(f"Coupling map: {coupling_map}")
```

The backend maps QDMI device operations to corresponding Qiskit gates,
including:

- **Single-qubit Pauli gates**: `x`, `y`, `z`, `id`/`i`
- **Hadamard**: `h`
- **Phase gates**: `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg`, `p`, `phase`, `gphase`
- **Rotation gates (parametric)**: `rx`, `ry`, `rz`, `r`/`prx`
- **Universal gates (parametric)**: `u`, `u1`, `u2`, `u3`
- **Two-qubit gates**: `cx`/`cnot`, `cy`, `cz`, `ch`, `cs`, `csdg`, `csx`,
  `swap`, `iswap`, `dcx`, `ecr`
- **Two-qubit parametric gates**: `cp`, `cu1`, `cu3`, `crx`, `cry`, `crz`,
  `rxx`, `ryy`, `rzz`, `rzx`, `xx_plus_yy`, `xx_minus_yy`
- **Three-qubit gates**: `ccx`, `ccz`, `cswap`, `rccx`
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

1. **All parameters must be bound**: Circuits with unbound parameters raise
   {py:class}`~mqt.core.plugins.qiskit.exceptions.CircuitValidationError`
2. **Only supported operations**: Operations not supported by the device raise
   {py:class}`~mqt.core.plugins.qiskit.exceptions.UnsupportedOperationError`
3. **Valid shots value**: Must be a non-negative integer

### Parameter Binding

The backend supports automatic parameter binding through the `parameter_values`
argument. You can pass parameter values either as dictionaries or as sequences
of values:

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

The {py:class}`~mqt.core.plugins.qiskit.job.QDMIJob` wraps a QDMI job and
provides status tracking:

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

The backend supports both single-circuit and multi-circuit execution. You can
submit multiple circuits in a single call:

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

Use Qiskit's
[BackendSamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.BackendSamplerV2)
and
[BackendEstimatorV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.BackendEstimatorV2).
The backend factories construct these native objects with typed keyword options.
Qiskit supplies the defaults and validates the options:

```python
sampler = backend.sampler(default_shots=1024)
estimator = backend.estimator(default_precision=0.1, abelian_grouping=True)

samples = sampler.run([measured_circuit]).result()[0]
counts = samples.data.meas.get_counts()
estimate = estimator.run([(circuit, SparsePauliOp("ZZ"))]).result()[0]
expectation, standard_error = estimate.data.evs, estimate.data.stds
```

Sampler defaults to 1024 shots. Estimator requires positive precision and
defaults to `1/64` (4096 shots); it groups qubit-wise commuting measurements.
Both use Qiskit's broadcasting, metadata, and asynchronous primitive jobs.
Calling `result()` waits for completion. PUBs with equal shot counts share a
backend batch; different shot counts or precisions follow Qiskit's scheduling.
Primitive-job cancellation follows Qiskit's future semantics: it does not abort
an already-running backend call.

### Backend requirements

| Feature                                    | Required QDMI result support  |
| ------------------------------------------ | ----------------------------- |
| Counts-only execution and native Estimator | `HIST_KEYS` and `HIST_VALUES` |
| `memory=True` and native Sampler           | `SHOTS`                       |

Native Sampler requests memory automatically. DDSIM supports both primitives;
counts-only devices must add `SHOTS` support to run Sampler. The backend never
reconstructs shots from counts; when memory is requested, it derives counts from
those same genuine shots.

```{code-cell} ipython3
sampler = backend.sampler(default_shots=100)
samples = sampler.run([qc]).result()[0]
print(samples.data.meas.get_counts())
```

A device must advertise its supported operations and accept OpenQASM 2, OpenQASM
3, or a [registered program format](#program-serializers). Transpile circuits to
the backend target before submission. Estimator also needs the basis rotations
and measurements that Qiskit generates for the observables. Use the provider's
specialized backend when its program dialect requires one, such as
`amazon.braket.qdmi.qiskit.AmazonBraketBackend` for Braket.

Results must contain one binary digit per classical bit, with `clbits[0]` on the
right, including unmeasured bits initialized to zero. Classical registers must
partition `circuit.clbits` in register order; loose, aliased, and reordered bits
are rejected. Serializers and providers must preserve this mapping. Shot order
is unchanged across registers, so joint samples and postselection remain valid.

The backend accepts nonnegative integer `shots` and boolean `memory` options.
QDMI has no standard seed parameter, so the generic backend rejects non-`None`
`seed_simulator`. DDSIM's [custom seed parameter](ddsim_device.md) is available
through direct QDMI job submission; other providers can define different custom
parameters. Other execution options are unsupported. The backend validates the
whole batch before submission, submits jobs in circuit order, and collects
results in that order. Remote IDs are queried only when needed. Submission or
collection failure triggers best-effort cancellation of submitted jobs;
cancellation errors do not replace the original error. Missing memory, invalid
bitstrings or shot totals, and failed or canceled jobs raise instead of yielding
partial or zero-filled samples. Successful repeated reads reuse the result.

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

### Circuit Serialization

When you run a circuit, the backend:

1. Validates the circuit (checks for unbound parameters, supported operations,
   valid options)
2. Serializes the circuit into one of the program formats supported by the
   target device, through the program serializer registered for that format
3. Submits the program to the QDMI device via `device.submit_job()`
4. Returns a {py:class}`~mqt.core.plugins.qiskit.job.QDMIJob`

### Program Serializers

A _program serializer_ turns one circuit into one program in one program format.
MQT Core provides the serializers for OpenQASM 2 and OpenQASM 3. Every other
format belongs to the package that owns the device, which registers its
serializer through the same registry.

A format fixes the kind of payload it carries, so there are two signatures. A
text format takes a serializer that returns `str`:

```python
def serialize(circuit: QuantumCircuit, backend: QDMIBackend) -> str: ...
```

A binary format takes one that returns `bytes`:

```python
def serialize(circuit: QuantumCircuit, backend: QDMIBackend) -> bytes: ...
```

{py:func}`~mqt.core.qdmi.is_binary_program_format` states which kind a format
carries. The backend checks the returned type against the format and raises
{py:class}`~mqt.core.plugins.qiskit.exceptions.TranslationError` on a mismatch.
A serializer reads the device through
{py:attr}`~mqt.core.plugins.qiskit.backend.QDMIBackend.device` and the supported
operations through
{py:attr}`~mqt.core.plugins.qiskit.backend.QDMIBackend.target`.

A package advertises its serializers through the
`mqt.core.qiskit.program_serializers` entry point group. The entry point name is
the {py:class}`~mqt.core.qdmi.ProgramFormat` member name:

```toml
[project.entry-points."mqt.core.qiskit.program_serializers"]
IQM_JSON = "iqm.qdmi.serializers:qiskit_to_iqm_json"
```

{py:func}`~mqt.core.plugins.qiskit.serializers.register_program_serializer` does
the same at run time:

```python
from mqt.core.plugins.qiskit import register_program_serializer
from mqt.core.qdmi import ProgramFormat

register_program_serializer(ProgramFormat.IQM_JSON, qiskit_to_iqm_json)
```

Pass `replace=True` to take over a format that already has a serializer,
including OpenQASM 2 and OpenQASM 3.

A device usually accepts several formats. The backend walks them in the order of
{py:data}`~mqt.core.plugins.qiskit.serializers.PROGRAM_FORMAT_PREFERENCE` and
uses the first one that has a serializer, so the order of the list decides and
not the order the device reports:

```text
IQM_JSON, CUSTOM1 ... CUSTOM5,
QIR_ADAPTIVE_MODULE, QIR_ADAPTIVE_STRING,
QPY, QASM3,
QIR_BASE_MODULE, QIR_BASE_STRING,
QASM2
```

A device-native format comes first, because a package that registers a
serializer for its own device's format wants that format used. The standardized
formats follow in order of what a circuit may contain: the QIR adaptive profile
allows classical control, QPY carries a Qiskit circuit without loss, and
OpenQASM 3 expresses control flow, while the QIR base profile forbids classical
feedback and OpenQASM 2 has no control flow at all. Encoding only breaks a tie
within one profile, because it decides how the program travels rather than what
it may say. `CALIBRATION` and `BATCH_JOB` are absent because a serialized
circuit is not what they carry.

### Device Introspection

The backend builds its {py:class}`~qiskit.transpiler.Target` by:

1. Querying the QDMI device for available operations
2. Mapping each operation to the corresponding Qiskit gate
3. Determining qubit connectivity from the device's coupling map
4. Including operation properties (duration, fidelity) if available

## API Reference

For complete API documentation, see:

- {py:class}`~mqt.core.plugins.qiskit.provider.QDMIProvider` — Device provider
  interface
- {py:class}`~mqt.core.plugins.qiskit.backend.QDMIBackend` — BackendV2
  implementation
- {py:class}`~mqt.core.plugins.qiskit.job.QDMIJob` — Job wrapper and result
  handling
- {py:mod}`~mqt.core.plugins.qiskit.exceptions` — Exception types

---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  output_stderr: remove
---

# PennyLane interface for gate-based QDMI devices

The {py:mod}`mqt.core.plugins.pennylane` module runs PennyLane QNodes and
Catalyst-compiled functions on gate-based QDMI devices. MQT Core compiles the
circuits for the selected device and reconstructs finite-shot measurement
results.

The device must provide gate and topology metadata usable by MQT Core's
compiler, accept QIR, OpenQASM 3, or IQM JSON, and return computational-basis
samples or counts. OpenQASM 2-only devices are unsupported. The examples use the
local [DD-based simulator device](ddsim_device.md) and require no credentials or
remote resources.

Install MQT Core with the optional PennyLane dependency:

```bash
uv pip install "mqt-core[pennylane]"
```

For compilation with `qp.qjit`, also install Catalyst:

```bash
uv pip install "pennylane-catalyst>=0.15,<0.16.0a0"
```

The integration supports Catalyst 0.15.x and 0.16 development builds on Linux
x86_64/aarch64, and Catalyst 0.15.x on macOS arm64. Development builds on macOS
currently
[normalize library suffixes before loading](https://github.com/PennyLaneAI/catalyst/pull/2904),
replacing the Python extension's `.so` with `.dylib`; use the released Catalyst
version there. Ordinary QNodes do not require Catalyst. See the
[Catalyst installation guide](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/installation.html)
for platform requirements.

The version range admits 0.16 development builds when prereleases are enabled or
an official development wheel is installed explicitly. Keep the PennyLane
version required by that Catalyst wheel; development releases pin matching
PennyLane and Lightning builds.

## Quickstart

A Bell-state circuit illustrates device discovery by stable ID, circuit
conversion, execution, and finite-shot result reconstruction.

```{code-cell} ipython3
import pennylane as qp

bell_device = qp.device("mqt.ddsim.default", wires=2, job_parameters={"custom1": 7})


@qp.qnode(bell_device, shots=1000)
def bell_state():
    qp.Hadamard(0)
    qp.CNOT(wires=[0, 1])
    return qp.counts(wires=[0, 1])


bell_counts = bell_state()
assert sum(bell_counts.values()) == 1000
assert set(bell_counts) <= {"00", "11"}
print({str(key): int(value) for key, value in sorted(bell_counts.items())})
```

Only the computational-basis states $00$ and $11$ have nonzero probability, up
to finite-shot fluctuations in their relative frequencies.

Add `qp.qjit` to compile a QNode with Catalyst:

```{code-cell} ipython3
compiled_device = qp.device("mqt.ddsim.default", wires=2)


@qp.qjit
@qp.qnode(compiled_device, shots=1000, diff_method="parameter-shift")
def compiled_expectation(angle):
    qp.RY(angle, 0)
    qp.CNOT(wires=[0, 1])
    return qp.expval(qp.Z(0))


value = compiled_expectation(0.4)
```

The default `qp.qjit` mode applies PennyLane preprocessing and supports
parameter-shift gradients. Catalyst also offers experimental program capture;
its current preprocessing and differentiation limits are described under
[Supported gate-level scope](#supported-gate-level-scope).

## Program conversion

PennyLane preprocesses ordinary QNode tapes by decomposing unsupported
operations, splitting incompatible measurements, changing measurement bases, and
expanding parameter broadcasts. MQT Core then synthesizes gates and routes the
circuits for the selected QDMI target. Programs need not use the provider's
native gate names or physical qubit placement. Invalid target metadata and
unsupported target configurations fail before submission; all programs in an
ordinary QNode batch use one snapshot of the target and are prepared before any
jobs are submitted.

The compiler selects an advertised format in this order: Adaptive QIR
binary/text, OpenQASM 3, IQM JSON, then Base QIR binary/text. IQM targets must
provide physical site names and supported gate metadata; the compiler emits
terminal readout keys in logical output order. Results follow PennyLane's wire
order regardless of physical placement. If the provider returns only counts,
samples are reconstructed without temporal shot order. Terminal measurements of
a compiled circuit share one provider readout.

Provider helpers select `program_format`, optional `terminal_readout`, and
OpenQASM `physical_qubits` addressing. Both ordinary QNodes and Catalyst retain
the compiler's result layout before selecting PennyLane wires. Terminal
normalization supports measured subsets, permutations, and zero-filled holes;
the default preserves the selected format's broader readout semantics. See
[terminal readout](../mlir/target_compilation.md#terminal-readout-and-results).

Hardware parameter constraints come from the provider's QDMI operation metadata.
Both execution modes use the same compiler target, including fixed angles and
bounded parameters when the provider advertises them.

Catalyst compiles the host control flow ahead of time. MQT Core records and
compiles the quantum program at execution time using the current parameter
values. Include this device compilation cost when measuring repeated calls.

To inspect or transform a circuit without opening a device session, import a
preprocessed tape with {py:meth}`~mqt.core.mlir.QCProgram.from_pennylane`:

```{code-cell} ipython3
from mqt.core.mlir import QCProgram

tape = qp.tape.QuantumScript(
    [qp.Hadamard("a"), qp.CNOT(wires=["a", "b"])],
    [qp.sample(wires=["a", "b"])],
)
program = QCProgram.from_pennylane(tape, wires=["a", "b"])
print(program.to_openqasm3())
```

The importer expects supported gates and terminal measurements. It appends
computational-basis readout in the supplied wire order. Use PennyLane's device
preprocessing for decomposition, basis changes, and deferred measurements. The
resulting {py:class}`~mqt.core.mlir.QCProgram` supports compiler transformations
and export; {py:func}`~mqt.core.mlir.compile_program` prepares it for a QDMI
target.

## End-to-end use case: finite-shot MaxCut QAOA

This example uses Hamiltonian evolution and Catalyst to optimize MaxCut on a
four-node graph. The cost Hamiltonian is the negative cut size. Each layer
evolves under that Hamiltonian and an X mixer.

```{code-cell} ipython3
import catalyst
import jax.numpy as jnp
import networkx as nx
import numpy as np

graph = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3)])
cost_hamiltonian = qp.sum(
    *(0.5 * (qp.Z(u) @ qp.Z(v) - qp.I(u)) for u, v in graph.edges)
)
mixer_hamiltonian = qp.sum(*(qp.X(wire) for wire in graph.nodes))
qaoa_device = qp.device(
    "mqt.ddsim.default", wires=4, job_parameters={"custom1": 7}
)


def ansatz(parameters):
    for wire in graph.nodes:
        qp.Hadamard(wire)
    qp.ApproxTimeEvolution(cost_hamiltonian, parameters[0], 1)
    qp.ApproxTimeEvolution(mixer_hamiltonian, parameters[1], 1)


@qp.qnode(qaoa_device, shots=1000, diff_method="parameter-shift")
def correlations(parameters):
    ansatz(parameters)
    return tuple(
        qp.expval(qp.Z(first) @ qp.Z(second)) for first, second in graph.edges
    )


def cost(parameters):
    return (
        0.5 * jnp.sum(jnp.asarray(correlations(parameters)))
        - 0.5 * graph.number_of_edges()
    )


@qp.qjit
def step(parameters):
    return parameters - 0.15 * catalyst.grad(cost)(parameters)


@qp.qjit
@qp.qnode(qaoa_device, shots=1000)
def sample(parameters):
    ansatz(parameters)
    return qp.sample(wires=range(4))


compiled_cost = qp.qjit(cost)
parameters = jnp.array([0.5, 0.5])
objective_values = [float(compiled_cost(parameters))]
for _ in range(4):
    parameters = step(parameters)
    objective_values.append(float(compiled_cost(parameters)))

samples = np.asarray(sample(parameters), dtype=np.int8)
cut_sizes = sum(
    samples[:, first] != samples[:, second] for first, second in graph.edges
)
best = samples[np.argmax(cut_sizes)]
print(f"Objective estimates: {np.round(objective_values, 3)}")
print(f"Final parameters: {parameters}")
print(f"Best sampled partition: {best}, cut size: {max(cut_sizes)}")
```

The QNode measures edge correlations; the classical `cost` function combines
them into the MaxCut objective. Keeping this sum outside the QNode lets Catalyst
differentiate the objective with parameter-shift. Parameter-shift gradients
evaluate shifted circuits through QDMI. Finite-shot objective estimates need not
decrease at every step. Each compiled QNode uses the device's existing session
and compiles its quantum circuit for that target.

The terms within each Hamiltonian commute, so one Trotter step implements each
layer exactly. PennyLane decomposes these evolutions into gates before device
execution.

## Direct generic construction

Stable entry points such as `mqt.ddsim.default` provide the simplest
construction. A device integration package can register a
{py:class}`~mqt.core.plugins.pennylane.device.QDMIDevice` subclass under the
stable ID of another QDMI device. Applications may also construct the generic
class directly:

```python
import pennylane as qp

from mqt.core.plugins.pennylane import QDMIDevice

device_id = "stable ID reported by the QDMI driver"
device = QDMIDevice(
    device_id=device_id,
    wires=["a", "b", "c", "d"],
    session_parameters={
        "token": "...",
        "base_url": "https://device.example",
    },
    job_parameters={
        "custom1": "device-specific job value",
    },
)
```

Automatic retries are disabled (`max_retries=0`). To retry confirmed failed jobs
up to three times, set the option when creating the device:

```python
device = qp.device("mqt.ddsim.default", wires=2, max_retries=3)
```

Retries create additional executions and can incur charges. Successful jobs are
reused; errors that do not confirm job failure require
[explicit recovery](#recovering-results-after-a-failure).

An integration can return an already-open device. Pass that handle directly to
the generic class. Do not repeat session parameters because the session already
exists:

```python
from mqt.core.plugins.pennylane import QDMIDevice
from mqt.core.qdmi import builtin_driver

device = QDMIDevice(
    device=builtin_driver.open_device("mqt.ddsim.default"),
)
```

Ordinary QNodes accept arbitrary PennyLane wire labels. Compiled QNodes require
contiguous integer labels starting at zero. Every execution requires finite
shots: set them on the QNode or override them with `qp.set_shots`. Devices do
not accept a `shots` argument or provide a default shot count.

Ordinary QNode execution groups tapes and shot partitions with the same program
format and shot count into native QDMI jobs. Providers that decline a program
list receive independent submissions. All jobs are submitted before waiting for
results, which retain the input tape and shot-partition order. `submitted_jobs`
counts QDMI jobs.

Use `with qp.Tracker(device) as tracker:` to record accepted program executions
(`executions`), requested shots, batches, and the number of tapes in each batch
(`batch_len`). Shot-vector copies count as separate executions even when grouped
in one native job. Tracking includes accepted jobs whose execution later fails.

Keep the device alive while using compiled functions. They use its open QDMI
session and job parameters; saved executables cannot restore that session in
another process. Compiled calls are synchronous and do not use the Python
tracker. Catalyst's `async_qnodes=True` schedules separate QNodes without
combining them into one QDMI job. Concurrent calls require a provider that
supports them on the shared session; see the
[QDMI thread-safety contract](driver.md#python-bindings). QDMI waits and result
retrieval release the Python GIL.

## Supported gate-level scope

Supported gates include identity and Pauli gates; H, S, T, SX; RX, RY, RZ, phase
shift, global phase, and U gates; controlled Pauli, rotation, and phase gates;
Toffoli, SWAP, CSWAP, ISWAP, ECR; and Ising XX, YY, and ZZ rotations. Adjoints
and controls, including negative controls, are supported. PennyLane decomposes
higher-level operations, and MQT Core lowers the circuit to the target's
operations and topology.

Probabilities, counts, expectations, and variances are finite-shot estimates.
Ordinary QNodes support scalar products of observables, including sampled values
and variances. PennyLane splits incompatible observables into separate circuits
and applies the required measurement-basis changes. Pulse programming, arbitrary
Hermitian observables, and analytic execution are unsupported.

Ordinary QNodes use PennyLane's deferred-measurement transform for mid-circuit
measurements by default; `mcm_method="deferred"` also selects it explicitly.
Reset and feedback may require unused device wires; custom labels are supported.
Programs that need more wires than the device exposes, postselection, and
`one-shot` or `tree-traversal` execution are unsupported. Deferred measurements
do not require a native method in `supported_mcm_methods`.

Catalyst's measurement callback requires an immediate result for host control
flow, whereas this backend submits a complete program to QDMI. Core's support
for measurement and feedback within a compiled program does not provide that
callback contract. Catalyst mid-circuit measurements and `mcm_method="deferred"`
are therefore unsupported.

With Catalyst, `BasisState` accepts runtime bit arrays and reordered wire
subsets. It must precede every other operation; repeated preparation and resets
are unsupported. `QubitUnitary` accepts runtime matrices on at most two wires,
including controls, and supports adjoints and negative controls. Both operations
support `capture=True`. Larger matrices and general `StatePrep` require
PennyLane decomposition.

Both capture modes support computational-basis samples, counts, probabilities,
and Z/product expectations and variances. Under `capture=True`, apply explicit
basis changes and observable splitting for other measurements:

```python
import catalyst


@qp.qjit(capture=True, skip_preprocess=True)
@catalyst.passes.diagonalize_measurements
@qp.transform(pass_name="split-non-commuting")
@qp.qnode(compiled_device, shots=1000)
def captured_expectation(angle):
    qp.Hadamard(0)
    qp.S(0)
    qp.RY(angle, 1)
    return qp.expval(0.5 * qp.Y(0) + qp.Z(1))


value = captured_expectation(0.4)
```

`skip_preprocess=True` selects the explicit transforms above instead of the tape
preprocessing pipeline. The splitting pass supports expectations only; use
diagonalization alone for a variance. For gate decomposition under capture, load
Catalyst's capabilities with
`qp.devices.DeviceCapabilities.from_toml_file(device.config_filepath, "qjit")`
and pass `set(capabilities.operations)` as the `gate_set` of
`qp.transforms.decompose` on the QNode.

ECR, SX, U1, U2, and U3 use native Core gates in ordinary QNodes. Catalyst
lowers them through decomposition; program capture currently requires the
explicit transform. The default `qp.qjit` mode applies decomposition
automatically.

Catalyst counts return arrays rather than dictionaries. Non-capture shot vectors
support samples only; capture does not support shot vectors or QNode
differentiation. Native device gradients and variance gradients are unsupported.
Parameter broadcasting requires an explicit supported transform. Compiled counts
and probabilities allocate $2^{n}$ entries for $n$ measured wires; use samples
for large registers. Catalyst-level random seeds are unsupported; use
provider-specific job parameters where available. See
[Catalyst's limitations](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/sharp_bits.html).

## Recovering results after a failure

For ordinary QNodes, the device keeps accepted jobs and successful results after
a failure. Compiled QNodes do not use this recovery or automatic retry
interface.

Retrieve the batch handle from `error.job` on a
{py:class}`~mqt.core.plugins.pennylane.exceptions.PennyLaneExecutionError`, or
from `device.last_job` after an interruption inside `device.execute()`. Each
`submit()` or `execute()` call clears this attribute until its batch is
prepared. QNode preprocessing happens before `execute()` and does not update it.

To recover a QNode's measurement values, save its executable tapes and
postprocessor **before execution**. For the Bell-state QNode above, use
PennyLane's
[`construct_batch`](https://docs.pennylane.ai/en/stable/code/api/pennylane.workflow.construct_batch.html):

```{code-cell} ipython3
tapes, postprocess = qp.workflow.construct_batch(bell_state, level="device")()
job = bell_device.submit(tapes)
# Accepted QDMI handles are available before waiting for results.
job_ids = [attempt.handle.id for entry in job.entries for attempt in entry.attempts if attempt.handle is not None]
counts = postprocess(job.result())[0]
```

`submit()` accepts the same preprocessed tapes and execution configuration as
`execute()`, which submits and collects in one call. Job IDs identify provider
work; their availability does not guarantee retrieval after process termination.

If this execution stops, keep the same Python session. After resolving a
connection or result-read problem, recover the measurement values from the
retained jobs:

```{code-cell} ipython3
job = bell_device.last_job
assert job is not None  # A batch was prepared by submit().
entries = job.collect()
counts = postprocess(job.result())[0]
```

`collect()` reads accepted work without starting new executions; its entries
retain attempts, results, and errors. If needed, use `job.submit()` for
untouched inputs or `job.resubmit([i])` to replace a failed or cancelled
attempt, then retrieve the result again. Successful results are reused. See
{py:class}`~mqt.core.plugins.pennylane.job.PennyLaneJob` for uncertain
submissions and explicit cancellation.

This workflow recovers forward measurements, including broadcasts and shot
vectors. It does not restore arbitrary QNode return containers or resume an
interrupted gradient or optimizer.

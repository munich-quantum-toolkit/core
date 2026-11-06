---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  output_stderr: remove
---

# Qiskit Backend Integration

{py:class}`~mqt.core.plugins.qiskit.backend.QDMIBackend` exposes a QDMI device
as a Qiskit {py:class}`~qiskit.providers.BackendV2`. Compile circuits for that
backend before calling
{py:meth}`~mqt.core.plugins.qiskit.backend.QDMIBackend.run`.

## Installation

```console
uv pip install "mqt-core[qiskit]"
```

## Quickstart

```{code-cell} ipython3
from qiskit import QuantumCircuit
from mqt.core.plugins.qiskit import QDMIBackend

backend = QDMIBackend.from_device_id("mqt.ddsim.default")
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
native = backend.compile(qc)
job = backend.run(native, shots=1024, memory=True)
result = job.result()
print(f"Bell counts: {result.get_counts()}")
```

`run` validates, serializes, and submits; it does not synthesize or route. Bind
parameters before compilation when their values affect native legality. Use a
provider's specialized backend when its service requires one.

## Provider and Device Discovery

```{code-cell} ipython3
from mqt.core.plugins.qiskit import QDMIProvider

provider = QDMIProvider()
print([backend.name for backend in provider.backends()])
```

`provider.backends(name="DDSIM")` filters names. Open a known device directly
with `QDMIBackend.from_device_id`. For registry setup, see {doc}`configuration`.

## Authentication

`QDMIBackend.from_device_id` and `QDMIProvider.get_backend_by_device_id` accept
{py:class}`~mqt.core.typing.QDMISessionParameters`, including `base_url`,
`token`, and `auth_file`. The driver defines their meaning and precedence;
provider libraries may also use environment variables or credential stores.
Discovery opens fresh sessions without explicit authentication parameters.

## Device Capabilities and Target

`backend.target` projects MQT Core's device snapshot into Qiskit instructions,
ordered placements, parameter constraints, and calibration. Qubit indices remain
physical site IDs. Only sites with single-qubit gates and measurement are
placeable; disabled sites leave holes. Durations and coherence times use the
snapshot's timing unit and scale factor.

MQT Core resolves device aliases. Fixed parameter variants retain separate
Target names, and unrecognized native operations become opaque Qiskit gates.
Control flow and classical assignment follow the selected payload's
capabilities. All-to-all devices do not materialize a coupling map.

Devices with computational resonators expose single-qubit gates and measurement
in the Qiskit target. Use `backend.run(backend.compile(qc))` for circuits with
two-qubit gates; MQT Core inserts the required MOVE operations. Qiskit
`transpile` and primitives with two-qubit gates are unsupported on these
devices. See
[resonator routing](../mlir/target_compilation.md).

## Circuit Execution

All parameters must be bound before submission. `parameter_values` can bind
already-native parameterized circuits; it cannot make a non-native angle legal.
`shots` and `max_retries` must be nonnegative integers; `memory` is boolean. The
generic backend rejects other options, including non-`None` `seed_simulator`,
because QDMI has no standard seed parameter.

Pass a list of circuits to submit a batch. Equal-format, equal-shot circuits use
a native program list when the device supports it; otherwise they use separate
jobs. The backend validates the whole batch before submitting any circuit and
returns results in input order.

### Optional retries

Retries are disabled by default. `max_retries=3` retries confirmed failed jobs
up to three times and can incur additional charges. Successful results are
reused. Use `backend.set_options(...)` to change defaults for subsequent runs
and primitives.

## Job Handling

`job.status()` returns Qiskit's job status. `job.result()` waits and returns
counts, optional shot memory, and the original circuit/register metadata. The
adapter checks widths, binary digits, and shot totals. It never fabricates
samples from a histogram; when memory is requested, counts come from those same
shots.

Results use Qiskit's classical-bit order (`clbits[0]` on the right), preserving
leading zeros and correlations across registers. Classical registers must
partition `circuit.clbits` in register order; loose, aliased, and reordered bits
are rejected. Compiled programs retain a result layout that maps measured site
results to the input circuit's classical bits; see
[terminal readout](../mlir/target_compilation.md#terminal-readout-and-results).

### Recovering a failed batch

```python
from mqt.core.plugins.qiskit import JobExecutionError, JobSubmissionError

try:
    job = backend.run([native, native], shots=64)
    result = job.result()
except (JobSubmissionError, JobExecutionError) as error:
    job = error.job
    entries = job.collect()
```

After an interruption, `backend.last_job` retains the most recently prepared
batch. `collect()` reads accepted work without starting executions. A repeated
`result()` reuses successful results; `submit()` submits untouched inputs and
`resubmit([index])` replaces failed or cancelled attempts. Recovery currently
requires the same Python process. See
{py:class}`~mqt.core.plugins.qiskit.job.QDMIJob` for uncertain submissions and
cancellation.

## Qiskit Primitives

The factories return Qiskit's `BackendSamplerV2` and `BackendEstimatorV2` with
Qiskit's own options and PUB broadcasting. The Bell circuit above already uses
DDSIM's native gates:

```{code-cell} ipython3
from qiskit.quantum_info import SparsePauliOp

sampler = backend.sampler(default_shots=128)
samples = sampler.run([qc]).result()[0]
print(f"Sample counts: {samples.data.meas.get_counts()}")

circuit = qc.remove_final_measurements(inplace=False)
estimator = backend.estimator(default_precision=0.1)
estimate = estimator.run([(circuit, SparsePauliOp("ZZ"))]).result()[0]
print(f"Sampled ZZ expectation: {estimate.data.evs}")
```

Estimator also needs the basis rotations and measurements generated for its
observables. Apply the circuit's layout to logical observables after routing.
Sampler requires QDMI `SHOTS`; counts-only execution and sampled Estimator
require `HIST_KEYS` / `HIST_VALUES`. DDSIM supports both.

### Exact local expectations

{py:class}`~mqt.core.plugins.qiskit.estimator.DDEstimator` evaluates real sparse
Pauli sums with MQT Core's local DD engine. It imports each PUB once, then
prepares one state per parameter binding for all its broadcast observables
without materializing a dense state or observable matrix. This entry point does
not use a QDMI device; `backend.estimator()` remains the sampled estimator
above.

```{code-cell} ipython3
from mqt.core.plugins.qiskit import DDEstimator

exact = DDEstimator().run([(circuit, ["XX", "ZZ"])]).result()[0]
print(f"Exact XX and ZZ expectations: {exact.data.evs}")
```

Qiskit's PUB parameter ordering and observable/binding broadcasting apply. PUB
precision overrides run precision, which overrides `default_precision=0`. Every
finite nonnegative precision uses the same exact numerical calculation. Zero
`stds` means zero sampling uncertainty, not zero quantum variance or
floating-point error. Metadata records the requested precision and circuit
metadata without a shot count.

Eligible terminal measurements are deferred, so expectations describe the
uncollapsed state before readout. Resets, feedback and other nonterminal
measurements are rejected. For bound OpenQASM or existing compiler programs,
{py:func}`~mqt.core.mlir.expectation_values` exposes the same native calculation
with a sequence of mappings from Pauli strings to real coefficients. Labels span
all program wires; the rightmost character acts on wire zero.

## Compiled programs

The backend keeps one device environment and uses MQT Core's format selection.
`backend.compile(circuit)` returns a {py:class}`~mqt.core.mlir.CompiledProgram`;
`backend.run(backend.compile(circuit))` compiles and submits it. A sequence
returns an ordered list of programs. The lower-level
{py:func}`~mqt.core.mlir.compile_program` also accepts `target=backend`.

`run()` also accepts native `QuantumCircuit` inputs. It validates their gates,
parameters and placements, then uses {py:func}`~mqt.core.mlir.serialize_program`
without synthesis or routing. Compiled programs retain their exact payload and
result layout. Programs from another environment are checked for compatibility
before any job in the batch is submitted.

Payloads, result layouts and copied circuit headers survive retries without
reserialization. Persist all three for cross-process recovery; a remote job ID
alone cannot recover frontend outputs.

### Device-specific execution options

Backend subclasses declare options in `_default_options` and validate/encode
them in `_job_parameters(options)` as QDMI `custom1` through `custom5`. The hook
receives merged defaults and run overrides, excluding the base options `shots`,
`memory`, and `max_retries`. It runs before submission; unknown option names
fail. The base backend defines no provider-specific option names.

`backend.set_options(...)` also sets defaults for primitives. Sampler accepts
`run_options={...}`; Estimator uses backend defaults.

## API Reference

- {py:class}`~mqt.core.plugins.qiskit.provider.QDMIProvider`
- {py:class}`~mqt.core.plugins.qiskit.backend.QDMIBackend`
- {py:class}`~mqt.core.plugins.qiskit.job.QDMIJob`
- {py:mod}`~mqt.core.plugins.qiskit.exceptions`

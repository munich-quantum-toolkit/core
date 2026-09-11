---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Run programs through QDMI

The compiler decides how a program fits a device. QDMI provides the next part of
the workflow: discover that device, submit a job, wait for completion, and read
its results. How do individual shots relate to a histogram? When can a simulator
also return a statevector? Can you reuse a compiled program?

This notebook answers those questions with local DDSIM. It runs independently
with the [tutorial setup](index.md#run-the-notebooks), without credentials or an
external device. The {doc}`QDMI guide <../qdmi/index>` explains the driver,
client interfaces, device implementations, and SDK integrations.

{download}`Download this notebook <../_build/jupyter_execute/tutorials/qdmi_execution.ipynb>`.

```{code-cell} ipython3
:tags: [hide-input]
from collections import Counter
from math import cos, pi, sin

import numpy as np
from qiskit.visualization import plot_distribution

from mqt.core.mlir import compile_program, submit_program
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device, registered_device_ids
```

## Discover a device and inspect its capabilities

Device IDs identify configured device definitions. Listing them does not submit
a program. The bundled simulator has the stable ID `mqt.ddsim.default`:

```{code-cell} ipython3
print("Registered devices:", registered_device_ids())
device = open_device("mqt.ddsim.default")
print("Device:", device.name())
print("Status:", device.status().name)
formats = device.supported_program_formats()
print("Accepted formats:", [program_format.name for program_format in formats])
```

The accepted formats tell the client how to encode a program. Device queries
also expose available operations, sites, and calibration properties. The target
compiler uses that information; callers do not need to reproduce its gate
mapping logic.

Opening another registered device may initialize access to a provider. Keep
DDSIM selected here; use the {doc}`configuration guide <../qdmi/configuration>`
when connecting an external library or service.

## Compile a program with a predictable distribution

Choose an angle $\theta$. Applying $R_y(\theta)$ to qubit 0 and a controlled-X
to qubit 1 produces $\cos(\theta/2)|00\rangle + \sin(\theta/2)|11\rangle$. For
$\theta=\pi/3$, predict the probabilities of `00` and `11`.

```{code-cell} ipython3
theta = pi / 3
shots = 2048
seed = 17
source = f"""OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ry({theta}) q[0];
cx q[0], q[1];
bit[2] result = measure q;
"""

compiled = compile_program(source, target=device, program_format=ProgramFormat.QASM3)
print("Selected format:", compiled.program_format.name)
print(compiled.payload)
```

The returned `CompiledProgram` owns the payload and the target contract used to
produce it. We selected OpenQASM 3 explicitly so this example can also inspect
DDSIM's state before terminal measurements. Omitting `program_format` lets the
compiler choose from the device's supported formats.

## Submit, wait, and retrieve counts

Compilation does not submit a job. Name the destination explicitly when calling
`submit_program`:

```{code-cell} ipython3
job = submit_program(compiled, target=device, num_shots=shots, custom1=seed)
assert job.wait()
print("Final job status:", job.check().name)
counts = job.get_counts()
assert sum(counts.values()) == shots
assert set(counts) == {"00", "11"}
print(counts)
```

`job.wait()` waits for completion and reports a failed execution. A finite
`timeout` lets a client stop waiting without cancelling the job. DDSIM uses
`custom1` as a positive integer random seed; custom parameters are defined by
each device, not by the generic QDMI interface.

Compare the sampled distribution with the analytic prediction:

```{code-cell} ipython3
expected = {"00": cos(theta / 2) ** 2, "11": sin(theta / 2) ** 2}
assert abs(counts["00"] / shots - expected["00"]) < 0.05
plot_distribution([expected, counts], legend=["Analytic probabilities", "Sampled counts"])
```

:::{dropdown} Explain the result
The probabilities are $3/4$ for `00` and $1/4$ for `11`. Finite-shot counts
fluctuate around those values. Increasing the shot count reduces sampling noise;
it does not change the underlying state.
:::

## Recover individual shots

A histogram loses ordering. `get_shots()` returns the individual outcomes in
sampling order, and counting them must reconstruct the same histogram:

```{code-cell} ipython3
samples = job.get_shots()
assert len(samples) == shots
assert dict(Counter(samples)) == counts
assert job.get_counts() == counts
print("First 16 shots:", samples[:16])
```

Reading results again does not rerun the program. QDMI bitstrings put the
highest-index output bit first. The {doc}`QIR tutorial <qir_execution>` uses
asymmetric outputs to make that ordering visible.

## Inspect the state behind terminal measurements

DDSIM can prepare this program's state once and sample it repeatedly. It retains
that uncollapsed state, so the same job can also provide a statevector and
probabilities:

```{code-cell} ipython3
state = job.get_dense_statevector()
probabilities = job.get_dense_probabilities()
expected_state = np.array([cos(theta / 2), 0, 0, sin(theta / 2)], dtype=complex)
np.testing.assert_allclose(state, expected_state, rtol=0, atol=1e-12)
np.testing.assert_allclose(probabilities, np.abs(expected_state) ** 2, rtol=0, atol=1e-12)
print("Statevector:", state)
print("Probabilities:", probabilities)
```

These are simulator results, not measurements available from physical hardware.
They describe the state before terminal measurement, not one randomly collapsed
shot. Jobs with feedback, resets, or other ineligible execution paths do not
provide this state. The {doc}`DDSIM guide <../qdmi/ddsim_device>` specifies
which sampling and zero-shot extraction jobs support it.

## Reuse the compiled program

The number of shots is a job parameter. Submit the same compiled artifact again
with a smaller shot count; no recompilation is needed while the target contract
still matches:

```{code-cell} ipython3
small_job = submit_program(compiled, target=device, num_shots=128, custom1=seed)
assert small_job.wait()
assert sum(small_job.get_counts().values()) == 128
assert job.get_counts() == counts
plot_distribution([small_job.get_counts(), counts], legend=["128 shots", "2048 shots"])
```

**Experiment:** change `theta` to `pi / 2`, rerun from the source cell, and
predict the new probabilities. Changing the circuit requires recompilation;
changing only the shot count requires another submission.

## Change the program format, keep the job API

DDSIM also accepts QIR. Recompile the same source as Base Profile bitcode and
use the same submission and result methods:

```{code-cell} ipython3
qir_compiled = compile_program(source, target=device, program_format=ProgramFormat.QIR_BASE_MODULE)
assert isinstance(qir_compiled.payload, bytes)
qir_job = submit_program(qir_compiled, target=device, num_shots=shots, custom1=seed)
assert qir_job.wait()
qir_counts = qir_job.get_counts()
assert sum(qir_counts.values()) == shots
assert abs(qir_counts.get("00", 0) / shots - expected["00"]) < 0.05
plot_distribution([counts, qir_counts], legend=["OpenQASM 3", "QIR Base"])
```

The payload changed from text to bytes, but the QDMI job workflow stayed the
same. Different execution paths can use different random sequences; compare the
semantics and distribution rather than requiring identical counts.

Use {doc}`Qiskit <../qdmi/qdmi_backend>` or
{doc}`PennyLane <../qdmi/pennylane_device>` to access QDMI through an SDK, and
{doc}`Slurm <../qdmi/slurm>` to connect jobs to an HPC allocation. A compiled
artifact must still match the selected device's capabilities; a stable ID does
not make artifacts portable to an unrelated target.

Continue with {doc}`jeff_exchange` to exchange a structured program before
choosing its final execution format or device.

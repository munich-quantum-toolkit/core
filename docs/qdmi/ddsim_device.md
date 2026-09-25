---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# MQT Core DD-based Simulator QDMI Device

DDSIM executes quantum programs locally through QDMI using
[decision diagrams](../dd_package.md).

## Capabilities

The simulator device accepts OpenQASM 2, OpenQASM 3, and textual or binary QIR
programs using the Base or Adaptive Profile. See the
{doc}`OpenQASM support table <../mlir/OpenQASM>` and
[QIR Support in the MQT](../qir/index.md) for the supported operations, exact
QDMI program formats, and payload contracts.

The device can perform weak simulation for every supported format, i.e., sample
from the distribution produced by the program. It can also perform strong
simulation for OpenQASM and eligible QIR Base or Adaptive Profile programs,
i.e., compute a representation of the full state vector. Set the
`QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM` parameter to the desired number of shots,
or to `0` to request only the state. In Python, use `num_shots`.

Sampling jobs also retain an uncollapsed state when the existing terminal
sampling path prepares one state for all shots. Such jobs provide statevector
and probability results alongside shots and counts. Dense and sparse vectors are
materialized only when queried; queries do not rerun simulation or change the
samples. Jobs that execute separately for each shot do not expose their last
trajectory as a statevector and return `QDMI_ERROR_NOTSUPPORTED` for state
queries. Zero-shot extraction remains useful for eligible programs outside the
terminal-sampling fast path.

State extraction defers terminal measurements without collapsing the returned
state. Measurement-dependent computation, resets and subsequent operations on
measured wires are unsupported. Adaptive QIR may still execute classical loops,
branches, dynamic allocations and direct helpers; see the
[QIR extraction contract](../qir/index.md) for result-use and lifetime rules.

For reproducible stochastic execution, set `QDMI_DEVICE_JOB_PARAMETER_CUSTOM1`
to a positive `int` seed. The Python API exposes the same parameter as
`custom1`. If `custom1` is absent, the device seeds the random-number generator
from the system. The seed controls OpenQASM and QIR sampling. State extraction
does not use this seed.

Under the hood, the QDMI device imports OpenQASM into the compiler's QC
representation, lowers it to QCO, and executes it with the QCO DD utilities.
This is the same compiler-backed simulation path exposed by
{py:class}`~mqt.core.mlir.QCOProgram`.

Sampling returns ordered bitstrings through `QDMI_JOB_RESULT_SHOTS` and their
histogram through `QDMI_JOB_RESULT_HIST_KEYS` and `QDMI_JOB_RESULT_HIST_VALUES`.
Output bit zero is on the right, and leading zeros are preserved. Output
positions describe classical results, independently of physical qubit indices.
OpenQASM outputs follow declaration order, then increasing register indices; QIR
outputs follow recording order, including repeated recordings. Entirely binary
containers are flattened for these queries. Variable-length QIR outcomes remain
distinct histogram keys.

These three queries are available only when every selected output of every shot
is binary. Recording an integer or float, even zero or one, or producing an
undefined OpenQASM output makes all three return `QDMI_ERROR_NOTSUPPORTED` for
the job. Execution still succeeds, and complete output remains available.
Implicit final measurement is used only when a program has neither explicit
measurements nor program outputs.

OpenQASM 3 selects explicit outputs when present, otherwise global classical
variables, including constants. DDSIM returns their final values as a per-shot
JSON array through `QDMI_JOB_RESULT_QASM3_OUTPUT`, or `job.get_qasm3_output()`.
Bits are numbers, Booleans are JSON Booleans, and numeric values retain their
types. Registers retain increasing index order and undefined values are `null`.
The supported frontend subset currently provides scalar bit, Boolean, integer,
unsigned integer, float, and bit registers. Unsupported selected types,
including angles, are rejected. The current QIR exporter accepts bit-register
outputs; scalar OpenQASM outputs must be submitted as OpenQASM rather than
compiled to QIR. This restriction prevents treating output data as an
entry-point exit code. The Qiskit backend initializes every source classical bit
to preserve Qiskit's zero-initialized semantics.

For example, `bit[2] bits; bits[0] = 1; bool accepted = true; uint count = 1;`
produces `[{"bits": [1, null], "accepted": true, "count": 1}]` for one shot,
with binary queries unavailable. Fully initialized `bit[2] bits = "01";`
produces `[{"bits": [1, 0]}]` and the binary outcome `01`.

Base or Adaptive QIR programs with a static terminal measurement region can
sample one prepared DD; other QIR programs run once per shot. See the
[QIR execution contract](../qir/index.md) for eligibility and resource limits.
Full and binary results describe the same executions; querying results never
reruns a program.

Sparse statevector and probability results use ascending numerical basis-index
order. Their keys and values share that order. Sparse exports require at most 64
qubits on a 64-bit platform; wider states return `QDMI_ERROR_NOTSUPPORTED`
because their basis indices do not fit the sparse representation.

## QIR output capture

Set `QDMI_DEVICE_JOB_PARAMETER_CUSTOM2` to a `bool` value of `true` before
submission to capture textual QIR output. In Python, pass `custom2=True` to
`submit_job` or `submit_program`. This option requires a QIR Base or Adaptive
program and a positive shot count; other jobs reject it with
`QDMI_ERROR_NOTSUPPORTED`.

After successful execution, `QDMI_JOB_RESULT_QIR_OUTPUT` returns the complete,
null-terminated QIR output stream. Its reported size includes the terminator.
Python clients use `job.get_qir_output()`; C++ clients use `job.getQIROutput()`.
`QDMI_JOB_RESULT_CUSTOM1` remains an alias for existing clients.

DDSIM automatically captures the full stream when a program may record integers
or floats. For binary-only programs, `custom2=True` opts into full capture.
Without capture, the full-output query is unsupported (`None` in Python).

Capture retains the standard QIR header, per-shot boundaries, types, containers,
and labels in native recording order. It executes the program once per shot and
keeps the stream in memory. Binary queries remain available when the whole job's
output is binary; capture jobs provide no uncollapsed statevector. See the
[executable QIR example](../qir/index.md#retrieve-the-qir-output-stream-through-qdmi).

## Compile and execute

Compile a Bell circuit, sample its measurements, and inspect its statevector.
DDSIM retains the state before terminal measurements, so state extraction does
not require a separate job or zero shots. Circuits with mid-circuit measurements
or resets do not support state extraction.

```{code-cell} ipython3
from mqt.core.mlir import compile_program, submit_program
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device

bell_qasm = """OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
bit[2] result;
h q[0];
cx q[0], q[1];
result = measure q;
"""

device = open_device("mqt.ddsim.default")
program = compile_program(bell_qasm, target=device, program_format=ProgramFormat.QASM3)
job = submit_program(program, target=device)
job.wait()
print(job.get_counts())
print(job.get_dense_statevector())
```

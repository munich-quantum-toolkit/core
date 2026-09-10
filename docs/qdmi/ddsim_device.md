---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# MQT Core DD-based Simulator QDMI Device

## Objective

MQT Core provides a QDMI device that is powered by a classical quantum circuit
simulator based on decision diagrams (see
[the documentation of the DD Package](../dd_package.md)). This functionality is
exposed through the QDMI interface as a device, which can be used to classically
simulate quantum programs.

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
`QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM` parameter to the desired number of shots
for weak simulation or to `0` for strong simulation.

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

OpenQASM 3 output bits are undefined until written, so direct QDMI jobs with a
partially initialized output register fail during import. The Qiskit backend
preserves Qiskit's zero-initialized classical-bit semantics by writing every
classical bit before submitting its generated OpenQASM 3 program.

Sampling returns ordered bitstrings through `QDMI_JOB_RESULT_SHOTS` and their
histogram through `QDMI_JOB_RESULT_HIST_KEYS` and `QDMI_JOB_RESULT_HIST_VALUES`.
Both results come from the same samples, including mid-circuit measurements. QIR
Base or Adaptive programs with a static terminal measurement region can sample
one prepared DD; other QIR programs run once per shot. See the
[QIR execution contract](../qir/index.md) for eligibility and resource limits.
OpenQASM classical registers use reverse declaration order, with each register
most-significant-bit first. QIR samples follow the program's recorded outputs.

DDSIM advertises maximal compiler-supported language/profile capabilities using
MQT's versioned private `CUSTOM2` report. See the
[capability report](../mlir/target_compilation.md#private-qdmi-capability-report).

## Compile and execute

The compiler selects a supported payload and checks the device contract before
submission. This example selects OpenQASM 3 and retains the Bell state after
terminal sampling:

```{code-cell} ipython3
from mqt.core.mlir import compile_program
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device

bell_qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] result;
h q[0];
cx q[0], q[1];
result = measure q;
"""

device = open_device("mqt.ddsim.default")
program = compile_program(bell_qasm, target=device, program_format=ProgramFormat.QASM3)
job = device.submit(program, num_shots=1024, custom1=7)
job.wait()
counts = job.get_counts()
assert sum(counts.values()) == 1024
assert set(counts) <= {"00", "11"}
assert len(job.get_dense_statevector()) == 4
print(counts)
print("First eight shots:", job.get_shots()[:8])
```

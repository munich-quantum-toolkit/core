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
simulation for OpenQASM and QIR Base Profile programs, i.e., compute a
representation of the full state vector. Set the
`QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM` parameter to the desired number of shots
for weak simulation or to `0` for strong simulation. QIR Adaptive Profile
programs require at least one shot because their measurement-dependent control
flow cannot be represented by state extraction.

For OpenQASM state extraction, terminal output measurements are deferred. They
do not collapse the returned state. Mid-circuit measurements still execute and
may therefore collapse the state before subsequent operations.

For reproducible stochastic execution, set `QDMI_DEVICE_JOB_PARAMETER_CUSTOM1`
to a positive `int` seed. The Python API exposes the same parameter as
`custom1`. If `custom1` is absent, the device seeds the random-number generator
from the system. The seed controls OpenQASM and QIR sampling. During OpenQASM
state extraction, it also controls mid-circuit measurements and resets; QIR Base
state extraction does not use it.

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
Both results come from the same samples, including mid-circuit measurements.
OpenQASM classical registers use reverse declaration order, with each register
most-significant-bit first. QIR samples follow the program's recorded outputs.

## Compile and execute QIR

The compiler can snapshot the DDSIM device as an all-to-all target, compile a
program to QIR, and submit the resulting bitcode to the same device:

```python
from mqt.core.mlir import CompilerTarget, OutputFormat, compile_program
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device

device = open_device("mqt.ddsim.default")
target = CompilerTarget.from_device(device)
program = compile_program(
    "bell.qasm",
    target=target,
    output=OutputFormat.QIR_BASE,
)

job = device.submit_job(
    program.to_bitcode(),
    ProgramFormat.QIR21_BASE_BINARY,
    num_shots=1024,
    custom1=7,
)
job.wait()
print(job.get_counts())
print(job.get_shots())
```

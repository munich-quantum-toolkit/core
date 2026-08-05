# MQT Core DD-based Simulator QDMI Device

## Objective

MQT Core provides a QDMI device that is powered by a classical quantum circuit
simulator based on decision diagrams (see
[the documentation of the DD Package](../dd_package.md)). This functionality is
exposed through the QDMI interface as a device, which can be used to classically
simulate quantum programs.

## Capabilities

The simulator device supports all operations that our
[MQT Core IR](../mqt_core_ir.md) supports. It accepts OpenQASM 2, OpenQASM 3,
and textual or binary QIR programs using the Base or Adaptive Profile. See
[QIR Support in the MQT](../qir/index.md) for the exact QDMI program formats and
payload contracts.

The device can perform weak simulation for every supported format, i.e., sample
from the distribution produced by the program. It can also perform strong
simulation for OpenQASM and QIR Base Profile programs, i.e., compute a
representation of the full state vector. Set the
`QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM` parameter to the desired number of shots
for weak simulation or to `0` for strong simulation. QIR Adaptive Profile
programs require at least one shot because their measurement-dependent control
flow cannot be represented by state extraction.

Under the hood, the QDMI device uses the MQT Core OpenQASM parser (see
{cpp-api:func}`qasm3::Importer::imports`) to parse the program into a
{cpp-api:class}`qc::QuantumComputation` object. That circuit is then passed
either to the {cpp-api:func}`dd::sample` or {cpp-api:func}`dd::simulate`
function, depending on the mode. Consult the respective documentation for more
details and limitations.

The device implements the full QDMI job interface (except for the
`QDMI_JOB_RESULT_SHOTS` result format not supported by the simulator).

## Compile and execute QIR

The compiler can snapshot the DDSIM device as an all-to-all target, compile a
program to QIR, and submit the resulting bitcode to the same device:

```python
from mqt.core.fomac import ProgramFormat, open_device
from mqt.core.mlir import CompilerTarget, OutputFormat, compile_program

device = open_device("mqt.ddsim.default")
target = CompilerTarget.from_device(device)
program = compile_program(
    "bell.qasm",
    target=target,
    output=OutputFormat.QIR_BASE,
)

job = device.submit_job(
    program.to_bitcode(),
    ProgramFormat.QIR_BASE_MODULE,
    num_shots=1024,
)
job.wait()
print(job.get_counts())
```

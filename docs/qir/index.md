---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# QIR in the MQT

The [Quantum Intermediate Representation (QIR)](https://www.qir-alliance.org)
expresses quantum programs in [LLVM IR](https://llvm.org/). The MQT Compiler
Collection compiles OpenQASM, Qiskit circuits, and its own program objects to
QIR 2.1. The bundled DDSIM device executes the result through QDMI, with the
same job and result API used for OpenQASM.

This notebook follows that complete path: compile a program, inspect its profile
and capability flags, choose LLVM text or bitcode, and retrieve counts or QIR
output records. See {cite:p}`stadeTowardsSupportingQIR2025` for background on
QIR support in MQT.

Download {download}`this notebook <../_build/jupyter_execute/qir/index.ipynb>`
and the shared {download}`requirements.txt <../tutorials/requirements.txt>`.
Follow the {doc}`notebook setup <../tutorials/index>` to run it locally. All
examples use the bundled simulator and need no hardware account.

For experiments that compare static circuits with measurement feedback, continue
with the {doc}`QIR tutorial <../tutorials/qir_execution>`.

## Compile a Base Profile program

Use the Base Profile for a circuit whose measurements do not control subsequent
quantum operations. This Bell circuit produces the outcomes `00` and `11`.

```{code-cell} ipython3
from mqt.core.mlir import OutputFormat, QIRProfile, compile_program

bell_qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
bit[2] result = measure q;
"""

base = compile_program(bell_qasm, output=OutputFormat.QIR_BASE)
assert base.profile is QIRProfile.BASE
print(base.llvm_ir)
```

The returned {py:class}`~mqt.core.mlir.QIRProgram` owns the compiled program.
Its {py:attr}`~mqt.core.mlir.QIRProgram.llvm_ir` property is an LLVM assembly
string. The entry-point attributes identify `base_profile`, the output schema,
and the required static qubit and result capacities. The module flags identify
QIR 2.1 and whether dynamic resource management is used.

For a custom pipeline, a {py:class}`~mqt.core.mlir.QCProgram` also provides
`to_qir(QIRProfile.BASE)` and `to_qir(QIRProfile.ADAPTIVE)`. These conversions
consume that program unless `copy=True` is set. The `compile_program` function
copies program objects by default and runs the coordinated optimization
pipeline.

## Execute LLVM text or bitcode

The serialization and QDMI format must agree:

| Profile  | LLVM text (`str`)                   | LLVM bitcode (`bytes`)              |
| -------- | ----------------------------------- | ----------------------------------- |
| Base     | `ProgramFormat.QIR_BASE_STRING`     | `ProgramFormat.QIR_BASE_MODULE`     |
| Adaptive | `ProgramFormat.QIR_ADAPTIVE_STRING` | `ProgramFormat.QIR_ADAPTIVE_MODULE` |

Open DDSIM by its stable device ID and submit the Base program as text:

```{code-cell} ipython3
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device

device = open_device("mqt.ddsim.default")
job = device.submit_job(base.llvm_ir, ProgramFormat.QIR_BASE_STRING, num_shots=256, custom1=7)
assert job.wait()
counts = job.get_counts()
assert set(counts) <= {"00", "11"} and sum(counts.values()) == 256
counts
```

`custom1` is DDSIM's positive integer sampling seed. Use
{py:meth}`~mqt.core.mlir.QIRProgram.to_bitcode` to serialize the same program to
LLVM bitcode, then submit those bytes without an intermediate file:

```{code-cell} ipython3
bitcode = base.to_bitcode()
assert isinstance(bitcode, bytes)
binary_job = device.submit_job(bitcode, ProgramFormat.QIR_BASE_MODULE, num_shots=256, custom1=7)
assert binary_job.wait()
assert binary_job.program_bytes == bitcode
assert binary_job.get_counts() == counts
print(f"Bitcode: {len(bitcode)} bytes")
print(binary_job.get_counts())
```

To save a program, use `base.write_bitcode("bell.bc")` for bitcode or
`Path("bell.ll").write_text(base.llvm_ir)` for LLVM text after importing `Path`
from `pathlib`. The command-line equivalents are:

```console
mqt-cc bell.qasm --emit=qir-base -o bell.ll
mqt-cc bell.qasm --emit=qir-adaptive -o bell.bc
```

## Adaptive Profile and capability flags

The Adaptive Profile permits measurement-dependent control flow. This program
measures a superposition, then flips and measures the qubit again if the result
was `1`. It always returns `0`; the loop exits after at most one iteration.

```{code-cell} ipython3
feedback_qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit q;
h q;
bit result = measure q;
while (result) {
    x q;
    result = measure q;
}
"""

adaptive = compile_program(feedback_qasm, output=OutputFormat.QIR_ADAPTIVE)
assert adaptive.profile is QIRProfile.ADAPTIVE
```

The compiler derives capability flags from the lowered program. There is no need
to write LLVM metadata yourself. Inspect the entry-point attributes and flags:

```{code-cell} ipython3
:tags: [hide-input]

for line in adaptive.llvm_ir.splitlines():
    if line.startswith(("attributes #0", "!")):
        print(line)
assert '"qir_profiles"="adaptive_profile"' in adaptive.llvm_ir
assert '"backwards_branching"' in adaptive.llvm_ir
```

For this program, `backwards_branching` records the conditional loop, `arrays`
records the result array, and `dynamic_qubit_management` and
`dynamic_result_management` describe resource allocation. LLVM prints the
conditional-loop bit pattern as `i2 -2` (binary `10`). Other programs can also
carry `int_computations`, `float_computations`, `ir_functions`,
`multiple_target_branching`, and `multiple_return_points` flags when their
lowered instructions require those capabilities.

A profile name alone does not describe every device's capabilities. When
compiling for a device, use the
{doc}`target-compilation API <../mlir/target_compilation>` to select its
supported operations, program format, and control-flow capabilities. DDSIM
supports this Adaptive program in both encodings:

```{code-cell} ipython3
for payload, program_format in (
    (adaptive.llvm_ir, ProgramFormat.QIR_ADAPTIVE_STRING),
    (adaptive.to_bitcode(), ProgramFormat.QIR_ADAPTIVE_MODULE),
):
    adaptive_job = device.submit_job(payload, program_format, num_shots=32, custom1=7)
    assert adaptive_job.wait()
    assert adaptive_job.get_counts() == {"0": 32}
    print(program_format.name, adaptive_job.get_counts())
```

## From a Qiskit circuit to QIR

Pass a Qiskit `QuantumCircuit` directly to `compile_program`. After constructing
a circuit, conversion is one line; obtaining bitcode takes one more:

```{code-cell} ipython3
:tags: [hide-input]

from qiskit import QuantumCircuit

circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])
circuit.draw("mpl")
```

```{code-cell} ipython3
qir = compile_program(circuit, output=OutputFormat.QIR_BASE)
bitcode = qir.to_bitcode()
```

Select `OutputFormat.QIR_ADAPTIVE` for supported Qiskit control flow. The
{doc}`Qiskit interface guide <../mlir/qiskit>` describes the accepted operations
and circuits. Compilation preserves the input circuit.

When execution is the goal, let the compiler select a device-compatible format:

```{code-cell} ipython3
from mqt.core.mlir import submit_program

compiled = compile_program(circuit, target=device)
qiskit_job = submit_program(compiled, target=device, num_shots=256, custom1=7)
assert qiskit_job.wait()
assert set(qiskit_job.get_counts()) <= {"00", "11"}
print(compiled.program_format.name)
print(qiskit_job.get_counts())
```

## Retrieve the QIR output stream through QDMI

Counts summarize recorded measurement bits. QIR's textual output stream also
preserves output labels, array and tuple records, other recorded scalar values,
and shot framing. Enable capture with DDSIM's boolean `custom2` parameter, then
request its string result from `CustomProperty.CUSTOM1`:

```{code-cell} ipython3
from mqt.core.qdmi import CustomProperty

recorded_job = device.submit_job(
    adaptive.to_bitcode(), ProgramFormat.QIR_ADAPTIVE_MODULE, num_shots=2, custom1=7, custom2=True
)
assert recorded_job.wait()
output = recorded_job.get_custom_result(CustomProperty.CUSTOM1, str)
assert isinstance(output, str)
assert output.count("START\n") == 2 and output.count("END\t0\n") == 2
print(output, end="")
print("Counts:", recorded_job.get_counts())
assert recorded_job.get_counts() == {"0": 2}
```

The parameter and result slots are separate: **job parameter CUSTOM1** is the
seed; **job result CUSTOM1** contains the captured text. The same capture API
works for Base and Adaptive profiles, with text or bitcode input.

Capture is off by default. Enabling it executes the program for each shot and
retains the complete stream in memory, so start with a small shot count. The
stream uses QIR record order; QDMI shots and counts place the highest-index bit
first. Capture jobs still expose those normal results, but do not retain an
uncollapsed statevector. OpenQASM jobs and zero-shot state-extraction jobs
reject capture. See {doc}`DDSIM <../qdmi/ddsim_device>` for the C API parameter
types.

## Execution contracts

The following details matter when exchanging QIR with another compiler or
building on the C++ runtime. For ordinary compilation and execution, the Python
examples above handle serialization, runtime setup, and result retrieval.

### Runtime and QIS compatibility

QIR entry points take no arguments and return an `i64` exit code. Runtime and
QIS declarations are checked before JIT compilation; a mismatched or unsupported
declaration is reported with its actual and accepted LLVM function types.

MQT Core implements the QIR 2.1 Base and Adaptive Profile runtime APIs. The JIT
accepts one exact LLVM type for each runtime declaration, so unsupported or
outdated overloads fail before execution.

MQT Core provides dedicated QIS functions for variants with one or two control
qubits, using the `c<gate>` and `cc<gate>` names. Operations with three or more
controls use generic `__ctl` and `__ctladj` specializations. The control qubits
are passed in an Array; parameterized and multi-target gates pass their original
arguments in a Tuple, following the QIR-Runner calling convention. MQT accepts
these functions as implementation-specific extensions to the QIR 2.1 Base and
Adaptive profiles, so the entry point keeps its `base_profile` or
`adaptive_profile` attribute.

MQT's two-angle phased-X rotation gate uses the `prx` QIS stem. The incompatible
QIR-Runner Pauli-axis operation named `r` is not part of MQT's QIS.

### Payloads and result access

The QDMI Device accepts jobs in the following program formats: QASM2, QASM3, QIR
Base/Adaptive Profile Module (LLVM bitcode), and QIR Base/Adaptive Profile
String (LLVM assembly).

QDMI C++ applications submit textual programs through the
`Device::submitJob(const std::string&, ...)` overload, which includes the
terminating null byte required by QDMI. Binary module payloads use the
`Device::submitJob(std::span<const std::byte>, ...)` overload instead. It
preserves embedded null bytes and submits exactly the span's size without
appending a terminator. `Job::getProgramBytes()` retrieves such a payload
without interpreting its format or removing terminal null bytes; the existing
`Job::getProgram()` remains the textual, null-terminated accessor. It rejects
known binary and non-text formats based on their QDMI format identifier, even if
their payload happens to end in a null byte.

The Python API follows the same distinction: pass `str` to `Device.submit_job`
for a textual program and `bytes` for an exact binary payload.
`Job.program_bytes` always returns the unmodified payload, while `Job.program`
expects a null-terminated UTF-8 text payload and rejects known binary or
non-text formats. The `num_shots` argument is optional for device-defined
formats that encode their repetition count in the program payload.

Every DDSIM QIR job owns its JIT session, runtime, simulator state,
random-number generator, and output settings. QIR jobs can therefore execute
concurrently without sharing measurements or output records. DDSIM records
result bits directly and formats textual records only when capture is enabled.

### Sampling and state extraction

Sampling supports Base and Adaptive formats. With output capture disabled, for
either profile with an acyclic, unconditional entry path, constant gate
arguments, terminal Z measurements and scalar result records, DDSIM prepares the
DD once and samples it for all shots. The runtime retains repeated and reordered
result records in program order, including after SWAPs. The QDMI device reverses
each shot for most-significant-bit first serialization before constructing its
histogram. Programs with classical memory accesses, helper calls, conditional
branches, resets, dynamic resources or generic controlled argument arrays use
ordinary per-shot execution. These inputs remain supported by the runner; they
are not eligible for this sampling optimization. A fixed seed reproduces a shot
sequence for the same execution path; sequences need not match across different
sampling algorithms or software versions.

When provided for static resources, `required_num_qubits` and
`required_num_results` specify capacities, and out-of-range IDs are rejected.
Extracted states include unused qubits within the declared capacity, initialized
to zero. Programs without those attributes retain the runtime's inference of
static IDs or dynamic allocations. During sampling, dynamic qubit release resets
and recycles the simulator wire; the qubit limit applies to peak simultaneous
allocations rather than their cumulative number in a shot. Released handles
remain invalid.

Statevector extraction supports Base and Adaptive formats. Base extraction stops
before the first `irreversible` call and requires a terminal irreversible
region; defined or indirect helpers remain unsupported for that path.

Adaptive extraction executes classical loops, branches, dynamically computed
gate arguments, dynamic allocations and direct helper calls while deferring Z
measurements. A measured wire cannot participate in later gates, controls or
SWAPs; independent wires may still evolve. Resets and measurement-dependent
computation are unsupported. Result reads must be unused or feed only direct
boolean output records. Indirect calls and unknown external functions are
rejected before execution. Calls cannot re-enter the entry point.
Initialization, when present, must be the first instruction of the entry point.
These restrictions also apply inside helpers.

During Adaptive extraction, each executed allocation adds a zero-initialized
wire. Release calls mark lifetimes without resetting or recycling simulator
wires, so the exported state retains all allocated wires in allocation order,
including unused and released wires. The qubit limit therefore applies to all
allocations in one extraction run. Each run resets the quantum runtime. Output
records are suppressed, and unsupported operations produce a failed QDMI job.
Both profiles preserve global phase and logical wire order, including SWAPs.
LLVM target triples must match the host architecture and operating system
because the JIT executes in process.

The generic submission APIs reject QDMI calibration and batch-job formats. Use
{py:meth}`~mqt.core.qdmi.Device.submit_calibration_job` or
{cpp-api:func}`qdmi::Device::submitCalibrationJob` for calibration. These APIs
accept an optional provider-defined configuration payload and no shot count; the
payload is not an executable circuit. Batch jobs contain job handles rather than
serialized program bytes and require a separate typed API.

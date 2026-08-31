---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# QIR Support in the MQT

The [_Quantum Intermediate Representation_ (QIR)](https://www.qir-alliance.org)
is a standardized intermediate representation for quantum programs based on the
[_LLVM intermediate representation_ (LLVM IR)](http://llvm.org/).

## Compiling and Executing QIR

The MQT Compiler Collection generates QIR in LLVM assembly or bitcode form.
Execute this output with the DDSIM QDMI device or a compatible external QIR
runtime.

See {cite:p}`stadeTowardsSupportingQIR2025` for more details about QIR support
in MQT.

### Executing Generated QIR from Python

The [QIR-Runner](https://github.com/qir-alliance/qir-runner) project provides
the `qir-runner` command-line executable and the `qirrunner` Python package. The
Python package can execute statically allocated Base Profile bitcode without an
intermediate file. Install it with `uv pip install qirrunner`, then pass the
result of {py:meth}`~mqt.core.mlir.QIRProgram.to_bitcode` to `run_bytes`:

```{code-cell} ipython3
from qirrunner import OutputHandler, run_bytes

from mqt.core.mlir import OutputFormat, compile_program

bell_qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
ctrl @ x q[0], q[1];
bit[2] c = measure q;
"""

qir = compile_program(bell_qasm, output=OutputFormat.QIR_BASE)
output = OutputHandler()
run_bytes(qir.to_bitcode(), shots=4, rng_seed=7, output_fn=output.handle)

# Display the records produced for the first shot.
print(output.get_output().split("END", maxsplit=1)[0] + "END")
```

This path is tested for Base Profile programs with static qubit and result
allocation, including dedicated one- and two-control QIS functions and the
generic QIR controlled specialization used for three or more controls.
QIR-Runner does not currently implement every QIR 2.1 dynamic resource
management function supported by the DDSIM QDMI device. Submit dynamically
allocated programs to that device instead.

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

### QIR Support in the DDSIM QDMI Device

When {code}`BUILD_MQT_CORE_MLIR=ON`, the QDMI device accepts jobs in the
following program formats: QASM2, QASM3, QIR Base/Adaptive Profile Module (LLVM
bitcode), and QIR Base/Adaptive Profile String (LLVM assembly). With this option
set to {code}`OFF`, the device accepts QASM2 and QASM3 but not QIR.

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
random-number generator, and output sink. QIR jobs can therefore execute
concurrently without sharing measurements or interleaving runtime output.
Sampling supports Base and Adaptive formats. Statevector extraction is limited
to Base formats: the JIT stops the selected entry point immediately before the
first call to a function marked `irreversible`, following the semantic boundary
defined by the Base Profile. It rejects other profiles and Base Profile programs
whose irreversible region is not terminal.

The generic submission APIs intentionally reject QDMI calibration and batch-job
formats. Calibration jobs do not carry a program, while batch jobs contain job
handles rather than serialized program bytes. Their format identifiers remain
available for capability discovery; they require dedicated typed APIs.

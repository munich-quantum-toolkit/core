# QIR Support in the MQT

The [_Quantum Intermediate Representation_ (QIR)](https://www.qir-alliance.org)
is a standardized intermediate representation for quantum programs based on the
[_LLVM intermediate representation_ (LLVM IR)](http://llvm.org/).

## The QIR Runtime in MQT Core

MQT Core provides a runtime for QIR that is based on its decision diagram-based
quantum simulator. This allows for the execution of QIR programs using MQT
Core's high-performance simulation capabilities.

The runtime can be utilized in two ways:

1. As a standalone library that can be linked to any QIR program, resulting in a
   binary executable.
2. By using the `mqt-core-qir-runner` command-line tool, which interprets QIR
   programs directly.

See {cite:p}`stadeTowardsSupportingQIR2025` for more details.

### Building the Runner

To build this tool, the CMake option `BUILD_MQT_CORE_QIR_RUNNER` has to be
enabled. It follows `BUILD_MQT_CORE_MLIR` by default, but can be enabled
independently when the project already provides a compatible LLVM/MLIR
installation. From the root of the repository, you can build the runner as
follows:

```bash
cmake -S . -B build -DBUILD_MQT_CORE_QIR_RUNNER=ON -DBUILD_MQT_CORE_MLIR=ON
cmake --build build --target mqt-core-qir-runner
```

After building, the tool can be found in the build directory under
`bin/mqt-core-qir-runner`.

### Executing a QIR Program

The `mqt-core-qir-runner` can be used to execute a QIR file (typically with a
`.ll` extension).

```bash
./build/bin/mqt-core-qir-runner bell.ll
```

The runner prints the program's outputs to the console in one of the two
[QIR Output Schemas][output-schemas] (Labeled or Ordered): the two `HEADER`
records announce the schema, and each shot is wrapped in `START` and `END`
records with a `METADATA\toutput_labeling_schema\t<schema>` line inside.

The active schema is selected by the `output_labeling_schema` function attribute
on the entry-point function of the QIR program. The value `ordered` selects
Ordered; anything else, or a missing attribute, selects Labeled.

[output-schemas]: https://github.com/qir-alliance/qir-spec/tree/main/specification/output_schemas

### QIR Support in the DDSIM QDMI Device

The QDMI Device accepts jobs in the following program formats: QASM2, QASM3, QIR
Base/Adaptive Profile Module (LLVM bitcode), and QIR Base/Adaptive Profile
String (LLVM assembly). These QIR formats are only supported when the
`BUILD_MQT_CORE_QDMI_DDSIM_WITH_QIR` CMake option is enabled. It is enabled by
default when `BUILD_MQT_CORE_MLIR` is enabled, but can be selected
independently. This lets an embedding project reuse its existing LLVM/MLIR
installation for the QIR JIT without building MQT Core's compiler dialects.

FoMaC C++ applications submit textual programs through the
`Device::submitJob(const std::string&, ...)` overload, which includes the
terminating null byte required by QDMI. Binary module payloads use the
`Device::submitJob(std::span<const std::byte>, ...)` overload instead. It
preserves embedded null bytes and submits exactly the span's size without
appending a terminator. `Job::getProgramBytes()` retrieves such a payload
without interpreting its format or removing terminal null bytes; the existing
`Job::getProgram()` remains the textual, null-terminated accessor. It rejects
known binary and non-text formats based on their QDMI format identifier, even if
their payload happens to end in a null byte.

The `MQT::CoreFoMaC` CMake target advertises this API through the exported
`MQT_CORE_FOMAC_BINARY_PROGRAM_API` target property. The
`MQT::CoreQDMI_DDSIM_Device` target similarly reports whether it was built with
QIR support through `MQT_CORE_QDMI_DDSIM_WITH_QIR`. Embedding projects can use
these properties to reject an incompatible pre-existing MQT Core target during
configuration.

The Python API follows the same distinction: pass `str` to `Device.submit_job`
for a textual program and `bytes` for an exact binary payload.
`Job.program_bytes` always returns the unmodified payload, while `Job.program`
expects a null-terminated UTF-8 text payload and rejects known binary or
non-text formats.

The generic submission APIs intentionally reject QDMI calibration and batch-job
formats. Calibration jobs do not carry a program, while batch jobs contain job
handles rather than serialized program bytes. Their format identifiers remain
available for capability discovery; they require dedicated typed APIs.

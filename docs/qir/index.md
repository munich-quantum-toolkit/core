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

The runner is part of every MQT Core build. From the root of the repository, you
can build it as follows:

```bash
cmake -S . -B build
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
String (LLVM assembly).

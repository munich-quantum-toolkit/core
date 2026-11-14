# Share the compiler command for QPY export

## Design

The native executable and Python launcher call the same C++ compiler driver.
Python supplies an optional QPY writer using the existing Qiskit exporter and
serializer. The standalone executable remains independent of Python. The Python
entry point is private and runs once per process because LLVM owns global
command-line state and may exit for help or invalid options.

Compilation passes the QC program and target environment directly to the writer.
No subprocess, temporary MLIR file, or Python option parser is needed.
Serialization completes before opening the destination. QPY file export retains
phase and readout semantics; device submission uses its existing normalization.

Device contract inspection does not require a serializer. Executable format
selection does, and retains its supported-format fallback. These are separate
operations rather than a public mode flag.

## Validation

Exercise file and binary stdout output, explicit and discovered targets,
physical addresses, layout, aliases, symbolic parameters, phase, optional
Qiskit, informational commands, stdin, and output preservation on failure. Run
CLI tests in fresh processes to respect LLVM command-line ownership. Regenerate
bindings and run Python, native compiler, and lint checks.

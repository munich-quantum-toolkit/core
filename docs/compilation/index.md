# Compilation and execution

Start with a quantum program, compile it for a simulator or device, and inspect
its results. The MQT Compiler Collection (`mqt-cc`) accepts OpenQASM, Qiskit
circuits, and typed program objects. Python, command-line, and C++ interfaces
share its MLIR compiler infrastructure; see {cite:p}`MQTCompilerCollection2026`.

## Start with a result

The {doc}`QPE walkthrough <../getting_started>` compiles and executes phase
estimation with local DDSIM. It connects a structured benchmark, the compiler,
and QDMI in one example. No hardware account or MLIR knowledge is required.

Then work through the {doc}`compiler tutorials <../tutorials/index>` to inspect
representations, optimize gates, follow measurement feedback, and compile under
hardware constraints. Continue into QIR execution, QDMI jobs, and jeff exchange
between compilers. Each tutorial is an executable notebook with experiments to
change and rerun.

## Choose a workflow

- **Compile and inspect:** the
  {doc}`compiler guide <../mlir/mqt_compiler_collection>` covers the Python,
  command-line, and C++ interfaces, custom pipelines, and simulation.
- **Compile for a device:** the
  {doc}`target-compilation guide <../mlir/target_compilation>` uses device
  operations, connectivity, and supported payloads to produce a runnable
  program.
- **Exchange programs:** use {doc}`OpenQASM <../mlir/OpenQASM>` or
  {doc}`Qiskit <../mlir/qiskit>` at the frontend, or compile to
  {doc}`QIR <../qir/index>` and execute LLVM text or bitcode with DDSIM. Use
  {doc}`jeff <../jeff>` to exchange structured programs between compilers.

Continue with {doc}`QDMI devices and integrations <../qdmi/index>` to discover
devices, connect an SDK, or configure remote execution. For direct state and
operator manipulation, use the {doc}`decision-diagram package <../dd_package>`.
For repeatable workloads and reference results, use
{doc}`structured benchmarks <../benchmarks>`.

Compiler internals are documented in the {doc}`MLIR reference <../mlir/index>`
and the
[development policy](../development.md#mlir).

```{toctree}
:maxdepth: 1
:hidden:

../getting_started
../tutorials/index
../mlir/mqt_compiler_collection
../mlir/target_compilation
../qir/index
../jeff
../mlir/OpenQASM
../mlir/qiskit
```

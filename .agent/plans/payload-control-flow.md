# Payload control-flow ownership

Status: complete.

Payload legalization owns required static loop expansion for target compilation
and standalone Base QIR export. Its serialized `payload-spec` option takes
precedence over `mqt.target_env` and does not require hardware metadata. The
shared unroller promotes classical memory between expansion rounds so constant
array accesses can expose nested loop bounds. Its capture checks, integer
limits, and cumulative cloning budget remain in force.

QIR conversion owns residual legality checks. Adaptive preparation retains its
CFG support. Compiler-supported payload capabilities belong to
`PayloadSpecification`; QDMI translates its format and encoding into that model.
QIR optional capability IDs have one owner under `Target/QIR`.

Validation passed: compiler GoogleTests, Python compiler and parameter-binding
tests, all compiler CLI tests, generated MLIR reference docs, and repository/C++
lint. A direct Adaptive CFG export also passed. The bound-array regression
checks direct export and replay of a serialized module pipeline, including
explicit payload precedence.

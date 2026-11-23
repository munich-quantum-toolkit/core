---
tocdepth: 3
---

```{include} Dialects/MQTDialect.md

```

## C++ parameter helpers

The source-tree header `mqt/Dialect/MQT/Utils/Parameters.h` provides
`mlir::mqt::FloatParameter` and `mlir::mqt::FloatExpression` for C++ builders.
`FloatParameter` accepts either a host `double` or an MLIR `f64` SSA value, so a
builder can use the same interface for constants and values computed by the
program.

`FloatExpression` builds `f64` arithmetic and folds known constants through an
MLIR builder. The builder must outlive its expressions, and operands must
dominate its insertion point. These helpers handle arithmetic; QCO's rotation
utilities handle angle normalization and phase conventions.

## Compilation seed

`mqt.compilation_seed` is a signless `i64` module attribute used by the compiler
while running passes. Its bits encode an unsigned 64-bit seed and override
pass-local seeds, including passes on nested modules. An outer override takes
precedence. The driver restores the original attribute after compilation; crash
reproducers retain the effective value as input metadata.

## Staged target compilation

`mqt.target_compilation_original_env` is a module attribute of type
`#mqt.target_env`. `mqt-target-prepare` saves the execution environment while
passes use the routing target. `mqt-target-finalize` restores it on success and
removes this attribute. Explicitly attaching a target environment also removes
the saved value; see {doc}`target_compilation` for the pass-sequence
requirements.

## Passes

See {doc}`Transforms` for shared metadata and modifier passes.

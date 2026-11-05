---
tocdepth: 3
---

```{include} Dialects/MQTDialect.md

```

## Compilation seed

`mqt.compilation_seed` is a signless `i64` module attribute used by the compiler
while running passes. Its bits encode an unsigned 64-bit seed and override
pass-local seeds, including passes on nested modules. An outer override takes
precedence. The driver restores the original attribute after compilation; crash
reproducers retain the effective value as input metadata.

## Staged target compilation

`mqt.target_compilation_original_env` is a module attribute of type
`#mqt.target_env`. Target preparation saves the execution environment while
passes use the routing target. Successful finalization restores the saved
environment and removes this attribute. Explicitly attaching another target
environment removes the saved value; see {doc}`target_compilation`.

## Passes

See {doc}`Transforms` for shared metadata and modifier passes.

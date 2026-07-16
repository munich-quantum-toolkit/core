---
tocdepth: 3
---

The OQ3 dialect is an experimental typed semantic representation for OpenQASM 3
programs. It preserves OpenQASM-specific concepts, such as source-level gate
definitions and ordered modifiers, while ordinary classical computation and
structured control flow use standard MLIR dialects.

```{warning}
OQ3 is internal and experimental. Its textual representation is not a stable
public interface.
```

```{include} Dialects/OQ3Dialect.md

```

## Frontend stages

The OpenQASM frontend separates syntax, semantics, and MLIR construction. Its
LLVM-backed lexer reads source-manager buffers directly and records precise
source locations. A grammar-only parser builds persistent syntax, including
textual includes. Semantic analysis then resolves declarations, lexical scope,
types, definite initialization, broadcasting, gate visibility, and inclusive
range behavior without requiring an MLIR context.

Only a successfully analyzed program can be emitted. Classical expressions use
the `arith` and `math` dialects, structured `if`, `for`, and `while` statements
use `scf`, and mutable values cross regions through explicit operands, results,
and yields. Dynamically indexed qubits are dispatched among canonical QC qubit
references with structured control flow, avoiding additional references whose
aliasing would be invisible to later quantum conversions.

OQ3 is intentionally small. It retains resolved gate declarations, applications,
and ordered modifiers because these are source semantics that a particular
quantum target may not support. It does not duplicate MLIR's classical types,
arithmetic, storage, functions, or structured control flow.

## QC target boundary

The `oq3-to-qc` conversion expands reachable custom gates and rejects reachable
cycles or programs whose expansion exceeds its deterministic module-wide budget.
Its conversion target marks OQ3 illegal, so a successful conversion contains no
residual OQ3 operations.

Some valid OQ3 programs deliberately fail at this target boundary. QC does not
provide general power semantics, and it cannot faithfully apply inverse or
control modifiers to a custom gate whose body contains structured control flow.
These programs still parse, analyze, and produce verified OQ3; the QC conversion
reports the unsupported target capability instead of silently changing the
program.

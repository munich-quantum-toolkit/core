# Experimental OpenQASM frontend

MQT Core contains an experimental, staged OpenQASM frontend. It parses source
with the OpenQASM 3.1 ANTLR grammar, performs semantic checks, and produces
typed `oq3` IR before target-specific lowering. The textual dialect and its C++
interfaces are experimental and carry no compatibility guarantee yet.

The frontend accepts an explicit `OPENQASM 3.1;` declaration or defaults
versionless source to 3.1. `OPENQASM 2.0;` selects the compatibility mode. Other
explicit versions, including 3.0, are rejected. `stdgates.inc` is loaded only
when requested in 3.1 mode, and `qelib1.inc` is available only in 2.0
compatibility mode. Additional include directories can be supplied through
`OpenQASMTranslationOptions`.

## Current demonstrator boundary

| Semantic family | Typed OQ3 frontend | QC lowering |
| --- | --- | --- |
| Version policy and source diagnostics | Supported | Not applicable |
| Qubit declarations and constant indexing | Supported | Already represented with QC allocation |
| Builtin and standard-library gate calls | Supported | Supported except target-specific `cu` variants |
| Register broadcasting | Supported | Supported |
| Custom gate definitions and calls | Supported, including symbol verification | Supported through delayed inlining |
| Ordered `inv`, `ctrl`, `negctrl`, and `pow` modifiers | Preserved as typed operands and attributes | `inv`, `ctrl`, and `negctrl` supported; `pow` depends on target support |
| Reset and barrier | Supported | Already represented with QC operations |
| Inclusive constant integer ranges | Supported as `oq3.for` | Supported with widened, comparison-driven `scf.while` |
| Inclusive dynamic integer ranges | Representable in `oq3.for` | Rejected unless nonzero can be proven |
| Classical declarations, expressions, and assignments | Bit registers plus scalar `bool`, `int`, `uint`, and floating-point declarations, constants, assignments, and common expressions supported | Builtin `arith` and `memref` operations; complete signed/unsigned semantics, casts, and operator coverage remain planned |
| Program inputs and outputs | Declared `bit[n]` inputs and outputs preserve source order and width | Lowered to width-matched builtin integer function arguments and results |
| Measurement and classical registers | OpenQASM 3.1 and 2.0 forms supported | Already represented with QC measurement and `memref` storage |
| `if` and `while` | Supported with storage-backed mutable state | Builtin `scf` operations |
| `switch`, `break`, and `continue` | Feature-named diagnostic | Planned |
| Arrays, aliases, and subroutines | Feature-named diagnostic | Planned |
| Timing, calibration, annotations, and pragmas | Feature-named diagnostic | Planned |

The table is intentionally conservative: parsing a grammar production does not
mean its semantics are silently accepted. Unsupported families produce a
diagnostic naming that family. The legacy importer remains the production path
and serves as a differential oracle until the staged frontend reaches parity.

## Dynamic range steps

A constant zero step is rejected during semantic analysis. A dynamic step is
valid typed OQ3 IR because source validity does not depend on a particular
target. The initial QC lowering refuses such a loop when it cannot prove the
step nonzero and emits
`dynamic range step cannot be proven nonzero for the selected target`. It never
treats zero as an empty range.

## Grammar provenance

The grammar is pinned to the OpenQASM 3.1.0 release. The generated parser is
compiled once in a private library; generated ANTLR implementation details are
not exposed through the public frontend header. The exact upstream revision and
regeneration command are recorded next to the grammar sources.

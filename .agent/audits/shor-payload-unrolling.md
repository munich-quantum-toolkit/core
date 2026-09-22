# Why target compilation unrolls Shor for OpenQASM

Status: the investigation led to the compiler work merged in #2546. Indexed
logical qubits and constant floating-point table lookups are supported. Mapped
physical-qubit indexing still requires specialization.

## Current boundary

OpenQASM 3 permits [counted loops][control] and indexed logical registers, but
physical qubits cannot be aliased into a register. A logical `q[i]` and a
runtime choice among placed `$7`, `$19`, and `$42` therefore have different
export requirements.

`TargetEnvironment::supportsIndexedQubits` permits indexed placement for
Adaptive QIR on all-to-all targets without site-specific operation restrictions.
The payload legalization pass specializes statically bounded quantum loops when
indexed placement is unavailable. Its 65,536 cloned-operation budget bounds
compilation size; it is separate from the DD interpreter's conditional-loop
execution budget.

The OpenQASM exporter preserves logical register indexing and lowers constant
floating-point tables through selection. Mapped programs instead refer to
physical sites. Supporting runtime selection among those sites would require
extra dispatch or a target contract that explicitly permits deferred placement.

PR #2546 retained the existing bounded specialization instead of adding physical
reference dispatch. Silently discarding placement or emitting physical-register
aliases would be incorrect.

## Shor acceptance

Adaptive QIR can retain Shor's arithmetic loops, references, and tables. The
benchmark acceptance factors 21 through Adaptive QIR and 15 through mapped
OpenQASM 3. Exact Shor 21 exceeds the OpenQASM expansion budget. These checks
passed after #2546; results are recorded in the [Shor plan](../plans/shor.md).

DDSIM also accepts raw OpenQASM logical-register loops through direct
submission; that importer path does not imply that a mapped physical-register
program can be exported with the same loops. No benchmark-specific pass pipeline
or larger unrolling budget is introduced.

[control]: https://openqasm.com/versions/3.1/language/classical.html#looping-and-branching

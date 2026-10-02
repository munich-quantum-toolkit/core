# OpenQASM dynamic loop bounds

Status: complete.

## Goal and scope

`TranslateQCToOpenQASM3.cpp` exports signed `scf.for` loops with dynamic bounds
and positive constant steps in the entry function. Bounds support `index` and
integers of at most 64 bits. Empty loops preserve initial scalar values. Dynamic
steps, unsigned comparisons, dynamic gate-function bounds, and dynamic qubit
indices remain unsupported.

`OpenQASMToQCEmitter.cpp` uses 64-bit arithmetic for dynamic ranges with signed
bounds, positive signed constant steps, and no loop jumps. Other range forms
retain their existing lowering. No supported endpoint values were removed.

## Decisions

Reuse scalar expressions and loop-state emission. Snapshot bounds and guard
nonempty ranges before subtracting one from the exclusive upper bound. Gate
functions cannot contain the required declarations and conditionals in the
supported frontend subset, so their bounds must remain constant.

Use the unsigned remaining distance to decide whether another iteration exists.
This distance fits in 64 bits even when signed endpoints straddle zero. A final
increment may wrap, but the false continuation flag prevents its use in another
iteration. This avoids unchecked assumptions about small endpoint values.

Route index arithmetic through the existing integer expression emitter. In
particular, dynamic shifts need unsigned shift distances for OpenQASM import.

## Validation

The release QC translation binary passes 203 tests. The OpenQASM frontend binary
passes 185 tests. New round-trip tests use the existing DD simulator to check
loop counts, simultaneous scalar updates, bound snapshots, empty ranges, and
signed integer limits. The common dynamic importer path is checked for the
absence of wider integer arithmetic.

`uvx nox -s lint` passes. Full-file C++ lint passes on all three changed C++
files with zero findings, using the nox session's compilation database and
settings with explicit file selection. `git diff --check` passes.

# Resonator routing ownership

Status: complete.

## Outcome and decisions

`Pipeline.cpp` schedules QCO-to-QC conversion and `mqt-lower-resonator-cz`.
`TargetCompilation.cpp` owns target reduction and environment preparation.
`ResonatorRouting.cpp` owns virtual CZ route discovery and QC lowering; both
phases use the same route discovery. The private header adds no installed API.

The pass uses MLIR's C++ pass interface, like staged target preparation and
finalization. It has no options or statistics. Its factory stays in the compiler
library; dialect pass registration must not depend on this higher layer. A
separate compiler TableGen group would add generation infrastructure without
changing the pass's functionality.

Placement eligibility, deterministic route selection, native CZ handling, MOVE
operand order, and terminal-readout scheduling retain their behavior. QCO and
jeff checkpoints retain virtual CZs. Native serialization remains separate from
compilation.

The target-compilation guide separates placement policy from resonator
compilation. The glossary defines target sites, placeable sites, computational
resonators, and MOVE. The guide explains the effect of routing on available
qubit capacity and native output. Pass-level matching and failure behavior stay
in the private pass header; `Transforms.md` collects generated documentation.

## Validation

- Configured the release preset and built the compiler unit-test target.
- All 21 resonator tests passed, including direct textual-pass use.
- The full compiler suite passed 431 tests; two tests require `/dev/full` and
  were skipped on macOS.
- C++ lint checked every line of the eligible changed C++ files and passed.
- Repository lint passed.
- Executable HTML documentation and internal documentation links passed.

No hardware execution or hosted CI was run.

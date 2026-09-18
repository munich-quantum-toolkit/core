# Native trapped-ion gate targets

Status: in progress; implementation and validation remain.

## Goal and scope

Accept GPI, GPI2, MS, and ZZ directly in compiler targets, and emit those gates
from native synthesis. Core owns their definitions, phase conventions, and
parameter units. Parameters use turns, following the matrices in the
[IonQ native-gate specification](https://docs.ionq.com/features/getting-started-with-native-gates).
The existing fixed-parameter target API describes fully entangling MS and ZZ
capabilities when needed.

## Decisions

Use the existing QC/QCO operation traits and central gate registry for dialect
conversion, matrix simulation, and QIR. Preserve these gates through target
compilation. Exporters that need a gate definition use ordinary rotation gates
to define the same unitary. Do not add a provider SDK dependency.

Single-qubit synthesis uses a ZYZ decomposition followed by GPI2, GPI, GPI2. If
only GPI2 is available, replace GPI with two GPI2 pulses and the required global
phase. Two-qubit synthesis reuses the existing RXX/RZZ decomposers with fully
entangling MS(0, 0, 1/4) or ZZ(1/4). Broader angle-domain restrictions and
calibration-aware pulse optimization are outside this change.

## Work remaining

- [ ] Add gate semantics and conversion/export support.
- [ ] Add target recognition and constant/symbolic single-qubit synthesis.
- [ ] Test matrices, phase, physical placements, fixed entanglers, and exports.
- [ ] Expose target gate kinds and regenerate stubs.
- [ ] Update Bench to use the direct targets and fixed Rigetti rotations.
- [ ] Run required checks and open the separate draft contribution.

## Validation

No native-gate checks have run yet. Use independent matrix definitions to test
the gate conventions, full-unitary comparisons for synthesis, and round-trip
tests for parameters and exports. Downstream tests must check emitted target
instructions and qubit placements.

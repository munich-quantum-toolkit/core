# Native trapped-ion gate targets

Status: complete.

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

Circuit import recognizes the canonical exported definitions structurally,
including their phase and operand order. Cache successful matches within each
reader. Other definitions retain their custom semantics.

Single-qubit synthesis uses a ZYZ decomposition followed by GPI2, GPI, GPI2. If
only GPI2 is available, replace GPI with two GPI2 pulses and the required global
phase. Two-qubit synthesis reuses the existing RXX/RZZ decomposers with fully
entangling MS(0, 0, 1/4) or ZZ(1/4). Broader angle-domain restrictions and
calibration-aware pulse optimization are outside this change.

## Validation

The compiler suite passed 236 tests; native synthesis passed 66 tests; QCO IR
passed 573 tests; QC translation passed 213 tests; the QIR runtime passed 81
tests. Python target and export tests passed 424 cases; circuit translation
passed 419 cases. These cover numeric and symbolic native round trips and
changed custom definitions. Downstream checks also cover native gate
preservation and one-qubit circuits with wider target operations. Generated
stubs, repository lint, and full changed-file C++ lint passed.

## Follow-up

Bench consumes these capabilities in a separate adapter update. Symbolic
multi-gate fusion retains the export limitation tracked by #2559.

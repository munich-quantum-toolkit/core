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
reader. Other definitions retain their custom semantics. Native export reuses
the custom-gate path so controls, inverses, and powers keep their definitions.

Single-qubit synthesis uses a ZYZ decomposition followed by GPI2, GPI, GPI2. If
only GPI2 is available, replace GPI with two GPI2 pulses and the required global
phase. Numeric and symbolic paths share this pulse recipe. Two-qubit synthesis
reuses the existing RXX/RZZ decomposers with fully entangling MS(0, 0, 1/4) or
ZZ(1/4). Operation matching and basis selection share the fixed-parameter check.
Native inverse rewrites preserve symbolic parameters. GPI2 uses three pulses and
a global phase correction because adding half a turn can lose precision for
large phases. MS placement reversal exchanges both qubits and their phases;
native matching checks the reordered fixed values before emission. Broader
angle-domain restrictions and calibration-aware pulse optimization are outside
this change.

## Validation

The compiler suite passed 245 tests; native synthesis passed 82; QCO IR passed
577; mapping passed 125. All 533 Python MLIR tests passed. These cover numeric
and symbolic native round trips and inverses, large phases, fixed GPI2 phases,
and reversed MS placements with unequal phases and fixed-parameter checks. Stub
generation left the public API unchanged.

## Follow-up

Bench consumes these capabilities in a separate adapter update.

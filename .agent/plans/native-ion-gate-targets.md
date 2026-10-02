# Native trapped-ion gate targets

Status: complete.

## Goal and scope

Support Forte-style GPI/GPI2/RZZ targets, with MS as a secondary entangler. Use
radians for every parameter and reuse the existing RZZ operation. Core owns
matrices, phase conventions, synthesis, and target conformance. Provider
adapters own unit conversion, serialization, physical labels, and parameter
domains.

## Decisions

Use the existing QC/QCO operation traits and central gate registry for dialect
conversion, matrix simulation, and QIR. Preserve native gates through target
compilation. Exporters define GPI/GPI2/MS using ordinary rotations. GPI equals
`i R(pi, phi)`, including its global phase. Do not add a provider SDK
dependency.

Circuit import recognizes canonical exported definitions structurally, including
their phase and operand order. Other definitions retain their custom semantics,
including same-named gates whose parameters use turns. Native export reuses the
custom-gate path so controls, inverses, and powers retain their definitions.

Numeric and symbolic single-qubit synthesis share a GPI2/GPI/GPI2 recipe. When
only GPI2 is available, replace GPI with two GPI2 pulses and the required global
phase. This recipe is independent of the ZSXX RX-pulse configuration. Two-qubit
synthesis reuses the RZZ/RXX decomposers at RZZ(pi/2) or MS(0, 0, pi/2).
Operation matching and basis selection share the fixed-parameter check.

GPI2 inversion uses three pulses and a phase correction because adding pi can
lose precision for large phases. MS placement reversal exchanges its phases as
well as its qubits, and checks reordered fixed values before emission. Target
compilation requires a single-qubit synthesis basis on every site, even for
native inputs. General symbolic two-qubit synthesis remains unsupported.

## Validation

All 1,870 native tests passed across compiler, native synthesis, QCO IR,
decomposition, optimization, mapping, QIR runtime, and QC translation. All 676
Python MLIR and translation tests passed, including full Forte target
compilation, radian matrices, custom definitions using turns, symbolic round
trips and inverses, large phases, and reversed MS placements. Stubs were
regenerated. Whole-file C++ lint, repository lint, and executable documentation
with generated-page link checks passed.

## Follow-up

Bench consumes the radian API and maps provider ZZ to RZZ in a separate adapter
update. Provider submission and IQM feedforward are separate work.

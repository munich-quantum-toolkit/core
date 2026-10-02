# Native trapped-ion gate targets

Status: complete; implementation and local validation passed.

## Goal and scope

Support current Forte-style targets with radian GPI/GPI2 and existing RZZ. Core
owns matrices, phase, synthesis, and target conformance. Provider adapters own
unit conversion, serialization, physical labels, and parameter domains. Bench's
catalogue and compiler adapter are maintained in Bench #1027.

## Decisions

Retain first-class GPI/GPI2 through the existing QC/QCO operation traits and
central gate registry. Their one-parameter signatures and GPI's phase,
`GPI(phi) = i R(pi, phi)`, belong to the gate semantics. A fixed-parameter R
capability alone would require parameter projection and phase handling in each
consumer. No provider SDK is required.

Require both GPI and GPI2 for single-qubit synthesis. Every current consumer has
both. Numeric synthesis uses at most three pulses and shorter forms for common
rotations; symbolic synthesis uses GPI2/GPI/GPI2. The general GPI2 inverse uses
unchanged pulses and a phase correction, preserving fixed phases and large
symbolic inputs. Numeric fusion can shorten it when phases are free. Arbitrary
RZ remains native when a target advertises virtual Z. Two-qubit synthesis reuses
RZZ(pi/2), subject to target parameter restrictions.

MS has no current consumer in this catalogue. Retired Aria required IonQ's
three-parameter MS; AQT's interaction is already represented by RXX. Omit MS, a
GPI2-only basis, and general pulse search.

Qiskit circuit and target import share exact canonical-definition recognition,
including phase. Aliases retain target names, fixed parameters, and placements.
Other definitions retain their semantics, including same-named gates using
turns. Standard fixed RX aliases work in both compilers; Qiskit's U equivalence
uses ordinary named gates defined by one RX. CY is a supported native controlled
Pauli capability, without a new entangling synthesis basis.

OpenQASM and Qiskit exports define the pulses with ordinary rotations. jeff uses
its existing custom-gate representation. QIR uses the existing one-qubit,
one-parameter runtime extension path. Controls, inverses, and symbolic values
retain their full phase through these consumers.

## Validation

The compiler, native synthesis, QCO IR, decomposition, optimization, mapping,
QIR runtime, and QC translation tests check matrices, native conformance, short
pulse forms, and large/symbolic parameters. Python MLIR, Qiskit target, and
translation suites check import/export aliases and jeff round trips. Bench tests
cover both compilers, current target families, native and mapped compilation,
exact U recipes, mirrors, control flow, and QIR execution.

Final local check results are recorded in the PR descriptions. Hosted CI is a
separate publication check. See the
[joint audit](../audits/ion-targets-and-bench.md) for catalogue evidence and the
resolved boundary findings.

## Limitations

Target compilation requires a single-qubit synthesis basis at every site.
General symbolic two-qubit synthesis remains unsupported. Virtual RZ must be
absorbed into pulse phases by a submission layer for pulse-only hardware. IQM
experimental feedforward and cloud submission are separate work.

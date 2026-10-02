# Native target synthesis

Status: complete; implementation, independent review, and local checks passed.

## Scope and ownership

Represent native ion operations with existing R and RZZ gates. RZ plus fixed R
quarter turns reuse ZSXX synthesis and final native-gate lowering. Native half
turns provide the existing X shortcut. GPI/GPI2 are Qiskit target aliases, not
new IR operations: target import checks their exact radian definitions, and
export projects the phase parameter and corrects GPI's global phase. Circuit
import uses ordinary custom gate definitions. QIR, OpenQASM, and jeff continue
to use existing R operations and need no ion-specific support.

Virtual RZ is an explicit target capability. A provider accepting only
GPI/GPI2/ZZ must absorb it into gate phases before submission. Providers own
units, angle ranges, and hardware serialization. MS remains unnecessary for the
current Bench catalogue; AQT uses RXX, and retired Aria models are omitted.

## Synthesis

Use the existing Euler machinery for numeric and symbolic single-qubit gates.
For unrestricted parameterized entanglers, emit the existing Cartan factors with
native angles. Keep fixed-angle entangler synthesis for constrained targets.
Runtime RZZ uses a CX/RZ/CX decomposition when it is not native; runtime CP uses
one arbitrary native RZZ where supported. Native cost analysis must use the same
capabilities and gate counts as emission.

Bench uses a private standard-gate target and local equivalences for final
native alias lowering in both IonQ and Rigetti compilation. It preserves the
public target, phase, layout, and the Qiskit session equivalence library.

## Validation

Full matrices, symbolic binding, fixed and unrestricted entanglers, reversed
placements, native counts, cache separation, and Qiskit/compiled-jeff exchange
are regression-tested. Local suites passed 999 C++ compiler/synthesis tests and
871 Python MLIR/Qiskit tests. Bench passed 507 tests with the implementation and
358 on its minimum Qiskit 2.1.2 environment. Generated stubs, whole-file C++
lint, repository lint, executable docs, and generated-page links passed.

A fresh correctness and complexity review found no outstanding issues after
fixing unrestricted operand placement, capability-based Bench lowering, and
variable-angle matrix reconstruction. Phase-sensitive 2/4/6-qubit QFT probes
reduced RZZ counts from 2/12/30 to 1/6/15 against the preceding PR revision.
These counts describe compiler output, not hardware execution time.

Runtime lowering covers CP and RZZ; other symbolic two-qubit gates must already
be native. Provider submission and general fixed-angle synthesis remain outside
scope. Hosted CI results are recorded separately from local validation.

# Audit: native ion gates and the Bench compiler

Status: complete; fresh review findings addressed and local validation passed.
Scope: Core #2578 and Bench #1027. Hardware snapshot: 2026-10-02.

## Hardware scope

A read-only Braket census across its five device regions returned 29 distinct
ARNs. Metadata for all 26 QPUs identified six online gate QPUs, one online
analog QPU, and 19 retired QPUs. No quantum tasks were submitted.

| Online gate QPU               | Active qubits | Native metadata                |
| ----------------------------- | ------------: | ------------------------------ |
| IonQ Forte / Forte Enterprise |       36 each | GPI, GPI2, ZZ                  |
| Rigetti Cepheus               |           107 | RX, RZ, CZ                     |
| AQT IBEX                      |            12 | PRX, XX, RZ                    |
| IQM Garnet / Emerald          |       20 / 54 | PRX, CZ, experimental feedback |

The live gate devices support verbatim programs. Provider adapters must enforce
units, numeric domains, physical labels, measurement restrictions, and feature
flags. Numeric domains are not fully specified by the native-name whitelist.
Rigetti uses RX at +/-pi/2 and +/-pi. Cepheus physical labels are 0..107 except
8; Bench keeps an explicit dense-to-physical map and all 386 directed CZ edges.
IQM's ideal architecture models are labeled as such; Emerald's 90 model edges
include four beyond the live 86-edge snapshot. Aquila is an analog device and is
outside this gate compiler.

IonQ describes arbitrary virtual Z as phase propagation. Bench advertises RZ for
that capability; the submission layer absorbs it into GPI/GPI2 gate phases. This
compiler abstraction does not claim RZ is in Braket's literal IonQ whitelist.

Aria is retired; its IonQ MS gate has no remaining catalogue consumer. AQT's
physically named MS interaction is already RXX. Remove Aria and MS. Replace the
Braket Ankaa model with Cepheus; this does not claim Ankaa is unavailable on all
other services. Remove retired Falcon/Eagle and the obsolete Heron-133 model;
keep Heron-156, IQM architecture models, Quantinuum H2, and artificial logical
gate sets. Add AQT IBEX. Benchmark creation performs no cloud queries.

Sources: [IonQ Aria](https://www.ionq.com/quantum-systems/aria),
[IonQ native gates](https://docs.ionq.com/features/getting-started-with-native-gates),
[Braket gate submission](https://docs.aws.amazon.com/braket/latest/developerguide/braket-submit-tasks.html),
[Braket experimental capabilities](https://docs.aws.amazon.com/braket/latest/developerguide/braket-experimental-capabilities.html),
[IBM retirements](https://quantum.cloud.ibm.com/docs/en/guides/retired-qpus),
[IQM Spark](https://iqm.tech/products/iqm-spark/), and
[Quantinuum availability](https://docs.quantinuum.com/systems/support/system_reference.html).

## Implementation boundaries

Core uses existing R and RZZ operations throughout synthesis, IR, and runtime
consumers. Qiskit targets expose GPI/GPI2 through exact fixed-R definitions;
only that import/export boundary projects their phase parameter and accounts for
GPI's global phase. Arbitrary renamed custom aliases and provider-specific angle
units are not inferred. CY follows the existing controlled-Pauli target
contract.

RZ plus fixed RX, RY, or R quarter turns use ordinary ZSXX synthesis with direct
native emission. Native parameter constraints remain explicit. Parameterized
entangler synthesis reuses Weyl factors and native cost analysis; constrained
entanglers retain the fixed-angle path. Runtime Pauli rotations and controlled
phase use the shared Pauli decomposition routines with explicit global phase.

Bench delegates target import to Core and uses a private standard-gate target
for both IonQ and Rigetti Qiskit compilation. Final local equivalences preserve
native aliases without modifying the session library. Native compilation drops
physical topology; mapped compilation retains it. Layout and mirror semantics
remain consumer contracts.

Bench's base compiler requires Qiskit 2.1.2 or newer; Core and QIR integration
require Qiskit 2.5.x for the native C API bridge. The uv minimums group and Core
extra select these environments independently. No provider SDK, new native IR
gates, arbitrary fixed-angle solver, or MS support is required.

## Verification

Regressions cover phase-sensitive matrices, symbolic binding, fixed and
unrestricted target conformance, native gate counts, aliases, compiled jeff
exchange, placement, and mirrors. The fresh audit identified unrestricted
entangler placement, description-based Bench target selection, and matrix
reconstruction issues; all are fixed and covered. Its final correctness and
complexity passes found no outstanding issues. See the
[native target decision record](../plans/native-ion-gate-targets.md) for local
validation. Publication reports hosted CI separately.

## Synthesis complexity audit

The 2026-10-03 audit covers Euler and Weyl decomposition, Pauli lowering,
one-qubit fusion, target capability resolution, native costing, and the Qiskit
boundary. Findings are addressed:

- Use Pauli conjugation for runtime rotations on fixed Clifford bases. Remove
  the CX lowering prerequisite and numerical cost queries; iSWAP now needs two
  native gates. With native U available, CZ-based RZZ needs three local gates
  instead of five. Arbitrary-angle entanglers remain the one-gate path.
  SQRTISWAP retains an explicit four-gate fallback.
- Reuse numerical Euler synthesis for constant one-qubit fusion. Remove the
  duplicate numeric quaternion backend, scalar templates, and unreachable
  parameter-conversion failures. Runtime fusion retains scalar SSA arithmetic.
- Keep the CX matrix private to its numerical decomposer. Generalize fixed
  quarter-turn capabilities to RY through constant Euler-frame offsets; no
  fixed-angle solver or additional basis enum is needed.
- Compare equivalent U matrices in numerical merge tests instead of selecting
  one set of Euler coordinates. Full-circuit matrix checks retain phase
  coverage; identity cancellation requires no remaining gate.
- Incorporate #2649's explicit fusion policy, operation-scoped greedy rewrites,
  and shared traversal. Preserve controlled-body phase wires through the
  synthesis listener. Keep fusion policy out of the Euler header and retain the
  complete native basis at emission. The U-specific optimizer is no longer
  needed in target compilation.
- Omit phase-only U gates and redundant full turns with exact phase accounting.
  Select shorter equivalent Euler representatives without changing tolerances.
  Import regression coverage for independent wires, symbolic run boundaries,
  controlled U2 restoration, and phase-generated control wires. Retain Python
  integration coverage rather than making unrelated test reductions.
- Balance global-phase sums in the shared normalizer. Sequential accumulation
  made symbolic U-target exports exceed the parameter-depth limit at 20 qubits.
  QC/QCO depth and value checks plus Qiskit/jeff export regressions cover the
  correction. The 100-qubit U workload also exceeds the old 4,096-node budget;
  align it with the existing 16,384-node classical-expression budget and retain
  depth, size, and failure-without-mutation regression checks.
- Emit signed sums incrementally in the Qiskit adapter to avoid Qiskit 2.5
  repeatedly expanding balanced additions. Balance additive replay chains on
  import so exported circuits round-trip and remain bindable.

Core #2649 is subsumed by this implementation; #2578 owns resolution of #2614.
No additional dependencies or unresolved complexity findings remain.

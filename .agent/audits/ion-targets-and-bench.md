# Audit: native ion gates and the Bench compiler

Status: complete; findings addressed and local consumer checks passed. Scope:
Core #2578 and Bench #1027. Hardware snapshot: 2026-10-02.

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
that capability; the submission layer absorbs it into pulse phases. This
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

## Findings and disposition

- **Target semantics:** Bench's duplicate importer lost open controls and angle
  bounds. It now delegates identity, fixed values, aliases, and placements to
  Core's strict importer. Native compilation removes physical topology before
  import; mapped compilation retains it. Bound predicates are checked before
  copying native capabilities. Structural global-phase entries are omitted.
- **Incorrect equivalences:** IonQ's CX recipe and Rigetti's RX recipe were
  incorrect; both are deleted. Corrected U recipes preserve complete matrices,
  including global phase. Ordinary RZZ uses Qiskit's existing CX decomposition.
- **Canonical native gates:** Core now recognizes exact radian GPI/GPI2
  definitions in circuit and target import, including aliases. Same-named
  foreign definitions retain their semantics. Target-aware export preserves
  capability names. CY completes the existing controlled-Pauli capability set.
- **Consumer coverage:** GPI/GPI2 now round-trip through jeff's custom gates as
  well as Qiskit and OpenQASM. Local QIR Base/Adaptive and direct DD probes
  verify their runtime semantics.
- **Pulse count:** Numeric RX(pi/2), RX(pi), and RZ use one, one, and two native
  pulses where admitted. Fixed-phase and large symbolic inverses keep the safe
  fallback. No general pulse search or arbitrary fixed-angle synthesis is added.
- **Catalogue:** Remove obsolete models, add current Cepheus and AQT snapshots,
  preserve arbitrary virtual RZ, and remove unsupported hardware feedback.
  Device docs distinguish real snapshots from ideal architecture models.

## Complexity review

Remove MS-specific operations, matrices, placement reversal, runtime macros,
export definitions, and tests. Delete the GPI2-only basis and public callback
emission helper. Bench deletes recursive unit conversion, output restoration,
and Rigetti pulse subclasses. Standard RX target aliases plus two ordinary named
gates in the U equivalence work on Qiskit 2.5. Bench requires that version:
Qiskit 2.1 still inserts a broken post-layout pass at optimization level 3
despite an explicit layout method. No compatibility pass surgery, provider SDK,
backend framework, or general angle solver is needed.

SDK support alone does not establish a current hardware requirement. Qiskit,
PennyLane, CUDA-Q, and Azure contain provider-specific or legacy MS interfaces;
current Braket metadata and provider retirement notices determine this scope.
First-class GPI/GPI2 remain justified by their signature and phase across
controls, import/export, and runtime consumers.

## Verification

Regressions compare full matrices and emitted capability names/parameters,
including numeric and symbolic inputs, inverses, controls, multiple fixed RX
angles, placement, mirrors, and measurement wiring. Independent audit probes
covered phases up to 1e16, DD functionality, local QIR execution, and the exact
Cepheus graph. A fresh review found the native-topology, explicit-global-phase,
and jeff gaps above; focused regressions now cover all three. The final local
suite and documentation results are reported with publication.

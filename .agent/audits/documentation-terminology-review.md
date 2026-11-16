# Documentation terminology review

Status: all 12 terminology decisions accepted, applied, and locally validated.
Baseline: `2af9b2df72af1596efb7ef3951200631371663f5`. Date: 2026-11-09.

## Scope

The glossary mixed scientific terms, compiler terminology, API names, and local
implementation rules. This audit records the evidence and applied decisions.
Public identifiers, schema keys, and behavior are unchanged. Confidence concerns
the cited evidence, not whether a term is absent from all literature.

## Findings and decisions

### 1. Semiclassical QFT is defined as a one-qubit reuse algorithm

**Priority: high. Confidence: high.**

- **Uses:** `docs/glossary.md`, `docs/benchmarks.md`, `docs/getting_started.md`,
  and the QFT, QPE, and Shor benchmark API descriptions.
- **Evidence:** the glossary makes measurement, reset, and reuse of one qubit
  part of the definition.
  [Griffiths and Niu](https://arxiv.org/abs/quant-ph/9511007) define the
  semiclassical transformation through measurements and classical feed-forward
  replacing controlled quantum gates. Single-qubit recycling is an additional
  construction, as used in
  [Beauregard's order-finding circuit](https://arxiv.org/abs/quant-ph/0205095).
- **Resolution:** Separate the concept and implementation. The glossary now
  defines the measurement-and-feedback construction with its primary source. The
  QFT guide and C++/Python API descriptions separately explain Core's
  single-qubit implementation; QPE and Shor descriptions connect recycling with
  the semiclassical inverse QFT.

### 2. “GPI basis” hides the distinction between rotations and native gates

**Priority: high. Confidence: high for the distinction; medium for the name.**

- **Uses:** `docs/glossary.md`, native-synthesis descriptions in
  `mlir/include/mqt/Dialect/QCO/Transforms/Passes.td`, and
  `mlir/include/mqt/Compiler/Target.h`.
- **Evidence:**
  [IonQ's native-gate reference](https://docs.ionq.com/features/getting-started-with-native-gates)
  names GPi and GPi2. Core's definition describes equatorial rotations with
  fixed angles. Its native export also accounts for phase conventions. The
  reviewed source establishes the gate names, not “GPI basis” as a general
  scientific name for Core's synthesis category.
- **Resolution:** Distinguish rotations from native gates. The glossary uses
  “GPi/GPi2 gate set”; the target guide, synthesis pass reference, and target
  enum documentation explain Core's rotation representation. Source inspection
  of nativeRAlias and OpenQASM export confirms the GPi phase correction. The
  guide distinguishes radians from IonQ's API turns.

### 3. Generic floating-point concepts are defined as Core helper types

**Priority: high. Confidence: high.**

- **Uses:** glossary entries “floating-point parameter” and “floating-point
  expression”; `FloatParameter` and `FloatExpression` in
  `mlir/include/mqt/Dialect/MQT/Utils/` and quantum builders.
- **Evidence:** the definitions specify a host `double` or MLIR `f64` value and
  local builder folding. Those are the representations of Core helpers, not
  general definitions of floating-point parameters or expressions. A class name
  is not a scientific alias.
- **Resolution:** Move the entries to the API reference. The glossary entries
  were removed; Parameters.h documents FloatParameter and FloatExpression at
  their declarations. No public type names changed.

### 4. Static, dynamic, program, virtual, logical, and device qubits overlap

**Priority: high. Confidence: high for ambiguity; medium for preferred
wording.**

- **Uses:** `docs/glossary.md`, compiler and target guides, tutorials, QC/QCO
  operation descriptions, and `CompilerTarget` documentation.
- **Evidence:** the glossary says “dynamic” means known only at execution, but
  also includes fixed-size dynamic allocations. The guides use “logical” for
  unmapped circuit wires; fault-tolerant readers may understand encoded qubits.
  [Qiskit's circuit model](https://quantum.cloud.ibm.com/docs/en/api/qiskit/circuit)
  distinguishes virtual qubits from hardware qubits. Core additionally has
  static site identifiers and allocated quantum resources.
- **Resolution:** Use program qubits before hardware mapping. Guides, dialect
  descriptions, and connected API documentation distinguish program qubits,
  physical sites, and allocation. The static/dynamic definitions now identify
  the property being described, including fixed-size runtime allocation. Encoded
  logical qubits and logical classical outputs retain their distinct meanings.

### 5. API objects have been promoted to general glossary definitions

**Priority: medium. Confidence: high.**

- **Uses:** “compiled program”, “target environment”, “selected payload
  specification”, “operation capability”, “compiler target”, “compilation seed”,
  “placeable site”, and “C++ QDMI library” in the glossary; their owning APIs in
  `mlir/include/mqt/Compiler/` and `include/mqt-core/qdmi/`.
- **Evidence:** definitions specify Core classes, immutability, target pairing,
  seed precedence, or placement eligibility. These are useful public API
  concepts, but not universal meanings. “Compiled program”, for example,
  ordinarily covers more than Core's serialized `CompiledProgram` object.
- **Resolution:** Move all API-specific entries to their owning references.
  Removed the eight listed glossary entries. Their meanings remain in target
  compilation, the QDMI guide, and the owning API documentation; CompiledProgram
  and CompilerTarget descriptions now explicitly explain their stored data and
  relationship to a live device.

### 6. “Payload” covers two unrelated objects

**Priority: medium. Confidence: high.**

- **Uses:** glossary entries “payload”, “execution payload”, and “selected
  payload specification”; target, QIR, and jeff guides; compiler API strings.
- **Evidence:**
  [MLIR's Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/)
  uses payload IR for the IR being transformed. Core also uses payload for
  serialized program data submitted to a device. The unqualified glossary entry
  makes the former sound like the general meaning in every guide.
- **Resolution:** Use concrete workflow wording. Payload IR remains a sourced
  Transform dialect term; execution documentation uses program data, serialized
  programs, formats, or supported program features as appropriate. Public
  PayloadSpecification names, metadata keys, and example variables remain.

### 7. Gate counts use a project-specific counting convention

**Priority: medium. Confidence: high.**

- **Uses:** “static gate count” in the glossary; `QuantumProgramInfo`,
  `num_gates`, and the compiler inspection tutorial.
- **Evidence:** `docs/mlir/mqt_compiler_collection.md` and
  `mlir/lib/Compiler/ProgramInspection.cpp` include measurement/reset and count
  a modifier or call once.
  [Qiskit's circuit model](https://quantum.cloud.ibm.com/docs/en/api/qiskit/circuit)
  distinguishes unitary gates from measurements and resets. Counts also differ
  from runtime executions or counts with helper calls expanded.
- **Resolution:** Keep Core's static gate count. The glossary, inspection guide,
  and API identify the metric as Core-specific and spell out measurements,
  resets, calls, modifiers, barriers, and control-flow counting. The separate
  operation_counts metric retains its meaning.

### 8. Wire correspondence and linear semantics need separate definitions

**Priority: medium. Confidence: high for the distinction.**

- **Uses:** glossary entries “wire correspondence”, “linear semantics”, “linear
  ownership”, “reference semantics”, “value semantics”, and QTensor; dialect
  references and `WireOpInterface`.
- **Evidence:** [QSSA](https://arxiv.org/abs/2109.02409) provides a prior
  quantum SSA representation with explicit input/output relationships. Linear
  quantum typing and SSA are established techniques. Core's exactly-one-use
  rule, positional helper results, stable register slots, and SWAP
  interpretation are more specific requirements checked by its verifier.
- **Resolution:** Separate concepts and conventions. General reference/value and
  linear semantics stay in the glossary with quantum-language sources. Wire
  correspondence moved to WireOpInterface and the QCO/resource references, where
  exactly-one-use and positional requirements are explained separately.

### 9. Layout terminology mixes data, transformations, and conventions

**Priority: medium. Confidence: high.**

- **Uses:** “layout metadata”, “routing permutation”, and “target site” in the
  glossary; `docs/mlir/qiskit.md`, target compilation, and `ResultLayout`.
- **Evidence:** the glossary's routing permutation specifies one direction of a
  map. The compiler also carries source-to-site maps and readout maps with
  different domains. `ResultLayout` concerns classical results, not qubit
  placement. The Qiskit guide already documents `TranspileLayout` separately.
- **Resolution:** Define concepts globally and maps locally. The glossary
  explains placement/routing. Qiskit, QubitLayout, and the MQT dialect reference
  specify map domains and directions; ResultLayout explains the separate
  program-output-bit to raw-sample-bit map.

### 10. Benchmark entries combine ordinary vocabulary with a private schema

**Priority: medium. Confidence: high.**

- **Uses:** “benchmark instance”, “benchmark instance specification”, “benchmark
  manifest”, and the generic alias “manifest”; `docs/benchmarks.md` and
  `include/mqt-core/bench/JSON.hpp`.
- **Evidence:** these entries describe Core's resolved defaults, case IDs,
  reference objects, and definition versions. “Manifest” is also used for QDMI
  device discovery with a different schema.
- **Resolution:** Move entries to the benchmark reference. Removed the instance,
  instance-specification, and manifest glossary entries and their generic
  aliases. The benchmark guide now defines the objects together and explicitly
  distinguishes benchmark manifests from device-discovery manifests.

### 11. Algorithm definitions mix the general concept with one benchmark

**Priority: medium. Confidence: high for scope.**

- **Uses:** “magic state”, “magic-state distillation”, “iterative QPE”, “quantum
  fan-out”, and “modular multiplier” in the glossary and benchmark guide.
- **Evidence:** the magic-state entries append Core's ideal-input checks and
  retained state to their definitions;
  [Litinski](https://arxiv.org/abs/1905.06903) is the source for the chosen
  protocol. [Høyer and Špalek](https://arxiv.org/abs/quant-ph/0208043)
  distinguish the unbounded fan-out primitive from its implementation with
  bounded gates. “On zero targets” in the glossary can mean no targets rather
  than targets initialized to zero.
- **Resolution:** Separate sourced definitions from implementations. General
  definitions of magic states, distillation, iterative QPE, fan-out, and modular
  multiplication link to primary literature. The benchmark guide retains Core's
  circuit choices, ideal-input checks, limits, and reference results.

### 12. Compiler shorthand is overused outside compiler references

**Priority: medium. Confidence: high for readability; not evidence of
invention.**

- **Examples:** “coordinated pipeline”, “canonical terminal readout”, “logical
  zero holes”, “owning wrappers”, “preflight”, “closed input”, “structured
  benchmark”, “contract”, and “boundary”.
- **Uses:** compiler binding strings, target/QIR guides, QDMI guides, and
  benchmark descriptions. The revised descriptions explain the operation as
  preparing a program, ordering measurements, retaining a session, checking a
  batch, or binding program inputs.
- **Resolution:** Use plain language throughout documentation. The review covers
  guides, detailed API descriptions, generated pass and dialect descriptions,
  and the compiler diagram. Concrete actions replace unnecessary shorthand;
  precise upstream terms and public identifiers remain. Writing guidance now
  makes this scope and the glossary/API division explicit.

## Established terminology

Compiler terms follow the
[MLIR glossary](https://mlir.llvm.org/getting_started/Glossary/) and
[canonicalization reference](https://mlir.llvm.org/docs/Canonicalization/).
Provider terms such as MOVE and computational resonator retain their
[IQM context](https://docs.iqm.tech/iqm-client/user_guide_qiskit.html#computational-resonators).
Project names and public identifiers remain unchanged. The revised writing
policy separates general definitions from local behavior and keeps editorial
instructions out of glossary entries.

## Source checks

Source checks also corrected stale statements:

- `PROGRAM_FORMAT_PREFERENCE` in `mlir/lib/Compiler/QDMIAdapter.cpp` includes
  OpenQASM 2 and does not always select Adaptive QIR for DDSIM.
- `ntrials` in the QCO pass definition uses at least eight trials, increasing
  with available logical CPUs.
- An OpenQASM import/export/reimport probe retained a loop indexing logical
  qubits; the compiler guide's blanket static-index requirement was wrong.
- A four-qubit QAOA template retained `gammas` and `betas` through binary jeff
  serialization and accepted array bindings after import. The old benchmark
  instruction to bind before serialization because names were lost was stale.
  `serializeInputNames` in QCO-to-jeff supplies the metadata.

Release notes and migration records retain their historical purpose. The review
covers the changed guides and connected API descriptions; it does not certify
every statement in every generated API page.

## Final validation

- `uvx nox --non-interactive -s docs`: passed, including all 18 executable
  notebooks, Sphinx/Doxygen generation, and generated-site link checks.
- `uvx nox -s stubs`: passed; compiler, QFT/QPE/Shor, and PennyLane binding
  descriptions were regenerated into their Python interface files.
- `uvx nox -s lint` and `git diff --check`: passed.
- Structural checks against the baseline confirmed unchanged executable cells in
  all 27 edited Markdown pages, Python ASTs apart from docstrings, C++ headers
  apart from comments, and TableGen definitions apart from documentation.
  Binding edits were reviewed as documentation strings; public identifiers and
  schema keys are unchanged.
- A QFT probe with `qubits=4`, `period_exponent=2`, 256 shots, and seed 17
  confirmed four output bits for both methods, four versus one allocated qubits,
  and samples within the reference distribution's support.
- A small complex-matrix check confirmed the documented GPi phase correction and
  GPi2 rotation identity for axis angles -2.1, 0, 0.37, and pi. The nativeRAlias
  definitions and native OpenQASM exporter supply the corresponding
  implementation evidence.
- Chrome DevTools screenshots of the rebuilt glossary, prose policy, and target
  guide showed readable text, equations, tables, and links. Their consoles were
  clear and all observed network requests succeeded.
- Standalone runtime suites, Windows validation, hosted CI, and an exhaustive
  external-link check were not run. Primary sources used for the revised
  definitions were checked individually.

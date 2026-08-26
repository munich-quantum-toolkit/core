(glossary)=

# Glossary

This glossary records MQT Core's preferred terms and the distinctions that
matter to its public interfaces and compiler design. It is not a copy of the
[MLIR glossary][mlir-glossary]. Upstream definitions are useful context, but MQT
Core owns the meanings documented here.

Update this page in the same pull request when introducing or changing a public
or potentially ambiguous term. Each new entry must name the preferred term, list
accepted aliases, and be understandable without detailed knowledge of the
implementation.

## Abbreviations and names

```{glossary}
:sorted:

CDA
Chair for Design Automation
  **Preferred term:** Chair for Design Automation. **Accepted abbreviation:**
  CDA. A
  [research chair at the Technical University of Munich](https://www.cda.cit.tum.de/)
  that develops design methods for fields including quantum computing.

DD
DDs
decision diagram
decision diagrams
  **Preferred term:** decision diagram. **Accepted abbreviations:** DD and DDs
  for the plural. A graph representation that shares repeated substructures to
  store and manipulate quantum states and operations compactly.

IR
intermediate representation
  **Preferred term:** intermediate representation. **Accepted abbreviation:**
  IR. A program representation used between a source language and final output
  so compiler analyses and transformations can operate on explicit structure.

jeff
  **Preferred term:** jeff. **Accepted aliases:** none. A structured, extensible
  [interchange format for quantum programs](https://github.com/unitaryfoundation/jeff).
  `jeff` is a name, not an
  abbreviation, and is always written in lowercase.

LLVM
  **Preferred term:** LLVM. **Accepted aliases:** none. The compiler
  [infrastructure project](https://llvm.org/) on which MLIR is built. LLVM is
  the current project name; do not expand it as an abbreviation in MQT prose.

MLIR
Multi-Level Intermediate Representation
  **Preferred term:** MLIR. **Accepted expansion:** Multi-Level Intermediate
  Representation. [LLVM's reusable infrastructure](https://mlir.llvm.org/) for
  building compilers with several interoperating abstraction levels.

MQSC
Munich Quantum Software Company
  **Preferred term:** Munich Quantum Software Company. **Accepted abbreviation:**
  MQSC. The company that develops and supports parts of MQT Core.

MQSS
Munich Quantum Software Stack
  **Preferred term:** Munich Quantum Software Stack. **Accepted abbreviation:**
  MQSS. The
  [Munich Quantum Valley compilation and runtime ecosystem](https://www.munich-quantum-valley.de/research/research-areas/mqss)
  that connects quantum software to local and remote quantum devices.

MQT
Munich Quantum Toolkit
  **Preferred term:** Munich Quantum Toolkit. **Accepted abbreviation:** MQT.
  The [open-source software toolkit](https://mqt.readthedocs.io/) of which MQT
  Core is the shared foundation.

MQV
Munich Quantum Valley
  **Preferred term:** Munich Quantum Valley. **Accepted abbreviation:** MQV. A
  [Bavarian initiative](https://www.munich-quantum-valley.de/) that develops
  quantum computing research, technology, education, and infrastructure.

OpenQASM
Open Quantum Assembly Language
  **Preferred term:** OpenQASM. **Accepted expansion:** Open Quantum Assembly
  Language. A [versioned language](https://openqasm.com/) for describing quantum
  programs. Include the major version when behavior depends on it.

QASM
quantum assembly language
  **Preferred term:** quantum assembly language. **Accepted abbreviation:**
  QASM. A generic category of assembly-like languages for quantum programs. Do
  not use QASM as an alias for a specific OpenQASM version.

QDMI
Quantum Device Management Interface
  **Preferred term:** Quantum Device Management Interface. **Accepted
  abbreviation:** QDMI. An
  [interface](https://munich-quantum-software-stack.github.io/QDMI/) for
  discovering quantum-device properties and submitting and controlling work
  without coupling software to one device implementation.

QIR
Quantum Intermediate Representation
  **Preferred term:** Quantum Intermediate Representation. **Accepted
  abbreviation:** QIR. An
  [LLVM-based representation and runtime interface](https://www.qir-alliance.org/)
  for exchanging and executing quantum programs.

TUM
Technical University of Munich
  **Preferred term:** Technical University of Munich. **Accepted abbreviation:**
  TUM. The university that hosts the Chair for Design Automation.
```

## Compiler terms

```{glossary}
:sorted:

operation
op
  **Preferred term:** operation. **Accepted alias:** op in code and compact
  technical prose. The basic unit of work and structure in MLIR. An operation
  has a name and can have inputs, results, attributes, and nested regions.

dialect
  A named family of related MLIR operations, types, attributes, and rules. A
  dialect lets one IR contain concepts from several abstraction levels without
  forcing them into one universal instruction set.

pass
  A procedure that inspects or changes IR while preserving the invariants
  declared by its input and output contracts. A pass normally runs as one step
  of a pass pipeline.

rewrite pattern
  A local rule that recognizes one IR shape and replaces or updates it. A
  pattern reports failure without changing IR when its input does not match.

canonicalization
  A semantics-preserving rewrite toward a simpler or preferred representation.
  Canonicalization is not a general optimization pipeline and must not depend on
  a particular downstream target.

conversion
  A change from one legal set of MLIR operations or types to another within the
  MLIR framework. Conversion can be partial or complete and is governed by a
  conversion target that states what is legal.

translation
  A change across the boundary between MLIR and a non-MLIR representation, such
  as OpenQASM text, QIR, or another external program model. Prefer conversion
  when both the source and destination are MLIR dialects.

import
  A translation from an external representation into MQT Core or MLIR.

export
  A translation from MQT Core or MLIR into an external representation.

lowering
  A transformation from a higher-level representation to one closer to the
  operations supported by a target. Lowering can use conversion, rewrites, or
  several passes; it does not imply one specific MLIR mechanism.

legalization
  The act of replacing or rejecting IR until every remaining operation and type
  satisfies a declared conversion target or target capability.

compiler target
  An immutable MQT description of the operations, topology, and properties that
  a compiler pipeline may use for one destination. It is a snapshot used for
  compilation, not a live device connection.

payload
  The program IR on which a transform, schedule, or target-specific action
  operates. Use a more specific term when the exact object, such as a function
  or circuit, matters.

static
  Known while compiling the program. Static does not necessarily mean a C++
  object with static storage duration.

dynamic
  Known only while executing the compiled program or interacting with a target.

QC
  MQT's compatibility-oriented quantum-circuit dialect. QC uses reference
  semantics: operations act on qubit references rather than producing a new SSA
  value for each updated qubit state.

QCO
  MQT's optimization-oriented quantum-circuit dialect. QCO uses value semantics:
  a quantum operation consumes input qubit values and produces output qubit
  values.

QTensor
  MQT's dialect for one-dimensional collections of qubits used with QCO. Its
  operations preserve the linear ownership of the contained quantum values.

reference semantics
  A model in which an operation changes an object reached through a stable
  reference. QC qubit operations use this model.

value semantics
  A model in which an operation consumes input values and produces new output
  values. QCO uses this model to make quantum data flow explicit in SSA form.

linear semantics
  A value-ownership rule under which a quantum value has one live use along a
  program path. The rule prevents copying unknown quantum state and makes
  ownership transfers explicit.
```

## Index

Every glossary entry appears in the
{ref}`alphabetical documentation index <genindex>`.

[mlir-glossary]: https://mlir.llvm.org/getting_started/Glossary/

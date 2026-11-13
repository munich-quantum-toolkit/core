(glossary)=

# Glossary

Use this glossary to look up recurring quantum-computing and compiler terms.
General definitions are separate from Core-specific behavior. API objects and
benchmark schemas are explained in their owning guides and references.

For general MLIR terminology, see the [MLIR glossary][mlir-glossary]. The linked
guides explain how the concepts fit together in a program.

## Abbreviations and names

```{glossary}
:sorted:

CDA
Chair for Design Automation
  A [research chair at the Technical University of
  Munich](https://www.cda.cit.tum.de/) that develops MQT Core together with
  [MQSC](https://mq.sc).

DD
DDs
decision diagram
decision diagrams
  A graph representation that shares repeated substructures to store and
  manipulate quantum states and operations compactly.

program qubit
  A qubit in the input program before assignment to a physical qubit or device
  site. Some frameworks call this a virtual qubit. This describes its role in
  compilation, independently of how the program allocates it or whether it
  represents an encoded qubit in a fault-tolerant computation.

ancilla qubit
ancilla
  A helper qubit used as temporary workspace. A
  clean ancilla starts and ends in the zero state. A dirty ancilla may have an
  unknown or entangled state. A compilation step that borrows a dirty ancilla
  must restore its state and correlations before returning it.

device qubit
physical qubit
  A qubit provided by the hardware. A device site identifies its location;
  other kinds of site, such as resonators, need not be qubits.

target site
  A hardware location, such as a physical qubit or computational resonator.
  Core represents these locations by site IDs in a compiler target; see
  [placement and routing](mlir/target_compilation.md#placement-and-routing)
  for how sites are selected.

computational resonator
  In [IQM devices](https://docs.iqm.tech/iqm-client/user_guide_qiskit.html#computational-resonators),
  a resonator that temporarily stores quantum states and mediates operations
  between qubits. See the [MOVE compilation guide](mlir/target_compilation.md#resonator-mediated-compilation)
  for Core's supported use of resonators.

MOVE
MOVE gate
  An operation used by IQM devices to transfer quantum
  state between a device qubit and a computational resonator. Its ordered
  operands identify the qubit and resonator for transfers in either direction.

qubit placement
placement
  Assignment of program qubits to physical qubits or device sites. Together
  with routing, this lets a circuit respect the device's connectivity. See
  [Qiskit's transpilation stages](https://quantum.cloud.ibm.com/docs/en/api/qiskit/transpiler).

routing
  Inserting operations, often SWAPs, so that interacting qubits can reach
  connected device sites. Routing can change where each program qubit's state
  resides. A routing permutation records that change; its direction depends
  on the interface. Core's maps are described in the
  [Qiskit layout reference](mlir/qiskit.md#transpiler-layouts).

Pauli string
  Also called Pauli product. A tensor product of single-qubit identity or
  Pauli X, Y, and Z operators.

IR
intermediate representation
  A program representation used between a source language and final output so
  compiler analyses and transformations can operate on explicit structure.

QPY
  Qiskit's versioned binary format for storing one or more quantum circuits.

jeff
  A structured, extensible [interchange format for quantum
  programs](https://github.com/unitaryfoundation/jeff). `jeff` is a name, not
  an abbreviation.

LLVM
  The compiler [infrastructure project](https://llvm.org/) on which MLIR is
  built.

MLIR
Multi-Level Intermediate Representation
  [LLVM's reusable infrastructure](https://mlir.llvm.org/) for building
  compilers with several interoperating abstraction levels.

modular multiplier
  A circuit that computes multiplication modulo an integer. Reversible
  implementations can update an input register or write the result to a
  separate register. [Beauregard](https://arxiv.org/abs/quant-ph/0205095)
  describes modular multiplication for quantum order finding. See the
  [benchmark guide](benchmarks.md#modular-multiplier) for Core's chosen circuit.

MQSC
Munich Quantum Software Company
  [MQSC](https://mq.sc) develops MQT Core together with the {term}`Chair for
  Design Automation <CDA>` at {term}`TUM`.

MQSS
Munich Quantum Software Stack
  The [Munich Quantum Valley compilation and runtime
  ecosystem](https://www.munich-quantum-valley.de/research/research-areas/mqss)
  that connects quantum software to local and remote quantum devices.

MQT
Munich Quantum Toolkit
  The [open-source software toolkit](https://mqt.readthedocs.io/) of which MQT
  Core is the shared foundation.

MQT Compiler Collection
  MQT Core's framework for compiling, optimizing, and exchanging quantum
  programs through Python, C++, and the `mqt-core` command-line driver.

MQV
Munich Quantum Valley
  A [Bavarian initiative](https://www.munich-quantum-valley.de/) that develops
  quantum computing research, technology, education, and infrastructure.

OpenQASM
Open Quantum Assembly Language
  A [versioned language](https://openqasm.com/) for describing quantum
  programs. Language features differ between versions.

QASM
quantum assembly language
  A generic category of assembly-like languages for quantum programs. The name
  alone does not identify a particular language or version.

QDMI
Quantum Device Management Interface
  An [interface](https://munich-quantum-software-stack.github.io/QDMI/) for
  discovering quantum-device properties and submitting and controlling work
  without coupling software to one device implementation.

QDMI Client interface
  The standard C interface consumed by applications to open sessions, query
  devices, and manage jobs. MQT Core's C++ and Python Client wrappers consume
  this interface; they are not a driver.

QDMI driver
  An implementation of the QDMI Client interface. The builtin MQT Core QDMI
  driver loads QDMI device libraries. A replacement driver owns its own
  discovery, configuration, and device access; applications must not assume it
  provides MQT Core's private driver extension.

QDMI session
  A connection to a QDMI driver, represented by `qdmi::Session` in C++ and
  `mqt.core.qdmi.Session` in Python. A device session is the driver's
  connection to one device implementation.

QIR
Quantum Intermediate Representation
  An [LLVM-based representation and runtime
  interface](https://www.qir-alliance.org/) for exchanging and executing
  quantum programs. Refer to its Base Profile and Adaptive Profile as Base
  Profile QIR and Adaptive Profile QIR.

TUM
Technical University of Munich
  The university that hosts the Chair for Design Automation.
```

## Compiler terms

```{glossary}
:sorted:

operation
op
  The basic unit of work
  and structure in MLIR. An operation has a name and can have inputs, results,
  attributes, and nested regions.

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
  Canonicalization is not a general optimization pipeline and must not depend
  on a particular downstream target.

parity network
  **Preferred term:** parity network. **Accepted aliases:** none. A CNOT circuit
  that computes specified XOR combinations of input qubits on its wires. Phase
  gates can act on these combinations before the required output wire values
  are restored.

conversion
  A change between or within MLIR dialects, from one legal set of operations
  or types to another. Conversion can be partial or complete and is governed
  by a conversion target that states what is legal.

translation
  A change across the boundary between MLIR and a non-MLIR representation,
  such as OpenQASM text, QIR, or another external program model. Prefer
  conversion when both the source and destination are MLIR dialects.

import
  A translation from an external representation into MQT Core or MLIR.

export
  A translation from MQT Core or MLIR into an external representation.

serialization
  Encoding a program as bytes or writing that encoding to a file.
  Deserialization reads the encoding back into a program. Converting QCO to
  the `jeff` MLIR dialect prepares a serializable program;
  `JeffProgram.to_bytes` and `JeffProgram.write` serialize it.

lowering
  A transformation from a higher-level representation to one closer to the
  operations supported by a target. Lowering can use conversion, rewrites, or
  several passes; it does not imply one specific MLIR mechanism.

legalization
  The act of replacing or rejecting IR until every remaining operation and
  type satisfies a declared conversion target or target capability.

native multi-program job
  One submitted job containing an ordered list of programs with common
  parameters. Results and optional individual outcomes use the input indices.

GPi/GPi2 gate set
  Single-qubit gates used by [IonQ](https://docs.ionq.com/features/getting-started-with-native-gates):
  GPi applies a half turn and GPi2 a quarter turn about an axis in the Bloch
  sphere's equatorial plane. Their phase parameter selects the axis. See the
  [target reference](mlir/target_compilation.md#native-single-qubit-gates)
  for Core's rotation representation, angle units, and global-phase convention.

conversion target
  The MLIR legality rules used by a dialect conversion. It is distinct from an
  MQT compiler target, which describes a compilation destination.

payload IR
  In the [MLIR Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/),
  the IR being inspected or changed by transform operations. A transform
  handle refers to operations, values, or parameters in that IR. This use of
  “payload” does not mean a serialized program submitted to a device.

static
  Known at compile time. The property being described matters: a static
  register size can be known even when its storage is allocated at runtime.
  In `qc.static` and `qco.static`, the known quantity is a device-site
  identifier, not the quantum state.

dynamic
  Determined or performed at runtime. A runtime-dependent index or branch is
  dynamic because its value or choice is not fixed at compile time. Dynamic
  allocation acquires resources during execution, even when the requested
  size is constant. Allocation and qubit placement are separate concepts.

static gate count
  Core's count of quantum operations in the entry-point IR, including
  measurements and resets. Each branch and loop body is visited once; calls
  and modifiers each count once without expanding their bodies. Barriers are
  excluded. This metric does not count runtime executions or every MLIR
  operation. See the [inspection reference](mlir/mqt_compiler_collection.md#inspect-a-quantum-program)
  for the API and complete counting rules.

QC
  MQT's compatibility-oriented quantum-circuit dialect. QC uses reference
  semantics: operations act on qubit references rather than producing a new
  SSA value for each updated qubit state.

QCO
  MQT's optimization-oriented quantum-circuit dialect. QCO uses value
  semantics: a quantum operation consumes input qubit values and produces
  output qubit values.

QTensor
  MQT's dialect for one-dimensional collections of QCO qubits, such as
  registers. The [dialect reference](mlir/QTensor.md) explains its allocation,
  extraction, and reinsertion rules.

reference semantics
  A model in which references identify objects whose state can change.
  Several references may identify the same object. QC uses this approach for
  qubits; see the [representation tutorial](tutorials/compiler_representations.md)
  for how it differs from QCO's explicit data flow.

value semantics
  A model in which values describe data rather than references to mutable
  objects. In a quantum SSA representation, updated qubits are represented by
  operation results; [QSSA](https://arxiv.org/abs/2109.02409) is an example.
  Restrictions on copying or discarding quantum values require additional
  rules, such as QCO's linear use rules.

linear semantics
linear ownership
  Rules that require a resource to be used exactly once, rather than copied or
  silently discarded. Linear and affine type systems are used to control
  quantum resources; [Selinger and Valiron](https://arxiv.org/abs/cs/0404056)
  give an affine formulation, which permits discarding but not duplication.
  Core's [quantum resource rules](mlir/index.md#quantum-resource-rules)
  specify its exact-use and wire correspondence requirements separately.
```

## Quantum algorithms

```{glossary}
:sorted:

magic state
  A non-stabilizer resource state used with Clifford operations and
  measurements to implement non-Clifford operations. For example,
  $T|+\rangle$ enables a T gate through state injection.
  [Bravyi and Kitaev](https://arxiv.org/abs/quant-ph/0403025) describe how magic
  states enable universal computation with Clifford operations.

magic-state distillation
  A protocol that uses Clifford operations and measurements to convert several
  noisy resource states into fewer, higher-quality magic states, accepting
  outputs according to measurement outcomes. See
  [Bravyi and Kitaev](https://arxiv.org/abs/quant-ph/0403025) for the general
  construction and the [benchmark guide](benchmarks.md#magic-state-distillation)
  for Core's chosen protocol and ideal-input checks.

iterative quantum phase estimation
iterative QPE
iQPE
  Estimation of an eigenphase through successive rounds of controlled unitary
  evolution and measurement, using a reused query qubit. In the
  [feedback-based algorithm](https://arxiv.org/abs/quant-ph/0610214),
  earlier measurement outcomes determine later phase corrections. It combines
  phase estimation with a semiclassical inverse QFT; a semiclassical QFT alone
  does not require qubit reuse. See the
  [QPE benchmark](benchmarks.md#quantum-phase-estimation) for Core's circuit.

Shor's algorithm
  A factoring algorithm
  that combines quantum order finding with classical arithmetic to recover and
  verify nontrivial factors.

order finding
  Finding the smallest positive exponent $r$ for which a given base $a$
  satisfies $a^r \equiv 1 \pmod{N}$, with $a$ coprime to $N$.

continued fraction
  A nested representation of a number by integer parts and reciprocals.
  Truncations, called convergents, give rational approximations used to
  recover candidate exponents from measured phases in Shor's algorithm.

QAOA
quantum approximate optimization algorithm
  A variational algorithm that alternates cost and mixer unitaries to prepare
  candidate solutions to a discrete optimization problem.

quantum fan-out
fan-out
  An operation that applies a controlled X from one control qubit to each
  target qubit. When the targets start in $|0\rangle$, it copies a
  computational-basis bit; a superposed control generally becomes entangled
  with the targets rather than being cloned. The
  [unbounded fan-out model](https://arxiv.org/abs/quant-ph/0208043) treats this
  as one primitive regardless of the number of targets. Its decomposition
  into two-qubit gates has a different depth cost.

semiclassical quantum Fourier transform
semiclassical QFT
  A QFT followed by measurement, implemented with single-qubit gates and
  measurements whose outcomes control later gates, replacing the QFT's
  controlled two-qubit gates. [Griffiths and Niu](https://arxiv.org/abs/quant-ph/9511007)
  describe this construction. Qubit recycling is an additional choice that
  depends on how the input is prepared; it is not part of the general
  definition. See [Core's QFT benchmark](benchmarks.md#quantum-fourier-transform)
  for its one-qubit implementation.

W state
  The equal, positive-amplitude superposition of all computational-basis
  states with exactly one qubit in state one.
```

## Index

Every glossary entry appears in the
{ref}`alphabetical documentation index <genindex>`.

[mlir-glossary]: https://mlir.llvm.org/getting_started/Glossary/

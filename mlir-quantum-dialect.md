What I want to be able to describe:

- A `Qubit` type including operations to dynamically `allocate` and `deallocate` qubits as well as a static `qubit` operation that yields a qubit reference from an index
- A `reset` operation that acts on a qubit
- A `measure` operation that takes a qubit and produces an `i1` classical measurement result
- A way to describe unitary operations that drive the quantum computation.

The last part is the most important and critical part as it forms the center of the quantum dialect.
These operations (also referred to as quantum gates or simply gates) have the following properties

- they are unitary, which is an essential trait to match to
- each type of gate has a compile time fixed number of target qubits (typically one or two). This is regulated by a target arity trait at the moment, which doesn't seem perfectly feasible, but maybe it is.
- each type of gate may have a compile time fixed number of parameters. This is currently indicated similarly to the target arity by a parameter arity trait. Parameters may either be defined statically or dynamically through values. Mixtures are possible. Canonicalization should ensure that parameters that can be statically defined are indeed statically defined.
- each type of gate has a unitary matrix associated with it. The size of the matrix is 2^n\*2^n with n being the target arity. For gates without parameters or with only static parameters this is compile-time fixed and known. For gates with dynamic parameters, this description is symbolic (mostly in terms of rotation angles and trigonometric functions).
- several types of modifiers may be applied to unitary gates to transform and extend them. An `inv` (or `adj`) modifier can be added to invert the gate, which corresponds to forming the adjoint of the underlying matrix. A control modifier may be used to add a variable list of qubits as control qubits to the operation. Control qubits may be positive or negative, which means they trigger on the control qubit being in state |1> or |0>, respectively. Last but not least, the `pow(r)` powering modifier can be used to compute powers of the unitary gate. For simplicity, `r` is mostly assumed to be an integer, but any real number is feasible through computing the principle logarithm. Modifiers are generally evaluated lazily, that is, they are simply tagged onto the gate and only resolved once such a resolution is necessary. There should be a way to query the effective unitary of a gate, which takes into account the modifiers. Modifiers should have canonicalization rules. They all commute. Control qubits must be unique and multiple control modifiers can be combined into one. Two inverses cancel another. Negative powers can be translated to an inverse modifier and the positive power. powers of 1 can be removed. powers of 0 can be replaced by an identity gate.

The difference between the MQTRef and the MQTOpt dialects is that the MQTRef dialect uses reference/memory semantics for its operations, while the MQTOpt dialect uses value semantics. The MQTOpt dialect is used for optimization and transformation passes, while the MQTRef dialect is used for initial representation and code generation.

I want to (at least)

- query an operation for the qubits it acts on
- query an operation whether it is a single-qubit gate (single target, no controls), two-qubit, single-target, two-target, no-controls, no-parameters, only static parameters, or dynamic parameters gate.
- query an operation for its underlying unitary
- apply `inv`, `ctrl`, `negctrl` and `pow(r)` modifiers to the gates
- conveniently construct individual basis gates in C++ without much coding overhead (e.g. mqtref.swap(q0, q1) or mqtref.rx(q0, theta)); the current setup makes this increadibly hard through the builder interface. The target and parameter arity traits should provide the necessary information to generate the correct builder overloads.
- have a construct for sequences of (unitary) gates that modifiers can also be applied to. these "compound" operations or sequences may only contain unitary operations. These should be inlinable in an MLIR native way. Canonicalization should remove empty sequences and merge nested sequences as part of inlining.
- have convenient shortcuts for constructing common gates such as the CNOT (=ctrl X) or CZ (=ctrl Z) and potentially have some syntactic sugar for these.

### Guiding principles

- Prefer idiomatic MLIR: keep base ops minimal and uniform, push optional features into wrapper ops/regions, and express semantics via interfaces and canonicalization rather than baking everything into each gate.
- Make composition first-class: modifiers are explicit IR constructs that can wrap single gates or entire sequences, and they compose/lazily defer.
- Ergonomics: provide sugar in the parser/printer and rich C++ builders so that writing and constructing circuits is pleasant.
- Analysis-friendly: expose queries through a single interface so passes never need to pattern-match concrete ops.
- If possible, there shouldn't be too much duplication between both dialects as they essentially share a lot of structure.
- Make use of MLIR-native traits and interfaces wherever they are useful. For example, `AttrSizedOperandSegments` or `InferTypeOpInterface`.
- Provide canonicalization where they make sense to keep the IR normalized and simple.
- Provide verifiers for IR constructs

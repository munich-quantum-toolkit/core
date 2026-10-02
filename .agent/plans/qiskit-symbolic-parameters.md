# Support symbolic Qiskit parameter expressions

Status: historical implementation record.

## Goal and scope

Users can import Qiskit circuits whose gates and global phase use free
parameters or real-valued parameter expressions. The compiler represents each
free parameter as a named `f64` function input and represents arithmetic with
frontend-neutral Arith and Math dialect operations. Users can export the
program, bind the reconstructed Qiskit parameters, and obtain the same numeric
circuit. Parameter names are unique across free and lexically bound Qiskit
parameters. Import rejects circuits that violate this source contract.

This work completes issue #2067. It extends the Qiskit circuit translation
introduced by #2031 and builds on the CBit representation from #2158. It must
not weaken the existing preflight checks, mutate input circuits, or expose a
partially constructed output circuit after a failure. Exact
`ParameterVectorElement` provenance is a separate follow-up.

## Constraints

- Qiskit 2.5 has no public expression-tree reader that works without an optional
  SymPy installation. Its own parameter-expression code records a stable postfix
  replay sequence in `_qpy_replay`. Evidence: nested expressions expose
  `OPReplay` records with `op`, `lhs`, and `rhs`; reverse subtraction, division,
  and power use distinct opcodes.

- Qiskit rejects two free parameters with the same name in one circuit, but it
  permits a lexically bound loop parameter and a distinct free parameter to
  share a name. Evidence: the existing name-keyed local map incorrectly captured
  the free parameter in such a loop body.

- Qiskit can construct parameter objects that share a UUID but disagree on their
  name. Such objects are outside Qiskit's intended contract. The importer does
  not inspect UUIDs; it validates the supported unique-name contract instead.

- a custom gate's definition already contains the actual symbols or expressions
  supplied at its call site. The importer does not need a separate
  formal-parameter substitution scheme. It must validate the definition against
  the current global and lexical bindings.

- an expression can convert to a number while still tracking free parameters.
  The version-specific reader must inspect `parameters` before it treats a value
  as a numeric constant.

- treating `ParameterVectorElement` as an ordinary standalone symbol changes
  positional binding order. This layer therefore rejects vector elements instead
  of inferring semantics from names such as `theta[10]`.

- Merely collecting a named function argument does not preserve it in Qiskit.
  The writer only creates parameters reached from emitted gate or global-phase
  expression trees, so an unused input would otherwise disappear.

- `mqt.input_name` and `mqt.qubit_register_name` are raw strings in
  `mlir/include/mlir/Dialect/Utils/Utils.h`; no dialect owns or verifies the
  namespace. MLIR dialect ODS can declare typed discardable attributes and
  generate helpers for them.

- QC/QCO conversion copies only `mqt.qubit_register_name`, and the QC and
  QTensor register-shrinking rewrites drop it when they replace an allocation.
  Compatible discardable metadata must be transferred as a group.

- The project namespace `::mqt` and the existing MLIR utility namespace
  `::mlir::mqt` require explicit qualification in translation units that import
  both namespaces. The metadata dialect belongs in the existing `::mlir::mqt`
  namespace; changing its C++ namespace would split related MQT MLIR APIs to
  avoid a local lookup issue.

- Both dialects expose gate parameters, including power exponents, through
  `UnitaryOpInterface`. An interface verifier is the one shared MLIR hook that
  covers standard gates and modifiers without adding a verifier to each
  operation.

- MLIR's dialect documentation generator requires `OpBase.td` even for an
  operation-free dialect. `DialectBase.td` declares the dialect but does not
  make the `Op` base class visible to the documentation backend.

- `cbit.alloc` modeled its non-semantic `source_name` as an inherent operation
  field. The name has the same cross-frontend contract as quantum register names
  and belongs in the shared discardable metadata.

- QC, QCO, and jeff used LLVM dialect `passthrough` metadata to identify the
  program entry point before LLVM lowering. The string array had no high-level
  owner or verifier and mixed the program model with a target encoding.

## Decisions

- Use one immutable, copyable scalar expression tree at the generic
  reader/writer boundary. Rationale: Qiskit-specific replay objects remain in
  `Qiskit2_5.cpp`, while import and export share one frontend-neutral contract.

- Support finite numbers, symbols, add, subtract, multiply, divide, power,
  negate, sine, cosine, tangent, inverse sine, inverse cosine, inverse tangent,
  exponential, logarithm, absolute value, and real conjugation. Rationale:
  Arith, Math, and Qiskit's 2.5 C API represent this real-valued subset
  directly. Operations without matching compiler semantics fail with a precise
  diagnostic.

- Require unique names across all free and lexically bound Qiskit parameters,
  then key import state by name. Continue to key compiler export by SSA value
  and use `mqt.input_name` for the public name. Rationale: Qiskit programs that
  reuse a parameter name for a distinct parameter are ambiguous and outside the
  supported source contract.

- Use the source name as parameter identity and do not inspect or preserve
  Qiskit's UUID across a round trip. Rationale: the writer creates exactly one
  Qiskit symbol for each named compiler input and reuses it throughout gates and
  global phase.

- Bound normalized expression depth and node count before compiler or circuit
  construction. Rationale: the existing definition and control-flow readers are
  bounded, and parameter replay must have the same fail-closed behavior for
  adversarial input.

- Reject `ParameterVectorElement` input in this PR and implement exact vector
  provenance as a stacked follow-up. Rationale: scalar symbols complete issue
  #2067, while vector identity, allocation bounds, sparse indices, and
  vector-level binding form an independently reviewable contract.

- Require every named `f64` input to occur in the normalized parameter trees
  that will be emitted. Rationale: Qiskit circuits cannot declare an otherwise
  unused parameter, so failing before writer allocation avoids silently changing
  the public parameter set.

- Define `mqt.input_name` and `mqt.register_name` as typed discardable
  attributes in an operation-free `mqt` dialect. Verify them with the dialect's
  operation and region-argument hooks. Rationale: MLIR assigns the semantics of
  a dialect-prefixed discardable attribute to that dialect; this provides one
  frontend-neutral owner and generated type-safe helpers.

- Use one function-wide namespace for named inputs and quantum or classical
  registers. Rationale: duplicate public names are ambiguous even when a source
  library happens to accept some cross-kind collisions. Rejecting them in MQT
  metadata keeps import and export contracts deterministic.

- Mark the single defined module-level program function with the unit attribute
  `mqt.entry_point`. Preserve it as discardable metadata through high-level
  conversions. Materialize LLVM `passthrough = ["entry_point", ...]` only when
  QIR metadata is attached, then remove the MQT marker. Rationale: the MQT
  dialect owns the frontend-neutral program contract, while LLVM passthrough
  attributes remain a QIR target detail.

- Query, set, and remove `mqt.entry_point` through MQT dialect helper functions.
  Declare the MQT dialect as a pass dependency only when the pass creates MQT
  metadata. Rationale: this keeps the attribute key private to its owner and
  follows the MLIR pass contract for dependent dialects.

- Keep `mqt.input_name` independent of the argument type. Rationale: the name is
  shared program metadata, while Qiskit and future OpenQASM exporters decide
  which input types they can represent.

- Copy compatible discardable attributes when a conversion or rewrite replaces
  their owner. Rationale: this preserves current and future shared metadata
  without source-format-specific key handling.

- Represent normalized parameters as a private variant of number, symbol, unary,
  and binary nodes. Use separate unary and binary operation enums and factory
  methods that always allocate the required operands. Rationale: consumers no
  longer validate redundant kind and pointer combinations because malformed tree
  shapes are not representable.

- Make finite statically known parameter values a `UnitaryOpInterface` invariant
  in QC and QCO. Traverse pure expression DAGs and memoize folding so a
  non-finite literal or folded subexpression cannot be hidden below a dynamic
  root. Keep runtime finiteness as a precondition for dynamic values. Rationale:
  dialect verification owns valid quantum IR, while import readers still reject
  invalid source values before construction and exporters can assume verified
  IR.

## Outcome and validation

The scalar implementation is complete. The shared MQT metadata dialect owns
input names, register names, and the program entry point. QC/QCO conversions
preserve all compatible discardable metadata without duplicate dialect-specific
attributes. The Qiskit boundary uses unique names instead of UUID edge cases and
a closed normalized expression variant. QC and QCO reject non-finite statically
known unitary parameters through their shared interface contract.

The final Release build and all 4,301 configured tests pass; one QDMI test is
skipped by its environment guard. All 206 focused Python MLIR tests pass after a
fresh extension build. MLIR documentation, stub generation, and the repository
lint suite also pass. The Sphinx build remains unverified because both attempts
failed to resolve the host for its external QDMI tag file.

## Code and ownership

`bindings/mlir/qiskit/QiskitTranslation.h` defines the normalized objects shared
by the generic translation and one Qiskit-version adapter.
`bindings/mlir/qiskit/Qiskit2_5.cpp` is the only file that reads Python
parameter objects, `_qpy_replay`, or calls Qiskit's `qk_param_*` C functions.

`bindings/mlir/qiskit/QiskitImport.cpp` validates a complete source circuit,
creates a QC program, inserts one named `f64` entry argument per free symbol,
and lowers normalized expressions to SSA values.
`bindings/mlir/qiskit/QiskitExport.cpp` performs the reverse preflight: it
recognizes a supported `f64` SSA expression graph, builds normalized
expressions, and only then asks a version-specific writer to allocate a Qiskit
circuit.

The importer uses `mqt.input_name` for the stable public name of each compiler
input. `mlir/include/mlir/Dialect/MQT/IR/MQTDialect.td` declares this metadata,
`mqt.register_name`, and `mqt.entry_point`; the operation-free `mqt` dialect
owns their contracts. The compiler representation uses `arith.addf`,
`arith.subf`, `arith.mulf`, `arith.divf`, and `arith.negf`, plus matching
real-valued Math dialect operations. A local `for` induction parameter is a
temporary SSA value keyed by its unique source name. It is not a function input.

## Acceptance

Import a Qiskit circuit with two shared free symbols in nested arithmetic, gate
arguments, and global phase. The QC entry function must have one named `f64`
argument per symbol and must contain the matching Arith and Math operations.
Export it, bind the parameters, and compare its numeric operator and global
phase with the source circuit.

Import partially bound expressions and a parameterized custom gate. Both must
resolve the remaining symbols without source mutation. Reject a `for` loop whose
binder has the same displayed name as a distinct free symbol before module
construction.

Export hand-written QC with supported `f64` Arith and Math expressions. The
result must contain shared Qiskit parameters and bind to the same numeric
values. Duplicate or unused named inputs, unsupported SSA operations,
unsupported Qiskit functions, non-finite constants, malformed trees, and
excessive expressions must fail during preflight.

Reject a `ParameterVectorElement` before module construction and leave the
source circuit unchanged. Continue to accept standalone scalar parameters whose
names contain brackets without inferring vector semantics.

## Interfaces

`Parameter` in `QiskitTranslation.h` is a copyable immutable tree. Its private
variant contains a number, a symbol, a unary node, or a binary node. Factory
methods create all nodes and allocate every required child. `Loop::parameter` is
`std::optional<Parameter>` and must contain a symbol when present.
`CircuitReader` returns normalized trees for instruction parameters and global
phase. `CircuitWriter` accepts the same tree and reconstructs Qiskit parameters
with the version-specific C and public Python APIs.

No SymPy dependency is added. No Qiskit object or expression string is stored in
MLIR. The supported compiler operations remain frontend-neutral Arith and Math
dialect operations on `f64` values.

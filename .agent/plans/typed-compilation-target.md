# Record compiler targets as typed MQT IR

Status: historical implementation record.

## Goal and scope

MQT Core represents a compilation target as the context-free C++
`mlir::CompilerTarget` value. After this change, clients can also store and
exchange the same immutable target facts as a typed `#mqt.compilation_target`
MLIR attribute. Converting the C++ value to the attribute and back preserves
names, ordered sites, timing data, connectivity, native operations, and fixed or
variadic operation arities.

The attribute deliberately describes only compiler-target facts. It does not
define a payload format, execution capability, combined target environment, or
module attachment. Those contracts require separate design work.

## Constraints

- The original implementation combined compiler-target facts with payload and
  DLTI contracts. Evidence: the complete original implementation required
  `PayloadEnvAttr`, `TargetEnvAttr`, and `MLIRDLTIDialect`, while the
  compiler-target conversion itself uses none of them.

- The final #2218 model treats missing connectivity and operation support as
  target-inference errors and gives operations a fixed or variadic arity.
  Evidence: `CompilerTarget::Connectivity` and `NativeOperations` have no
  unknown state, and `CompilerTarget::Operation::Arity` distinguishes fixed
  widths from inclusive variadic minima.

## Decisions

- Keep only `CompilationTargetAttr` and its leaf attributes. Rationale: MQT Core
  4.0 needs stable compiler-target serialization, while the payload capability
  model is still under design.

- Do not define a canonical module attribute in this change. Rationale: A module
  attachment would imply ownership and composition rules that belong to the
  deferred target-environment design.

- Keep the existing C++ target enums and map them explicitly to the MQT dialect
  enums. Rationale: The C++ target API remains independent of the generated enum
  types, while the two-state mappings are exhaustive and small.

- Represent operation arity as `OperationArityAttr` with a fixed or variadic
  kind and one value. Rationale: Fixed zero represents `gphase`, while a
  positive variadic minimum represents a base gate that accepts additional
  controls. Variadic and fixed-zero operations cannot have site tuples because
  one tuple cannot describe all accepted widths.

## Outcome and validation

The focused implementation records compiler-target facts without adding a
payload-capability or DLTI model. The typed target round trip preserves fixed
zero-arity and positive-minimum variadic operations. The MQT dialect and
compiler tests, generated dialect documentation, C++ lint, and repository lint
all pass.

## Code and ownership

`mlir/include/mlir/Compiler/Target.h` declares the immutable C++ target model.
`mlir/lib/Compiler/Target.cpp` validates target data, prepares routing and
synthesis caches, and implements conversions. The MQT dialect is declared in
`mlir/include/mlir/Dialect/MQT/IR/MQTDialect.td` and implemented in
`mlir/lib/Dialect/MQT/IR/MQTDialect.cpp`. TableGen generates the public C++
attribute and enum declarations from that dialect file.

A site is a hardware location identified by a target-defined nonnegative
integer. Its position in the site's array defines the compiler's dense vertex.
Connectivity is all-to-all or an explicit set of undirected couplings.
Native-operation support is unrestricted or an explicit list. An explicit
operation has a fixed width, including zero, or a positive variadic minimum.
These facts must survive every conversion without inference.

## Acceptance

Acceptance requires parsing and printing a detailed `#mqt.compilation_target`
attribute without structural change. Materializing a detailed `CompilerTarget`,
reconstructing it from the attribute, and materializing it again must produce
the same attribute. The same round trip must preserve all-to-all or explicit
connectivity, unrestricted or explicit operation support, and fixed or variadic
operation arities. Invalid or disconnected reconstructed targets must retain the
existing C++ diagnostics.

Generated dialect documentation must show concrete syntax for the composite
attribute and its leaf records. Repository lint must report no formatting,
spelling, metadata, or generated-file problems.

## Interfaces

The public additions are:

    static llvm::Expected<CompilerTarget>
    CompilerTarget::create(mqt::CompilationTargetAttr attribute);

    mqt::CompilationTargetAttr
    CompilerTarget::materialize(MLIRContext& context) const;

The MQT dialect exports `DurationUnitAttr`, `SiteAttr`, `CouplingAttr`,
`SiteTupleAttr`, `OperationArityAttr`, `NativeOperationAttr`,
`CompilationTargetAttr`, `ConnectivityKind`, `NativeOperationsKind`, and
`OperationArityKind`. The implementation uses existing LLVM and MLIR libraries
and adds no third-party dependency.

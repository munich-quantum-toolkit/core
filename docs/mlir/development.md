# MLIR development policy

This page defines how MQT Core uses MLIR. It condenses the parts of the MLIR and
LLVM guidance that most often affect design, review, tests, and debugging. MQT
Core is an MLIR consumer, so this policy is normative for this repository even
where existing code differs.

Reviewed against LLVM and MLIR **22.1.8**. Revisit this page and the MLIR
clang-tidy configuration on every major LLVM/MLIR upgrade.

## C++ const and IR handles

MLIR's intermediate representation (IR) is a mutable graph. `Value` and its
`TypedValue`, `BlockArgument`, and `OpResult` forms, `Operation`, `Block`,
`Region`, `ModuleOp`, and typed operation wrappers are small handles into that
graph. A `const` handle does not make the referenced IR immutable and creates a
false model of const-correctness. Follow MLIR's
[rationale for the usage of `const`][mlir-const]:

```cpp
void inspect(Value value, Operation* operation);

for (Value operand : operation->getOperands()) {
  /// Use operand without implying that the IR graph is immutable.
}
```

Do not write:

```cpp
void inspect(const Value value, const Operation* operation);
```

This rule also applies to local variables, lambda parameters, range variables,
structured bindings, typed wrappers such as `func::FuncOp`, and `const auto`
that deduces one of these types. `ValueRange`, `OperandRange`, and `ResultRange`
are cheap non-owning views over the same handles. Copy these handles and views
instead of binding them as `const` values or references. Do not add top-level
`const` to any by-value parameter. Continue to use normal const-correctness for
ordinary C++ objects, references, pointers, containers, and strings; do not
distort a generic interface or access through a const container merely because
one contained value is an MLIR handle. MLIR `Type` and `Attribute` objects are
immutable values and are not mutable IR graph handles.

The dependency-free source gate checks only explicitly named core handles and
views. A text check cannot infer the type behind `auto` or distinguish an MLIR
operation wrapper from an unrelated C++ type whose name ends in `Op`. This
policy still applies in both cases.

## Passes, verifiers, and rewrites

Follow the [MLIR Developer Guide][mlir-developer-guide] and these repository
rules:

- A pass may assume that its declared input operation is verified. It must not
  crash or assert on valid IR, and its successful output must verify.
- A verifier checks only invariants owned by its operation. Do not make an
  operation verifier depend on enclosing pipelines or unrelated operations.
- Declare every dialect that a pass can create or load as a dependent dialect.
- Use bounded recursion. Treat unbounded recursive IR walks or pattern
  application as correctness risks, not only performance risks.
- Make rewrite-pattern return values truthful. Return failure without changing
  IR; report success only after performing the promised rewrite.
- Use established matchers such as `m_Constant` instead of manually recognizing
  one producer shape.
- Use traits for static properties and interfaces when behavior varies by
  operation implementation.
- Treat a memref as a shaped memory abstraction, not as a C++ pointer.
- Search upstream MLIR for an operation, interface, trait, conversion, or helper
  before adding an MQT-specific equivalent.

Use diagnostics for invalid input or unsupported behavior. Reserve assertions
for internal invariants that valid input cannot violate. Diagnostics must state
what failed and, when useful, which form is supported.

## Data structures and performance

Use LLVM views and abstract range types at MLIR-facing boundaries. Prefer an
LLVM data structure such as `SmallVector`, `DenseMap`, or `MapVector` when its
storage, lookup, ordering, or API behavior provides a concrete benefit. Keep a
standard-library type when it already expresses the required contract.

Do not convert containers in bulk for style. Require a profile, benchmark, or a
specific allocation or complexity argument for a performance rewrite. Keep
user-visible output deterministic: never use pointer identity or unspecified
iteration order as an observable ordering rule.

## Tests

MQT Core uses GoogleTest and CTest for MLIR code. Do not add `lit` or FileCheck
infrastructure. Adapt the useful principles from the [MLIR Testing Guide]
[mlir-testing] as follows:

- Parse and transform IR in-process. Use a subprocess only for irreducible
  command-line behavior.
- Use the smallest input that isolates the contract.
- Give the test a name that states the behavior.
- Check semantic operations, types, attributes, and diagnostics instead of a
  large textual snapshot.
- Test valid and invalid cases when both form part of the contract.
- Verify input and successful output around pass-pipeline tests.
- Add a regression test for every behavioral bug fix.

## Debugging

Start from the [MLIR debugging workflow][mlir-debugging]:

1. Reduce the input to a small `.mlir` file and identify the first failing pass.
2. Run only the relevant pass pipeline.
3. Print generic IR when custom syntax may hide malformed state.
4. Print IR before the relevant pass or after a failure.
5. Disable multithreading when output order obscures the failure.
6. Enable dialect-conversion tracing for a conversion failure.
7. Save a pass-pipeline crash reproducer for crashes that are not immediately
   local.
8. Turn the reduced case into the smallest direct regression test.

`mqt-cc` registers MLIR's standard pass-manager options. Useful options include
`--mlir-print-op-generic`, `--mlir-print-ir-before-all`,
`--mlir-print-ir-after-failure`, `--mlir-disable-threading`,
`--mlir-print-stacktrace-on-diagnostic`, and
`--mlir-pass-pipeline-crash-reproducer=<path>`. `--debug-only` traces require a
build of LLVM/MLIR and MQT Core with debug logging enabled; do not assume that a
release build provides them.

## Upstream references

- [MLIR Developer Guide][mlir-developer-guide]
- [MLIR rationale for the usage of `const`][mlir-const]
- [MLIR Testing Guide][mlir-testing]
- [MLIR debugging guide][mlir-debugging]
- [MLIR FAQ][mlir-faq]
- [LLVM Coding Standards][llvm-coding-standards]

[llvm-coding-standards]: https://llvm.org/docs/CodingStandards.html
[mlir-const]: https://mlir.llvm.org/docs/Rationale/UsageOfConst/
[mlir-debugging]: https://mlir.llvm.org/getting_started/Debugging/
[mlir-developer-guide]: https://mlir.llvm.org/getting_started/DeveloperGuide/
[mlir-faq]: https://mlir.llvm.org/getting_started/Faq/
[mlir-testing]: https://mlir.llvm.org/getting_started/TestingGuide/

# MLIR development policy

This page defines how MQT Core uses MLIR. It condenses the parts of the MLIR and
LLVM guidance that most often affect design, review, tests, and debugging. MQT
Core is an MLIR consumer, so this policy is normative for this repository even
where existing code differs.

Reviewed against LLVM and MLIR **23.1.0**. Revisit this page and the MLIR
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

## Linear quantum values

Every `!qco.qubit` and one-dimensional qubit tensor or vector SSA value in valid
QCO IR has exactly one use, including block arguments. `qco::verifyLinearity`
owns this whole-IR check; ordinary MLIR operation verification alone does not
establish it. Builders and transformations must preserve the invariant, and
public QCO pipeline boundaries must validate it.

Rewrites on valid QCO IR can use `*value.user_begin()` to obtain the sole
consumer, or `*value.use_begin()` for its `OpOperand` and operand number. Do not
repeat `hasOneUse()` guards in these rewrites. Keep linearity checks in the
verifier and optional debug assertions at internal boundaries. This rule does
not apply to QC references or classical SSA values, and does not permit assuming
that results preserve wire order: linearity and wire correspondence are separate
contracts.

When forwarding a linear value requires deleting its producer, use a rewrite
that removes both operations. A fold can only change its root; do not rely on
later dead-code elimination to restore linearity before other rewrites run.

## Data structures and performance

Use LLVM views and abstract range types at MLIR-facing boundaries. Prefer an
LLVM data structure such as `SmallVector`, `DenseMap`, or `MapVector` when its
storage, lookup, ordering, or API behavior provides a concrete benefit. Keep a
standard-library type when it already expresses the required contract.

When code in the `mlir` namespace or one of its nested namespaces uses an LLVM
name that `mlir/Support/LLVM.h` imports, include that header and use the
unqualified name, such as `SmallVector`, `StringRef`, or `function_ref`. Do not
rely on `mlir/Support/LLVM.h` for type definitions. Include each LLVM header
that the source file needs. Keep the `llvm::` qualifier for names that
`mlir/Support/LLVM.h` does not import.

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

Build the driver with the configured LLVM/MLIR installation:

```sh
cmake --preset release
cmake --build --preset release --target mqt-cc
mqt_cc=build/release/mlir/tools/mqt-cc/mqt-cc
```

Save this reduced example as `reduced.mlir`:

```mlir
module {
  func.func @f(%q: !qco.qubit) -> !qco.qubit {
    %h0 = qco.h %q : !qco.qubit -> !qco.qubit
    %h1 = qco.h %h0 : !qco.qubit -> !qco.qubit
    return %h1 : !qco.qubit
  }
}
```

Run exactly the selected pipeline with `--run-pipeline`:

```sh
"$mqt_cc" reduced.mlir --run-pipeline \
  --pass-pipeline='builtin.module(canonicalize)'
"$mqt_cc" reduced.mlir --run-pipeline \
  --pass-pipeline='builtin.module(hadamard-lifting)' \
  --mlir-print-op-generic --mlir-print-ir-before-all \
  --mlir-print-ir-after-failure --mlir-disable-threading
```

The first command removes the two Hadamards. The second prints the input to
HadamardLifting on stderr and its result on stdout. An empty `builtin.module()`
pipeline preserves the input. Use `--mlir-print-ir-before=hadamard-lifting` to
select one pass's dump in a longer pipeline. Non-stdout MLIR output selected
with `-o` is bytecode; redirect stdout to save textual IR.

Isolated execution requires MLIR input and a `builtin.module(...)` pipeline. It
skips frontend conversion, compiler preparation, default optimizations, and
output lowering. Parsing verifies the input, and QCO linearity is checked before
and after the pipeline. Pass-manager verification remains enabled. Do not
combine this mode with `--emit`, target compilation, or the decomposition
convenience flag. Supply already reduced input for the pass's supported subset.

Ordinary compilation with `--pass-pipeline` retains its required preparation and
cleanup stages around the supplied QCO pipeline. `--passes` is an alias for the
same textual syntax; individual pass flags are not supported. The CLI and
library share QCO pass and upstream transform registration. The driver also
registers its conversion passes so they can be selected for debugging.

### Diagnostics and reproducers

Use `--mlir-print-stacktrace-on-diagnostic` to attach a trace when the LLVM
build supports stack traces. Source locations, excerpts, and attached operation
notes remain available throughout compilation, including for stdin. Initial jeff
conversion also uses the requested pass-manager instrumentation.

For a runnable failure example, save this valid but unsupported QC input as
`unsupported.mlir`:

```mlir
module {
  func.func @f() {
    cf.br ^next
  ^next:
    return
  }
}
```

Generate and replay its failing conversion pipeline:

```sh
"$mqt_cc" unsupported.mlir --run-pipeline \
  --pass-pipeline='builtin.module(qc-to-qco)' \
  --mlir-print-ir-before-all --mlir-print-ir-after-failure \
  --mlir-disable-threading \
  --mlir-pass-pipeline-crash-reproducer=failure.mlir
"$mqt_cc" failure.mlir --run-reproducer --mlir-print-ir-before-all
```

Both commands fail with the unsupported unstructured-control-flow diagnostic.
`--run-reproducer` applies the recorded pipeline, threading, and verification
settings. It requires a non-empty recorded module pipeline and cannot be
combined with a custom pipeline or ordinary compiler output modes. It follows
the recorded verification policy, including disabled verification, and does not
add QCO linearity checks. Ordinary file loading does not activate replay.
Disable threading when generating a local reproducer with
`--mlir-pass-pipeline-local-reproducer`.

For conversion tracing, add `--debug-only=dialect-conversion`. This requires
LLVM/MLIR and MQT Core code built with debug logging enabled; build-type names
alone do not establish that support. Keep threading disabled when tracing to
make the output easier to read. Stack addresses and frame counts vary by
platform and are unsuitable regression assertions.

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

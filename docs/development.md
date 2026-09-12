# Development policy

This page defines MQT Core's repository-owned development policy. It supplements
the {doc}`contribution guide <contributing>`, which describes the contribution
process, and the {doc}`AI usage policy <ai_usage>`, which defines accountability
for AI-assisted work.

## Sources of authority

Apply guidance in this order:

1. This policy, including its subsystem-specific sections.
2. Repository configuration and scoped agent instructions that enforce or
   summarize that policy.
3. Upstream guidance that MQT Core has explicitly adopted.
4. Existing code, which provides useful evidence but can preserve obsolete or
   inconsistent practice.

When these sources conflict, follow the higher source and correct the lower
source in the same focused change when practical. Do not copy a nearby pattern
only because it already exists.

MQT Core largely follows the [LLVM Coding Standards][llvm-coding-standards]. We
also adopt selected practices from the [Google C++ Style Guide][google-cpp] when
they improve clarity or fit the wider MQT code base. MQT policy resolves
differences between these guides; neither upstream document is imported in full.

## C++ choices

- Use C++20 standard-library facilities before adding a project abstraction or
  dependency.
- Keep variables local, initialize them when declared, and give each name one
  clear meaning.
- Use `auto` when the initializer makes the type clear or when spelling the type
  would hide the important part of an expression. Spell the type when it
  communicates a contract or prevents a surprising conversion.
- Preserve deterministic user-visible output. Do not rely on pointer values or
  unspecified container iteration order.
- Keep cleanup separate from behavioral changes unless the cleanup is required
  to make the behavior correct.
- Do not add flexibility, configuration, or abstraction without a current use.

The [MLIR section](#mlir) explains the deliberate differences for code built on
LLVM and MLIR.

### C++ includes

Order include groups as follows, sorting each group alphabetically:

1. The matching header for the source file.
2. MQT Core headers, including MQT's MLIR headers.
3. Other private project headers.
4. Third-party library headers.
5. Upstream MLIR headers.
6. LLVM headers.
7. System and standard-library headers.

Use quotes for project and third-party headers. Use angle brackets for system
and standard-library headers, or when a library requires them. Keep includes
that depend on macros or declarations at the point where they are needed. For
example, `<qiskit.h>` needs angle brackets to avoid the local `Qiskit.h` on
case-insensitive file systems. Keep its extension function table,
`<qiskit/funcs_py.h>`, in angle brackets so that it sorts after the umbrella
header.

The root `.clang-format` enforces this order. MQT's MLIR headers use the `mqt/`
prefix, for example `"mqt/Dialect/QCO/IR/QCOOps.h"`. Upstream MLIR headers use
`mlir/`, for example `"mlir/IR/MLIRContext.h"`. When adding a dependency,
include its header prefix in the third-party formatter category. Other quoted
includes belong to the private project header group.

### C++ documentation comments

Use `///` for Doxygen documentation comments. The first sentence is the summary;
separate additional paragraphs with a blank `///` line instead of using `\brief`
or `\details`. Document parameters and return values only when the explanation
adds information that the name and signature do not already provide. Preserve
existing documentation when changing comment style.

Prefer Unicode for simple mathematical notation in comments, such as `π`,
`R(θ, φ)`, `U†`, and `|0⟩`. Put spaces around arrows in prose and rewrite
diagrams, as in `QCO → jeff` and `QC ↔ QCO`. Preserve the spelling of code
identifiers, language syntax, and literal output in examples. Use Doxygen math
(`\f$...\f$` or `\f[...\f]`) for formulas that need typeset fractions,
subscripts, or other mathematical layout.

Keep `//!<` or `///<` for trailing member documentation. Keep `/** ... */`
documentation inside backslash-continued macros: line comments there can consume
the following declarations after line splicing. Preserve explicit `@brief`
commands there when Doxygen needs them to retain summaries after macro
expansion.

The `cpp-documentation-style` lint hook checks project-owned C++ files. It
rejects block documentation and explicit summary or detail commands, except on
continued macro lines.

Keep top-level `@file` documentation and put its summary on the next line,
without `@brief`:

```cpp
/// @file Circuit.h
/// Defines the circuit representation.
```

```cpp
/// Returns the number of qubits in the circuit.
[[nodiscard]] size_t getNqubits() const;
```

```cpp
/// Applies an operation to the selected qubits.
///
/// Rejects duplicate indices in \p qubits.
///
/// \param qubits Qubit indices in application order.
/// \returns The created operation.
Operation apply(llvm::ArrayRef<Qubit> qubits);
```

Keep public API documentation in the declaration and do not duplicate it in the
implementation. Use ordinary implementation comments for details that do not
belong to the API contract.

Use `//` for ordinary implementation and namespace closing comments. Inline
`/* ... */` comments remain valid, including unused parameter names such as
`OpAdaptor /*adaptor*/` and argument labels such as `/*isSigned=*/false`.

### Reproduce C++ lint locally

Before pushing a C++ change, run:

```console
uvx nox -s cpp-lint
```

The session configures the `lint` preset and builds `mqt-core-lint-headers` to
prepare generated headers without compiling or linking the project. It then runs
the same `cpp-linter` release and options as CI against every line of each
changed C++ file. It compares against `origin/main` by default. Pass a different
Git diff base after `--` when needed:

```console
uvx nox -s cpp-lint -- upstream/main
```

Use `--all` to check every eligible project C++ file instead:

```console
uvx nox -s cpp-lint -- --all
```

Changed-line `clang-tidy` commands remain useful for quick iteration, but they
do not reproduce CI's whole-changed-file scope. Update this session when the
reusable C++ lint workflow changes its action version or inputs.

## Commit messages

MQT Core adapts Chris Beams's [commit-message guidance][commit-messages] to its
gitmoji convention:

- Start with the established gitmoji prefix and an imperative subject.
- Target 50 characters and never exceed 72 characters, including the prefix.
- Do not end the subject with a period.
- Add a blank line before the body.
- Use the body to explain why the change is needed, its constraints, and any
  non-obvious tradeoffs. Do not restate the diff.
- Wrap prose at 72 characters where practical.
- Preserve legitimate human `Co-authored-by` trailers. Record AI assistance with
  `Assisted-by`, never by representing an AI system as an author.

## Prose and terminology

Apply [Orwell's six rules for writing][orwell] to documentation, comments,
diagnostics, tests, commit messages, and review communication:

1. Avoid familiar metaphors, similes, and figures of speech.
2. Use a short word when it has the same meaning as a long word.
3. Remove every word that does not add meaning.
4. Use active voice when possible.
5. Use everyday English instead of jargon when precision permits.
6. Break a rule before making the text unclear or incorrect.

Apply the relevant [ASD-STE100 Simplified Technical English][ste] principles:
keep sentences short and direct, give each sentence one main idea, use one term
per concept, and prefer explicit nouns to vague pronouns. These are required
style rules, not a claim of formal ASD-STE100 compliance. Use established terms
from quantum computing, LLVM/MLIR, HPC, and computer science. Explain
differences between communities' terms once; do not invent synonyms for variety.

[orwell]: https://www.orwellfoundation.com/the-orwell-foundation/orwell/essays-and-other-works/politics-and-the-english-language/
[ste]: https://www.asd-ste100.org/

Use one established term for one concept. The
{doc}`MQT Core glossary <glossary>` records preferred names, accepted aliases,
and distinctions that matter to public APIs or compiler design. Update the
glossary in the same pull request when introducing or changing a public or
potentially ambiguous term. Do not add entries for ordinary language or private
implementation details.

## Maintenance

Review subsystem policy, agent instructions, formatting, lint rules, and
exceptions when a major dependency changes. Update the recorded upstream version
and remove obsolete exceptions. Existing code does not override a new decision
merely because migration is incomplete.

### Agent guidance

Treat `AGENTS.md` as a concise routing and guardrail layer, not a second copy of
the development policy. Keep rationale, examples, and detailed procedures in the
canonical documentation and link to the applicable sections. Repeat only short,
non-obvious rules that agents must keep in immediate context to avoid a
recurring mistake. Enforce mechanical rules in repository tooling instead of
relying on prose.

Scope instructions to relevant work. Keep skill descriptions short and load
workflow details on demand. Define outcomes and constraints rather than
prescribing each step. Complete authorized work through relevant validation;
repeat checks only for new changes, failures, or unresolved concerns.

Plans and audit files should preserve useful decisions, not log routine tasks.
Benchmark only when requested or needed to answer a concrete performance
question. Keep ad hoc experiments outside the repository unless explicitly
requested.

[commit-messages]: https://chris.beams.io/posts/git-commit/
[google-cpp]: https://google.github.io/styleguide/cppguide.html
[llvm-coding-standards]: https://llvm.org/docs/CodingStandards.html

## Release documentation

Record notable user-facing changes in the current Unreleased section of
`CHANGELOG.md`, with PR references and all contributing human authors. Group
entries by user workflow and subsystem. Fold refinements to functionality that
has never shipped into its feature entry. After release, describe subsequent
changes relative to that published behavior.

Lead major architectural releases with the new program model, enabled workflows,
and consequences for existing users before listing individual changes. Keep the
release overview and migration paths visible from the README and documentation
home page. Preserve PR and contributor references when regrouping entries.

Document breaking changes to released interfaces in both `CHANGELOG.md` and
`UPGRADING.md`. Give the replacement, changed semantics, or lack of a
replacement. Do not add migrations between intermediate unreleased APIs.
Preserve published release sections except when correcting a verified error.

Before a release, reconcile the Git history, merged PRs, and existing entries.
Account for maintenance branches, backports, and frontports; merge dates alone
do not define what ships for the first time. Include relevant feature and fix
PRs, verify contributor attribution, and record why maintenance-only changes do
not need public entries. Check migration examples against the released and
proposed APIs, build the executable documentation, and check its links.

Keep Unreleased until the final version and release date are set. Version tags
drive package metadata; do not edit generated version files to stage a release.
Release preparation does not itself publish a release.

## Documentation validation

Install Python 3.14, LLVM/MLIR 23.1, Doxygen, and Graphviz before building the
complete documentation. The Doxygen configuration is validated with Ubuntu
24.04's version 1.9.8 and version 1.17. Use `uvx nox --non-interactive -s docs`
to build generated references and execute the MyST notebooks. This command fails
on Sphinx and Doxygen diagnostics and checks local links in the generated HTML,
including the native C++ reference. Use
`uvx nox --non-interactive -s docs -- -b linkcheck` to check external links
separately.

Notebook execution is forced on each build. The docs session isolates the QDMI
registry from system, user, project, and inline environment definitions while
retaining packaged devices. Executable examples use local DDSIM and require no
credentials. Configuration recipes for external providers do not execute.

Use `{code-cell}` blocks in pages with MyST-NB front matter. Keep required setup
visible or collapsible with `hide-input`, display useful computed output, and
assert the demonstrated semantics. Use checked subprocess calls for CLI
examples. Ordinary code fences document configuration or interfaces without
executing them. Give figures descriptive alternative text and preserve full
contracts in linked references when they would interrupt a tutorial.

## MLIR

These rules apply to MLIR dialects, transforms, tools, and bindings. MQT Core is
an MLIR consumer; this policy applies even where existing code differs.

Reviewed against LLVM and MLIR **23.1.0**. Revisit this page and the MLIR
clang-tidy configuration on every major LLVM/MLIR upgrade.

### C++ const and IR handles

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

### Passes, verifiers, and rewrites

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

### Linear quantum values

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

### Data structures and performance

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

### Tests

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

### Debugging

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

#### Diagnostics and reproducers

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

### Upstream references

- [MLIR Developer Guide][mlir-developer-guide]
- [MLIR rationale for the usage of `const`][mlir-const]
- [MLIR Testing Guide][mlir-testing]
- [MLIR debugging guide][mlir-debugging]
- [MLIR FAQ][mlir-faq]
- [LLVM Coding Standards][llvm-coding-standards]

[mlir-const]: https://mlir.llvm.org/docs/Rationale/UsageOfConst/
[mlir-debugging]: https://mlir.llvm.org/getting_started/Debugging/
[mlir-developer-guide]: https://mlir.llvm.org/getting_started/DeveloperGuide/
[mlir-faq]: https://mlir.llvm.org/getting_started/Faq/
[mlir-testing]: https://mlir.llvm.org/getting_started/TestingGuide/

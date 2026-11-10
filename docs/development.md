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
- Use LLVM `raw_ostream` for output, `MemoryBuffer` for file input, and
  `StringRef` for borrowed text. Reuse synchronized diagnostics for concurrent
  logging. STL stream headers are forbidden, including in tests. Use GoogleTest
  scoped traces for test context. Keep owned strings where needed; use
  `SmallString` for suitable temporary buffers.
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
relative to configured CMake include roots; parent-directory traversal is
forbidden. Keep includes that depend on macros or declarations at the point
where they are needed. For example, `<qiskit.h>` needs angle brackets to avoid
the local `Qiskit.h` on case-insensitive file systems. Keep its extension
function table, `<qiskit/funcs_py.h>`, in angle brackets so that it sorts after
the umbrella header.

The root `.clang-format` enforces this order. MQT's MLIR headers use the `mqt/`
prefix, for example `"mqt/Dialect/QCO/IR/QCOOps.h"`. Upstream MLIR headers use
`mlir/`, for example `"mlir/IR/MLIRContext.h"`. When adding a dependency,
include its header prefix in the third-party formatter category. Other quoted
includes belong to the private project header group.

### C++ documentation comments

Use `//` for ordinary code comments. Never use `///` for them.

Use `///` for Doxygen documentation. Keep public API documentation at the
declaration; do not duplicate it in the implementation.

Start descriptions with a short sentence stating the entity's purpose or
observable behavior, on one source line where practical. Separate any further
content, including `\param` and `\returns`, with a blank `///` line. Add details
only when they explain something the summary and signature leave unclear.
Separate further paragraphs with blank `///` lines. Do not use `\brief` or
`\details` outside the macro exception below.

Structural commands such as `@defgroup` and `@name` remain valid; marker-only
blocks such as `@{` and `@}` need no summary. Do not add a summary solely to
precede an `@see` cross-reference.

Preserve existing documentation when changing comment style. Inline `/* ... */`
comments remain valid, including unused parameter names
(`OpAdaptor /*adaptor*/`) and argument labels (`/*isSigned=*/false`).

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
continued macro lines. It also rejects lint directives written as documentation
comments.

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

## Python docstrings

Use [Google-style docstrings][google-python-docstrings], consistent with
[PEP 257][pep-257], including in nanobind binding strings and generated stubs.
Start with a short sentence on one line stating the Python API's purpose or
observable behavior. Separate any further content, including `Args:`,
`Returns:`, and `Raises:`, with a blank line. Add input, output, side-effect,
and failure details only when the summary and signature leave them unclear.
Separate further paragraphs and sections with blank lines.

```python
"""Applies an operation to the selected qubits.

Qubit indices follow application order. Duplicate indices raise ValueError.
"""
```

Change binding docstrings in the C++ source and regenerate stubs with
`uvx nox -s stubs`; never edit generated `.pyi` files manually.

[google-python-docstrings]: https://google.github.io/styleguide/pyguide.html#381-docstrings
[pep-257]: https://peps.python.org/pep-0257/#multi-line-docstrings

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

Write for someone who knows the task but has not read the implementation or
followed its design. Prioritize technical accuracy, reader understanding, and
natural flow, in that order. Concision helps when it removes distractions; it
hurts when it removes the connections that make an explanation understandable.

In a guide, start with what the reader can do and why it is useful. Introduce
concepts before using them, connect steps with their reasons, and put examples
near the explanation they support. Use paragraphs for reasoning and lists or
tables for steps and comparisons. Detailed API descriptions should explain how
to use the interface, including ownership, side effects, and failure behavior
that the signature does not convey. Keep specialist implementation details in
the owning reference and link to them when needed.

Integrate additions into the surrounding explanation rather than appending a
record of each feature or fix. Remove duplicated caveats and contrasts with
unreleased designs, but preserve limitations that affect a reader's choices.
Keep prompts, conversations, review instructions, and iteration history out of
public documentation, docstrings, comments, and PR descriptions. A rationale
belongs in the text when it explains the current design; the discussion that led
to it does not. Keep release history and migration advice in their designated
documents.

Apply [Orwell's six rules for writing][orwell] to documentation, comments,
diagnostics, tests, commit messages, and review communication:

1. Avoid familiar metaphors, similes, and figures of speech.
2. Use a short word when it has the same meaning as a long word.
3. Remove every word that does not add meaning.
4. Use active voice when possible.
5. Use everyday English instead of jargon when precision permits.
6. Break a rule before making the text unclear or incorrect.

Use [ASD-STE100 Simplified Technical English][ste] principles where concise,
unambiguous statements help readers look up a fact:

| Kind of text                                    | Writing guidance                                                                                    |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Implementation comments                         | Use short, direct statements to explain a reason or constraint the code cannot show.                |
| API summary or Doxygen brief                    | State the purpose or observable behavior concisely.                                                 |
| Parameter, return, and error descriptions       | Give precise meanings, units, ranges, defaults, and conditions where needed.                        |
| Detailed API descriptions                       | Use natural, connected prose; explain concepts and consequences with examples. STE is not required. |
| Guides, tutorials, and conceptual documentation | Write a coherent explanation for the intended reader. STE is not required.                          |

These are writing guidelines, not a claim of formal ASD-STE100 compliance. Do
not split a connected explanation into isolated sentences just to make each
sentence short. Use pronouns when their reference is clear, and retain familiar
technical terms when a substitute would be less precise.

[orwell]: https://www.orwellfoundation.com/the-orwell-foundation/orwell/essays-and-other-works/politics-and-the-english-language/
[ste]: https://www.asd-ste100.org/

### Terminology and glossary

Prefer established terms from quantum computing, LLVM/MLIR, HPC, and computer
science. Explain specialist language where readers first need it; a glossary
link alone does not introduce a concept. Use a consistent term for each concept,
but do not invent a compound name for an ordinary action or implementation
detail. Preserve project names such as `jeff` and `jeff-mlir`.

The {doc}`glossary` helps readers interpret recurring terms across guides. It is
not an inventory of classes or a place to establish new field-wide definitions.
Before adding an entry, check that readers need it beyond one API description.
For established terms, cite a primary specification, paper, or upstream manual
and keep the general definition separate from MQT Core's particular behavior.
For a necessary MQT-specific concept, label it as such and link to its API or
guide. Do not present class names as accepted scientific aliases or add
"accepted aliases: none" boilerplate. Keep API-object definitions, constructor
requirements, and schema details in their owning references rather than the
glossary. Use “program qubits” for circuit qubits before hardware mapping,
distinguishing placement from allocation and from encoded logical qubits.
Explain each map's domain and direction beside its API. Review technical
references as well as guides for unnecessary shorthand; precision does not
require unexplained compound names.

When terminology is uncertain, record the term, affected uses, source evidence,
and proposed action in an audit for review. Lack of a source is a reason to
investigate, not proof that a term was invented. Coordinate approved terminology
changes across guides, API descriptions, and examples without silently renaming
public symbols.

### Reviewing prose

Read the affected section as a whole after editing it. Check that a newcomer can
identify the task, understand each introduced concept, and follow the example.
Verify claims against the implementation and relevant primary sources; avoid
unsupported assurances. Link to symbols or sections rather than brittle line
numbers. Remove sentences that only answer an earlier review comment or repeat a
nearby rule. Check generated API documentation at its source; do not edit
generated stubs or reference pages. Build the documentation and inspect the
rendered pages as described under
[documentation validation](#documentation-validation).

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

Do not change `CHANGELOG.md` or `UPGRADING.md` except in release-preparation PRs
or when explicitly asked. Feature and fix PR descriptions must explain
user-facing changes, limitations, and required migrations, with relevant labels
for Release Drafter.

During release preparation, record notable user-facing changes with PR
references and all contributing human authors. Group entries by user workflow
and subsystem. Fold refinements to functionality that has never shipped into its
feature entry. After release, describe subsequent changes relative to that
published behavior.

Lead major architectural releases with the new program model, enabled workflows,
and consequences for existing users before listing individual changes. Keep the
release overview and migration paths visible from the README and documentation
home page. Preserve PR and contributor references when regrouping entries.

Document breaking changes to released interfaces and migration instructions in
the PR that introduces them. Give the replacement, changed semantics, or lack of
a replacement. During release preparation, collect these instructions in
`UPGRADING.md` and summarize the changes in `CHANGELOG.md`. Do not add
migrations between intermediate unreleased APIs. Preserve published release
sections except when correcting a verified error.

Before a release, prepare the changelog and upgrade guide from the Git history,
merged PR descriptions, the Release Drafter draft, and any existing Unreleased
notes. Account for maintenance branches, backports, and frontports; merge dates
alone do not define what ships for the first time. Include relevant feature and
fix PRs, verify contributor attribution, and record why maintenance-only changes
do not need public entries. AI tools may help draft entries, but a maintainer
must verify their claims, PR links, and attribution against the release diff and
merged PRs before accepting the draft. Check migration examples against the
released and proposed APIs, build the executable documentation, and check its
links.

Between releases, `CHANGELOG.md` and `UPGRADING.md` contain only published
releases. Create their next sections during release preparation and finalize the
version and release date before publishing. Version tags drive package metadata;
do not edit generated version files to stage a release. Release preparation does
not itself publish a release.

## Testing priorities

Prioritize workflows that users run: import from an SDK or program format,
compile for a device, submit, and interpret the result. Exercise the real local
compiler and DDSIM device where possible. Use an independent SDK simulator or a
known analytical result as the oracle. Provider mocks belong at unavailable
external boundaries and for failures that a local device cannot reliably
produce. Keep remote QPU tests opt-in.

Extend an existing workflow before adding a test. A regression is a reason to
improve an oracle or input, not automatically a new test function or file. Keep
direct unit tests when they protect a distinct contract that workflows cannot
observe clearly or cheaply, such as numerical limits, ownership, error recovery,
or bounded resource use. Do not fabricate impossible internal states merely to
increase coverage. Investigate whether unreachable branches should exist.

Assign each contract to its owning layer. C++ tests cover native semantics;
Python tests cover SDK behavior, argument conversion, exceptions, object
lifetime, and the complete public workflow. An independent Python oracle can
justify a native semantic check, but duplicating the native case matrix cannot.
Every Python case runs across multiple interpreters and dependency resolutions.

Check results, phase, logical wire and output order, and required resource
bounds. Avoid private helper calls, exact IR text, chosen layouts, and rewrite
sequences unless they are the public contract. For an efficiency requirement,
prefer an upper bound on work or size to an exact operation count or a
wall-clock timeout.

Use the smallest circuit, shot count, array, and parameter set that
distinguishes the behavior. Choose representative cases for different paths
instead of Cartesian products of unrelated options. Preserve a large input only
when its size crosses a relevant limit or exposes a complexity defect. Keep
statistical tolerances well above sampling noise and justify expensive sampling
budgets.

Share expensive setup and conversion work when several independent observations
fit in one small program. Keep each result separately observable and failures
easy to identify; combining test names without reducing work adds no value.

Coverage is a diagnostic for missing workflows, not a requirement to exercise
every branch. Aim for roughly 90% overall; inspect meaningful losses rather than
adding tests to satisfy a patch percentage. When pruning, identify the retained
oracle and any lost contract, then compare collection, runtime, and coverage on
the same build. Fewer test names alone do not mean less test work.

## Python test selection

`pytest` runs package tests by default. Repository-script tests use the
`linkcheck` and `qiskit_c_api` markers registered in `pyproject.toml`. In a
source checkout, use `pytest -m linkcheck` to select one suite or `pytest -m ""`
to run all suites. The command-line marker expression replaces the default
selection, including when selecting a test file explicitly. Nox uses the same
default selection; pass `nox -s tests -- -m linkcheck` to run a script suite.

## Documentation validation

Install Python 3.14 and Doxygen before building the complete documentation.
LLVM/MLIR follows the {ref}`source-build SDK setup <setting-up-mlir>`. The
Doxygen configuration is validated with Ubuntu 24.04's version 1.9.8 and version
1.17. Use `uvx nox --non-interactive -s docs` to build generated references and
execute the MyST notebooks. This command fails on Sphinx and Doxygen diagnostics
and checks local links, scripts, and stylesheets in the generated HTML,
including the native C++ reference. Use
`uvx nox --non-interactive -s docs -- -b linkcheck` to check external links
separately.

Notebook execution is forced on each build. The docs session isolates the QDMI
registry from system, user, project, and inline environment definitions while
retaining packaged devices. Executable examples use local DDSIM and require no
credentials. Configuration recipes for external providers do not execute.

Use `{code-cell}` blocks in pages with MyST-NB front matter. Keep required setup
visible or collapsible with `hide-input` and display useful computed output. Use
assertions when they check a property that the displayed output does not make
clear. Link API names to their references, including external APIs when
intersphinx provides them. Use checked subprocess calls for CLI examples.
Ordinary code fences document configuration or interfaces without executing
them. Give figures descriptive alternative text and preserve full contracts in
linked references when they would interrupt a tutorial.

The **Documentation** GitHub Actions workflow builds each PR's head commit and
each main-branch or release-tag commit. It runs the same docs session and saves
the checked HTML, an offline ZIP, and a source manifest as one artifact. The
session reports dependency installation, package building, and rendering times.
Compiler caching does not skip notebook execution or documentation checks.

The **Publish documentation** workflow runs from the trusted default branch. It
downloads only the triggering run's artifact, verifies its commit, and checks
that the PR or Git ref still points to that commit. It uploads static files
through the pinned [Read the Docs upload client][rtd-direct-upload]; it does not
run artifact code. Fork PRs use the same isolated build path. Closed PRs,
superseded commits, merge-group builds, and manual validation runs are not
published.

### Activate artifact publication

Publication requires the following one-time project setup:

1. Keep the existing repository-backed project mode so its tracked `latest`
   version and existing version settings remain available. Direct upload is
   enabled for `mqt-core`.
2. Add an administrator's RtD API token as the GitHub Actions secret
   `READTHEDOCS_TOKEN`, then set the repository variable
   `READTHEDOCS_UPLOAD_ENABLED` to `true`. The token is used only by the trusted
   publisher. Do not expose it to the documentation build.
3. After the workflows are on main, verify one branch upload, one tag upload,
   and a PR preview, including the `latest` and `stable` URLs and offline ZIP.
   The publisher runs only after a successful build.
4. Replace the required `docs/readthedocs.org:mqt-core` status in the main
   ruleset with the GitHub Actions check **Build and validate documentation**.
   Keep the other required checks. This removes the RtD queue from the merge
   gate while retaining executable documentation validation.

Before activation, GitHub Actions still builds and retains artifacts; the
existing `.readthedocs.yaml` continues to serve source builds. RtD stops source
builds for a version after its first direct upload. Old versions retain their
own source-build configuration. To roll back a migrated version, disable the
publisher variable and clear **Uploaded** in that version's RtD settings; RtD
then rebuilds it from source. Reinstating the RtD required status also restores
the old merge gate.

[rtd-direct-upload]: https://docs.readthedocs.com/platform/stable/guides/build/direct-upload.html

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
  // Use operand without implying that the IR graph is immutable.
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
owns this whole-IR check and positional quantum state correspondence through
`scf.for`, both `scf.while` regions, `qco.if`, and `qco.index_switch`. Functions
return their quantum arguments as trailing results in argument order. Classical
state can change shape across `scf.while` regions. Ordinary MLIR operation
verification alone does not establish these QCO invariants. Builders and
transformations must preserve the invariant, and public QCO pipeline boundaries
must validate it.

Rewrites on valid QCO IR can use `*value.user_begin()` to obtain the sole
consumer, or `*value.use_begin()` for its `OpOperand` and operand number. Do not
repeat `hasOneUse()` guards in these rewrites. Keep linearity checks in the
verifier and optional debug assertions at internal boundaries. This rule does
not apply to QC references or classical SSA values, and permits relying on the
positional correspondence checked by `qco::verifyLinearity`. Exactly-one-use and
wire correspondence remain separate checks.

When forwarding a linear value requires deleting its producer, use a rewrite
that removes both operations. A fold can only change its root; do not rely on
later dead-code elimination to restore linearity before other rewrites run.

### Data structures and performance

Use LLVM views and abstract range types at MLIR-facing boundaries. Prefer an
LLVM data structure such as `SmallVector`, `DenseMap`, or `MapVector` when its
storage, lookup, ordering, or API behavior provides a concrete benefit. Keep a
standard-library type when it already expresses the required contract.

Follow [LLVM's `SmallVector` guidance][llvm-smallvector]: use `SmallVector<T>`
without an explicit inline capacity by default. Retain `SmallVector<T, N>` when
a known bound or measured storage need justifies `N`, such as synthesis results
containing one or two qubits. `N` sets inline storage, not a size limit. Keep
explicit capacities required by LLVM for large element types. Comment only on
reasons that are not clear from the type, contract, or nearby code. Keep the
rationale near the owning type or declaration; do not repeat it at each use or
restate how inline capacity works. Consider `std::array` for exactly fixed-size
data; do not change unrelated APIs merely to use it.

For parameters that borrow a sequence, use `ArrayRef<T>` or a suitable MLIR
range when only reading elements, `MutableArrayRef<T>` when changing elements in
place, and `SmallVectorImpl<T>&` when resizing or appending. These interfaces do
not depend on inline capacity. A function that transfers ownership may use a
concrete container type. Audit unexplained capacity choices individually; do not
replace explicit capacities in bulk or claim performance gains without evidence.

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
- Expose behavioral fixes through an existing workflow when possible; add a
  focused regression only for a contract it cannot cover.

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
8. Use the reduced case to strengthen an existing workflow or, where necessary,
   add a small direct regression test.

Build the driver with the configured LLVM/MLIR installation:

```sh
cmake --preset release
cmake --build --preset release --target mqt-core
mqt_core=build/release/bin/mqt-core
```

Save this reduced example as `reduced.mlir`:

```mlir
module {
  func.func @f(%q: !qco.qubit) -> !qco.qubit {
    %h0 = qco.h %q : !qco.qubit
    %h1 = qco.h %h0 : !qco.qubit
    return %h1 : !qco.qubit
  }
}
```

Run exactly the selected pipeline with `--run-pipeline`:

```sh
"$mqt_core" compile reduced.mlir --run-pipeline \
  --pass-pipeline='builtin.module(canonicalize)'
"$mqt_core" compile reduced.mlir --run-pipeline \
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
"$mqt_core" compile unsupported.mlir --run-pipeline \
  --pass-pipeline='builtin.module(qc-to-qco)' \
  --mlir-print-ir-before-all --mlir-print-ir-after-failure \
  --mlir-disable-threading \
  --mlir-pass-pipeline-crash-reproducer=failure.mlir
"$mqt_core" compile failure.mlir --run-reproducer --mlir-print-ir-before-all
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
[llvm-smallvector]: https://llvm.org/docs/ProgrammersManual.html#llvm-adt-smallvector-h

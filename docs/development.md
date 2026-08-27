# Development policy

This page defines MQT Core's repository-owned development policy. It supplements
the {doc}`contribution guide <contributing>`, which describes the contribution
process, and the {doc}`AI usage policy <ai_usage>`, which defines accountability
for AI-assisted work.

## Sources of authority

Apply guidance in this order:

1. This policy and any MQT Core subsystem policy, such as the
   {doc}`MLIR development policy <mlir/development>`.
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

The {doc}`MLIR development policy <mlir/development>` explains the deliberate
differences that apply to code built on LLVM and MLIR.

### C++ documentation comments

Use `///` for Doxygen documentation comments. Do not use `/** ... */`. The first
sentence is the summary; separate additional paragraphs with a blank `///` line
instead of using `\brief` or `\details`. Document parameters and return values
only when the explanation adds information that the name and signature do not
already provide.

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

### Reproduce C++ lint locally

Before pushing a C++ change, run:

```console
uvx nox -s cpp-lint
```

The session configures and builds the `lint` preset, then runs the same
`cpp-linter` release and options as CI against every line of each changed C++
file. It compares against `origin/main` by default. Pass a different Git diff
base after `--` when needed:

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

## Terminology

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

[commit-messages]: https://chris.beams.io/posts/git-commit/
[google-cpp]: https://google.github.io/styleguide/cppguide.html
[llvm-coding-standards]: https://llvm.org/docs/CodingStandards.html

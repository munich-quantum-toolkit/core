# Documentation and comment quality audit

Status: applied. Baseline: `82a8f3a85`, initially clean. Date: 2026-09-11.
Scope: repository documentation, docstrings, code comments, and compact
prevention guidance in `AGENTS.md`.

## Result

Removed repetitive implementation walkthroughs and obvious narration, corrected
stale API claims, and replaced change history with current constraints. The
changes preserve executable code, signatures, tests, and build commands.

The most consequential corrections are:

| Location                             | Finding and applied change                                                                                                                                                                                                                   |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `QIRProgramBuilder`                  | Constants emit LLVM operations, allocation depends on the selected profile, and the four-block layout describes this builder's Base implementation. Replaced contradictory examples and clarified measurement, reset, and register behavior. |
| `dd::Package::expectationValue`      | The implementation throws `std::invalid_argument` for an oversized observable and asserts realness in debug builds. Removed claims of `std::runtime_error` and garbage collection.                                                           |
| `QCOToQC` and `QCToQCO`              | Replaced repeated conversion narratives with the actual state, allocation, wire-correspondence, and region-ordering responsibilities. QCO-to-QC does retain lowering state.                                                                  |
| Qiskit backend guide                 | Removed multi-controlled gate names absent from the backend's gate lookup; documented its supported names and aliases.                                                                                                                       |
| DD guide and implementation comments | Distinguished normalized outgoing weights from the incoming factor, removed a blanket input-size complexity claim, and removed incorrect reverse-traversal comments. Kept the operation-specific complexity limits and executable examples.  |
| Windows package initialization       | Described the DLL search path used by `os.add_dll_directory`, rather than the process `PATH`.                                                                                                                                                |

The remaining cleanup removes trivial conversion examples, repeated DD algorithm
walkthroughs, CMake command narration, and test comments that retell a fix
rather than explain the invariant. Binding docstrings were edited at their
source and the corresponding DD stub was regenerated.

`AGENTS.md` now asks for current contracts and reasons, bans unsupported
assurances and change narration in code/API docs, and explicitly preserves
useful summaries, examples, ownership, numerical limits, and workaround reasons.

## Coverage and method

The baseline inventory contains 1,090 tracked files: 1,048 first-party text
files, 10 template-managed files, 17 generated stubs, 11 vendored files, and
four binary files. Repository-wide text screening covered documentation and
comment anti-patterns. Extracted 8,320 comment/docstring blocks from 507 files
for length, repetition, history, and hedging checks; binding raw string
docstrings and Markdown prose were reviewed separately. Candidates were checked
against surrounding code before editing. This is repository-wide screening with
contextual review, not a proof of every statement in the documentation.

[Windbag](https://github.com/scale-venture-partners/windbag) supplied the
initial anti-pattern categories. Its Python-oriented check was used as a
candidate finder, alongside C++/TableGen and prose screening. The baseline
command `uvx --from windbag==0.1.1 windbag check --all --json` reported 89
candidates: 86 verbose blocks, two obvious comments, and one hedge. The verbose
findings were dominated by license headers, so their count is not a measure of
first-party documentation quality. The final run reported 81 verbose-block
flags: 71 license headers and 10 retained metadata or technical-context blocks.
No obvious-comment or hedge flags remain. No linter dependency was added.

Deliberately retained:

- Public API summaries and examples that explain types, ordering, units,
  ownership, failure behavior, or supported input limits.
- Mathematical derivations, phase-sensitive regression reasons, reference-count
  invariants, and explanations of active workarounds and suppressions. Concrete
  TODOs keep their removal conditions; an unqualified parallelization plan was
  removed.
- Changelog and migration history, durable audit/plan decisions, published-paper
  text, and attribution/license headers. History is appropriate in those places.
- Generated and externally maintained content. Generated references were
  rebuilt; generated stubs were changed only by the supported regeneration
  session.

## Validation

- `uvx nox -s lint`: passed. Earlier runs applied formatting; the final run
  passed without modifications.
- `uvx nox -s stubs`: passed. Only `python/mqt/core/dd.pyi` changed, with the
  intended docstring updates and no signature changes.
- `uvx nox --non-interactive -s docs`: passed after the final source changes.
  Built the generated MLIR and native C++ references, executed all 11 MyST
  notebooks, and passed `scripts/check_docs_links.py` on generated HTML.
- `uvx nox --non-interactive -s docs -- -b linkcheck`: passed with the
  repository's configured skips. A GitHub rate limit cleared on retry. The
  initially rate-limited uv contribution-guide link was also confirmed through
  GitHub's contents API at its pinned revision.
- C++ lint: `uvx nox -s cpp-lint -- HEAD` prepared the generated headers, but
  its commit-to-commit selector omitted uncommitted files. This empty selection
  was not counted as validation. The installed `cpp-linter` then ran the same CI
  checks explicitly on all 31 eligible changed source/test files, with zero
  findings. It used `--files-changed-only=false`, `--lines-changed-only=false`,
  `--ignore=*|!<changed-file>...`, the `build/cpp-lint` database, and the
  session's Clang 23 and extra-argument settings. Public include directories
  retain the CI exclusion; the documentation/stub builds compiled their
  consumers.
- Baseline comparison: all 37 changed C++ token streams match after removing
  comments and binding documentation strings. All eight changed Python/stub
  syntax trees match after removing docstrings. All 33 changed CMake files
  retain their non-comment content, and all three TableGen files retain their
  definitions outside descriptions. Executable notebook cells are unchanged.
- `git diff --check`: passed. Generated notebook scratch files were removed.

The audit does not claim a new behavioral or performance result. Production
algorithms and test assertions are unchanged; a full standalone runtime test
suite was not needed for these documentation-only edits.

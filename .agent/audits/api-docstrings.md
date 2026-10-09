# API comment and docstring audit

Status: fixes applied. Original audit: 2026-10-09, baseline
`afe3809a615cc0e8edc852282b9e5fe380df66b9`. Review: 2026-10-10, rebased onto
`81c570c6884ab1e9ab70d214369cf848aa943345`.

## Result

Apply the merged
[C++ documentation policy](../../docs/development.md#c-documentation-comments)
and [Python docstring policy](../../docs/development.md#python-docstrings) to
existing comments and docstrings. This follows
[issue #2671](https://github.com/munich-quantum-toolkit/core/issues/2671).
PR #2578 already supplied a purpose summary for `normalizeGlobalPhases`; that
change required no further fix.

## Scope and method

Screened all 676 tracked first-party source files with relevant extensions: 549
C++ files, 25 TableGen files, 81 Python files, and 21 generated stub files.
Headers, implementations, bindings, benchmarks, tests, tools, scripts, and
configuration were included. Vendored code and generated build output were
excluded from the source inventory.

Python's AST supplied docstrings. A C++ lexer supplied comments and binding
literals; adjacent string literals were joined before checking their contents.
TableGen description fields were excluded from comment screening so fenced
examples were not mistaken for source comments. Their separate summary and
description fields already provide the required structure.

Screening covered the full inventory. Context review checked implementation
comments, test headings, summaries made only of detail commands, binding
paragraphs, qualified definitions, and longer first paragraphs. Doxygen XML and
native and Python API HTML supplied rendering evidence. This audit checks
comment policy; it does not establish every documented semantic contract or
require documentation for every undocumented declaration.

## Applied findings

- Convert ordinary implementation notes, test headings, namespace comments, and
  lint directives to `//`. In mixed blocks, retain the API documentation and
  convert only the directive. Previously, Doxygen used the C ABI naming
  rationale as the brief for `MQT_CORE_QDMI_driver_add_manifest_v1` and rendered
  its `NOLINTBEGIN` directive in the detailed description.
- Add short purpose summaries to 14 blocks that began with detail commands.
  Reference-only `@see` comments are valid Doxygen and need no added summary.
  The baseline XML had 48 documented members with empty briefs: 47 used `@see`,
  and the singleton getter in `qdmi/common/Common.hpp` needed a purpose summary.
- Separate summaries from detail commands in 339 blocks. Separate further
  sentences and lists from summaries; shorten unclear or lengthy summaries while
  retaining numerical limits, preconditions, references, and attribution.
- Keep the public contracts for `hasCompleteTensorLifetime`,
  `Statistics::toString`, and `PowOp::getUnitaryMatrix` at their declarations.
  Remove the duplicate or displaced implementation documentation.
- Add summary separators to eleven binding docstrings, shorten the summary for
  `QCOProgram.decompose_multi_controlled`, and regenerate stubs from the binding
  source. Authored Python docstrings needed no changes.
- Join the affected prose paragraphs and let `clang-format` choose line breaks.
  Format embedded C++ comment paragraphs separately when TableGen formatting
  treats them as strings. Preserve paragraph separators, lists, command blocks,
  and fenced examples. Separate the allocation summaries from their insertion
  point requirements in three builder comments.
- Extend the existing comment-style hook to reject `NOLINT` directives anywhere
  in a `///` comment, including after prose. Prose intent and summary quality
  still require context review; the temporary audit scanner is not a new
  repository tool or a sentence-counting rule.
- Correct the QIR array argument descriptions and the Hadamard-lifting control
  selection rule to match their implementations.

Preserve structural Doxygen markers, trailing member documentation, and block
comments inside continued macros. The local `Trial::score` member in
`Mapping.cpp` remains documentation even though it is declared inside a
function. Wrapped single sentences and abbreviations are not defects by
themselves.

## Original audit validation

- `uvx nox -s stubs`: final rerun passed; the generated diff contains only
  eleven changed docstrings in three stub files.
- `prek run cpp-documentation-style --all-files`: passed with the extended hook.
- Final source screening permits `@see` comments and found no missing summaries,
  missing separators, multi-sentence binding summaries, or Doxygen lint
  directives. Its remaining candidates are two abbreviations and the local
  `Trial::score` member above.
- Token comparisons preserved code in all 212 changed C++ and TableGen files;
  only eleven binding docstrings differ, including one shortened summary. AST
  comparisons preserved the APIs of the three changed generated stubs. All 56
  trailing documentation blocks and 81 macro documentation blocks are unchanged.
- `uvx nox -s cpp-lint -- --all`: checked 412 eligible files, including all 143
  eligible changed C++ files. Its nine findings concerned includes, exception
  escape, and casts; each reproduced on the unchanged baseline with the same
  dependencies and generated headers. Whole-file analysis of the final changes
  reported six of those baseline findings. Direct builder-header checks also
  reproduced their diagnostics on the baseline; no unrelated fixes were added.
- `uvx nox --non-interactive -s docs`: passed, including notebooks and internal
  links. Rendering preserves the 47 reference-only entries, the singleton and
  driver-registration summaries, and eleven binding summary paragraphs. Lint
  directives do not render.
- `uvx nox -s lint`: final rerun passed.
- Hook counterexamples passed for lint directive forms, ordinary comments,
  trailing documentation, and continued macro exceptions.

## Rebased validation

Repository lint and whole-file C++ lint passed (143 changed files, no findings).
Native Doxygen generation, nine hook counterexamples, C/C++ compatibility of QIR
declarations, and binding/stub docstring comparisons passed. Native tests, full
Sphinx builds, and stub regeneration were not repeated during this review.

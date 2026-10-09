# API comment and docstring audit

Status: fixes applied and validated; C++ lint has baseline findings. Date:
2026-10-09. Baseline: `afe3809a615cc0e8edc852282b9e5fe380df66b9`. The baseline
includes merged
[PR #2723](https://github.com/munich-quantum-toolkit/core/pull/2723). The
working tree was clean at the start.

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

The baseline inventory contained 4,762 Doxygen line-comment blocks, 56 trailing
member documentation blocks, 81 macro documentation blocks, 1,251 authored
Python docstrings, 597 stub docstrings, and 587 binding docstring literals.
Counts are source occurrences, not independent API defects.

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
  Remove the 49 summaries initially added before cross-references, including two
  type comments that were not present in the baseline XML.
- Separate summaries from detail commands in 339 blocks. Separate further
  sentences and lists from summaries; shorten unclear or lengthy summaries while
  retaining numerical limits, preconditions, references, and attribution.
- Keep the public contracts for `hasCompleteTensorLifetime`,
  `Statistics::toString`, and `PowOp::getUnitaryMatrix` at their declarations.
  Remove the duplicate or displaced implementation documentation.
- Add summary separators to eleven binding docstrings, shorten the summary for
  `QCOProgram.decompose_multi_controlled`, and regenerate stubs from the binding
  source. Use raw C++ string literals for these docstrings; their values are
  unchanged by the follow-up. Authored Python docstrings needed no changes.
- Join the affected prose paragraphs and let `clang-format` choose line breaks.
  Format embedded C++ comment paragraphs separately when TableGen formatting
  treats them as strings. Preserve paragraph separators, lists, command blocks,
  and fenced examples. Separate the allocation summaries from their insertion
  point requirements in three builder comments.
- Extend the existing comment-style hook to reject `/// NOLINT` directives.
  Prose intent and summary quality still require context review; the temporary
  audit scanner is not a new repository tool or a sentence-counting rule.

Preserve structural Doxygen markers, trailing member documentation, and block
comments inside continued macros. The local `Trial::score` member in
`Mapping.cpp` remains documentation even though it is declared inside a
function. Wrapped single sentences and abbreviations are not defects by
themselves.

## Validation

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
  eligible changed C++ files, and reported nine findings. All nine were
  reproduced by rerunning their translation units from an archive of the
  unchanged baseline with the same dependencies and generated build inputs.
  These concern includes, exception escape, a widening cast, and an enum cast;
  no unrelated code fixes were added. The default mode selected no files because
  these changes were not committed, so its result was not used.
- Follow-up `uvx nox -s cpp-lint`: checked 143 units with whole-file analysis,
  including all 28 eligible changed C++ files in this follow-up. Its six
  findings match the warnings already reproduced on the unchanged baseline.
- Direct whole-header checks for the two builders were rerun after the final
  paragraph edits. Their source and generated-header diagnostics also reproduce
  on the unchanged baseline.
- Whole-file `clang-tidy` for `bindings/mlir/register_mlir.cpp` was rerun after
  the final docstring edit; it reported no findings in project-owned source.
- `uvx nox --non-interactive -s docs`: both audit and follow-up passed,
  including notebooks and internal links. The follow-up rendering check
  preserves the 47 reference-only native entries without adding briefs. The
  singleton getter and driver registration entry point retain their purpose
  summaries, and lint directives do not render. All eleven binding summaries
  remain separate Python paragraphs.
- `uvx nox -s lint`: final rerun passed.
- Hook counterexamples passed for lint directive forms, ordinary comments,
  trailing documentation, and continued macro exceptions.

Follow-up checks: stub regeneration passed and all generated stubs are
byte-for-byte unchanged from the first audit commit. Token comparisons preserve
all code and string values in the 67 changed C++ and TableGen files. Exactly
eleven audited docstrings now use raw literals. Trailing and continued-macro
documentation remains unchanged.

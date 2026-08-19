# Add compile-time fixed-width OpenQASM angles

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core currently recognizes the OpenQASM `angle` keyword but rejects angle
declarations. After this change, users can declare initialized fixed-width
angles, cast finite constant float expressions to them, use the supported
compile-time angle arithmetic, and pass the results to quantum gates. MQT Core
honors the fixed-width modulo arithmetic before it lowers the result to the
existing binary64 radians gate interface.

A user can observe the feature by importing `angle[8] theta = pi / 2;` followed
by `rx(theta) q;`. The QC program contains the binary64 value of pi divided by
two. Exporting that QC program produces valid OpenQASM 3.1 with a floating-point
gate argument. The export does not reproduce the source declaration or attach
OpenQASM-specific metadata to MLIR.

## Progress

- [x] (2026-08-19 11:59Z) Rechecked current `origin/main`, repository policy,
      the old pull-request stack, the live OpenQASM 3.1 rules, and the angle
      representations used by common quantum frameworks.
- [x] (2026-08-19 11:59Z) Added private parser syntax for angle declarations,
      width designators, and angle casts.
- [x] (2026-08-19 11:59Z) Added exact compile-time quantization, resizing,
      modular arithmetic, comparisons, and the binary64 lowering boundary.
- [x] (2026-08-19 11:59Z) Added focused frontend and canonical export tests; the
      new tests pass.
- [x] (2026-08-19 12:17Z) Updated the OpenQASM support documentation and
      completed release, non-unity, test, documentation, and lint validation.
- [x] (2026-08-19 12:17Z) Reviewed the final nine-file change and recorded the
      outcome. No remote state was changed.
- [x] (2026-08-19 12:37Z) Rebased the signed implementation on current `main`,
      including a non-overlapping workflow update that landed during
      publication. Opened draft pull request #2169 and folded its reference into
      the existing unreleased OpenQASM changelog entry.
- [x] (2026-08-19 13:23Z) Resolved all four Clang-Tidy 22.1.8 findings from the
      first CI run. The exact checks, release and non-unity builds, and both
      affected test suites pass locally.

## Surprises & Discoveries

- Observation: The typed OpenQASM frontend and QC emitter already represent gate
  parameters as binary64 radians. An angle constant can therefore become an
  ordinary `ScalarType::Angle` constant at the end of semantic analysis without
  changing the public frontend types or the QC dialect. Evidence: the focused
  export test emits `rx(1.5707963267948966)` and reparses it in strict mode.
- Observation: The source width is needed only while constant expressions are
  evaluated. Keeping a private pair of residue and width removes the need for
  source-format attributes, integer angle operations in QC, and pattern
  reconstruction in the exporter.
- Observation: Binary64 cannot distinguish every adjacent fixed-angle residue at
  width 53 near two pi. Width 52 has a step larger than the largest binary64
  spacing in the represented interval. The supported range is therefore 1
  through 52, and an omitted width resolves to 52.
- Observation: OpenQASM defines `pi`, `tau`, and `euler` as 64-bit floats, and
  inverse trigonometric functions return floats. The previous frontend treated
  some of these values as angles. The fixed-angle work exposes and corrects that
  mismatch.
- Observation: The live specification first describes binary angle operations on
  equal-width operands, then states that mixed-width angles use unsigned integer
  promotion. The implementation widens both fixed values before it applies the
  operation.
- Observation: The normal documentation build needs a QDMI 1.3.2 tag file from
  GitHub Pages. The first build failed during a temporary DNS outage. Reusing
  the identical cached 1.3.2 tag file made the warnings-as-errors build pass.

## Decision Log

- Decision: Support fixed-width angles only as compile-time values. Rationale:
  the requested use case does not require runtime angle storage, and the
  existing backend interfaces use binary64 gate parameters. Date/Author:
  2026-08-19 / Codex.
- Decision: Keep fixed-angle state private to
  `mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp`. Rationale: no other format
  or dialect needs the OpenQASM storage width when the selected output contract
  is canonical semantic output rather than source round-trip. Date/Author:
  2026-08-19 / Codex.
- Decision: Accept widths 1 through 52 and use 52 for unsized angles. Rationale:
  every supported residue remains distinguishable after conversion to the
  existing binary64 radians boundary. Date/Author: 2026-08-19 / Codex.
- Decision: Use round-to-nearest, ties-to-even for float conversion and angle
  narrowing. Widening appends zero low bits. Rationale: this is one of the
  OpenQASM-defined narrowing policies and is the required float-to-angle rule.
  Date/Author: 2026-08-19 / Codex.
- Decision: Treat initialized declarations with or without `const` as write-once
  compile-time bindings. Rationale: this accepts common source while avoiding
  runtime storage and assignment. Reassignment and missing or dynamic
  initializers receive diagnostics. Date/Author: 2026-08-19 / Codex.
- Decision: Leave bare gate parameters as binary64 and defer global gate-angle
  quantization. Rationale: other quantum formats use continuous floating-point
  parameters, and unconditional modulo reduction is unsafe for phase-sensitive
  or controlled operations. Date/Author: 2026-08-19 / Codex.

## Outcomes & Retrospective

The implementation is published as draft pull request #2169. The source, test,
and feature-documentation diff changes eight files with 577 insertions and 65
deletions. The changelog and ExecPlan bring the pull request to ten files. The
old pull-request stack changed 30 files with about 6,000 insertions. The smaller
design needs no public frontend change, no MLIR dialect or operation, no
format-specific attribute, no exporter reconstruction, and no Python or
generated-file change.

The release build passed. CTest passed all 4,104 tests; one device query test
was skipped by its fixture. The complete OpenQASM target and QC translation
binaries passed 168 and 173 tests. The same two targets built without unity and
their test binaries passed. The repository lint session and the warnings-as-
errors documentation build passed. `git diff --check` passed.

The first ordinary documentation run failed only because DNS resolution could
not download the QDMI 1.3.2 tag file. The successful retry used the identical
cached tag file from another current MQT Core worktree through Sphinx's normal
configuration override. The existing unreleased staged-OpenQASM changelog entry
now references pull request #2169. The entry already credits the contributing
authors.

The first CI lint run found four warnings in the new code: one signed/unsigned
comparison, two nested conditional expressions, and one redundant cast. The
follow-up uses `std::cmp_greater`, explicit branches, and the literal exponent
type. These changes preserve behavior. The four exact checks pass with
Clang-Tidy 22.1.8, and both affected test binaries pass in release and non-unity
builds.

## Context and Orientation

The parser in `mlir/include/mlir/Target/OpenQASM/Detail/OpenQASMParser.h`
creates a private syntax tree declared in `OpenQASMSyntax.h` and copied by
`mlir/lib/Target/OpenQASM/OpenQASMSyntax.cpp`. Semantic analysis in
`mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp` converts that syntax into the
public typed frontend. The QC emitter already maps typed angle expressions to
binary64 MLIR values. The reverse exporter maps binary64 values to OpenQASM
`float`, which is valid as a gate argument because gate arguments can promote to
the language's unknown-width angle type.

An OpenQASM fixed angle of width N is an unsigned N-bit residue k whose value is
two pi times k divided by 2 to the power N. Float conversion first reduces the
value modulo the binary64 representation of two pi and then selects the nearest
residue, with ties selecting an even low bit. Mixed-width angle operands widen
to the larger width before arithmetic or comparison. Multiplication and division
use an integer literal that can be represented at the angle width. These
operations and unary negation use unsigned modular arithmetic. Comparisons use
the unsigned residue order.

This task supports initialized declarations, casts from finite constant floats,
angle-to-angle resizing, unary negation, angle addition and subtraction,
nonnegative integer-literal multiplication, positive integer-literal division,
comparisons, `sin`, `cos`, `tan`, and gate arguments. It does not support angle
inputs or outputs, mutable or runtime angles, bit casts, bitwise operations,
shifts, rotations, population count, angle divided by angle, or a global
quantization pass.

## Plan of Work

Extend the private parse vocabulary with an angle scalar kind and an angle-cast
expression. Preserve an optional width expression on declarations and casts so
semantic analysis can evaluate named constant widths.

Inside semantic analysis, store fixed constants as a private `FixedAngle` value
containing an unsigned residue and a width. Use integer arithmetic to quantize
every finite binary64 input exactly relative to the binary64 value of two pi.
Keep every operation in the residue domain. Convert to radians only when a
folded constant enters the existing typed frontend.

Do not change `mlir/include/mlir/Target/OpenQASM/Frontend.h`, QC, QCO, QIR,
Python bindings, generated stubs, or the QC-to-OpenQASM exporter. Document the
compile-time input subset and the canonical floating-point output boundary in
`docs/mlir/OpenQASM.md`.

Add parser and semantic tests under `mlir/unittests/Target/OpenQASM/`. Add one
end-to-end import/export test under `mlir/unittests/Dialect/QC/Translation/`.
The tests must cover the specification halfway example, negative and multi-turn
conversion, widths 1 and 52, omitted width, widening and narrowing, large and
subnormal binary64 inputs, modular arithmetic, comparisons, trigonometric use,
unsupported runtime behavior, and strict reparsing of canonical output.

## Concrete Steps

Run all commands from the repository root. Build and run the focused frontend
tests first:

    cmake --build --preset release --target mqt-core-mlir-unittest-openqasm-target
    ./build/release/mlir/unittests/Target/OpenQASM/mqt-core-mlir-unittest-openqasm-target

Build and run the QC translation tests:

    cmake --build --preset release --target mqt-core-mlir-unittest-qc-translation
    ./build/release/mlir/unittests/Dialect/QC/Translation/mqt-core-mlir-unittest-qc-translation

Then complete release, documentation, and repository validation:

    cmake --build --preset release
    ctest --preset release
    uvx nox --non-interactive -s docs
    uvx nox -s lint
    git diff --check

## Validation and Acceptance

The OpenQASM target tests must pass and show that a width-8 halfway input rounds
to the even residue, negative and multi-turn inputs reduce modulo two pi, mixed
widths widen correctly, narrowing uses ties-to-even, and omitted width retains
the width-52 low step. Width 0, width 53, missing initializers, dynamic values,
reassignment, angle division, and oversized integer literals must fail with a
diagnostic.

The QC translation test must import a fixed angle, export the QC program, find
no `angle` declaration or `mqt.openqasm` attribute in the output, observe the
expected binary64 gate argument, and reparse the emitted OpenQASM 3.1 in strict
mode. Existing OpenQASM, compiler, and repository tests must remain green.

## Idempotence and Recovery

Builds and tests write only under existing build and cache directories and are
safe to repeat. The implementation changes no generated files and performs no
remote action. If a focused test fails, rerun its GoogleTest filter after the
source fix, then rerun the complete binary so shared frontend behavior is not
missed. Preserve unrelated worktree changes when inspecting or revising the
diff.

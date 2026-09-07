# Add compile-time fixed-width OpenQASM angles

Status: historical implementation record.

## Goal and scope

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

## Constraints

- The typed OpenQASM frontend and QC emitter already represent gate parameters
  as binary64 radians. An angle constant can therefore become an ordinary
  `ScalarType::Angle` constant at the end of semantic analysis without changing
  the public frontend types or the QC dialect. Evidence: the focused export test
  emits `rx(1.5707963267948966)` and reparses it in strict mode.

- The source width is needed only while constant expressions are evaluated.
  Keeping a private pair of residue and width removes the need for source-format
  attributes, integer angle operations in QC, and pattern reconstruction in the
  exporter.

- Binary64 cannot distinguish every adjacent fixed-angle residue at width 53
  near two pi. Width 52 has a step larger than the largest binary64 spacing in
  the represented interval. The supported range is therefore 1 through 52, and
  an omitted width resolves to 52.

- OpenQASM defines `pi`, `tau`, and `euler` as 64-bit floats, and inverse
  trigonometric functions return floats. The previous frontend treated some of
  these values as angles. The fixed-angle work exposes and corrects that
  mismatch.

- The live specification first describes binary angle operations on equal-width
  operands, then states that mixed-width angles use unsigned integer promotion.
  The implementation widens both fixed values before it applies the operation.

## Decisions

- Support fixed-width angles only as compile-time values. Rationale: the
  requested use case does not require runtime angle storage, and the existing
  backend interfaces use binary64 gate parameters.

- Keep fixed-angle state private to
  `mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp`. Rationale: no other format
  or dialect needs the OpenQASM storage width when the selected output contract
  is canonical semantic output rather than source round-trip.

- Accept widths 1 through 52 and use 52 for unsized angles. Rationale: every
  supported residue remains distinguishable after conversion to the existing
  binary64 radians boundary.

- Use round-to-nearest, ties-to-even for float conversion and angle narrowing.
  Widening appends zero low bits. Rationale: this is one of the OpenQASM-defined
  narrowing policies and is the required float-to-angle rule.

- Treat initialized declarations with or without `const` as write-once
  compile-time bindings. Rationale: this accepts common source while avoiding
  runtime storage and assignment. Reassignment and missing or dynamic
  initializers receive diagnostics.

- Leave bare gate parameters as binary64 and defer global gate-angle
  quantization. Rationale: other quantum formats use continuous floating-point
  parameters, and unconditional modulo reduction is unsafe for phase-sensitive
  or controlled operations.

- Let #2169 close #1128 and track the remaining runtime storage and target-aware
  quantization work in #2174. Rationale: #2169 satisfies the issue's stated
  design and parser or lowering acceptance criterion. The remaining work needs a
  format-independent compiler contract and is not part of compile-time OpenQASM
  input support.

## Outcome and validation

Compile-time fixed angles are implemented in semantic analysis and QC emission
without new dialect operations, format-specific attributes, exporter
reconstruction, or Python API changes. Release and non-unity affected tests, the
configured CTest suite with one expected skip, Clang-Tidy, lint, and strict
documentation passed.

Issue `#2174` owns runtime fixed angles and target-aware quantization.

## Code and ownership

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

## Acceptance

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

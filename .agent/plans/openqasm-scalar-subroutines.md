# Classical subroutines and borrowed arrays

Status: complete.

## Goal and scope

Implement #2593 with scalar subroutines and borrowed array parameters. The
[OpenQASM guide](../../docs/mlir/OpenQASM.md) defines supported syntax,
initialization, aliasing, and backend limits. Array-valued outputs (#2592)
remain separate.

## Decisions

- Use `func.func`/`func.call` and ranked memrefs. Bit vectors remain by-value;
  borrowed arrays retain caller storage without argument copies.
- Infer readonly use and export canonical scalar types instead of storing source
  metadata. Preserve signed arithmetic and bit patterns.
- Inline borrowed helpers before jeff's value-array lowering; retain scalar
  helpers. CBit permits mapped inlining for the QIR path.
- Run SCCP after memory promotion but before canonical cleanup and dead-value
  removal: promoted helper-call operands must fold before liveness handles
  quantum yields. Canonical cleanup also normalizes SCCP's constants.
- Preserve constant signedness when constructing integer attributes, including
  narrow negative elements in bulk initializers passed to borrowed helpers.
- Static empty rank-reduced views use shape-only casts because LLVM 23 subviews
  reject zero extents. Remove this workaround when upstream accepts them.

## Validation

The owning binaries under `build/release/mlir/unittests/` pass: compiler (320
passed, two expected platform skips), OpenQASM frontend (214), OpenQASM
translation (239), jeff round-trip (167), and QCO/DD utilities (257). Coverage
includes retained calls, one-time index evaluation, signed bulk initializers
passed to borrowed helpers, single-cleanup promotion of constant call operands,
QIR JIT, and serialized jeff. The compiler fixture independently constructs the
specialized unitary reference, retaining its call, constant angle, and wire.

The rebuilt package passes all 483 cases in `test/python/test_mlir.py` and
`test/python/test_mlir_integer_interchange.py`, including the four explicit
seed-override cases. Their exact-IR reference matches the custom pipeline's
cleanup boundaries instead of assuming incidental constant order is unchanged.

Repository lint and `git diff --check` pass. Whole-file C++ lint remains
unverified locally because clang-tidy 23 is unavailable. Local arrays inside
subroutines, array-valued returns, and recursive calls remain unsupported.

# Support wide CBit comparisons in Qiskit

Status: complete.

## Goal and scope

Allow Qiskit interchange for unsigned comparisons between one complete named
classical register and one same-width literal when the width exceeds 64 bits.
Both operand orders and all equality and ordering predicates are supported.
Values must round-trip without truncation, including values large enough to
exceed Python's decimal string-conversion limit.

The private interchange model lives in
`bindings/mlir/qiskit/QiskitTranslation.h`. Python conversion is implemented in
`bindings/mlir/qiskit/Qiskit2_5.cpp`, while
`bindings/mlir/qiskit/QiskitExport.cpp` and
`bindings/mlir/qiskit/QiskitImport.cpp` translate between that model and MLIR.
Focused coverage is in `test/python/test_mlir_qiskit_translation.py`. The
supported boundary is documented in `docs/mlir/python_compiler_collection.md`.

Computed wide integer expressions, packed loose bits, and signed comparisons
wider than 64 bits remain unsupported. Those cases continue to fail with an
explicit diagnostic instead of silently widening the generic expression path.

## Decisions

- Store normalized unsigned literals as `llvm::APInt`, with the declared Qiskit
  width kept separately. Active bits are validated before MLIR emission. This
  preserves exact values and avoids allocating storage based only on an
  untrusted declared width.
- Convert Python integers through unsigned hexadecimal text in both directions.
  Python exempts power-of-two bases from its configurable decimal digit limit,
  so this supports very wide values without changing interpreter-wide settings.
- Recognize the wide form before generic expression handling. Export accepts a
  direct `cbit.read` and `arith.cmpi` with an integer constant; import emits the
  same standard MLIR operations. The complete-register check prevents this
  exception from expanding support to aliases or arbitrary computed values.
- Keep the existing 64-bit boundary everywhere else. Signed ordering currently
  requires a computed sign-bit transform in Qiskit, so it is intentionally not
  part of the direct-only wide path.

## Validation

Validation from the repository root completed as follows:

    cmake --build build/python/Release --target mqt-core-mlir-bindings -j 6
    uvx nox -s stubs
    uvx nox -s lint
    git diff --check

The binding build passed. The focused selection passed all seven cases, and the
complete `test/python/test_mlir_qiskit_translation.py` file passed all 261
tests. Stub generation completed without a generated-file diff, repository lint
passed, and `git diff --check` passed. The focused cases cover 65-, 151-, and
301-bit values, reversed ordering, Python's decimal digit limit, and the
deliberate computed-wide and signed-wide rejections.

`uvx nox -s cpp-lint -- origin/main` could not start because this host does not
provide the required clang-tidy 22; it reported no source finding.

## Outcome

Direct unsigned complete-register comparisons now round-trip through Qiskit at
arbitrary representable widths using standard MLIR operations. The generic
64-bit boundary and explicit rejections remain intact, no dependency or public
Python API was added, and the user-facing support table records the exception.

# Support wide CBit comparisons in Qiskit

Status: complete.

## Goal and scope

Support unsigned comparisons between one complete classical register and one
same-width literal beyond 64 bits, including both operand orders and all six
comparison predicates. Qiskit expression conditions and tuple equalities use the
same normalized expression path. Out-of-range tuple equalities become false.
Computed, packed, and signed wide expressions remain unsupported.

## Decisions

- Store unsigned literals in `llvm::APInt`; transfer Python integers through
  hexadecimal text to preserve values without Python's decimal digit limit.
- Import validation owns the supported-expression boundary. Only the immediate
  register and literal leaves of a direct comparison may exceed 64 bits. Generic
  emission handles constants, reads, and comparison predicates.
- Emission checks mapped register storage. A wide read requires one complete
  storage object in bit order; packing remains limited to 64 bits.
- Preserve operand order during import. MLIR comparison folding normalizes
  constant-left predicates where needed; jeff already enables this folding.
- Export retains the direct `cbit.read` and `arith.cmpi` recognition path and
  the shared initialization and snapshot checks.

## Validation

The LLVM/MLIR 23.1 binding build passed. All 281 Qiskit translation tests
passed, including 27 focused wide-comparison cases. The 12 predicate/order cases
also verify conversion through QCO to jeff. Stub generation produced no tracked
changes. C++ lint passed with zero findings. Repository lint passed.

Commands from the repository root:

```console
uv run --no-sync pytest -n 0 test/python/test_mlir_qiskit_translation.py
uvx nox -s stubs
uvx nox -s cpp-lint
uvx nox -s lint
```

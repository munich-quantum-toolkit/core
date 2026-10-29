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

## Validation and maintenance decision

Maintenance on 28 October 2026 checked this PR against main
`66968a5f44d2b0eddb108b6c7c5c49c463c57205`. Retain the implementation and test
scenarios: frontend type/alias checks, export round trips, DD diagnostics, QIR
execution, serialized jeff, and the single-cleanup entry protect distinct
contracts. The 17-element initializer crosses the bulk-emission threshold; three
524,288-bit results cross the aggregate snapshot limit.

The current native suites passed: OpenQASM frontend 214, translation 239,
compiler 320 with two `/dev/full` platform skips, jeff round-trip 167, QCO/DD
utilities 257, and CBit IR 17. Logs and review evidence are in the Benchpress
workspace under `.benchmarks/open-pr-maintenance-20261028/pr2816/`.

Prior validation also passed 483 Python cases across `test_mlir.py` and
`test_mlir_integer_interchange.py`; these were not rerun for this plan-only
maintenance. The seed-override reference retains the custom pipeline's cleanup
boundaries. Whole-file C++ lint was not rerun; there are no C++ edits in this
maintenance. Repository lint and the independent final review passed.

No runtime optimization is retained without useful evidence. Constant array
indices could avoid eager extent construction, but no representative array-heavy
performance workload was available. Local arrays inside subroutines,
array-valued returns, and recursive calls remain unsupported.

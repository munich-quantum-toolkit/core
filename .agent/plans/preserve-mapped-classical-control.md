# Preserve classical control during target mapping

Status: complete. The implementation and regressions are rebased onto current
`main` and validated through the Core and Benchpress paths.

## Goal and scope

Target mapping must preserve measurement-fed classical control while inserting
routing SWAPs and repairing block order. Before this change, routing could
create a backward quantum dependency through later structured control, and the
topological repair could move a classical-register read before the measurement
result was stored. Cleanup could then remove the conditional quantum work.
Routing could also split a measurement from its direct classical destination,
which native Qiskit export rejects.

The implementation is confined to
`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` and
`mlir/lib/Dialect/QCO/Utils/Sorting.cpp`. Five regressions in
`mlir/unittests/Dialect/QCO/Transforms/Mapping/test_mapping.cpp` cover the
routing frontier, mutable-register ordering, static indexed loads, repeated
measurement destinations, and the native measurement-destination contract. No
public API or dependency changes are required.

## Decisions

- Track direct, statically indexed CBit loads and stores per register element.
  Effects on distinct constant indices do not conflict. Whole-register,
  dynamic-index, and indirect effects remain conservative barriers that alias
  every element. This per-index refinement removes false cross-index
  dependencies without creating a dependency cycle.
- Select the earliest original block-position operation from the topological
  sorter's ready set. This retains source order and measurement/store adjacency
  whenever dependencies permit it.
- Retain the mapper's existing decrement, insert, and increment protocol. For
  the first use of each SWAP endpoint, rewind only when its wire crossed a
  `qco.if`, `qco.index_switch`, `scf.for`, or `scf.while` beyond the earliest
  unresolved frontier. Rewinding every endpoint breaks valid pure-quantum
  routing state.
- When a SWAP consumes a measured qubit, delay it until after the first later
  direct memory-write consumer of the measured bit. Stop at structured control
  and never move the insertion point earlier than the normal definition anchor.
- Represent register conditions with the current fixed-width model: `cbit.read`
  followed by standard `arith` operations. The sorter orders the store before
  the read; pure comparisons do not own the memory dependency.

## Validation

- The five focused mapping regressions passed 25 consecutive repetitions each:
  125 of 125 test executions.
- The complete mapping unit-test binary passed all 99 tests.
- The nine target-compilation compiler regressions passed.
- A wheel built from this source passed the full affected Benchpress matrix when
  combined with the independent target-pipeline inlining from Core PR #2344: all
  31 control cases exported to native Qiskit, preserving all 3,461 conditionals
  and satisfying target validation.
- Without that independent inlining, this PR passes 23 of 31 control cases and
  preserves 3,422 conditionals. The remaining eight fail before this mapping
  logic because reusable quantum functions reach target compilation.
- The BV100 regression exported natively with all 99 measurement destinations
  intact.
- Repository lint, Markdown lint, formatting, and whitespace checks passed. The
  exact C++ lint session is delegated to PR CI because this machine does not
  have the required clang-tidy 23 executable.

## Outcome

Routing no longer creates a backward dependency through later structured
control, sorting preserves only potentially aliasing mutable-register effects,
and routing keeps measurements with their direct classical destinations. Core
pull request 2344 remains independently necessary for the eight current-`main`
cases whose reusable quantum functions must be inlined before target
compilation; it is not part of this repair.

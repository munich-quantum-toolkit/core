# Preserve classical control during target mapping

Status: complete on main `3f801880a`. The routing cycle is fixed; separate
Benchpress integration and exporter gaps remain.

## Goal and scope

Prevent routing SWAPs from creating a cyclic dependency through later classical
control. An independent wire can advance through a conditional whose measurement
depends on an unresolved two-qubit gate. Using that conditional's output as a
SWAP input can make the unresolved gate depend on its own result.

The production change belongs in
`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`. The mapping regression
checks valid output, target connectivity, and preservation of conditional work.
Two sorter regressions cover repeated stores to one register element and a
whole-register write followed by an indexed load during dominance repair.

## Decisions

- Keep main's recursive memory-effect ordering and FIFO topological sorter from
  #2436 unchanged. The sorter preserves mutable-register dependencies but cannot
  repair a cycle introduced by routing.
- Do not require measurement/store adjacency in the mapper. Native Qiskit export
  supports intervening quantum operations since #2439.
- Retain the mapper's decrement, insert, and increment protocol. Before the
  first use of each SWAP endpoint, rewind its wire only if it crossed a
  `qco.if`, `qco.index_switch`, `scf.for`, or `scf.while` beyond the earliest
  unresolved frontier. Rewinding every endpoint changes valid pure-quantum
  routing state.
- Leave the existing composite-boundary helper unchanged. No public API or
  dependency changes are required.

## Validation

- The isolated routing regression aborts at the sorter's cyclic-dependency
  assertion on main. With this fix, it passes 25 consecutive repetitions.
- All 96 mapping, 185 QCO utility, and 167 compiler tests pass.
- A fresh Python wheel passes all 306 Qiskit translation tests and the two
  one-qubit synthesis regressions. All 68 Benchpress integration tests pass.
- Repository lint and whole-changed-file C++ lint pass without findings.

## Remaining Benchpress gaps

The constrained feed-forward matrix is not fully enabled by this change. Six of
31 guarded profiles pass unchanged; 25 stop at the integration's strict textual
event-order check. The check rejects reordered independent events, so these
failures alone do not establish a semantic regression. The small deterministic
feed-forward counterexample now preserves its measured result.

BV100 still fails native Qiskit export when mapping groups measurements before
their stores. The exporter supports intervening quantum operations, but not
another measurement. This restriction belongs in the exporter, not SWAP
placement. Keep the integration guards and export fallback until their separate
contracts are resolved; no full Benchpress corpus result is claimed here.

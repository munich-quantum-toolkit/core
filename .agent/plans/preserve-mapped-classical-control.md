# Preserve classical control during target mapping

Status: validation in progress on main `3f801880a`.

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

## Validation and remaining work

- Confirm that the isolated routing regression still fails on main, then passes
  with the routing fix.
- Run the mapping, QCO utility, and compiler tests, plus repeated focused cases.
- Recheck the known Benchpress gaps with the refreshed branch. Do not restart
  the full corpus suite.
- Run repository lint and the whole-changed-file C++ lint session before push.

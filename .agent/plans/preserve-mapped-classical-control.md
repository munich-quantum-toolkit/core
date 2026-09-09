# Preserve classical control during target mapping

Status: complete. Routing, terminal-measurement, and performance regressions
pass.

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
- Defer composites while an earlier wire operation still needs routing. Terminal
  sinks and output-only measurements must not block independent quantum work.
- Use one measurement classification for advancement and composite deferral.
  Follow SSA result uses and forward whole-register effects until reaching
  quantum work. Register accesses before the measurement run do not require
  measured-qubit reuse. Output-only loads and overwrites remain terminal;
  register accesses inside quantum composites still impose ordering.
- Keep consecutive measurements in scope so an earlier measurement does not hide
  a later result used for quantum control. Reuse LLVM slice analysis and a
  bounded worklist. Traverse it by index because discovering more effects can
  append entries and invalidate iterators. No public API or dependency changes
  are required.
- Cache consecutive-measurement suffix classifications within each `advance`
  invocation, sharing slice and register work across the run. Discard the cache
  before hot routing mutates the graph.
- Backward traversal of an idle nested block argument can reach its sentinel;
  that exhausted wire does not defer another composite.
- Select the wire value crossing the composite's block-order boundary when
  extending it. Another wire may already have advanced through a gate that
  consumes a classical result of that composite; its current value would
  introduce an SSA cycle.
- Prefer terminal measurements even when the target accepts adaptive programs.
  Target permission does not replace program dependencies or placement
  readiness.

## Validation

Release unit tests pass: 100 mapping tests, 192 QCO utility tests, and 181
compiler tests. The added cases cover idle nested wires, a conditional angle
consumed on another wire, terminal measurements after earlier register control,
and consecutive measurements with first/last result control and shared stores.

The consecutive-measurement probe at 4,000 measurements improved from 327.85 ms
to 2.398 ms before the boundary and earlier-access fixes. This measures mapping
alone on a synthetic workload, not a whole-program or corpus speedup.

Repository lint and full-file C++ lint against fixed base `ce608b082` pass.
These are local checks, not hosted CI or a full Benchpress corpus run.

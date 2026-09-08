# Preserve classical control during target mapping

Status: complete. The routing regressions and required local checks pass.

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
  Follow all SSA result uses and the sorter's whole-register effect order until
  reaching quantum work. Output-only loads and overwrites remain terminal;
  register accesses inside quantum composites still impose ordering.
- Keep consecutive measurements in scope so an earlier measurement does not hide
  a later result used for quantum control. Reuse LLVM slice analysis and a
  bounded worklist; no public API or dependency changes are required.

## Validation

Build the release mapping, QCO utility, and compiler unit-test targets. The
mapping binary passes all 100 tests, the QCO utility binary passes all 187, and
the compiler binary passes all 171. The focused mapping tests cover terminal
measurements and sinks before independent control, consecutive measurements,
multiple result users, output-only register reads and overwrites, and register
writes inside a quantum conditional.

Run `uvx nox -s lint` and `uvx nox -s cpp-lint -- <main-base>` for the change.
The repository hooks pass. Full-file C++ lint passes for all three C++ files in
the PR, using main base `b75b02fa9`. These are local checks, not hosted CI or a
full Benchpress corpus run.

# Simplify routing without changing its heuristics

Status: complete.

## Outcome and decisions

Mapping skips repair for equal layouts and updates RoutingBundle directly.
WireInfos derives membership from its inverse vectors. Graph traversal borrows
adjacency lists; unused distance-matrix and graph APIs are removed. The mapping
boundary diagnoses qubit-carrying calls, multi-block entry functions, and
invalid options. Classical calls remain supported. Documentation describes the
dense temporary workspace and its later cleanup.

Branch convergence, voting, A* ordering, and traversal semantics are unchanged.
Payload control-flow legalization remains owned by #2162. Supported structured
regions already have single-block verifiers; the entry-function restriction is
checked by mapping. No payload capability checks were duplicated.

The optional mapping benchmark and its data, figure, and reproduction steps are
in `.agent/benchmarks/routing/README.md`. Identical mapped-IR hashes and SWAP
counts accompany a 1.56–1.82 times speedup for unchanged branch layouts. The
routing workload is essentially unchanged. Graph measurements are synthetic.

## Validation

Release builds passed 104 mapping tests, 192 QCO utility tests, and 182 compiler
tests. `uvx nox -s lint` passed. Full changed-file C++ lint passed with
`uvx nox -s cpp-lint -- 91a9e0ba514af938680cdd394d6d63195872dc9a`.

A disposable combination with #2162 at `1c5d4cc66` applied cleanly using
three-way patch application. All 100 existing mapping tests passed there. The
combined compiler suite passed 193 of 195 tests. Its two failing tests,
`PayloadControlRejectsLinearStateInGenericSCFControl` and
`PayloadControlRejectsUnstructuredCFG`, fail during parsing because they
allocate qubits outside the entry block. Both failures were reproduced with the
routing changes removed, before the mapping pass runs. Updating those #2162
inputs is outside this PR's scope.

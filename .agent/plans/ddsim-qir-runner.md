# DDSIM QIR execution

Status: implemented and locally validated.

## Goal and scope

Implement the eight findings in `.agent/audits/ddsim-qir-runner.md` and submit
one PR. Preserve Adaptive execution and the public textual QIR runtime ABI.
Terminal sampling must preserve recorded-result order and fall back for
unsupported inputs. State extraction must preserve logical wires and phase.

## Decisions

- Build on current main. Shared gate-validation allocation work in #2455 stays
  with that PR; QIR address-storage improvements are independent.
- Declared static qubit capacity determines extracted state width. Missing
  metadata keeps the existing direct-ABI and legacy-session inference behavior.
- Recycle physical dynamic wires, not opaque handles, to reject stale handles.
- Enable terminal sampling only after proving a straight-line static program
  with known QIS calls, terminal measurements and static output mapping. Keep
  full JIT execution for other sampling programs.
- Keep canonical matrices and DD construction shared with QCO. A general gate
  cache and an LLVM optimization pipeline remain benchmark-dependent candidates,
  outside the eight confirmed findings.

## Validation

Release suites passed: 38 JIT/analysis, 78 runtime, 65 DDSIM, 169 DD and 122 QIR
IR tests. The audit records CPU-time and allocation comparisons with their
limits. Full Clang CTest: 4013 tests, no failures, one existing skip. Both
`uvx nox -s lint` and `uvx nox -s cpp-lint` pass; C++ lint inspected all changed
implementation and test files. The final JIT suite also passes after the
naming-only cleanup. Hosted CI is not part of this local validation.

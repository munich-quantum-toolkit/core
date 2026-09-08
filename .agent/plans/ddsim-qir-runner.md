# DDSIM QIR execution

Status: implemented and locally validated.

## Goal and scope

Implement the findings in `.agent/audits/ddsim-qir-runner.md` and
`.agent/audits/ddsim-qir-runner-refresh.md` in one PR. Preserve Adaptive
execution, ordered shot results and the public textual QIR runtime ABI. State
extraction must preserve logical wires, declared width and phase.

## Decisions

- Keep canonical gate matrices and DD construction shared with QCO. General
  caching and LLVM optimization remain dependent on workload evidence.
- Declared static capacities bound resource IDs and determine extracted width.
  Missing metadata retains direct-ABI and legacy-session inference behavior.
- Start packages at zero capacity. Size declared resources exactly; grow unknown
  resources geometrically from the DD default capacity on first quantum use.
  Warm resets retain storage, and moved-out packages are recreated lazily.
- Recycle physical dynamic wires, not opaque handles, to reject stale handles.
- Prove terminal sampling eligibility from the actual Base or Adaptive body:
  unconditional acyclic execution, constant gates and static resources, terminal
  measurements, and known output mapping. Other programs execute for every shot.
- Keep batch DD sampling in Runtime. Move full ascending physical sample strings
  directly into the batch; preserve generic mappings, the final measurement
  record and RNG consumption.
- Initialize only appended unique-table levels in the shared DD implementation.
  Preserve existing lookups, statistics, roots and GC behavior.
- Reject defined and indirect helpers during state-extraction analysis before
  changing IR. Do not infer effects through function bodies.

## Validation

Focused release tests passed: 46 JIT/analysis, 79 runtime and 180 DD tests. Full
Clang CTest passed 3,190 tests with no failures and one existing skip. Both
required lint sessions and the complete strict documentation build passed. C++
lint inspected the full changed-file set, including shared DD code. The
refreshed audit records combined CPU-time and allocation measurements, including
noisy growth timings and unchanged seeded output mappings. Hosted CI is separate
from these local results.

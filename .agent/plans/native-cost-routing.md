# Native cost in routing selection

Status: implemented, reviewed, and locally validated.

## Scope and ownership

Rank completed routing candidates by final native two-qubit gate count, then
maximum block depth, preserving candidate order on exact ties. Keep bounded A*
and forward/backward refinement unchanged. Reuse target synthesis through a
shared QCO-owned pipeline, including cleanup and target conformance.

Topology-only mapping retains its supported input contract. Speculative native
failures must not emit user diagnostics or hide errors from final compilation.
The sink canonicalizer must retain a producer while SCF still uses it during
loop-result forwarding.

## Decisions

Use the existing native synthesis on temporary routed modules. A separate cost
model would repeat fusion, numerical, symbolic, and target-direction rules.
Release search storage before synthesis. Duplicate-layout bookkeeping did not
reduce measured CPU cost and is excluded.

Gate count is static; maximum block depth is a secondary proxy for structured
control flow. Runtime branch probabilities and unknown loop bounds are outside
this change. No routing heuristic or new compiler option is introduced.

## Review and validation

Ponytail Review removed duplicate score fields, an unused SWAP result, and a
repeated semantic-test builder. No separate cost model or layout cache was
added.

The release build passed 979 tests across mapping, target synthesis, QCO IR, and
compiler suites. Native-basis tests cover 256 semantic configurations. Full
changed-file C++ lint, repository lint, MLIR reference generation, and the
complete executable documentation build passed.

The production integration reproduced all 40 prior QASMBench outputs byte for
byte. Native counts improved in 18 cases against the recorded baseline, with no
regressions. The 1,000-qubit random grid case completed in 64.36 seconds at
10.75 GiB peak RSS, with the same 71,453 native gates and depth 5,379. Small
controls used 25.8 seconds aggregate process CPU versus 3.94 seconds in the
recorded baseline; different host load prevents a speed comparison.

Harnesses, raw results, and the detailed review remain outside the repository in
`/tmp/mqt-native-routing-pr-20260915`. The temporary CMake hook was removed.
Submit the signed branch for human review; hosted CI is separate from these
local checks.

# Native-cost routing with physical wire slots

Status: complete.

## Contracts and decisions

Native synthesis owns cost decisions. Routing ranks estimated native two-qubit
count, then maximum block dependency depth, then candidate order. Unknown costs
fall back to SWAP count after available estimates. Final synthesis owns emitted
gates and unsupported-operation diagnostics.

Index routing wires by physical site and retain Layout as the sole mapping to
logical qubits. Apply the same traversal rules with and without native costs.
Keep optional cost tracking beside each block state. Virtual routing and IR
emission differ only at SWAP and region boundaries. Preserve frontier traversal,
A*, bounded Arena storage, branch convergence, loop restoration, seeds, trial
generation, options, and cache capacities. Add no public API or strategy
framework.

The independent SinkOp fix preserves producers still used while SCF forwards
idle loop results. It was published as PR #2610 and merged. PR #2607 now targets
main and excludes that fix from its diff.

### Completed work

- [x] Publish the independently validated idle-loop fix (#2610).
- [x] Replace WireInfos with physical-site slots in one reviewable commit.
- [x] Simplify routing and region flow in a separate commit.
- [x] Validate cold/hot agreement, semantics, fallback, boundaries, and
      determinism.
- [x] Compare quality and resources with frozen revision 3bd27caba.
- [x] Update pass documentation and #2607 with fresh plots and measured
      limitations.

### Validation and acceptance

Run mapping, synthesis, compiler, and QCO IR suites; compile changed tests with
GCC and Clang; run whole-file C++ lint and repository lint. Preserve coherent
state, linearity, target conformance, symbolic boundaries, topology-only
targets, idle sites, tensor tails, all region forms, negative credits, and
bounded fallback.

Reuse the archived broad cohorts with seeds 7/99 and separate largest-input
results. No lost successful compilations or semantic regressions are allowed.
Aggregate native two-qubit count may increase at most 1% against 3bd27caba,
separately for flat and structured inputs. Original seven-case and expanded
timing cohorts may each increase at most 3% in summed median wall time. Report
SWAPs, depth, CPU, RSS, family/architecture/width breakdowns, individual losses,
and Grover scaling. Keep benchmark artifacts outside the repository. The prior
comparison with main remains historical evidence until replaced by fresh
results.

### Results

The independent fix merged as #2610. The signed routing commits separate
physical-site state (`ef3a2e1b3`) from traversal and region cleanup
(`7516fc490`). The full mapping implementation loses 405 lines and 21
conditional statements. Native-cost availability no longer selects wire or
region semantics.

Against `3bd27caba`, flat native count increases 0.19% and structured native
count falls 17.96%. No successful compilation is lost. The separate largest
cohort increases 1.23% in native count and 7.62% in dependency depth. Original
seven-case wall time increases 2.92%; expanded 92-case wall time increases
0.15%. All four acceptance gates pass, with little margin on the first timing
gate. CPU increases 22.08% and 9.70% in those timing cohorts; RSS changes +6.62%
and -0.60%. These results do not imply uniform performance improvement.

All 997 affected tests, GCC and Clang compilation, whole-file C++ lint, and
repository lint pass. The 52 ordinary semantic cases per revision and ten wide
RUS circuit/seed checks per revision pass. Grover repeats and serial/parallel
outputs match within each revision. Source, binary, input, and harness hashes
are recorded; original untracked files are unchanged.

External artifacts, including 14 plots, all individual losses, family,
architecture, width, timing, CPU, RSS, and resource-grid breakdowns, live in
`/home/nvidia/.codex/experiments/pr2607-routing-simplification-20260923/`.

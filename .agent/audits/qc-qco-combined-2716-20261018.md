# Combined QC/QCO audit and PR #2716

Date: 2026-10-18. Combined implementation for PR #2716.

## Decision

Combine the [QC/QCO audit](qc-qco-upstream-20261017.md) with
[PR #2716](https://github.com/munich-quantum-toolkit/core/pull/2716). The audit
repairs builder ownership, region and tensor contracts, and consumer traversal;
the PR simplifies forward conversion. The changes preserve the same supported
input contract and share the consolidated conversion fixtures.

The combined implementation is based on upstream `main` at
`470094596af808df39cd25563377534bd44c2129`. Upstream advanced during the
measurements; the frozen four-way comparison remains based on
`056bef24722dd0a1f0813f3f7b4197c10f8eb950`. The original PR head was
`9d929c5947ddb34bed3a9ef729635509d91674bf`, originally based on
`575acd1b9ca52e4608ccf4833e7d60640913cef4`. The sole application conflict was
the call-pattern signature beside an obsolete callee lookup. Keep main's removal
of that lookup and use the PR's ordinary pattern signature. The original audit
worktree remains unchanged. The subsequent upstream commits apply cleanly; their
only overlapping builder edit changes loop canonicalization walk order and is
retained. The latest main change only reorders lint hooks. The original PR
commits retain their authorship after rebasing; the audit and further
simplifications follow in a separate commit.

## Further simplifications

- Release preflight register provenance before normalization and rebuild it in
  the same state afterward. Keep both validation phases: folding can reveal
  aliases that were not apparent in the original indices.
- Move structured capture vectors out of their source SetVectors; erase the old
  entries instead of retaining two copies and their membership tables.
- Resolve structured outputs through borrowed views into one reserved vector.
  Release completed region mappings after copying terminator operands. Retain
  shared output ordering while sibling regions still need it.
- Keep unitary attributes throughout lowering. The direct rewrite driver does
  not verify partially converted IR, so temporary removal/restoration is unused.
- Refresh load provenance directly in the source-retention branch and cache
  dialect handles. Remove the load-only pattern and the final QC scan: every
  source QC operation must be rewritten or checked and erased. Boundary
  verification and checks for remaining reference uses stay in place.

These follow-up changes remove another 44 net converter lines from the initial
combination. Against upstream, the complete prototype removes 581 net
source/test lines: 146 production and 435 test lines. It preserves the shared
132 forward and 133 reverse conversion cases and all five PR regressions; this
follow-up adds no tests. Independent design and correctness reviews, including
the final ponytail review, found no remaining concrete issue or necessary
regression gap. The final review covered converter lifetimes and scaling,
builder ownership and callers, iterator correspondence, consumer traversal, and
public validation boundaries.

The source-operation snapshot remains deliberate. Structural rewrites move
children, and deferred erasure preserves reference identity. Replacing it with a
custom mutation-aware walker would add lifetime rules without measured benefit.
The entry-block allocation limit, single-block structured input, tensor slot
contract, and positional quantum correspondence remain explicit constraints.

## Measurement method

Four variants isolate the contributions: audit alone, PR alone, initial
combination, and simplified combination. All use Clang 23.1.4, portable
LLVM/MLIR 23.1.0, release `-O3`, ThinLTO, and assertions on DGX Spark. Five
fresh-process rounds rotate variant order on CPU 0; factor247 retains three
complete rounds. Further factor247 repetitions stopped after external CPU
contention became pronounced; its incomplete fourth round is saved separately
and excluded. The host/core is not reserved. MLIR multithreading is disabled.
Medians are descriptive local measurements, not a statistical claim for small
changes. Other compiler, lint, and benchmark jobs were observed on the host,
including CPU 0; wall-clock results are diagnostic.

The native probe times `QCProgram::intoQCO`; full-pipeline cases also time
`compileForTarget` and `intoQC`. Import, initial copy, target construction,
extra output checks, inspection, and serialization are outside timing. Peak RSS
includes import and the retained source/copy, and is sampled before output
serialization. Output IR hashes and gate counts must agree across every variant
and repetition. Hashing in the parent uses streaming to avoid inflating child
RSS; an interrupted initial run with an inflated RSS floor is saved separately
and excluded.

Seven frozen QASMBench circuits come from Benchpress
`36d550d38fe03fdbb2f00f045942b4315d9e4864`; four generated MLIR cases isolate
flat gates, register loads, sibling loops, and wide helper signatures. Full
compilation uses seed 42, ten mapping trials, one iteration, and the
`sx/x/rz/cz/measure/reset` basis without calibration metadata. Results do not
replace the PR's historical broader topology campaign.

## Results

All 252 retained samples pass output verification and agree on output IR hashes
and gate counts within each case: 212 forward-conversion and 40 full-pipeline
samples. No additional test case was needed.

Forward-conversion medians follow. RSS columns compare the initial and
simplified combination. The Audit and PR columns separate their contributions.

| Case                | Audit (ms) |   PR (ms) | Combined (ms) | Simplified (ms) | Combined RSS (MiB) | Simplified RSS (MiB) |
| ------------------- | ---------: | --------: | ------------: | --------------: | -----------------: | -------------------: |
| qft_n18             |     13.113 |     8.015 |         8.068 |           7.807 |              18.57 |                18.57 |
| qram_n20            |      1.487 |     0.893 |         0.922 |           0.892 |              16.35 |                16.54 |
| ising_n26           |      4.964 |     3.114 |         3.111 |           3.055 |              17.38 |                17.38 |
| factor247_n15       |  17702.400 | 11910.900 |     10539.000 |       11205.000 |            1722.50 |              1702.43 |
| cat_n260            |     14.662 |    10.188 |        10.215 |          10.085 |              18.32 |                18.32 |
| wstate_n380         |     43.493 |    29.210 |        29.218 |          28.839 |              22.41 |                22.44 |
| qft_n160            |   1404.180 |   856.447 |       873.410 |         823.506 |             196.03 |               190.62 |
| flat-1m             |   1873.690 |   835.906 |      1188.950 |         818.760 |             492.00 |               492.07 |
| register-loads-100k |   1245.130 |   805.668 |       770.021 |         751.645 |             172.32 |               166.30 |
| sibling-loops-2k-32 |    185.160 |   103.468 |       102.871 |          93.460 |              64.99 |                59.27 |
| wide-helper-8192    |      7.503 |     2.564 |         2.583 |           2.613 |              17.72 |                17.91 |

The further cleanup reduces peak RSS by 3.5% for register loads and 8.8% for
sibling loops. QFT-160 saves 5.4 MiB; factor247 saves 20.1 MiB. The flat
million-gate case is unchanged: imported/copied IR dominates its peak.

The broader combination has larger memory effects: audit-only versus final peak
RSS falls from 736.3 to 492.1 MiB for one million flat gates, and from 259.5 to
190.6 MiB for QFT-160. These are process peaks for this harness, not
steady-state converter allocations.

Full pipeline medians (QFT-18 linear; Ising-26 all-to-all):

| Case      | Audit (ms) | PR (ms) | Combined (ms) | Simplified (ms) | Combined RSS (MiB) | Simplified RSS (MiB) |
| --------- | ---------: | ------: | ------------: | --------------: | -----------------: | -------------------: |
| qft_n18   |    187.668 | 183.307 |       182.307 |         182.356 |              23.23 |                23.29 |
| ising_n26 |     22.152 |  20.375 |        20.250 |          20.218 |              20.41 |                20.41 |

The further cleanup gives no meaningful full-pipeline improvement in these two
cases. Wall-time differences under contention do not establish small speedups or
regressions; in particular, factor247's simplified median is slower than the
initial combination's. Do not multiply component ratios or claim a universal
compiler speedup. The PR's historical factor247/square regression was not
remeasured here.

## Validation and artifacts

- The final `470094596` full `release-clang-ipo` build and CTest pass: 4,030
  passed, one expected `ScQDMIJobSpecificationTest.QueryJobId` skip; 2,925 MLIR
  tests included. This includes the new upstream OpenQASM and jeff cases.
- Both full-pipeline cases were rerun with the refreshed binary; output IR
  hashes and gate counts match the frozen comparison.
- Repository whole-file C++ lint passes for its 19 selected files. The final
  capture-vector transfer also passes a complete converter translation-unit
  recheck with no source diagnostics.
- MLIR reference generation, repository lint, and final diff checks pass.
- The production and test files are byte-identical to the reviewed and
  C++-linted `8ed992b64` combination. Rebasing onto `470094596` adds only a
  lint-hook ordering change; repository lint also passes with that change.
- Python bindings and sanitizer builds were not run for this combined
  implementation. Hosted checks on the original PR head do not validate the
  combination; final-head CI status is tracked on PR #2716.

The earlier builder change makes builders move-only; borrowed contexts must
outlive their builders and IR. No in-tree caller needs builder copies. The
prototype does not broaden conversion support or promise a general full-compiler
speedup.

External artifacts are in
`/home/nvidia/.codex/experiments/core-qc-qco-combined-20261018/measurements/`:
source patches, native probe/build/runner scripts, generated inputs,
environment, input/binary hashes, raw samples, and summaries. The combined
worktree is its sibling `core/`; `pr-only/` supplies the separate PR baseline.

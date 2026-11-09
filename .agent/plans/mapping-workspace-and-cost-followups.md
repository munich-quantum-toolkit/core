# Mapping workspace and graph-walk follow-ups

Status: blocked on routing quality; implemented for draft review. Semantic
checks pass, but the adopted routing-quality gate fails. Graph-walk experiments
did not justify a production change.

## Scope and decisions

The `place-and-route` pass deduplicates A* states by the positions of every
original program qubit, including idle spectators. Workspace labels are not part
of the key. Root hashing and incremental SWAP hashing use the same
program-position pairs as equality. Lowest accumulated cost still determines
which representative is retained.

The full inverse permutation remains in each node for neighbor lookup. Arena
ownership, the node budget, queue behavior, the root-only native-cost
adjustment, and shortest-path fallback are unchanged. Region reconciliation
still restores the complete layout. No workspace shrinking, trial-default
change, cache-sharing API, or new statistic is included.

This is a bounded heuristic, not an optimality guarantee. Removing equivalent
workspace states can change subsequent queue exploration and the selected route.
Acceptance requires semantic correctness and measured routing quality;
byte-identical output to the baseline is not required for this change.

The implementation stays in `Mapping.cpp`. `Passes.td` describes the state
identity. The existing `PreserveBasisStatesAfterNativeScoredCompilation` test
adds a 1 MiB search budget to its fallback budgets, checking flat and structured
programs in the X/Z bases across CX, CZ, ECR, and square-root-iSWAP targets.

## Graph-walk investigation

The baseline attributed material sampled CPU time to `walkProgramGraph`. Three
smaller alternatives to persistent traversal state were tested after
program-position deduplication:

- Remove the eager pending/frontier reservations based on target width.
- Give the pending map inline storage while retaining the frontier reservation.
- Specialize the traversal for the callback type.

The allocation variants showed no consistent gain; removing both reservations
regressed dense native QFT. Callback specialization also failed to improve the
longer QFT cases. None is retained. These experiments preserve release order and
are smaller than a persistent frontier.

Persistent state would have to renumber stored physical indices after SWAPs,
rebuild after composite placement and while-region handoff, and maintain a
separate speculative lookahead state. Fresh traversals discover their initial
frontier in site order; retained state would preserve historical arrival order.
That difference affects native-cost append order and requires an explicit new
ordering contract. The current evidence does not justify that additional state.

## Deferred follow-ups

- Workspace-aware reconciliation has real cold-trial overhead, but the item 0
  cohort emitted no workspace-only boundary SWAPs. Branch reconciliation was not
  exercised. Its modified partial-token algorithm still needs a termination
  argument and caller-side workspace relabeling that preserves physical wires.
- Trial-local numerical caches showed little estimated benefit at capacity 64 in
  the expensive workloads. Larger capacity and shared ownership are separate
  hypotheses. Keep numerical caches distinct from traversal accounting.
- A hot-route-only fallback statistic would hide the observed cold fallbacks. A
  trial-inclusive statistic needs a separate scope decision.

## Validation and evidence

The performance and routing-quality comparison is pinned to main revision
`bb60821de90ab110940d3f7a4befcbd230d4356a`. It reuses the item 0 cohort: QFT,
QAOA, Grover, weak-measurement Grover, and repeat-until-success on
144-/1024-site grids, with native and topology-only targets, plus two long
measurement chains in both modes. Case prefixes name grid side, not program
width. The final branch is based on `161a19b773`; its additional DD sampling
change does not modify mapping. Timings were not repeated on that base.

Timing uses five alternating pairs, fixed seed 7, eight trials, default
refinement/lookahead/memory settings, and the same optimized build. Profiles and
quality checks run separately; node counts below are historical. Quality
comparisons include additional seeds and inspect final emitted two-qubit
counts/depth and inserted SWAP statistics. Structured counts are static, without
execution-frequency weighting; depth is the maximum within a block. Shared-host
timings are not isolated-host performance claims.

The earlier item 0 study at `58a6541655` established the search opportunity:
wide native QAOA had 1,228,219 distinct accepted full-layout keys versus 142,182
program-position keys summed per search, five full arenas, and 13 cold
fallbacks. These are historical baseline counts, not measurements of the new
implementation or a predicted speedup.

### Correctness and routing quality

The release suite passed all 4,387 tests on the final base. Debug mapping (133
tests) and QCO utilities (262 tests) passed on the benchmark base, including
cold/hot native-score assertions. MLIR reference generation, repository lint,
and whole-file C++ lint passed before publication. External inspection verified
all 144 baseline/candidate modules, including QCO linearity.

The following totals cover 24 cases at seeds 7, 17, and 99. Counts are static;
SWAPs are the mapping pass's inserted-SWAP statistic. Each cell is baseline →
program-position keys.

| Cohort               | Two-qubit operations |   Two-qubit depth |  Inserted SWAPs |
| -------------------- | -------------------: | ----------------: | --------------: |
| Native, flat         |    224,470 → 225,229 | 125,074 → 125,665 | 63,848 → 64,169 |
| Native, structured   |      86,420 → 86,406 |   57,937 → 58,015 | 23,010 → 23,058 |
| Topology, flat       |    114,628 → 114,856 |   69,140 → 69,225 | 57,377 → 57,422 |
| Topology, structured |      46,351 → 46,332 |   37,226 → 37,189 | 19,405 → 19,387 |

Two-qubit gate count is the primary optimization objective; depth and inserted
SWAPs are secondary diagnostics. The original cohort still fails the count
criterion: flat native counts increase 0.34%. Dense native Grover at seed 17
grows from 24,224 to 24,704 operations (1.98%). Wide native weak-measurement
Grover at seed 99 grows from 961 to 1,065 (10.82%). Correctness and compilation
speed do not resolve these count regressions.

### Regression investigation

Diagnostic builds reproduce both regressions byte-for-byte. They trace each
trial's final score and the searches of the former winning trial, starting from
the same initial layout. Dense Grover first diverges at search 8: both selected
goals have priority 53.4260139 and cost 42, but different 14-SWAP paths. Weak
Grover first diverges at search 16: both goals have priority 27.3582497 and cost
12, but different four-SWAP paths. Removing duplicate states changes the heap's
unspecified equal-priority order and subsequent refinement trajectory.

Trial ranking correctly chooses the smallest tracked native count in these
cases. Weak Grover's former winning trial rises from 961 to 1,257; the unchanged
1,065-gate trial then wins. Dense Grover's former winner rises from 24,224 to
25,169, so the unchanged 24,704-gate trial wins. No score-selection defect was
found. Workspace labels can also affect full-layout reconciliation, but the
traced first divergences already occur inside search. Local surrogate costs do
not identify the best complete emitted circuit.

The broader cohort adds 164 cases at compiler seeds 41 and 2027: QFT, QAOA, QPE,
QFT adder, W states, GHZ, repeat-until-success, weak-measurement Grover, and
noncommuting random circuits. Nominal widths are 16/32, with full/partial
occupancy on lines, rings, grids, and a synthetic irregular mesh. Weak Grover
caps the marked string at ten bits. All 140 primary cases use CX/CZ; 24
additional QFT/adder/random cases use ECR or square-root-iSWAP. Matching CX/CZ
counts are not independent circuit samples.

Across 328 baseline/PR pairs, two-qubit counts decrease from 260,924 to 260,666
(0.099%): seven wins, 318 ties, three losses. The worst loss is 0.66%, on a
square-root-iSWAP random circuit. All 656 outputs verify, including QCO
linearity. These results limit the observed scope of the regression; they do not
erase the original cohort's larger losses. Structured counts remain static.

Four queue prototypes were screened on the same 140-case seed-41 cohort:

| Equal-priority ordering         | Native count change | Wins / ties / losses |
| ------------------------------- | ------------------: | -------------------: |
| Lower accumulated cost          |              +0.87% |         6 / 120 / 14 |
| Greater accumulated cost        |              +3.05% |          6 / 92 / 42 |
| Lexicographic program positions |              +0.22% |         38 / 66 / 36 |
| Insertion order (FIFO)          |              +2.89% |          6 / 90 / 44 |

None is retained. Greater-cost and FIFO ties regress ring QFT by 30%; canonical
position ordering regresses a QFT adder by 14.4%. Lower-cost ties also increase
native structured counts by 4.57% on the original three-seed cohort. The
previous greater-cost improvement on the original cohort does not generalize.
Adding tie-policy options, replaying complete routers, or retaining duplicate
states merely to reproduce heap order is not justified by these measurements. No
lean quality fix has been validated; the production implementation and its
measured performance gains remain unchanged. Broader quality timings overlapped
builds and are not performance evidence.

### Architectural disposition

The state-key change belongs in the mapping search and introduces no public API
or persistent traversal state. Hashing and equality use every program qubit,
including spectators. Full layouts still own physical routing and region
reconciliation. Independent architectural review found no additional blocker.
Keep this as a draft until routing quality meets the gate or maintainers
explicitly accept the measured tradeoff. Item 4 remains a measured no-change
result; persistent traversal is not justified by the tested smaller variants.

### Performance

Five alternating baseline/candidate pairs produced stable output hashes within
each variant for all 24 cases. Representative median wall times and maximum RSS
across the five repetitions follow (RSS in MiB):

| Case                              | Baseline seconds | Candidate seconds | Baseline RSS | Candidate RSS |
| --------------------------------- | ---------------: | ----------------: | -----------: | ------------: |
| 32-QAOA native                    |            0.435 |             0.247 |          765 |           346 |
| 32-QAOA topology                  |            0.432 |             0.255 |          759 |           338 |
| 32-Grover native                  |            0.173 |             0.088 |          203 |            68 |
| 32-Grover topology                |            0.149 |             0.064 |          201 |            65 |
| 32-weak-measurement Grover native |            0.190 |             0.106 |          212 |            79 |
| 12-QFT native                     |            3.262 |             3.245 |          428 |           428 |

The sum of per-case median wall times falls from 11.774 to 10.947 seconds
(7.0%). This is a descriptive cohort total, not a workload-weighted deployment
estimate. Wide sparse cases improve materially; dense QFT is essentially
unchanged. Small changes on short cases remain sensitive to shared-host noise.
Fresh candidate profiles were collected separately for dense native/topology QFT
and wide native QAOA. No instrumented node-count comparison was rerun; the
historical key counts establish the hypothesis, not a new fallback claim.

Item 4's reservation removal increased dense native QFT median time from 3.024
to 3.293 seconds in its separate five-pair comparison. Inline pending-map
storage and callback specialization showed no consistent benefit. Callback
results use four pairs; a fifth overlapped validation work and is excluded.

Windows and hosted CI are unverified. The PR remains draft because routing
quality fails the adopted gate, independently of these validation limits.

# Decomposition and native-synthesis performance

Status: the coordinate-reuse and Z-frame improvements are published in PR #2959.
The larger-circuit ancilla follow-up is complete and locally validated.

## Outcome and scope

Reuse Weyl coordinates for SWAP-related keys in `NativeCostTable::precompute`,
in `mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/TargetSynthesis.cpp`. The
table keeps its bitwise keys, insertion order, collision handling, seed, and
1024-entry bound. No public API, synthesis algorithm, routing heuristic, pass,
option, or cross-trial mutable state is added.

SWAP conjugation preserves the coordinates of `M` and of `SWAP * M`. The six
orientation keys form groups of **two and four**, not three and three. The
implementation reuses an existing representative through the table's
collision-checked lookup. A missing representative falls back to decomposition.
Keeping six keys means table coverage does **not** increase.

Z-frame propagation also avoids repeated control-flow cleanup and reuses cached
all-site RZ support, with the existing per-site check as a fallback. Ancilla
discovery advances through live wires in source order and classifies tensor
slots during history traversal, preserving candidate selection.

The regression in `test_target_synthesis.cpp` compares shared and independent
cost queries for all six orientations, two seeds, near-zero and chamber-boundary
coordinates, and forward/reversed CX and parameterized RXX targets. Existing
numerical-failure, eviction, synthesis-equivalence, and mapping tests remain.

## Evidence and decisions

Baseline: `c7e8795c8c` (the plan branch). Environment: DGX Spark, Linux AArch64,
Clang 23.1.6, LLVM/MLIR 23.1.0, Release with ThinLTO and mold. Compilation uses
seed 2023, four mapping trials, and affinity to CPUs 5–8. The host is shared;
affinity does not reserve CPUs. Wall-clock results are diagnostic, not an
isolated throughput claim.

Temporary instrumentation counted each Weyl caller and coordinate-cache hit,
ancilla searches and visited operations. It also measured thread CPU time around
Weyl calls, `SingleQubitRunFusion::apply`, and `propagateZFrames`, and process
CPU time around owning passes. Timed figures below use medians of five runs
after coordinate reuse. Instrumentation and all experiment artifacts are kept
outside the submitted change. `perf record` on the performance-core PMU
collected samples, but executable call stacks did not unwind sufficiently to
attribute inclusive pass costs. The CPU-time probes supply that attribution
instead.

### 1. Reuse coordinates: implemented

Garnet workloads and deterministic call counts:

| Workload                          | Precompute before | Precompute after | Mapping coordinate misses | Shared-table hits |
| --------------------------------- | ----------------: | ---------------: | ------------------------: | ----------------: |
| QFT, 16 qubits, period exponent 3 |                36 |               34 |                        56 |              3185 |
| QAOA, 16 qubits, depth 4          |                 2 |                2 |                         4 |                23 |
| Grover, 8 qubits, 4 iterations    |               186 |               94 |                      3121 |             62979 |
| Modular multiplier, 2-bit modulus |               152 |               84 |                       305 |              6959 |
| Modular multiplier, 3-bit modulus |               212 |              118 |                       710 |             14230 |
| Modular multiplier, 4-bit modulus |               272 |              152 |                      1140 |             24588 |
| Multiplexer, 5 / 9 / 17 qubits    |      18 / 34 / 66 |     10 / 18 / 34 |               4 / 17 / 55 |    49 / 137 / 339 |

Symmetric or previously cached matrices already avoid some calls in the
baseline, so this is not a universal threefold reduction. Each newly inserted
orientation uses its representative when available. All baseline workload
programs matched byte for byte in the instrumented prototype, including native
gate counts and placement.

Final uninstrumented comparison: all 17 workload/device cases produced identical
text in all five baseline/final pairs (170 compilations). Illustrative median
wall times in milliseconds:

| Garnet workload      | Baseline ms | Final ms |
| -------------------- | ----------: | -------: |
| qft-16               |      114.47 |   105.89 |
| qaoa-16              |       56.76 |    62.16 |
| grover-8             |     1171.60 |   865.38 |
| modular-multiplier-4 |      508.16 |   544.37 |

The shared host was heavily oversubscribed during these runs. Differences are
inconsistent across workloads; they do not establish an end-to-end speedup. The
supported improvement is fewer numerical decompositions with identical measured
output.

### 2. Repeated mapping work: stop at the measurement gate

Weyl calls take 1.5–6.7% of mapping CPU time after reuse, below the 20%
threshold. This stopping decision applies after the first optimization. For
Grover, the shared table handles 85.4% of coordinate requests; the last-result
and local caches handle 2.7% and 7.7%, leaving 4.2% misses. QFT has 76.5%
shared-table hits; QAOA instead resolves 91.5% through the last-result cache.

Do not add approximate matrix keys, a synchronized memo, a coordinates-only
numerical API, or reversed local-factor construction. Aggregate counters across
trials suffice for this stopping decision; no per-traversal cache redesign is
justified by these workloads.

### 3. Ancilla search: initial measurement gate

The proposed multiplexer and modular-multiplier workloads make **zero**
`findAncillas` calls: their controls are below the ancilla-candidate threshold.
Both were compiled at increasing sizes for Garnet and all-to-all DDSIM. Garnet's
topology is sparse, not a nearest-neighbour chain.

Supplemental Grover workloads hold the width at eight and double iterations:

| Iterations | Ancilla searches | Operations visited | Decomposition CPU ms | Decomposition wall ms |
| ---------: | ---------------: | -----------------: | -------------------: | --------------------: |
|          4 |                8 |                844 |                6.271 |                15.710 |
|          8 |               16 |               3032 |               13.892 |                28.773 |
|         16 |               32 |              11440 |               31.449 |                64.503 |

The scan count is superlinear, but pass CPU time grows 2.22× and 2.26×, below
the threefold threshold. Retain the current clean-first/source-order selection.
At these sizes, a forward sweep did not meet this investigation's acceptance
gate. The larger-circuit follow-up below supersedes that stopping decision. This
does not establish linear scaling for larger circuits or long tensor histories.

### 4. Cross-pass sharing: no change

On Garnet, `FuseTwoQubitGates` makes no Weyl calls in these workloads. Native
synthesis makes 42 / 1 / 664 / 441 calls for QFT / QAOA / Grover / the largest
modular multiplier, respectively. Weyl work is 0.2–2.6% of synthesis CPU time.
On DDSIM, the modular-multiplier fusion and synthesis passes each make 7 / 8 / 9
coordinate decompositions, but synthesis spends below 0.2% of its CPU time
there. Sharing target-independent data across passes is not justified.

### 5. Z-frame propagation: two measured improvements

The follow-up baseline includes the coordinate-reuse change above. Temporary
thread-CPU timers split `propagateZFrames` into capability setup, initial
fusion, and the frame walk. Initial fusion and the walk account for roughly
equal shares on Garnet. On DDSIM, setup instead dominates: about 2.8 ms versus
tens of microseconds for the walk.

Retained changes:

- Disable region simplification only for the local fusion inside Z-frame
  propagation. Fusion does not change control flow, and the target pipeline
  already owns control-flow cleanup. Keep scalar folding, constant CSE,
  traversal order, and listener notifications.
- Reuse `CompilerTarget::findOperation`'s cached all-site coverage and
  unknown-angle query as a sufficient test for unrestricted RZ support. Retain
  the per-site fallback for heterogeneous aliases and unrestricted
  native-operation sets. A global supported-gate flag alone is insufficient: it
  can mean support on only some sites.

Five alternating baseline/final pairs on the same instrumented binary isolate
these changes. Median Z-frame CPU times:

| Workload                        | Before ms | After ms | Reduction |
| ------------------------------- | --------: | -------: | --------: |
| qft-16 (Garnet)                 |    10.580 |    9.268 |     12.4% |
| grover-8-iterations-16 (Garnet) |   304.918 |  257.890 |     15.4% |
| multiplexer-17 (DDSIM)          |     2.923 |    0.041 |     98.6% |

For the larger Grover case, initial-fusion CPU time fell from 180.7 to 121.7 ms.
For DDSIM multiplexer setup, it fell from 2.892 to 0.018 ms. The frame walk is
unchanged. Sampled block-order maintenance and site-listener work did not
justify adding mutable state or weakening scalar-dominance checks. Disabling
folding/CSE was not retained: immediate constant recognition and standalone
constant sharing remain required contracts. No timing code, profiling option,
cache, or new API is added.

Uninstrumented public CLI measurements also use five alternating pairs for each
of the 17 workload/device cases. All 170 compilations emit identical programs,
including native counts and layout. Representative median process CPU times (sum
of user and system CPU time, including worker threads):

| Workload                        | Before ms | After ms | Reduction |
| ------------------------------- | --------: | -------: | --------: |
| qft-16 (Garnet)                 |    46.841 |   44.353 |      5.3% |
| grover-8-iterations-16 (Garnet) |  1046.312 |  982.693 |      6.1% |
| modular-multiplier-4 (Garnet)   |   163.526 |  157.938 |      3.4% |
| multiplexer-17 (DDSIM)          |    86.666 |   77.644 |     10.4% |

CPU medians improve in 16 of 17 cases; DDSIM multiplexer with nine qubits is
flat (+0.2%). Shared-host load remains a limitation: wall time is noisy, and
these measurements do not establish isolated throughput or gains for other
workloads.

`ZFramesRequireUnrestrictedRZOnEverySite` checks propagation across CZ with
uniform, heterogeneous, partial, fixed-angle, and unrestricted RZ capabilities.
It checks IR verification, linearity, target conformance, native gate counts,
and complete unitary equivalence. Replacing the shortcut with the global
supported-gate flag makes the partial-coverage case fail its gate counts and
target-conformance check; the correct implementation passes. Independent source
critique and a Ponytail review found no remaining issue.

## Reproduction and validation

Use the repository's native release build and test presets, with supported local
toolchain overrides. Generate QC instances outside the repository with the
following command; get schemas using `bench describe`.

```sh
mqt-core bench generate --instance-specification <spec.json> \
  --format qc --output <directory>
```

Workload parameters:

- `qft`: `{"qubits":16,"period_exponent":3}`.
- `qaoa`: `{"qubits":16,"depth":4}`.
- `grover`: `{"marked_bitstring":"10101010","iterations":4}`, then 8 and 16.
- `multiplexer`: `{"qubits":5}`, then 9 and 17.
- `modular-multiplier`: modulus `11` / `111` / `1111`, multiplier `10` / `010` /
  `0010`, multiplicand `++` / `+++` / `++++`, control `+`.

Wrap parameters with `schema_version: 1` and the benchmark ID. The executable
benchmark ID uses a hyphen: `modular-multiplier`.

Compile with `--seed=2023 --mapping-trials=4 --emit=qco-optimized` and
`--qdmi-device=mqt.sc.iqm.garnet`. Select the executable payload explicitly:

```text
--payload-spec='#mqt.payload_spec<format = <id = "iqm-json", version = "", profile = "", encoding = text>, capabilities = [], optional_capabilities_known = false>'
```

For the multiplexer and modular-multiplier all-to-all runs, use
`--qdmi-device=mqt.ddsim.default` without that payload argument. Compare textual
stdout, including layout and native operations. Pair five uninstrumented
baseline/final executions per case, alternating their order.

Baseline: all 4337 CTest tests passed. The coordinate-reuse step passed 4338;
the Z-frame follow-up passes all 4339 tests after restoring the deliberately
incorrect mutation. All 170 uninstrumented baseline/final paired compilations
produce identical output. The full release build passes. C++ lint is clean for
all changed source files; the test file was rechecked after fixing its sole
style finding, and all 108 native-synthesis tests passed again. Repository lint
passes. Hosted CI and Windows/macOS validation were not run.

Before PR submission, the changes were applied to current `main` at
`0b17881c6c`. The native release build and all 4348 CTest tests passed;
repository lint also passed. The benchmark measurements above retain their
original baseline and were not repeated on this updated base.

## Larger-circuit ancilla follow-up

Baseline: PR #2959 head `4d1389d4c4`. The toolchain and CPU affinity match the
experiments above. Inputs and temporary probes are outside the repository in
`ancilla-scaling-20261107`. The isolated public CLI runs
`compile --run-pipeline --pass-pipeline='builtin.module(decompose-multi-controlled)'`.

Workloads cover 7 and 15 controls, 64/128/256 repeated MCX gates with one clean
spare wire, a control with enough early clean spares, dense and sparse tensor
histories of 256/1024/4096 extract/H/insert steps followed by 32 seven-control
gates, and 1024/4096 live scalar wires with four seven-control gates. An H gate
between MCX gates prevents adjacent inverses from cancelling. Dense histories
cycle through ten slots; sparse histories touch only slot zero.

The larger controlled cases exceed the original scaling gate. For 15 controls,
128 to 256 gates increases baseline median decomposition CPU from 262.847 to
856.463 ms (3.26x), and search CPU from 127.593 to 571.698 ms. Prefix operation
visits increase from 3,523,776 to 14,141,824.

Retained changes:

- Advance a cursor through the current block during the existing input-gate
  sweep. Keep live quantum values in source order, with indexed removal when
  their sole user is reached. Anchor the cursor at a surviving predecessor so
  the next search sees the emitted replacement. Reset on block changes and
  discard all state before greedy rewriting. The ordered list avoids both
  repeated block-order reconstruction and quadratic removal at large live width.
- Classify tensor slots at their nearest insertion/allocation origin during
  discovery, retaining unknown-state handling and sorted slot order. Stop when
  every slot has been touched. This avoids rewalking the tensor prefix for each
  returned slot without adding a persistent occupancy cache.

Five alternating instrumented pairs produced identical output in all 15 cases.
Median thread CPU times in milliseconds:

| Workload                   | Pass before | Pass after | Search before | Search after |
| -------------------------- | ----------: | ---------: | ------------: | -----------: |
| 15 controls, 128 gates     |     262.847 |    134.243 |       127.593 |        1.756 |
| 15 controls, 256 gates     |     856.463 |    299.781 |       571.698 |        3.815 |
| 7 controls, 256 gates      |     238.127 |    119.742 |       111.706 |        0.807 |
| Dense history, 4096 steps  |      43.197 |     17.889 |        25.527 |        0.851 |
| Sparse history, 4096 steps |      27.821 |     20.412 |        19.788 |       12.026 |
| 4096 scalar wires, 4 gates |       5.368 |      5.288 |         0.466 |        0.503 |

The 15-control prefix visits become 55,024 and 110,448 for 128 and 256 gates.
Sparse histories still require repeated backward walks; no general linear-time
claim applies to tensor discovery. The wide-scalar case slightly increases
search cost while total pass CPU is flat. Shared-host timing variation remains.

Untargeted generated Grover programs preserve loops: even 256 iterations produce
only two candidate searches. These are a control case, not evidence of expanded
circuit scaling. Garnet target compilation expands the circuits. Three
alternating instrumented pairs for Grover at 8 qubits with 64/128 iterations and
16 qubits with 16/32 iterations produced identical output in all 24
compilations. Median search CPU fell from 1.417/5.551/0.206/0.738 ms to
0.149/0.294/0.085/0.156 ms, respectively, but median total compilation CPU
remained approximately 4.4/10.0/9.0/16.9 seconds. Search is too small a fraction
to claim a meaningful end-to-end improvement for these generated programs.

Five alternating uninstrumented pairs for each of the 15 isolated cases also
produced identical output in all 150 compilations. These include parser,
verification, decomposition, and textual output costs. Representative median
process CPU times (milliseconds):

| Workload                   |  Before |   After |
| -------------------------- | ------: | ------: |
| 15 controls, 256 gates     | 842.459 | 568.426 |
| 7 controls, 256 gates      | 290.611 | 237.957 |
| Dense history, 4096 steps  |  68.922 |  49.078 |
| Sparse history, 4096 steps |  61.289 |  53.314 |
| 4096 scalar wires, 4 gates |  29.035 |  29.440 |

The supported benefit is eliminating repeated prefix work on long expanded
controlled circuits, with lower tensor-history overhead. The small wide-scalar
case is effectively flat, with about 0.4 ms additional process CPU. Neither
instrumented nor uninstrumented runs reserve the shared host; small timing
changes remain inconclusive.

`AncillaDiscoveryRestartsForEachBlock` uses two borrowed-qubit helper functions,
repeated gates, and full seven-qubit functionality comparison. Removing the
block-state reset makes it fail with cross-region operands; the correct version
passes. Existing tests cover clean/dirty preference, repeated restored ancillas,
dynamic-index barriers, sparse huge tensors, and target capacity. Independent
specialist critique and a final Ponytail review found no remaining issue. The
final native release build passes, as do all 4349 CTest tests and whole-file C++
lint for both changed source files. Repository lint passes. No profiling code or
mutation remains. Hosted CI and Windows/macOS validation were not run.

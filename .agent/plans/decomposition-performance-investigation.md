# Decomposition and native-synthesis performance

Status: complete. The coordinate-reuse and Z-frame improvements are implemented
and locally validated.

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
all-site RZ support, with the existing per-site check as a fallback.

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

### 3. Ancilla search: stop at the measurement gate

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
A forward sweep would introduce mutable wire/occupancy state without meeting
this investigation's acceptance gate. This does not establish linear scaling for
larger circuits or long tensor histories.

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

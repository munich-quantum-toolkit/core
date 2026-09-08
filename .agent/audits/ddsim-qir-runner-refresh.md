# DDSIM QIR runner after DD and QCO improvements

Status: all five confirmed findings implemented and locally validated. Baseline:
PR #2466 at `b115e36209ee337b4f196472ec322779a7f6e919`, based on upstream main
`b75b02fa95e7e686a6d14c825d8d3ffa3a9aa421`. Scope: QIR analysis, execution and
ordered sampling; runtime DD capacity; shared unique-table growth;
state-extraction validation.

## Existing shared behavior

Main already supplies shared DD gate construction (PR #2455), zero-capacity
packages for the QCO sampler (PR #2459), and QIR lowering cleanup (PR #2463).
QIR retains shared gate construction through the QCO adapter. The QC/QCO
conversion and modifier work in PR #2464 does not replace the sampling or
allocation changes here. No additional gate cache or execution engine is needed.

## Implemented findings

### Static Adaptive programs can reuse terminal sampling

`getStaticSamplingOutputs` in `IRRewriter.cpp` now accepts Base and Adaptive
profiles. The same body proof requires constant gate arguments and static IDs,
an acyclic unconditional path, terminal measurements, known output mapping and a
successful return. Allocation, reset, feedback, memory operations, unknown calls
and loops still use ordinary per-shot execution. This optimizes static programs
carrying an Adaptive profile; it does not defer genuinely adaptive measurements.
State extraction remains Base-only.

The sampling-plan tests run for both profiles, including rejected effects and
control flow. The missing-initialize fallback test explicitly reads a result, so
it still exercises ordinary per-shot resets after the eligibility change.

### Full-register samples move directly into the output batch

`Runtime::sampleMeasurements` owns the complete DD sampling loop. It classifies
physical output order once after state preparation. For a full ascending
register, it reverses the DD sampler's string in place and moves it into the
batch. Subsets, repetitions and other permutations retain generic mapping. The
runtime's last measurement string still matches the last returned shot. Empty
output maps and zero-shot calls retain their behavior.

`QIRBatchSampling.PreservesWideSeededOutputMappings` checks asymmetric 64-bit
states, ascending and descending outputs, repeated/subset records, SWAPs and
seeded repeatability. An independent comparison against the baseline executable
also produced byte-identical seeded sequences for all four mappings. RNG
consumption within each sampling path is unchanged.

### Static capacity is exact; unknown resources grow geometrically

`Runtime::QState` and package recreation start at zero capacity. Declared
resources grow directly to their capacity. Dynamic and metadata-free execution
allocate at least `Package::DEFAULT_QUBITS` on first quantum use, then double
capacity as needed, bounded by the DD qubit limit. The state itself spans only
used or declared qubits. Warm resets retain package capacity; transferring a
state leaves lazy recreation for the next job.

Native runtime tests cover phase, logical width and state transfer while
incrementally growing static IDs and dynamic allocations. JIT tests cover exact
declared capacity, unused qubits, zero-qubit states and repeated transfers.

### Unique-table growth initializes only appended levels

`UniqueTable::resize` resizes the outer storage and initializes bucket storage
and invariant statistics only for appended levels. Existing buckets and
statistics survive unchanged. It no longer constructs a full temporary bucket
table on every resize. This shared change benefits both QIR and QCO.

`DDPackageTest.UniqueTableGrowthPreservesLookupsStatisticsAndRoots` checks
vector and matrix lookup identity, preserved statistics, new levels, retained
roots, forced garbage collection and regrowth after clearing. Shrinking
populated tables retains its existing unsupported cleanup limitation.

### State extraction rejects unproven helper effects

Previously, a Base-tagged entry containing `H(q0); call readout(); return`, with
measurement inside `readout`, was accepted and returned a collapsed state. Such
helper calls are outside the supported flat Base structure. The extraction scan
now rejects defined or indirect callees and non-ordinary call instructions
before changing IR. This closes the acceptance gap without interprocedural
analysis. Regressions check direct and indirect helpers, calls after a direct
irreversible boundary, and unchanged valid IR after rejection.

## Combined performance evidence

Seven-trial median process CPU times on native ARM64, GCC release and LLVM/MLIR
23.1. Each comparison ran serially on CPU 0 and was repeated in reverse order.
Batch measurements include preparation and output collection, but exclude JIT
construction. Allocation counters measure C++ allocation requests, not peak RSS
or all malloc calls. These workloads have small, compressed DDs.

| Workload                                      |       Baseline | Combined patch |
| --------------------------------------------- | -------------: | -------------: |
| Adaptive Bell, 10,000 shots                   |       12.04 ms |        0.38 ms |
| Adaptive GHZ32, 10,000 shots                  |     470-471 ms |   2.34-2.35 ms |
| Base GHZ32, 100,000 full-register shots       | 33.27-33.52 ms | 23.11-23.23 ms |
| Base GHZ64, 100,000 full-register shots       | 60.57-61.09 ms | 41.00-41.48 ms |
| Empty runtime, requested bytes                |     26,075,208 |      8,767,048 |
| Incremental 256-qubit growth, requested bytes |    376,938,454 |    143,106,006 |

Full-register output allocation calls fall from about 200,000 to 100,000 per
100,000-shot batch. Generic single-output sampling did not regress in either
comparison. Incremental 32-qubit growth also retained its allocation and timing
behavior. At 128 and 256 qubits, growth timings varied between runs and did not
establish a stable combined latency benefit; the allocation reduction is the
supported claim. Do not multiply gains from the earlier isolated prototypes.

## Deferred candidates

Skipping RNG draws on deterministic DD branches changes seeded sequences and
regressed uniform superposition sampling in the isolated experiment. It remains
deferred. A general gate cache, an LLVM optimization pipeline, replacing QIR's
direct growth with Kronecker construction, removing ordered shots, and removing
the forwarding DD adapter have no evidence warranting changes in this PR.

## Validation

Focused release tests passed: 46 JIT/analysis, 79 runtime and 180 DD tests. Full
Clang CTest passed 3,190 tests with no failures and one existing skip.
`uvx nox -s cpp-lint -- b75b02fa95e7e686a6d14c825d8d3ffa3a9aa421` checked all
changed C++ implementation and test files with zero findings. Both
`uvx nox -s lint` and the complete warning-as-error
`uvx nox --non-interactive -s docs` build passed.

The combined benchmark's output correlation, widths and final-record checks
passed, and baseline/updated seeded mapping output matched byte for byte. Hosted
CI is separate from these local results.

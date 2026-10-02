# Final QIR execution performance and determinism audit

Status: all three findings implemented and re-audited. Date: 2026-09-10.
Baseline: upstream main `ad74680f1ef380456a1b89a810ef33ee8d218f69`. Scope: QIR
JIT analysis and execution, the QIR runtime, shared DD calls on its hot path,
and DDSIM QDMI result serialization. The accompanying Git commit identifies the
implementation tested against that baseline.

This audit applies the evidence standard in
[issue #2253](https://github.com/munich-quantum-toolkit/core/issues/2253).
It covers this subcomponent, not the issue's whole MLIR scope.

## Result

1. **Resolved P1: variable-width histogram buffer sizing.** DDSIM sums each
   key's actual length. A client buffer of the queried size holds every key and
   the final terminator; shorter buffers are rejected before any write.
2. **Resolved P2: quadratic Base extraction boundary search.** Candidate
   selection and validation each make one pass over the irreversible calls. The
   existing terminal-region checks and rejection-before-mutation behavior remain
   intact.
3. **Resolved P2: unordered sparse serialization.** DDSIM caches sparse entries
   in a vector sorted once by numerical basis index. All sparse key and value
   results traverse the same sequence.

The final boundary review also found and fixed an unsupported-width sparse
export: all four DDSIM sparse-result requests now reject states wider than their
`size_t` basis-index representation before DD traversal. No further actionable
issues remain in the reviewed paths and immediate consumers. The histogram
release blocker is resolved. The performance and scope limits below still apply.

## Histogram contract and verification

Sources: `src/qdmi/devices/dd/Device.cpp`, `submitQIRProgramSampling`,
`getHistogram`, and `getShots`; `src/qdmi/Client.cpp`, `getSparseResult`.

Adaptive control can record a result a second time only for one measurement
outcome. The resulting shots and histogram keys need not have equal lengths. The
old serializer multiplied the first key's length by the number of keys, but then
wrote every complete key. For keys `0` and `11`, it reported four bytes and
wrote five: `0,11\0`.

The fixed size calculation follows the adjacent shot serializer's sum of actual
lengths. The regression checks the exact size, complete output, unchanged
sentinel, rejection without writes for a short buffer, and agreement between
histogram values and ordered shots. The C++ client parses variable-length keys
without imposing a uniform width. Existing empty-output behavior is preserved.

A separate public C API probe now reports and writes five bytes, leaving the
next sentinel byte unchanged. The experimental artifacts are retained locally.

## Extraction analysis and performance

Source: `mlir/lib/Dialect/QIR/Execution/JIT/IRRewriter.cpp`,
`prepareForStateExtraction`.

The old search visited irreversible calls in function storage order and compared
each candidate against the others. A valid terminal chain with blocks stored in
reverse execution order caused a triangular number of dominance comparisons.
Repeated terminal measurement of one static qubit is enough to trigger this; it
does not require a large quantum state.

The first linear pass selects a candidate. A call that dominates every other
call either replaces the current candidate or is itself dominated by a valid
candidate. The second pass verifies that the chosen call dominates every other
irreversible call. Self-comparisons remain excluded. The terminal-region walk,
helper-call restrictions, IR rewrite, and sampling eligibility are unchanged.

Final seven-trial process CPU medians, pinned to CPU 0:

| Measurement blocks | Baseline reverse layout | Implementation |
| -----------------: | ----------------------: | -------------: |
|              1,000 |                7.564 ms |       0.647 ms |
|              2,000 |               30.665 ms |       1.343 ms |
|              4,000 |              122.618 ms |       2.661 ms |
|              8,000 |              495.142 ms |       5.795 ms |

At 8,000 blocks, the ranges were 488.709-521.535 ms and 5.622-6.033 ms. The
already-favorable forward layout took 5.388 ms baseline and 5.998 ms with the
implementation. The extra linear pass has a cost on that layout; this is not a
claim that every input becomes faster. Ordinary compiler output with a flat
terminal region does not incur the demonstrated quadratic cost.

Measurements include dominance analysis, terminal checking, and rewriting. They
exclude parsing, verification, JIT compilation, and quantum execution. The table
uses the final implementation comparison.

The added native regression covers reverse block storage order and multiple
measurements in one block. Rejected independent regions must remain unchanged.
The final differential experiment compares complete results and rewritten IR for
1,024 identical generated control-flow inputs:
**131 rewritten, 65 unchanged, and 828 rejected**. Both versions match in every
case; rejected IR remains unchanged and all input/output modules verify. The
inputs include shuffled blocks, unreachable blocks, branches, cycles, and
non-terminal quantum work. This bounded check complements the reasoning; it is
not an equivalence proof for all LLVM IR.

## Sparse ordering and ownership

Sources: `include/mqt-core/dd/DDDefinitions.hpp`, `src/dd/Edge.cpp`, and
`src/qdmi/devices/dd/Device.cpp`, `getSparseResults`.

The shared `SparseCVec` API returns an unordered map. DDSIM previously
serialized that map directly. With the same type and ascending insertion of
indices 0 through 31, the libstdc++ probe begins `31,30,29,12,11,...`, while
libc++ begins `31,30,29,28,27,...`. The mathematical mapping is the same; this
demonstrated noncanonical serialization across libraries, not random variation
between identical runs of one binary. QDMI itself requires paired keys and
values but does not prescribe ascending order. The project's
deterministic-output policy and #2253 require avoiding observable unordered
traversal.

The shared DD map API remains unchanged. DDSIM copies the materialized entries
into its existing lazy cache and sorts the pairs by their integer keys once.
Keys, complex amplitudes, and probabilities all use that vector. `call_once`
publishes the completed sort before any reader proceeds; no reader can observe
partially sorted entries. The retained DD still owns the source state.

The regression initializes the cache through a probability query, checks exact
basis order against unequal real and imaginary amplitudes, compares squared
magnitudes, and repeats both sparse queries. Existing buffer, dense-result,
job-lifetime, and concurrency tests continue to pass. An additional width
regression preserves a 64-qubit export on this platform, including its highest
set bit, and rejects all four sparse result kinds at 65 qubits. This prevents
out-of-range shifts in DD traversal and bitstring serialization. The guard lives
at the shared QDMI result boundary; the raw DD sparse-map API is unchanged.

Sorting costs O(k log k) once for k sparse entries and uses a temporary vector
alongside the materialized map during conversion. The retained vector is
contiguous and subsequent reads do not sort. No large sparse-export latency or
peak-memory improvement is claimed. The alternate-library probe isolates the
container; it is not a complete libc++ build of Core.

## Retained behavior and deferred work

- Static terminal sampling still prepares once. Seeded ordered-shot and SWAP
  mapping tests pass, including repeated/subset outputs and fallback execution.
  Different algorithms or software versions need not reproduce the same draws.
- Runtime maps and the immutable ABI registry are used for lookup, not
  observable iteration. Metadata and shots retain explicit sequence order. DD
  root-map traversal marks reachability; it does not define output or arithmetic
  order. No wholesale container conversion is justified.
- Warm 100,000-gate probes request 100,000 ordinary C++ allocations for X and
  200,000 for CX. Shared DD root insertion/erasure and `dd::Controls`
  construction explain these costs. A shared root-update or control-view change
  still needs evidence covering multiple roots, garbage collection, and control
  validation. These deferred observations are not additional unimplemented audit
  findings.
- The allocation counter excludes aligned allocation and direct malloc calls.
  Tiny repeating DD workloads do not establish behavior for large or low-reuse
  states; X and CX are different workloads, not an A/B comparison.
- Dynamic allocation and feedback retain per-shot execution. No cross-job JIT
  cache, general gate cache, or new LLVM optimization pipeline is justified by
  this evidence. Dense state and probability queries already share a cache.
- Other local DD audits were checked for overlap and left untouched.

## Validation

Environment: DGX Spark ARM64, Linux 6.17.0-1032-nvidia, Clang 23.1.1, LLVM/MLIR
23.1.0. Core targets use `release-clang-ipo`, Clang ThinLTO and mold. The
isolated analysis programs use Clang `-O3` with the installed LLVM libraries.

Publication validation uses upstream base
`9d6526f4827ed96f7a16880325c61fe725f2f737`. The benchmark table retains its
original baseline and measurements; rebasing did not change the measured code.

Final local validation:

- **51 QIR JIT/analysis tests, 80 runtime tests, and 75 DDSIM QDMI tests pass.**
- The isolated implementation passes all **14 IR analysis tests** and the
  **1,024-case differential check**.
- The public API histogram probe confirms the exact fixed size and sentinel.
- C++ lint reports **zero findings** for all changed implementation/test files;
  the required lint-header target also builds.
- Full repository lint and strict documentation generation, including local link
  checking, pass.

No full unrelated CTest sweep or hosted CI validation was performed.
Experimental harnesses, raw measurements, and logs are retained locally and are
excluded from the pull request.

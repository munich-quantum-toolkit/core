# Finite parameters and global phases

Status: complete; all findings applied and locally validated. Base: upstream
main `8ed992b640ccb9bc0936f3929e840330b9dabb65`.

## Outcome and scope

Finite quantum parameters are a precondition. Direct constants receive cheap
sanity checks. Numerical guards remain where finite arithmetic can overflow,
lose phase information, or fail matrix reconstruction. General classical
floating-point behavior and external-format validation are unchanged.

The changes cover shared MQT angle and constant-folding utilities, modifier
canonicalization, and QC/QCO global-phase normalization. The
[contract audit](../audits/finite-parameters-20261017.md) records the findings
and retained consumer contracts. Existing tests are consolidated or extended;
the two MQT suites have 69 cases instead of 83.

## Decisions

- Leave power subtrees unchanged during phase normalization. Large exponents
  amplify phase rounding. Modifier canonicalization owns safe power folding.
- Keep balanced symbolic sums and bound newly merged phases with
  `2 * atan(tan(angle / 2))`. Existing Qiskit, OpenQASM, and DD consumers
  support these operations. Single phases and negations need no wrapping.
- Evaluate constant expressions with a cached depth-first stack. Advance the
  operand cursor before descent and seed pending results as failed, so the first
  unknown operand can end evaluation immediately. Reuse operand and result
  buffers. Retain complete evaluation for constant-dependent exporters.
- Cache scalar results within each isolated scope. Nested classical regions
  share the cache because normalization preserves scalar meanings.
- Hoist eager scalar producers once, in source order, before the outermost
  enclosing inverse or control. Verified modifier bodies exclude classical
  regions, memory effects, and scalar dependencies on qubits. Classical scope
  boundaries and power bodies remain intact.
- Merge known phases before emitting arithmetic, move phase-expression buffers,
  skip recursive dispatch for regionless operations, and remove inverse/control
  wrappers emptied by extraction. Reuse existing modifier inlining helpers.
- Keep synthesis failure checks relative to independent phase preparation. The
  pass permits input changes on failure; unsupported lowering must preserve the
  normalized IR. Search modifier ancestors once and omit final phase preparation
  when native-only lowering cannot introduce phases.
- Use nullable attributes for cache entries and folding results. A missing entry
  is unevaluated, a null entry records failure, and a nonnull entry records a
  constant. Numeric results retain `std::optional<double>`. Document the source
  and ABI migration in the PR description; compiled consumers must rebuild.
- Start one-off expression caches with capacity for four entries. Literals and
  block arguments bypass allocation. This reduces storage and CPU time for short
  queries, with a small growth cost for intermediate expressions.
- Reuse folded angles when counting Pauli entanglers and emitting bounded
  entanglers. Skip evaluation for local terms and fixed or unrestricted domains
  during emission. A known constant also selects bounded angle reduction;
  materialize scaled arithmetic only when the emitted rotation needs it.
- Keep the explicit evaluation stack: plain recursion exhausts the stack on
  valid deep input, and a depth cap rejects constant-dependent exports. Combine
  PR #2775's scratch-buffer reuse and cache fast path with pending failure
  entries, which remove the need for its separate failure-propagation loop.

## Validation

The final `cmake --build --preset release-clang-ipo` build and
`ctest --preset release-clang-ipo --parallel 8` run passed 4015 tests, with one
existing skip (`ScQDMIJobSpecificationTest.QueryJobId`). This is the complete
configured native suite, including native synthesis. The configuration uses
Clang 23.1.4, LLVM/MLIR 23.1.0, ThinLTO, and mold on Linux aarch64. This run
includes nullable folding results, compact caches, and deferred Pauli angle
materialization.

The Python command

```sh
uvx nox -s tests-3.14 -- test/python/test_mlir.py \
  test/python/test_mlir_parameter_binding.py \
  test/python/test_mlir_qiskit_translation.py \
  test/python/test_mlir_qiskit_target.py -n 4
```

passed all 840 cases at `e814d4fad` with Python 3.14.7 and Qiskit 2.6.0. The 243
phase/symbolic/parameter cases also passed with Qiskit 2.5.2. Both versions
passed six QC/QCO full-unitary binding, Qiskit export, and OpenQASM round-trip
probes. All four original deep-expression and large-power reproducers passed.

`uvx nox -s lint`, `uvx nox -s cpp-lint`, and `git diff --check` passed. C++
lint checked every changed implementation/test file in full. No binding
declarations changed. Implementation and tests are 229 lines smaller. Existing
numerical and consumer oracles remain covered.

One-off correctness probes and matched timing experiments live in the ignored
`.agent/benchmarks/finite-parameters/` folder. Each comparison retains its exact
source patch, binary hashes, raw samples, reproduction commands, and plots. CPU
affinity does not reserve host resources; timings are diagnostic and do not
establish application-wide speedups. Measurements at `e814d4fad` against the
preceding implementation show 8192 scalar producers at depth 64 fell from a
median 14.82 ms to 2.17 ms; a shared expression across 128 classical regions
fell from 280.2 ms to 25.8 ms. The phase-only control and observed allocation
requests were unchanged within sample spread. Allocation-request counts are not
peak or resident memory measurements.

Nullable attributes reduce returned handle size from 16 to 8 bytes. A map of
100001 entries uses 4,227,072 bytes instead of 6,324,224 bytes, a 33% reduction.
The small initial capacity uses 132 bytes instead of 1032 bytes for two or four
entries. These are raw cache-storage measurements, excluding referenced IR,
attributes, and allocator overhead; they do not measure peak RSS.

Seven matched CPU-time samples compare the compact folder against `7ef355046`.
The same process evaluates shared, verified IR with three implementations and
rotates their order. A 64,000-operation constant chain changed from 40.57 ms
median (40.42–41.12 ms) to 37.19 ms (36.97–37.89 ms); a dynamic chain changed
from 13.56 ms (13.46–13.63 ms) to 11.74 ms (11.61–12.22 ms). Against nullable
attributes with default cache capacity, the small initial capacity reduced
short-query CPU time by 1.3–5.0%, with a 0.4% growth cost at depth 16 and no
material deep-chain regression. All 126 samples returned the expected result.
The local `compact-cache/` benchmark folder retains all medians, ranges,
controls, raw samples, exact sources, and plots. Earlier wall-time results were
noisy. CPU affinity does not isolate shared hardware; neither experiment
establishes an application-wide speedup.

## Limits

Non-finite quantum parameters remain unsupported. Power normalization is owned
by modifier canonicalization. Hosted CI and other platforms are separate from
local validation.

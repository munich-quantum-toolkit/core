# DD simulation performance investigations

Status: complete. Two changes retained; estimator API and collection-default
changes rejected by their measurement criteria. Called-function sampling has a
separate proposed follow-up.

## Scope and decisions

The investigation covers Pauli expectations, estimator parameter sweeps, worker
allocation, garbage collection, and adaptive sampling in the DD package, QCO
interpreter, QIR runtime, and DDSIM worker. Public APIs, gate embedding,
terminal sampling ownership, and garbage-collection invalidation stay unchanged.

- Keep direct Pauli contraction in `Package::expectationValues`. It meets the
  1.5× large-workload threshold and improves the small workload.
- Close estimator API work: module copying and package construction do not reach
  the required 30% of the small-circuit sweep. Parameter substitution and its
  compiler work must not be counted as module copying.
- Keep one automatic QCO worker-count owner. `WorkerMain.cpp` passes zero for
  automatic selection and caps external requests at the job maximum. Explicit
  counts retain their path.
- Reject larger default collection limits. They do not improve two workloads by
  10% within the 25% memory budget.
- Keep structured branching serial and QIR replay unchanged. Investigate called
  functions separately in
  [the focused follow-up](dd-called-function-branching.md).

## Measurement conditions

Baseline: `858c9b718b4dd01c98dbbfe29efe11e38835c6c1`, from
`origin/claude/optimistic-cray-o2goe1`. Native builds use the machine-local
`release-clang-ipo` preset: Clang 23.1.6, LLVM/MLIR 23.1.0, Release, ThinLTO,
and mold on NVIDIA DGX Spark, Linux AArch64. Python is 3.14.10 and Qiskit 2.6.1.

Every table reports five fresh-process runs unless stated otherwise. Values are
medians with minimum–maximum ranges. Package peak memory estimates exclude
unused capacity and memo tables; process maximum RSS is measured separately.
Final paired Pauli and program runs used performance core 19; estimator runs
used core 17. Pauli pairs alternate baseline/candidate order. The host had
concurrent builds and other workloads. These are local diagnostic measurements,
not isolated-machine throughput claims. Process CPU timings also accompany the
Pauli and estimator wall times.

Harnesses, exact inputs, raw JSON statistics, RSS, and logs are outside the
repository in
`/home/nvidia/.codex/experiments/dd-simulation-performance-20261107-858c9b718b/`.
The working copies remain in `/tmp/dd-performance/`. The baseline DD archive and
executable are retained independently of the candidate. Sampling and collection
probes exercise code unchanged by the Pauli implementation. Temporary sampling
instrumentation counts group visits/replay paths and reports package statistics;
it is not part of the production diff.

## 1. Pauli expectations

The state has 20 qubits, two layers of seeded Ry and phase rotations, and
alternating nearest-neighbor CNOTs. The Hamiltonian has 1,000 distinct,
Jordan–Wigner-style four-endpoint X/Y strings with intervening Z parity strings
and real coefficients. Seed: 42. The small case uses eight terms on the same
preparation. This is a synthetic molecular-style workload, not a chemical
accuracy benchmark.

| Terms | Baseline time                | Direct contraction time   | Speedup | Baseline/candidate peak DD MiB | Baseline/candidate maximum RSS KiB |
| ----- | ---------------------------- | ------------------------- | ------- | ------------------------------ | ---------------------------------- |
| 8     | 129.313 µs (126.401–135.538) | 72.080 µs (70.897–87.761) | 1.79×   | 0.145 / 0.112                  | 6,984 / 6,924                      |
| 1,000 | 81.930 ms (73.770–83.001)    | 25.821 ms (1.925–25.961)  | 3.17×   | 2.406 / 0.112                  | 10,024 / 7,084                     |

Time coefficients of variation are 3.29%/10.03% for eight terms and 4.87%/51.06%
for 1,000 terms, baseline/candidate respectively. Peak used vector nodes fall
from 29,188 to 1,408 in the large case. Both versions produce the same reported
observable values.

The recursion factors incoming conjugated bra and ket weights outside its
memoized complex contraction. Keys include both node pointers and the level; the
memo belongs to one Pauli string. Below the lowest nonidentity operator it uses
the existing private inner product, including scalar terminals. It handles
omitted levels and preserves validation, identity terms, repeated-term reuse,
finite-result checks, and borrowed roots. It creates no intermediate DD nodes.
The memo accepts at most `ctVecInnerProdNumBucket` node pairs (4,096 by
default); further pairs are recomputed without storage. This uses the existing
inner-product cache setting and bounds scratch memory. The large-case process
CPU medians are 9.979 ms and 1.950 ms, a 5.12× reduction; the wall-time spread
reflects host scheduling. The initial unbounded prototype increased RSS in a
less-compressible four-layer probe and was rejected.

A single additional four-layer, 20-qubit/eight-term probe checks scratch-memory
scaling. Baseline, unbounded prototype, and capped candidate respectively use
276,480 / 401,920 / 276,548 KiB peak RSS; all have 227.508 MiB peak DD storage.
Expectation evaluation takes 32.381 / 22.658 / 7.832 seconds and reports the
same result. State preparation dominates total process time and host load
varies; this diagnostic establishes the memory tradeoff, not a repeatable
speedup claim.

The exhaustive complex three-qubit oracle uses a one-entry cache to exercise
uncached recomputation. It also repeats high-only strings across observables and
evaluates again after forced collection. Reversing the candidate Y sign in an
external mutation build makes that oracle fail.

## 2. Estimator profile

The Python module was relinked against the saved baseline DD archive for this
profile; the other measured code is unchanged from the baseline. One process
runs one warm-up sweep followed by five measured sweeps. Each sweep has 100
parameter bindings, Ry gates with one shared parameter and qubit-dependent
offsets, a CNOT chain, and one Z expectation. The profile runs
`DDEstimator._run_pub`, times its actual calls, and separates `program.copy()`
from in-place parameter substitution. A separate 100-constructor probe measures
package setup. Numerical outputs are checked against cosine values.

All times below are milliseconds, with median (minimum–maximum).

| Qubits | Total                  | Module copy         | Parameter substitution | Expectation evaluation | Package setup probe |
| ------ | ---------------------- | ------------------- | ---------------------- | ---------------------- | ------------------- |
| 6      | 41.730 (41.346–50.465) | 0.854 (0.845–0.880) | 10.449 (10.258–10.615) | 28.474 (28.279–37.119) | 0.351 (0.313–0.371) |
| 20     | 72.253 (71.572–73.460) | 2.306 (2.285–2.320) | 21.377 (21.293–21.484) | 45.549 (44.760–46.485) | 0.398 (0.387–0.418) |

Copy plus setup is approximately 2.9% of the six-qubit sweep and 3.7% of the
20-qubit sweep. Median copy CPU times are 0.844 ms and 2.286 ms; median total
CPU times are 41.714 ms and 72.237 ms. Independently measured phase medians need
not sum to the total median. Earlier candidate profiles under heavier host load
are retained as diagnostics, not mixed into this baseline table. The 30%
threshold is not met; no Python API or supported-circuit change is justified.

## 3. Worker selection

`sampleProgram` retains `clamp(shots / 256, 1, 8)`. The QASM worker sends
`workers = 0` in automatic mode and applies `min(desired, maxWorkers_)` in its
external callback. Existing explicit worker and QIR selection paths remain.

The called-function QCO regression compares automatic sampling with an external
serial grant against a full automatic grant at 1,024 shots. It checks exact
seeded output equality under the upstream logical-shot-stream contract. QDMI
sampling also checks explicit one-worker output. Top-level structured QASM
programs return through branch sampling before requesting workers, so that QDMI
test alone does not exercise the callback. OpenQASM `def` subroutines are
currently rejected by the frontend; do not claim a QASM called-measurement
end-to-end test.

## 4. Collection limits

The experimental variant changes constructor defaults only: vector/matrix limits
131,072 → 524,288, real-number limit 65,536 → 262,144. Invalidation rules are
identical. Programs use the existing generators: QFT with 24 qubits and period
exponent 12, ring QAOA with 20 qubits and one layer (gamma 0.31, beta 0.19), and
Grover searching for the 20-bit zero string.

| Program | Baseline time, s             | Larger-limit time, s         | Baseline/larger peak DD MiB | Baseline/larger maximum RSS KiB | Node collections |
| ------- | ---------------------------- | ---------------------------- | --------------------------- | ------------------------------- | ---------------- |
| QFT     | 0.000641 (0.000623–0.000722) | 0.000636 (0.000628–0.003754) | 0.082 / 0.082               | 54,068 / 48,244                 | 0 / 0            |
| QAOA    | 6.996 (3.427–8.410)          | 5.987 (4.738–6.107)          | 89.908 / 87.531             | 189,776 / 165,276               | 5 / 2            |
| Grover  | 0.903 (0.870–1.317)          | 2.094 (1.702–2.246)          | 8.976 / 36.007              | 61,008 / 99,192                 | 141 / 47         |

QFT has no collection to avoid. QAOA has a 40.8% matrix-vector hit rate under
both limits and zero vector-add hits. The experiment does not show improved
multiplication-cache reuse. Grover's fourfold DD memory growth exceeds the
budget and its time regresses. Matrix-vector hit rates are 8.8% for both QFT
variants and 46.53%/46.60% for Grover; Grover vector-add hits rise from 1.59% to
3.75%. Real-number collection counts match the node counts in the table. Reject
the default change. A two-layer QAOA probe exceeded a 180-second wall limit
under shared load; it is not included in this table.

## 5. Adaptive sampling and replay

All cases use 10,000 shots and seed 42. Weak Grover searches four bits with its
default measurement strength; distillation uses the default level-one generator.
The QFT cases use the semiclassical generator with k measurements and period
exponent k/2. QCO receives the generated program; QIR uses the standard Adaptive
compilation pipeline before timing execution. Compiler/JIT preparation is
outside these execution timings, so the two columns are not an end-to-end
backend ranking.

| Program      | QCO time, ms         | QIR time, ms        | QCO groups / QIR replay paths | QCO / QIR matrix-vector hit rate |
| ------------ | -------------------- | ------------------- | ----------------------------- | -------------------------------- |
| Weak Grover  | 29.39 (20.42–36.62)  | 2,133 (1,798–2,645) | 47 / per-shot                 | 28.3% / 97.1%                    |
| Distillation | 1,430 (1,051–1,559)  | 0.684 (0.681–6.730) | per-shot / 1                  | 99.92% / 28.7%                   |
| QFT k=4      | 0.613 (0.584–15.701) | 0.315 (0.296–3.344) | 4 / 4                         | 63.6% / 75.0%                    |
| QFT k=8      | 0.701 (0.661–15.826) | 0.392 (0.384–1.558) | 16 / 16                       | 79.6% / 90.6%                    |
| QFT k=12     | 0.961 (0.958–0.990)  | 0.775 (0.752–3.936) | 64 / 64                       | 82.6% / 94.3%                    |

No node or real-number collection occurs in these probes. The low QIR hit rate
for distillation accompanies only one path, not lost replay reuse. QIR weak
Grover takes the per-shot path; its high hit rate does not remove native-program
replay work. No QIR execution redesign follows from these data.

Forcing weak Grover through QCO's per-shot path with three performance cores
available gives 2.331 s (2.018–2.619), versus 0.029 s for serial grouped
sampling. The comparison does not justify parallel structured branching.
Distillation measurements in generated helpers are a supported producer of the
fallback; the separate follow-up first tests existing compiler inlining.

## Validation

- Native `release-clang-ipo` build with shared libraries and Python bindings
  passed. A broad native run before the final memo bound passed all 4,337 tests.
- After rebuilding the bounded implementation, all affected native binaries
  passed: DD 234, QCO DD 116, QIR JIT 76, QIR runtime 82, QDMI DDSIM 93 (601
  total). The automatic external serial-grant regression is included.
- Python estimator and expectation tests: 22 passed against the final built
  module, with pytest workers disabled so the build-tree import setup is shared.
- The dense-oracle Y-sign mutation was rejected. The final exhaustive oracle
  also passes with a one-entry memo capacity.
- `uvx nox -s cpp-lint -- 858c9b718b` passed all five changed C++ files. The
  final two cache-bound edits were additionally checked with clang-tidy using
  the same compilation database; no diagnostics concern changed files.
- Repository lint: `uvx nox -s lint` passed after formatting the plans.

The focused review found and resolved unbounded memo growth and strengthened the
automatic serial-grant test. No public API, collection threshold, invalidation
rule, gate cache, or QIR execution change is retained. Windows and hosted CI
were not run for the initial investigation. Performance numbers above remain
tied to the recorded baseline, rather than the later upstream rebase.

## Upstream integration validation

Rebased onto upstream `main` at `49466928bf77d4df7fe8090bfb9792c2e3abddf1`. The
automatic-worker regression checks exact seeded output equality between full and
serial-capped grants, consistent with upstream logical shot streams. Production
Pauli code is unchanged from the measured candidate.

The rebased release build passed, along with all 4,346 native tests and the 22
Python estimator/expectation tests. All five changed C++ files passed
`uvx nox -s cpp-lint -- origin/main`; repository lint also passed. Windows and
hosted CI are separate from these local results.

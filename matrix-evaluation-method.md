🤖 *AI text below* 🤖

# Matrix root-range evaluation method

This evaluation compares the isolated matrix root-range fix with upstream main
`e7b37b7a42291a9816b735ef8ae2e53b0f37b990`. It contains only matrix workloads:
36 ordinary-scale performance processes and 12 wide-Hadamard accuracy controls.
The numerical harnesses, oracles, seeds, fixed repeat counts and paired plans
are unchanged from the earlier matrix evaluation. All binaries and results
are rebuilt and rerun for this exact baseline and candidate source tree.

## Ordinary-scale performance controls

`matrix-composition.plan` contains six cases, three repetitions, and adjacent
upstream/candidate pairs (36 processes). Pair order alternates. Each circuit
uses seed 7301; repeats are fixed for both variants.

| Family | Qubits | Layers | Complete builds per process | Timed gates |
| --- | ---: | ---: | ---: | ---: |
| Dense entangling circuit | 4 | 16 | 16 | 2,816 |
| Dense entangling circuit | 6 | 8 | 4 | 544 |
| Dense entangling circuit | 8 | 4 | 1 | 92 |
| Independent dense two-qubit blocks | 16 | 12 | 8 | 3,840 |
| Independent dense two-qubit blocks | 32 | 12 | 4 | 3,840 |
| Nearest-neighbor diagonal circuit | 32 | 12 | 4 | 3,024 |

Dense and block circuits use seeded RY/RZ gates and CNOT layers. The diagonal
circuit uses seeded RZ and controlled-phase gates. No iteration count is adjusted by variant. The final report includes measured
durations and observed trial ranges.

CPU and wall time include package construction, gate construction, matrix
composition, reference-count maintenance, normal garbage collection, and
destruction of the earlier repeated builds. The operation list is generated
before timing; each repetition starts with a fresh package and retains its
current root. The final package remains live for validation. The last package's
destruction is outside timing for both variants.

Use `cpu_seconds` and `simulation_maxrss_kib` for the performance comparison.
RSS is captured immediately after the timed builds and before any oracle,
statistics traversal, or inverse composition. It covers every repeated build.
Table statistics and `final_nodes` describe the final build. The runner's
`maxrss_kib` and `process_seconds` include validation and must not replace these
simulation measurements. RSS also includes allocations omitted from DD bucket
statistics, including the private matrix-root index.

Correctness is checked after timing and the simulation RSS snapshot:

- Dense cases compare every represented matrix entry and every column against
  a separate dense simulation of the input gate matrices in `long double`.
- Block cases independently build each dense 4x4 block and contract eight
  complete sampled columns against their tensor product. They also compare
  sampled diagonal entries.
- Diagonal cases independently multiply each sampled column's scalar gate
  phases and check the complete column against that basis vector.
- Every case additionally composes the DD conjugate transpose with the DD
  result and checks eight resulting columns against identity columns.

The independent column norm and overlap routines read stored weights and
handle omitted identity levels; they do not call DD multiplication or vector
addition. The inverse check is additional evidence, not the independent oracle.
The contraction residual uses `long double` Gram subtraction, records that
type's precision and a roundoff floor, and is a numerical check rather than a
formal error certificate. The acceptance target is column L2 error 1e-8 plus
that recorded floor. Nonfinite values fail explicitly. The entry error field
records whether its scope is all entries or sampled diagonal entries.

## Wide Hadamard accuracy controls

`wide-matrix-accuracy.plan` contains one upstream/candidate pair at each width
32, 64, 96, 128, 256, and 512 (12 processes). It builds the full all-qubit
Hadamard matrix, checks 21 deterministic paths against the analytic signed
amplitude `2^(-n/2)`, then applies the Hadamards in reverse order and checks
identity entries. It records the original and inverse root weights and node
counts. The maximum relative matrix-entry error and maximum absolute inverse
entry error must both be at most 1e-10 for `check_pass`.

This is an accuracy comparison, not a speed benchmark. Upstream is expected to
fail the numerical control at the tested widths 96, 128, 256 and 512. The plan records the
expected outcome separately; the executable returns zero on a numerical
failure so that a measured baseline error remains distinct from a process
failure. Any baseline or candidate outcome differing from the expectation
needs inspection. Expected upstream accuracy failures are separate from the ordinary-scale
performance controls. Do not infer a speedup from timings of an incorrect
upstream computation.

## Reproduction and provenance

Both libraries use the release-no-MLIR preset and clang 23, with matched
optimization flags. A clean detached baseline builds only `mqt-core-dd`.
Each harness is linked against a frozen library copy; manifests record the
compiler command, source and header hashes, full Git revision and full diff
against the shared baseline. `build.py` refuses to overwrite a binary.

The evaluation ran serially on CPU16, with the plan's 60-second process cap
and a 48-GiB address-space limit. All builds completed before measurement.
Upstream/candidate pairs were adjacent and their order alternated. ASLR used
the host's normal setting. Simulation CPU and RSS exclude numerical validation;
raw process timing and process RSS also include validation and serve as
separate diagnostics.

The public artifacts contain the two fixed plans, all 48 raw result rows,
per-case summaries, numerical verification counts and compact provenance.
The unchanged harness, oracle, build and runner sources are retained in the
local evaluation archive. This branch does not add a benchmark framework to
the product or publish any executable.

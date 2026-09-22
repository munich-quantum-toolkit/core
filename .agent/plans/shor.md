# Dynamic Shor benchmark

Status: in progress. The benchmark is implemented on current main, which
includes its compiler prerequisites from #2546. Fresh native, Python,
documentation, and lint validation remains before handoff.

## Delivered behavior

- Native `Shor` and Python `bench.shor.Shor` implement semiclassical order
  finding with real controlled modular arithmetic. Options are `number`,
  `base=2`, and optional positive `qft_cutoff`; omission selects exact
  arithmetic.
- Odd circuit inputs satisfy `3 <= number <= 2**31 - 1`, with
  `1 < base < number` and a coprime base. For `n = bit_width(number)`,
  generation uses `2*n+3` qubits and returns `2*n` big-endian phase bits. Phase
  integers, denominators, modular products, and continued fractions fit in
  `uint64_t`.
- Shared Fourier arithmetic builds reusable private accumulate, uncompute, and
  in-place multiply helpers. Tables contain repeated-squaring powers and their
  modular inverses. Generation is polynomial and does not compute orders or
  enumerate modular orbits. Exact arithmetic clears workspace; approximate
  Fourier arithmetic can change the sampled success rate.
- Native evaluation verifies continued-fraction candidates with modular
  exponentiation and gcds. Its dedicated result contains optional sorted factors
  and the observed fraction of successful shots. Manifest reference
  `verification/shor_factors` and evaluation JSON report no TVD or Hellinger
  metrics. Registry, strict JSON, CLI, bindings, and generated stubs are wired.
- C++ and Python `factor(number, run, ...)` return one verified factor pair,
  prime status, or exhausted attempts. Classical even/prime/perfect-power
  prechecks precede base 2 and seeded retries; the default limit is 16 attempted
  bases. The callback owns compilation, device selection, shots, and execution
  seeds. Callback errors propagate.
- README and getting-started share an executable callback that factors 21 on
  DDSIM with 64 shots and seed 17. The walkthrough explains an actual phase,
  continued fractions, and gcd recovery. Focused QPE examples now live in
  `docs/benchmarks.md`; the glossary defines the new terms.

## Compiler contract

The benchmark uses the existing QC borrowed-register helper ABI. Calls borrow
complete fixed-size registers and return updated quantum values in argument
order after ordinary results in QCO. Register slots retain qubit identity;
controlled SWAP operations exchange states without moving qubits between slots.
The existing inliner, QC/QCO conversions, target compiler, and DD interpreter
own these contracts. This benchmark adds no compiler or runtime machinery.

## Validation

The acceptance checks are native factor recovery and strict JSON validation,
small modular multipliers on basis states and coherent superpositions, inverse
composition with clean workspace, sampled Shor distributions for 15, 21, and 35,
and a structured 31-bit generation/Adaptive QIR compilation check. Python tests
exercise the factoring callback through DDSIM using Adaptive QIR for 21 and
mapped OpenQASM 3 for 15. The README and walkthrough execute the factoring
example. The largest input's 65-qubit state is not simulated.

Fresh validation on 2026-09-22:

- Optimized native configure/build succeeded. The full CTest suite completed
  3,602 entries without failures, with one expected Slurm integration skip.
- The benchmark and QDMI compilation Python suites passed all 116 tests,
  including verified DDSIM factoring through both payloads.
- Stub generation, whole-changed-file C++ lint, and executable documentation
  including rendered-link checks passed. The walkthrough factors 21 as `(3, 7)`.
- Repository lint passed all hooks except the type checker, which reports 28
  existing diagnostics in unrelated untracked audit scripts. The repository's
  pinned type checker passes with `.agent` excluded. Those scripts and the
  pre-existing QDMI audit edit remain untouched.

## Limits and deferred scope

The 31-bit input bound controls representation size, not feasible simulation
cost. Mapped OpenQASM compilation specializes indexed physical-qubit loops
within the existing 65,536-operation budget. The exporter supports indexed
logical registers and angle tables, but cannot express runtime indexing of
physical qubits. Adaptive QIR retains supported indexed loops. Exact Shor 21
exceeds the OpenQASM expansion budget; the current compiler reproduces that
diagnostic. See the
[unrolling investigation](../audits/shor-payload-unrolling.md).

Dynamic-size helper signatures, register ownership transfers, recursive
factoring, wider integers, and the independent distillation benchmark remain
outside this change.

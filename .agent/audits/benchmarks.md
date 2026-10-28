# Benchmark audit

Status: applied and validated locally. Date: 2026-10-23.

Review baseline: PR 2803 at `f7e864f9422d47150251ff09a35a386c8bea4357`, based on
merged QAOA support.

## Scope and result

The review traced the 16 benchmark families through native validation and
reference evaluation, strict JSON parsing, Python bindings, structured MLIR
generation, serialization, and tests. Supported inputs include large structured
programs and sparse counts. Public APIs, canonical schemas, semantic case IDs,
and circuit definitions remain unchanged.

The implementation shares repeated contracts through existing helpers and the
family registry. It also fixes small-probability evaluation and repeated source
locations in JSON diagnostics. Family-specific semantics remain explicit.

## Numerical evaluation

`Multiplexer::probability` computes target-zero amplitudes from the complemented
control fraction. For `c` controls,
`1 - sum(b_i * 2^-i) = 2^-c + sum((1-b_i) * 2^-i)`. This avoids cancellation
near one and permits one sine formula for both targets. With 64 all-one controls
and target zero, the original formula returned `2.0325535182104675e-52` instead
of `3.9307967343631768e-58`. The endpoint regression checks relative error; an
absolute tolerance could accept that wrong result.

`detail::evaluate` computes squared Hellinger fidelity as
`(sum(sqrt(count_i * ideal_i)))^2 / total_shots`. Delaying normalization avoids
multiplying two small probabilities and uses one square root per outcome.
Positive integer counts cannot make an ideal probability smaller before the
square root. For normalized ideal probabilities, Cauchy-Schwarz bounds the
squared sum by the total shots. Existing shot-overflow validation remains.

The GHZ regression uses two equally sampled even-parity outcomes at the maximum
X-basis width. It requires exactly `2 * double::denorm_min()`. The old product
formula loses this result with binary64 accumulation. This host has a 113-bit
`long double` mantissa and does not reproduce that loss in the original native
code; the earlier audit's binary64 host claim was incorrect.

An external C++ formula probe compared 913 cases with a 150-digit Decimal
reference, including subnormals and maximum shot totals. Native accumulation
matched the rounded reference throughout. Explicit binary64 accumulation
preserved all positive representable results, with maximum absolute error
`7.8e-16`; the old formula lost the GHZ result. This models narrow `long double`
arithmetic, not another operating system's build. Final results below the
`double` range can still round to zero. No runtime speedup was measured.

## Shared contracts and diagnostics

- `registerBenchmarkIO` owns generation, three metadata properties, and two
  strict JSON parsers. Metadata getters bind directly through nanobind. MLIR
  remains lazily imported. Constructors, options, output layouts, and evaluation
  stay explicit because QAOA, Shor, and other families have different contracts.
- `BenchmarkFamilies.inc` supplies internal emitter declarations and public
  typed `generate` declarations, as well as the existing definitions and
  dispatch. Adding a family no longer requires another declaration list.
- `phaseAdditionLoop` reuses `advancePhaseCorrection` for `phase/2 + pi*bit`.
  Floating-point operations and loop shape are preserved; extracting the bit
  before the independent halving does not change the recurrence.
- The existing schema helper supplies shared parameter-object constraints.
  Family-specific fields and validation remain explicit. All 16 serialized
  schemas match merged main byte for byte.
- Multiplexer, W-state, and modular-multiplier fields are parsed before the
  constructor error wrapper. A field error now identifies its source and JSON
  pointer once. Constructor validation remains wrapped, including QPE's phase
  construction. One regression covers the three affected parsers.

## Test scope

Keep the numerical regressions: small sampled circuits cannot exercise those
limits. Keep both static and dynamic BV sampling paths, using one asymmetric
`001` input to cover leading zeros and bit order. This replaces four cases with
two while retaining the distinct circuit methods.

Remove the old GHZ maximum-width positivity check: the exact fidelity regression
exercises that same supported boundary through the shared evaluator.

The removed central Python serialization matrix duplicated family-level
round-trip tests and repeated the family list. Native tests retain both parser
paths. A fresh-process probe verified both Python parser bindings, custom source
forwarding, ValueError translation, and lazy MLIR loading. Generated stubs are
unchanged. Independent circuit oracles, negative inputs, phase-sensitive checks,
wire order, and resource limits remain covered by existing subsystem tests.

## Validation and limits

The local release preset uses LLVM/MLIR 23.1.0 with assertions enabled. Results:
163 native benchmark tests, 53 MLIR benchmark tests, the benchmark CLI CTest,
and 82 Python 3.14 benchmark tests passed. Stub regeneration passed with no
generated changes. Repository lint passed with Python 3.14 for the local hooks.
Whole-file C++ lint passed for every changed C++ file, including an explicit
check of the public generation header and the JSON regression test. The complete
executable documentation build and local-link check passed. The contributor
section was inspected in Chrome, including its rendered layout, console, and
network requests; only the preview server's favicon was missing.

No hardware execution, performance benchmark, or other operating system was
checked. Compiler backends were exercised by the benchmark tests, not
exhaustively audited. Keep distinct structured and unrolled diffusion builders,
family-specific output layouts, and the CLI's exclusive-file publication: their
similar-looking code implements different contracts.

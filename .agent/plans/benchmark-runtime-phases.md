# Benchmark runtime phases

Status: complete.

## Scope and contracts

For issue #2613, QPE, constant QFT addition, modular multiplication, and Shor
compute bounded scalar phases. Parameters, manifests, case IDs, output order,
and evaluators retain their contracts. All-family consumer checks include
WState, MagicStateDistillation, and Shor. General tensor support remains for
independent consumers. Multiplexer reuses the shared geometric phase loop.

- QPE carries unsigned i64 residues and converts to f64 after exact modular
  reduction. Overflow-safe doubling preserves the full uint64 phase domain.
- Iterative QPE computes one highest residue in O(log precision) using 128-bit
  modular multiplication during generation. For `d = 2^s*m` with odd `m`,
  reverse doubling uses parity above power `s` and at most 63 packed wrap bits
  below it. Odd denominators use residue parity directly. Every emitted shift is
  below 64.
- Iterative QPE, semiclassical QFT, and Shor carry one correction
  `c = c/2 +/- pi*bit/2`. The angle remains bounded by pi; a contractive halving
  limits rounding error independently of precision. Feedback takes O(precision)
  work and one P gate per round. Measurements supply the i1 bit directly.
- Constant addition stores exact i1 input bits and a zero carry bit when needed.
  The shared recurrence `theta/2 +/- pi*bit` keeps angles in `[-2*pi, 2*pi]`.
  Independent long-double references allow 3e-15 radians of f64 rounding.
- Modular arithmetic requires `1 <= bits <= 63`, unsigned residues below the
  modulus, a reduced accumulator, and zero overflow/work qubits. Controlled
  translations commute on this domain; subtraction consumes ascending powers.
  Forward/inverse modular addition shares its sequence of diagonal phase sweeps
  and overflow checks. Signed recurrence coefficients avoid per-phase negation.
- Shor stores exact powers and inverses. It computes one base inverse, then
  squares both residues. Its single in-place helper emits shared accumulation
  directly around the controlled swap, avoiding two single-use function calls.
- Single-qubit fusion uses direct P/H identities and H-Z-H conjugation through
  the shared Pauli emitter. U pairs retain their exact phase correction without
  quaternion extraction. Controlled unitary tests cover all seven bases.
- DD accepts supported scalar integer tensors. jeff preserves index-cast widths
  and rejects unsupported widths before native fallback. OpenQASM uses its
  existing table dispatch for integer input tensors.

## Validation scope

Exact QPE residues and long-double angles cover odd/even uint64 boundaries at
precision 1026. Maximum generation checks bound million-bit QPE QC/jeff
payloads, 1024-qubit addition, 63-bit modular multiplication, and 31-bit Shor.
Small exhaustive and coherent arithmetic tests check phases and workspace
cleanup. QC/QCO DD, serialized/reloaded jeff DD, Adaptive QIR bitcode execution,
and supported static Base-QIR/OpenQASM execution retain semantic coverage.
Direct QPE and WState OpenQASM round trips and constant-adder export remain.
Eight-bit QPE and QFT sampling exercise longer feedback histories. Redundant
table-layout, operation-count, duplicate generation and exporter-failure checks
were removed; the cleanup removed 298 lines of benchmark tests.

Current consumer limits also reproduce on upstream: adaptive QPE/Shor require
Adaptive QIR; direct Shor OpenQASM rejects its helper arguments; direct nested
QFT re-import cannot prove register bounds. Static target routes are covered.

## Measurements

DGX Spark, `release-clang-ipo`, Clang/MLIR 23.1, ThinLTO and mold. Baseline
phase-table generators and current generators use the same consumer libraries.
Payload sizes exclude manifests. Full pipeline stages were measured separately
in an external harness; timing medians below use five process runs.

| Case                  | QC bytes, baseline -> current | jeff bytes, baseline -> current |
| --------------------- | ----------------------------: | ------------------------------: |
| QPE 16                |                  2352 -> 2479 |                    6456 -> 7592 |
| QPE 1024              |                 18466 -> 2521 |                   14536 -> 7592 |
| Iterative QPE 1024    |                 17872 -> 1758 |                   13120 -> 5912 |
| Constant adder 16     |                  2752 -> 2863 |                    8144 -> 9224 |
| Constant adder 1024   |                 18885 -> 4867 |                   16224 -> 9400 |
| Modular multiplier 16 |                13193 -> 10023 |                  32752 -> 34416 |
| Modular multiplier 63 |                74124 -> 10102 |                  63184 -> 34416 |
| Shor 31               |              2049352 -> 21228 |                1078120 -> 67536 |

| Case                             | Adaptive QIR compile ms |     DD sample ms |    QIR sample ms |
| -------------------------------- | ----------------------: | ---------------: | ---------------: |
| QPE 8, 4096 shots                |            5.76 -> 6.27 |     4.15 -> 4.06 | 176.65 -> 176.75 |
| Iterative QPE 8, 4096 shots      |            5.14 -> 5.70 | 180.01 -> 160.50 |   72.82 -> 65.30 |
| Constant adder 8, 4096 shots     |            6.20 -> 6.25 |     3.77 -> 3.78 | 103.37 -> 104.07 |
| Modular multiplier 5, 4096 shots |          16.19 -> 15.72 |   12.72 -> 13.67 | 695.21 -> 697.37 |
| Shor 15, 64 shots                |          39.52 -> 39.32 | 439.05 -> 530.43 |   90.55 -> 89.84 |

Scalar arithmetic repeats per shot in adaptive DD simulation. Relative to the
phase-table baseline, iterative QPE sampling is faster; Shor DD sampling is
about 21% slower in this run. Small jeff payloads can grow.

Seven interleaved runs isolate the carried-feedback optimization against
`1222261ee`, using the same compiler libraries for both generated inputs:

| Case                                 |     DD sample ms |  QIR sample ms |
| ------------------------------------ | ---------------: | -------------: |
| Iterative QPE 8, 4096 shots, phi=1/3 | 215.98 -> 159.86 | 72.98 -> 65.10 |
| Semiclassical QFT 16, 4096 shots     | 364.39 -> 151.83 | 72.39 -> 61.83 |
| Shor 15, 64 shots                    | 530.92 -> 528.15 | 89.58 -> 89.36 |

The QFT period exponent is 4. IQPE uses 26% less DD time and 11% less QIR time;
QFT uses 58% less DD time and 15% less QIR time. Shor remains dominated by
modular multiplication. Complete seeded histograms match before/after and
through jeff reload for these cases. Setup arithmetic matches independent exact
references for random phases and numerical boundaries. Specialist probes bound
million-round feedback error below 6e-16 radians and verify fusion identities
against matrices for 10,015 boundary and random angles. Generation timings
include CLI startup and are not speedup claims. Raw harnesses and data remain
outside the repository.

## Gates

- Repository lint passed (`uvx nox -s lint`).
- Native `release-clang-ipo` build passed. Full CTest: 3941 passed and one
  expected `ScQDMIJobSpecificationTest.QueryJobId` skip.
- Whole-file C++ lint passed against `origin/main` with zero findings.

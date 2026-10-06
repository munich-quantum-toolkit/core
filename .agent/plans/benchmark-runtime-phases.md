# Benchmark runtime phases

Status: complete.

## Scope and contracts

Issue #2613 is based on upstream `main` at `a3b5fed6f`. QPE, constant QFT
addition, modular multiplication, and Shor compute bounded scalar phases.
Parameters, manifests, case IDs, output order, and evaluators retain their
contracts. All-family consumer checks include WState, MagicStateDistillation,
and Shor. General tensor support remains for independent consumers. Multiplexer
reuses the shared geometric phase loop.

- QPE carries unsigned i64 residues and converts to f64 after exact modular
  reduction. Overflow-safe doubling preserves the full uint64 phase domain.
- Iterative QPE computes one highest residue in O(log precision) using 128-bit
  modular multiplication during generation. For `d = 2^s*m` with odd `m`,
  reverse doubling uses parity above power `s` and at most 63 packed wrap bits
  below it. Odd denominators use residue parity directly. Every emitted shift is
  below 64.
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
Direct small QPE and WState OpenQASM round trips and constant-adder export
remain. Redundant table-layout, operation-count, duplicate generation and
exporter-failure checks were removed; benchmark test source shrank by 298 lines.

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
| Iterative QPE 1024    |                 17872 -> 1856 |                   13120 -> 6048 |
| Constant adder 16     |                  2752 -> 2863 |                    8144 -> 9224 |
| Constant adder 1024   |                 18885 -> 4867 |                   16224 -> 9400 |
| Modular multiplier 16 |                13193 -> 10023 |                  32752 -> 34416 |
| Modular multiplier 63 |                74124 -> 10102 |                  63184 -> 34416 |
| Shor 31               |              2049352 -> 21289 |                1078120 -> 67160 |

| Case                             | Adaptive QIR compile ms |     DD sample ms |    QIR sample ms |
| -------------------------------- | ----------------------: | ---------------: | ---------------: |
| QPE 8, 4096 shots                |            6.58 -> 7.35 |     4.16 -> 4.05 | 175.70 -> 176.54 |
| Iterative QPE 8, 4096 shots      |           5.49 -> 10.59 | 180.14 -> 223.93 |   72.67 -> 72.81 |
| Constant adder 8, 4096 shots     |            6.99 -> 6.56 |     3.67 -> 3.73 | 103.52 -> 103.64 |
| Modular multiplier 5, 4096 shots |          20.25 -> 14.79 |   12.71 -> 13.02 | 693.30 -> 694.76 |
| Shor 15, 64 shots                |          32.09 -> 29.92 | 436.07 -> 524.05 |   90.78 -> 89.60 |

Scalar arithmetic repeats per shot in adaptive DD simulation. Relative to the
phase-table baseline, iterative QPE and Shor DD sampling are about 24% and 20%
slower in this run; QIR sampling is comparable. Small jeff payloads can grow.
Seven interleaved runs against the initial runtime implementation measured IQPE
8, phase 1/3, 4096 shots at 254.33 -> 214.85 ms DD sampling (15.5% less), 7992
-> 6144 jeff bytes, and 72.73 -> 72.83 ms QIR sampling. Complete seeded
histograms matched before/after and through jeff reload. Setup arithmetic also
matched independent exact references for random phases and numerical boundaries.
Generation timings include CLI startup and are not speedup claims. Raw harnesses
and data remain outside the repository.

## Gates

- Repository lint passed (`uvx nox -s lint`).
- Native `release-clang-ipo` build passed. Full CTest: 3940 passed and one
  expected `ScQDMIJobSpecificationTest.QueryJobId` skip.
- Whole-file C++ lint passed against `a3b5fed6f`; the final Multiplexer reuse
  also passed a separate whole-file check.
- No remote publication performed.

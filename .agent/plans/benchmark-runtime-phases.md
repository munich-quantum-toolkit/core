# Benchmark runtime phases

Status: complete.

## Outcome

Issue #2613 is implemented on upstream `main` at `a3b5fed6f`. QPE, constant QFT
addition, modular multiplication, and Shor no longer store derived
floating-point phase tables. The all-family checks include WState,
MagicStateDistillation, and Shor. Benchmark parameters, manifests, case IDs,
output order, and evaluators retain their contracts. General tensor support
remains for independent consumers. The generators live in
`mlir/bench/programs/`, their semantic checks in `mlir/unittests/bench/`, and
the public arithmetic contract in [benchmarks.md](../../docs/benchmarks.md).

## Decisions and numerical limits

- QPE carries unsigned i64 residues. Its overflow-safe doubling compares the
  residue with its complement before selecting the wrapped result. Conversion to
  f64 happens only after exact modular reduction.
- Iterative QPE initializes the highest residue in linear time. For denominator
  `d = 2^s*m` with odd `m`, higher powers reverse by halving and adding
  `ceil(d/2)` when `(residue >> s)` is odd. One i64 packs the first `s` wrap
  decisions; both operands of every select keep shift counts below 64.
- Constant addition retains exact i1 input bits, including a zero carry bit. The
  shared emitter computes `theta/2 + pi*bit` in f64, keeping angles in
  `[0, 2*pi]`. Independent long-double references allow 3e-15 radians of
  rounding difference, including 1024-bit sparse and all-ones addends.
- Modular multiplication carries unsigned doubled residues below a modulus
  smaller than 2^63. Controlled translations commute on the supported
  clean-workspace domain, so subtraction consumes the same ascending powers.
- Shor retains only exact powers and inverses, avoiding runtime Euclid and cubic
  angle storage. Scalar helper arguments use the shared modular emitter.
- DD constant lookup accepts supported scalar integer tensors. jeff preserves
  index-cast widths and rejects unsupported widths before native fallback.
  OpenQASM reuses its existing dispatch lookup for integer input tensors.

## Evidence

Regression checks cover exact QPE residues against 128-bit modular arithmetic,
long-double phases, million-bit QPE generation, 1024-qubit addition, 63-bit
modular generation, and maximum-size Shor generation. Million-bit QPE also
serializes and reloads as jeff with bounded payload size. Small exhaustive and
coherent arithmetic checks preserve relative phases and workspace cleanup.

QC/QCO DD, jeff serialize/reload then DD, Adaptive QIR bitcode execution, and
supported static Base-QIR/OpenQASM target execution are covered. Direct
iterative QPE and small standard QPE OpenQASM round trips, WState round trips,
and direct integer-table export are covered.

Remaining boundaries reproduce on upstream main: measurement-dependent iterative
QPE and Shor do not target static Base QIR; direct Shor OpenQASM rejects its
helper arguments; direct carry-adder OpenQASM re-import cannot prove nested QFT
loop bounds. The same direct-import limit affects standard QPE at precision two
and three; precision one round-trips. Supported static target routes round-trip.

## Measurements

DGX Spark, native `release-clang-ipo`, Clang/MLIR 23.1, ThinLTO and mold.
Baseline generators were rebuilt from `main`; both versions use the same
consumer libraries. Values are medians of five process runs. Generation times
include CLI startup and file writing; they are not speedup claims. Payload sizes
exclude manifests.

| Case                  | QC bytes, before -> after | jeff bytes, before -> after | QC generation ms | jeff generation ms |
| --------------------- | ------------------------: | --------------------------: | ---------------: | -----------------: |
| QPE 16                |              2352 -> 2479 |                6456 -> 7592 |     6.72 -> 3.31 |      10.41 -> 5.82 |
| QPE 1024              |             18466 -> 2521 |               14536 -> 7592 |     6.11 -> 6.53 |      10.15 -> 5.95 |
| Iterative QPE 1024    |             17872 -> 2489 |               13120 -> 7800 |     3.98 -> 6.27 |      8.59 -> 10.74 |
| Constant adder 16     |              2752 -> 2863 |                8144 -> 9224 |     3.88 -> 6.56 |      6.52 -> 11.21 |
| Constant adder 1024   |             18885 -> 4867 |               16224 -> 9400 |     3.11 -> 3.50 |       5.38 -> 6.85 |
| Modular multiplier 16 |            13193 -> 10385 |              32752 -> 38792 |     8.24 -> 5.26 |     13.63 -> 13.71 |
| Modular multiplier 63 |            74124 -> 10464 |              63184 -> 38792 |     8.53 -> 5.37 |     21.29 -> 15.16 |
| Shor 31               |          2049352 -> 23288 |            1078120 -> 74464 |    14.02 -> 8.33 |     40.44 -> 29.66 |

| Small case                       | Adaptive QIR compile ms |     DD sample ms |    QIR sample ms | QIR bitcode bytes |
| -------------------------------- | ----------------------: | ---------------: | ---------------: | ----------------: |
| QPE 8, 4096 shots                |            8.15 -> 7.52 |     4.10 -> 3.94 | 176.24 -> 176.64 |      3832 -> 3760 |
| Iterative QPE 8, 4096 shots      |            9.30 -> 6.72 | 190.47 -> 253.98 |   72.76 -> 72.94 |      3792 -> 3772 |
| Constant adder 8, 4096 shots     |           13.08 -> 6.43 |     5.39 -> 3.66 | 105.58 -> 103.45 |      3760 -> 3736 |
| Modular multiplier 5, 4096 shots |          13.20 -> 19.66 |   12.89 -> 12.59 | 694.89 -> 692.31 |      5228 -> 4976 |
| Shor 15, 64 shots                |          32.71 -> 27.36 | 437.08 -> 542.11 |   90.72 -> 90.36 |      9212 -> 6412 |

Iterative DD simulation repeats the classical arithmetic per shot; measured QPE
and Shor sampling costs rise about 33% and 24%. Static DD cases and compiled QIR
sampling stay comparable. Small jeff programs can grow despite large-case
savings. Parsing, QC/QCO/jeff conversion, jeff serialization/reload, QIR bitcode
serialization, and JIT initialization were timed separately in the ad hoc
harness; raw measurements remain outside the repository.

## Validation

- Repository lint passed (`uvx nox -s lint`).
- Native `release-clang-ipo` build passed. Full CTest: 3948 passed, one expected
  `ScQDMIJobSpecificationTest.QueryJobId` skip.
- Whole-file C++ lint passed (`uvx nox -s cpp-lint`) with zero findings.
- No remote publication has been performed.

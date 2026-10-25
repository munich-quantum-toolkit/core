# LLVM JSON migration

Status: complete; local validation passed. Hosted CI remains a separate gate.

## Scope and decisions

Stock LLVM JSON replaces nlohmann/json in benchmark APIs, DD diagnostics, QDMI
configuration, bindings, and the IQM exporter. There is no CoreJSON support
library. LLVM owns generic parsing and value equality. Configuration parsing
retains LLVM’s C numeric-locale requirement for decimal numbers; there is no
separate UTF-8 or finite-number parser policy. Core retains typed parameter,
calibration, and boundary checks. The library does not change the process
locale.

IQM streams finite gate parameters through LLVM APFloat for locale-independent
decimal output. DD statistics stream their fixed schema and finite floating
fields the same way, without a locale requirement or a generic JSON writer. QIR
validation rejects nonfinite angles. Pulse transport and execution scope are
unchanged.

Manifest validation compares LLVM values with a reconstructed benchmark. Typed
integer parameters remain exact; redundant metadata uses LLVM numeric
comparison. Case IDs hash LLVM's sorted compact JSON. Benchmarks are unreleased,
so there is no intermediate encoding migration path.

CoreBench remains static because LLVM Error type IDs and their handlers must
share one image, including on Windows. Checked factories and native benchmark
methods return owned LLVM errors. CoreBench, the benchmark generator, and the
CLI follow LLVM's exception and RTTI policy. Python retains its constructors and
error categories without a benchmark-specific platform translator. See the
[benchmark error contract](benchmark-owned-errors.md).

The CLI retains sibling-file creation and hard-link publication. LLVM's unique
file helper substitutes percent characters throughout the path, breaking the
supported literal-percent directory case. Errors return LLVM values without
weakening overwrite refusal or cleanup.

## Evidence and validation

Against `89232f06ce5a5b200ec09081d2d57eb8c41dfada`, nine alternating warmed
public-API pairs showed changes between -2.3% and +2.9%; four compiler workloads
were within 1.2%. These shared-host measurements establish no material speedup.
GHZ and QAOA emitted identical LLVM IR and semantically equal IQM JSON. In the
larger QAOA case, target compilation and QIR lowering each took about 53-55 ms;
JSON export took 2.57 ms. Allocation traffic rose slightly, while peak tracked
live memory and memory retained at return were essentially unchanged. Process
peak RSS is a separate measure and does not establish a memory saving.

Local validation passed 4,089 native tests, 2,036 Python tests, full
changed-file C++ lint, repository lint, executable documentation and local link
checks, and an installed CMake consumer without CoreJSON. The trimmed benchmark
section was inspected in Chrome. One specialist Ponytail review found a
numeric-spelling test that no longer exercised a floating token; the test now
supplies that token directly. Mutants for locale-sensitive IQM output, rounded
large parameters, and a zero-allocation DD usage ratio each fail their targeted
regression. DD reports also match 12 independently parsed pre-change reports;
the comma-locale regression fails against the previous installed library.

External compiler I/O and QDMI exception boundaries remain. This is not a
repository-wide exception-free migration.

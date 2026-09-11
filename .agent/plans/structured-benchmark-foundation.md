# Complete the structured-benchmark foundation

Status: historical implementation record.

The two family lists were later replaced by
[one family catalog](centralize-benchmark-families.md).

## Goal and scope

MQT Core needs structured quantum workloads whose inputs, expected outputs, and
generated programs describe the same exact instance. After this work, C++,
Python, and `mqt-core-bench` users can configure GHZ, Grover, QPE,
Bernstein--Vazirani (BV), and QFT with family-specific parameters. Each instance
has an analytic reference, a canonical JSON instance specification, a
self-checking manifest, a stable case ID, and a structured QC or `jeff` program.

The foundation is also the extension path for Daniel Haag's remaining structured
programs. It keeps his original commits in the branch history and does not copy
simulator support from the pull-request stack that starts at
[#2077](https://github.com/munich-quantum-toolkit/core/pull/2077).

## Constraints

- The current `jeff-mlir` pin already lowers both BV methods and both QFT
  methods. Evidence: focused generation produced nonempty `jeff` files for all
  four cases. The draft pull request #44 is not needed.

- Loading MLIR objects from two Python extensions creates incompatible
  process-local MLIR type IDs. Evidence: the semantic extension passes canonical
  instance specification JSON to the existing `mqt.core.mlir` extension, so one
  extension owns every returned `QCProgram`.

- Exact QPE powers stay finite above 1,024 bits when rational turns are reduced
  before conversion to `double`. Evidence: the precision-1,025 emitter test
  keeps the module below 150 operations.

- The stack that starts at #2077 already owns structured DD execution. The
  benchmark branch therefore owns instances, emission, and references only.

- The combined DD check exposed a shared QFT ordering error in both methods.
  Evidence: the old emitters gave total variation distance above 0.28 for QFT
  `(3, 1)`; descending physical targets and matching feed-forward reduced it
  below 0.05 for both methods.

- The full Python matrix reaches all 570 or 581 tests per version, but six
  unrelated QDMI tests fail because the test installs do not register the
  built-in `mqt.sc.default` and `mqt.sc.iqm.garnet` devices. The focused
  benchmark and CLI tests pass for Python 3.11 through 3.14 with minimum direct
  dependencies.

## Decisions

- Use the singular public names `mqt::bench`, `mqt.core.bench`,
  `mqt-core/bench/`, and `MQT::CoreBench`. Rationale: the CLI and component name
  use `bench`, and the old public API is unreleased.

- Ship typed GHZ, Grover, QPE, BV, and QFT families. Standard and dynamic or
  iterative circuit forms are methods of one semantic family. Rationale: method
  changes must not duplicate the instance reference.

- Keep one private semantic registry and one private MLIR registry. Rationale:
  generic JSON dispatch needs a fixed ID table, while public C++ and Python
  remain typed and do not expose a base class, variant, or option map.

- Install only `MQT::CoreBench`. Expose typed generation through the
  source-build target `MQT::CoreBenchGenerate`. Rationale: the wider MLIR C++
  package is not installed yet.

- Qubit allocation establishes the initial zero state. Emitters reset only
  qubits that they reuse after measurement. Rationale: allocation-adjacent
  resets add work without changing the program.

- Mirror QPE inverse-QFT operands and QFT result indices instead of emitting
  final swaps. Rationale: result mapping is cheaper than quantum swaps.

- Never overwrite CLI outputs. Rationale: case-ID paths are stable; atomic
  no-clobber publication and a manifest-last completion marker are enough.

- Keep DD execution changes above #2077. Rationale: the simulator owns execution
  and returned-register sampling.

- Do not add a benchmark option that expands registers into separate scalar
  allocations. Rationale: storage representation does not change the benchmark
  instance or reference, indexed loops require an indexable value, and scalar
  expansion would make large structured programs linear in the qubit count. A
  reusable compiler transformation is the correct owner if a concrete consumer
  needs this representation.

- Call the strict JSON configuration a benchmark instance specification and
  reserve benchmark instance for a resolved typed value. Rationale: the JSON
  specifies the configuration used to construct an instance; it does not request
  generation or execution.

## Outcome and validation

Typed semantic models and structured emitters cover GHZ, Grover, QPE, BV, and
QFT. Analytic references, canonical instance specifications, manifests, and
stable case IDs describe the same instance. The separate DD integration checks
each family against those references.

The release/native suites, installed C++ consumer, wheel/launcher checks,
focused Python minimums, and executable documentation passed. The full Python
matrix did not pass because of unrelated provider-registration failures. Family
metadata was later centralized in the linked catalog record.

## Code and ownership

Public C++ types live under `include/mqt-core/bench/`; their implementations
live under `src/bench/`. These files own option validation, resolved defaults,
logical output names, analytic probabilities, count evaluation, canonical JSON
instance specifications, manifests, and case IDs. `src/bench/JSON.cpp` contains
the private semantic registry: one entry per ID with a definition version,
instance specification schema callback, and evaluation callback.

Structured emitters live in `mlir/bench/programs/`. Each benchmark has one
self-contained source file. `QFT.cpp` and `QPE.cpp` each keep their phase-loop
rules with the bit order that uses them. `mlir/bench/Generate.cpp` contains the
private MLIR generation registry and the typed `generate(...)` overloads
declared in `mlir/include/mlir/bench/Generate.h`.

`mlir/bench/MQTCoreBench.cpp` implements the `list`, `describe`, `generate`, and
`evaluate` commands. Generation stages both outputs, atomically publishes
without replacing existing paths, and publishes the manifest last.

Python types live in `bindings/bench/register_bench.cpp`. Their `generate()`
methods pass canonical instance specification JSON to `_generate_benchmark` in
`bindings/mlir/register_mlir.cpp`. This boundary keeps MLIR objects in the one
extension that defines them. Generated declarations live in
`python/mqt/core/bench.pyi`.

All references use big-endian strings for the returned `result` register. BV is
a point distribution at the hidden bitstring. QFT with period exponent `k`
assigns probability `2^-k` to outcomes whose final `n-k` bits are zero. GHZ,
Grover, and QPE retain their analytic state, search, and phase-estimation
references.

## Acceptance

All five families must round-trip through typed C++, strict instance
specification JSON, Python, and the CLI with one case ID per resolved semantic
instance. BV `101` must be deterministic in both methods. QFT `(3, 1)` must have
peaks `000` and `100`, and QFT `(4, 2)` must have four quarter-probability
peaks.

Emitter tests must show that allocation-adjacent resets are absent and reuse
resets remain. Standard QPE and QFT must contain no `qc.swap`. BV must use
structured secret lookup, direct big-endian result indexing, and the specified
static or dynamic resources. Large QPE and QFT programs must stay compact and
finite. Every method must produce valid QC and `jeff` with the existing
dependency pin.

The CLI must reject malformed instance specifications, unknown IDs, invalid
formats, and any output collision without changing an existing file. The
executed notebook must discover all five IDs and complete its CLI round trip.
Python must import only `mqt.core.bench` and generated programs must support
`.ir` and `.to_qco()`.

The combined #2077 stack must sample small BV and QFT programs according to the
same references. Those simulator-dependent test commits must remain outside this
feature branch.

## Interfaces

The installed semantic target is `MQT::CoreBench`. Public family and option
types use namespace `mqt::bench` and C++20 standard-library types. nlohmann JSON
and the compact SHA-256 implementation remain private.

The source build exposes `MQT::CoreBenchGenerate` and these typed overloads:

    std::optional<mlir::QCProgram> generate(const BV&);
    std::optional<mlir::QCProgram> generate(const GHZ&);
    std::optional<mlir::QCProgram> generate(const Grover&);
    std::optional<mlir::QCProgram> generate(const QFT&);
    std::optional<mlir::QCProgram> generate(const QPE&);

It also has one generic instance specification overload for the CLI and Python
bridge. No public base class, plugin interface, all-family variant, generic
option map, size-only callback, compatibility alias, or new dependency is part
of this foundation.

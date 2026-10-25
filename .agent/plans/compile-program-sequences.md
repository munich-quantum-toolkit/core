# Compile homogeneous program sequences

Status: complete; local validation passed.

## Outcome and scope

`compile_program` accepts homogeneous `collections.abc.Sequence` inputs and
returns an ordered list, including for empty and singleton batches. Scalar
calls, native pipelines, and submission APIs retain their behavior. The binding
and signature source are `bindings/mlir/register_mlir.cpp` and
`bindings/patterns.txt`; `python/mqt/core/mlir.pyi` is generated with
`nox -s stubs`.

## Contracts and decisions

- Accept generic sequences, preserve order and scalar precedence, and apply
  scalar output selection and ownership to each element. Source strings or paths
  can infer different formats from their contents.
- Support one Python input kind, including subclasses. Iterators, nested
  sequences, mixed kinds, and binary buffers are outside the supported subset.
- Snapshot references and validate kinds before consuming typed inputs.
  Compilation failure returns no results and can consume its processed prefix,
  including the failing input. Later inputs remain untouched; `copy=True`
  preserves all inputs.
- Compile serially against one target snapshot. An empty sequence skips target
  resolution. Keep existing program and context ownership.
- Use nanobind handles and `list_builder` to borrow tuple-owned inputs and
  transfer results into a preallocated list. Native scalars bypass the Python
  sequence ABC. Classify the first supported family once; exact Python types
  share that classification and differing subclasses use the canonical base
  checks. Input families are determined by their classes. Keep the owning tuple,
  scalar precedence, and lazy Qiskit integration. Initialize stable Python types
  once per module and cache QuantumCircuit only after it becomes available.
  Captured state owns these types without retaining sys.modules. Borrowed
  entry-point handles remain owned by the Python call; the batch tuple owns its
  elements across callbacks. Imported integration classes retain their identity;
  replacing modules or class definitions at runtime is outside this contract.
- Constrained type variables reject mixed kinds; scalar and sequence overloads
  preserve output types. Qiskit target defaults precede same-family defaults
  because Qiskit's partly untyped `Target` otherwise matches `CompilerTarget` in
  ty. A closed `TypedDict` and `Unpack` share the common keyword arguments
  without accepting unknown keywords. Further shape-TypeVar experiments returned
  `Sequence` instead of `list`, union signatures lost scalar/list precision, and
  a bound union accepted mixed kinds. Keep the 28 precise overloads.

The
[typing specification](https://typing.python.org/en/latest/spec/distributing.html#syntax)
permits current typing features in stubs when their syntax parses on the oldest
supported Python. Thus `TypedDict(closed=True)` needs no runtime backport;
`type Alias = ...` would break the Python 3.11 parser and is excluded. Python
typing cannot exclude strings from `Sequence[str]`; a bare string stays scalar
at runtime, even when a caller annotates it as a sequence.

## Performance evidence and limits

The full-file binding audit compared rebased baseline `1d10fa936e` with the
candidate, both on upstream `c8dcc7f775`, using Clang 23/ThinLTO, Python 3.14.8,
nanobind 3.1, and Qiskit 2.6.0. Seven alternating fresh-process pairs used the
public APIs, five warmed groups per process, and fixed inputs and seeds. Final
pairs shared CPU 8 to reduce migration between the host's different core types.
This is a shared-host comparison, not a general compiler speedup claim.

| Public workload                           | Baseline CPU time | Candidate CPU time |
| ----------------------------------------- | ----------------: | -----------------: |
| 32-site target metadata snapshot          |           6.39 us |            5.79 us |
| GHZ(2), scalar, QC_IMPORT                 |           2.33 us |            2.30 us |
| GHZ(2), eight-element list, QC_IMPORT     |          19.33 us |           18.50 us |
| GHZ(2), eight-element UserList, QC_IMPORT |          20.77 us |           20.04 us |
| Copy and bind 24 efficient-SU2 parameters |          86.33 us |           85.49 us |
| Eight Qiskit efficient-SU2(2) imports     |           4.58 ms |            4.57 ms |
| Eight QFT(10) programs to target OpenQASM |          60.41 ms |           60.45 ms |

Metadata and GHZ improvements occurred in every pair. Parameter binding has a
small, noisy timing gain; fewer allocations support retaining its simpler
snapshot. Qiskit import, QFT compilation, sampling, Jeff views, and module
import show no convincing latency change. Input and output fingerprints matched.

The retained changes and their correctness boundaries are:

- Repeated module/type lookup adds dispatch work. Per-module positive caching
  and exact builtin container checks remove it from common paths. Freezing
  Qiskit's absence was rejected because the late-import regression fails.
- Mapping-to-map-to-map duplicates nodes and keys. A private nanobind dictionary
  snapshot feeds the native parameter map directly, preserving Mapping inputs,
  callback ownership, array-rank checks, and validation before mutation. An
  intermediate items tuple provided little benefit and was discarded.
- ArrayRef-to-vector-to-list creates temporary containers. A typed list builder
  copies directly into owned Python elements; string_view avoids intermediate
  strings. Returning aliases instead fails the metadata copy regression.
- Python wrapper allocation remains visible in metadata access. The nanobind
  pools for QCProgram, Site, and OperationCapability improved all seven isolated
  pairs: roughly 7-8% for metadata and 1.5% for eight QC outputs. Native
  destructors run before wrapper parking, and subclasses are not pooled. The
  three default pools retain at most 50 KiB of raw wrappers and slots, plus
  allocator and instance-index overhead. No other classes are pooled.

A warmed QFT singleton spent 5.2 ms in target passes within an 8.1 ms public
call; loop unrolling and native synthesis accounted for 1.5 and 0.9 ms.
Parameter binding phases were approximately 8 us copying, 79 us binding, and 2
us destroying the result. Emitted QFT input IR contains loops; target OpenQASM
uses physical qubits. A same-source unstripped native profile attributes about
64% of target snapshot cycles to libc allocation/free paths, with another 19% in
Python. These costs do not justify changing native compilation or metadata
ownership.

Memray with Python allocator tracing measured warmed metadata calls at 305 to
262 allocations, 26.6 to 17.4 kB allocation traffic, and 23.2 to 17.1 kB peak
live tracked memory. Parameter binding fell from 1,347 to 1,322 allocations and
249.9 to 248.5 kB traffic. QFT still allocates 82.74 MB with about 856 kB peak
live tracked memory. Post-call and isolated-process peak RSS remain about 131
MB; allocation traffic, live memory, and RSS establish different limits. There
is no material process peak-memory reduction. Repeated construct/consume/delete
and subclass probes retained about 7 kB after either 200 or 2,000 cycles, with
all 2,400 subclass destructors observed.

Dense NumPy arrays and Jeff segment views already transfer owned buffers without
copying. Keep their ownership capsules, required bytes materialization, and
existing GIL boundaries. Blanket GIL release would change concurrent mutation
safety; frozen classes and conversion restrictions would narrow the API. Do not
add payload caching, shared contexts, internal nanobind APIs, or a general cache
framework.

## Validation

The sequence tests retain distinct contracts: generic containers and order,
per-source output inference for text and paths, device compilation and
execution, ownership on success and failure, homogeneity preflight, and Qiskit
target reuse with owning input references. Existing scalar tests cover output
conversions, consumed-program rejection, option forwarding, and input parsing.

Ty LSP probes compare keyword completions, return-type hovers, and diagnostics
before and after edits (zero diagnostics, precise list return types, and common
keyword completions). Positive consumer checks target Python 3.11; negative
probes reject mixed inputs, iterators, nesting, incorrect keyword types, and
unknown keywords. Ruff checks the generated stubs.

A compiled mutation selecting every output from the first source was detected by
the retained mixed-format test. One Ponytail review covered the connected
compiler/sample/submit paths; specialist critique also checked type precedence
and stub compatibility.

After rebasing on upstream `c8dcc7f775`, the optimized editable build and 1,181
compiler, Qiskit, QDMI-compilation, DD, and QIR-runner tests passed. Existing
tests also cover late Qiskit import, UserDict parameter binding, and metadata
copy independence without adding test cases. Compiled frozen-negative-cache and
reference-return mutations fail the corresponding regression checks.

Stub regeneration produces no signature changes; Ruff and Python 3.11 ty probes
pass, with eight expected invalid-call diagnostics. Whole-file C++ lint passes
through cpp-linter directly. The nox wrapper's result-reading step is blocked by
git-bot-feedback 0.3's local mode not writing GITHUB_OUTPUT; no repository
tooling was changed. Repository lint is required before publication. Earlier
executable documentation and Chrome checks cover the unchanged user
documentation. Hosted CI remains separate from these local results.

# OpenQASM classical arrays

Status: complete. Public API timings and allocation profiles at baseline
`41a988bb4f` did not justify another performance change. A redundant QTensor
guard was removed and validated.

## Scope and decisions

The frontend supports global, fixed-size, one-dimensional arrays of `bool`,
`int`, `uint`, `float`, and compile-time-quantized `angle` values. It reuses
scalar casts, negative-index normalization, and definite-initialization
analysis. Empty arrays are valid. Runtime indices must be in bounds; the
frontend emits no runtime checks. Views, whole-array copies, multidimensional
arrays, and whole-array input or output remain unsupported.

Typed stack storage and upstream SROA/mem2reg avoid a separate array runtime. DD
execution retains rank-zero slots because promotion can produce them. QIR
converts classical memory and index types together in its final stage; checking
all operand types earlier would reject valid array accesses. Base validates
measurement ordering before its restricted instruction subset. Only Base needs
conversion-time canonicalization for that validation; Adaptive uses the existing
shared cleanup pipeline.

OpenQASM, jeff, and Qiskit exports require array storage to disappear. Qiskit
export preserves its existing scalar-only folding contract, so callers must run
`cleanup()` explicitly before exporting promotable arrays. Runtime storage stays
supported by DD execution and Adaptive QIR, subject to their capabilities.

## Performance decisions

DD sampling branches keep distinct storage identities for caller-visible aliases
but share array buffers until a write. Constants remain in immutable IR instead
of each branch snapshot; runtime results and arguments remain in the
environment. Constant type validation still runs when the operation executes.

Complete array initializers establish an immutable initialization fact only
after every initializer has been checked. Partially initialized arrays retain
the existing branch-sensitive bitmaps. Repeated writes to initialized elements
do not detach these bitmaps.

Run upstream promotion only when allocation interfaces are present. Keep its
leading canonicalizer: constant index normalization must fold before SROA.
LLVM/MLIR 23.1 limits memref destructuring to 16 elements. Document that
boundary instead of adding affine scalar replacement, whose repeated scans made
a 10,000-element initializer exceed 30 seconds in a diagnostic experiment.

Do not gate promotion on an allocation's current slot list. Canonicalization can
turn a constant-sized dynamic memref into static, promotable storage. Running
SROA/mem2reg before that fold leaves the allocation and its accesses.

QTensor branch analysis uses main's existing `hasOneUse()` check before reading
the current tensor's user. A separate initial `use_empty()` check is redundant,
including during singleton-loop folding.

## Performance evidence

Seven alternating process pairs compared the implementation with `17a74e3d7d` on
a DGX Spark, using Clang 23.1.4, LLVM/MLIR 23.1.2, and Release with ThinLTO.
Each process ran three samples; timing medians exclude the first. CPU affinity
was fixed, but cores were not reserved. These are workload-specific diagnostics.

- Sampling 4,096 shots with 12 measurement decisions and a sparse 90,000-element
  array: 172.3 → 18.7 ms; peak process RSS: 32.8 → 24.1 MiB.
- The same sampling setup with a fully initialized 10,000-element table: 51.5 →
  21.8 ms; peak process RSS: 33.3 → 28.9 MiB.
- Semantic analysis of 50,000 dynamic reads from a 90,000-element array: 48.2 →
  18.3 ms. Parsing is outside the timer.
- Cleanup of 20,000 array-free RX/CX gates: 327.0 → 292.5 ms. Copying is outside
  the timer. Small-array timings were unchanged within observed variation.

Seeded histograms and cleaned IR matched the baseline. RSS includes process
setup; semantic-analysis memory did not improve. The later main changes in
`7862345f13` concern fan-out benchmarks and Qiskit bindings, outside this
harness.

At `41a988bb4f`, a further experiment measured the public C++ path from OpenQASM
import through QC/QCO conversion, optional cleanup, and 4,096-shot DD sampling.
Inputs came from the guide and compiler regressions; the existing literal-table
test was also scaled to 1,024 and 10,000 entries. These larger inputs test
scaling, not the frequency of such tables in user workloads.

Seven alternating process pairs tested top-down traversal in the first
canonicalizer. End-to-end medians were 20.08 → 20.10 ms at 1,024 entries and
164.91 → 166.48 ms at 10,000 entries with cleanup. Small examples showed no
consistent gain. The change was rejected. Userspace CPU profiles placed the
large-table cost mainly in generic cleanup; the guide example spent most of its
time in generic DD sampling and compiler-context setup.

A separate heap profile attributed 222,398 allocations and 138.7 MiB of
cumulative allocation volume to large-table cleanup, with 5.2 MiB live at its
end. The traversal change did not reduce that volume. These instrumented heap
figures are distinct from uninstrumented process peak RSS. Measurements used the
same Release toolchain and CPU affinity as above, without reserved cores.
Avoiding copies on same-value stores was not pursued: the selected programs did
not establish a useful workload for the extra store-path check.

## Validation

- Release build and CTest: 4,027 passed; one existing job-ID test skipped. This
  includes `CleansUpNestedSingletonLoops` after removing the duplicate guard.
- At `41a988bb4f`, Python compiler, loops, integer interchange, parameter
  binding, Qiskit translation, DD, and QIR-Runner suites: 978 passed on Python
  3.14. These suites were not rerun for the redundant-guard deletion.
- Export regressions reimport OpenQASM, jeff, and Qiskit and check the resulting
  value with DD sampling. Existing tests retain bounds, empty-array, angle,
  parameter, allocation, and QIR capability coverage.
- At `41a988bb4f`, the OpenQASM guide's array example, full documentation build,
  and local link checks passed. Chrome inspection covered the rendered page,
  console, network, and screenshot; only the preview server's default favicon
  request returned 404. The guide is unchanged.
- Whole-file C++ lint passes for all 19 changed C++ source and test files.
  Repository lint and diff-whitespace checks pass.

An isolated mutation that shared the mutable array handle across branches failed
`BranchesIsolateMemRefWritesAndPreserveAliases`; the baseline passed. This
confirms that the existing test detects loss of branch isolation. No mutation
was applied to production sources.

Hosted CI is separate from these local checks.

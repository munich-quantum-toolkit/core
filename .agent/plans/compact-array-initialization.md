# Compact array initialization

Status: complete.

## Scope and decisions

OpenQASM and jeff numeric literals above 16 elements use an `arith.constant`
dense tensor and `bufferization.materialize_in_destination` into existing
mutable storage. DD, QIR, jeff, and OpenQASM consumers preserve this compact
representation. Small arrays retain scalar initialization for LLVM/MLIR 23.1
memory promotion. Runtime expressions and unsupported attribute conversions
retain scalar emission and evaluation order.

Initialization writes whole local storage. DD reuses uniquely owned buffers and
replaces shared branch buffers without copying overwritten values; aliases
retain storage identity. OpenQASM export requires initialization to precede
every use of an entry-block allocation and emits a declaration initializer,
which preserves definite initialization on reimport.

Equal literals share immutable data. The jeff importer rematerializes numeric
literals and zero arrays when updates or mutating control flow need independent
storage; live computed arrays retain their existing restriction. Export uses
jeff's array constant type conversion to flatten and widen whole-array
initializers directly, without an intermediate dense attribute. MLIR region
dead-code elimination removes unused zero placeholders using jeff's
memory-effect interfaces.

Existing MLIR facilities suffice; no custom initialization operation, pooling
pass, general constant evaluator, or broader ownership analysis is needed.

## Validation

`CompilerPipelineTest.ConstantArrayInitializationPreservesValues` checks narrow
integers, row-major ordering, signed zero, and independent mutable arrays across
branches through DD, jeff serialization, OpenQASM reimport, and native QIR. The
regression detected shared-literal storage defects before their fix; its
weighted checksum detects reordered elements that an unweighted sum misses. The
existing branch/alias test also covers initialization after measurement;
removing the buffer-sharing guard makes its outcome check fail.

Historical measurements of `1c214f629d` against `04de176835`, using the same
Release toolchain and seven alternating process pairs, reduced the imported IR
for 16,384 varying literals from 65,568 operations to 34. Median QIR-path CPU
time fell from 543 to 16 ms and allocation volume from 975 to 29 MB. These are
workload-specific results on a shared heterogeneous host. Literal payload and
mutable storage still scale with element count. Raw evidence stays outside the
repository.

Follow-up measurements against `d88f5b5820` confirm one fewer 128 KiB buffer
allocation in the 16,384-element DD probe. Peak live heap remains 16.53 MB.
Fifteen alternating pairs on one core across nine public routes give median
CPU-time ratios of 0.994–1.011; no runtime speedup is established. The jeff pin
includes upstream PRs #96 and #97, replacing the local normalization rewrite and
zero-placeholder deletion loop.

Local validation on main `eb8f7db150` with jeff `ab4d73a590` passes: 4,082
native tests, 333 Python MLIR tests, and 14 existing OpenQASM programs through
cleanup, serialized jeff round-trip, and sampling. Five existing round-trip
tests caught redundant copies after CSE merged zero arrays; rematerialization
fixes them. The negative control-flow test uses a computed array to retain its
ownership contract. Full-file C++ lint and repository lint pass; the nox C++
wrapper's local reporting error is bypassed by running the same cpp-linter
command directly. Hosted CI and cross-platform validation remain separate.

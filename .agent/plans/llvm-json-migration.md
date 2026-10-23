# LLVM JSON migration

Status: complete; local validation passed with the limitations below.

## Goal and scope

LLVM JSON values and streams replace nlohmann/json in production, tests,
bindings, and build configuration. `support/JSON.hpp` owns strict parsing and
locale-independent serialization. Physical IQM semantics, Python exception
categories, and QDMI C status boundaries remain unchanged. Pulse and execution
scope are unchanged.

Consumers are benchmark specifications/manifests/evaluation, DD statistics, QDMI
registry and SC device configuration, Python session configuration, and the IQM
exporter. The IQM exporter streams instructions directly. Installed consumers
receive LLVM include directories through `MQT::CoreJSON`.

## Decisions

A small Core parser owns strict grammar, numeric conversion, and duplicate-key
checks while reusing LLVM string decoding and JSON values. LLVM 23.1's parser
uses locale-sensitive `strtod`, accepts malformed numeric spellings, and
overwrites duplicate keys. Remove the corresponding Core checks only when the
supported LLVM parser enforces these contracts. Nesting is limited to 256
containers to bound recursive LLVM value destruction and serialization.

Floating-point input uses `StringRef::getAsDouble`; output uses `std::to_chars`
through `json::OStream::rawValue`. These conversions are locale-independent.
Floating-point `std::from_chars` is not available across the supported macOS
deployment range, so parsing uses LLVM on every platform. Non-finite output is
`null`, preserving DD statistics behavior; non-finite input is rejected.

Affected fallible C++ JSON APIs return `llvm::Expected`. Python APIs retain
their values and exception categories. Existing non-JSON throwing operations
keep a contained boundary; this change does not import the broad native API
migration from PR #2545.

Benchmark manifests use schema version 2 and case IDs use the domain
`mqt-core:benchmark-case:v2`. Standard floating-point formatting can change
spelling while preserving the double value, so version 1 manifests require
regeneration as described in `docs/benchmarks.md`. Instance specifications and
counts retain schema version 1. No legacy floating-point formatter is retained.

Allocation failures follow LLVM policy: recoverable out-of-memory handling is
not guaranteed. Retain Core exception cleanup where removing it leaks
allocations during unwinding, and RTTI for public polymorphic DD statistics.
Ordinary JSON validation uses explicit errors. CoreBench and SC configuration
disable RTTI when LLVM does; CoreJSON follows the same rule. Exception handling
remains enabled where existing native calls and stream boundaries need it.

CoreBench must expose LLVM error type IDs across shared-library boundaries;
hiding its LLVM archive symbols breaks typed error handling. QDMI keeps its LLVM
symbols private behind the C ABI. C++ translation units using public LLVM APIs
must match the SDK's RTTI setting, while DD-only consumers retain RTTI.

## Validation and limits

The optimized native suite passed 4,100 tests; Python passed 2,017 tests.
Installed CMake consumers, shared-library typed errors, stub generation,
whole-file C++ lint, and repository lint passed. Executable documentation and
local links passed; rendered documentation was inspected. External linkcheck
failed on two unchanged QIR specification anchors. Hosted CI, Windows, and macOS
execution remain unverified.

`test/support/test_json.cpp` covers grammar, Unicode, duplicates, depth, numeric
limits, locale-independent output, and round trips. Deliberately broken locale
formatting, duplicate detection, and depth checking each failed the targeted
regression test. An independent sanitizer probe round-tripped 20,000 finite
doubles in each of the C and German numeric locales. Throwing stream buffers are
covered by SC configuration tests.

Against base `ecc3a2c7cf488321bf48b226bb9cf3ad4fee72de`, all 16 benchmark
schemas were equal after independent JSON parsing, 757 SC configuration
mutations preserved acceptance/rejection, and four public compiler workloads
produced equal parsed IQM payloads. Existing phase and source-slot checks
remain. An independent review found and validated the shared error visibility,
exception-cleanup, and DD RTTI constraints above.

Paired public-entry measurements on a shared host showed mixed compile-time
changes, including a small-program regression. No overall speedup is claimed.
Profiling separated cumulative allocations, tracked live memory, and process
peak RSS. Mapping and native synthesis consumed two thirds of the measured
target pipeline for the deeper workload; other compilation phases were not
isolated. No further performance change had sufficient evidence to retain.

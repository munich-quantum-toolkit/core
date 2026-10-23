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

Decoded bytes, owning strings, integer precision, deterministic case IDs, and
the Python/C error boundaries are required semantics. Duplicate-key rejection,
paired surrogates, finite input numbers, and the depth bound define the explicit
supported subset. Parsing unescaped strings twice was incidental: after the
UTF-8 and control-byte checks, construct an owning string directly. Escaped
strings still use LLVM decoding.

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

CoreBench links statically, like CoreJSON and SC configuration, to keep LLVM
error producers and handlers in the same final image. Separate static LLVM
copies across a shared-library boundary can make even `llvm::toString` abort;
exporting symbols on ELF does not solve Windows DLL identity. QDMI keeps its
LLVM symbols private behind the C ABI. C++ translation units using public LLVM
APIs must match the SDK's RTTI setting, while DD-only consumers retain RTTI.

## Validation and limits

Rebased on main `0e0826b1bf3136aa1d56b943b3fe4d7703f9e547`, the refactor passed
4,091 native tests and 2,021 Python tests after reinstalling the extension. The
rebase preserves main's corrected DD memory estimates and single active-count
traversal. Whole-file C++ lint and repository lint passed. Earlier validation
also covered installed CMake consumers, LLVM error handling, and stub
generation. Executable documentation and local links passed; rendered
documentation was inspected. External linkcheck failed on two unchanged QIR
specification anchors. Hosted CI, Windows, and macOS execution remain
unverified.

`test/support/test_json.cpp` covers grammar, Unicode, duplicates, depth, numeric
limits, locale-independent output, and round trips. Deliberately broken locale
formatting, duplicate detection, and depth checking each failed the targeted
regression test. Borrowed-string and missing-control-check mutations also fail
the ownership and strict-syntax tests. An independent sanitizer probe
round-tripped 20,000 finite doubles in each of the C and German numeric locales.
Throwing stream buffers are covered by SC configuration tests.

Against base `ecc3a2c7cf488321bf48b226bb9cf3ad4fee72de`, all 16 benchmark
schemas were equal after independent JSON parsing, 757 SC configuration
mutations preserved acceptance/rejection, and four public compiler workloads
produced equal parsed IQM payloads. Existing phase and source-slot checks
remain. An independent review found and validated the LLVM error ownership,
exception-cleanup, and DD RTTI constraints above.

Against `769a22d9b092b353e1362409a88ccdc02c5c4f00`, nine alternating pairs
through public C++ APIs showed about 13% lower CPU time for GHZ/QAOA manifest
parsing and 5% lower CPU time for Garnet configuration parsing. Wall times were
affected by shared-host contention. Compiler timings stayed within about 1.5%;
no overall compiler speedup is claimed. Temporary timers in the public CLI
showed that target compilation, QCO-to-QC conversion, and QC-to-QIR lowering
accounted for about 94% of the deeper workload's measured phase CPU time; IQM
export accounted for less than 1%. Profiling separated cumulative allocations,
tracked live memory, and process peak RSS.

Manifest validation uses LLVM value equality and an exact output-width check.
Numeric spelling is incidental for reference versions and output widths; typed
integer parameters retain integer tokens. Decimal widths use parsed-double
semantics, so `3`, `3.0`, and `3e0` are equivalent. Integer widths retain exact
64-bit precision. W-state descriptors and compact generated IR support widths
above 2^53, where LLVM equality alone can equate distinct unsigned integers. The
reader reconstructs the benchmark from parameters, so this check protects
metadata consistency; the supplied output descriptor does not drive execution.
After LLVM equality, the signed-64-bit dimension limit bounds a floating width
by 2^63. Converting it to unsigned and comparing exactly avoids signed overflow
and rejects rounding of the expected width. Generated reference versions use
unsigned values to avoid LLVM's signed/double conversion at the INT64_MAX
boundary. Canonical serialization still owns emitted manifests and case IDs;
evaluation no longer serializes a manifest just to compare it. The registry
constructs serialized metadata only for callers that request it.

Against `f689b8aac8a79a9857b79b4bea7be2df04689eda`, nine alternating public-API
pairs showed 36-37% lower manifest-read CPU time and 29% lower GHZ evaluation
CPU time. Manifest writing and instance resolution controls stayed within 1%.
These are warmed CPU measurements on a shared host, not compiler-wide speedups.
Removing the width guard and restoring serialization comparison each fail their
targeted regression. A specialist reviewed the changed data flow and found no
further issues. The second object-key lookup remains: no measured benefit
justifies changing early duplicate rejection.

Case-ID serialization streams into the existing domain-prefixed string through
LLVM `raw_string_ostream` and `json::OStream`, removing a temporary JSON string.
Against `35b23a162fef5f5ce5446e3e6d3b325a511500e9`, five public API workloads
used three fewer ordinary C++ allocations and 478-479 fewer requested bytes per
call. Nine alternating warmed pairs showed only 0.4-1.6% lower median CPU time;
this does not establish a compiler-wide improvement or reduced peak RSS. Five
manifests were byte-identical and their hashes matched an independent Python
SHA-256 calculation. Restoring the strict width-token rule and removing the
precision guard each failed their targeted regression. A specialist's Ponytail
review found no further issues. Arbitrary capacity tuning and a custom hashing
stream lack evidence and were not added.

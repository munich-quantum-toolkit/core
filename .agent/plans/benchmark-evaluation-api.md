# Benchmark evaluation boundaries

Status: complete; local validation passed. Hosted CI remains a separate gate.

## Contracts and decisions

Raw-count evaluation remains fallible and owns its validated shot total.
Generic, QAOA, and Shor results expose `shots` in C++ and Python. Private JSON
formatters consume evaluator output and a verified manifest; the public
arbitrary-result serializer is removed. No validated-result wrapper is needed.
ModularMultiplier reuses the result's total instead of summing counts again.

The Shor runner borrows a valid, nonempty, nonthrowing callable synchronously
through `llvm::function_ref`. Python still catches callback failures before
returning through native frames and restores the original exception afterward.

Keep `std::map` counts. Three alternating public Python comparisons against
`std::unordered_map` plus sorted pointers showed regressions of roughly 15% for
4,096-outcome QFT and 32% for QAOA, with smaller regressions for sparse and wide
histograms. Shor improved about 7%, but that does not justify changing shared
storage. Native phase probes isolate the added sort cost. Allocation traffic
increased even where peak live memory decreased; retained memory was zero. These
measurements do not establish a whole-program speedup for the retained API
changes.

LLVM StringMap would require custom binding conversion. A sorted vector would
require adapting mapping producers, and std::flat_map requires C++23. None
simplifies the current C++20 and nanobind boundaries enough to justify more
code.

## Validation

The release suite passes 4,096 native tests and 2,042 Python tests. Six native
checks fail when shot propagation is replaced with zero, including all result
types and JSON output. Restored evaluators retain their existing numerical and
input-error checks. Stubs expose exactly the three new `shots` properties.

C++ lint, repository lint, and stub generation pass. Native benchmark IR retains
no exception-handling instructions, and all three compile units use LLVM's
no-EH/no-RTTI policy. Independent review found no remaining semantic issue.
Hosted CI and non-Linux execution remain separate.

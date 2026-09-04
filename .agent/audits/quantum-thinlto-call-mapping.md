# Contract audit: quantum-call target compilation

Status: confirmed architectural finding; runtime scaling remains unresolved.
Baseline: `d126765488e36985adc796363172f9498390345b` (2026-09-04).
Source: `mlir/include/mlir/Dialect/QCO/IR/QCOOps.td`,
`mlir/lib/Dialect/QCO/IR/Operations/CallOp.cpp`,
`mlir/include/mlir/Dialect/QCO/Transforms/Passes.td`,
`mlir/lib/Compiler/TargetCompilation.cpp`,
`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`, and
`mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/TargetSynthesis.cpp`.

## Result

Current MQT Core has a forced-flattening boundary for reusable quantum
functions. QCO supports a verified positional unitary-call ABI, but target
compilation has no call-summary or inlining stage that can carry the call to a
target result.

This is a sound motivation for a ThinLTO experiment. It is not yet evidence of
a runtime or memory improvement over eager inlining.

## Confirmed finding: target compilation cannot retain `qco.call`

`qco.call` implements `UnitaryOpInterface` and has one qubit result for each
qubit input. Its unitary matrix is deliberately unavailable at compile time.
The completed QCO function model in PR #2336 documents that calls expose
positional correspondence but do not perform interprocedural mapping or cache
callee bodies.

The mapper documentation requires qubit-carrying calls to be inlined before
mapping. The canonical target pipeline runs cleanup, decomposition,
optimization, fusion, mapping or placement, native synthesis, and conformance;
it does not add an inliner.

For a restricted native target, target synthesis cannot lower a `qco.call`
because its unitary matrix is unavailable. Independently, final conformance
walks private functions and rejects quantum function inputs that have not become
static target sites. The mapper only has compositional handling for structured
control-flow operations, not a call boundary.

**Benefit.** A summary-aware mapper has a concrete supported-behavior target:
allow eligible unitary calls to participate in physical planning, then
materialize an equivalent target-native recipe late.

**Risk and limit.** The current behavior is an intentional narrow boundary from
the function-model work, not a defect. The initial prototype must preserve that
fallback for unsupported calls and must not alter local wire traversal.

## Analytical balloons

The executed balloons use small deterministic cost models. They establish design
requirements, not MQT mapper performance.

| Balloon | Model | Result | Implication |
| --- | --- | --- | --- |
| Hierarchy scale | six-gate leaf; eight calls per tile; repeated tile calls | At 128 repetitions, 142 modular planning inputs become 6,144 forced-inline gates (43.27x). | Measure pre-materialization work separately from final native output. |
| Layout ABI | endpoint two-qubit call on a three-site line | At 64 calls, a restoring ABI needs 128 routing swaps; layout propagation needs 1. | Summaries require an output layout delta. |
| Variant selection | two output layouts and alternating continuations | At 32 calls, one fixed variant needs 48 swaps; context-aware selection needs 32. | Cache a bounded Pareto set of physical variants, not one compiled body. |

The hierarchy result does not imply that native output can remain compact. The
layout results compare a layout-transforming summary against a deliberately
fixed-layout outlined ABI; a whole-program eager mapper can see the same
continuation context after inlining.

## Related work in this repository

PR #2336, “Add reusable quantum functions to QC and QCO,” merged immediately
before this baseline. Its design explicitly avoids speculative interprocedural
mapping and caches. This audit treats that as the correct logical foundation,
not work to undo.

No overlapping issue or pull request was found by an issue/PR search for
`qco call` beyond PR #2336. The next overlap check should include any future
mapping or interprocedural-compilation work before implementation begins.

## Validation

Source inspection:

```sh
rg -n -C 3 "inlin|CallOp|call" mlir/lib/Compiler \
  mlir/lib/Dialect/QCO/Transforms \
  mlir/include/mlir/Dialect/QCO/Transforms/Passes.td
rg -n "createInlinerPass|InlinerPass|inline" mlir --glob '*.{cpp,h,td}'
```

Analytical balloon command:

```sh
UV_CACHE_DIR=/private/tmp/mqt-thinlto-uv-cache \
  uv run --no-project python ../thinlto_balloon.py
```

The available MLIR package is 22.0.0, while `cmake/SetupMLIR.cmake` requires
23.1 or newer. `cmake --preset release` therefore stopped with the expected
version diagnostic before any MQT target compiled. A compatible build is the
next required check.

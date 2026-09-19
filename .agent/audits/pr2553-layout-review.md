# Review of PR #2553: compiler layouts

Status: historical review of PR #2553 on 2026-09-19 at
`0ba1b2c71acd987a197b82ed158d890c271dbd6b`, against merge base
`b897f04be98edc43cd9cdf53c5e53a55b9b8cc81`. Findings and measurements below
describe that reviewed implementation. The
[implementation record](../plans/compiler-layout-controls.md) contains the
completed changes and their current validation.

The review covers all 37 changed files, their producers and consumers, the
existing routing layout implementation, and the relevant Qiskit and MLIR
contracts. The principal simplifications are one entry-point-owned layout and a
synchronous, single-program compilation API. Results must remain private until
the entire owned pipeline and its postconditions succeed.

## Revised scope

The user selected one layout tied to the program entry point. The existing
`verifyEntryPoint` checks uniqueness among direct siblings only, and
`verifyQuantumAllocations` skips child modules. Thus the old race and
nested-loss reproducers remain valid against this head, but the proposed
implementation should reject multiple program scopes rather than support them
with more state. Apply that restriction at layout-bearing program and
layout-compilation boundaries; unrelated library-only IR needs no redesign.

Attach the retained layout or invalidation marker to the selected entry-point
`func.func`. Existing QC/QCO function conversions update that operation in
place, so they already provide the ownership boundary required for preservation.
Reject module-level, helper-function, and competing nested specifications. The
public entry point survives ordinary symbol DCE; recursive marker propagation
and the root fallback are no longer needed. Raw IR editors retain responsibility
for preserving or explicitly discarding metadata.

The native snapshot does not become another serialized layout specification.
Keep imported provenance and the detached native result distinct; automatic
composition is still outside scope.

## Confirmed findings

### P1: cloned pipelines share mutable tracking state

Locations: `mlir/lib/Compiler/TargetCompilation.cpp:164`,
`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp:93`, and the tracking pass
constructors.

`populateTargetCompilationWithLayoutPipeline` accepts an `OpPassManager`, but
allocates one `shared_ptr<LayoutTracking>` for all its passes. When the manager
is nested on sibling modules, MLIR clones the passes, copying the same pointer.
Their preparation, mapping, and result publication then mutate the same vectors
and flags concurrently. Each child is otherwise a valid program with local,
fixed-size entry-block allocations.

Reproduced with two sibling modules, each containing one allocation, H, and
sink:

```cpp
PassManager pm(program->module().getContext());
MappingResult result;
populateTargetCompilationWithLayoutPipeline(
    pm.nest<ModuleOp>(), environment, result);
auto status = runWithCompilationOptions(pm, program->module(), {});
```

Multithreading was enabled. Four of five tracked runs failed, with either
`input qubit identity was lost during layout preparation` or
`layout compilation did not produce a mapping result`. The ordinary pipeline on
the same input succeeded. This is a shared-state race, not an unsupported
quantum operation. One destination also cannot represent separate results for
multiple programs.

[MLIR's operation-pass contract](https://mlir.llvm.org/docs/PassManagement/#operation-pass)
requires passes to support cloning and prohibits relying on mutable state across
operation invocations. A shared pointer manages lifetime; it does not isolate
per-operation state.

**Disposition:** keep `QCOProgram::compileForTargetWithLayout` and make its pass
wiring internal to one program invocation. Remove the new public result-output
pipeline-population API. Do not implement per-module tracking, result maps, or
locks for an unsupported case. The ordinary top-level Python method was not
observed to suffer this race.

### P2: Qiskit export bypasses the nested-layout loss policy

Location: `bindings/mlir/qiskit/QiskitExport.cpp:2927`.

The exporter checks `mqt.layout_invalidated` and reads `mqt.layout` only on the
root module. The schema accepts these attributes on nested modules, and the
documented discard policy covers the complete module tree.

A verified `QCProgram` containing a root entry point and a child module with
either retained layout metadata or an invalidation marker exports successfully
to Qiskit with `layout is None`. OpenQASM export of the same program correctly
requires explicit discard. The latest fix made the shared loss checks recursive
but missed this independent Qiskit boundary.

**Disposition:** supersede the earlier recursive-export recommendation. Reject
unsupported metadata ownership during verification and have every exporter read
the same entry-point owner. Replace nested-preservation tests with rejection
tests; do not keep both ownership models.

### P2: source register membership does not round-trip

Locations: `bindings/mlir/qiskit/Qiskit2_5.cpp:985` and `:2271`.

There are two related assumptions:

- Import obtains registers from each qubit's private `_register`. Qiskit permits
  a register made from existing loose qubits; those bits have `_register=None`
  even when `initial_layout.get_registers()` contains their register. Import
  silently records `registers=[]` for this ordinary transpiler output.
- Export builds `Layout(initial_dict)` without restoring the layout's register
  collection. Even a standard `QuantumRegister(2, "logical")` round-trip changes
  `initial_layout.get_registers()` from `{QuantumRegister(2, "logical")}` to
  `set()`.

Both were reproduced with Qiskit 2.5.2,
`transpile(..., initial_layout=[1, 0], optimization_level=0)`, and the built PR
adapter. The index maps remained correct; these are metadata preservation
failures, not observed gate-semantics failures.

[Qiskit's public Layout API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.Layout#get_registers)
provides register enumeration separately from its bit map.

**Recommended change:** handle register membership explicitly in the versioned
adapter, using the public register collection where available. Preserve a
deliberate fallback for manually constructed layouts whose registered bits have
no layout register collection. Do not blindly call `add_register` for partial
layouts: it also fills missing bit assignments. Preserve the PR's promised gaps,
or explicitly narrow that part of the contract. No new Core register abstraction
is needed.

### P2: the success-only result contract stops before the caller's pipeline ends

Locations: `mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp:642` and
`mlir/include/mqt/Compiler/TargetCompilation.h:59`.

The result pass publishes into the caller's object before `PassManager::run`
returns. Appending a pass that calls `signalPassFailure()` demonstrates the
problem: the run fails, but a sentinel result
`{allocationSizes=[7], initialLayout=[8], finalLayout=[9]}` becomes
`{[1], [0], [0]}`.

The existing failure tests exercise invalid placement and synthesis failure,
both before publication. They do not establish the broader success-only promise
for a composed pass manager.

**Disposition:** publish only after the synchronous runner, target conformance,
and final QCO linearity verification succeed. Success of the mapping stage or a
subpipeline is insufficient. Remove the destination pointer and publication
pass. Program contents after failure remain unspecified; this does not require
cloning the module or transactional rollback.

## Simplification and optimization opportunities

- **Remove the public result-publication scaffolding.** Prefer one synchronous
  owner over `createLayoutTracking`, `createLayoutResultPass`, a borrowed
  destination inside shared state, and an externally populated result pipeline.
  Keep the existing mapping algorithm and source-index tracking. Approximately
  40–60 net lines appear removable, depending on the chosen private wiring; this
  is a source estimate, not a measured patch.
- **Defer a separate ancilla-scan optimization.** `QubitLayout::fromAttr` scans
  `result.ancillas` for every register slot; `NativeCircuitWriter::setLayout`
  scans it again for every logical input. Both become quadratic for large
  ancillary registers, but the measurements at the requested sizes do not
  establish this as a material compiler bottleneck. Keep the simple
  representation unless profiling the primary or secondary workloads justifies
  an existing LLVM membership container. No cache or helper class is needed.
- **Avoid building throwaway Qiskit qubits.** `setLayout` creates a loose Python
  object for every logical input, then replaces every registered slot with a
  register member. Populate registered slots first and construct loose bits only
  for unfilled slots. This also reduces redundant ancilla checks. The benefit is
  fewer Python allocations; its runtime improvement was not benchmarked.
- **Remove the redundant source-tag width guard in tensor shrinking**, once
  retaining the existing verifier contract. `ShrinkRegisters.cpp:111` repeats
  the invariant owned by `MQTDialect::verifyOperationAttribute`: one source
  index per static allocation slot. Keep the actual metadata remapping and
  external input validation. The valid-IR precondition, rather than defensive
  checks in each rewrite, should be explicit.
- **Consider removing the import-only encode/decode validation round trip.**
  `QiskitImport.cpp:2920` decodes the just-created attribute and discards the
  decoded object; `QCProgram::fromModule` verifies the attached attribute again.
  One owning verifier suffices for correctness. Retain the early check if
  rejecting malformed metadata before translating a large circuit is an
  intentional performance requirement; it is not another independent semantic
  check.

Entry-point ownership and success-only result publication take priority. Fold
small local deletions into those changes; do not make them separate cleanup
projects. Add no generalized layout service, frontend interface, invalidation
observer hierarchy, or custom cache.

## Performance evidence

The target is 50–1,000 qubits, with 100–150 primary, 300 secondary, and 1,000
exploratory. This is an optimization priority, not a new supported-input cap.
The earlier 2,000–16,000-input schema probe is outside that profile and no
longer drives the optimization recommendation.

The replacement probe imports and exports valid circuits with one source
register, one H gate, and identity layout. The physical register width equals
the logical input count. The source register is either an ordinary
`QuantumRegister` or an `AncillaRegister`; the latter stresses ancillary
metadata. It also reparses each imported program's MLIR. Circuit construction,
initial import for export inputs, and text serialization are outside timed
regions.

DGX Spark ARM64, Clang 23 Release with ThinLTO/mold, LLVM/MLIR 23.1.0, CPython
3.14, Qiskit 2.5.2. Medians of 15 pairs after three warmups, alternating
ordinary/ancillary order. Import and parse include context construction and
verification; export includes construction of SDK objects. No implementation was
changed.

| Qubits | Import ordinary / ancillary | Export ordinary / ancillary | Parse ordinary / ancillary |
| -----: | --------------------------: | --------------------------: | -------------------------: |
|     50 |            0.654 / 0.660 ms |            0.283 / 0.301 ms |           0.663 / 0.655 ms |
|    100 |            0.720 / 0.735 ms |            0.376 / 0.380 ms |           0.779 / 0.790 ms |
|    150 |            0.767 / 0.802 ms |            0.433 / 0.450 ms |           0.902 / 0.913 ms |
|    300 |            1.040 / 1.092 ms |            0.704 / 0.758 ms |           1.326 / 1.350 ms |
|  1,000 |            2.283 / 2.548 ms |            1.966 / 2.262 ms |           3.377 / 3.521 ms |

At 100–150 qubits the total measured import and export calls are each below one
millisecond. The ancillary differences are tens of microseconds or less. They
include SDK and metadata differences, so they do not isolate the linear scans or
predict a replacement's speedup. At 1,000 qubits the differences remain below
0.3 ms for each operation. These light-circuit results do not establish native
routing performance: active width, gate count, topology, and search settings
must accompany the implementation's compilation measurements.

## Contracts worth retaining

- `MappingResult` uses actual target site IDs; the internal mapper's `Layout`
  uses dense hardware indices. Imported `QubitLayout` supports partial
  provenance and source groups. Their different domains justify separate simple
  value types; merging them into one generalized abstraction would make
  assumptions less clear.
- Keep the complete-placement and fixed local allocation restrictions. Dynamic
  inputs, partial placement constraints, and automatic composition with imported
  layouts need distinct requirements before implementation.
- Keep initial placement separate from routing permutation, and preserve the
  explicit `-1` partial-map convention. Qiskit's final routing layout is not
  itself the final logical-input map. The current versioned adapter is the right
  place for SDK details.
- Keep source tags through tensor shrinking and removed-input tracking through
  workspace permutations. The unitary regression with an idle input is a useful
  semantic check, not redundant scaffolding.
- Retain explicit layout-loss handling as the chosen API policy. Blanket
  invalidation is conservative; replacing it with per-pass observers would add
  complexity without an established requirement. Apply it to the sole entry
  point, not a tree of independently preserved layouts.
- No new OpenQASM or QIR language construct is introduced. The concrete upstream
  compliance concern here is MLIR pass isolation. Partial SDK layouts are an
  intentional permissive interchange choice, already documented as unsupported
  by some Qiskit convenience methods.

## Validation

The following tests validated the reviewed implementation in the preceding
review, before this plan revision. Only documentation and the new size probes
changed in this follow-up; implementation checks must run again after changes.

- Built the PR head with `release-clang-ipo`: compiler tests, `mqt-cc`, and the
  MLIR, DD, and QDMI bindings.
- All 243 compiler/CLI tests passed, including 15 layout-focused cases.
- All 599 tests in `test/python/test_mlir.py` and
  `test/python/test_mlir_qiskit_translation.py` passed, including 61 selected by
  `-k layout`, using Qiskit 2.5.2.
- The first broad Python run lacked packaged device discovery in the temporary
  binding staging directory. Configuring the built DDSIM and SC test devices
  resolved all seven setup-related failures/errors; no product change was
  needed.
- Separate native probes confirmed sibling-module interference and premature
  result publication. Separate Python probes confirmed nested metadata loss and
  both register-membership failures.
- `git diff --check` passed. No implementation patch, commit, or GitHub review
  was submitted. Hosted checks were not used as evidence for these findings.

## Ponytail review of the revised plan

Lean already. Ship.

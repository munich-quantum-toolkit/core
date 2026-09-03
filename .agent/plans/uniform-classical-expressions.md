# Standardize classical integer expressions

This ExecPlan follows `.agent/PLANS.md` and the repository development and AI
usage policies. It records one implementation task and grants no remote GitHub
authority.

## Purpose and contract

OpenQASM, Qiskit, and jeff should exchange classical computations through the
same QC/QCO integer operations. CBit represents mutable register storage;
standard MLIR integer values and operations represent computation. Remove the
unreleased CBit comparison operation without a compatibility alias. Preserve
snapshot ordering and existing wide register comparisons, and extend jeff
expressions to widths 1 through 64 without adding multiword arithmetic.

## Progress

- [x] Inspected the clean starting worktree, producers, exporters, and backends.
- [x] Remove the CBit comparison operation and migrate producers and consumers.
- [x] Unify typed source expressions and guarded shifts; add selection export.
- [x] Implement bounded jeff integer legalization and array snapshot
      preservation.
- [x] Apply QIR call-boundary and permissive OpenQASM 2 condition fixes.
- [x] Replace shape-dependent tests, run semantic round trips, and update docs.
- [x] Run full relevant tests, stubs, documentation, lint, and C++ lint.

## Discoveries

The initial jeff adapter accepts CBit comparisons at widths 3, 64, and 65 but
rejects the equivalent read plus arithmetic comparison at each width. Its
integer constants support widths 1, 8, 16, 32, and 64, and it lacks integer
casts and selection. The pinned adapter also maps jeff logical right shift to
signed right shift. Source exporters reject some standard arithmetic solely
because it is not rooted in a register read. OpenQASM fixed-width casts
currently only accept matching-width register operands.

## Decisions

The approved design retains whole-register reads/writes and removes all CBit
computation. Frontends build exact-width values; consumers determine signedness.
jeff promotes non-native widths only at its boundary and masks results to retain
source widths. Existing wider register-versus-constant comparisons use a narrow
read/comparison lowering with reads at the snapshot point. Other wider jeff
expressions remain unsupported. Source exports support integer selection using
Boolean or bit-mask expressions, without temporary public registers. Preserve
existing unrelated function and loop restrictions and fail on stale snapshots.

## Context and implementation milestones

The CBit operation definitions and shared decomposition live in
`mlir/include/mlir/Dialect/CBit/IR/CBitOps.td` and
`mlir/lib/Dialect/CBit/IR/CBitOps.cpp`. Replace comparison construction in
`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` and
`bindings/mlir/qiskit/QiskitImport.cpp` with a read, integer constant, and
`arith.cmpi`. Remove dedicated comparison cases in the DD evaluator and QIR Base
backend. Move the existing bit-comparison implementation to the jeff backend,
where it is a lowering rather than an IR concept. Migrate operation tests to
standard comparisons and storage memory effects.

Next, extend the two source exporters to use operand/result types instead of
register ancestry. Signed comparisons use explicit OpenQASM casts and Qiskit
sign-bit XOR biasing. Support zero/sign extension, truncation, and integer
selection. A shared integer-expression helper builds zero-filling shifts by
checking the original distance before narrowing and selecting a safe count and
zero result. Extend OpenQASM semantic analysis and typed expressions to import
the fixed-width scalar operations these exporters produce. Keep default machine
integers at 64 bits and honor source-language promotion rules.

In `mlir/lib/Conversion/QCOToJeff/QCOToJeff.cpp`, legalize integer computation
using native widths, masks, and bounded cast expansion. Convert reads/writes to
bit-array access and updates. Convert selection to structured switch, implement
all comparison predicates, and lower rotations and population count to standard
integer operations. Use existing type-conversion infrastructure for region and
function signatures. In `mlir/lib/Conversion/JeffToQCO/JeffToQCO.cpp`, correct
right-shift semantics and copy an array on update when a prior SSA value remains
observable. Never modify fetched dependencies or the exchange schema.

Finally, reject register arguments/results on direct and indirect calls in
shared QIR result preparation, and remove condition-only OpenQASM 2 version
gates while preserving zero initialization and gate-library differences.

## Validation and acceptance

From the repository root, configure with `cmake --preset release`, then build
with `cmake --build --preset release`. Run the focused CBit, OpenQASM, jeff,
QIR, QC/QCO, and DD CTest tests before `ctest --preset release`. Run
`uv run --no-sync pytest test/python/test_mlir_qiskit_translation.py` and the
other compiler integration tests. Tests must compare observable behavior, not
require an incidental operation graph. Cover widths 1, 3, 8, 9, 32, 64 and wide
constant comparisons at 65 and 301; all predicates; sign boundaries; casts;
wraparound; shifts at zero, width minus one, width, and truncation-wrapping
distances; shared reads; intervening writes; and old jeff array values. Repeat
important cases after cleanup and through serialized jeff and both source
formats. Unsupported wider expressions and CBit calls must diagnose.

Regenerate stubs with `uvx nox -s stubs`, build dialect docs with
`cmake --build --preset release --target mlir-doc`, and build full docs with
`uvx nox --non-interactive -s docs`. Run `uvx nox -s cpp-lint`, then
`uvx nox -s lint`. Record failures separately from unrun checks and report the
production code delta relative to the starting revision.

## Safety and recovery

Preserve unrelated changes. Build artifacts belong under existing build paths.
Do not reset or discard changes; rerun focused commands after repairs. Any
commits must be signed and verified, with no AI co-author attribution. Do not
push or edit GitHub metadata without fresh authorization.

## Implementation log

The comparison operation, CBit comparison/ancestry helpers, and dedicated
DD/QIR/export cases have been removed. Producers now use read/constant/cmpi. A
first compiler build passed, followed by the source-expression and jeff changes.
The semantic interchange suite now passes alongside existing integration tests.

The frontend now tracks explicit integer widths separately from default machine
integers, parses bool/bit/integer casts, and supports typed bitwise expressions.
Arithmetic retains machine-width promotion. Runtime integer arithmetic wraps at
that width, as do explicit narrowing casts: expanding every signed add into i128
bounds assertions prevented otherwise ordinary narrow computations from reaching
the 64-bit backend. Compile-time invalid expressions retain semantic
diagnostics. Document this implementation-defined boundary and test wraparound
directly.

The jeff boundary uses native widths, bounded bit-based casts, masks, and
switches for selection. Integer switch imports must preserve yielded values, not
replace them with input aliases. Shared array updates are identified before
rewriting, because conversion order must not decide whether an old SSA value
needs a copy. Large general expressions remain rejected.

The Python Release build initially referred to a removed uv build-environment
interpreter; configuring through a no-build-isolation editable install repaired
that infrastructure. The release compiler and Python module use separate build
trees. No fetched dependency was edited.

## Outcomes and retrospective

Final validation passed: the complete Python suite has 912 passing tests; CTest
has 3,828 passing tests and one environment-dependent QDMI job-ID test skipped
(3,059 tests carry the MLIR label). The final 60-case comparison rerun also
passed after making both operands computed values. The 140-case interchange
matrix now feeds measurements into register values and shift distances, and
checks cleanup plus serialized jeff and source-format round trips. It includes
zero/full-width rotations, constant narrowing, narrow-unsigned promotion, shared
snapshots, signed boundaries, and wide constant comparisons.

Full documentation, generated dialect documentation, stub generation, general
lint, new-file lint, and C++ lint passed. C++ lint checked changed production
and test files, including the new shared helper; the last affected-file rerun
has zero diagnostics. Stub regeneration left no tracked stub changes. No remote
writes or commits have been made.

The larger expression suite exposed an existing simulator memory-safety defect:
`ClassicalEnv::bindFrom` assigned a map element from a reference into that same
map while insertion could reallocate it. Copying the attribute before insertion
fixes the shared cause. A 1,024-alias regression exercises map growth. Temporary
`NOSTRIP` in the Python binding CMake helper enabled diagnosis and has been
removed; normal symbol stripping is restored. No fetched dependency was changed.

Balanced bit reconstruction avoids the 64-level Qiskit nesting ceiling. Bounded
casts between jeff's native integer widths need more expression nodes, so the
classical Qiskit import/export budget is now 16,384 (parameter limits
unchanged). Rotations and population count share one target-independent
expansion used only by targets that lack these operations. Population count uses
a compact mask-and-add algorithm rather than one source-tree copy per bit;
one-bit intrinsics are identities. Source exporters retain snapshot checks.

The final audit also corrected constant-width truncation for LLVM's checked
APInt constructors and unified narrow-unsigned constant promotion in arithmetic
and comparisons. Floating-point/integer conversions remain outside the common
round-trip subset; jeff has no corresponding cast operation. The old builtin
boundary fixture also used a population count as a floating gate parameter, so
it remains a negative test under that precise name. Integer builtin interchange
is covered by the semantic matrix instead.

The final production delta from the starting revision is +1,827/-1,030 lines
(net +797), including the new shared integer helper. Tests add 589 and remove
397 lines; documentation adds 78 and removes 69 lines. The execution log itself
is excluded from these counts. This is a capability expansion, not a claimed net
line reduction: CBit has one fewer operation and no comparison/ancestry helpers,
while standard exact-width integer expressions now interchange through three
format paths. Backend-only bounded legalization replaces competing IR
representations, and stale/cross-region source snapshots still fail explicitly.

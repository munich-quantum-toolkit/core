# Add value-semantic functions and calls to QCO

This ExecPlan is a living document maintained according to `.agent/PLANS.md`.

## Purpose / Big Picture

QCO needs a direct representation for reusable unitary definitions and a
loss-minimizing convention for ordinary functions that thread qubits through SSA
results. After this change, `qco.call` is an ordinary unitary operation, generic
`func.call` is an explicit wire boundary, and QC/QCO conversion uses one
positional function ABI instead of per-result annotations or deriving
interprocedural correspondence by walking callee bodies.

## Progress

- [x] (2026-09-02 21:36Z) Compared current main, PR #2196, the QCO builder,
  WireIterator, and both conversion passes.
- [x] (2026-09-02 22:28Z) Chose a positional QCO function ABI and removed the
  proposed result annotation.
- [x] (2026-09-02 22:28Z) Added `qco.call` and QCO unitary-function
  verification.
- [x] (2026-09-02 22:28Z) Added callback-complete QCO builder function and call
  APIs.
- [x] (2026-09-02 22:28Z) Made both QC/QCO conversions preserve scalar-qubit
  functions and calls.
- [x] (2026-09-02 22:28Z) Removed generic-call inference and caching from the
  wire and tensor iterators.
- [x] (2026-09-02) Added focused generic-call and QC/QCO round-trip tests.
- [x] (2026-09-02 23:33Z) Added PR #2336 to the existing general compiler launch
      changelog entry.
- [x] (2026-09-02) Ran 1,157 focused QC, QCO, and conversion tests and lint
      before the final rebase.
- [x] (2026-09-02 22:55Z) Applied the independent MLIR/C++ specialist review:
      malformed calls fail safely, call metadata round-trips or fails explicitly
      when QC cannot represent it, and builder/call contracts match across the
      two dialects.
- [x] (2026-09-02 23:25Z) Applied the specialist's final corrections: QC-to-QCO
      rejects duplicate or explicitly returned borrowed qubits before mutation,
      QCO-to-QC rejects attributes it cannot preserve on stripped pass-through
      results, and no redundant nested call verifier remains. The final delta
      review found no remaining actionable issues.
- [x] (2026-09-03 00:17Z) Rebased onto `origin/main` after PR #2337 fixed the
      multiplexer benchmark. The release build, all 3,805 runnable repository
      tests (3,806 registered, one expected skip), and the full lint pass.

## Surprises & Discoveries

- PR #2196 adds 212 builder implementation lines largely because it exposes
  paired start/end state and recomputes qubit and tensor correspondence from
  callee bodies. A positional ABI and `qco.call` make both responsibilities
  local and remove those failure modes.
- Current QC-to-QCO preflight rejects function qubit block arguments before
  dialect conversion starts. Multi-function support therefore requires an
  explicit function conversion path; changing only `func.call` is insufficient.
- MLIR function signature conversion temporarily creates type-conversion casts
  that do not satisfy the final unitary-function signature. Both conversion
  passes therefore hide the marker under a scope guard and restore it after the
  whole conversion, including failure paths.
- Converted QC block arguments must retain the original QC SSA value as their
  state-map key. Treating the converted QCO argument as a second key returns the
  stale argument instead of the latest qubit at `func.return`.
- The existing `build/release` directory still contains generated Neutral Atom
  QDMI manifests from an older checkout. They add a fifth discovered device and
  make two unrelated registry tests fail; final validation therefore uses a
  fresh build directory rather than deleting the user's existing build state.
- A malformed `qco.call` was parsed far enough for the function attribute
  verifier to query its unitary interface before the operation verifier ran. Its
  correspondence accessors and enclosing verifier must therefore be total even
  on invalid IR rather than relying on assertions or verifier order.
- `func.call`, `qc.call`, and `qco.call` carry argument, result, and discardable
  attributes. Conversion must preserve all representable metadata and reject
  nonempty attributes on synthetic QCO qubit results instead of silently
  dropping them when converting to resultless QC calls.

## Decision Log

- Decision: A QCO function places source-language results first, followed by one
  updated qubit for every scalar qubit argument in qubit-argument order.
  QCO-to-QC validates the correspondence before stripping those trailing values.
  No result annotation is used. Rationale: the formats being targeted borrow
  fixed qubit operands rather than returning arbitrary qubit identities; one
  positional convention is explicit, loss-minimizing, and cannot become stale
  independently of the signature. Date/Author: 2026-09-02, user and Codex.
- Decision: A marked QCO unitary function has `f64` parameters followed by
  qubits and returns those qubits positionally. `qco.call` has the same direct
  input/output mapping and an unknown compile-time matrix. Rationale: this is
  enough for gate definitions, modifiers, WireIterator, and format frontends
  without inlining or matrix synthesis. Date/Author: 2026-09-02, Codex.
- Decision: Generic `func.call` is a WireIterator boundary. Rationale: generic
  functions may measure, reset, allocate, branch, or return unrelated qubits;
  interprocedural consumers use the positional ABI directly rather than making
  local wire iteration infer whole-callee behavior. Date/Author: 2026-09-02,
  Codex.
- Decision: Builder APIs take complete callbacks and function handles.
  Rationale: insertion and linear-tracking state cannot leak across a paired
  start/end API, while result types are inferred once from the completed body.
  Date/Author: 2026-09-02, Codex.
- Decision: Keep malformed-call safety and metadata preservation local to the
  call operations and conversion patterns. Rationale: these are trust-boundary
  correctness checks; another ABI descriptor or annotation layer would duplicate
  the positional convention. Date/Author: 2026-09-02, Codex and independent
  specialist review.

## Context and Orientation

`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td` defines value-semantic quantum
operations and `qco::UnitaryOpInterface`. `QCOProgramBuilder` tracks live qubit
SSA values. Before this change, `WireIterator` followed generic calls through a
cached `CallQubitMapping`. `QCToQCO.cpp` and `QCOToQC.cpp` currently rely on
MLIR's type-only function/call conversion and therefore do not add or strip
qubit results.

## Plan of Work

Extend the unitary marker verifier to accept the QCO signature and prove each
returned qubit traces back to the corresponding argument through QCO unitary
operations.

Define `qco.call` with call/symbol interfaces and the QCO unitary interface. Its
qubit inputs and outputs correspond positionally. Add complete callback builder
APIs, validate the trailing positional results from local QCO wire flow, and
update live-value tracking from the function signature at generic calls.

Teach QC-to-QCO to append the latest value of each qubit function argument to
the function return and to convert `qc.call` to `qco.call`. Teach QCO-to-QC to
validate and strip those trailing pass-through qubit results from function
signatures, returns, and call sites, replacing each stripped call result with
the corresponding operand. Earlier results remain ordinary converted results.
Preserve call attributes in both directions; reject result attributes attached
to QCO-only pass-through qubits because QC has nowhere to store them.

Delete `CallQubitMapping`, its cache/invalidation API, and the special
`func.call` branches from WireIterator. `qco.call` needs no special iterator
code because it implements `UnitaryOpInterface`.

## Concrete Steps

Run focused builds and tests while iterating:

    cmake --build --preset release --target mqt-core-mlir-unittest-qco-ir mqt-core-mlir-unittest-qco-utils mqt-core-mlir-unittest-qc-to-qco mqt-core-mlir-unittest-qco-to-qc
    ctest --test-dir build/release -R 'QCO|QCToQCO|QCOToQC' --output-on-failure

Run final validation:

    cmake --build --preset release
    ctest --preset release
    uvx nox -s lint

## Validation and Acceptance

A marked QCO helper verifies and can be nested under QCO modifiers. Its call is
traversed by WireIterator through the unitary interface. A generic call ends
wire traversal. QC-to-QCO followed by QCO-to-QC preserves a helper call and does
not leave redundant pass-through qubit results in QC. QCO-to-QC followed by
QC-to-QCO reconstructs the same positional ABI for the supported one-block outer
function shape.

## Idempotence and Recovery

The work is isolated on `codex/qco-function-model` as the second commit of one
self-contained branch from `origin/main`. It will be published as one new PR,
independent of PR #2196 and its stack. Builds and tests are repeatable.

## Outcomes & Retrospective

The positional ABI supports generic and unitary scalar-qubit functions in the
QCO builder and in both conversion directions. `qco.call` gives local quantum
analyses an explicit unitary edge, while generic calls deliberately stop local
wire traversal. Removing the speculative generic-call inference deleted both
mapping caches and their failure-prone body analysis. The implementation passes
the complete test suite in a fresh release build; the existing release build
remains contaminated by obsolete generated QDMI manifests and was left intact.
After the specialist corrections, the call accessors are safe on malformed IR,
and call metadata is either preserved losslessly or rejected when QC cannot
represent it. No result annotations, recursive matrix synthesis, or generic-call
mapping abstraction was added. The specialist's final delta review found no
remaining actionable findings and judged the positional ABI and local iterator
boundary an idiomatic MLIR 23/C++20 foundation for OpenQASM, Qiskit, and jeff
integration.

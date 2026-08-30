# Rebase and simplify the call-mapping stack

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Pull requests `#2194` and `#2196` add call-aware linear-value tracking to the
QCO and QTensor MLIR utilities and to `QCOProgramBuilder`. The repository has
since merged new static-qubit validation, stricter conversion boundaries, and
new C++ and MLIR development rules. Rebase both pull requests onto the current
`main`, remove code that has no current consumer, and keep unsupported call
shapes fail-closed. A developer can then build multi-function QCO test programs
while the iterator utilities derive linear-value correspondence only when the
callee body proves it.

The remaining pull requests `#2197` through `#2201` build on `#2196`. Their
production code is outside this audit. Rebase them only as required to preserve
the stack and make each pull request add its own changelog reference.

## Progress

- [x] (2026-08-30 13:36Z) Read the root and MLIR agent guides, canonical C++ and
      MLIR policies, AI policy, and Ponytail skills.
- [x] (2026-08-30 13:36Z) Fetch current `main`, `#2194`, and `#2196` and save
      local backup refs for both remote heads.
- [x] (2026-08-30 13:36Z) Inspect merged `#2281` and `#2282` and identify
      changed files and contracts.
- [x] (2026-08-30 14:10Z) Rebase `#2194` on current `main`, reduce its mapping
      tests to the downstream contracts, and pass both focused utility suites.
- [x] (2026-08-30 14:10Z) Rebase and simplify `#2196` on rewritten `#2194`.
- [x] (2026-08-30 14:10Z) Rebase `#2197` through `#2201` without production-code
      changes and restore one changelog reference per pull request.
- [x] (2026-08-30 14:10Z) Run focused tests, all configured C++ tests, and both
      required lint sessions; all pass after accepting the plan's Markdown
      formatting.
- [x] (2026-08-30 14:16Z) Verify all nine rewritten commits, inspect every
      pull-request diff, publish the seven branches atomically with exact
      leases, and update the two audited pull-request descriptions and labels.

## Surprises & Discoveries

- Observation: The current `#2194` and `#2196` production files do not overlap
  the files changed by `#2281` or `#2282`. Evidence: the only path changed by
  both current `main` and `#2194` since their merge base is `CHANGELOG.md`;
  `#2196` has no overlapping path.
- Observation: `#2281` strengthens the module-wide QCO linearity contract by
  requiring unique static-qubit indices and program-level static roots in the
  entry block. This does not replace call correspondence because function
  arguments and call results still need body-derived pairing.
- Observation: `#2282` does not touch call mapping or the builder. Its relevant
  precedent is to reject unsupported MLIR shapes before mutation and report
  failure instead of guessing or terminating.
- Observation: Production consumers pass only type-filtered operands belonging
  to the call. `#2196` uses both mappings for builder tracking; `#2199`
  additionally shares and invalidates the qubit cache after callee mutation. No
  production consumer queries classical or foreign values, uses a reverse
  mapping, or invalidates tensor correspondence.

## Decision Log

- Decision: Keep the fail-closed `FailureOr<Value>` contract for call mapping.
  Rationale: Declarations, recursion, incomplete bodies, and multi-block bodies
  do not prove a linear-value correspondence. Positional pairing can join
  unrelated values. Date/Author: 2026-08-30, Codex.
- Decision: Audit `#2194` and `#2196` only; mechanically rebase later stack pull
  requests without production-code cleanup. Rationale: The user named the two
  audit targets and separately requested per-pull-request changelog ownership.
  Date/Author: 2026-08-30, Codex.
- Decision: `#2194` and `#2196` add their own references to the general compiler
  infrastructure entry. `#2197` through `#2201` each add only their own
  reference to the interprocedural-pass entry. Rationale: Iterator and builder
  support are infrastructure, while an early pull request must not promise later
  passes that can still change or be removed. Date/Author: 2026-08-30, Codex.
- Decision: Keep only three mapping scenarios per value kind: nested reordered
  correspondence, kept/created values where applicable, and fail-closed
  declarations/recursion. Rationale: These cover every distinct downstream
  contract; the removed pass-through, foreign-value, and duplicated traversal
  cases did not exercise another production behavior. Date/Author: 2026-08-30,
  Codex.
- Decision: Require helper functions to be finished before operations are added
  to `main`. Rationale: no production consumer needs to suspend partially built
  `main`; the ordering rule removes copied tracking sets and a speculative
  function-scope abstraction while preserving every downstream stack use.
  Date/Author: 2026-08-30, Codex.

## Outcomes & Retrospective

The Ponytail audit reduced `#2194` production and tests from 1,126 added lines
to 762, excluding changelog and plan edits, and reduced `#2196` from 650 to 393.
The retained code has production consumers in the stack or enforces fail-closed
behavior; duplicate input-shape tests, unused flexibility, copied builder state,
and a non-discriminating tensor-swap test were removed.

The focused QCO utility, QTensor utility, builder, and complete QCO IR suites
pass. A release build succeeds, all 3,923 configured tests pass with one
expected skip, and C++ lint reports zero clang-format or clang-tidy failures.
The general lint gate also passes cleanly. All nine commits verify as signed,
and every pull request adds only its own changelog reference. The stack was
published atomically with exact leases. Pull requests `#2194` and `#2196` retain
no assignees and use the `enhancement`, `c++`, and `MLIR` labels. Hosted checks
were not monitored because that was not requested.

## Context and Orientation

`mlir/include/mlir/Dialect/QCO/Utils/WireIterator.h` and
`mlir/lib/Dialect/QCO/Utils/WireIterator.cpp` define traversal over one linear
qubit value. Pull request `#2194` extends traversal across `func.call` by
tracing each callee argument through a supported body to its returned result.
`mlir/include/mlir/Dialect/QTensor/Utils/TensorIterator.h` and its
implementation provide the equivalent mapping for qubit tensors.

`mlir/include/mlir/Dialect/QCO/Builder/QCOProgramBuilder.h` and
`mlir/lib/Dialect/QCO/Builder/QCOProgramBuilder.cpp` own test-program
construction. Pull request `#2196` adds additional functions and calls. The
builder consumes the mapping utilities to transfer its tracked linear values
across calls.

The pull requests form a Git stack: `#2194` is based on `main`; `#2196` is based
on `#2194`; `#2197` through `#2201` each use the preceding pull-request branch
as their base. Rewriting an early branch requires rebasing every later branch so
that GitHub continues to show only each pull request's own change.

### Plan of Work

Rebase `#2194` onto the current remote `main`. Resolve the changelog conflict by
retaining current `main` and adding only `#2194` to the general compiler
infrastructure entry. Compare the rebased diff with upstream MLIR and current
repository helpers. Use Ponytail review to identify public methods, caches,
wrappers, comments, tests, or branches with no current consumer. Remove each
finding that can be deleted without weakening the fail-closed contract or its
smallest regression test. Convert changed public documentation to `///`, remove
forbidden `const` qualifiers from MLIR handles and by-value parameters, and use
current terminology.

Rebase `#2196` onto rewritten `#2194`. Add `#2196` and its link definition to
the same infrastructure changelog entry. Trace every new builder state field and
helper from `startFunction`, `endFunction`, and `call` to its callers. Remove
duplicated tracking paths or tests that only pin implementation details. Keep
one focused test for each supported behavior and each concrete failure contract.

Rebase `#2197` through `#2201` in order. Do not change their production code.
Make each pull request add only its own number to the changelog entry and define
only its own link. Preserve human authorship and replace no legitimate human
trailer.

### Concrete Steps

Run all commands from the repository root. Refresh the refs with:

    git fetch --prune origin main mlir/call-aware-iterators mlir/builder-call-support

Before rewriting, record each remote head and create a local backup ref. Rebase
the branches in stack order. After each substantive edit, inspect:

    git diff --check
    git diff --stat <pull-request-base>..HEAD
    git range-diff <old-base>..<old-head> <new-base>..HEAD

Build and run the two `#2194` utility binaries:

    cmake --build --preset release --target mqt-core-mlir-unittest-qco-utils mqt-core-mlir-unittest-qtensor-utils -j2
    ./build/release/mlir/unittests/Dialect/QCO/Utils/mqt-core-mlir-unittest-qco-utils --gtest_brief=1
    ./build/release/mlir/unittests/Dialect/QTensor/Utils/mqt-core-mlir-unittest-qtensor-utils --gtest_brief=1

Build and run the `#2196` builder tests:

    cmake --build --preset release --target mqt-core-mlir-unittest-qco-ir -j2
    ./build/release/mlir/unittests/Dialect/QCO/IR/mqt-core-mlir-unittest-qco-ir --gtest_filter='QCOTest.Builder*' --gtest_brief=1

Run the repository-required validation on the final stack head:

    ctest --preset release --output-on-failure
    uvx nox -s cpp-lint
    uvx nox -s lint

Expected focused output reports zero failed tests. `cpp-lint` and `lint` must
complete successfully and leave the worktree clean.

### Validation and Acceptance

Pull request `#2194` is acceptable when traversal crosses supported
straight-line calls in both directions, reports failure for unsupported callees
without positional guessing, and both QCO and QTensor utility binaries pass. Its
diff must not expose cache controls or reverse mappings without a production
consumer.

Pull request `#2196` is acceptable when the builder creates completed additional
functions, tracks qubits and qubit tensors across supported calls, rejects
unsupported callee shapes with the documented diagnostic, and the
builder-focused QCO IR tests pass. The builder must preserve outer tracked
values and reject leaks.

The stack is acceptable when each pull request adds only its own changelog
reference, every changed C++ file follows current MLIR and documentation policy,
all configured C++ tests pass, both lint sessions pass, all rewritten commits
verify as signed, and every GitHub diff contains only that pull request's scope.

### Idempotence and Recovery

Fetching, building, testing, and linting are repeatable. Local backup refs named
for the pull request and rewrite date preserve the pre-rewrite remote heads.
Before pushing, compare every remote head with its recorded lease. Push the
stack atomically with one exact `--force-with-lease` per branch. If a lease no
longer matches, stop without pushing and inspect the new remote work instead of
overwriting it.

### Artifacts and Notes

The pre-rewrite heads recorded on 2026-08-30 are:

    #2194 79d8ea2492224e6aca16e0d78ccf682bc76e9e10
    #2196 0c727ef274245a4d9ce45363466ad4c1fe1b38e4
    #2197 eabcd73a06054eed7b868b9e1df23c43fb55e9e8
    #2198 84eafd4d77493677aa3b1e4edb5dd0c7131a83be
    #2199 1ae55bf064d88ace2a25584d98c3743e3faaba8c
    #2200 6615c429b4f4514215f54dbff65110a08f03e567
    #2201 f4edd2d3ec57c4cdbff8a2be380bb722d61d531c

The current `main` head after #2282 is:

    d1c19982952c79862c95a4903d302b2a8a295752

### Interfaces and Dependencies

Use MLIR's `FailureOr<Value>` and `LogicalResult` for mapping failure. Use
`func::CallOp`, `func::FuncOp`, `Value`, `ArrayRef`, `SmallVector`, `DenseMap`,
and `DenseSet` from the existing LLVM and MLIR dependencies. Add no dependency,
new dialect operation, interface, trait, or general-purpose abstraction.

Revision note (2026-08-30): Created after refreshing the stack, then updated
after the Ponytail audit and validation to record simplifications,
per-pull-request changelog ownership, and the required mechanical restack.

# Run the MQT Core v4 SpecAudit campaign

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root. Every subsystem audit must follow `.agent/AUDITS.md`.

## Purpose / Big Picture

MQT Core v4 and the MQT Compiler Collection add substantial code, tests, and
documentation. This campaign checks whether the tests defend deliberate
contracts or preserve choices that no user, specification, or maintainer asked
the project to keep. The result is a set of small, evidence-backed SpecAudits.
Each audit identifies safe ways to narrow tests, remove code, simplify
ownership, or unlock measured performance work. The audits do not apply their
findings.

A maintainer can inspect each file under `.agent/audits/`, reproduce every
experiment from a clean checkout, accept or reject individual verdicts, and then
commission one focused change at a time. The campaign is complete when every
declared subsystem is either audited or explicitly deferred with a reason, and
the scope map contains no unexplained gap or duplicate owner.

## Progress

- [x] (2026-08-19 21:00Z) Read `.agent/AUDITS.md`, `.agent/PLANS.md`, and
      `docs/ai_usage.md` in full.
- [x] (2026-08-19 21:00Z) Pinned the initial campaign snapshot to
      `cb5cf0103bd9841726c8ec6c5abb725758afea58` and confirmed that it matched
      the remote default branch.
- [x] (2026-08-19 21:00Z) Refreshed the live GitHub intake: 47 open issues and
      18 open pull requests.
- [x] (2026-08-19 21:00Z) Selected global-phase normalization as the pilot and
      confirmed that no open pull request directly changes its source or test.
- [x] (2026-08-19 22:15Z) Completed every pilot wave and produced the
      global-phase-normalization audit for maintainer review.
- [x] (2026-08-20) Recorded the maintainer decision: accept pilot verdicts 1-6,
      defer verdict 7, and keep verdicts 8-9.
- [ ] Resolve accepted pilot verdicts in one pull request with one commit per
      verdict.
- [ ] Run the collision-free Core cleanup cohort.
- [ ] Run the stable compiler leaf cohort.
- [ ] Reconcile and run scopes that current pull requests block.
- [ ] Present accepted verdicts to a maintainer for resolution selection.
- [ ] Reconcile each audit after selected verdicts merge.

## Surprises & Discoveries

- Observation: the repository has no `mlir/test/` tree. The MLIR census must
  cover `mlir/unittests/`, the compiler-facing Python tests under
  `test/python/`, and `test/qir/mqt-cc/`. Evidence: the repository file census
  at the initial baseline.
- Observation: the fixture libraries under `mlir/unittests/programs/` contain
  substantial scaffolding but no direct assertions. They are architecture-level
  audit inputs, not independent assertion scopes. Evidence: the initial MLIR
  scope census.
- Observation: several high-value scopes have direct open-pull-request overlap.
  Auditing a synthetic merge would produce evidence for a state that never
  existed on the default branch. Evidence: the GitHub path-overlap census in
  `Artifacts and Notes`.
- Observation: the native agent tree reached its lifetime thread limit after
  four threads, even when earlier agents had completed. Fresh role-isolated
  sessions in detached disposable worktrees preserved the required separation.
  Evidence: the pilot agent registry and clean-worktree checks.
- Observation: one requested-behavior cartographer followed a broad pull-request
  timeline and received embedded test diffs. The controller discarded the entire
  output and repeated that source class in a fresh session. Evidence: the pilot
  contamination record.
- Observation: the read-only process sandbox was not usable for one isolated
  session on this host. A normal disposable worktree with explicit read-only
  role instructions and final status checks provided the same audit boundary.
  Evidence: the pilot session log and empty final worktree status.
- Observation: the first red-team pass found two verdicts whose supplied
  mutations had not failed the accused assertion. Two fresh executors closed
  those gaps before the audit editor started. Evidence: pilot mutations M8 to
  M10.
- Observation: dynamic addition and integral-power scaling can turn individually
  valid runtime phase angles into a result outside the documented phase-angle
  precondition. This is a correctness follow-up, not permission to change code
  inside the audit. Evidence: the pilot red-team analysis.

## Decision Log

- Decision: run a portfolio of small SpecAudits instead of one repository-wide
  audit. Rationale: `.agent/AUDITS.md` requires one subsystem at a time and an
  executable evidence boundary. Date/Author: 2026-08-19, Codex.
- Decision: give each active audit a persistent scope steward, but use separate
  agents for every adversarial role. Rationale: continuity helps iteration, but
  an agent that sees tests while deriving promises contaminates the ledger.
  Date/Author: 2026-08-19, Codex.
- Decision: permit at most two read-only audits, one global probe executor, and
  one audit editor at once. Rationale: probes mutate disposable worktrees and
  C++ coverage builds compete for shared machine resources. Date/Author:
  2026-08-19, Codex.
- Decision: defer a scope when an open pull request directly changes its source
  or tests. A downstream-only pull request permits read-only audit work but
  blocks resolution until reconciliation. Rationale: this keeps evidence tied to
  a real, pinned default-branch state. Date/Author: 2026-08-19, Codex.
- Decision: use global-phase normalization as the pilot. Rationale: it has one
  focused implementation, one focused test target, meaningful quantum semantics,
  and no direct open-pull-request collision. Date/Author: 2026-08-19, Codex.
- Decision: spawn a fresh resolution agent for each maintainer-selected verdict,
  not for each audit. Rationale: the audit method requires each verdict to
  remain independently reviewable. Date/Author: 2026-08-19, Codex.
- Decision: discard a role's complete output after forbidden-input exposure.
  Rationale: trying to retain the uncontaminated part would make the promise
  ledger impossible to verify. Date/Author: 2026-08-19, Codex.
- Decision: use detached disposable worktrees and fresh sessions when the native
  agent tree cannot allocate another role. Rationale: `.agent/AUDITS.md`
  requires fresh context and clean baselines, not a specific process launcher.
  Date/Author: 2026-08-19, Codex.
- Decision: require the red team to check assertion arithmetic and the exact
  failing site for every mutation. Rationale: a nearby failure is not evidence
  for the assertion under judgment. Date/Author: 2026-08-19, Codex.
- Decision: treat the pilot's runtime-angle finding as a separate correctness
  follow-up. Rationale: a SpecAudit may identify a defect, but it does not apply
  a repair or broaden authorization beyond its verdict ledger. Date/Author:
  2026-08-19, Codex.

## Outcomes & Retrospective

The campaign control plan, scope graph, baseline rules, GitHub gates, and pilot
are complete. The pilot audited 120 assertion sites at one pinned baseline. It
found ten over-specified sites, two redundant sites, three contract-free sites,
and no coverage-driven sites in the focused file. The five transitive matrix
helper assertions remain anchored. The pilot earned test narrowing and one
correctness-enabling representation freedom. It did not earn a production
deletion or a measured performance claim.

The pilot changed the operating procedure for later cohorts. Allocate every role
in a fresh context before work starts. Give cartographers narrow GitHub requests
that cannot return patches or test timelines. Require the census to publish
checkable verdict arithmetic. For every shape verdict, run an equivalent-shape
mutation that fails the exact assertion. When two regression tests repeat a
guard, inject the fault into each path instead of assuming one result covers
both. Run the red team before the editor, and repeat any affected probe before
the editor writes the audit.

The remaining cohorts stay gated by the live overlap rules above. On 2026-08-20,
the maintainer accepted verdicts 1-6, deferred verdict 7, and kept verdicts 8-9.
The accepted verdicts may now proceed through the resolution workflow in one
pull request with separate commits. Verdict 7 remains blocked on an ownership
decision.

## Context and Orientation

A SpecAudit examines assertions, not files or test functions. Each assertion
must be compared with a promise in the spec ladder from `.agent/AUDITS.md`.
External specifications and machine-checked declarations are strongest, followed
by published documentation and maintainer requests. Existing code and tests are
not promises by themselves.

Every audit is stored at `.agent/audits/<scope-slug>.md`. It records the exact
source and test scope, a pinned commit, a numbered promise ledger, every
assertion verdict, executed evidence, confirmed anchors, unlocked changes, and
items deliberately left alone. An audit stops at this ledger. It does not edit
production code or tests.

The campaign controller owns the queue, scope graph, resource leases, baseline
commits, GitHub refresh times, and agent registry. The controller does not
author verdicts. Each active audit has a scope steward that preserves status and
routes feedback. The steward does not act as cartographer, prosecutor, defender,
executor, unlock analyst, red team, or adjudicator.

Record each role assignment in this form in the relevant audit progress log:

    role | agent identifier | baseline | allowed inputs | output | status

Agent continuity is useful only within one role. Durable audit files, not agent
memory, are the source of truth.

## Audit Scope Graph

The pilot is global-phase normalization:

- Source: `mlir/lib/Dialect/Utils/Transforms/NormalizeGlobalPhases.cpp` and the
  public declaration that registers or describes the pass.
- Tests: the assertions in
  `mlir/unittests/Dialect/Utils/test_global_phase_normalization.cpp`.
- Focused target: `mqt-core-mlir-unittests-dialect-utils`, restricted to the
  global-phase tests.

The first Core cleanup cohort contains these scopes:

1. Algorithms ownership under `include/mqt-core/algorithms/` and
   `src/algorithms/`, linked to issue `#2095`.
2. Three CircuitOptimizer scopes: structural normalization, local algebra and
   block formation, and measurement or dynamic normalization, linked to issue
   `#2088`.
3. Classic OpenQASM serialization in the operation hierarchy, linked to issue
   `#2098`.
4. The standalone QIR CLI and JIT boundary, linked to issue `#2096`.
5. DD simulation and state-generation APIs, linked to issues `#2102` and
   `#2103`.

The classic Core foundation cohort contains:

1. Symbolic expressions and symbolic operations.
2. The operation, compound-operation, and conditional-operation hierarchy.
3. `QuantumComputation`, registers, permutations, and container behavior.
4. Separate classic OpenQASM lexer/parser/AST, semantic-pass, and importer
   audits, linked to issue `#2097`.
5. Separate DD numeric-table, package/storage, and Python-surface audits.
6. Classic Qiskit conversion, Qiskit QDMI backend, Qiskit primitives, and the
   top-level Python loading and command-line surface.

The compiler leaf cohort contains:

1. CBit IR and CBit-to-memref lowering as separate audits.
2. QC builder/allocation, dense-unitary semantics, modifier canonicalization,
   and modifier unrolling as separate audits.
3. QCO builder/allocation, structured control flow, gate or modifier folding,
   and dense-unitary matrix semantics as separate audits.
4. QTensor IR or alias behavior and TensorIterator or register shrinking as
   separate audits.
5. One audit for each QCO optimization pass.
6. Separate Euler, Weyl, multi-control, single-qubit-run fusion, target
   synthesis, mapping-pass, and layout/graph audits.

The conversion and external-format cohort contains:

1. QC-to-QCO and QCO-to-QC as separate directional audits. Every round-trip
   assertion has one primary owner.
2. QIR Base lowering, Adaptive lowering, and QIR builder/profile metadata.
3. The paired `jeff` import/export boundary.
4. OpenQASM syntax/parser, semantic analysis, OpenQASM-to-QC, and QC-to-OpenQASM
   as four separate audits.
5. Native Qiskit import, export, and C API/version handling as three audits.

The integration cohort contains:

1. Compiler program ownership and format handling.
2. The default compiler pipeline.
3. The compiler target model and target-compilation pipeline.
4. The compiler QDMI adapter.
5. After the QDMI redesign settles: registry/configuration, manager/lifetime,
   device/job/site/operation, and Slurm as four audits.
6. Separate DDSIM and SC QDMI provider audits.

Refuse scopes such as all of MLIR, all Core IR, all DD, all QDMI, all Python
bindings, all tests, or one audit per top-level directory. Split any proposed
scope whose promises cannot be listed in one afternoon, whose tests cannot run
with one command, or whose prosecutors would need substantially more than twenty
tests each.

## GitHub Intake and Drift Rules

Before each audit, refresh all open issues and pull requests. Attach relevant
items to the audit with one of these dispositions:

- `promise candidate`: request text may define a rung-3 promise;
- `reported defect`: a regression test must survive;
- `direct baseline changer`: the open pull request edits scoped source or tests;
- `downstream consumer`: the open work depends on the scoped behavior;
- `defer`: the audit cannot produce stable evidence yet; or
- `not relevant`: reviewed and excluded.

An open issue can state requested behavior. An unmerged pull request's code,
tests, and implementation choices do not become promises for the default-branch
baseline. A cartographer may read issue, discussion, and review text within its
assigned source class, but must not read the pull request patch or tests.

Refresh GitHub state before the census, before probes, before red-team review,
before human acceptance, and before resolution. Never rebase a worktree during
an experiment. If the default branch changes, apply these rules:

- A contract or public-surface change refreshes the ledger and affected
  prosecution.
- A test change requires a new census and affected assertion work.
- An implementation-only change requires rereading citations and rerunning
  affected probes and unlock analysis.
- An unrelated change preserves the pinned experiment, but the audit still
  reconciles to current `main` before acceptance.
- A directly overlapping open pull request defers the scope. Never audit a
  synthetic merge.

At campaign start, the main direct-overlap gates are QDMI pull request `#1901`,
QCO/DD pull requests `#2077` through `#2080`, Qiskit pull request `#2150`,
classical-control pull request `#2162`, OpenQASM-angle pull request `#2169`,
mapping pull requests `#1955` and `#1956`, and pass-family pull requests
`#1845`, `#1970`, and `#2062`. Pull request `#2032` adds a Quake boundary that
does not exist at the initial baseline. If it merges, create a new Quake audit.

## Agent Waves for One Audit

Run these waves in order. Parallel execution is optional; role isolation is
mandatory.

1. Four highest-reasoning cartographers independently derive promises from
   external or machine-checked contracts, published material, requested text,
   and the declared public surface. They do not read tests.
2. One census agent enumerates every test and assertion with a stable ID and no
   judgment. A stable C++ ID consists of the GoogleTest suite, test name, and
   assertion ordinal. A stable Python ID consists of the pytest node ID and
   assertion ordinal.
3. Independent prosecutors receive the accepted ledger, scoped implementation,
   and no more than about twenty tests each. Every assertion receives one
   proposed class and a ledger reference or explicit `no promise`.
4. Provenance agents receive suspect assertion IDs and return commits, linked
   issues, change history, and AI-assistance signals. They return facts, not
   verdict opinions.
5. Defenders receive the clean ledger and accused assertions, but never the
   prosecution reasoning. Their only job is to find missed promises, reported
   defects, safety properties, or outside consumers.
6. One executor uses `.agent/audit-probe.sh` or an equivalent uniform command to
   run T1 coverage and T2 fault injection. The executor reports exact output and
   does not interpret it beyond the protocol. Use T3 scoped mutation for
   high-value verdicts.
7. Unlock analysts receive confirmed verdicts and executed evidence. Each unlock
   names the exact lifted assertion and exact enabled change. Run exactly one
   additional architecture-altitude analyst.
8. A fresh red team tries to find the verdict most likely to break a real user.
   A fresh editor then produces the self-contained audit file.

When feedback arrives, send a corrected promise to the matching cartographer, a
missed assertion to the census agent, a verdict objection to a fresh defender or
probe, a reproduction problem to the executor, and an editorial correction to
the editor. Repeat the red team after any material draft change.

## Quality Gates

An audit is ready for human acceptance only when:

- it names one bounded source scope, test scope, focused command, clean
  baseline, and GitHub snapshot;
- all four cartographers completed their work without reading tests;
- the census covers every scoped assertion, including indirect Python or
  command-line assertions assigned to the scope;
- every assertion has exactly one verdict and ledger reference or an explicit
  statement that no promise exists;
- every verdict cites reproducible executed evidence;
- reported defects, external contracts, and safety properties are identified as
  anchors;
- coverage consequences are explicit;
- every unlock names the blocking assertion and enabled change;
- ordinary cleanup observations remain separate from SpecAudit unlocks;
- the architecture analyst and red team completed their work;
- relevant GitHub work was refreshed and classified;
- every citation was reread at the pinned baseline;
- prose follows the repository style and wraps at 80 columns; and
- `uvx nox -s lint` passes for the audit-file batch.

Abort or defer an audit when the baseline is dirty, its baseline tests fail,
evidence cannot be reproduced, or overlapping work makes the conclusions
unstable.

## Resolution Workflow

An accepted audit does not authorize a code change. A maintainer selects
individual verdicts. For each selected verdict, a fresh resolution agent:

1. refreshes current `main`, the audit, open issues, open pull requests, and
   relevant consumers;
2. confirms that the verdict and unlock still hold;
3. writes an independent ExecPlan when the change is a significant refactor;
4. narrows or removes the assertion in the first commit;
5. applies the enabled simplification or performance change in a second commit;
6. runs focused tests, affected suites, coverage, lint, and downstream checks;
7. records reproducible before/after measurements for performance claims; and
8. hands the result to a fresh verifier before any publication.

One verdict normally produces one pull request. Do not assign an AI resolution
to a `good first issue`. Pushing, opening a pull request, or posting GitHub text
requires explicit human authorization. Public agent-authored text must include
the disclosure required by `docs/ai_usage.md`.

After a resolution merges, reconcile the audit against current `main`. Mark the
verdict `applied`, `narrowed`, or `superseded`, preserve the reason, and rerun
dependent audit evidence when necessary.

## Concrete Steps

From the repository root, initialize an audit only from a clean, pinned default
branch commit. Create a dedicated detached worktree and a distinct build tree
for mutable probes. Do not let two agents mutate one worktree.

For C++ or MLIR work, configure and validate with the commands documented in
`AGENTS.md`:

    cmake --preset release
    cmake --build --preset release --target <focused-target>
    ./build/release/<focused-test-binary> --gtest_filter='<owned-tests>'

For coverage probes:

    cmake --preset coverage
    ./.agent/audit-probe.sh t1 --lang cpp --source <source> \
      --target <target> --ctest <regex>
    ./.agent/audit-probe.sh t2 --lang cpp --target <target> \
      --inject <file:line> --with '<replacement>' --ctest <regex>

For Python scopes:

    uv run --no-sync pytest <test-path> --collect-only
    uv run --no-sync pytest <test-path>
    ./.agent/audit-probe.sh t1 --lang python --source <source> \
      --tests <tests> --drop <pytest-node-id>
    ./.agent/audit-probe.sh t2 --lang python --tests <tests> \
      --inject <file:line> --with '<replacement>'

After each audit-document batch:

    uvx nox -s lint
    git diff --check
    git status --short

Capture the exact command, baseline, mutation, exit status, failing tests, and
coverage delta in the audit. Restore the probe worktree and confirm that it is
clean before reusing or retiring it.

## Validation and Acceptance

The global-phase pilot proves that the process works when:

- four uncontaminated source-class ledgers reconcile into one numbered ledger;
- the census assigns every assertion a stable identifier;
- prosecution and defense disagree where warranted without sharing reasoning;
- every final verdict cites an executed probe;
- an architecture analyst distinguishes test-enabled unlocks from ordinary
  cleanup;
- a red team reports residual risk;
- the audit file can reproduce each result from a clean checkout; and
- the audit document passes repository lint.

The campaign proves repository coverage when each scope in `Audit Scope Graph`
has an accepted audit or a current, explicit deferral. No assertion may belong
to two audits. Round-trip and integration assertions must record one primary
owner and any secondary consumers.

Rank findings by complexity removed per unit of risk. Track public APIs,
targets, dependencies, branches, abstractions, source lines, test lines, build
time, test time, runtime, memory, coverage, and measured compiler performance
when relevant. Do not set a test-deletion target. A high anchored percentage is
a useful result.

## Idempotence and Recovery

Cartography, census, history searches, and baseline test runs are read-only and
repeatable. Probe mutations occur only in a clean dedicated worktree. The probe
script restores its injected line unless explicitly told to keep it. Never
continue after a probe leaves the worktree dirty.

Do not rebase an experiment worktree. If `main` moves, finish or discard the
pinned experiment, create a new clean baseline, and reconcile through the rules
above. Preserve old verdict text and evidence when marking a finding applied,
narrowed, or superseded.

If an agent becomes unavailable, restart its role from the audit file, baseline,
allowed inputs, and recorded output. Do not compensate by giving a contaminated
agent a forbidden role.

## Artifacts and Notes

Initial campaign snapshot:

    baseline: cb5cf0103bd9841726c8ec6c5abb725758afea58
    date: 2026-08-19
    open issues: 47
    open pull requests: 18
    global-phase direct path collisions: none

Completed pilot artifact:

    .agent/audits/global-phase-normalization.md

The existing `.agent/audits/pennylane-plugin.md` is a living audit. Do not
restart it. Reconcile it after direct-overlap Python-support and QDMI work
settles.

## Interfaces and Dependencies

The campaign depends on Git, CMake presets, CTest, GoogleTest, `uv`, Nox, the
repository's `.agent/audit-probe.sh`, and read access to GitHub issue and pull
request state. No new production or test dependency is permitted merely to run
the campaign. T3 mutation remains a manual scoped mutation; do not add a
mutation-testing package.

Every checked-in audit and this ExecPlan must use repository-relative paths and
must not record local checkout locations, account names, credentials, or
ephemeral branch names.

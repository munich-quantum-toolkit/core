# Spec Audits (SpecAudits)

This document describes how to audit a part of this repository for *spec debt*:
tests that assert behavior nobody ever promised, and the code complexity those
tests hold in place. It is the counterpart to [`.agent/PLANS.md`](PLANS.md). An
ExecPlan is the forward pass, where an agent turns a request into working code.
A SpecAudit is the backward pass, where a later agent checks what that autonomy
actually froze.

Treat the reader as a complete beginner to this repository. They have only the
current working tree and this file.

## What spec debt is

An agent receives a request that does not state every detail. It fills the gaps
and writes code. In the same commit it writes tests for that code. The tests
pass, so they are correct. They are also a specification that nobody wrote on
purpose.

The next agent reads those tests as the contract. It builds on them. A third
agent tries to simplify the code and finds that the change breaks tests, so it
reverts. The accidental choice is now permanent.

Spec debt costs twice. The test suite grows without covering more intent. And
the code below it freezes, because every simplification looks like a regression.

This repository already forbids the first half. `AGENTS.md` says:

> Add tests that protect intended behavior or reproduce a concrete regression.
> Never test provisional implementation choices that are not part of the
> supported contract.

`PLANS.md` says the same thing from the other side:

> Err on the side of over-explaining user-visible effects and under-specifying
> incidental implementation details.

Nothing checks either rule after the work lands. A SpecAudit is that check.

## How to use SpecAudits and AUDITS.md

When you audit a subsystem, follow this file to the letter. If it is not in your
context, read all of it first.

Store one SpecAudit per audited scope at `.agent/audits/<scope-slug>.md`. The
audit is a living record. Reconcile it on later runs instead of rewriting it.

A SpecAudit produces a ranked list of verdicts with evidence.
**It never lands the changes it recommends.** A human decides which verdicts to
apply, and the work happens in separate pull requests afterwards.
`docs/ai_usage.md` requires that a human stays accountable, and a bulk deletion
of tests that nobody reviewed is the "extractive contribution" that document
warns about.

## Requirements

NON-NEGOTIABLE REQUIREMENTS:

- Every verdict must cite evidence from an executed experiment. An argument is
  not evidence.
- Every verdict must name the promise it tested the assertion against, or state
  that no promise exists.
- Every claim of `file:line` must come from reading that file at the pinned
  baseline commit. Not from memory, and not from a search snippet.
- Never delete a test that reproduces a reported defect.
- The audit stops at the ledger. It does not delete tests, change code, or open
  pull requests.
- Every SpecAudit must be self-contained. A reader with only the audit file and
  a clean checkout must be able to re-run every experiment in it.

## The unit of audit is the assertion

Not the file, and not the test function. One test can hold three assertions that
each deserve a different verdict: one that guards a documented contract, one
that pins an exact error message, and one that is dead weight.

Judge each assertion. Then report remedies per test, because that is how a patch
is written.

## The spec ladder

Ask one question of every assertion:
**which promise does this defend, and who made it?**

Sources rank as follows, strongest first.

1. **External and machine-checked.** A specification this repository does not
   own, or one a tool verifies. The QDMI headers that
   `cmake/ExternalDependencies.cmake` fetches at a pinned revision are the
   clearest case: the repository cannot quietly change them. The MLIR TableGen
   files under `mlir/include` are the next: an operation's arguments, results,
   traits, verifier, and assembly format are a checked contract. Then the
   OpenQASM grammar and the QIR profiles that `docs/qir/` names.
2. **Published.** A promise this project made to its users in writing.
   `CHANGELOG.md` entries, `UPGRADING.md` migrations, the rendered pages under
   `docs/`, and Doxygen comments on headers in `include/mqt-core/`.
3. **Requested.** The issue, discussion, or review comment that caused the work,
   in the requester's own words.
4. **Planned.** An ExecPlan under `.agent/plans/`. Treat this rung with care.
   Most plans in this repository were committed in the same commit as the code
   and tests they describe, so they are the agent's own account of what it did,
   not a specification that existed first. A plan's stated purpose counts as
   rung 3. A detail the plan invented on its way to that purpose does not.
5. **Implemented.** The code behaves this way.
6. **Asserted.** A test says so.

**Rungs 5 and 6 are not promises.** An assertion whose only support is "the code
does this" is a mirror. It cannot fail unless somebody changes the code, which
is exactly when you want the freedom to change it.

The first product of an audit is a **spec ledger**: a numbered list of the
promises that apply to the scope, each with a citation and its rung. Write it
before you look at a single test. Number the entries `S1`, `S2`, and so on.
Every later verdict refers to those numbers.

For MLIR work this gets sharp. If a unit test asserts something about an
operation that you cannot derive from that operation's TableGen definition, the
assertion is a candidate by construction.

## Verdict classes

Give every audited assertion exactly one of these.

**Anchored.** It defends a ledger entry at rung 1, 2, or 3, and it does not pin
more than that entry says. Leave it alone. Record it anyway: a scope where 90%
of assertions are anchored is a good result, and the next auditor needs to know
somebody already checked.

**Over-specified.** It defends a real promise but constrains more than the
promise states. The remedy is to narrow it, never to delete it. An exact
exception message becomes the exception type plus a stable substring. A full
serialized blob becomes the fields the contract names. An exact float rendering
becomes a comparison with a tolerance.

**Redundant.** Another assertion already covers the same equivalence class. The
remedy is to merge or to parametrize.

**Contract-free.** No rung 1 to 3 source promises this behavior. The remedy is
to delete the assertion, and then to treat the code that exists only to produce
that behavior as a removal candidate. This class is where the value is.

**Coverage-driven.** The assertion exists to satisfy a coverage gate over code
that cannot be reached through the public interface. Codecov sets patch targets
of 90% for C++ and 95% for Python, and this repository excludes almost nothing
from coverage, so unreachable defensive branches create real pressure. An agent
relieves that pressure by manufacturing an impossible input.
`test/fomac/test_fomac.cpp` casts `0` to an enumeration behind a suppression
comment for `EnumCastOutOfRange` in order to reach a `detail::` throw. The
remedy here is usually a coverage exclusion plus removal of both the test and
the branch, not a better test.

## Provenance signals

Version history tells you where to look first. It does not decide the verdict;
the ladder does. Keep the two apart, or you will convict good tests for the
crime of having been written recently.

This matters here more than it might elsewhere. Writing implementation and tests
in one commit is the normal shape of a change in this repository, not an
anomaly, so co-introduction alone convicts almost everything and therefore
convicts nothing. Use these signals in order of how well they discriminate.

1. **The assertion changed in the same commit that changed the behavior it
   asserts.** This is the strongest signal in the set. It proves the assertion
   follows the code rather than constraining it. Watch for the case where the
   rewrite came out *tighter* than the original, because that is spec debt
   accumulating in real time.
2. **The assertion has never failed.** `git log -S '<assertion text>'` returns
   one commit. It was born green and stayed green.
3. **The commit carries an AI-assistance trailer and the ExecPlan it implements
   says nothing about this specific behavior.** The behavior came from the
   gap-filling, not from the request.
4. **Test and code arrived together with no linked issue text stating the
   requirement.** Weakest of the four. Common and mostly innocent.

## Smell catalogue

Each entry is a pattern, a reason, and a remedy. The searches are starting
points for the census, not verdicts.

| Smell                       | Search                                               |
| :-------------------------- | :--------------------------------------------------- |
| Exact error message, C++    | `EXPECT_EQ(.*\.message,`, `EXPECT_STREQ(e.what()`    |
| Exact error message, Python | `pytest.raises` with a long `match=`                 |
| Internal symbol             | `detail::` or a leading underscore in a test         |
| Incidental ordering         | `assert \[.* for .* in .*\] ==`                      |
| Float rendering             | a string literal matching `\d\.\d{8,}`               |
| Golden blob                 | `EXPECT_EQ(ss.str(), "` with many continuation lines |
| Vacuous assertion           | `assert .* is not None` as a whole test body         |

**Exact error messages.** The contract is usually that an error occurs and which
kind. The wording is not. Pinning it blocks every improvement to diagnostics and
freezes typos: this repository holds an assertion on the string
`"Gate 'my_x' takes 1 targets, but 2 were supplied."` Narrow to the type plus a
stable substring.

**Internal symbols.** A test that reaches into a `detail::` namespace, or an
underscore-prefixed Python name, asserts an implementation choice by definition.
Those names exist to say "not part of the contract". Either the behavior matters
through the public interface, in which case test it there, or it does not, in
which case delete.

**Mocks that assert calls.** Counting calls or checking their order tests the
mechanism, not the result. Such a test fails on every refactor and passes
through every bug that keeps the same call shape. Assert the result instead.

**Incidental ordering.** Dictionary iteration, set iteration, filesystem
listing, and registry order are contracts only when a document says so.
Otherwise compare as a set or sort both sides.

**Float rendering.** A literal such as `rxx(0.10000000000000001)` pins the
formatting precision of an emitter to seventeen significant digits. Nobody
promised that. Compare parsed values with a tolerance.

**Golden blobs.** An assertion on a whole serialized output, including
indentation and coordinate formatting, converts every cosmetic change into a
test failure. Assert the fields the contract names.

**Indistinguishable parameter matrices.** A parametrized suite whose cases the
code under test cannot tell apart adds runtime and no oracle. Keep one case per
equivalence class and say what the classes are.

**Names that restate the implementation.** A test called
`ProductionTranslationUsesTheStagedPipeline` promises to fail when somebody
replaces the staged pipeline with a better one. Rename to the observable
outcome, and if no observable outcome exists, that is the finding.

**Vacuous assertions.** A test whose body is `assert thing is not None`, with a
docstring conceding that the interesting parameters are ignored, tests that
construction did not raise. Say that, once, and delete the rest.

**Fixtures that dwarf their assertion.** Scaffolding is a cost. When the setup
is fifty lines and the check is one, ask what the check is for.

## Anchors that must survive

Prosecution is easy and the failure mode is over-deletion. Any assertion with
one of these properties stays, and the audit records why.

- It reproduces a reported defect. Find the linked issue or the fix commit. This
  is the highest-value class of test in any repository.
- It encodes a contract with something outside this repository: a QDMI header, a
  QIR profile, an OpenQASM construct, a serialization format another tool reads.
- It guards a safety property: resource cleanup, absence of a leak, absence of
  undefined behavior, a bound on memory or time.
- It is the only assertion that fails when you inject a fault that a rung 1 to 3
  promise makes visible. It earned its place; say so in the ledger.

## Evidence protocol

**No verdict without an executed experiment.**

Escalate through three tiers, and stop as soon as one settles the question.
Configuring and building the C++ targets in this repository takes several
minutes, so spend the expensive tiers only where the payoff justifies them.

### T1, coverage delta

Measure coverage of the source in scope. Remove the candidate assertion. Measure
again.

For Python, use the repository's own command and configuration:

```sh
uv run --no-sync pytest --cov-config=pyproject.toml
```

For C++, use the coverage preset:

```sh
cmake --preset coverage
cmake --build --preset coverage
ctest --preset coverage
```

No change in line or branch coverage is evidence of redundancy. It is never
proof on its own. An assertion can duplicate another's coverage while holding
the only oracle that would catch a wrong value.

### T2, fault injection

This tier does the real work. Restore the assertion. Now break the code it
claims to guard: invert a condition, delete a guard, change a bound by one,
return the wrong value. Run the whole suite for the scope.

Read the result as follows.

- **Another test fails.** The candidate is **redundant**. The suite already
  catches this fault. Delete the candidate and name the test that covers it.
- **Nothing fails, and no rung 1 to 3 promise makes the fault visible.** The
  behavior is **contract-free**. Delete the assertion. Then look at the code you
  just broke: you have proven that nothing outside it depends on what it does.
  It is a removal candidate.
- **Nothing fails, but a promise does make the fault visible.** The candidate is
  the sole anchor for that promise. **Keep it.** Record the injected fault in
  the ledger so the next auditor does not repeat the experiment.

The second outcome is the point of the whole method. A fault that nobody notices
means the behavior is unspecified *and* the code producing it is weight.

### T3, scoped mutation

For high-value verdicts only. Mutate the one translation unit or module under
audit and compare the set of surviving mutants before and after the change you
propose. If the set is unchanged, the suite is exactly as strong as it was.

This repository has no mutation-testing dependency, and a SpecAudit does not add
one. Drive T3 by hand through the probe script.

### The probe script

[`.agent/audit-probe.sh`](audit-probe.sh) runs the T1 and T2 loop so that every
candidate gets the same treatment. It refuses to run on a dirty working tree,
restores what it changes, and prints an evidence block for the ledger. Read
`./.agent/audit-probe.sh --help` for the arguments.

Running the loop by hand is allowed. Running it differently for different
candidates is not.

## Unlock analysis

Cleaning the suite is half the work. The other half is spending the freedom it
buys. For every assertion you narrow or delete, ask:
**what does this now permit?**

An unlock claim must name the constraint that was lifted and the change that
becomes legal. "This code could be simpler" is not a claim.

Look for these classes.

**Defensive branches that nothing can reach.** A guard against an input the
public interface cannot construct. Look for messages that admit the branch is
dead, such as `"unhandled gate type after pre-check"`, which names the earlier
check that already excluded the case.

**Parameters with one value.** A flag, option, or argument that every call site
passes the same way. Once no test forces the other path, the parameter and its
branch go.

**Abstractions with one implementation.** An interface kept alive because a test
mocks the seam. When the mock goes, the interface can collapse into its single
implementation.

**Test-only seams.** Production code that exists so a test can hook into it.
This repository ships a context manager in a compatibility module whose
documented purpose is to let tests fake a missing dependency. Such a seam is
worth keeping only when the test it enables defends a real promise.

**Duplicated logic held in place by two tests.** Two copies stay in sync because
an assertion pins each. This repository defines an `unreachable` helper in two
headers, and a test fixture hardcodes a gate table that also exists as a
generated definition file. One copy should win.

**Optimization unlocks.** This is the class that most often goes unnoticed. Work
the code does *only* to satisfy an over-specified assertion:

- sorting to produce a deterministic order nobody promised;
- defensive copies enforcing an immutability nobody documented;
- a slower stable algorithm chosen so a golden file keeps matching;
- a formatting precision frozen by a literal in a test;
- caching or bookkeeping that exists so a call-counting test can observe it.

Each of these is a real cost paid for an imaginary contract.

**Architecture altitude.** Run one pass that ignores individual assertions and
asks whether a whole module, layer, or indirection exists only because tests
imposed it. An auditor working assertion by assertion never sees this.

### Do not claim an unlock the tests never blocked

An audit reads a lot of code and will find ordinary inefficiency along the way:
a value recomputed per call, a helper nobody needed. Report it, in its own
section, and say plainly that no assertion was holding it. A SpecAudit that
takes credit for every improvement it noticed becomes impossible to evaluate,
and the next reader cannot tell which findings the method actually produced.

The test is simple. Name the assertion that would have to change first. If there
is none, it is a normal cleanup, not an unlock.

## The agent roster

A SpecAudit is adversarial by design. The method depends on
**isolation between roles**, not on any particular tool. Nothing below names a
vendor or a model. Read "highest tier" as the strongest reasoning configuration
your agent offers.

Run the waves in order.

**Wave 1, spec cartographers.** Four in parallel, highest tier, read-only. One
per source class: external and machine-checked; published; requested; and the
declared public surface, meaning exported headers, the nanobind bindings, the
generated type stubs, and `__all__` lists. Each returns numbered ledger entries
with citations.

> A cartographer **must not read the tests**. One that does will rediscover the
> tests' assumptions and write them down as promises. That is the exact
> contamination the audit exists to detect, and it silently turns the whole run
> into a rubber stamp.

**Wave 2, census.** One agent, cheap tier. Enumerate every test and assertion in
scope with a stable identifier. No judgment. Cover both C++ test roots: `test/`
and `mlir/unittests/`. A scan of `test/` alone misses hundreds of tests.

**Wave 3, prosecutors.** One per twenty or so tests, in parallel, highest tier.
Each receives the spec ledger, its own cluster, and the code under test. For
every assertion it must cite the ledger entry defended or declare it unanchored
and name the smell. The prosecutor's default is that an assertion is not a
specification until shown otherwise. Prosecutors do not see each other's work.

**Wave 4, provenance.** One per suspect batch, in parallel, middle tier. Runs
the version-history checks above and returns commits, not opinions.

**Wave 5, defenders.** One per suspect batch, in parallel, highest tier. Its
only job is to save the tests: find the ledger entry the prosecutor missed, the
defect the test reproduces, the outside consumer that depends on it.

> A defender **must not see the prosecution's reasoning**, only the assertions
> it accused. Where a second vendor's agent is available, run the defenders and
> the red team there. An agent from the same family as the prosecutor shares its
> blind spots and will agree too easily.

**Wave 6, executors.** Serialized, cheap tier, one at a time. They run the probe
script. Builds and coverage runs contend for the same directories, so never run
two in parallel against one worktree.

**Wave 7, unlock analysts.** In parallel, highest tier: one per confirmed
contract-free verdict, plus exactly one working at architecture altitude.

**Wave 8, red team.** One or two, highest tier. Given the complete draft ledger,
find the verdict that will break a real user. Rank the residual risk of each
surviving verdict.

### Running without subagents

If your agent has no way to spawn subagents, run each wave as a
**separate fresh session**, passing only that wave's inputs. The parallelism is
a speed optimization. The isolation is the method. A single session that plays
every role will anchor on its first reading of the tests and confirm it for the
rest of the run.

## Guardrails

1. Never delete a test that reproduces a reported defect.
2. Narrow before you merge. Merge before you delete.
3. No verdict without an executed experiment.
4. No agent both prosecutes and adjudicates.
5. The audit produces a ledger and stops. It does not apply its own findings.
6. A dirty working tree or an unpinned baseline aborts the run.
7. When a maintainer later applies a verdict, removing the assertion and
   changing the code it constrained belong in **separate commits**, so a
   bisection can tell which one broke.
8. Audit one subsystem at a time. Refuse a whole-repository audit; the suite is
   far too large for the evidence protocol to mean anything at that size.
9. State the coverage consequence plainly. If a deletion moves the project
   number, that is a fact the maintainer needs, not something to hide.

## Scoping an audit

Choose a scope with a boundary somebody can defend: one plugin, one dialect, one
device, one pass. Aim for a scope whose tests you can run in a single command
and whose promises you can enumerate in an afternoon.

Prefer a first scope where the tests run without a full C++ build. Fault
injection then costs seconds instead of minutes, and you will run more of it.

Record the scope as a pair: the source under audit and the tests that cover it.

## Skeleton of a good SpecAudit

Store at `.agent/audits/<scope-slug>.md`. Wrap prose at 80 columns.

```markdown
# SpecAudit: <scope>

<Scope as a source path and a test path. The baseline commit. The date. A
sentence stating that every file:line below was read at that commit.>

## Spec ledger

<S1..Sn: the promise, its rung, and its citation.>

## Summary

| # | Assertion | Class | Remedy | Tier | Unlock | Risk |

## Verdicts

### N. <one-line claim> — <class>

<The assertion, with file:line. The ledger entry it was tested against, or
"none". The experiment that was run and what it showed, quoted. The remedy.
The unlock, if any, naming the lifted constraint and the enabled change.>

## Anchors confirmed

<Assertions that survived fault injection, and the fault each one caught. This
section is why the next auditor does not repeat your work.>

## Deliberately not touched

<Per item: what it is, and why this audit left it alone.>

## Progress

- [x] (YYYY-MM-DD) <what was done>
```

Rank the summary table by **complexity removed per unit of risk**, not by count.
Ten narrowed error-message assertions are worth less than one deleted
abstraction.

## Reconciling

Re-derive an existing SpecAudit against the current default branch rather than
rewriting it. Mark each verdict **applied**, **narrowed**, or **superseded**,
and keep the reason. The record of why something stopped being a problem is the
part that saves the next auditor time.

Record verdicts that something else resolved by accident, and say so. Re-check
partial applications honestly: a change that narrows three of twenty assertions
narrows the finding; it does not close it.

## MQT Core requirements

- Use the build and test entry points `AGENTS.md` documents, so every candidate
  is measured the same way.
- Run `uvx nox -s lint` after each batch of changes to the audit file.
- Wrap Markdown at 80 columns. The formatter reflows prose, and a bare `#123`
  that lands at the start of a line silently becomes a heading. Write pull
  request references in backticks.
- Write plain, direct prose in the style `AGENTS.md` mandates. Short words,
  active voice, no term of art left undefined.
- Do not record local filesystem paths, account names, or branch names. A
  checked-in audit must make sense from any clone.
- Do not edit files whose header says they are generated from an external
  template.

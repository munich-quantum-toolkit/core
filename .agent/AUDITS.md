# Contract audits (SpecAudits)

A SpecAudit checks a concrete concern about a subsystem's tests and the
production code they constrain. Its purpose is to find useful, safe changes, not
to maximize findings or delete tests. Store the result in
`.agent/audits/<scope-slug>.md`. Follow the root `AGENTS.md`.

## Scope and stopping point

Choose a bounded question, source paths, and tests that can answer it. For
example: does an exact output check prevent a simpler serializer while
preserving precision? Start with the suspected cost and the relevant behavior;
an exhaustive assertion census is optional and rarely needed. A broader audit
can be divided into independently useful scopes without requiring a campaign.

Record the baseline commit, any in-scope uncommitted changes, and relevant
dependency versions. Check related issues and pull requests before investigating
overlapping work, and refresh when that scope or baseline changes. Record actual
overlap, not a repository-wide snapshot of unrelated items. Missing access is a
limitation, not evidence that no related work exists.

An audit request authorizes investigation and a report. It does not by itself
authorize applying findings or publishing changes. If the user has also
requested fixes, implement the supported findings within that scope. Group
related changes by behavior and reviewability; neither one PR per assertion nor
a fixed commit split is required.

## Establish the contract

Read the request, relevant specifications, documentation, declarations,
implementation, callers, and tests. Distinguish externally required behavior,
maintainer decisions, compatibility obligations, internal invariants, and
incidental representation choices. Cite the source of each consequential claim.
Resolve conflicting evidence rather than assigning truth by a rigid source
ranking.

Missing documentation does not mean missing intent. Existing tests, verifiers,
interfaces, and downstream uses can expose requirements omitted from prose.
Conversely, a plan or implementation alone does not make every detail permanent.
For MLIR, TableGen is only part of the contract: inspect custom verifiers,
interfaces, pass postconditions, pipeline consumers, and semantic requirements.

Treat these as questions, never automatic findings:

- Exact text, operation order, or operation counts may protect serialization,
  deterministic output, a normal form, or a performance bound.
- Internal tests may isolate a safety invariant or a bug that is expensive to
  reproduce through a public entry point.
- Mock call counts may enforce resource ownership, batching, or network cost.
- A parameter matrix may cover different numerical or backend failure modes.
- A defensive branch may handle external input even when in-tree callers cannot
  construct it. Search public and downstream boundaries before calling it dead.

Authorship, AI assistance, co-introduction of code and tests, and a test edit in
an implementation commit do not establish a defect. Git history helps recover
intent and regressions; `git log -S` cannot tell whether a test has ever failed.

## Investigate candidates

For each candidate, state the proposed change, the behavior that must survive,
and the concrete benefit. Search callers and consumers before claiming that
production code exists only for a test. For exported APIs, absence of in-tree
callers is insufficient evidence of no users.

Use the smallest check that can settle the question. Source reasoning can prove
a duplicate check or establish ownership; a behavioral or performance claim
needs a suitable reproducer or measurement. Preserve regression coverage,
resource safety, input validation, numerical accuracy, and compatibility. Narrow
an assertion only when the replacement still detects the defect it was meant to
catch. For deletion, identify the surviving oracle and justify the relevant
input classes, paths, and failure modes.

Keep experimental edits in a disposable checkout when they would interfere with
user work. Record the baseline result, exact change or self-contained
reproducer, command, exit status, and relevant outcome. Rebuild the affected
target before testing a mutation, verify that the intended tests actually ran,
and restore and recheck the pre-experiment state, including any pre-existing
changes. Serialize experiments that share a checkout or build directory. A
read-only investigation does not require aborting merely because unrelated user
edits exist.

### What an experiment proves

- A behavior-preserving variation that breaks an assertion shows a possible
  representation constraint. First prove that the variation preserves all
  relevant contracts, including downstream postconditions.
- Another test catching one injected defect proves overlap for that defect. It
  does not prove redundancy across other inputs or failure modes.
- A surviving mutation may be equivalent, unreachable, inadequately exercised,
  or evidence of a coverage gap. It never proves that the behavior is unwanted
  or the code removable. A missed observable defect calls for stronger coverage.
- Equal line coverage says nothing about the strength of value assertions.
  Matching results for a finite mutation set support only that set.
- A build failure, collection error, crash before the target assertion, empty
  test selection, or broken result parser cannot be counted as a passing probe.

Prefer direct test exit status and diagnostics. The former `audit-probe.sh`
helper suppressed failures and inferred verdicts from failure counts; it has
been retired. Do not recreate that inference in another wrapper. Coverage and
mutation tools are optional aids, not mandatory stages or verdict generators.

## Report findings people can act on

Put the result first. Use a short ranked list of confirmed findings. Each needs:

1. **Change and benefit:** what to change, where, and why it matters.
2. **Contract and evidence:** the promise retained, source references, and what
   the check actually established.
3. **Risk and limits:** downstream impact, untested cases, and any decision
   needed before implementation.

Use confidence separately from impact. Keep unresolved candidates in a clearly
separate short section with the missing evidence and next useful check. Reject
candidates contradicted by evidence; do not keep them in the actionable count.
“No actionable findings” is a valid result. Avoid catalogs of every passing
assertion or hypothetical improvement.

For a test change, name any production simplification it enables and the
consumer checks needed. It is also valid to conclude that only test clarity
improves or that no safe production change follows. Report unrelated correctness
or performance findings separately; do not credit them to removing an assertion
that never blocked them. Do not force a production deletion to justify an audit.

Review each proposed finding against its strongest counterexample before
handoff. An independent reviewer can help with a disputed or high-risk finding,
but no fixed agent roster, model tier, isolated role ceremony, or multi-agent
workflow is required.

## Maintain the record

Keep the baseline, durable reasoning, minimal reproducible evidence, and latest
known disposition: proposed, accepted, applied, deferred, rejected, or
superseded. Acceptance is not implementation. Link to the implementing change
when known; label historical results and unverified current status explicitly.
Recheck affected findings after code or contracts change, rather than rerunning
an entire audit by default.

Remove role transcripts, assertion censuses that no longer help a decision,
service outages, repeated status snapshots, and superseded recommendations.
Preserve a rejected finding's reason when it prevents a likely repeated mistake.
Use repository-relative paths and stable symbols; line numbers need a baseline.
Keep only enough historical evidence to explain or reproduce the decision.

## Suggested structure

```markdown
# Contract audit: <scope>

Status: <disposition>. Baseline: <commit and any in-scope edits>. Date: <date>.
Source and tests: <repository-relative paths>.

## Result

<Ranked actionable findings, or no actionable findings.>

## Findings

### <Concrete change>

<Contract, source evidence, experiment and result, benefit, risk, disposition.>

## Unresolved questions

<Only material gaps and the next check or decision needed.>

## Validation

<Commands or reproducers, results, and limitations.>
```

# Execution plans (ExecPlans)

An ExecPlan records the goal, current implementation approach, consequential
decisions, and evidence for a complex task. Store one plan per independently
implemented task in `.agent/plans/<task-slug>.md`. Small fixes and documentation
edits do not need a plan unless unresolved design or coordination makes one
useful. Follow the root `AGENTS.md` and any scoped guidance.

## Write for the next contributor

The reader has the repository and the plan, but not the conversation. Explain
what changes for users, which layer owns the behavior, what remains unsupported,
and how to demonstrate success. Name the relevant repository-relative source,
test, and documentation paths. Link to canonical guidance instead of copying it,
and explain only the concepts needed to understand this task.

Before prescribing an implementation, trace the input through its producers and
consumers. Check existing helpers and dependency capabilities. Record a design
choice only when its rationale will help someone maintain or change the result.
A plan is evidence of intent, not authority over a user request, public
contract, or the implementation that ultimately landed.

State assumptions and unresolved decisions explicitly. Resolve routine choices
within the authorized scope. Ask when a consequential ambiguity remains or a
choice would exceed the requested scope or change a supported contract beyond
what the user authorized. A plan does not authorize commits, publication,
external actions, or work in another task's checkout.

## Keep one current account

Update the plan when a milestone completes, a decision changes, or work stops.
Edit the affected sections in place. Do not append a second account of the same
work under progress, discoveries, outcomes, and a revision note.

Keep:

- the supported behavior, exclusions, and ownership boundaries;
- consequential decisions and the reasons for them;
- a rejected approach only when its reason prevents a likely repeated mistake;
- remaining work and real blockers;
- the smallest reproducible validation procedure and its latest relevant result.

Remove:

- timestamped activity logs, review rounds, rebase narratives, and agent
  rosters;
- resolved local setup problems, temporary service outages, and retry
  transcripts;
- obsolete implementation instructions, copied diffs, and repeated test counts;
- machine paths, local environment overrides, ephemeral branches, and temporary
  authorization or user preferences from a past task.

A dependency defect that still constrains the design is worth keeping: record
the affected version or condition, a reproducer or upstream reference, and the
condition for removing the workaround. A transient failure belongs in the task
handoff. If it blocks completion, keep one sentence naming the unverified check
until it is resolved. Never convert blocked, skipped, or queued checks to
passes.

## Active plans and completed records

Start with a status: proposed, in progress, blocked, complete, or superseded.
State what remains for an active plan. Use a short milestone checklist only for
work that is still being tracked; timestamps on every activity are unnecessary.
Each milestone should end in observable behavior and a focused check. Prototype
only a concrete uncertainty, with a criterion for keeping or discarding it.

At completion or before handoff of a completed change, compact the plan into a
decision record: outcome and scope, durable decisions, validation, and remaining
limitations or follow-ups. Remove the implementation recipe and completed
checklist. Git history preserves the sequence of edits. Link to a successor when
an API or design changes; do not leave incompatible designs presented as
current.

Do not infer completion from a checked-in file, a passing intermediate run, or a
PR being opened. When cleaning an older plan, preserve its last recorded outcome
and label historical validation as such. Check the current code before claiming
that an old API still exists or a follow-up was resolved. Reconcile remote
status only when it matters to the task; do not turn document cleanup into CI
monitoring.

## Validation

Use repository build and test entry points. Record the focused command, the
behavior it checks, and the result; link to the root guide for routine setup and
lint. Keep an exact revision or dependency version when needed to interpret an
experiment. Distinguish local validation from hosted CI and from unrun checks.
Do not copy every successful rerun. Broaden validation only for an affected
contract or a required repository check.

Write commands from the repository root. Include recovery instructions only for
steps that can lose data or leave a migration incomplete. An ordinary rebuild
does not need its own recovery section.

## Suggested structure

Use ordinary Markdown. Omit sections that have nothing useful to say and combine
ones that would repeat each other.

```markdown
# <Task>

Status: <state; remaining work or successor if relevant>.

## Goal and scope

<User outcome, supported boundary, exclusions, and source/test entry points.>

## Decisions

<Choice, reason, and evidence. Distinguish accepted decisions from proposals.>

## Work remaining

- [ ] <Milestone with observable acceptance; active plans only.>

## Validation

<Focused commands and expected behavior; latest results with any limits.>

## Outcome

<Delivered behavior and unresolved follow-ups; compact at completion.>
```

# Codex Execution Plans (ExecPlans)

This document describes the requirements for an execution plan ("ExecPlan"), a
design document that a coding agent can follow to deliver a working feature or
system change. Treat the reader as a complete beginner to this repository: they
have only the current working tree and the single ExecPlan file you provide.
There is no memory of prior plans and no external context.

## How to use ExecPlans and PLANS.md

When authoring an executable specification (ExecPlan), follow PLANS.md
_to the letter_. If it is not in your context, refresh your memory by reading
the entire PLANS.md file. Be thorough in reading (and re-reading) source
material to produce an accurate specification. When creating a spec, start from
the skeleton and flesh it out as you do your research.

When implementing an executable specification (ExecPlan), do not prompt the user
for "next steps"; simply proceed to the next milestone. Keep all sections up to
date, add or split entries in the list at every stopping point to affirmatively
state the progress made and next steps. Resolve ambiguities autonomously, and
commit frequently.

When discussing an executable specification (ExecPlan), record decisions in a
log in the spec for posterity; it should be unambiguously clear why any change
to the specification was made. ExecPlans are living documents, and it should
always be possible to restart from _only_ the ExecPlan and no other work.

When researching a design with challenging requirements or significant unknowns,
use milestones to implement proof of concepts, "toy implementations", and
similar work that validates whether the proposal is feasible. Read relevant
library source code, research deeply, and include prototypes that guide a fuller
implementation.

## Requirements

NON-NEGOTIABLE REQUIREMENTS:

- Every ExecPlan must be fully self-contained. Self-contained means that in its
  current form it contains all knowledge and instructions needed for a novice to
  succeed.
- Every ExecPlan is a living document. Contributors are required to revise it as
  progress is made, discoveries occur, and design decisions are finalized. Each
  revision must remain fully self-contained.
- Every ExecPlan must enable a complete novice to implement the feature
  end-to-end without prior knowledge of this repository.
- Every ExecPlan must produce demonstrably working behavior, not merely code
  changes that meet a definition.
- Every ExecPlan must define every term of art in plain language or not use it.

Purpose and intent come first. Begin by explaining, in a few sentences, why the
work matters from a user's perspective: what someone can do after this change
that they could not do before, and how to see it working. Then guide the reader
through the exact steps to achieve that outcome, including what to edit, what to
run, and what they should observe.

The agent executing the plan can list files, read files, search, run the
project, and run tests. It does not know prior context and cannot infer intent
from earlier milestones. Repeat every assumption the plan relies on. Do not
point to external blogs or documentation; embed required knowledge in the plan
in your own words. If an ExecPlan builds on a checked-in prior ExecPlan,
incorporate it by reference. Otherwise, include all relevant context from that
plan.

## Formatting

Format and envelope are simple and strict. Each ExecPlan must be one single
fenced code block labeled `md` that begins and ends with triple backticks. Do
not nest additional triple-backtick code fences inside it. Present commands,
transcripts, diffs, and code as indented text within that single fence instead.
Use indentation for clarity rather than code fences that could prematurely close
the ExecPlan.

Use two newlines after every heading, use `#`, `##`, and deeper headings
correctly, and use correct ordered and unordered list syntax. When writing an
ExecPlan to a Markdown file whose content is only that single ExecPlan, omit the
outer triple backticks.

Write in plain prose. Prefer sentences over lists. Avoid checklists, tables, and
long enumerations unless brevity would obscure meaning. Checklists are permitted
only in the `Progress` section, where they are mandatory. Narrative sections
must remain prose-first.

## Guidelines

Self-containment and plain language are paramount. If you introduce a phrase
that is not ordinary English, define it immediately and explain how it appears
in this repository by naming the files or commands where it occurs. Do not say
"as defined previously" or "according to the architecture document." Include the
needed explanation in the ExecPlan, even if that repeats information.

Avoid common failure modes. Do not rely on undefined jargon. Do not describe a
feature so narrowly that the resulting code compiles but does nothing
meaningful. Do not outsource key decisions to the reader. When ambiguity exists,
resolve it in the plan and explain why. Err on the side of over-explaining
user-visible effects and under-specifying incidental implementation details.

Anchor the plan with observable outcomes. State what the user can do after
implementation, the commands to run, and the outputs they should see. Acceptance
should be phrased as behavior a human can verify rather than internal
attributes. If a change is internal, explain how to demonstrate its impact, for
example with a test that fails before the change and passes after it or through
a small end-to-end scenario.

Specify repository context explicitly. Name files with full repository-relative
paths, functions and modules precisely, and where new files belong. If touching
multiple areas, include a short orientation paragraph explaining how those parts
fit together. When running commands, show the working directory and exact
command line. State environment assumptions and reasonable alternatives.

Be idempotent and safe. Write steps that can be repeated without damage or
drift. If a step can fail halfway, include how to retry or adapt. If a migration
or destructive operation is necessary, spell out backups or safe fallbacks.
Prefer additive, testable changes that can be validated as work proceeds.

Validation is not optional. Include instructions to run tests, start the system
when applicable, and observe useful behavior. Describe comprehensive tests for
new capabilities. Include expected outputs and error messages so a novice can
distinguish success from failure. Where possible, prove the change beyond
compilation through an end-to-end scenario, CLI invocation, or equivalent. State
the project's exact test commands and how to interpret their results.

Capture evidence. Put concise, focused transcripts, diffs, and logs inside the
ExecPlan when they prove success. Prefer file-scoped diffs or small excerpts to
large patches.

## Milestones

Milestones are narrative, not bureaucracy. Introduce each milestone with a short
paragraph describing its scope, what will exist at its end, the commands to run,
and the acceptance to observe. Keep the sequence readable as a story: goal,
work, result, proof. Progress and milestones are distinct: milestones tell the
story, while progress tracks granular work. Both must exist.

Never abbreviate a milestone merely for brevity or omit details crucial to a
future implementation. Each milestone must be independently verifiable and
incrementally implement the overall goal.

## Living plans and design decisions

- ExecPlans are living documents. Record each key design decision and its
  rationale in the `Decision Log`.
- Every ExecPlan must contain and maintain `Progress`,
  `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective`
  sections.
- Capture unexpected behavior, performance tradeoffs, and bugs that shape the
  approach in `Surprises & Discoveries`, with short evidence snippets such as
  test output.
- If the implementation changes course, document why in the `Decision Log` and
  reflect the implications in `Progress`.
- At the end of a major task or the complete plan, add an
  `Outcomes & Retrospective` entry that records what was achieved, what remains,
  and lessons learned.

## Prototyping milestones and parallel implementations

Explicit prototyping milestones are encouraged when they de-risk a larger
change. Keep prototypes additive and testable. Label them as prototyping,
describe how to run and observe them, and state the criteria for promotion or
discarding.

Prefer additive changes followed by safe subtractions that keep tests passing.
Parallel implementations, such as an adapter alongside an older path during a
migration, are appropriate when they reduce risk or keep tests running. Explain
how to validate both paths and retire one safely. When several libraries or
feature areas are involved, consider independent spikes that prove each external
dependency has the required behavior in isolation.

## MQT Core requirements

In addition to the requirements above, an MQT Core ExecPlan must state the task
worktree and branch it owns, preserve unrelated user changes, and never modify
another task's worktree. It must follow the repository's `AGENTS.md` and
`docs/ai_usage.md`, including the rules for generated files, secrets, AI
disclosure, authorization, and human review. An ExecPlan does not itself
authorize external GitHub actions.

Use the exact Core commands relevant to the change. For example, name the
applicable CMake preset and focused C++ or MLIR test binary; for Python changes,
name the relevant `pytest` or Nox command; for documentation, include the Nox
documentation command. End with `uvx nox -s lint` unless an existing, documented
limitation prevents it, and record the limitation and its evidence.

## Skeleton of a good ExecPlan

## <Short, action-oriented description>

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

### Purpose / Big Picture

Explain in a few sentences what someone gains after this change and how they can
see it working. State the user-visible behavior to enable.

### Progress

Use a checkbox list to summarize granular steps. Document every stopping point,
including partially completed work split into completed and remaining parts.
This section must always reflect the current state.

- [x] (2026-01-01 00:00Z) Example completed step.
- [ ] Example incomplete step.
- [ ] Example partially completed step (completed: X; remaining: Y).

Use timestamps to measure the rate of progress.

### Surprises & Discoveries

Document unexpected behavior, bugs, optimizations, or insights discovered during
implementation. Provide concise evidence.

- Observation: … Evidence: …

### Decision Log

Record every decision made while working on the plan in this form:

- Decision: … Rationale: … Date/Author: …

### Outcomes & Retrospective

Summarize outcomes, gaps, and lessons learned at major milestones or completion.
Compare the result against the original purpose.

### Context and Orientation

Describe the relevant current state as if the reader knows nothing. Name key
files and modules by full path. Define non-obvious terms. Do not refer to prior
plans.

### Plan of Work

Describe the sequence of edits and additions in prose. For each edit, name the
file, location, and exact change. Keep it concrete and minimal.

### Concrete Steps

State the exact commands and working directories. When a command generates
output, show a short expected transcript. Update this section as work proceeds.

### Validation and Acceptance

Describe how to exercise the system and what to observe. Phrase acceptance as
behavior with specific inputs and outputs. Name the relevant Core test commands
and what success looks like.

### Idempotence and Recovery

State which steps are repeatable. For risky steps, provide a safe retry or
rollback path. Keep the environment clean after completion.

### Artifacts and Notes

Include the most important transcripts, diffs, or snippets as indented examples.
Keep them concise and focused on evidence.

### Interfaces and Dependencies

Be prescriptive. Name the libraries, modules, and services to use and why.
Specify the types, interfaces, and function signatures that must exist at the
end of the milestone. Prefer stable names and repository-relative paths.

When you revise an ExecPlan, ensure the change is reflected across all sections,
including the living-document sections, and add a note at the end describing
what changed and why. An ExecPlan must describe not only what to do, but why.

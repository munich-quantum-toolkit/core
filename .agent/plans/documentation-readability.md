# Documentation readability and terminology

Status: complete. Baseline: `2af9b2df72af1596efb7ef3951200631371663f5`.

## Scope

Prioritize technical accuracy, reader understanding, and natural flow. Retain
Orwell's writing rules throughout; limit STE guidance to implementation
comments, API summaries, and concise reference fields. Detailed API descriptions
and guides use connected prose. Glossary entries need a reader purpose and must
distinguish general terminology from MQT-specific behavior.

Reworked 27 documentation pages and connected compiler, binding, plugin, and
MLIR descriptions. Removed implementation and review narration from user
workflows, organized explanations around reader tasks, and corrected
source-verified stale claims. Preserved executable examples, public interfaces,
and the historical purpose of release notes and migration records.

The [terminology audit](../audits/documentation-terminology-review.md) records
all 12 findings, source evidence, applied decisions, validation, and limits.
Definitions belong beside their owning APIs when they describe Core-specific
objects, schemas, or map conventions. General scientific definitions cite
primary sources and distinguish benchmark implementations.

## Validation

Stub generation, the strict documentation build with all 18 executable
notebooks, generated-site link checks, repository lint, and structural
comparisons passed. Chrome DevTools inspection covered the rebuilt glossary,
prose policy, and target guide. The audit records additional probes and
validation limits.

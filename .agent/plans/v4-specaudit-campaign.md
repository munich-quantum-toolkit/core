# MQT Core v4 contract-audit campaign

Status: historical proposal, only partly completed in the checked-in record. The
mandatory eight-wave process is superseded by [AUDITS.md](../AUDITS.md). This
record does not schedule or authorize further audits.

## Goal and recorded outcome

The campaign aimed to find tests that constrain incidental choices and trace
whether changing those tests could simplify production code or improve its
performance. It proposed small subsystem audits across Core and the compiler
collection.

The global-phase pilot ran against `cb5cf0103bd9841726c8ec6c5abb725758afea58` on
2026-08-19. The maintainer accepted findings 1-6, deferred 7, and retained 8-9
on 2026-08-20. See the
[global-phase record](../audits/global-phase-normalization.md) for the actual
reasoning, evidence, and limits. The checked-in campaign does not establish
completion of the remaining scopes. The separately recorded
[PennyLane audit](../audits/pennylane-plugin.md) preserves its own baseline and
applied decisions.

## Lessons retained

- Trace the relevant producer, implementation, tests, and consumers together.
  Reading TableGen or a plan in isolation misses pipeline and downstream
  requirements.
- Challenge a suspected representation constraint with a behavior-preserving
  alternative, then establish that all relevant consumers still accept it. A
  changed assertion alone does not prove that the proposed change is safe.
- A missed injected defect can reveal a weak oracle. It does not establish that
  the behavior or its implementation is unnecessary.
- Test changes may improve clarity without enabling production deletion. State
  that outcome honestly; investigate production ownership without forcing a
  deletion to justify the audit.
- Refresh relevant issue and PR overlap when selecting or changing a scope.
  Repeated repository-wide snapshots and role-isolation administration added
  noise without strengthening individual findings.

## Candidate scope map

These are historical candidate areas, not a current backlog. Some APIs and
modules have since moved or been removed. Check current source ownership and
related work before selecting a new audit.

| Area              | Useful boundaries                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| Core IR           | Operation semantics, circuit containers, symbolic expressions, classical parser and serializer |
| Decision diagrams | Numeric tables, package/storage, simulation, Python ownership                                  |
| Compiler dialects | CBit, QC/QCO builders and modifiers, QTensor aliasing and iterators                            |
| Transformations   | One pass or consumer contract at a time: fusion, synthesis, mapping, or layout                 |
| Conversions       | Each direction separately; assign one owner to round-trip assertions                           |
| External formats  | OpenQASM parsing/semantics/emission, Qiskit import/export, QIR profiles, jeff                  |
| Integration       | Program ownership, default and target pipelines, target model, QDMI adapter                    |
| Devices           | Registry and session lifetime, jobs, provider models, Slurm                                    |

Core tests live under `test/`; MLIR tests live under `mlir/unittests/`, with
compiler-facing Python coverage under `test/python/`. Shared fixtures are inputs
to an audit, not independent assertion scopes. Downstream consumers matter when
considering exported API or representation changes.

## Resuming a scope

Start with a concrete concern and the smallest runnable check. Pin the relevant
baseline, check actual overlapping work, and follow the current audit guide.
Keep unresolved candidates separate from findings. Do not recreate the old
assertion census, role roster, or historical PR blocking list as prerequisites.

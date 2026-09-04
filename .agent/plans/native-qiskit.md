# Native Qiskit primitives

Status: in progress; validate and publish DDSIM Sampler integration after the
separate DDSIM ordered-shots change.

## Goal and scope

Replace the QDMI-specific primitives with Qiskit's `BackendSamplerV2` and
`BackendEstimatorV2`. Keep QDMI v1.3.3 compatibility and provider submission
single-threaded. The backend submits a validated batch before collecting it.

## Decisions

- Native Qiskit owns PUB grouping, broadcasting, statistics, and primitive jobs.
  Backend factories forward typed keyword options without a compatibility layer;
  Qiskit retains ownership of defaults and validation.
- Memory requires genuine `SHOTS`; counts for memory execution come from those
  same shots. Counts-only execution requires histogram results. Missing or
  malformed requested results raise instead of becoming zero samples.
- Snapshot classical-register headers, defer remote IDs, cache completed
  results, and cancel submitted jobs on failure. Query formats once per batch.
- Use Qiskit 2.1 or newer: it includes native Sampler run options and removes
  the mandatory SymEngine dependency that prevents minimum-version installation
  on Python 3.14. No compatibility branches are needed.

## Validation

`uv run --no-sync pytest test/python/plugins/qiskit test/python/qdmi` passes 414
checks with Qiskit 2.5.2. The same suites pass with Qiskit 2.1.0 (413 passed,
one compiler-translation check skipped because it requires Qiskit 2.5). Lint and
documentation builds pass locally.

Five offline recording runs of 400 circuits at 100 shots show native Sampler
submits all 400 jobs before collection, compared with one for the custom
Sampler. Native Estimator needs 400 jobs for four commuting Pauli terms,
compared with 1,600. These are scheduling checks, not remote speed measurements.

Three SV1 comparisons using identical converted programs give median times of
31.79 seconds for native Sampler through QDMI and 40.51 seconds for the Braket
SDK (1.27× faster). Native Estimator completes 400 four-term commuting PUBs in
22.04 seconds using 400 tasks. All results pass program-order and shot-total
checks; native samples and histograms match their stored results. Independent
auditing is excluded from execution timings.

These measurements use the existing optimized Braket provider and its
`AmazonBraketBackend` OpenQASM adapter; the gain is not attributable to Core
alone. Preparation takes 0.125 seconds for 400 circuits. No additional Core
scheduler or speculative cache is justified. Mixed shot/precision groups retain
native scheduling, and shared binding GIL work remains in its existing change.

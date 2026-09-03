# Measure and simplify the PennyLane-to-SV1 path

This ExecPlan is a living document maintained according to `.agent/PLANS.md`.
Keep Progress, Surprises & Discoveries, Decision Log, and Outcomes &
Retrospective current. It coordinates Core with the separately checked-out
Amazon Braket QDMI Device repository; never modify another task's checkout.

## Purpose / Big Picture

Run 400 distinct four-qubit circuits with 100 shots through one synchronous
PennyLane execution call, efficiently using Amazon Braket SV1. Improve measured
overhead without changing the QDMI 1.3.3 interface, introducing a generic
provider thread pool, or changing sampling semantics. Compare identical OpenQASM
payloads with both ordinary and tuned Amazon Braket SDK execution. The user
authorized up to USD 100 for staged investigation; conservatively include
earlier estimated spending of USD 7.5825 in that limit. Only SV1 in us-east-1 is
authorized.

## Progress

- [x] (2026-09-03) Inspect the existing concurrent Core and provider changes and
      trace conversion, submission, waiting, and result retrieval through both
      layers.
- [x] (2026-09-03) Confirm provider uses unchanged QDMI 1.3.3 and identify
      repeated Python topology reconstruction and SDK per-download S3 resource
      creation.
- [x] (2026-09-03) Repeat eight-worker provider measurements: 21.82, 23.47, and
      21.14 seconds. Compare tuned SDK workers and shared-client diagnostics.
- [x] (2026-09-03) Cache successful gate/location validation, preserve original
      errors during cleanup, and release the GIL for shot/count retrieval in
      Core.
- [x] (2026-09-03) Replace the provider's prefetch future with its lifetime
      gate; requeue unfinished polls instead of retaining workers until remote
      completion.
- [x] (2026-09-03) Pass 130 provider native tests, 130 ASan/UBSan tests, 125
  repeated lifecycle/concurrency checks, and Core's full C++ lint session.
- [x] (2026-09-03) Finish SDK comparisons. Tuned unmodified SDK median: 35.00 s;
  shared-S3 diagnostics: 29.46 s at 32 workers and 35.10 s at 64 workers.
- [x] (2026-09-03) Pass all eight Core and all eight provider Python test and
  minimum-dependency sessions, provider documentation/lint, and Core stubs.
- [x] (2026-09-03) Pass Core documentation; generated stubs have no API changes.
- [x] (2026-09-03) Pass final Core lint. Sign and verify the three Core code
  commits and all six provider commits; keep documentation separate.
- [x] (2026-09-03) Record measured results, limitations, and cost in the report
  and PR descriptions prepared for the final handoff.

## Surprises & Discoveries

The prior provider experiment completed in 21.8 seconds versus 89–96 seconds for
SDK batches, but the SDK used 100 workers and had long S3 request tails. That is
not sufficient evidence of a general advantage over a tuned SDK. Core conversion
takes approximately 0.67 seconds for 400 tapes, mostly rebuilding advertised
topology sets per gate. The C++ QDMI driver simply forwards job calls and adds
no job-wide scheduling lock. Provider prefetch already uses independent bounded
submission and result pools; queued callbacks have a cancellation gate so
freeing a queued job does not wait behind unrelated remote jobs.

Sixteen submission workers exposed 161 HTTP 429 responses that the bundled C++
SDK did not retry. Its existing AWS_NEW_RETRIES_2026 opt-in fixes the error
classification without custom retry code: a subsequent 400-task run recovered
104 retries and finished in 33.17 seconds. It did not beat eight workers. A
100-worker SDK run also exhausted retries; all 276 created tasks were
reconciled.

The failed provider experiment exposed Core cleanup masking the original error:
canceling a terminal job raises ValueError, which the old RuntimeError-only
suppression did not catch. Regression coverage now includes RuntimeError,
ValueError, and KeyboardInterrupt during cancellation.

The cached Python test wheels omitted the SC provider because the stubs session
builds with it disabled. Reinstalling the package explicitly with
BUILD_MQT_CORE_QDMI_SC_DEVICE=ON restores full-suite validation; this is a local
build-cache issue, not a production-code fix in this task.

## Decision Log

Prefer fewer requests, shared clients, and immutable session metadata over more
workers. Retain all validation and cancellation/lifetime safety. Date:
2026-09-03. Keep the existing Core PR's main-then-backport order; no merge or
release is authorized. Keep provider changes in its existing single PR with
distinct signed commits. This plan does not authorize additional GitHub actions.

Keep eight submission workers. Use the SDK's documented retry opt-in in the
application environment rather than mutate process-wide AWS settings or add a
custom retry strategy. Date: 2026-09-03.

Cache successful checks keyed by PennyLane gate name and device-wire tuple,
owned by each converter. This avoids another metadata representation and still
validates every gate's shape, wires, and parameters. Preparation dropped from
0.67 seconds to 0.06 seconds for 400 circuits. Date: 2026-09-03.

Keep the thin C++ QDMI client/driver forwarding layers unchanged. Release the
GIL declaratively in sample/count bindings because result retrieval can block on
S3 after wait returns; do not add a duration-dependent concurrency test. Date:
2026-09-03.

## Context and Orientation

`python/mqt/core/plugins/pennylane/converter.py` translates tapes (PennyLane's
circuit records) to OpenQASM and checks device-supported gate locations.
`python/mqt/core/plugins/pennylane/device.py` prepares the entire batch, submits
every job, then returns samples in input order. `bindings/qdmi/qdmi.cpp`
connects Python to `src/qdmi/Client.cpp`, which calls the C driver in
`src/qdmi/driver/Driver.cpp`. In the provider repository, `src/Device.cpp` and
`include/amazon-braket-qdmi-device/Device.hpp` implement metadata caching,
asynchronous HTTP submission, background status polling, and S3 result
retrieval. The C API still submits one program per job and callers still
synchronously wait for PennyLane results. No QDMI interface addition is needed.

## Plan of Work

First reproduce the current provider and SDK with the same deterministic
programs. Tune only existing SDK worker and poll settings, then separately
measure a diagnostic shared-S3-client variant; never label that variant
unmodified SDK. Profile conversion without creating quantum tasks. Measure
result decoding on already completed tasks where possible. Record request
counts, retries, wall time, CPU time, ordered payload/sample validation, and
cost in ignored benchmark artifacts under `build/benchmarks/`.

Only then change production code. A session-local cache in `_ProgramConverter`
may eliminate repeated topology reconstruction without caching mutable global
device metadata. Trace all callers before changing shared C++ helpers. Do not
add a second scheduler unless a measured workload justifies it. Each behavioral
change needs an existing-suite regression test, and independent deletion-only
cleanup belongs in its own commit.

## Concrete Steps

From the Core repository root run focused Python tests using the existing test
environment, followed by `uvx nox -s tests`, `uvx nox -s minimums`, and
`uvx nox -s lint`. If bindings change, run `uvx nox -s stubs`; if any Core C++
changes, run `uvx nox -s cpp-lint` with the normal CMake lint preset. From the
provider repository root build with `cmake --build build -j 6`, then run
`ctest -C Release --test-dir build --output-on-failure --timeout 30` only with
live tests disabled. Run `uvx nox -s tests minimums docs` and `uvx nox -s lint`
for provider changes. Read each repository's AGENTS.md first.

## Validation and Acceptance

Offline tests must preserve rejected invalid gates, directed operation loci,
undirected fallback topology, ordered results, shape, shot totals, async errors,
pending cancellation, and safe job/session destruction. Live stages must
validate 400 distinct payloads, samples shaped (100, 4), binary values, and 100
total shots. Compare distributions of repeated wall-clock runs, not unrelated
random sample equality. Faster-than-SDK is an aspiration, not a correctness
assertion.

## Idempotence and Recovery

Live benchmarks are not idempotent: use unique run labels, persist reservations
before submitting, record task identifiers privately for reconciliation, and
disable automatic failed-task resubmission. Verify SSO authentication before a
live stage and reserve a safety margin for S3 and uncertain simulator time.
Never print credentials, account identifiers, private ARNs, or bucket names. On
failure, account for already created tasks before another run. Do not force-push
or discard user changes. Verify every signed commit before pushing.

## Artifacts and Notes

Existing measurement: original QDMI 333.7 s; asynchronous submission alone 151.8
s; submission plus result prefetch 21.8 s. These are historical single provider
runs, not new controlled medians. QDMI 1.3.3 is pinned to upstream revision
18cfb67fd9042761d3005c2f8655751c1758f9c5 in the provider build.

## Interfaces and Dependencies

Keep the existing QDMI C ABI and Core Python methods unchanged. Reuse Python's
standard library, NumPy, and the installed AWS C++ SDK. Core caches must be
owned by the converter/session, not retain sessions through a global cache.

## Outcomes & Retrospective

The provider's final three-run median is 21.82 seconds (21.14–23.47), compared
with 35.00 seconds (31.73–40.56) for the tuned, unmodified 32-worker SDK: about
1.6 times faster on this workload. SDK diagnostics with a shared S3 client were
also measured separately; they are not stock-SDK results. This is a cloud
workload comparison with service/network variability, not a universal speedup.

The later-completed-job prefetch regression fails before requeuing and passes
afterward. Separate commits preserve lifetime simplification versus scheduling
behavior. Keep eight workers, shared native clients, cached terminal results,
and native SDK retries; additional workers increased throttling. Python sample
decoding took about 0.1 seconds in profiling and does not justify a rewrite.

All 7,737 created tasks are accounted for, including partially failed
experiments. Estimated cumulative SV1 compute is USD 29.01375 before credits and
separately unmetered S3 costs, within the USD 100 allowance. No more paid runs
are needed. Local implementation and validation are complete. Delivery uses
provider PR #205 and Core PR #2349 with signed commits; check the final remote
heads when publishing and report hosted CI separately. Core review/merge and the
subsequent v3.x backport remain outside this implementation stage.

Revision note: updated with measured preparation savings, failed high-worker
experiments, native retry configuration, cleanup correctness, prefetch fairness,
completed sanitizer/native/Python validation, final SDK comparisons, and cost.

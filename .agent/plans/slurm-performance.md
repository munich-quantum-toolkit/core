# Slurm performance and test reliability

Status: complete; implemented and validated locally. Baseline:
`552feffdbf5e2396660bb3e91a21936a9d7adf63`.

## Outcome and scope

The static-license adapter opens the selected registered device without copying
the full catalog. It preserves fresh sessions, IDLE/BUSY acceptance, and input
errors. Selection remains separate from provider authorization. Availability
synchronization and optional SPANK/checker work in #2310-#2313/#2366 remain
separate workstreams.

The real Slurm fixture bounds commands, checks terminal job states and exit
codes, and isolates each invocation's Docker resources and artifacts. Fourteen
fast runner tests cover failures before CI builds a wheel. CI uses the existing
compiler-cache integration; Docker installs the wheel without retaining its
archive in an image layer.

## Decisions and evidence

Reuse `Session::openDevice` rather than adding an adapter lookup or cache. Keep
provider sessions and status fresh; callers can reuse an opened device for
multiple quantum jobs. Provider calls retain provider-owned timeout settings.

Keep the two-node contention and independent-device execution tests. Process
group termination handles Compose children; unique projects prevent cross-run
cleanup. Controller polling belongs only to this disposable test fixture.

The controlled local compiler-cache test reduced a clean wheel build from 63.27
to 41.90 seconds with 62 hits from 67 requests. Image size fell from 516 to 450
MB. Native and Python adapter tests, the runner tests, and real Slurm tests
passed, including simultaneous independent runs. Hosted cache behavior remains
unverified. Full contract evidence and measurement limits are in the
[Slurm audit](../audits/slurm-performance-2026-09-08.md).

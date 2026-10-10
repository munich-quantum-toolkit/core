# Native cleanup and test runtime

Status: implementation complete; Windows timeout diagnosis remains open.

## Outcome

The independent commits preserve Core's throwing APIs: DD imaginary-unit
parsing, positive allocation capacities, measurement-fidelity boundaries,
serialized-file write errors, benchmark manifest reuse, typed gate parameters,
removal of an unused adapter argument, QIR lifecycle binding and output capture,
cheaper Shor verification, and accurate benchmark API documentation and isolated
documentation output. Result types and diagnostic adapters remain in PR #2545.
Most tests previously deleted from that PR were never on main.

## CI evidence

The
[reported Windows job](https://github.com/munich-quantum-toolkit/core/actions/runs/38073032802/job/114274242213)
failed on main because one sampling test reached 600 seconds. Successful main
and #2545 Windows suites took about 77–78 seconds. The hang needs a reproduction
or host and worker stacks; changing timeouts does not establish a fix.

The Shor test queries the functionality DD directly and retains every input,
plus the separate coherence, inverse, and workspace checks. Matched GCC 13.3
Release binaries on Linux arm64, pinned to CPU 0, gave medians of 13.770 seconds
before and 13.053 seconds after across three runs each. Ranges overlap on this
shared host, so this does not establish a reliable CI speedup.

## Validation

The release suite ran 4,014 tests with one existing skip. The affected DD and
DDSIM targets were rebuilt and rerun after final edits. The 455 focused Python
tests and executable documentation passed. Repository and whole-file C++ lint
passed after the reported test-style findings were fixed and all affected files
were rechecked. Windows and macOS execution remain unverified locally; the
failed-write regression uses `/dev/full` where available.

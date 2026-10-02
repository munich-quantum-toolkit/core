# QDMI follow-up fixes

Status: complete; local validation passed. Hosted CI and human review remain
separate acceptance steps.

## Goal and scope

Implement the accepted follow-up audit findings: preserve Qiskit placements and
physical durations, deduplicate loaded QDMI providers, validate SC names, reuse
parser topology sets, and consolidate Python tests while preserving binding
coverage. Extract independently useful mainline fixes from the deferred driver
and staging work in pull requests 2229, 2230, and 2231.

## Decisions

Keep QDMI 1.3 APIs and packaging boundaries. Provider identity includes both the
loaded module and symbol prefix. Retain providers for the driver lifetime to
prevent finalization racing with a new session. Preserve global and explicit
Qiskit placements, subclass hooks, and raw duration semantics in the C++ client;
conversion to seconds belongs in the Qiskit adapter.

Python boundary tests remain in Python. Native semantic duplicates use their
existing C++ oracles; isolate registry smoke tests without adding a production
reset API.

## Deferred PR extractions

From #2230, retain loaded providers for process lifetime, accept warning
statuses through session setup, reject null session and child handles, and
release allocated sessions on setup failure. From #2229, contain
configuration-loading exceptions at `QDMI_session_alloc` and distinguish valid
unsupported custom enums from invalid enum values. These fixes do not require
the replaceable-driver ABI.

No independent extraction from #2231 was justified: its shared-client and
runtime-staging changes depend on the deferred driver split. Windows packaging
and the remaining work in all three PRs stay deferred.

## Validation

Implementation base: `d0a2f7e71`, which adds two unrelated mainline
QIR/control-flow fixes after the audit baseline. The QDMI sources were unchanged
between those revisions.

- Rebuilt the native driver and SC tests in `build/cpp-lint`: 103 driver tests,
  two diagnostic tests, and 43 SC tests passed; one existing SC job-ID test is
  skipped. Another 240 client, 14 registry, and 63 DDSIM tests passed. The 10
  QDMI staging/configuration CTest checks passed.
- Rebuilt the Python package and ran
  `uv run --no-sync pytest -n0 test/python/qdmi test/python/plugins -q`: 468
  tests passed, including QDMI registry tests before provider enumeration.
- `uvx nox -s lint` passed. `uvx nox -s cpp-lint` checked all six changed C++
  source files against `origin/main`, with zero findings. Driver and diagnostic
  tests passed again after the lint fixes.

Tests cover provider aliases and lifetime, session warnings and malformed
handles, custom enums, invalid configuration, embedded NUL names, ordered
three-qubit placements, calibration units, and Python binding isolation.

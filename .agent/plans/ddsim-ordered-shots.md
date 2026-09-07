# Ordered DDSIM shots

Status: complete.

## Scope and decisions

Expose genuine ordered samples through `QDMI_JOB_RESULT_SHOTS` for DDSIM's
OpenQASM and QIR sampling paths. Preserve the existing histogram and state
extraction contracts. No QDMI interface or Python binding changes are needed.

The QCO sampler already encodes each outcome with the returned classical
register mapping. An optional output sequence retains those outcomes before
aggregation; callers that only need counts allocate no sequence. Terminal
measurements still simulate once, while dynamic programs execute once per shot.
The QIR runtime supplies each shot's output directly.

## Validation

The release build and `ctest --preset release` passed all 3,870 registered
tests, with one existing SC-device skip. The DDSIM device binary passed 63
tests; QCO sampling passed 14 tests. Python QDMI passed 251 tests using the
built device, including repeated reads and empty shot strings. These checks
cover terminal and dynamic measurements, QIR Base/Adaptive, classical mapping,
histogram consistency, reproducible ordering, and buffer/state errors.

`uvx nox -s lint`, `uvx nox -s cpp-lint`, and the documentation build passed.
The device builds against QDMI 1.3.3. Qiskit native Sampler integration is
covered by its separate plugin change.

# Ordered DDSIM shots

Status: in progress.

## Scope and decisions

Expose genuine ordered samples through `QDMI_JOB_RESULT_SHOTS` for DDSIM's
OpenQASM and QIR sampling paths. Preserve the existing histogram and state
extraction contracts. No QDMI interface or Python binding changes are needed.

The QCO sampler already encodes each outcome with the returned classical
register mapping. An optional output sequence retains those outcomes before
aggregation; callers that only need counts allocate no sequence. Terminal
measurements still simulate once, while dynamic programs execute once per shot.
The QIR runtime supplies each shot's output directly.

## Work remaining

- [ ] Retain and expose samples in both execution paths.
- [ ] Check ordering, mapping, dynamic measurements, counts consistency, and
      QDMI buffer and state contracts.
- [ ] Build, run focused tests, and run required lint checks.

## Validation

Use the release CMake preset and DDSIM device test binary. Qiskit native
Sampler integration is covered by its separate plugin change.

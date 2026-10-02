# Quantum program inspection

Status: complete; combined inspection APIs validated on current main.

## Scope and decisions

QC gate histograms use the existing entry-point gate count semantics: visit
static IR once, skip barriers and modifier bodies, and do not expand calls.
Static depth follows quantum resource dependencies, joins alternative SCF
branches by maximum, and visits each loop region once. Return unknown for
unresolved references or unsupported control flow rather than undercount.

QC and QCO resource inspection reports total declared allocated width, distinct
physical site IDs, and control-flow presence. Unknown widths remain unknown;
site IDs are not dense wire counts. Include helper declarations but exclude
nested module scopes. These opt-in queries do not change compilation.

Parameter binding and circuit construction are separate features.

## Validation

The compiler unit binary passes 258 tests. The Python compiler and Qiskit
translation/target suites pass 591 tests. Generated stubs, full executable
documentation with local link checks, repository lint, and whole-changed-file
C++ lint pass. Adapter adoption checks pass without changing its dependency pin.

Gate histograms and depth remain QC APIs. QC and QCO share resource inspection.
Static depth returns unknown for register views or stored quantum references,
unstructured control flow, and nesting of 128 regions or more. It does not
estimate executed depth, classical latency, or filtered two-qubit depth.

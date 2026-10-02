# Adaptive QIR statevector extraction

Status: implemented and locally validated.

## Contract and approach

QDMI zero-shot QIR jobs accept Base and Adaptive profiles. Preserve the existing
Base terminal-region rewrite. Adaptive execution defers measurements while
running classical control flow and direct helper calls. Operations on measured
wires fail extraction; independent wires may still evolve. Measurement-dependent
result reads, resets, indirect calls and unknown external effects are rejected
before JIT execution. Unused reads and direct boolean output records are
allowed. Output records do not produce fictitious measurement bits.

Dynamic allocations create distinct zero-initialized wires. Releases are
lifetime markers during extraction: they neither reset the state nor recycle
wires. Sampling retains its existing release/reset/recycling behavior.

Runtime terminality failures are reported after returning from generated code,
so validation does not depend on C++ exceptions unwinding through JIT frames.
Preflight excludes measurement-dependent control, including in helpers, so dummy
measurement values cannot control execution. Repeated extraction starts fresh.

## Validation

Native JIT and QDMI string/bitcode regressions cover amplitudes, global phase,
classical control, helper calls, allocation and release, repeated execution,
output suppression, and rejection of non-terminal measurements and feedback.

The full Clang build and CTest suite pass: 3380 tests, with one existing job-ID
capability test skipped. C++ lint and strict documentation generation, including
local link checks, pass. Full repository lint passes.

# Exception-free API audit

Status: applied; validation is recorded in the
[execution plan](../plans/exception-free-core.md). Reassessed on main
`d3fbd39e0`, with PR #2545 rebased as `b94419052`, on October 10, 2026.

## Findings addressed

1. **Diagnostic formatting must survive allocation failure.**
   `diagnostics::detail::emitFormatted` now shares the allocation-free stderr
   writer with unhandled diagnostics. The fallback neither constructs an owning
   diagnostic nor invokes handlers. A regression disables allocations while
   formatting a long message: the original code aborts; the fix writes the raw
   format string and returns. The test compiles the production sources into its
   executable so allocation replacement also covers shared-library builds.

2. **Relative library loads must retain their own resource directory.**
   `qdmi::detail::moduleDirectory` identifies the executable image before using
   the executable-path fallback on Linux and macOS. Other relative paths remain
   library paths. The Linux regression loads the existing driver fixture by
   absolute and relative paths; an actual-source probe fails before the fix and
   passes afterwards. Executable-path probes also pass after changing directory,
   including a non-PIE build. Windows is unchanged. This defect also exists on
   main. Resolving a relatively loaded DSO after a later process `chdir` remains
   a pre-existing limitation; this change does not broaden that contract.

3. **Private driver APIs need only scoped diagnostics.** Removed
   diagnostic-output parameters from seven private factory, registry, and driver
   methods, their local capture guards, and the test helper's special invocation
   branch. Driver tests link the same static CoreSupport as their handlers. They
   still check message, category, original status, and silent success.
   Independently loaded libraries retain C-status translation and hidden
   support; no C++ diagnostic object crosses that boundary.

4. **DD numerical warnings must reach diagnostic handlers.**
   `Package::measureAll` emits a numerical warning through the shared
   dispatcher. `CorruptedBellState` now checks warning capture while preserving
   successful measurement and its existing invalid-state cases. The capture
   assertion fails against the former direct stderr write.

## Selected refactors

- Benchmark evaluation counts shots once and lets probability functions validate
  outcomes, including zero-count entries. Shor validates its own outcomes
  because it has no probability callback. Public width, encoding, and overflow
  checks stay.
- Generic benchmark JSON parsing uses one validated envelope, then dispatches
  directly to parameter parsers. Manifest generation supplies the existing case
  ID without a second hash; checked owned strings move into options.
- QDMI bindings reuse nanobind optional casting and `bindResult`. Python
  fallback diagnostics are constructed only when failure provides no captured
  error.
- The MLIR DD adapter drops an unused qubit-count argument. QCO yield binding
  uses the verifier-defined result order once; it retains mapping checks and
  atomic updates. Parallel sampling moves its completed map after all futures
  are drained.

## Retained checks and limits

DD capacity, operands, numerical measurement, serialized inputs, and C ABI
status handling protect public contracts. They remain checked. A blanket
checked/unchecked API split or propagation macro would add machinery without
removing those needs. Preinitializing Python's fallback would allocate on
successful calls and remains rejected. Broader benchmark evaluation JSON changes
would alter diagnostic order or require a larger redesign; they are outside this
cleanup.

The final independent source review found no further required changes. It traced
validation, ownership, result ordering, and binding conversions; it did not run
tests. Current Windows/macOS execution and performance need separate evidence.

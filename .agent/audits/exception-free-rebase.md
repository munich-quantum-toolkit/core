# Exception-free rebase audit

Status: recommendations, unapplied. Reviewed PR head `5c26a01a3` and its rebase
onto main `81c570c68` on October 9, 2026. Rebase integration fixes are separate.

## Findings

1. **Allocation failure can terminate while reporting a recoverable C error.**
   `diagnostics::detail::emitFormatted` catches formatting failures, then
   allocates another string in its `noexcept` fallback. Under persistent
   allocation failure, this terminates before SC's `guardDeviceCall` can return
   `QDMI_ERROR_OUTOFMEM`. A probe linked against the rebased CoreSupport archive
   with failing `operator new` reached its termination handler (exit 86). Use an
   allocation-free stderr fallback accepting a string view. Preserve the
   existing C status contract; add a regression with actual allocation failure,
   not only an explicit throw while later allocations still succeed.

2. **Relative library loads can resolve resources beside the wrong executable.**
   `qdmi::detail::moduleDirectory` treats every relative `dladdr` filename as
   the main executable. Loading `./provider.so` also returns a relative
   filename. A Linux loader probe reproduced that case: the library was in a
   temporary directory, but the current branch selected `/usr/bin`.
   Configuration and worker lookup use this helper. Distinguish the executable
   before using its path; retain the loaded library path otherwise. Cover
   relative library loads. This bug is also present on main; the rebase did not
   introduce it. The probe checks the loader result and the branch's
   path-selection algorithm; macOS and Windows were not exercised.

3. **Private driver APIs no longer need diagnostic-output parameters.** Main
   hides driver internals and exports only C entry points. Driver tests link
   `mqt-core-qdmi-driver-testing` statically with the same CoreSupport as their
   capture helpers. Production callers leave every optional `Diagnostic*` null;
   non-null arguments occur only in diagnostic tests. Remove these parameters
   and local capture guards from `LoadedDeviceAPI::create`,
   `QDMI_Device_impl_d::create`, `Driver::registerDevice`,
   `Driver::registerDeviceIfAbsent`, `Driver::registeredDeviceIds`,
   `Driver::open`, and `DeviceRegistry::discover`. Adapt tests to install
   ordinary handlers. The rebase removes the public-driver documentation example
   because those implementation headers are no longer installed. Keep C-status
   translation, hidden static support, and metadata coverage. The test named
   `CopiesFailureAcrossLibraryBoundary` now exercises one executable's static
   library, so it does not demonstrate a DLL boundary. Estimated remaining
   reduction: 50–70 net lines, no dependency removal.

4. **A DD numerical warning bypasses scoped diagnostics.** `Package::measureAll`
   writes a normalization warning directly to `std::cerr`. `CorruptedBellState`
   already exercises this with a root weight of 0.5. Native handlers and DDSIM
   diagnostic frames miss the warning. Emit a `Numerical`/`Warning` diagnostic
   and extend that test to assert capture. This is a pre-existing stream write
   left outside the new diagnostic contract.

## Rejected simplifications

- Preinitializing Python's captured diagnostic with its generic fallback message
  would shorten `bindings::invoke`, but allocates on every successful call.
  Retain failure-only storage and formatting.
- DD capacity, gate operands, numerical measurement, and serialized-input checks
  protect real public inputs. No blanket conversion to assertions is justified.
- Main's runtime benchmark arithmetic replaces old static-angle-table tests.
  Preserve its numerical and execution checks rather than restoring obsolete
  representation assertions while rebasing.

## Scope and limits

Traced diagnostics through native APIs, Python, private driver code, C entry
points, JSON boundaries, QIR, and reusable DDSIM workers. Reviewed conflict
resolutions for main's concrete DD exports, parallel sampling, runtime
packaging, and benchmark arithmetic. Assumed concurrent jobs and independently
loaded libraries; this is not a hostile-code sandbox review. Performance and
native Windows/macOS packaging require separate evidence. Validation of the
rebased implementation is recorded in the execution plan.

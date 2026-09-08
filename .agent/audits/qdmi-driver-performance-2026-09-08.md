# QDMI driver performance audit

Status: both confirmed findings implemented and locally validated. Date:
2026-09-08. Audit baseline: `e37fc41a49176c772ef855a12cfeadcccb471440`,
initially clean. Driver source matched `origin/main` at
`3be5ee96f3907659bd99fdd6d54cd74c4eea7da9`, the implementation base. Dependency:
QDMI v1.3.3. Measurements: ARM64 DGX Spark, Clang 23.

## Result

Two confirmed opportunities in session opening were implemented. They do not
establish faster quantum execution or higher remote-job throughput.

### 1. Provider initialization no longer holds the global cache mutex

`getDynamicDeviceLibrary` in `src/qdmi/driver/Driver.cpp` held a process-wide
mutex across loading, symbol resolution, and the provider's `device_initialize`
callback. A slow first open blocked fresh opens of unrelated cached providers.
Public `Session::openDevice`, Python `open_device`, and compiler device opening
reach this path; an already-open `Driver::open` device bypasses it.

The global mutex now protects only the module map. Each loaded module has a
mutex protecting initialization and its prefix-to-provider map. This preserves
serialization for providers sharing module state, alias identity, one successful
initialization, failure cleanup/retry, and process-lifetime retention. Module
entries are never erased and an opener owns a loader reference while waiting.

A gated initializer blocked an unrelated cached provider for the full 200 ms
observation interval in four baseline runs (200.206–200.867 ms). With the fix,
the unrelated open completed before release in all four runs (0.050–0.060 ms).
The test shared object was preloaded to isolate the provider callback from OS
loader initialization. This is a controlled concurrency demonstration, not a
measurement of real provider startup. OS loader serialization can remain.

Durable regression:
`DynamicDeviceLibraryDeathTest.InitializesUnrelatedModulesWhileRetryingConcurrentAliases`.
It gates the first initialization, queues an alias, opens a cached unrelated
provider, then makes the first initialization fail. The alias retries, and a
later open shares its successfully initialized provider. Existing alias/lifetime
coverage checks process-lifetime retention and lack of premature finalization.

### 2. Inline configuration avoids two redundant payload copies

`Driver::openFresh` copied the registered definition, `mergeSessionConfig`
copied its configuration again, and parameter forwarding implicitly converted
inline JSON into an owning `optional<string>`. The last copy repeated for each
child. Raw `custom1`, already optional, avoided the last conversion.

The merge helper now takes defaults by value. Fresh opening moves its owned
snapshot into the helper; registry callers passing lvalues retain copy
semantics. The setter borrows existing NUL-terminated strings through optional
string views. The registry snapshot, explicit empty overrides, absent
parameters, NUL-inclusive lengths, conflict checks, and synchronous provider
call remain intact.

A no-op provider and allocations of at least 1 MiB isolated a 16 MiB synthetic
payload. Snapshot/merge/construction allocated three payload-sized buffers for
typed inline configuration before the fix and one after it. One hundred isolated
constructors allocated 100 such buffers before and zero after. Constructor time
was 42.613–63.504 ms before and 0.012–0.017 ms after across four runs each.
These synthetic results exclude JSON parsing, provider copies, Python
conversion, networking, and whole-device opening. The payload is opaque
synthetic text, not a simulator configuration benchmark; allocation counts are
the useful result.

Durable regression: `DeviceSessionConfigTest.MovesAndBorrowsInlineConfiguration`
checks buffer identity through the merge and the parent plus two child sessions.
Existing tests cover configuration contents, overrides, errors, and ownership.

## Candidates not promoted to findings

- Linear registry lookup and quadratic repeated registration: no representative
  catalog measurement establishes a practical bottleneck. Preserve public order.
- Per-client copies of the immutable handle catalog: savings depend on catalog
  size and session churn; no workload justifies changing ownership here.
- Eager catalog and child-session initialization: laziness changes error timing
  and lifetime behavior, so it is not part of these optimizations.
- Repeated OS-loader calls on warm fresh opens: no measured cost justifies an
  additional cache and its identity/invalidation obligations.

Property/site/operation queries and job submit/check/wait/results already avoid
bulk copies and global driver locks. Job creation/free holds the per-device
mutex only for ownership bookkeeping; provider creation/free run outside it. No
evidence supports a job pool, custom allocator, lock-free registry, or
driver-wide metadata cache.

## Related work and validation

REST metadata and changed files were checked on 2026-09-08. PR #2229
(`d76d6681`) changes the Client ABI; #2230 (`1f544f1a`) changes default-driver
configuration and still holds the cache mutex during construction in its older
path-based cache. PR #2373 (`c8b4ac3d`) changes batch/job forwarding, separate
from these opening costs. Pending branches were not performance-tested.

The optimized probe compiled `Driver.cpp` with `clang++-23 -std=c++20 -O2` and
linked existing Debug Client/registry/common support; registry startup was
outside timing. The rebuilt native driver binary passed 106 tests. Repository
lint passed. Changed-file C++ lint passed with zero findings, including all
three changed C++ translation units. The full QDMI CTest tree passed 478 tests
with one existing skip (`ScQDMIJobSpecificationTest.QueryJobId`). Both new tests
fail when linked against the original driver translation unit and pass with the
fixed driver. The permanent tests replace the temporary benchmark's
implementation-specific assertions as the regression checks.

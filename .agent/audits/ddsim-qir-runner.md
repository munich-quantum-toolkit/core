# DDSIM QIR runner audit

Status: implemented. Date: 2026-09-08. Audit baseline:
`33dbc843e589d9e9166308084e825c9f8b2ff89d`; implementation based on main
`3be5ee96f`. Scope: DDSIM, QIR execution, resource metadata and shared DD roots.

## Findings and disposition

1. **Repeated static simulation:** DDSIM executed every gate for every shot. A
   conservative Base-profile analysis now proves a straight-line path with known
   constant QIS calls, terminal measurements and static recorded outputs.
   Eligible batches execute once and sample the prepared DD. Result order,
   repeated records and logical wire mapping are preserved. Unknown calls,
   memory operations, branches, loops, resets and feedback retain per-shot JIT
   execution. Text-enabled sessions also retain per-shot execution.
2. **Unused output:** DDSIM retained formatted text it never read. Explicitly
   disabled output now skips formatting while preserving measurement records.
   Public text output and shot framing remain available.
3. **Temporary allocations:** Address and generic-control translation now use
   inline SmallVector storage; scalar measurement translates directly. Generic
   control bytes still use memcpy. Shared DD gate validation remains with #2455.
4. **Unused lazy JIT:** Replace LLLazyJITBuilder with LLJITBuilder and remove
   its unused lazy manager, duplicate process-symbol generator and moved-out
   state. Cache the immutable ABI registry. Preserve debugger and MinGW support,
   host CPU tuning and explicit codegen options; reject incompatible
   architecture/OS.
5. **Logical SWAP extraction:** X(q0); SWAP(q0,q1) previously exported physical
   order. Materialize the final permutation once when transferring state.
6. **Phase and root ownership:** State growth discarded an initial scalar phase;
   global phase changed a tracked edge without updating its root reference.
   Growth now preserves weight, and the shared DD helper replaces the reference.
   Regression tests cover following gates and forced collection.
7. **Static resources:** Configure validated capacities from QIR attributes,
   initialize declared width and use indexed static result storage. Missing
   metadata retains inference. Avoid a second shot reset only when the entry
   starts with runtime initialization. The metadata producer also counts
   measured and read result IDs that have no output record.
8. **Dynamic lifetime:** Recycle reset wire slots while issuing distinct handles
   within a shot. Unused released wires are not materialized; the DD limit now
   bounds peak live allocation. Tests retain stale-handle rejection and SWAP
   behavior across reuse.

## Retained boundaries

Canonical gates and DD construction remain shared with QCO. No second
interpreter, new dependency, general gate cache or LLVM optimization pipeline
was added. Gate-cache evidence only covered repeated X gates; ownership, GC and
large-DD behavior need separate evaluation. State-extraction analysis remains
separate from the stricter sampling proof. Seeds are repeatable within a path;
optimized sampling does not promise the previous RNG sequence.

## Validation

Release tests passed: 38 JIT/analysis, 78 runtime, 65 DDSIM, 169 DD and 122 QIR
IR cases. These cover fallback side effects, no-initialize shot reset, output
mapping, phase, resource limits and metadata production. Required lint is
tracked in the companion plan and PR.

A local GCC/LLVM 23.1 ARM64 comparison uses the same current DD implementation
for both runners, including JIT construction, sampling, histogram insertion and
destruction. Five warmed trials pinned to CPU 0 gave these median process CPU
milliseconds (baseline / implementation): Bell 10,000 shots 43.3 / 17.9;
16-qubit GHZ with 256 rotations and 4096 shots 409.0 / 21.6; 8-qubit dense
circuit with 256 shots 37.8 / 16.5; Adaptive Bell 10,000 shots 32.8 / 24.1;
one-shot Bell 11.4 / 9.6. Bell ordinary C++ allocations fell from about 284,000
to 8,100. Host contention makes wall-clock comparisons unstable. These are
diagnostic local measurements, not portable latency claims or large-DD
benchmarks. Local sources and raw measurements are retained under
`build/qir-audit/`.

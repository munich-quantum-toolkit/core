# OpenQASM frontend performance

Status: implemented and locally validated. Baseline:
`ea1e672125a7a3645ee163966d57952855cd47ce`. Date: 2026-09-09.

## Changes and contracts

- Include expansion has a separate one-million-context limit. The statement
  limit alone did not bound branching empty includes. Textual repetition,
  recursion checks, source order, and per-occurrence provenance remain intact.
  The expansion test checks both empty leaves and nonempty leaves large enough
  to reach the statement limit first.
- Barrier distinctness uses the existing gate-call algorithm: group references
  by register and compare only pairs involving an affine index. Static
  duplicates, full-register overlaps, uncertain aliases, and affine proof work
  remain checked. A 64,000-qubit barrier test covers an unrelated affine
  operand.
- Source-position lookup is separate from include-stack construction. Logical
  include frames no longer construct and discard physical ancestor stacks.
  Existing nested and repeated-include provenance tests remain the oracle.
  Locations are not cached across textual include occurrences.
- The string importer uses a temporary non-owning MemoryBuffer and retains the
  frontend's owning copy. A regression imports a non-null-terminated StringRef
  and verifies the module after overwriting the original storage.
- Canonical gate names reuse standard-gate descriptors with the sole spelling
  exception `U3` -> `u3`. All 32 names were compared before replacing the
  switch. Gate availability, arity, aliases, and phase-sensitive U/u2 lowering
  are intact.
- Two scope guards use LLVM 23's direct `scope_exit` constructor instead of the
  deprecated `make_scope_exit` factory.

## Reproducers and measurements

The original audit used `0c3fac2cb2861e9e25262857a84650cba651f466`, GCC 13.3,
LLVM/MLIR 23.1.0, `-O3 -DNDEBUG`, and native ARM64. Relevant frontend production
files were unchanged on the implementation baseline.

For empty includes, create D supplied buffers, each including its successor
twice, with an empty last buffer. The old parser created `2^D - 1` contexts and
zero statements. At D=22 it succeeded in 270 ms using 106,408 KiB peak RSS. The
new parser diagnoses the expansion limit before allocating another context.

Barrier source: `OPENQASM 3.1; qubit[64000] q; qubit[2] r;`
`for int i in [0:1] { barrier q, r[i]; }`. The experimental grouping change
reduced median analysis time from 1062.91 ms to 5.10 ms over five interleaved
runs.

For include ancestry, put `qubit q;` and 2,000 `U(0,0,0) q;` statements beneath
64 nested includes. Removing unused ancestry construction reduced experimental
median analysis time from 87.22 ms to 6.49 ms over five interleaved runs. Both
results retained 2,001 statements and 128,064 include frames.

These are focused historical frontend measurements, not end-to-end compiler
speedups. Source-copy removal and canonical-name simplification have allocation
and duplication arguments, without a separate latency claim.

## Validation

The release build of the OpenQASM target and QC translation test binaries passes
with local IPO disabled. All 194 OpenQASM target tests and 209 QC translation
tests pass. Repository lint and full-file C++ lint against the baseline pass,
with zero C++ findings. No binding API changed; hosted CI remains separate.

A probe linked to the final release frontend retained the four barrier
statements and all 128,064 nested-include frames. Five interleaved runs against
the original baseline gave median analysis times of 1087.20 -> 5.12 ms for the
barrier and 88.06 -> 6.97 ms for nested includes. The D=22 empty-include input
now fails with the intended expansion-limit diagnostic and 28,792 KiB peak RSS.
A 20,000-gate input without includes showed comparable analysis times (16.30 ->
15.87 ms median); no general flat-input speedup is claimed.

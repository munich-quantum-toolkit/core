# QDMI query performance and decoding

Status: complete.

## Scope and decisions

The standard C++ decoder rejects array byte counts that are not multiples of
an element's size and strings without a final terminator. Session arrays and
job IDs use this decoder. Histograms use the existing sparse key/value reader,
which rejects excess keys and preserves the zero-width outcome.

SC sessions sort internal supported tuples and calibration overrides once;
queries borrow the supplied tuple. The public flattened site list retains its
configured order. Compiler-target snapshots reuse operation names, site indices,
known sites, and canonical couplings within one conversion. The known-site set
is built only when an operation reports tuples. No metadata cache survives a
snapshot or changes session ownership.

JSON key validation searches the existing fixed lists without allocating sets.
Public APIs and QDMI versions remain unchanged. Future adapter work in #2227 can
retain the per-snapshot cache; allocation-failure containment in #2271 and
batching in #2373 remain separate work.

## Validation

Regression tests cover malformed property buffers, histogram key/value counts,
unsorted tuple ordering, partial calibration overrides, and provider query
counts on repeated snapshots. Local checks pass: 469 QDMI tests (one existing
SC job-ID skip), 164 compiler tests, 463 Python QDMI/plugin tests, `nox -s lint`,
and `nox -s cpp-lint`.

A synthetic 4,000-site chain reduces site-index queries from 27,994 to 4,000 and
operation-name queries from ten to two. Calibration checksums match. The target's
all-pairs distance computation remains outside this change.

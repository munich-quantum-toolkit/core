# Qiskit translation simplification

Status: complete.

## Outcome and decisions

Implemented [the audit findings](../audits/qiskit-translation.md), preserving
supported inputs, source immutability, and the version-specific native boundary.

- Each writer owns one unpublished Python circuit. Blocks inherit exact parent
  resources and symbol identities. Numeric appends borrow native data; symbolic
  and classical instructions use Python. This removes deferred repair state.
- Snapshot materialization follows writes, region and control-flow edges, and
  expression depth. A shared write index avoids repeated linear scans.
  Unsupported widths and unsafe measurement scheduling still fail closed.
- Balanced dispatch bounds generated nesting for irregular loop lists and
  variable-bearing switches. Arithmetic progressions use ranges only when the
  normalized endpoint satisfies the existing precision contract.
- Both ABI snapshots use the same extractor, including transitive signature
  types. Test support comes from the adapter registry. Broad SDK dependency
  support remains separate from direct compiler translation support.

## Validation

The focused suite passed: 506 tests, one deselected. C++ lint against
integration base `3be5ee96f` passed with zero findings across all six changed
C++ files. Repository lint and stub generation passed; generated stubs did not
change. One adoption fixture test is excluded because it creates an unsigned
commit, contrary to the checkout signing requirement.

The focused suite is reproducible with:

```sh
uv run --no-sync pytest -n0 -q \
  test/python/test_mlir_qiskit_translation.py \
  test/python/test_mlir_loops.py \
  test/python/test_mlir_integer_interchange.py \
  test/python/test_qiskit_c_api_adopt.py \
  -k 'not restartable_worktree_rejects_unrelated_changes'
```

## Performance and limits

Development measurements on ARM64 with Qiskit 2.5.2 compared audit baseline
`33dbc843e` with the implementation before final integration. Medians of three
runs after warmup used 10,000 RX gates; the mixed case inserted ten conditional
symbolic RY gates.

| Workload                   | Baseline | Implementation |
| -------------------------- | -------: | -------------: |
| Numeric import             |  45.0 ms |        34.0 ms |
| Numeric export             |  12.6 ms |        11.8 ms |
| Symbolic import            |  74.3 ms |        52.4 ms |
| Symbolic export            |  10.9 ms |        15.2 ms |
| Mixed import               |  45.7 ms |        34.6 ms |
| Mixed export               |  13.6 ms |        12.9 ms |
| Shared read, 4,000 stores  |   247 ms |        40.6 ms |
| Shared read, 16,000 stores | 3,494 ms |         163 ms |

The shared-read probe exports one `cbit.read` reused by writes to a distinct
one-bit register. The accepted symbolic-export tradeoff is about 4.3 ms per
10,000 gates for one output owner and removal of deferred symbolic repair. The
final benchmark rerun was distorted by heavy concurrent host builds and is not
treated as a comparable measurement. No final-revision speedup is inferred from
it. Native ABI provenance and behavioral validation remain required for future
Qiskit minor versions; the adoption extractor is not an ABI proof.

# Contract audit: PennyLane QDMI plugin

Status: historical decisions applied or narrowed in PR `#2147`, last recorded
reconciliation `3354fdaab` on 2026-08-18. Current behavior and the historical
experiments were not revalidated during document cleanup.

Source: `python/mqt/core/plugins/pennylane/`. Tests:
`test/python/plugins/qdmi_pennylane/`. Original baseline: `18b619af5`; probes
ran on `c44d83d66`, whose changes did not affect those source or test paths.
Symbol and line references below describe that baseline unless stated otherwise.

## Result

The main improvement was a device-owned converter that reuses device metadata.
For the measured 100-gate circuit on an 18-operation device, preprocessing and
conversion went from 101 `operations()` queries and 1818 `name()` calls to one
and 18 at device construction. The source lost 37 lines; tests grew by 57. The
benefit was fewer device calls and one format-selection owner.

| ID  | Change                                                                              | Recorded disposition                                      |
| --- | ----------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 1   | Compare parsed gate parameters; use shortest round-tripping float output            | Applied                                                   |
| 2   | Select program format once at the device boundary                                   | Applied with 6                                            |
| 3   | Check reconstructed histogram multiplicity without requiring shot order             | Applied                                                   |
| 4   | Relax an exact rotation literal; retain the single-rotation check                   | Narrowed; claimed regression protection was unproven      |
| 5   | Check accumulated execution time with a controlled clock                            | Applied                                                   |
| 6   | Replace public conversion helpers with a device-owned converter and cached metadata | Applied; public API change required maintainer acceptance |

## Decisions and evidence

### 1. Preserve parameter precision without fixing its spelling

`test_converter.py:131-133` required 17-digit strings for the values 0.1, 0.2,
and 0.3. Replacing `_format_parameter`'s `.17g` rendering at `converter.py:228`
with `repr(_finite_parameter(parameter, operation_name))` failed only
`test_qasm3_resolves_ddsim_aliases_and_inverse_gates`; the DDSIM integration
tests passed. The replacement parses the emitted values and checks exact float
equality, preserving round-trip precision. This does not justify loosening
numerical accuracy to an arbitrary tolerance.

### 2. Give format selection one owner

The device and converter both selected QASM3, then QASM2, then failed, with
separate wording checks. Replacing the converter's error at `converter.py:146`
with the device wording failed only `test_rejects_device_without_qasm`. The
useful evidence for consolidation was the duplicated implementation and the
device's already-selected format. The device now passes that format to
conversion. QASM3 translation failures must still fail without silently retrying
QASM2.

### 3. Preserve histogram counts

`test_histogram_only_device_reconstructs_samples` required zero rows followed by
one rows. Removing `sorted` at `device.py:274` passed because insertion order
happened to agree; reversing it failed only that test. The test was the only
counts-only-device check, so deletion would have lost coverage. It now compares
a `Counter` of rows. The production sort was retained: the test change allows
other expansion strategies without claiming a measured speedup.

### 4. Do not mistake an inert mutation for regression coverage

Changing `rotations=False` to `True` at `converter.py:384` failed no test.
Characterization showed why: device preprocessing had already replaced
observable measurements with `SampleMP`, making the flag inert on that path. It
still affected a direct public converter call with a Pauli-X expectation. The
exact angle literal was narrowed to a single `ry(` occurrence. The flag remained
unchanged. The original claim that the test caught double rotation was rejected;
a future preprocessing change needs a suitable semantic regression, not reliance
on this probe.

### 5. Test timing accumulation

The old `execution_time >= 0.0` assertion also passed when accumulation was
removed and the value stayed zero. A replacement test controls the clock over a
three-job shot vector and checks the total. Mutating accumulation to `+= 0.0`
then fails. The broad execution test retains a finiteness check. A strict
positive-time check would be unreliable for a coarse clock and a short job.

### 6. Reuse device metadata during conversion

The original exported `convert_program(tape, device, wires)` interface
repeatedly queried supported operations during per-operation preprocessing. The
accepted change introduced `_ProgramConverter`, bound to one device session, and
moved conversion tests through `QDMIDevice` and submitted payloads. Direct
`execute` tests retain rejection coverage that preprocessing would otherwise
intercept.

The original audit called the exported helpers “contract-free” because it found
no documentation or in-tree external callers. That reasoning is insufficient:
exported APIs can have downstream users. The recorded maintainer acceptance
allowed this change; the same search would not authorize another API removal.
Internal unit tests are not inherently unnecessary either. Use the boundary that
most directly protects the actual behavior.

## Coverage and limits

Reversing the QDMI bit-string conversion at `device.py:300` failed
`test_stable_entry_point_and_wire_order` and two cases of
`test_gate_semantics_against_pennylane_reference`. Stub Bell histograms were
palindromic and missed it. Keep asymmetric inputs and differential simulator
checks for wire ordering, even when prose omits the convention.

The rewrite retained non-finite parameter rejection and device-side format
validation, and added tests for advertised sites, site pairs, and coupling maps.
The recorded plugin coverage increased from 85% to 90%, and converter coverage
from 82% to 94%. Those coverage numbers alone do not prove equivalent or
stronger assertion coverage.

The exported exception hierarchy was deliberately retained. Lack of in-tree
catch sites does not establish that downstream users do not catch those types.
Batch and shot-vector ordering and one-job-per-executable-tape behavior remain
supported contracts.

Two questions were not settled by executed probes: whether the exact QASM2
payload assertion catches meaningful serializer drift across PennyLane versions,
and whether every non-finite parameter case covers a distinct failure mode.
Retain their substantive coverage until a proposed replacement is justified.

## Reproducing the historical checks

At the recorded baseline, build/install the package using the repository guide,
then run from the repository root:

```sh
uv run --no-sync pytest test/python/plugins/qdmi_pennylane -q
```

Apply only one source change described above, run the same suite, record its
exit status and named failures, then restore and rerun the baseline. Use a
compatible dependency environment; a collection failure is not probe evidence.
The timing mutation refers to the post-resolution test at `3354fdaab`. The
device-call measurements above describe the original instrumentation; the
instrumentation was not retained as a benchmark, so they are historical evidence
rather than a ready-to-run performance harness.

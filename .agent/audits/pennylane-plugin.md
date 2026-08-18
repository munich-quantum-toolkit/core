# SpecAudit: PennyLane QDMI plugin

Scope: `python/mqt/core/plugins/pennylane/` (915 lines) with
`test/python/plugins/qdmi_pennylane/` (834 lines, 26 test functions, 33
collected cases).

Baseline `18b619af5` for every citation. The probes ran on `c44d83d66`, which
adds this method to `.agent/` and changes nothing under `python/` or `test/`.
Every `file:line` below was read at that commit. Every verdict names an
experiment that was executed, and the failing-test lists are copied from the
probe output.

Method: [`.agent/AUDITS.md`](../AUDITS.md). Probe tool:
[`.agent/audit-probe.sh`](../audit-probe.sh).

Reconciled on 2026-08-18 against `3354fdaab`. Every `file:line` below was
re-read at that commit and still resolved to the line it named, so the drift
since the baseline invalidated no verdict. Pull request `#2147` applied all six.
Each verdict now carries an **Applied** paragraph saying what landed and what
did not.

After that pull request the scope is 878 source lines and 891 test lines across
26 test functions and 33 collected cases. The source lost 37 lines; the tests
gained 57, because driving conversion through `QDMIDevice` costs QNode
boilerplate that a direct converter call did not. The payoff of verdict 6 is the
round trips it removed, not the lines, and the summary table below says so.

## Why this scope

The plugin arrived in one commit, `077b73f80` (`#2005`), which added the four
source files, the tests, the documentation page, and
`.agent/plans/qdmi-pennylane-device.md` together. `git log --diff-filter=A` on
the plan returns only that commit, and the plan as committed already carries ten
completed checkboxes and a retrospective. It is the agent's account of what it
did, not a specification that existed first, so it sits at rung 4 throughout.

No issue in this repository asks for a PennyLane device. The umbrella QDMI issue
`#772` lists a FoMaC library, a driver, device implementations, and "high-level
Python bindings", and never mentions PennyLane. The human record for this scope
is the approved pull request summary plus eight review comments, all of which
concern documentation wording and lint.

## Spec ledger

Full ledger in the four source classes; the entries the verdicts below rely on:

| ID   | Promise                                                                             | Rung | Source                                |
| :--- | :---------------------------------------------------------------------------------- | :--- | :------------------------------------ |
| S-A  | Sample columns follow the emitted measurement order                                 | 1    | PennyLane `process_samples`           |
| S-B  | Only advertised program formats may be submitted                                    | 1    | QDMI `constants.h:424-432`            |
| S-C  | A successful `wait` does not imply success                                          | 1    | QDMI `client.h:961-981`               |
| S-D  | Each executable tape becomes one distinct QDMI job                                  | 2    | `docs/qdmi/pennylane_device.md:190`   |
| S-E  | Jobs are submitted sequentially                                                     | 2    | `docs/qdmi/pennylane_device.md:191`   |
| S-F  | Shot vectors and batches execute in order                                           | 2    | `docs/qdmi/pennylane_device.md:314`   |
| S-G  | QASM3 first, QASM2 only if QASM3 is absent, else a format error before job creation | 2    | `docs/qdmi/pennylane_device.md:63-67` |
| S-H  | A QASM3 failure is never retried as QASM2                                           | 2    | `docs/qdmi/pennylane_device.md:70`    |

Nothing published states any of the following, and the verdicts turn on that:
error message wording; the rendering of a gate parameter; the order in which a
histogram is expanded into shots; the bit-string endianness convention; or how
many QDMI jobs a shot vector decomposes into.

## Summary

Ranked by complexity removed per unit of risk.

| #   | Assertion                   | Class          | Remedy       | Unlock                              | Risk | Status   |
| :-- | :-------------------------- | :------------- | :----------- | :---------------------------------- | :--- | :------- |
| 1   | `test_converter.py:131-133` | Over-specified | Narrow       | Readable QASM output                | Low  | Applied  |
| 2   | `test_converter.py:212`     | Contract-free  | Delete       | −16 lines, −1 QDMI call per tape    | Low  | Applied  |
| 3   | `test_device.py:89-90`      | Contract-free  | Narrow       | Frees the ordering strategy         | Low  | Applied  |
| 4   | `test_device.py:180`        | Over-specified | Narrow       | None; the claimed guard is not real | Low  | Narrowed |
| 5   | `test_device.py:73`         | Contract-free  | Strengthen   | None                                | Low  | Applied  |
| 6   | `test_converter.py:27-33`   | Contract-free  | Rewrite file | −99% QDMI calls in preprocessing    | High | Applied  |

## Verdicts

### 1. The emitted parameter rendering is pinned to seventeen digits

`test/python/plugins/qdmi_pennylane/test_converter.py:131-133` asserts the
literals `rxx(0.10000000000000001)`, `ryy(0.20000000000000001)`, and
`rzz(0.29999999999999999)`.

These come from `converter.py:228`, `format(value, ".17g")`, inside
`_format_parameter`, which is private. The docstring states the intent: "Format
one QASM parameter without losing double precision." That intent is a real
promise. The *spelling* is not: nothing published mentions how a parameter is
rendered.

**Probe.** Replaced `converter.py:228` with
`return repr(_finite_parameter(parameter, operation_name))`.

```text
failing tests   : 1
  - test_converter.py::test_qasm3_resolves_ddsim_aliases_and_inverse_gates
```

Nothing else in the suite noticed, including the end-to-end DDSIM tests.

`repr` gives the shortest representation that round-trips exactly, so the stated
intent is preserved: `float("0.1") == 0.1`. For every value the tests use, the
two renderings agree except for `0.1`, `0.2`, and `0.3`, where `.17g` emits
trailing noise.

**Remedy.** Parse and compare instead of matching text:

```python
emitted = dict(re.findall(r"(rxx|ryy|rzz)\(([^)]+)\) q\[0\],q\[1\];", payload))
assert {name: float(value) for name, value in emitted.items()} == {"rxx": 0.1, "ryy": 0.2, "rzz": 0.3}
```

Exact float equality, so any real precision loss still fails.

**Unlock.** `converter.py:228` may then emit `rxx(0.1)` rather than
`rxx(0.10000000000000001)`. Every OpenQASM program this plugin sends to every
device becomes readable, at identical precision. Three assertions were the only
thing preventing it.

**Applied.** The three assertions now parse the emitted literals and compare
them as exact floats, and `_format_parameter` became `repr` at its single call
site. `IsingXX(0.1)` reaches a device as `rxx(0.1)`.

### 2. The same format rule is implemented twice, and each wording is pinned

`device.py:187-202` and `converter.py:132-147` are the same rule: QASM3, then
QASM2, then raise. They differ only in the message.

```text
converter.py:146  "supports none of the required program formats: OpenQASM 3 or OpenQASM 2."
device.py:201     "advertises neither OpenQASM 3 nor OpenQASM 2."
```

`test_converter.py:212` pins the first, `test_device.py:218` pins the second.
Neither wording is published. The duplication survives because unifying the two
messages breaks a test either way.

**Probe.** Rewrote `converter.py:146` to the `device.py` wording.

```text
failing tests   : 1
  - test_converter.py::test_rejects_device_without_qasm
```

**Remedy.** Delete `converter._preferred_format` and give `convert_program` the
format the device already selected. `test_converter.py:207-213` cannot be
narrowed to survive that: the code path it exercises stops existing. Its
coverage is already duplicated by `test_device.py:213-219`, which asserts the
same rule (S-G) at construction.

**Unlock.** −16 lines, and one `supported_program_formats()` round trip removed
per converted tape. `device.py:163` already caches the selected format as
`self._program_format` and uses it at `device.py:225-228`; `device.py:346`
simply does not pass it, so `converter.py:413` re-derives it. Removing the
second implementation also removes the standing risk that the two diverge —
which, in wording, they already have.

**Applied.** `converter._preferred_format` is gone, and
`test_rejects_device_without_qasm` with it. The device passes the format it
already selected to the converter, so the rule has one implementation and one
wording. Applied together with verdict 6, because on its own it would have
changed a public signature that verdict 6 removes.

### 3. Histogram expansion order is asserted but never promised

`test_device.py:89-90` asserts that when a device exposes counts but not shots,
the reconstructed samples are the four zero rows followed by the four one rows.
That order comes from `sorted(counts.items())` at `device.py:274`. QDMI relates
histogram key order to shot order nowhere, and neither does the documentation.

**Probe A.** Removed `sorted(...)`, leaving `counts.items()`.

```text
failing tests   : 0
```

**Probe B.** Replaced it with `sorted(counts.items(), reverse=True)`.

```text
failing tests   : 1
  - test_device.py::test_histogram_only_device_reconstructs_samples
```

Together these say something precise. The assertion pins *an* order, so it
blocks any future change to the expansion strategy. It does not protect the
`sorted` call, because the stub builds its counts from
`Counter(["00"] * 4 + ["11"] * 4)`, whose insertion order already equals sorted
order.

The load-bearing part of this test is multiplicity, not order: each bit string
must repeat exactly `count` times, or every probability derived from a
counts-only device is wrong. No other test exercises `expose_shots=False`.

**Remedy.** Keep the defence, drop the ordering claim:

```python
assert Counter(map(tuple, samples.tolist())) == {(0, 0): 4, (1, 1): 4}
```

**Unlock.** The expansion strategy becomes free to change. Worth saying plainly:
a synthesized shot order looks meaningful and is not, so pinning it in a test
also advertises a guarantee the device does not make.

**Applied.** The test now asserts
`Counter(map(tuple, samples.tolist())) == {(0, 0): 4, (1, 1): 4}`. The `sorted`
call at `device.py:274` was left in place: the point is that the expansion
strategy is now free to change, not that it must.

### 4. A defence that the experiment refuted

The advocate for these assertions argued that `test_device.py:180`,
`qdmi.submissions[0][0].count("ry(-1.5707963267948966) q[0];") == 1`, is the
only guard against a double diagonalizing rotation: drop `rotations=False` at
`converter.py:384` and PennyLane would add a second `RY(-pi/2)`, silently
returning a wrong expectation value on a QASM2-only device. It was the strongest
argument offered for any assertion in this scope.

**Probe.** Replaced `converter.py:384` with `rotations=True`.

```text
failing tests   : 0
```

**Characterization.** Calling `qp.to_openqasm` both ways on the two tape shapes:

```text
device path (SampleMP):        rotations flag changes output = False
public path (expval PauliX):   rotations flag changes output = True
```

By the time a tape reaches `_convert_qasm2` through the device,
`measurements_from_samples` has already replaced the observable with a
`SampleMP`, so there are no diagonalizing gates left to add. The flag is inert
on the only production path. It matters solely to a direct caller of the public
`convert_program`, which today means the tests.

Two consequences. The assertion does not defend what it was believed to defend,
so the seventeen-digit angle literal in it is unearned; narrow to
`count("ry(") == 1`, which keeps the "applied once" claim. And `rotations=False`
is currently protected by nothing at all.

Recording this verdict matters more than the lines it saves. The argument was
careful, cited the plan and the pipeline, and was wrong, and only running it
showed that.

**Narrowed, not applied.** The assertion is now `count("ry(") == 1`. The probe
was re-run at `3354fdaab` and reproduced: `rotations=True` still fails no test.
No code change followed, because there is nothing here to change --
`rotations=False` remains correct and remains protected by nothing. Anyone
changing the preprocessing pipeline so that a diagonalizing gate survives to the
serializer will get a wrong expectation value with no test failing.

### 5. An assertion that cannot fail for its own purpose

`test_device.py:73` asserts `device.execution_time >= 0.0`. `device.py:349-353`
accumulates the timer in a `finally`. Delete that accumulation and the attribute
stays `0.0`, so the assertion still passes. It catches the property vanishing,
being `None`, or going `NaN`, and nothing else.

**Remedy.** Strengthen rather than delete — freeze the clock and assert the
accumulated value, or drop to an explicit finiteness check. `> 0.0` is not
available: a coarse-resolution clock can return a zero delta for the stub's
instantly-completing job.

**Applied.** A new test freezes the clock over a three-job shot vector and
asserts the exact accumulated total; `QDMIDevice` now imports `monotonic` by
name so a test can replace this device's clock alone. Replacing the accumulation
with `+= 0.0` fails the new test. The broad execution test keeps an explicit
finiteness check, which is what the old assertion really did.

### 6. A public function that exists for its tests, and what it costs

`convert_program` and `ConvertedProgram` are exported from the package
`__all__`. Searching the repository for callers outside the package finds
`device.py:346` and `test_converter.py` — nothing else. Neither name appears in
`docs/qdmi/pennylane_device.md`, which documents only `QDMIDevice`. The sibling
Qiskit plugin does document its conversion surface; this one does not.

The three-argument signature `convert_program(tape, device, wires)` is what
prevents the device from handing down state it has already computed.
`supports_operation` (`converter.py:167`) therefore calls `_device_operations`
on every invocation, and it is PennyLane's per-operation `stopping_condition`
(`device.py:225-228`).

For a 100-gate circuit on an 18-operation device, one preprocessing pass makes
101 `device.operations()` round trips and 1818 `Operation.name()` calls where 1
and 18 would do. `test_device.py:147` asserts four jobs for one parameter-shift
gradient, which multiplies that to roughly 404 and 7272.

**Blocked by.** `test_converter.py:27-33` — the file imports `convert_program`
from the package and drives all nine of its tests through it. Making the
function private breaks the file at import.

**Remedy.** Rewrite `test_converter.py` against `QDMIDevice` plus
`StubDevice.submissions`, a technique `test_device.py:125-126` already uses.
Then make the conversion surface private and cache the advertised operation
table at construction.

**Risk: high**, and the largest payoff here. It is a 236-line test rewrite, and
it should be a maintainer decision rather than a drive-by change. It is listed
last for that reason, not because it is worth least.

**Applied.** `test_converter.py` is rewritten against `QDMIDevice` and
`StubDevice.submissions`, and `ProgramConverter` binds the conversion to one
opened device session. Measured at the QDMI boundary on a 100-gate circuit over
an 18-operation device: 101 `operations()` round trips and 1818 `name()` calls
per preprocessing and conversion pass become 1 and 18 once, for the life of the
device.

Two branches lost their old callers when the tests moved to the device boundary:
the OpenQASM 3 and OpenQASM 2 refusals of an unadvertised operation, which
`decompose` now reaches first. Both tests regained them by also executing a tape
through `Device.execute` directly, which is the path where conversion, not
preprocessing, has to refuse.

Codecov then rejected the patch at 92.8% of the diff, which exposed something
the audit had not looked for: `_validate_qdmi_contract` reads advertised sites,
site pairs, and the device coupling map, and no test exercised any of it beyond
one site-pair rejection. Those tests now exist. Plugin coverage went from 85% to
90%, the converter from 82% to 94%, and every source line this change added is
covered. The one branch that could not be covered, a third program format in
`supports`, was unreachable and is gone.

`.agent/plans/qdmi-pennylane-device.md:385-387` still documents
`ConvertedProgram` and `convert_program` as public. It is left as the record of
what that task did, not corrected to match the code.

## Anchors confirmed

**The endianness convention.** `device.py:300` reverses each QDMI bit string.
Nothing published states this; the only statement of it is the code comment
above the line. Replacing `clean[::-1]` with `clean`:

```text
failing tests   : 3
  - test_ddsim.py::test_stable_entry_point_and_wire_order
  - test_ddsim.py::test_gate_semantics_against_pennylane_reference[operations0]
  - test_ddsim.py::test_gate_semantics_against_pennylane_reference[operations1]
```

Those three earn their place. Note which tests did *not* catch it: every
stub-based test in `test_device.py` stayed green, because the stub's Bell
histogram is palindromic. The end-to-end tests against the real DDSIM device are
the only defence for the single most consequential unwritten convention in this
plugin, and `test_gate_semantics_against_pennylane_reference` — a differential
comparison against `default.qubit` — is the best test in the scope.

**Believed but not executed.** Two assertions the advocate defended, which no
probe here tested. Severity and confidence are separate; these are unverified,
not cleared.

- `test_converter.py:171-181`, exact equality on the QASM2 payload. The argument
  is that it is the only detector of drift in PennyLane's serializer, which the
  plugin pre-validates against a hand-copied gate table (`converter.py:99-124`).
  That table is already out of sync with upstream, which lacks the plugin's
  `GlobalPhase` entry. Worth its own probe against a bumped PennyLane.
- `test_converter.py:216-223`, the `nan`/`inf`/`-inf` matrix. The argument is
  that each value kills a different mutant of `math.isfinite`. Plausible and
  untested here.

Both survived the verdict 6 rewrite unchanged in substance. After `#2147` they
live at `test_converter.py:184-194` and `:240-254`, and both now assert against
what the device receives rather than against a direct converter call.

## Deliberately not touched

**The seven-class exception hierarchy.** No `except` clause anywhere in the
repository catches any of the seven by type; the subtype relation is never
exercised; `PennyLaneExecutionError` is raised five times and named by no test.
Collapsing to two classes would remove roughly 56 lines, and five assertions
block it. Left alone regardless: these are exported names, so removing them is a
public API decision that belongs to the maintainers, not to an audit. Recorded
because the reason the hierarchy exists is worth knowing — the plan attributes
the `PennyLane...Error` naming to a Sphinx cross-reference collision with the
Qiskit plugin, so a documentation-build accident became permanent public API.

**Everything in `test_ddsim.py`.** End-to-end, tolerance-based, and differential
against a reference simulator. This is what the rest of the scope should look
like.

**`test_device.py:106` and `:124-126`.** The `[5, 5, 7]` shot decomposition and
the batch ordering. S-D, S-E, and S-F cover order and one-job-per-tape, and the
advocate showed the shot list blocks a plausible "submit one 17-shot job and
slice it" change that would destroy the statistical independence a shot vector
exists to provide. Anchored.

## Found along the way, not blocked by any test

These are ordinary cleanups. No assertion holds them, so the audit claims no
credit for them; they are recorded so the reading is not wasted.

- `device.py:247-249` duplicates the finite-shots check from `device.py:104` and
  is unreachable: the transform at `device.py:212` runs first in the pipeline,
  and every later transform preserves `tape.shots`. Replacing the condition with
  `if False:` fails no test. It reads as live validation, and confirming
  otherwise costs a reader a seven-transform ordering derivation.
- `device.py:225` and `device.py:226-228` pass textually identical lambdas as
  `stopping_condition` and `stopping_condition_shots`. PennyLane discards the
  first whenever the tape has shots, which is always.
- `_finite_parameter` and `_format_parameter` (`converter.py:202-228`) are 25
  lines across two functions, each with one caller.
- `converter.py:348-353` and `converter.py:392-397` construct `ConvertedProgram`
  twice with identical measurement-decoding arguments.

**Applied.** All four landed in `#2147`, alongside the verdicts whose commits
already touched the same lines.

## Progress

- [x] (2026-08-15) Spec ledger built from four isolated source classes, none of
      which read the tests.
- [x] (2026-08-15) Census of 26 test functions, 33 collected cases.
- [x] (2026-08-15) Independent advocate argued every accused assertion without
      sight of the prosecution.
- [x] (2026-08-15) Seven fault-injection probes executed; two overturned a
      verdict that argument alone had settled the wrong way.
- [x] (2026-08-18) All six verdicts and the four cleanups applied in `#2147`,
      one commit pair per verdict.
- [x] (2026-08-18) Reconciled against `3354fdaab`; every citation re-read and
      still resolving, every verdict marked.
- [ ] Probe the exact QASM2 payload assertion against a bumped PennyLane.

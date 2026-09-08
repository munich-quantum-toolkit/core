# Contract audit: QDMI jobs and Python plugins

Status: findings 1, 2, and 5 fixed locally; findings 3 and 4 declined by the user.
PennyLane specialist findings and Qiskit native preflight are implemented. Audit baseline:
`1bf02dd7694d511abe19032b5d80a649ab5f8d20` (merged PR #2454). Date: 2026-09-07.

Scope: native job creation/retrieval, Qiskit target and submission behavior,
PennyLane conversion and sampled execution. This is a bounded follow-up, not a
whole-repository audit. All probes used local test doubles or the SC metadata
provider; no external hardware was used.

## Confirmed findings

### 1. Retain job handles returned with a warning

`QDMI_Device_impl_d::createJob` and `retrieveJobById` in
`src/qdmi/driver/Driver.cpp` return immediately for any status other than
`QDMI_SUCCESS`. A provider returning `QDMI_WARN_GENERAL` with an allocated job
therefore leaves the client output null and loses ownership of the native job.
`qdmi::throwIfError` explicitly accepts warnings, so `Device::submitJobImpl`
continues to set parameters on a null handle; retrieval returns an unusable
`Job`. Both SDK plugins use this shared submission path.

A scratch copy of `test/qdmi/driver/session_device.cpp`, changing the two
successful job-allocation returns from `QDMI_SUCCESS` to `QDMI_WARN_GENERAL`,
reproduced both failures through the Python bindings:

- `submit_job(...)`: `ValueError: Setting program format: Invalid argument.`
- `retrieve_job_by_id("session-job").check()`:
  `ValueError: Checking job status: Invalid argument.`

Accept success and warnings before wrapping the handle, preserve the warning
status, and reject a null handle reported as successful. Test both entry points.
Keep the separate allocation-failure work in #2271 out of this fix.

### 2. Share PennyLane validation across both formats and all fixed arities

`_ProgramConverter._convert_qasm2` checks only advertised gate spellings. It
bypasses shape, finite-parameter, and placement validation performed by the
QASM3 path. `_validate_qdmi_contract` also returns immediately for gates on more
than two wires, even when explicit site tuples are present.

Normal QNode execution with the existing `StubDevice` reproduced:

- QASM2: advertise CX only on `(0, 1)`; CNOT on `(1, 0)` is submitted.
- QASM2: RX with `nan` is submitted as `rx(nan) q[0];`.
- QASM3: advertise CCX only on `(0, 1, 2)`; Toffoli on `(1, 2, 3)` is submitted.

Use one validation path before either serializer, retaining their distinct gate
spelling rules. Validate complete ordered tuples for every fixed arity. Preserve
the existing treatment of unspecified placements and finite-shot preprocessing.
These probes demonstrate invalid submissions, not incorrect hardware results.

### 3. Do not infer unrestricted MCX support from its Qiskit class

`QDMIBackend._add_operation_to_target` computes explicit placements, then
ignores them whenever the mapped gate is a class. A real SC configuration with
`{"name": "mcx", "numParameters": 0, "numQubits": 3, "sites": [[0, 1, 2]]}` on
four qubits produced a Target accepting both `(1, 2, 3)` and `(0, 1, 2, 3)`. The
latter also violates the advertised arity.

Reserve unrestricted class instructions for providers that advertise genuinely
variable arity and unrestricted placement. Represent a known fixed-arity gate as
an instance with its placements, or reject the unsupported representation. Do
not assume arbitrary classes in `_EXTRA_GATES` have a common constructor.

### 4. Preserve placements from equivalent Qiskit aliases

`_add_operation_to_target` returns when `gate_name` is already in
`seen_gate_names`. A real SC model advertising `cx` on `(0, 1)` and `cnot` on
`(1, 2)` produced a Target containing only `(0, 1)`. Both operation names are
valid, unique SC entries and map to the same canonical Qiskit gate.

Merge disjoint placements and their calibration into the existing instruction.
Keep a deliberate rule for overlapping aliases with conflicting calibration; do
not silently broaden explicit placements to global support. Retain alias
semantics through serialization and subclass overrides.

### 5. Cache PennyLane placement metadata, rather than successful gate uses

The converter caches successful `(operation.name, indices)` validations, but
each new tuple reads and reconstructs the operation's entire supported-site set.
For one operation advertised on N sites and used once on each site, validation
performs N full site-list queries and N-squared index reads.

The existing boundary doubles measured:

| Distinct sites used | Full site-list queries | Site-index reads |
| ------------------- | ---------------------- | ---------------- |
| 10                  | 10                     | 100              |
| 20                  | 20                     | 400              |

Cache normalized capability metadata once per operation/session and use set
membership for each gate. This can replace `_validated_loci` and complement the
shared validation in finding 2. Continue validating each gate's parameters and
wire arguments. Counts prove redundant work; no end-to-end speedup is claimed.

## Additional preflight hardening

`QDMIBackend.run` checks operation names but not circuit width or operation
placements. A backend whose Target rejects CX on `(1, 0)` still submitted that
circuit successfully to a test double.

Rejecting known-invalid native placements before submission would give earlier,
more useful errors. However, a blanket check against the public Target is not
safe: `_preprocess_circuit` can expand logical circuits to hidden device sites,
and custom serializers can perform additional translation. Validate the native
operation contract at the appropriate boundary while preserving those hooks.
This is a proposed tightening of preflight behavior, distinct from the confirmed
Target construction defects above; real providers may already reject the job.

## Overlap with open work

Refreshed open issues and PRs during the audit. #2226 changes payload selection,
serializer registration, and control-flow support in both plugins, but its patch
does not fix these placement or finite-parameter validation paths. Coordinate
shared converter edits with it. #2230's warning handling concerns provider and
session setup, not the two job-handle entry points above. #2229 and #2231 cover
the replaceable driver and staging; they are not prerequisites for these fixes.

PR #2373 and issues #2363/#2364 cover native multi-program batching; do not
duplicate them. PR #2233 removes obsolete calibration/pulse metadata, so a
separate duration-query cache optimization would need coordination. No additional compiler/DD changes
are justified by this investigation.

### Validation and limits

- `uv run --no-sync pytest -n0 test/python/qdmi test/python/plugins -q`: 468
  tests passed in 5.72 seconds. The existing suite does not cover the failing
  combinations above.
- `/tmp/qdmi-plugin-audit/probe.py` exercises ordinary PennyLane QNodes, real SC
  Target construction, and Qiskit submission to a boundary double.
- `/tmp/qdmi-plugin-audit/metadata.py` counts full-list queries and index reads.
- The warning provider was compiled from the existing session fixture with only
  its two successful job-allocation statuses changed, then opened in a separate
  Python process through `register_device` and `open_device`.
- These audit probes preceded the fixes below. Windows and real hardware were
  not used. No new commit or remote write was made.


## Resolution

- Finding 1: preserve warning-status native jobs in both entry points and reject
  null handles before wrapping them. The regression verifies status, ownership,
  and exactly one native free for both entry points.
- Findings 2 and 5: share preparation between QASM2 and QASM3, validate finite
  parameters and fixed-arity placements, and normalize metadata once per
  converter session. At 10/20 distinct loci, the probe now makes one site-list
  query and 10/20 index reads, compared with 10/20 queries and 100/400 reads.
- Findings 3 and 4: no changes to MCX or alias Target construction.
- Qiskit preflight: built-in QASM serializers validate native width and ordered
  placements after preprocessing. A bad circuit rejects its entire batch before
  submission. Native metadata is cached; the public Target cannot substitute for
  it because extensions can hide sites or expose fictional pairs. Custom
  serializers retain compilation control. Control-flow bodies use their parent
  qubit mapping and require explicit support in the extension's Target.

## PennyLane specialist resolution

Reviewed against PennyLane 0.45.1, both installed and
[latest stable](https://api.github.com/repos/PennyLaneAI/pennylane/releases/latest)
at the review date. All four specialist findings are addressed:

1. **Named-wire deferral:** map used wires to contiguous indices, defer using
   PennyLane's existing transform, then restore device labels. This supports
   reset and feedback with named/nonconsecutive wires and rejects insufficient
   spare-wire capacity. Real DDSIM tests cover both spare-last and spare-middle
   device wires.
2. **Graph decomposition:** derive `target_gates` from the converter's existing
   format-specific supported operation names. QASM2 and QASM3 regressions verify
   the resulting operation matrix and serialization. Placement checks remain
   authoritative; graph support does not imply routing or hardware costs. See
   the [plugin guide](https://docs.pennylane.ai/en/stable/development/plugins.html#custom-device-decompositions).
3. **Shot API:** remove device-level shots, the implicit 1024-shot default, and
   the private `_shots` assignment. QNodes and tapes must specify finite shots;
   `qp.set_shots` can override a QNode's budget. Omission fails before submission.
   Examples and tests use the modern APIs. See
   [deprecations](https://docs.pennylane.ai/en/stable/development/deprecations.html).
4. **Tracking:** use standard `Tracker.update/record` for batches, tape counts,
   accepted jobs, and shots. Shot-vector copies count individually; jobs that
   subsequently fail remain counted. See
   [Tracker](https://docs.pennylane.ai/en/stable/code/api/pennylane.Tracker.html).

The independent #2226 extraction declares deferred-only PennyLane capabilities,
so explicit one-shot/tree-traversal requests fail instead of silently changing
methods. Qiskit nested-operation validation also needs no new QDMI API. Exact
payload descriptors, feature inference, native one-shot execution, and opaque
result decoding remain in #2226. Native batching remains with #2364/#2373.

## Final validation

- `uv run --no-sync pytest -n0 test/python/qdmi test/python/plugins`: 500 passed.
- Full repository lint passes, including Ruff and ty.
- The unchanged native fix passes 104 driver tests and the warning-provider
  binding probe for both job entry points.
- `uvx nox -s cpp-lint -- e90db8be2` completed the full configured build and
  checked six C++ files with zero findings. The older base selects both changed
  driver files through the preceding PR's diff; lint inspects current contents
  with `--lines-changed-only=false`.
- The two previously failing MLIR binaries pass 500 QCO IR and 47 target-synthesis
  tests. Running formatting hooks alongside the build likely exposed partially
  rewritten headers; sequential checks pass without unrelated source changes.
- No hosted CI, Windows, or real hardware execution was performed.

See [the v4 decision record](../plans/qdmi-v4-plugin-validation.md).

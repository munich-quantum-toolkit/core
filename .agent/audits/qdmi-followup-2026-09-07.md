# Contract audit: QDMI follow-up

Status: findings implemented; validation is recorded in
[the execution plan](../plans/qdmi-followup-fixes.md). Audit baseline:
`f601bb7968c99bbe154f6c01c537b1fe1b79b5c1`, clean at start. Includes merged pull
request 2440. Date: 2026-09-07.

Scope: QDMI client/driver, SC configuration and metadata, compiler adapter and
its consumers, Qiskit target construction, and Python-to-C++ test migration.
Three production subagents and a separate test-migration subagent contributed.
DD implementation internals were excluded.

## Result

Five production findings and one test-isolation defect are confirmed. Native
semantic duplication can leave Python, but the measured savings are small. The
compiler adapter audit found no further justified change.

## Production findings

### 1. Preserve explicit higher-arity Qiskit placements

`python/mqt/core/plugins/qiskit/backend.py:573–629` reads explicit placements
only for one- and two-qubit operations. It returns `[None]` for three or more
qubits, even when the provider supplies a finite site list. Qiskit interprets
this as unrestricted placement.

A real SC configuration with four sites and one `ccx` placement `[0, 1, 2]`
produces `backend.target['ccx'] == {None: None}`. Its target reports support for
`(1, 2, 3)`, which uses a different site set and cannot be explained by gate
symmetry. CCX is a fixed-arity gate instance, not the variable-arity MCX class.
The model-only SC device is a supported source of target metadata; the probe
does not claim execution of the invalid placement.

Honor explicit tuples for fixed arities, retaining tuple-specific calibration,
or reject shapes that cannot be represented. A generic tuple path could also
replace the duplicated one-/two-qubit calibration loops. Preserve legitimate
global/variable-arity behavior and the documented subclass hooks.

### 2. Convert Qiskit instruction durations to seconds

`backend.py:501–506`, `:520–525`, and `:535–540` pass raw QDMI duration values
to Qiskit's `InstructionProperties`. QDMI `constants.h` duration-unit and
scale-factor documentation requires multiplication by the scale and
interpretation in the advertised unit. Qiskit's installed `target.py:76–77`
requires seconds. `Operation::getDuration` and its Python binding return the raw
integer, so no lower layer supplies this conversion.

The same real SC probe reports duration `20`, unit `ns`, scale `0.5`. The
resulting Qiskit duration is `20.0` seconds instead of `1e-8` seconds. Convert
all three paths using one shared conversion. An absent scale defaults to one; an
absent duration remains absent. Missing or unconvertible units must not silently
become seconds. This is a metadata-correctness finding; scheduling consequences
were not benchmarked.

Reproduce findings 1 and 2 with an inline SC configuration:

```python
config = {
    "schema-version": 1,
    "name": "QDMI target probe",
    "numQubits": 4,
    "durationUnit": {"unit": "ns", "scaleFactor": 0.5},
    "qubitProperties": {"defaults": {}, "overrides": []},
    "couplings": [],
    "operations": [
        {"name": "x", "numParameters": 0, "numQubits": 1, "duration": 20},
        {"name": "ccx", "numParameters": 0, "numQubits": 3, "sites": [[0, 1, 2]]},
    ],
}
```

Pass `json.dumps(config)` as `device_config` to `open_device('mqt.sc.default')`,
construct `QDMIBackend(device)`, and inspect `target['x'][(0,)].duration` and
`target.instruction_supported(operation_name='ccx', qargs=(1, 2, 3))`.
`/tmp/qdmi-audit2/qiskit-target.py` exited zero and reproduced both results; an
independent subagent repeated it successfully.

### 3. Cache actual loaded provider identity

`src/qdmi/driver/Driver.cpp:180–189` keys the cache by a requested path resolved
against the current directory, while `dlopen` resolves a bare library name
through its search path. An absolute name and a loader-search-path basename can
therefore create two wrappers for one module and prefix. Both initialize the
same provider; separate destruction can also finalize it prematurely.

A test provider derived from `test/qdmi/driver/session_device.cpp`, with an
exported initialization counter, was registered by absolute path and basename
with the same `TEST_SESSION` prefix. With its directory in `LD_LIBRARY_PATH`,
`/tmp/qdmi-client-audit/alias` printed
`same provider initialized 2 times; same wrapper=0` and exited zero. The root
repeated the result. QDMI `device.h:63–75` guarantees exactly-once
initialization and finalization.

Determine identity from the loaded module before initialization and keep the
symbol prefix in the cache key: different prefixes may represent different
providers. Preserve bare-name loading. Retain the existing `/./` alias test and
add an actual loader-search-path alias regression. Linux was reproduced; Windows
needs its corresponding loader-resolution check.

### 4. Reject embedded NULs in SC names

`src/qdmi/devices/sc/Configuration.cpp:188,241,305` checks complete C++ strings
for nonempty and unique names, but the C API exports them through `c_str()` and
`strlen`. Escaped JSON NULs bypass those public-name requirements.

`/tmp/qdmi-sc-nul-probe.py` initialized device/site names beginning with NUL and
two operation names `x` and `x\u0000different`. Queries returned empty
device/site names and duplicate operation names `x`. Initialization returned
success; the probe exited zero, and the root repeated it. This violates the
nonempty/unique-name contracts in `docs/qdmi/sc_device.md:59,70`.

Reject embedded NUL alongside the existing checks for all three name fields.
Retain ordinary UTF-8 support. The acceptance change affects only names that the
C API cannot represent faithfully.

### 5. Reuse existing parser membership sets

`Configuration.cpp:379–382` scans every coupling for each calibration override
although `uniqueCouplings` already contains the same directed edges. Replace
that search with `uniqueCouplings.contains({first, second})`, reducing the
membership work from O(overrides × edges) to O(overrides × log(edges)) without
changing public ordering or directed-edge semantics.

An independently compiled baseline parser and scratch variant with only that
expression changed produced these median whole-file parse times over five runs,
with one override per edge:

| Edges/overrides |  Baseline | Existing-set lookup |
| --------------: | --------: | ------------------: |
|           2,000 |   5.55 ms |             4.79 ms |
|           8,000 |  27.42 ms |            19.00 ms |
|          32,000 | 177.41 ms |            43.56 ms |

Both variants used `g++ -O3 -std=c++20`; every run exited zero. Harness:
`/tmp/qdmi-sc-parser-bench.cpp`; binaries `/tmp/qdmi-sc-parser-{old,new}`;
inputs `/tmp/qdmi-sc-bench-{2000,8000,32000}.json`. This demonstrates a large
synthetic-model benefit, not a material speedup for bundled 20–100-site models.
The explicit-site branch at `:374` could similarly retain its already-created
`uniqueSites` set; that variant was not benchmarked.

## Test isolation and migration

### Fix process-global registry pollution first

The registration tests at `test/python/qdmi/test_qdmi.py:896,908,917` leave
nonexistent libraries in the process-global driver. Later Qiskit provider
enumeration warns when it opens them; warnings-as-errors fails the test.

```sh
uv run --no-sync pytest -n 0 \
  test/python/qdmi/test_qdmi.py::test_register_device_does_not_load_nonexistent_library \
  test/python/plugins/qiskit/test_provider.py::test_provider_default_constructor
```

Both the subagent and root observed one pass and one failure. Reversing the
order produced two passes. Running the QDMI tree before plugins produced 458
passes and five failures. This proves order dependence; no hosted xdist failure
was established.

C++ already covers deferred loading, duplicate/idempotent validation, and
ordered enumeration in `test/qdmi/driver/test_driver.cpp:918,944,1040`.
Consolidate the Python registration boundary checks in one isolated subprocess
smoke while retaining constructor/path conversions, boolean returns, and
exception translation. Leave exhaustive native semantics in C++. Do not add a
production unregister/reset API solely for test cleanup. Subprocess isolation
may cost more; its benefit is reliability.

### Small native semantic copies can leave Python

| Python test in `test/python/qdmi/test_qdmi.py` | Surviving C++ oracle in `test/qdmi/test_client.cpp` |
| ---------------------------------------------- | --------------------------------------------------- |
| `test_device_submit_job_preserves_num_shots`   | `DDSimulatorDeviceTest.SubmitJobPreservesNumShots`  |
| `test_job_ids_are_unique`                      | `JobTest.IdIsUnique`                                |
| `test_job_get_counts_is_consistent`            | `JobTest.MultipleGetCountsCalls`                    |

Keep the Python valid-job smoke, omitted-shot argument branch, histogram
conversion, and shots/counts agreement. The three redundant cases totalled 0.040
seconds per serial suite run, including setup/teardown: about 1.28 aggregate
runner-seconds across 32 repeats on this machine.

The four Bell-result tests at `:838–895` can share one job in one Python smoke,
retaining all four bound getters and their value/type checks. Their combined
measured cost was 0.044 seconds; savings after retaining one execution are
necessarily smaller. Neither cleanup enables production simplification.

Keep Qiskit and PennyLane integration tests in Python, including serializers,
wire/register ordering, sampling, gradients, shot vectors, and SDK result
assembly. Keep bytes/text payloads, overloads, exception translation, optional
arguments, paths, and lifetime coverage at the Python boundary. Binding-only
changes do not trigger C++ tests under the pinned change-detection workflow.

### Frequency and timing limits

Ready-PR/main CI has four platforms × two Nox sessions × four Python versions:
32 Python suites, versus seven C++ matrix configurations plus coverage. Draft
Python CI has four suites. Wheel tests add further supported-wheel pytest runs,
with import-only exceptions for `cp315*` and Windows ARM64. Sources:
`.github/workflows/ci.yml`, `noxfile.py`, `pyproject.toml`, and the pinned
reusable Python workflow at `9d0a300a990563b5d70b6cd5e7549c3e678d3cf7`.

A clean serial run with plugin directories before QDMI passed 463 tests in 5.32
seconds. Recorded case/setup/teardown totals were about 0.52 seconds for
QDMI/Slurm, 3.60 for Qiskit, and 0.43 for PennyLane. XML/log:
`/tmp/qdmi-audit-migration-clean.{xml,log}`. CI uses xdist; aggregate work saved
is not the same as wall time saved. Python process/import overhead remains.

## Deferred and rejected candidates

- PR #2230 already replaces the weak library cache with strong ownership,
  addressing a separately reproduced initialize-during-finalize race. Its
  current patch still keys by requested path, so finding 3 remains distinct.
- PRs #2226, #2227, #2229, #2231, #2233, and #2373, plus issues #2271,
  #2363/#2364, and #2367, cover adjacent payload, driver, staging, metadata,
  batching, and allocation-failure work. Do not duplicate those scopes. The
  inspected Qiskit changes in #2226 do not repair findings 1 or 2; #2233 does
  not remove the SC name parser or operation-duration contract.
- Dropping uncalibrated compiler placement lists would widen one-way CX to
  unrestricted support. `Target.cpp:690` and
  `PreservesOneWayDirectionalOperationSupport` show why those lists must stay.
- Removing higher-arity homogeneity checks expands the documented compiler
  adapter contract. It is not a behavior-preserving simplification.
- Eager all-pairs topology distances remain a known scaling limit, but this
  round establishes no additional QDMI-specific case for changing them.
- SC job-object removal conflicts with the current create-job contract and needs
  an explicit behavior decision; it is not a confirmed safe deletion.

## Validation provenance

Python was rebuilt at the baseline with
`uv sync --inexact --no-dev --no-build-isolation-package mqt-core`. Native
driver/SC probes used existing builds whose relevant production sources were
unchanged from the baseline;
`git diff 55d3c56ad..HEAD -- src/qdmi include/mqt-core/qdmi` was empty. The
compiler adapter audit was source-based; its old binary was not counted as
current validation because `Target.cpp` changed after that build.

The initial audit was read-only. The accepted fixes and independent driver
extractions are recorded in the execution plan.

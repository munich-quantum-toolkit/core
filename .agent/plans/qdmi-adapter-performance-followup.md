# QDMI adapter performance follow-up

Status: proposed. Baseline: `7681f45` (2026-11-08). Nothing below is measured on
this baseline yet. Each item is a hypothesis with the probe that settles it.

The first round is recorded in
[qdmi-adapter-performance](qdmi-adapter-performance.md). Read it first; this
plan does not repeat its evidence.

## Status of the first round

| First-round item                       | State on `7681f45`                                                        |
| -------------------------------------- | ------------------------------------------------------------------------- |
| Scope DDSIM PennyLane sessions         | Done (#2960)                                                              |
| Native Qiskit artifact submission      | Done (#2955, #2960): `prepareProgramSubmission`, `_try_submit_programs`   |
| Provider name filtering                | Done (#2960)                                                              |
| Single site materialization            | Done (#2960)                                                              |
| Shared decoding and Catalyst readout   | Done (#2953, #2954, #2960)                                                |
| Compatibility identity fast path       | Rejected: measured too small                                              |
| Shot-vector copies as separate entries | Retained by decision                                                      |
| Per-tape compilation profile           | Measured; parallel batch helper proposed, not built (item 1 below)        |
| Qiskit native-run validation split     | Not measured (item 3 below)                                               |
| Import-once sampler                    | Prototyped (659 ms to 533 ms on W5), not built (item 4 below)             |

The record's main conclusion still holds: with session scoping in place,
compilation is the largest adapter cost.

## Scope and fixed decisions

Same files and ownership rules as the first round. The Qiskit backend, PennyLane
device, Catalyst runtime device, `qdmi_batch.Batch`, and the native helpers in
`mlir/lib/Compiler/QDMIAdapter.cpp` and `bindings/mlir/register_mlir.cpp`.

These decisions stay in force; do not reopen them without new evidence:

- No payload caching, shared MLIR contexts between independent programs, or a
  general cache framework
  ([compile-program-sequences](compile-program-sequences.md)).
- Sequential QDMI submission; no global provider lock or second scheduler.
- DDSIM keeps its 65,535-site default; session `custom1` sets the width.
- No identity fast path in `validateTargetCompatibility`; no decoder cache
  changes in `decodeShots` (both measured in the first round).
- Shot-vector copies remain separate `Batch` entries.

## Related open work

- #2791 adds native compiler profiling. Use it for item 2 if it lands first; do
  not build a second profiler.
- #2974 prepares a release. Keep changes out of release preparation.
- #2226, #2227, and #2626 change payload contracts, snapshots, and binary
  outputs; they touch the code of items 1 and 8.
- #2688 and #2689 change placement cost, which affects item 2 numbers.

## Measurement protocol

As in the first round: `cmake --preset release`, editable install, harnesses
outside the repository, the median of at least five alternating runs, and a
record of the environment and dependency versions. The host may be shared, so
report alternating paired runs and do not claim portable throughput.

| ID  | Workload                                                                                                                                                            |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| W1  | `DDSIMDevice(wires=4)`, 4-qubit RY/CNOT ansatz, 8 parameters, parameter-shift gradient, 1,000 shots                                                                 |
| W2  | W1 circuit under `qp.qjit`, 50 calls with new parameters                                                                                                            |
| W4  | `backend.run` of native 2-qubit RX/RZ/CX circuits with 10,000 and 100,000 gates (DDSIM, `custom1="2"`)                                                              |
| W5  | `backend.sampler().run`, 50-gate parametrized circuit, 200 bindings                                                                                                 |
| W7  | 20-qubit circuit, 10^6 shots: PennyLane `qp.expval` and `qp.sample`; Qiskit `memory=True`                                                                           |
| W8  | Full-width DDSIM backend from `QDMIProvider().get_backend(...)`, Bell circuit through `generate_preset_pass_manager(backend=backend)`, then `backend.sampler().run` |

## Investigation items

### 1. Native batch compilation (P1, ready for maintainer go-ahead)

Problem: `compilePythonProgram` compiles a sequence one program at a time.
`QDMIDevice._prepare` calls `compile_program` once per tape, so it cannot
benefit even from a batch API. A four-worker prototype in the first round cut W1
compile time from 35.94 ms to 10.79 ms and the gradient from 69.49 ms to 43.83
ms.

Design (from the first-round record):

- Add one native helper that compiles a list of owned inputs against one
  `TargetEnvironment`. It uses the existing shared `llvm::DefaultThreadPool` in
  `createCompilerContext` (`mlir/lib/Compiler/Programs.cpp`).
- Import stays under the GIL. Compilation runs without the GIL.
- Results keep input order. An error names its input index, and the first error
  by index wins, so the outcome does not depend on scheduling.
- Inputs that share an `MLIRContext` (program copies) compile serially. Only
  independently imported programs run in parallel.
- Route `compilePythonProgram` sequences and `QDMIDevice._prepare` through it.
  `_prepare` must still skip tapes without measurements before it compiles.

Must check before parallel use: static state reached from the pipeline
(`registerMQTCompilerPasses`, the `qpyExporter` hook, diagnostic handlers). QPY
output calls into Python, so it needs the GIL or a serial fallback. Run the
compiler suites under ThreadSanitizer once.

Must keep: per-input consumption and copy semantics, `TranslationError` for the
failing tape, deterministic output, and GIL release around native work. Catalyst
makes one synchronous call per execution and does not benefit (see item 2).

Acceptance: W1 compile and gradient medians improve over serial on the same
host. Identical payloads for all inputs. Indexed error tests for an early and a
late failing input.

### 2. Fixed per-program compile overhead (P1, measure first)

Problem: after scoping, one small W1 tape costs about 2.2 ms to compile. Each
Catalyst call (W2) pays this serially, and item 1 cannot help there. Nobody has
measured where the time goes. Candidate sources:

- `createCompilerContext` loads all 15 registered dialects, including the LLVM
  dialect, for every program (`loadAllAvailableDialects`), even when the payload
  is OpenQASM.
- Each pipeline stage builds a fresh `PassManager` in `runWithPassManager`
  (`mlir/lib/Support/Passes.cpp`). Earlier work measured about 20 µs of
  canonicalizer pattern freezing per fresh pass manager
  ([compiler-canonicalization-cost](compiler-canonicalization-cost.md)).
- Each stage ends with a full `verify(mod)`, even in release builds.

Probe: W2 and one W1 tape. Use `CompilationOptions(enable_timing=True)` (or

## 2791) for pass time. Time context creation, import, the number of pass

managers per `CompiledProgram::compile`, verification, and export separately.
Repeat with a 50-qubit, 1,000-gate circuit, so that fixed and per-gate costs can
be told apart.

Possible fixes, each only with numbers:

- Load only the dialects a program needs, and rely on passes declaring their
  dependent dialects. The risk is a pass that creates operations from an
  undeclared dialect; the full suites catch this.
- Merge adjacent stages that differ only in their pass lists.
- Keep the final verification. Dropping intermediate verification moves error
  attribution and needs maintainer approval.

No context sharing or pass-manager cache (fixed decision above).

### 3. Qiskit native-run validation split (P2, measure only)

Problem: `QDMIBackend.run` validates each instruction in Python
(`validate_circuit(native=True)`, which calls `Target.instruction_supported`),
then imports natively in three passes and serializes. #2957 enforces native
placement in exports to Qiskit, not in `serialize_program`, which still
documents that the caller validates.

Probe: W4 phase split: `assign_parameters`, `validate_circuit`,
`serialize_program`, and submission.

Follow-up only if validation dominates: native validation during serialization
against `CompilerTarget::supportsOperation`. This changes validation ownership,
so it needs maintainer approval. Error types and messages stay the same.

#### 4. Import-once sampler (P2, maintainer decision)

The first-round prototype kept all 200 W5 payloads identical and cut the median
from 659 ms to 533 ms. Estimated size: 120–180 production lines plus 100–180
test lines, using private Qiskit hooks that match in Qiskit 2.5.0 and 2.6.1.

Decision needed: is a 19% W5 gain worth an internal `BackendSamplerV2` subclass
and future Qiskit compatibility work? If yes, the implementation must keep PUB
shapes, registers, metadata, per-PUB shots, preprocessing overrides, and `Batch`
recovery. Re-measure after items 1 and 2 first; they reduce the same per-binding
cost without a new commitment.

#### 5. Aggregate-first readout in the PennyLane Python path (P2)

Problem: the three consumers use different result policies.

- Catalyst prefers counts for aggregate measurements and samples only when
  needed (#2953).
- Qiskit reads counts unless `memory=True`.
- `PennyLaneJob._samples` always calls `samples()`, which requests SHOTS first
  and falls back to counts. The result is a shots-by-wires array even when every
  measurement in the tape is an expectation, variance, probability, or count.

For histogram-native providers and large shot counts, this transfers and decodes
one string per shot instead of one per distinct outcome.

Probe: W7 PennyLane `qp.expval`, and a mock provider that counts SHOTS versus
histogram queries.

Fix if material: when every tape measurement can be computed from counts, read
counts through the existing `decode_counts` binding and use PennyLane's
`process_counts`. Otherwise, keep the current samples path. Match the Catalyst
fallback order and error text.

Must verify: `process_counts` exists with the needed signature for every used
measurement class in PennyLane 0.45.1 and the latest supported release. Count
keys must use PennyLane wire order: `decode_counts` returns logical MSB-first
strings, and `samples()` currently reverses them. Shot-vector entries and
`Batch` recovery stay unchanged.

#### 6. Full-width Qiskit backends and ISA circuits (P2, measure first)

Problem: `QDMIProvider` cannot know a desired width, so the DDSIM backend it
returns has a 65,535-qubit `Target`. Qiskit's preset pass managers allocate
ancillas up to the target width. The ISA circuits that users pass to
`backend.sampler()` may then have 65,535 qubits. `validate_circuit`, native
import, and serialization would then scale with idle qubits.

Probe: W8. Record the transpiled `num_qubits` and the `backend.run` phase split,
compared with a `custom1="2"` backend.

Fix if confirmed (smallest first):

1. Documentation in `docs/qdmi/qdmi_backend.md` and `docs/qdmi/ddsim_device.md`
   that steers primitives users to `from_device_id(..., custom1=...)`.
2. Make validation and import skip idle qubits without changing site IDs. This
   is only safe if exports and result layouts never depend on idle wires. Prove
   that first.

The 65,535-site default stays.

#### 7. Qiskit memory readout round trip (P3)

Problem: with `memory=True`, `QDMIJob._collect_result` copies one shot list
three times. `Job.get_shots` builds a Python list. `decode_shots` copies it into
`std::vector<std::string>` and returns a new list. Python then deduplicates and
hex-encodes.

Probe: W7 Qiskit `memory=True`. Report the share of the time before Qiskit's own
`BackendSamplerV2` post-processing. Act only if that share is material.

Fix if material: one native call from the job to decoded shots, so the list is
built once. Leave the decoder itself unchanged (fixed decision).

#### 8. Text payload copy at submission (P3, probe only if cheap)

`trySubmitPrograms` copies every text payload into a `std::vector<std::string>`,
because `qdmi::Device::trySubmitJob` takes `std::span<const std::string>`.
Single-program groups could use the `const std::string&` overload. For several
programs, a view-based overload is a Client API change; coordinate with #2626.
The expected saving is about one memory copy of the payload. Measure a
100,000-gate OpenQASM payload; drop the item if the copy is below 1% of
submission time.

### Milestones

- [ ] Baseline phase splits for W1, W2, W4, W5, W7, and W8 on `7681f45`. Share
      the table before changing code.
- [ ] Item 1 after maintainer go-ahead; item 2 changes only with numbers.
- [ ] Items 3, 5, 6: report the measurements; implement 5 and the item 6 docs if
      confirmed.
- [ ] Items 4, 7, 8: report the numbers and the decision needed.

### Validation for any change

- `uv run --no-sync pytest test/python/plugins test/python/qdmi test/python/test_mlir.py`.
- `ctest --preset release`, including the Catalyst runtime tests and
  `mlir/unittests/Compiler/test_compiler_qdmi_adapter.cpp`.
- ThreadSanitizer run of the compiler suites for item 1.
- `uvx nox -s stubs` after binding changes, `uvx nox -s cpp-lint`, and
  `uvx nox -s lint`.
- Re-run the affected workloads and report the before and after medians with the
  environment. A change without a measured saving does not land.

# Keynote compiler changes

All four compiler changes prepared for this keynote are merged into upstream
`main`. The remaining native changes are optional compiler and DDSIM tracing.
The capture retains the revision that produced each artifact; rebasing the
presentation does not relabel those observations as a new run.

## Upstream compiler changes

| Change                                    | Upstream PR                                                       | Scope                                                                                                                                                             |
| ----------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Adjacent controlled-gate canonicalization | [#2752](https://github.com/munich-quantum-toolkit/core/pull/2752) | Combines matching control shells while preserving wire order and controlled relative phase. Target-aware synthesis retains responsibility for native constraints. |
| Fixed equatorial-pulse synthesis          | [#2753](https://github.com/munich-quantum-toolkit/core/pull/2753) | Adds the `GPI` synthesis basis for fixed π and π/2 equatorial pulses, including runtime-angle synthesis and full-unitary tests.                                   |
| Explicit compiler target on the CLI       | [#2754](https://github.com/munich-quantum-toolkit/core/pull/2754) | Accepts validated `--target` attributes and Python target serialization, including empty-input diagnostics.                                                       |
| Scoped classical storage and source names | [#2755](https://github.com/munich-quantum-toolkit/core/pull/2755) | Preserves local OpenQASM declarations and labels, and initializes reusable Qiskit storage at each region entry.                                                   |

The changes use the existing compiler and exporter infrastructure. Their PRs
contain the focused diffs and semantic regression coverage. They do not add
general runtime bounds or classical indexing to Qiskit export.

Relative to main `ee77292f8`, compiler tracing adds **305 lines and removes 5**;
DDSIM tracing adds **103 lines and removes 9**. Their native tests and CMake
configuration add **505 lines and remove 4**. Counts include comments and
includes, and exclude presentation sources, capture scripts, and Python tests.

## Presentation telemetry

These are intentional keynote patches with regression coverage,
**not a proposed stable tracing API**. They are disabled unless their
environment variable is set.

| Trace                         | Implementation scope                                                                                                                | What it records and its limits                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MQT_MQSF_MAPPING_TRACE`      | 219 added / 5 deleted lines in `Mapping.cpp`                                                                                        | Actual trial layouts, forward/backward passes, selected trial, routed gates/SWAPs, and a bounded search sample. The search records only the first search per routing state and retains at most 256 generated-node descriptions; truncation is reported. Trial records are emitted after parallel refinement to prevent interleaving. JSON/stderr schema and overhead are presentation-specific. |
| `MQT_MQSF_SYNTHESIS_TRACE`    | 86 added lines in `TargetSynthesis.cpp`                                                                                             | Accepted two-qubit fusions: exact before/after IR, operation indices, composed matrix, and native basis use. The recorder disables MLIR multithreading; concurrent writers and a durable public trace format are outside its contract.                                                                                                                                                          |
| DDSIM program and shot traces | 103 added / 9 deleted implementation lines across `Device.cpp` and `WorkerMain.cpp`; 213 added / 2 deleted test/configuration lines | Real program completions, QDMI/JIT timing, worker progress, and indexed results. Adaptive execution retains native parallel workers and samples in 64-shot recording batches. Timestamps are batch-completion observations, not individual-shot execution timestamps. Recording overhead is included.                                                                                           |

Mapping tests compare traced/untraced output and replay the emitted routing
steps. Synthesis tests compare full unitaries, exact unchanged final IR, and
highlight indices. DDSIM tests compare seeded traced/untraced ordered results,
validate every indexed arrival and worker partition, and exercise unwritable
trace paths. The worker-partition test honors the runtime's actual worker count:
requesting four workers does not override a two-CPU runner or an explicit worker
budget. Production sampling is unchanged by that fix.

## Capture assumptions and work that is not a compiler fix

- Target models and naming/parameter conversions live in
  [capture_devices.py](https://github.com/munich-quantum-toolkit/core/blob/codex/mqsf-2026/presentations/mqsf2026/capture_devices.py).
  The three hardware examples compile against SC QDMI mock targets assembled
  from provider metadata. DDSIM executes the resulting programs. This is not
  execution or calibration validation on IQM, IBM, or IonQ hardware.
- IonQ GPI/GPI2 are represented internally as fixed-angle equatorial rotations;
  the recorder exports the corresponding native gate definitions and accounts
  for their phase convention. Provider metadata calls its entangler `zz`; the
  presentation uses the equivalent `rzz` angle convention. These adapters are
  presentation integration code, not a new provider implementation.
- The structured syndrome and iterative-QPE examples use portable adaptive QIR.
  Their captures do not establish arbitrary structured-program support on the
  three hardware targets.
- The new nested-loop iQPE needs **no native compiler patch**. Its eight-round
  outer loop contains a runtime-dependent controlled-U power loop, reset,
  measurement, and floating-point feedback. Both loops survive into executable
  adaptive QIR. Qiskit export currently cannot represent this combination of
  dynamic bounds, classical indexing, and runtime feedback; its diagram is
  derived from the submitted OpenQASM source instead.

Keep these trace mechanisms on the keynote branch unless a concrete public
telemetry use case justifies a shared design. Their schemas, bounds, and timing
cost are tailored to this presentation.

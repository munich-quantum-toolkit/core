# Fixed-parameter compiler targets

Status: implementation and local validation complete on main `bb0bf98dc`.

## Scope and ownership

[Core #2575](https://github.com/munich-quantum-toolkit/core/pull/2575) supports
fixed operation parameters, multiple native alternatives, Qiskit target
exchange, and RX gate synthesis with RZ and an existing entangler such as CZ.
Current Braket devices guide coverage. Existing Core `R`, `RXX`, `RZZ`, and `CZ`
represent the corresponding PRX, XX, ZZ, and CZ operations; representing an
operation does not establish a complete synthesis basis or provider execution
support.

`CompilerTarget` owns capability validation and basis selection. Native
synthesis owns phase-correct lowering. Qiskit import/export owns named
instruction alternatives. Provider adapters own verbatim syntax, physical
labels, parameter domains, and submission constraints.

## Device evidence

An unfiltered, paginated AWS `SearchDevices` scan on 2026-10-01 at 23:55 UTC
covered all five [Braket regions][devices]: `us-east-1`, `us-west-1`,
`us-west-2`, `eu-north-1`, and `eu-west-2`. `GetDevice` inspected every online
regional entry. The six online gate QPUs advertised these
`paradigm.nativeGateSet` values:

| Device                  | Region     | Native operations                              |
| ----------------------- | ---------- | ---------------------------------------------- |
| Rigetti Cepheus-1-108Q  | us-west-1  | `rx`, `rz`, `cz`, `barrier`                    |
| IonQ Forte-1            | us-east-1  | `GPI`, `GPI2`, `ZZ`                            |
| IonQ Forte-Enterprise-1 | us-east-1  | `GPI`, `GPI2`, `ZZ`                            |
| IQM Garnet              | eu-north-1 | `cz`, `prx`, `cc_prx`, `measure_ff`, `barrier` |
| IQM Emerald             | eu-north-1 | `cz`, `prx`, `cc_prx`, `measure_ff`, `barrier` |
| AQT IBEX Q1             | eu-north-1 | `prx`, `xx`, `rz`                              |

Aquila was online but accepts analog Hamiltonian programs. SV1 and DM1 were
online simulators, with no native gate set. Ankaa-3 and IonQ Aria were retired.
Use Cepheus/CZ for the current Rigetti example; retain general iSWAP support.

All six gate QPUs advertised verbatim, physical qubits, sparse indices, and
subset measurements. Partial verbatim boxes were enabled except on IonQ. Only
IonQ allowed unassigned measurements. The broader `supportedOperations` list
describes compiled inputs and must not replace the native whitelist. Physical
labels also need translation: Cepheus had 107 active sites in `0..107`,
excluding `8`, while IQM labels started at `1`. Refresh topology instead of
embedding these labels.

[AWS documents][gates] Rigetti RX angles as ±π/2 and ±π. The live gate metadata
contains no numerical parameter domains; no additional Braket acceptance bounds
were verified. In particular, direct AQT API bounds and retired IonQ MS bounds
must not be assumed to constrain Braket Forte ZZ. Braket angles use radians.

## Implementation decisions

The closest ecosystem model is [Qiskit's target][qiskit-target]: standard gate
instances carry fixed parameters and distinct names for alternatives. PennyLane
separates [operation capabilities][pennylane-capabilities] from
[decomposition predicates][pennylane-decomposition]. CUDA-Q selects
[backend lowering passes][cudaq-backend]; Microsoft's neutral-atom simulator
uses [SX/RZ/CZ lowering][qdk-native]. These support separating capability facts,
small decomposition recipes, and provider execution rules; they do not justify a
general fixed-angle solver or a common schema across all SDKs.

- Keep finite fixed values per parameter. Omitted values are unrestricted;
  repeated capabilities form a union across values and placements. Unbound
  parameters cannot satisfy fixed constraints. Preserve absolute matching
  tolerance and phase; do not wrap parameters during capability matching.
- Reuse the existing `ZSXX` decomposition for IBM SX/X and fixed RX gates with
  unrestricted RZ. Replace `FixedRotation` with optional gate choices in the
  synthesis configuration. Account for the phase between SX/X and RX exactly.
  Both quarter-turn signs and optional half turns are valid choices; emission
  and routing costs must use the selected capabilities.
- Require a complete supported synthesis basis for target compilation. This
  permits ordinary canonicalization before final native lowering and removes
  special cleanup passes for incomplete native-only targets. Capability queries
  can still represent such targets. Final output must satisfy the target.
- Preserve named Qiskit instructions and their constant parameters on target
  import and export. Multiple RX variants must survive together, including
  different placements. Do not infer operation semantics from an arbitrary name.
- Keep general Euler helpers independent of target policy. Do not introduce an
  arbitrary-angle solver, rational angle representation, or six-axis gate
  framework without a concrete device need. Rewrite user documentation around
  the supported contract and examples.

[Core #2578](https://github.com/munich-quantum-toolkit/core/pull/2578) is
already stacked on #2575 and owns GPI/GPI2 coverage. On rebase, reassess its
unreleased turn-based gates: prefer radians and existing RZZ, prioritizing Forte
over retired Aria MS while preserving GPi's phase relative to `R(π, φ)`. Its GPI
gate recipe is independent of fixed RX synthesis. Existing R/PRX covers IQM and
AQT unitary representations. Provider serialization and IQM
[experimental feedforward][dynamic] remain separate work, including feedback
groups and result semantics. Follow [verbatim rules][verbatim] at the adapter
boundary; fixed parameters do not provide complete execution support.

## Validation

- [x] Finish capability-driven ZSXX lowering and remove obsolete cleanup APIs.
- [x] Validate multiple named/fixed Qiskit capabilities through target round
      trips.
- [x] Verify phase, signs, optional half turns, symbolic inputs, routing costs,
      and rejection of unsupported compilation targets; review the final diff.
- [x] Regenerate binding stubs and run the required native, Python, and lint
      checks.

Local validation on the revised implementation passed 985 native tests (254
compiler, 86 native synthesis, 126 mapping, 315 decomposition, 204 optimization)
and 631 Python MLIR/Qiskit tests. Binding stubs were regenerated. Independent
correctness and Ponytail reviews found a missing `canonical_name` stub override
and obsolete includes; both were corrected. No remaining design or correctness
finding was identified. Repository lint and full changed-file C++ lint passed.

The docs session now rebuilds the local package with documentation generation
enabled, preventing cached builds from omitting MLIR reference pages. The full
executable documentation build passed with no Sphinx warnings, followed by the
generated-page link check.

Refresh the device catalogue with these read-only commands; keep AWS CLI
pagination enabled and repeat the lookup for each returned online device ARN:

```sh
for region in us-east-1 us-west-1 us-west-2 eu-north-1 eu-west-2; do
  aws braket search-devices --profile braket --region "$region" --filters '[]'
done
aws braket get-device --profile braket --region "$region" --device-arn "$device_arn"
```

After building the release preset, run:

```sh
build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler
build/release/mlir/unittests/Dialect/QCO/Transforms/NativeSynthesis/mqt-core-mlir-unittest-target-synthesis
build/release/mlir/unittests/Dialect/QCO/Transforms/Mapping/mqt-core-mlir-unittest-mapping
build/release/mlir/unittests/Dialect/QCO/Transforms/Decomposition/mqt-core-mlir-unittest-decomposition
build/release/mlir/unittests/Dialect/QCO/Transforms/Optimizations/mqt-core-mlir-unittest-optimizations
QISKIT_NUM_PROCS=1 uv run --no-sync pytest test/python/test_mlir*.py
uvx nox -s stubs
uvx nox -s cpp-lint
uvx nox -s lint
uvx nox --non-interactive -s docs
```

Adapter tests must also inspect native spellings, radians, placements, and
rules.
[devices]: https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html
[gates]: https://docs.aws.amazon.com/braket/latest/developerguide/braket-submit-tasks.html
[dynamic]: https://docs.aws.amazon.com/braket/latest/developerguide/braket-experimental-capabilities.html
[verbatim]: https://docs.aws.amazon.com/braket/latest/developerguide/braket-constructing-circuit.html#verbatim-compilation
[qiskit-target]: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.Target
[pennylane-capabilities]: https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/devices/capabilities.py
[pennylane-decomposition]: https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/devices/preprocess.py
[cudaq-backend]: https://nvidia.github.io/cuda-quantum/latest/using/extending/backend.html
[qdk-native]: https://learn.microsoft.com/en-us/azure/quantum/neutral-atom-noise-models

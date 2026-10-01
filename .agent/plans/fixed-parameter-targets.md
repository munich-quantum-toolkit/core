# Fixed-parameter targets for Bench and Braket

Status: in progress. The #2575 simplification is implemented and tested. Native
ion gates and provider integration remain proposed follow-ups.

## Goal and scope

Compile MQT Bench circuits for its device gate sets, prioritizing native gates
exposed by Amazon Braket for verbatim execution. Separate three contracts:
representing a native operation, synthesizing into a supported native basis, and
serializing a program the provider accepts. Completion requires checking all
three; a compiler target or generic OpenQASM round trip alone is insufficient.

Keep [Core #2575](https://github.com/munich-quantum-toolkit/core/pull/2575)
focused on fixed-parameter capabilities and Rigetti synthesis. Coordinate native
ion gates through
[Core #2578](https://github.com/munich-quantum-toolkit/core/pull/2578), then
complete Bench import and Braket export in their owning adapters. This plan
records the dependency order without combining those changes into one PR.

The initial coverage matrix follows the
[Braket native-gate catalogue](https://docs.aws.amazon.com/braket/latest/developerguide/braket-submit-tasks.html)
and
[Bench target definitions](https://github.com/munich-quantum-toolkit/bench/tree/d23df0d5d2791b44633858783e62a7468480f362/src/mqt/bench/targets).

| Target family                  | Required native gates                             | Core representation and work                                                                          |
| ------------------------------ | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Rigetti Ankaa / Bench Ankaa-84 | `rx`, `rz`, `iswap`; RX restricted to ±π/2 and ±π | Constrained `RX`, unrestricted `RZ`, existing `iSWAP`; Bench exposes +π and ±π/2 through named gates. |
| IonQ Forte                     | `gpi`, `gpi2`, `zz`                               | Native `GPI`/`GPI2`; reuse `RZZ` for Braket `zz`.                                                     |
| Bench IonQ Aria                | `gpi`, `gpi2`, `ms`                               | Retain native `MS` support even though the current Braket catalogue lists Forte.                      |
| IQM Garnet / Emerald           | `prx`, `cz`                                       | Reuse Core `R` and `CZ`; check synthesis and external spelling.                                       |
| AQT IBEX-Q1                    | `prx`, `xx`, `rz`                                 | Reuse Core `R`, `RXX`, and `RZ`; check synthesis and external spelling.                               |

Device fixtures must record their identity and capability snapshot. Derive a
verbatim target from `paradigm.nativeGateSet`, topology, and documented
parameter restrictions, not the broader service `supportedOperations` list. The
current Braket catalogue also lists Rigetti Cepheus, but only specifies Ankaa's
native set; obtain its device metadata before claiming complete current-device
coverage. Preserve existing support for other Bench targets, including IBM and
Quantinuum. QuEra analog programs, pulse control, and general approximate
Clifford+T synthesis are outside this work.

## Decisions and follow-up design

### Keep capability matching general; keep synthesis small

Retain optional finite fixed values per parameter. An omitted value means an
unrestricted parameter; separate capabilities express alternatives and
placements. Matching, serialization, compatibility, and final verification must
agree. Unbound symbolic values cannot satisfy a fixed parameter. Use the
existing absolute comparison tolerance, without periodic wrapping that loses
phase.

In #2575, `mlir/lib/Compiler/Target.cpp` selects unrestricted RZ with fixed RX
quarter turns and optional half turns. The shared Euler recipe preserves phase
for both pulse signs. The arbitrary-angle solver, 64-pulse fallback, coordinate
transforms, and variable-length pulse plans are removed. The target stores only
the chosen quarter-turn angle and optional half-turn angle. Routing costs and
numeric/symbolic emission use the same chosen basis.

Most device constants being dyadic fractions of π does not require a rational
angle type, angle snapping, or synthesis for every dyadic fraction. Keep
ordinary floating-point constants. Add another explicit recipe, such as repeated
π/4 pulses, only when an identified device requires it. Other declared fixed
values remain valid for native matching without automatically becoming synthesis
bases.

Structural/classical cleanup, target-aware inlining, and QCO lifetime and
control-flow canonicalization remain available. Native sequences are preserved
when no universal basis exists. Restricted cleanup merges or cancels adjacent RZ
gates before and after synthesis; it leaves fixed RX pulses alone. Symbolic
merging reuses the existing phase-safe angle normalization. Numeric fusion must
shorten an already native run. General symbolic pulse costing and unrelated
square-root-iSWAP cleanup remain outside this change.

### Use radians in Core and convert at the producer boundary

Recommend revising the unreleased #2578 design to use radians consistently with
Core's existing rotations and the
[Braket SDK gate matrices](https://amazon-braket-sdk-python.readthedocs.io/en/stable/_modules/braket/circuits/gates.html).
Bench's IonQ gate classes use turns: import their GPI/GPI2 phases, all three MS
parameters, and ZZ interaction angles with one multiplication by 2π. Identify
these conventions from the producer's gate definitions, not the name `zz` alone.
Braket input/output then requires no unit conversion. A direct IonQ interface
would own its own conversion.

With radian parameters, `GPI(φ) = i R(π, φ)`, `GPI2(φ) = R(π/2, φ)`, and
`MS(φ₀, φ₁, θ)` is RXX(θ) conjugated by the corresponding RZ phases. Braket
`zz(θ)` is Core `RZZ(θ)`: remove the duplicate ZZ operation and its
matrix/runtime plumbing from #2578. Retain GPI/GPI2/MS identities where needed
to preserve native output. Their names do not justify a new external QIR runtime
ABI; lower them to existing operations at the QIR boundary unless a concrete
consumer requires native QIS entry points.

GPI/GPI2 have fixed rotation amounts but continuously variable phase arguments.
Do not constrain the GPI2 parameter to π/2. MS phases belong to ordered
operands; its full-entangling interaction is π/2, including Braket's
omitted-argument default. Existing two-qubit synthesis can use a supported
full-entangling MS or RZZ instance. When a provider bounds an interaction
parameter, expose a known-valid synthesis subset, such as fixed RZZ(π/2), rather
than advertising unrestricted RZZ and relying on export-time rejection. Other
supported interaction angles remain representable as fixed capabilities without
promising synthesis from them. The adapter owns device-specific domain
validation; add richer Core constraints only if concrete Bench circuits require
more than this conservative subset.

### Validate the actual provider payload

Bench's IonQ models include a free RZ operation, while Braket's native IonQ set
does not. Keep the benchmark model usable, but require strict verbatim targets
to lower RZ into native pulses or absorb it into their phases. Import Bench's
named Rigetti gates through their fixed RX definitions instead of adding new
Core gates.

The Braket adapter owns native spellings (`prx`, `xx`, `zz`), provider argument
validation, and
[verbatim serialization](https://docs.aws.amazon.com/braket/latest/developerguide/braket-openqasm-verbatim-compilation.html).
Check the final native calls, radian values, and physical placements after
serialization. Custom OpenQASM definitions that expand into rotations and CNOTs
do not establish verbatim compatibility. Follow the provider's box and
measurement rules; IonQ requires all gates in the verbatim region. Apply the
documented
[rewiring requirements](https://docs.aws.amazon.com/braket/latest/developerguide/braket-constructing-circuit.html#verbatim-compilation),
including disabling rewiring for Rigetti. Keep ordinary service compilation and
strict native compilation as distinct target selections.

## Work remaining, in priority order

- [ ] **P0 — complete native ion coverage in #2578:** adopt radian conventions,
      reuse RZZ, and keep native GPI/GPI2/MS synthesis independent of the
      removed arbitrary-angle solver. Reconcile its plan and downstream
      dependencies before implementation. Verify full matrices, phase, and
      operand order.
- [ ] **P0 — close the consumer contract:** add Bench import and strict Braket
      target/export checks for Rigetti, Forte, and Bench's Aria model.
      Distinguish Bench-native output from provider-native output and verify
      parameter conversion exactly once. These checks are required before
      claiming the hardware use case complete, even though the adapter changes
      land separately.
- [ ] **P1 — finish the catalogue:** exercise IQM and AQT through existing Core
      operations, resolve missing device metadata such as Cepheus, and retain
      coverage for other Bench families. Extend recipes only for a demonstrated
      catalogue gap.

## Validation

Use offline device fixtures and Braket SDK matrices as the consumer reference;
no hardware execution is needed. Check each native gate and representative
compiled Bench circuits. Include numeric and bound symbolic parameters, global
phase, unequal MS phases with reversed operands, the default MS interaction, and
named Rigetti fixed gates. Verify native operation names, parameter values, and
legal placements after export, with failures for unsupported fixed values and
residual non-native gates such as IonQ RZ.

For #2575, tests retain native acceptance of RX(0.37) while rejecting it as a
synthesis basis. They cover both quarter-turn signs, optional half turns,
symbolic RZ merging and cancellation, parameter dominance, and consistency
between routing costs and emitted pulses. The former six-axis matrix tests are
replaced by 72 full-matrix cases for the supported recipes. Python checks
include bindings as large as 1e300 and the existing
[Qiskit input contract](../../docs/mlir/qiskit.md).

After building with the repository release preset, run:

```sh
build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler
build/release/mlir/unittests/Dialect/QCO/Transforms/NativeSynthesis/mqt-core-mlir-unittest-target-synthesis
build/release/mlir/unittests/Dialect/QCO/Transforms/Mapping/mqt-core-mlir-unittest-mapping
build/release/mlir/unittests/Dialect/QCO/Transforms/Decomposition/mqt-core-mlir-unittest-decomposition
build/release/mlir/unittests/Dialect/QCO/Transforms/Optimizations/mqt-core-mlir-unittest-optimizations
QISKIT_NUM_PROCS=1 uv run --no-sync pytest -n 4 test/python/test_mlir.py
uvx nox -s lint
```

Run the owning adapter tests for the final payload checks. Follow
[repository guidance](../../AGENTS.md#build-and-validation) for binding stubs
and C++ lint when their implementation changes.

The simplified implementation passed 245 compiler, 80 native-synthesis, 125
mapping, 315 decomposition, and 204 optimization tests, plus all 162 Python MLIR
tests. A subsequent complexity review removed obsolete Euler-plan RZ coalescing;
these results include that deletion. Radian ion gates and Braket consumer checks
remain unimplemented follow-ups.

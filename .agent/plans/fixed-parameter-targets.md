# Fixed-parameter compiler targets

Status: complete.

## Goal and scope

Allow operation capabilities to restrict individual parameters to finite fixed
values. Unspecified parameters remain unrestricted. Target matching, serialized
attributes, synthesis-basis selection, and final verification must preserve the
same restrictions. Target compatibility also compares these constraints.
Symbolic values cannot satisfy a fixed parameter.

Derive synthesis from one arbitrary rotation axis and a fixed pulse about a
different axis. Support all distinct RX/RY/RZ pairs through cyclic coordinates.
Precompute an effective quarter-turn sequence from its actual angle; use native
half turns when available. Bound construction to 64 pulses per effective quarter
turn. Direct native-gate targets are a separate Core change.

## Decisions

Use optional fixed values per parameter, not a general constraint language.
Multiple capabilities describe alternative fixed values and placements. Match
constants with the existing absolute parameter-comparison tolerance, without
reducing angles modulo a period: doing so could lose global phase.

Only inspect parameter values for constrained capabilities. Unrestricted targets
keep their existing pipelines. Basis selection must never treat a fixed-angle
rotation as an arbitrary rotation.

Numeric and symbolic synthesis share one fixed-pulse recipe in the existing
Euler utilities. Pulse-plan details stay internal; Python exposes target
capabilities and the selected basis kind. Routing and synthesis borrow the
cached basis, including its pulse sequence; Python property access retains value
semantics.

Target cleanup applies structural and classical canonicalization, preserving
gate forms for target-aware synthesis. The inliner's cleanup follows the same
rule. Target pass factories select canonicalization; target pipelines reuse the
shared QCO cleanup sequence with that canonicalizer. The selector includes QCO
lifetime and control-flow patterns, which cannot be omitted without breaking
linearity. Non-universal targets keep their native sequences; operations outside
the native set still need a usable basis. Numeric fixed-pulse fusion uses the
cached recipe and only shortens already native runs. Symbolic fixed-pulse runs
continue to use individual lowering because symbolic fusion does not model pulse
costs.

## Validation

Run these commands from the repository root after the build and environment
setup in [AGENTS.md](../../AGENTS.md#build-and-validation). The recorded results
below are from local validation, not hosted CI.

- `build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler`: 244
  tests passed, including native-only compilation and OpenQASM export.
- `build/release/mlir/unittests/Dialect/QCO/Transforms/NativeSynthesis/mqt-core-mlir-unittest-target-synthesis`:
  79 tests passed, including 1,944 full-matrix cases across all six axis pairs,
  routing-cost agreement, and phase-preserving fixed-pulse fusion. Append
  `--gtest_filter='TargetSynthesisTest.*FixedPulse*'` to run only those focused
  checks.
- `build/release/mlir/unittests/Dialect/QCO/Transforms/Mapping/mqt-core-mlir-unittest-mapping`:
  125 tests passed.
- `uv run --no-sync pytest test/python/test_mlir.py -k 'fixed_parameter or fixed_pulse'`:
  362 cases passed, covering the Python API and numeric and symbolic input
  gates.
- `uvx nox -s stubs`: passed; regenerated Python signatures include
  `FIXED_ROTATION` and the empty-tuple `parameters` default.

The native-suite results are from the preceding implementation validation. They
were not rerun for the binding-only changes; the Python cases and stub
generation were. Hosted CI is separate and was not rerun locally.

## Follow-up

Direct GPI/GPI2, MS, and ZZ target support is separate work. It will own gate
conventions and synthesis in Core. The downstream adapter will then expose these
capabilities and Rigetti's fixed rotations.

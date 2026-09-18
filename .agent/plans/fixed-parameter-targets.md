# Fixed-parameter compiler targets

Status: generic fixed-pulse synthesis implemented; validation in progress.

## Goal and scope

Allow operation capabilities to restrict individual parameters to finite fixed
values. Unspecified parameters remain unrestricted. Target matching, serialized
attributes, synthesis-basis selection, and final verification must preserve the
same restrictions. Symbolic values cannot satisfy a fixed parameter.

Derive synthesis from arbitrary RZ and a target-declared fixed X/Y pulse.
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

## Validation

Compiler and native-synthesis unit suites passed, including full-unitary phase
comparisons, parameter restrictions, invalid attributes, and ordered placements.
Python binding tests cover fixed and symbolic input gates. Earlier validation
passed for the initial fixed-pulse implementation. The generic construction
passes full-unitary tests across both axes, signs, fractional and non-Clifford
angles, and optional half turns. Final Python and lint checks remain.

## Follow-up

Direct GPI/GPI2, MS, and ZZ target support is separate work. It will own gate
conventions and synthesis in Core. The downstream adapter will then expose these
capabilities and Rigetti's fixed rotations.

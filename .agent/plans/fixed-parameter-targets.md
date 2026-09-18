# Fixed-parameter compiler targets

Status: complete.

## Goal and scope

Allow operation capabilities to restrict individual parameters to finite fixed
values. Unspecified parameters remain unrestricted. Target matching, serialized
attributes, synthesis-basis selection, and final verification must preserve the
same restrictions. Symbolic values cannot satisfy a fixed parameter.

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

## Validation

The compiler suite passed 234 tests; native synthesis passed 64 tests, including
648 full-matrix cases across all six axis pairs, both signs, fractional and
non-Clifford angles, and optional half turns. Python target tests passed 388
cases, including numerical and symbolic input gates. Generated stubs, repository
lint, and full changed-file C++ lint passed.

## Follow-up

Direct GPI/GPI2, MS, and ZZ target support is separate work. It will own gate
conventions and synthesis in Core. The downstream adapter will then expose these
capabilities and Rigetti's fixed rotations.

# Fixed-parameter compiler targets

Status: implementation and local validation complete.

## Goal and scope

Allow operation capabilities to restrict individual parameters to finite fixed
values. Unspecified parameters remain unrestricted. Target matching, serialized
attributes, synthesis-basis selection, and final verification must preserve the
same restrictions. Symbolic values cannot satisfy a fixed parameter.

Add synthesis using arbitrary RZ and RX(π/2), using existing rotation
operations. This basis covers fixed X-axis pulses without new vendor gate
operations. Direct native-gate targets are a separate Core change.

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
Python binding tests cover fixed and symbolic input gates. Stub generation,
repository lint, and full changed-file C++ lint passed.

## Follow-up

Direct GPI/GPI2, MS, and ZZ target support is separate work. It will own gate
conventions and synthesis in Core. The downstream adapter will then expose these
capabilities and Rigetti's fixed rotations.

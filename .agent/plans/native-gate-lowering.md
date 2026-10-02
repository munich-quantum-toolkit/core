# Native gate lowering

Status: superseded for Core synthesis by
[native target synthesis](native-ion-gate-targets.md). The Bench lowering design
below remains current.

## Scope and ownership

Keep fixed RX and R capabilities in the public target. Numeric and symbolic
Euler synthesis use the standard ZSXX basis. Select SX or SXdg to match the
available quarter-turn direction; enable X when a native gate or half-turn gate
supports it. The Euler emitter produces the selected RX or R gates directly and
preserves their exact global phase. Native gates remain valid inputs. Arbitrary
fixed-angle synthesis remains unsupported.

Bench's Qiskit compiler uses a private standard-gate target and a final
BasisTranslator with a local equivalence library. Core consumes the native
target directly. Native compilation ignores physical placement; mapped
compilation retains it. Preserve layout metadata needed to interpret native
compilation and mirror results.

## Design decisions

- Reuse ZSXX, native capability matching, and Qiskit's BasisTranslator. No new
  basis enum, public compiler pass, or provider dependency is needed.
- Select the quarter-turn direction before fusion counts gates. Each logical
  quarter turn then maps to one native rotation gate without additional RZ
  operations.
- Keep phase corrections exact, including under control and recompilation.
  Native compilation does not mutate Qiskit's session equivalence library.
- Keep native parameter validation separate from logical synthesis. Preserve
  natively supported gates of the opposite quarter-turn direction.

## Validation

See [the native target plan](native-ion-gate-targets.md) for the current
validation results and supported scope. Phase, short native forms, mixed named
and fixed operations, and symbolic inputs are regression-tested.

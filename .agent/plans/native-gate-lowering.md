# Native gate lowering

## Scope and ownership

Keep fixed RX capabilities in the public target. Numeric and symbolic Euler
synthesis use the standard ZSXX basis. Select SX or SXdg to match the available
quarter-turn direction; enable X when a native gate or half-turn gate supports
it. Target-native synthesis converts these gates to RX and preserves their exact
global phase. Native gates remain valid inputs. Arbitrary fixed-angle synthesis
remains unsupported.

Bench's Qiskit compiler uses a private standard-gate target and a final
BasisTranslator with a local equivalence library. Core consumes the native
target directly. Native compilation ignores physical placement; mapped
compilation retains it. Preserve layout metadata needed to interpret native
compilation and mirror results.

## Design decisions

- Reuse ZSXX, native capability matching, and Qiskit's BasisTranslator. No new
  basis enum, public compiler pass, or provider dependency is needed.
- Select the quarter-turn direction before fusion counts gates. Each logical
  quarter turn then maps to one native gate without additional RZ operations.
- Keep phase corrections exact, including under control and recompilation.
  Rigetti compilation does not mutate Qiskit's session equivalence library.
- Keep native parameter validation separate from logical synthesis. Preserve
  natively supported gates of the opposite quarter-turn direction.

## Validation

Core's native synthesis, decomposition, optimization, compiler, and mapping
suites pass 992 tests. Python MLIR/Qiskit and C API adoption pass 867 tests.
Regression tests cover negative-RX shortening, mixed named/fixed native
operations, and the optional named-X shortcut.

An eight-case local comparison against Core `574b4fc7` used six-qubit circuits
with eight U/CZ layers, numeric and symbolic parameters, and positive, negative,
quarter-turn-only, and named-SX targets. Native gate counts stayed unchanged.
Median compile times over five runs were 3.5-4.3 ms; the largest change was 0.14
ms. These measurements cover compiler time, not hardware execution time.

Before publication, run whole-file C++ lint, repository lint, affected
executable docs, Bench's full optional-Core suite, and its Qiskit 2.1.2 minimum
suite. Pin Bench to the signed Core commit and validate that exact dependency
selection.

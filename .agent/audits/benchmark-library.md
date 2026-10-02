# Contract audit: benchmark library

Status: Applied. Baseline: `f36ab46c07a389ac9faabb2ac9f4e7f97fffdffa` with no
in-scope edits. Date: 2026-09-09. Scope: `include/mqt-core/bench/`,
`src/bench/`, `bindings/bench/`, `python/mqt/core/bench/`, `mlir/bench/`,
`test/bench/`, `test/python/bench/`, and `mlir/unittests/bench/`.

## Result

The audit found no benchmark correctness defect. It removed repeated native
plumbing and tests of incidental QC builder structure. It retained tests of
benchmark semantics, algorithms, configuration choices, output order, numerical
limits, structured scaling, and serialization.

## Findings

### Test benchmark contracts instead of generic builder structure

The BV, Grover, GHZ, QFT, and QPE generation tests inspected generic loads,
stores, measurements, and SSA indices. The QC IR tests already cover those
builder contracts. The benchmark tests now use sampling where the QCO sampler
supports the generated program. The BV test retains only the resource choices
that distinguish its static and dynamic methods.

The teleportation feed-forward order, repeat-until-success retry behavior,
multiplexer angle recurrence, QFT-adder cross-register phases, QPE controlled
powers, and modular-multiplier coherence checks remain. These walks protect
algorithm-specific behavior rather than generic construction.

### Remove incidental counts and repeated checks

Exact gate, loop-bound, and tensor-extraction counts fixed tests to the current
emitter shape without protecting a supported result. Broad operation bounds
remain where they protect structured generation at maximum input sizes.

The common helpers no longer recount sampler shots or revalidate programs after
APIs that already verify them. The Python generation helper now checks the QCO
conversion boundary without searching printed IR.

### Share native serialization and evaluation plumbing

One analytic-reference builder replaces ten copies of the same JSON fields. One
constructor wrapper preserves the common parameter diagnostic, and the registry
lookup now supplies validation and definition metadata directly. A
benchmark-aware evaluation overload removes repeated forwarding lambdas while
the modular multiplier retains its separate arithmetic-success calculation.

### Remove unused Grover accessors

The exported marked and unmarked probability getters had no production or
binding callers. `probability()` already provides both values, so the redundant
getters and their dedicated assertions were removed.

### Use consistent class summaries and render mathematical notation

Every benchmark class uses the dry summary "A validated ... benchmark." Longer
Python descriptions retain algorithm details where they already exist. Doxygen
and Python binding text use their math roles for states, bases, phases,
rotations, and arithmetic. Generated stubs preserve the required escaped LaTeX.

## Unresolved questions

The QCO DD sampler cannot execute BV because its dense rank-one `i1` tensor
constant is unsupported. This audit does not change production code to serve a
test. The registry-wide QC-to-jeff round trip still covers BV generation and
serialization.

## Validation

- The 62 native benchmark tests pass.
- The 31 MLIR benchmark generation tests pass.
- Stub generation passes.
- The documentation build and link check pass.

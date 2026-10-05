# Quantum program inspection

Status: complete; QC and QCO share consolidated inspection results.

## Scope and decisions

QC and QCO gate histograms use the existing entry-point gate count semantics:
visit static IR once, skip barriers and modifier bodies, and do not expand
calls. Unitary calls count under their callee name. Measurements and resets do
not count; explicit global phases do. Controlled gates use `ctrl`, not frontend
names such as `cx` or `cz`. Both dialects share the counting traversal and
Python bindings; neither needs conversion for inspection. `inspect()` returns
all gate counts in `QuantumProgramInfo` together with resource information.
Individual counting methods retain their lightweight traversals.

QC and QCO resource inspection reports total declared allocated width, distinct
physical site IDs, and control-flow presence. Unknown widths remain unknown;
site IDs are not dense wire counts. Include helper declarations but exclude
nested module scopes. These opt-in queries do not change compilation.

Circuit depth is tracked in
[#2682](https://github.com/munich-quantum-toolkit/core/issues/2682). Its
treatment of control flow, barriers, classical dependencies, and calls must be
defined before implementation. Parameter binding is on main; circuit
construction remains separate.

## Validation

The compiler test binary passed all 275 tests, including the consolidated
counts, resource boundaries, control flow, opaque modifiers and calls, and
QCO-only native gates. All six focused Python counting and inspection tests
passed through both dialects, including comparison with frontend gate counts.

Reproduce the focused Python checks with:

```sh
uv run --no-sync pytest test/python/test_mlir.py -k 'program_num_gates or program_inspection'
```

Stubs were regenerated. Repository lint, whole-changed-file C++ lint, and
executable documentation with local link checks passed. These checks make no
compilation-time claim.

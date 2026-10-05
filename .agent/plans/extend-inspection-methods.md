# Quantum program inspection

Status: complete; inspection and counting are implemented without a depth API.

## Scope and decisions

QC gate histograms use the existing entry-point gate count semantics: visit
static IR once, skip barriers and modifier bodies, and do not expand calls.
Unitary calls count under their callee name. Measurements and resets do not
count; explicit global phases do. Controlled gates use `ctrl`, not frontend
names such as `cx` or `cz`.

QC and QCO resource inspection reports total declared allocated width, distinct
physical site IDs, and control-flow presence. Unknown widths remain unknown;
site IDs are not dense wire counts. Include helper declarations but exclude
nested module scopes. These opt-in queries do not change compilation.

Circuit depth is deferred: quantum-only dependency depth does not reproduce
Qiskit's handling of barriers and classical dependencies. Consumers can retain
frontend metrics outside compilation timing. Parameter binding is on main;
circuit construction remains separate.

## Validation

The compiler test binary passed all 272 tests. Native coverage includes static
sites, dynamic and overflowing widths, borrowed references, nested modules,
control flow, and opaque modifiers/calls. The selected Python suites
(`test_mlir.py`, `test_mlir_qiskit_translation.py`,
`test_mlir_qiskit_target.py`, and `test_mlir_parameter_binding.py`) passed all
801 tests, including four device-dependent tests rerun with the SC QDMI device
configured.

Both Qiskit and OpenQASM import have a regression comparing resource width, gate
histograms, and total/1Q/2Q gate counts against Qiskit. All 65 small/medium
QASMBench inputs agreed on QC/QCO declared width and gate-only counts after
excluding explicit global phases. This validates inspection, not compilation
timing. Temporarily replacing the Benchpress text-based helpers with these APIs
passed 114 regression and construction tests without changing its dependency
pin.

Generated stubs, repository lint, whole-changed-file C++ lint (with a corrected
test rechecked in full), and executable documentation with local link checks
passed. Hosted CI remains a separate check after publication.

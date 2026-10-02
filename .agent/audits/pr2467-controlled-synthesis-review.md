# Controlled-synthesis audit and quality comparison

Status: applied. Date: 2026-09-08. Original baseline:
`00a214ec3f49aab0eeee0352a05ef817100e8c55`. Changes rebased onto main
`4c5e45855e5bb50c42b68ddbf4f9a4dababb8737`.

## Findings and disposition

- **Identity modifier evaluation:** wide X/Y/Z synthesis can leave empty
  `qco.ctrl` bodies after angle folding. The original head's DD consumer
  rejected these valid identity regions. Main's shared `composeBodyMatrix` fix
  in #2464 resolves this on rebase. No duplicate or Y-specific workaround is
  needed. The 63-control coherent-state regression now passes without
  canonicalization.
- **Unreachable HP24 machinery:** `mczCoreForWidth` uses specialized synthesis
  below five controls and SP22 through 32. HP24 now asserts its actual minimum
  of 33 controls and directly selects one dirty helper for odd widths, two for
  even widths. Removed the inactive small-width table, ripple incrementer,
  recursive relative-phase planner, thread-local cache, and estimate branches.
  Active half-MCX widths exceed 11 and incrementer widths exceed 10, so those
  removed alternatives had no production caller. Changing the crossover must
  revisit this limit. This is a maintenance improvement, not a speed claim.
- **Wide-state oracle and coverage:** retained existing SP22 samples and added
  32/33/34, 47/48, and 63/64 controls for X/Y/Z. Compare the phase-sensitive
  norm of `actual - expected` at `1e-11`, rather than DD node identity. The
  coherent helper scopes DD arithmetic tolerance to `1e-15` and restores it
  afterward; the default merging tolerance accumulated about `1e-9` error at 32
  controls. These selected states are not a full-operator bound. Independent
  numeric and runtime rotation matrices now cover 2 through 10 controls.
- **Duplicate remapping:** remap each generated rotation half-plan in place;
  remove unused `GateEmitter` remapping. This removes a second plan allocation
  and move loop. Plans remain the sole owner of wire remapping.
- **Stale CLI descriptions:** both compiler options now name Y and rotations;
  the generated pass description names the active HP24 dirty-helper choice.

Source:
`mlir/lib/Dialect/QCO/Transforms/Decomposition/DecomposeMultiControlled.cpp`.
Regression tests: the corresponding
`mlir/unittests/Dialect/QCO/Transforms/Decomposition/test_multi_controlled_decomposition.cpp`.

## Contracts retained

The modifier verifier owns restrictions on classical support operations. No
second dependency walker is needed. Balanced halves borrow opposite controls and
restore them coherently. Arbitrary relative-phase MCX replacements have not been
proved safe in the four-MCX rotation shell. RX uses Hadamard conjugation of RZ;
Y uses target S conjugation of X. Controlled `2*pi` rotations retain their
conditional minus sign. Native-target and minimum-width policies still apply. No
new cache, public API, or dependency is justified.

## Reproducible Qiskit comparison

Run `uv run --no-sync python test/bench/compare_controlled_rotations.py` with
the locally built package and Qiskit 2.5.2. The script records numeric and
symbolic RX/RY/RZ at 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, and 64 controls. Raw
rows from this run are in
[controlled-rotation-quality.csv](controlled-rotation-quality.csv). Core was
built with Clang/LLVM 23.1, Release, ThinLTO, and mold on ARM64.

Both methods use exactly the input qubits. Nine-sample median synthesis times
exclude one warmup, input creation, import/export, basis conversion, and
routing. Both circuits receive the same `u,cx` normalization, then level-3
optimization with seed zero and `qubits_initially_zero=False`. The CSV includes
CX count, one-qubit count, total depth, CX depth, and synthesis time. Numeric
and symbolic operators through five controls pass phase-sensitive checks,
including symbolic binding at `2*pi`. Timings are local sequential measurements,
not an end-to-end compilation comparison or stable performance guarantee.

Numeric post-optimization examples (Core / Qiskit):

| Gate | Controls |          CX |       Depth |  Synthesis ms |
| ---- | -------: | ----------: | ----------: | ------------: |
| RX   |        2 |       4 / 8 |      9 / 13 | 0.272 / 0.034 |
| RY   |        3 |     14 / 20 |     21 / 28 | 0.351 / 0.076 |
| RY   |        8 |   104 / 104 |   186 / 171 | 0.507 / 0.238 |
| RY   |       16 |   232 / 232 |   442 / 427 | 2.214 / 0.346 |
| RY   |       64 | 1000 / 1000 | 1978 / 1963 | 4.251 / 0.350 |

Core saves CX gates for RX/RY at two and three controls. Other sampled CX counts
match, including all RZ cases and symbolic angles. Core's larger circuits are up
to 15 layers deeper. Core synthesis is slower in this measurement. Earlier
claims of uniformly faster Core synthesis included basis lowering in Qiskit's
timed work and are superseded by this comparison.

## Improvement beyond Qiskit: measured candidate, deferred implementation

Helper order affects scheduling even when CX count stays fixed. In an isolated
prototype, reverse the selected dirty-helper wires in both balanced half-MCXs,
keeping controls in order. Use Qiskit's exact `synth_mcx_n_dirty_i15` to
construct those halves, and the same four quarter-angle rotations and
normalization as above. This produces these numeric RY depths:

| Controls | Core | Qiskit | Reversed-helper prototype | CX (all three) |
| -------- | ---: | -----: | ------------------------: | -------------: |
| 9        |  214 |    210 |                       204 |            120 |
| 10       |  250 |    235 |                       232 |            136 |
| 16       |  442 |    427 |                       388 |            232 |
| 32       |  954 |    939 |                       804 |            488 |
| 64       | 1978 |   1963 |                      1636 |           1000 |

At eight controls, balanced reversed helpers give depth 180, worse than Qiskit's
171. A 5+3 split instead reduces CX from 104 to 96 but has depth 185. Neither
candidate dominates at every width. The reversed balanced prototype at nine
controls and both eight-control splits pass full phase-sensitive operator
comparisons with maximum element error below `1.2e-14`.

Minimal reproduction, after importing `QuantumCircuit`, `transpile`, and
`qiskit.synthesis.synth_mcx_n_dirty_i15`:

```python
k = 64
first = (k + 1) // 2
halves = []
for start, count in ((0, first), (first, k - first)):
    plan = synth_mcx_n_dirty_i15(count)
    spare = [i for i in range(k) if i < start or i >= start + count]
    spare = spare[: plan.num_qubits - count - 1]
    halves.append((plan, list(range(start, start + count)) + [k] + spare[::-1]))
circuit = QuantumCircuit(k + 1)
for _ in range(2):
    circuit.compose(*halves[0], inplace=True)
    circuit.ry(-0.73 / 4, k)
    circuit.compose(*halves[1], inplace=True)
    circuit.ry(0.73 / 4, k)
output = transpile(
    circuit,
    basis_gates=["u", "cx"],
    optimization_level=3,
    seed_transpiler=0,
    qubits_initially_zero=False,
)
assert output.count_ops()["cx"] == 1000
assert output.depth() == 1636
```

This establishes room beyond Qiskit's current public synthesis output, not
optimality or a ready Core patch. Before implementation, measure Core's own
helper ordering across all axes, symbolic expressions, and routing targets;
retain the exact-restoration tests. A width-specific split policy needs an
explicit objective because CX count and depth disagree at eight controls. The
existing construction has linear CX count; results requiring additional
clean/dirty qubits do not establish an improvement under this no-extra-qubit
contract.

## Validation

The native decomposition binary passes all 303 tests, including both active HP24
dirty-helper modes and numeric/runtime full operators through ten controls. The
benchmark completes all 144 backend rows. Final lint and Python validation are
recorded in the implementation plan.

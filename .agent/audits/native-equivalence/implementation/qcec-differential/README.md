# Executed native/QCEC differential

2026-10-25; 11 bounded cases. `probe.py` builds identical operation descriptions
in separate Python environments. Each JSON embeds exact QASM2, metadata global
phase, options, versions, verdicts, and (native) emitted QCO. QASM2 alone cannot
represent the metadata phase; that field must be retained when reproducing.

Native source baseline: `1d384a5d23aa1f02b26fd5684dab586ef26167fe` plus the
uncommitted native-equivalence implementation in this worktree. Built with
`CMAKE_BUILD_PARALLEL_LEVEL=4 nox -s stubs`; exact imported binary SHA-256
values are in `native-binaries.json`. These are executable artifacts, not a
committed implementation revision. Native Python is 3.14.8. The JSON records the
Qiskit version used by each environment; emitted inputs are included for
comparison.

QCEC: `4319be6bbf58cd7958a6447b42022721fb74ee3b`, built against Core
`80daeb5f9a2c862dcafb6e3ddb6e45a75b410029` (3.11.0), Python 3.12.15, Qiskit
2.2.3. Its environment remains isolated from native Core's newer ABI.

Run from the implementation checkout:

```sh
.nox/stubs/bin/python .agent/audits/native-equivalence/implementation/qcec-differential/probe.py native .agent/audits/native-equivalence/implementation/qcec-differential/native.json
/home/nvidia/.codex/experiments/native-equivalence-20261025/qcec-venv/bin/python .agent/audits/native-equivalence/implementation/qcec-differential/probe.py qcec .agent/audits/native-equivalence/implementation/qcec-differential/qcec.json
```

| Input/relation                                                                              | Expected                      | Native observed                   | QCEC observed                                                                        |
| ------------------------------------------------------------------------------------------- | ----------------------------- | --------------------------------- | ------------------------------------------------------------------------------------ |
| X versus X                                                                                  | Equal                         | Equivalent                        | equivalent                                                                           |
| X versus RY(.17)X                                                                           | Different                     | NotEquivalent                     | not_equivalent                                                                       |
| X versus exp(.37i)X, preserve phase                                                         | Different                     | NotEquivalent                     | equivalent; this QCEC result does not implement the native phase-preserving contract |
| Same metadata phase difference, ignore phase                                                | Equal projectively            | Equivalent                        | equivalent                                                                           |
| X versus XZXZX, ignore phase                                                                | Equal projectively            | Equivalent                        | equivalent_up_to_global_phase                                                        |
| X versus RY(pi/3)X, process distance threshold .49                                          | D=.5 exceeds threshold        | NotEquivalent (.4999999999999999) | not_equivalent                                                                       |
| Same, threshold .5                                                                          | Within native numerical guard | Inconclusive                      | equivalent                                                                           |
| Same, threshold .51                                                                         | Within threshold              | Equivalent                        | equivalent                                                                           |
| Two wires; X0; terminal labels c[0],c[3]; routed SWAP and reversed measurement destinations | Same effects for every input  | Equivalent                        | equivalent                                                                           |
| Same route, physical X moved to wire 1 incorrectly                                          | Different                     | NotEquivalent                     | not_equivalent                                                                       |
| Same route, measurement destination reversal omitted                                        | Different                     | NotEquivalent                     | not_equivalent                                                                       |

The last three use native TerminalProbabilities and QCEC partial construction.
The full declared four-bit register includes fixed zero holes; selected active
labels are sparse. This overlap does not validate QCEC partial construction in
general: the earlier H versus HZ superposition counterexample remains outside
this passing subset. Native comparison uses outcome effects, not basis-column
probabilities. The phase-preserving and near-threshold differences above are
explicitly expected contract differences, not a requirement to copy QCEC.

No performance claim follows from these tiny cases. This probe does not test all
routing metadata formats, every classical binding, or general channels.

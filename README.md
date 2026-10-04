🤖 *AI text below* 🤖

# Core PR #2578: synthesis evaluations

Latest: [large-circuit evaluation, 24–156 qubits, 2026-10-04](large-20261004/report.md), including scaling curves and bottleneck diagnostics.

Results for [Core PR #2578](https://github.com/munich-quantum-toolkit/core/pull/2578), comparing upstream main `32f1b331430ce4d580e2780f6fa63e60f6b9a0a3` with the synthesis implementation published in `a5c0292727ecf3e34dc713ba966327e0a33067f2`.

The subsequent CI fix `a00a2d8c0eb7fde8e36b8372bfd5acfc912bc78d` changes DD simulation and tests, not synthesis. These are the original measurements, not a rerun of that later commit.

The frozen corpus contains 50 inputs and eight native gate contracts: 24 generated Core benchmarks, 18 scalable numeric/symbolic kernels, and eight identity microcases. The branch produces 400 validated native exports, versus 271 on main. Comparisons exclude unsupported cases; aggregate ratios also exclude the microcases.

Both builds use Release LLVM/MLIR 23.1, Python 3.14.7, Qiskit 2.5.2, and NumPy 2.5.3 on DGX Spark arm64. Runs use one pinned CPU, one thread, one discarded warm-up, and five timed samples. Import, copying, export, and validation are outside synthesis timing. All-to-all targets isolate synthesis; results do not measure routing or hardware fidelity. The fixed-RX/iSWAP contract is exploratory, not a current online Rigetti-device claim.

Total native one- and two-qubit gate counts never increase on jointly validated cases. Numeric IQM W-state and Pauli-layer workloads take up to about 20% longer while using roughly two-thirds fewer R gates. Seven full circuit depths increase, while two-qubit depth never increases. Non-RZ one-qubit count increases in one exploratory iSWAP case, from 98 to 99, while total one-qubit count falls. Small timing differences should not be overinterpreted.

Validation includes full matrices for small synthetic circuits, sampled random statevectors for larger circuits, three symbolic bindings, and 512 seeded DD shots checked against Core benchmark reference distributions. Sampled checks are weaker than full-unitary equivalence.

## Figures and data

- [Paired quality and runtime](plots/paired_comparison.png)
- [Native one-qubit counts by algorithm](plots/algorithm_native_1q.png)
- [Native two-qubit counts by algorithm](plots/algorithm_native_2q.png)
- [Synthesis runtime by algorithm](plots/algorithm_synthesis_ms.png)
- [Absolute results, including fractional IBM and IonQ](plots/absolute_representatives.png)
- [Validated-export coverage](plots/applicability.png)
- [Raw results](raw_results.csv)
- [Reproduction bundle](synthesis-evaluation.zip): frozen corpus, exact measured harness, raw samples, native output IR, metadata, report, and PNG/PDF/SVG figures.

The reproduction bundle records the measured source and the publication state at the time of the run. The PR comment records later CI status separately.

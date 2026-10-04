🤖 *AI text below* 🤖

# Native synthesis refinement: large-circuit evaluation

This repeats the 2026-10-04 large-circuit experiment after aligning Pauli and
Weyl approximation, extending Z-frame propagation to native RZ targets, and
removing repeated inliner cleanup. The previous experiment remains unchanged.

## Inputs and measurements

- The 68 input circuits, hashes, and seven native contracts are copied unchanged
  from the previous experiment. Widths are 24, 36, 54, 104, and 156 qubits.
- Core families: GHZ, QFT, W state, QPE, Grover (two iterations), multiplexer,
  QFT adder, and BV. Grover's generator is limited to 62 qubits, so it is absent
  at widths 104 and 156.
- Numeric and symbolic kernels: EfficientSU2, Pauli layers, and equatorial
  frames. IQM, IonQ, and Rigetti contracts are tested through 54 qubits; four
  IBM/RX-RZ-RZZ contracts continue through 156.
- 398 case/target combinations per compiler. Main and the first candidate
  alternated execution order; the final PR was measured again after a pass-order
  correction, retaining those fresh main measurements. Runs affected by heavy
  background build activity were discarded. The final 104/156-qubit rerun detects
  compiler activity and retries affected cases. Three pairs had over 30% spread
  in synthesis samples; both compilers were repeated for all three pairs. Those
  original rows remain in `variance-originals`. Main and final PR results use
  one warmup and five timed samples,
  CPU 5, and one numerical-library thread. Each case runs in its own process
  with a 180-second timeout and a 12-GiB virtual-memory limit.
- DGX Spark, ARM64, Python 3.14.7, Qiskit 2.5.2, NumPy 2.5.3, Release builds,
  LLVM/MLIR 23.1. Full binary hashes, CMake flags, source status, and environment
  are recorded per worker in `metadata.json`.
- Copying is outside the synthesis timer. Parsing and Qiskit export have their
  own timers. Peak RSS includes the whole worker. Pass profiles run separately.
- Native conformance is checked for every exported operation, including fixed
  aliases and bounded parameters, at three bindings for symbolic circuits.
  This is not a full-width semantic proof. The separate 400-case smaller suite
  checks matrices, sampled statevectors, or benchmark output distributions.
- All targets are all-to-all: the results isolate synthesis and exclude routing,
  calibration fidelity, device scheduling, and hardware execution.

Main is `32f1b331430ce4d580e2780f6fa63e60f6b9a0a3`. The updated PR was built from
`a00a2d8c0eb7fde8e36b8372bfd5acfc912bc78d` plus `measured.patch`.
The corresponding source commit is
`cdf609911d4813a4e0c401a9d2baa0e9b8ece871`. Only plan prose changed after the
measured binary was built. Its version string still names the parent revision;
per-worker binary hashes identify the tested implementation. The previous PR
measurements are
the archived results published at asset commit
`ad92e19de2cce9d3ab4d1e0583884dd803084315`, rather than a simultaneous rerun.
Fresh main measurements provide the contemporaneous baseline. The final PR
head is `9fcefeeac0f3d85af060490980b1e1b7de6ed8a3`; its follow-up changes only tests
and plan prose, leaving the measured compiler unchanged.

## Reproduction

Set `CHECKOUTS` in `large.py` to the two Release checkouts and preserve their
separate Python environments. Do not regenerate the corpus. Verify the hashes
in `corpus/manifest.json`; `evaluate.py` checks each input before compiling.

```sh
python3 large.py sweep --output final-results --repetitions 5
python3 profile_cases.py
```

Each compiler environment must contain its own built `mqt.core.mlir` extension.
Inspect `metadata.json` to confirm the binary hash and package path. `large.py`
uses the shared `evaluate.py` helpers and enforces native contracts at large
width instead of attempting exponentially large statevectors.

The smaller semantic run uses the unchanged 50-case corpus from the original
experiment, included under `small-corpus` in the publication bundle:

```sh
QISKIT_NUM_PROCS=1 <PR-python> evaluate.py run --checkout <PR-checkout> \
  --corpus small-corpus --output small-validation --repetitions 5 --shots 512
```

Those timing samples were collected alongside correctness checks and are not
used for performance claims. `diagnose.py angle_scan` checks two-qubit matrices
including phase; `diagnose.py frame_probe` checks the diagonal-chain reduction.
Both use the active compiler environment. One noisy symbolic pass profile was
retained in `profile-variance-originals`; the five updated profiles were repeated
without concurrent local validation. Profile parsing excludes nested
inliner pipeline rows to avoid double-counting inclusive time.

Generate PNG, PDF, and SVG figures with:

```sh
uv run --no-project --with matplotlib==3.11.2 --with numpy==2.5.3 python analyze.py
uv run --no-project --with matplotlib==3.11.2 --with numpy==2.5.3 python plot_diagnostics.py
```

`previous-results` contains the previous PR's archived samples. The profile
runner reuses the older profiles included in the publication bundle. This is an
ad hoc
evaluation harness, kept outside the source PR.

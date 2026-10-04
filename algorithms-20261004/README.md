🤖 *AI text below* 🤖

# Reproducing the comparison

Use the revisions and measured source patch recorded in `source.json`.
Build both checkouts in Release mode with LLVM/MLIR 23.1, GCC 13.3, Python
3.14, and Qiskit 2.5.2. The worker metadata records exact versions and flags.
Install the MLIR Python extension in each checkout's `.venv`.

Set `CHECKOUTS` in `run_suite.py` and `large.py` to those checkouts. Change CPU 5
if necessary. Move the included `small/` and `large/` result directories aside
before running `python3 run_suite.py`; completed workers are otherwise skipped.
The frozen input hashes are checked before every run. Compiler activity causes
an attempt to be excluded and rerun. The resource limit is 12 GiB per large
worker and the timeout is 300 seconds.

Run `scaling.py main` and `scaling.py pr2578` with the corresponding checkout's
Python interpreter. The former measures canonicalization; the latter runs
scheduled classical simplification followed by top-down canonicalization.
Tensor scalarization uses the same canonicalizer settings on both revisions.
These stress tests exclude parsing and are distinct from synthesis timings.

Run `uv run --no-project --with matplotlib python analyze.py` to recreate the
plots and tables. `cartan_probe.py`, `diagonal_controls.py`, and `probe.py witness`
are phase-sensitive diagnostics; their previous-PR records are historical
quality measurements, not timing baselines. The bundled `probe.py` and
`diagnose.py` satisfy their imports. The historical absolute search paths can
be removed if those old experiment directories exist on the reproduction host.

The smaller suite checks semantics using matrices, sampled states, or Core
reference distributions. The large suite checks native gates and angle bounds
only. All targets use all-to-all connectivity: routing, hardware calibration,
and execution are excluded. Depth excluding RZ is a structural metric, not a
scheduled device duration. See the report for remaining depth tradeoffs.

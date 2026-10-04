🤖 *AI text below* 🤖

# Large-width synthesis evaluation

This extends the 2026-10-03 experiment using its native contracts, generators,
timing helpers, and counter definitions. No production repository changes are
needed. Results and diagnostics are stored outside the code PR.

## Scope

Circuit widths: 24, 36, 54, 104, and 156. Seven native contracts run through 54
qubits: IQM R/CZ; IBM SX/X/RZ + CX or CZ; RX/RZ/RZZ; bounded fractional IBM;
IonQ fixed R + virtual RZ + RZZ; and fixed-RX/RZ/CZ. At 104 and 156 qubits,
only the three IBM contracts and unrestricted RX/RZ/RZZ run. The exploratory
iSWAP contract from the earlier evaluation is omitted. These are native gate
contracts with all-to-all connectivity, not calibrated device snapshots or
routing benchmarks. The target width equals the circuit width; separate
padded-width probes check unused device capacity.

Each width includes the same eight Core families (GHZ, QFT, W state, QPE,
Grover, multiplexer, QFT adder, BV) and six numeric/symbolic kernels
(EfficientSU2, Pauli layers, equatorial frames). Grover uses two iterations;
Core's generator limits Grover to 62 qubits, so 104/156-qubit Grover is excluded.
There are 68 frozen inputs and 398 circuit/target combinations per revision.
Every raw input and target capability is in `corpus/manifest.json`; SHA-256
checks protect the input files.

The baseline is upstream main `32f1b331430ce4d580e2780f6fa63e60f6b9a0a3`.
The PR synthesis implementation is `a5c0292727ecf3e34dc713ba966327e0a33067f2`;
current checkout head is `a00a2d8c0eb7fde8e36b8372bfd5acfc912bc78d`.
The latter changes DD evaluation and tests, not synthesis. These use the same
installed Release synthesis binaries as the earlier experiment; their hashes
are recorded in every worker's metadata. Do not mistake checkout-head metadata
for proof that an unchanged installed binary was rebuilt.

## Measurement

DGX Spark arm64, CPU 5, one thread, Release/O3/LTO, LLVM/MLIR 23.1,
Python 3.14.7, Qiskit 2.5.2, NumPy 2.5.3. Each case gets an isolated process,
one warm-up, and five samples for parsing, synthesis, and export. Source
copying, stringification, gate counts, native validation, and I/O are outside
the synthesis timer. The baseline/PR execution order alternates between
case/target pairs. A worker has a 180-second wall-clock limit and a 12-GiB
virtual-memory limit. Peak RSS includes the interpreter, input, compilation,
export, and validation; it is not incremental synthesis allocation.

`final-results` is the measured sweep. `pilot`, `alias-check`, and `results`
are diagnostic runs, excluded from final statistics. The initial native checker
in `results` incorrectly treated canonical RX export and specialized R aliases
as unsupported; those failures are harness errors. The corrected checker has
positive/negative cases in `check_large.py`. No raw records were rewritten.

At full width, every exported operation is checked against its native gate
name/canonical alias, arity, fixed parameters, and bounds; symbolic outputs get
three deterministic bindings. This is a structural check, **not a semantic
equivalence proof**. The small-circuit semantic results from the earlier sweep
remain separate evidence. Any additional wide-circuit simulation checks are
recorded separately, outside performance timing. Do not construct an exponential
statevector or enumerate a large reference distribution.

## Reproduce

Use the two original Release environments listed in `large.py`, or change
`CHECKOUTS` to independently built matching environments. Run from this folder:

```sh
PR/.venv/bin/python check_large.py
python3 large.py sweep --output new-results --timeout 180
uv run --no-project --with matplotlib --with numpy python analyze.py
```

`generate` recreates inputs using the PR environment; do not regenerate between
baseline and branch runs. `worker` selects one pair. `profile` uses MLIR's
pass timing, separately from benchmark samples. `diagnose.py angle_scan` compares
two-qubit controlled-phase outputs against their exact matrices at tiny angles.
`diagnose.py semantic` runs bounded-shot DD checks for supported Core cases.
`analyze.py` reads `final-results`, writes raw CSV and summaries, and renders
PNG/PDF/SVG plots. Change its input directory when reproducing into a new folder.

`metadata_probe.py` isolates a SymbolDCE pass that leaves the input unchanged;
`profiles/metadata_*.jsonl` records its timing. `diagnose.py frame_probe` checks
a diagonal CZ/RZ chain and records its removable RZ operations. Raw diagnostic
pass logs and eight passing/two timed-out wide DD checks are retained. Initial
pilot runs and failed harness/profiler probes remain local and are excluded
from the reproduction bundle and all reported measurements.

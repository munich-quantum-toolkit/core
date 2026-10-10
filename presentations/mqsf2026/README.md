# MQSF 2026

## System Software for Quantum Computing: From the Metal to the User

A 12-slide, 35-minute keynote for a 1920 × 1080 projector. The white and blue
MQSC presentation opens with the shared software stack, follows a LiH
quantum-classical application, compiles one of its measurement programs for
three device models, and then shows structured programs and adaptive execution.
See [presenter-notes.md](presenter-notes.md) for the rehearsal narrative.

Every slide and progressive build works with a Logitech Spotlight configured to
send standard forward/back keys. The browser renders saved evidence and replays
measured events. Native compilation, simulation, metadata capture, and AFQMC run
before packaging. The presentation needs no server or network.

## Build and present

From the repository root:

```console
uv run --no-project presentations/mqsf2026/build.py
```

Open `build/mqsf2026/index.html` directly in a browser. Use full screen at 1920
× 1080 and rehearse with the actual clicker. The layout scales with the window
while retaining its 16:9 canvas. The build also creates
`build/mqsf2026/mqsf-2026.zip`, containing the self-contained HTML, the full
capture as `evidence.json.gz`, this README, and the presenter notes. Code
highlighting, data, SVG illustrations, logos, and fonts are embedded. External
links and QR destinations are optional audience resources.

| Action                              | Keys                                                 |
| ----------------------------------- | ---------------------------------------------------- |
| Next build, then next slide         | Right arrow, Page Down, Space, Enter                 |
| Previous build, then previous slide | Left arrow, Page Up, Backspace                       |
| First / last slide                  | Home / End                                           |
| Jump to a slide                     | Type its number, then Enter                          |
| Slide overview                      | G or Escape; arrows and Enter select; Escape returns |
| Full screen                         | F                                                    |
| Blackout                            | B                                                    |
| Show inline speaker notes           | P                                                    |

`P` displays notes on the projector; use the separate notes file for private
rehearsal. A URL fragment such as `#9.0` opens a slide and build. Printing
produces the final build of each slide.

Forward starts a playback when it enters that build. During a one-shot replay,
forward finishes the current replay; the next press advances. During a looping
replay, forward stops it and advances immediately. Back cancels playback and
restores the previous build. Playback never changes slides automatically.

| Slide | Playback                                                                            |
| ----- | ----------------------------------------------------------------------------------- |
| 3     | AFQMC workflow unfolds over 6.5 seconds; one further build shows source             |
| 4     | QDMI submission window over 8.5 seconds; parallel CPU/walker replay over 11 seconds |
| 5     | Energy and walker weights evolve over a 12-second loop                              |
| 6     | Placement refinement, routing and synthesis: 10 / 10 / 8 seconds                    |
| 7     | Each device selection replays query, placement, routing and synthesis               |
| 10    | One QPE replay over 9.5 seconds, with API calls and shots on one clock              |

Forward entry starts a build-zero replay automatically. Direct jumps, backward
navigation, printing, and reduced-motion mode show settled final states. Looping
builds hold their final frame for 2.4 seconds before restarting. The quantum
batch replay omits preprocessing and starts at the recorded submission call; its
displayed time is relative to that boundary. CPU task time and imaginary time
are separate, explicitly labelled coordinates.

SVG nodes persist across timeline frames. Morphs start at their current
intermediate geometry when interrupted; hidden future elements remain hidden.
References are present with the substantive content and never require a separate
click. Code is not shortened horizontally. Structured outputs use explicitly
labelled display folds with original source line numbers; the bundle retains all
unmodified artifacts.

## Evidence and its limits

The LiH example uses STO-3G at a stretched 2.4 Å bond, with the Li 1s core
frozen and two active electrons in three sigma spatial orbitals: six spin
orbitals and 15 possible two-electron determinants. The exact reference is FCI
within that active space, including the frozen-core and nuclear energy. The
fixed trial parameters were tuned classically at 1.6 Å and reused at 2.4 Å. This
more demanding geometry makes the projection visible without enlarging the
circuit. It is not a quantum VQE or a quantum advantage demonstration.

The recorded workflow submits 2,048 matchgate measurement programs with 256
shots each as one native QDMI job. Measured shadows supply 15 determinant
overlaps. Four local CPU processes then propagate 128 walkers for 480 steps per
trial, comparing the shadow-derived and Hartree–Fock trials. The step size is
0.02 Ha⁻¹. A checked Givens-angle sign adaptation enforces the Majorana
convention. Dense active-space energy and force-bias contractions replace the
pinned helper's larger-space estimator; the upstream phaseless propagation
update is unchanged. The 15-state cache is exact within this example and is not
a scalable general overlap algorithm.

Every curve can be recomputed from saved local energies and importance weights.
The band is one pointwise independent-walker standard error conditional on the
shared shadow data. It excludes shadow error and systematic effects from finite
projection time, finite step size, and the phaseless approximation. The two
curves reuse per-walker auxiliary-field seeds and are correlated. There is no
population resampling. Walker-grid positions are schematic; states, weights,
process assignments, task intervals, and completion observations are recorded.
The example uses ideal local DDSIM and makes no hardware or quantum-advantage
claim.

One actual six-wire LiH shadow circuit then passes through QC, QCO,
optimization, placement, routing, and target synthesis. Device metadata is dated
evidence:

- IQM Emerald: AWS Braket `GetDevice` response, with 54 qubits and 85 undirected
  couplings in the saved snapshot. PRX is represented by the compiler's
  `R(theta, phi)` operation; CZ retains its physical operands.
- IBM Nighthawk: the official `ibm_miami` public r1 snapshot, calibrated on 17
  April 2026, pinned in Qiskit IBM Runtime. It is not a current r2 calibration.
- IonQ Forte-1: an AWS Braket response with 36 all-to-all connected qubits. The
  compiler synthesizes to GPI/GPI2/RZZ. The local target adds explicit pulse
  constraints absent from the QDMI metadata API: fixed equatorial rotations and
  RZZ angles in [0, π/2]. MLIR/QIR retain their exact R representations; the
  target-aware Qiskit export and native OpenQASM definitions preserve global
  phase. Units are radians, as in Braket, rather than direct IonQ API turns.

The compiler queries the real local QDMI SC provider (`mqt.sc.default`),
populated with those recorded models. The IonQ snapshot is extended with its
recorded native-pulse constraints and passed as an explicit compiler target.
These are not calls to the live vendor QDMI providers. Target-specific payloads
execute unchanged on DDSIM; no physical hardware job is submitted. Calibration
summaries are source-reported values, not an execution noise model. Topology
positions are schematic. The routing animation uses actual recorded refinement
endpoints and emitted operations, with interpolated motion between them.

A separate three-qubit parity-feedback program exposes counted loops, reset, and
conditional correction. Its bounded-unrolled form still contains adaptive
feedback. Repeat-until-success illustrates a runtime-dependent loop. Four-qubit
iterative QPE, with eight output bits and phase 1/3, supplies the 2,048-shot
adaptive runtime distribution.

These teaching programs use the bundled Emerald demonstration model with
explicit reset and control-flow capabilities. Their execution on DDSIM does not
certify those capabilities on the dated physical-device snapshots.

The QIR replay uses opt-in `MQT_MQSF_SHOT_TRACE` instrumentation in an isolated
single-program DDSIM worker. Each `JitSession.sample(1)` executes one circuit,
preserves the RNG stream, and timestamps completion. On Linux the worker clock
is aligned with the client's monotonic call timestamps and checked against the
submit/wait interval and returned shot order. This mode disables terminal batch
sampling and includes instrumentation overhead. It demonstrates real execution
rather than ordinary simulator throughput. OpenQASM captures have no per-shot
trace. The AFQMC batch instead records actual Python adapter call boundaries,
including native waits and indexed result decoding.

## Refresh native captures

Use native bindings, `mqt-cc`, the DDSIM shared library, and its adjacent worker
from the same checkout and build. Follow the repository and machine-local build
instructions. Commands below assume that environment's `python` is active; the
static packaging command above needs no native components.

```console
python presentations/mqsf2026/capture_programs.py \
  --compiler build/release-clang-ipo/mlir/tools/mqt-cc/mqt-cc
```

This writes `captures/programs.json`. The default captures parity, iterative
QPE, and repeat-until-success; `--scenario parity`, `qpe`, or `rus` limits the
run. The timeout is configurable with `--timeout`.

AFQMC also needs PennyLane, NumPy, SciPy, PySCF, OpenFermion, and Numba. Check
out the
[public AFQMC example](https://github.com/amazon-braket/amazon-braket-examples/tree/16cd791da7c3ec7e104851eb3bc502d00be1c1c3/examples/hybrid_quantum_algorithms/Quantum_Monte_Carlo_Chemistry)
at the linked revision outside this repository. The recorder verifies its
imported files against pinned hashes.

```console
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python presentations/mqsf2026/capture_afqmc.py \
  --source /path/to/Quantum_Monte_Carlo_Chemistry --bond-length 2.4 --steps 480

python presentations/mqsf2026/capture_execution.py \
  --library build/release-clang-ipo/lib/libmqt-core-qdmi-ddsim-device.so \
  --application presentations/mqsf2026/captures/afqmc.json
```

AFQMC defaults are `--snapshots 2048 --shots 256 --walkers 128 --steps 240`,
with `--dtau 0.02 --seed 17 --processes 4 --bond-length 1.6`. The committed
presentation capture selects `--bond-length 2.4 --steps 480`. Its
permutation/propagation seed does not seed the DDSIM measurement stream; actual
ordered outcomes are retained. The execution recorder defaults to
`--shots 2048 --seed 7`, requires Linux for QIR clock alignment, and writes
`captures/demo.json.gz` with the application merged.

To refresh device compilation, provide already downloaded AWS responses and the
[official pinned IBM configuration and properties](https://github.com/Qiskit/qiskit-ibm-runtime/tree/fa4cecc76321f9559456b132cfdfc7d06999f802/qiskit_ibm_runtime/fake_provider/backends/miami):

```console
python presentations/mqsf2026/capture_devices.py \
  --emerald /path/to/emerald-get-device.json \
  --ibm-config /path/to/conf_miami.json \
  --ibm-properties /path/to/props_miami.json \
  --ionq /path/to/forte-get-device.json \
  --source presentations/mqsf2026/captures/afqmc-source.qasm \
  --compiler build/release-clang-ipo/mlir/tools/mqt-cc/mqt-cc \
  --library build/release-clang-ipo/lib/libmqt-core-qdmi-ddsim-device.so
```

This script only reads metadata files, compiles locally, and executes on DDSIM.
It writes `captures/devices.json`. Use the same source circuit generated by the
AFQMC capture. After regenerating the execution fixture, merge the saved targets
and reference metadata before packaging:

```python
import gzip
import json
from pathlib import Path

captures = Path("presentations/mqsf2026/captures")
fixture = captures / "demo.json.gz"
data = json.loads(gzip.decompress(fixture.read_bytes()))
devices = json.loads((captures / "devices.json").read_text())
data["targets"] = devices["targets"]
data["target_provenance"] = devices["provenance"]
data["references"] = json.loads((captures / "references.json").read_text())
fixture.write_bytes(gzip.compress((json.dumps(data, indent=2) + "\n").encode(), mtime=0))
```

Preserve the curated compressed fixture and its scripts together. Intermediate
program/application JSON is untracked. Provenance retains source, script,
payload, and binary hashes as well as dependency versions and adaptations. Use
`--help` for each recorder's options. Rebuild and validate after any refresh.

## Packaging and validation

Read the Docs builds only the presentation with
`python -m uv run --no-project presentations/mqsf2026/build.py` into its HTML
output directory. It exposes the same ZIP as the HTML download. Native capture
generation stays outside Read the Docs. This presentation-branch override must
remain separate from Core's normal documentation build.

```console
uv run --no-project --with pygments --with mlir-pygments \
  --with openqasm-pygments --with pytest \
  pytest -o addopts= -q test/python/presentation/test_build.py

python -m pytest -o addopts= -q test/python/presentation

uv run --no-project --with playwright playwright install chromium
uv run --no-project --with playwright presentations/mqsf2026/check_browser.py
uv run --no-project --with playwright playwright install firefox
uv run --no-project --with playwright presentations/mqsf2026/check_browser.py --engine firefox
```

Run the full presentation suite in the native environment; AFQMC tests need
NumPy and PennyLane. The offline browser check accepts `--html`, `--browser`,
`--screenshot`, `--screenshots-dir`, and `--engine chromium|firefox`. The last
option saves every final slide at FullHD. Repository lint and native checks are
separate; local success does not establish hosted CI or Read the Docs status.

Asset and reference provenance is recorded in `assets/sources.json` and
`captures/references.json`. The QDMI wordmark is the official MQSC website SVG,
adapted to the white background. Regenerate tightly cropped equation paths and
the repository/device QR codes with
`uv run presentations/mqsf2026/render_equations.py`.

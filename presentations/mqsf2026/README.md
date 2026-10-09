# MQSF 2026

## System Software for Quantum Computing: From the Metal to the User

A 28-slide, 35-minute keynote for a 1920 × 1080 projector. The white and blue
MQSC presentation follows a small feedback program through compilation, uses
iterative phase estimation to explain execution, and closes the application loop
with a complete small H₂ quantum-classical AFQMC capture.

Every slide and progressive build works with the forward and back buttons of a
Logitech Spotlight configured to send standard presentation keys. The browser
replays captured native work. It does not compile, simulate, contact a device,
or rerun AFQMC. The downloaded presentation needs no server or network. See
[presenter-notes.md](presenter-notes.md) for the rehearsal narrative.

## Build and present

From the repository root:

```console
uv run --no-project presentations/mqsf2026/build.py
```

Open `build/mqsf2026/index.html` directly in a browser. Set the display to 1920
× 1080, use full screen, and rehearse with the actual clicker. The layout scales
to other window sizes while retaining its 16:9 canvas. The build also creates
`build/mqsf2026/mqsf-2026.zip`, containing the self-contained HTML file, the
full capture as `evidence.json.gz`, and the README and presenter notes. Code
highlighting, captured data, SVG illustrations, logos, and the font are embedded
in the HTML. External links and QR destinations are optional audience resources.

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

The notes overlay appears on the presentation display; use the separate notes
file for private rehearsal. A URL fragment such as `#18.0` opens a particular
slide and build. Browser printing produces the final build of each slide.

Slides 17 and 18 start their replay on the first forward press. A forward press
during playback finishes that build; the next press advances. Back returns to
the preceding build. The first replay uses recorded time; the second expands the
same timeline to 18 seconds. Neither advances the slide automatically.

## What the evidence contains

- A three-qubit parity-feedback example, with two rounds, for readable compiler
  and circuit views. The compiler preserves loops, reset, and measurement
  feedback; an alternate path unrolls bounded loops.
- Four-qubit iterative phase estimation with eight output bits and phase 1/3,
  captured with 2,048 shots per successful payload. It supplies the runtime
  distribution and timed replay.
- H₂ at 0.75 Å in STO-3G, with two electrons and four spin orbitals: 512
  matchgate measurement programs, 512 shots each, one native QDMI multi-program
  job, reconstructed trial overlaps, and 160 classical propagation steps with 64
  walkers. The capture includes a Hartree–Fock-trial comparison and an
  independently checked full configuration interaction reference.

Compiler artifacts and emitted OpenQASM 3 / adaptive QIR are real outputs.
Captures verify that the submitted payload matches the retrieved program and
that ordered shots match independently retrieved counts. API traces identify
whether they observe the native device C interface or Python calls. The AFQMC
recorder requires a single native job with all indexed programs.

The QIR replay uses opt-in `MQT_MQSF_SHOT_TRACE` instrumentation in the DDSIM
worker. The capture client selects one worker and an isolated single-program
job. Each `JitSession.sample(1)` executes one circuit, preserves the RNG stream,
and records its completion using `steady_clock`. On Linux these timestamps are
aligned with the client's `monotonic_ns` call timestamps and checked against the
submit/wait interval and returned shot order. This disables terminal batch
sampling and includes logging and per-shot execution overhead. These are
measured demonstration timings, not normal batch-throughput benchmarks. OpenQASM
captures have no per-shot timestamp trace.

## Refresh native captures

Use the native bindings, `mqt-cc`, DDSIM library, and DDSIM worker from the same
checkout and build. Follow the repository build instructions and machine-local
preset guidance. The static build above needs none of those native components.

With the native development environment active, generate the compiler captures:

```console
python presentations/mqsf2026/capture_programs.py \
  --compiler build/release-clang-ipo/mlir/tools/mqt-cc/mqt-cc
```

`--scenario parity` or `--scenario qpe` limits regeneration; the default is
both. The script writes `captures/programs.json`. It has a configurable
`--timeout`.

For AFQMC, the native environment also needs PennyLane, NumPy, SciPy, PySCF,
OpenFermion, and Numba. Check out the
[public AFQMC example](https://github.com/amazon-braket/amazon-braket-examples/tree/16cd791da7c3ec7e104851eb3bc502d00be1c1c3/examples/hybrid_quantum_algorithms/Quantum_Monte_Carlo_Chemistry)
at revision `16cd791da7c3ec7e104851eb3bc502d00be1c1c3` outside this repository.
The recorder verifies the imported helper files against pinned hashes.

```console
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python presentations/mqsf2026/capture_afqmc.py \
  --source /path/to/Quantum_Monte_Carlo_Chemistry

python presentations/mqsf2026/capture_execution.py \
  --library build/release-clang-ipo/src/qdmi/libmqt-core-qdmi-ddsim-device.so \
  --application presentations/mqsf2026/captures/afqmc.json
```

Adjust the compiler and library paths for a different build layout. Keep the
matching worker beside the library as the build/install process provides it. The
execution recorder requires Linux for aligned QIR shot timing. Its defaults are
`--shots 2048 --seed 7`; omit `--application` to record only the compiler
examples. AFQMC defaults are
`--snapshots 512 --shots 512 --walkers 64 --steps 160 --dtau 0.02 --seed 17`.
Use each script's `--help` for its options. AFQMC records actual simulator
outcomes; its permutation/propagation seed does not seed the DDSIM measurement
stream.

The final execution command merges the application and writes
`captures/demo.json.gz`. Commit this curated fixture with its capture scripts;
the intermediate JSON files remain untracked. Provenance includes script and
payload hashes, source revision, dependencies, parameters, and adaptations.
After a capture change, rebuild and validate the offline package before
replacing a working presentation copy.

## Scientific and deployment boundaries

Emerald supplies the captured 54-site, 90-edge coupling graph and native R/CZ
gates. Reset and unrestricted classical control are explicit demonstration
assumptions. Compiled payloads execute on ideal local DDSIM; this is not a
physical Emerald run or a calibration-derived noise model. The topology uses
schematic positions, and highlighted operations follow captured program order,
not a live mapping search.

AFQMC uses a checked Givens-angle sign adaptation. Its cached six-determinant
overlap expansion is exact within this tiny reconstructed H₂ space and does not
establish a scalable replacement for general overlap evaluation. Energy bands
show a pointwise walker-only standard error conditional on the collected
shadows. They exclude shadow sampling error, time-step error, and phaseless
bias. Walker positions are schematic; weights and occupations are recorded. This
demonstrates the complete small workflow, without a quantum-advantage claim. The
Slurm/cloud slide is deployment context; the capture ran locally.

The final full Benchpress comparison is pending. Slide 26 currently states the
measurement dimensions and explicitly shows that status. It contains no invented
performance or coverage result.

## Packaging and validation

This branch's Read the Docs override builds only the presentation. It runs
`python -m uv run --no-project presentations/mqsf2026/build.py` into the hosted
HTML directory and exposes the same ZIP as the HTML download. Native capture
generation stays outside Read the Docs. This branch-specific override must not
replace Core's documentation build in an upstream software change.

```console
uv run --no-project --with pygments --with mlir-pygments \
  --with openqasm-pygments --with pytest \
  pytest -o addopts= -q test/python/presentation/test_build.py

python -m pytest -o addopts= -q test/python/presentation

uv run --no-project --with playwright playwright install chromium
uv run --no-project --with playwright presentations/mqsf2026/check_browser.py
```

Run the full Python suite in the native development environment; the AFQMC tests
need NumPy and PennyLane. The browser check accepts `--html`, `--browser`,
`--screenshot`, and `--screenshots-dir` (the final build of each slide at
FullHD). It checks the built file with networking disabled. Repository lint and
focused native checks remain separate from presentation packaging; local results
do not imply hosted CI or Read the Docs success.

Logo provenance is recorded in `assets/sources.json`. The QDMI wordmark comes
from the supplied AWS/QDMI tutorial's original SVG. Its vector paths are
preserved, with navy fill for this deck's white background.

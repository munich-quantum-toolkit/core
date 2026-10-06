# MQSF 2026

## System Software for Quantum Computing: From the Metal to the User

This presentation is a living testbed for MQT Core. The browser explores real
compiler artifacts and replays recorded QDMI calls and measurement outcomes. It
does not compile or simulate programs in JavaScript. No server or network
connection is needed after downloading the presentation.

## Build and open

From the repository root:

```console
uv run --no-project presentations/mqsf2026/build.py
```

Open `build/mqsf2026/index.html` directly in a browser. The same directory
contains `mqsf-2026.zip`, the portable presentation bundle. All styles, code
highlighting, and captures are embedded. Large source artifacts have a
highlighted excerpt; the full original remains available for viewing and
download.

The presentation CI packages this bundle and checks it with networking disabled.
This branch's Read the Docs configuration builds only the presentation, without
installing MQT Core or LLVM. Its PR preview opens directly into the
presentation. That configuration is specific to this presentation branch and
must not replace Core's documentation configuration in an upstream software fix.

## Refresh the evidence

Use native bindings and `mqt-cc` built from the same checkout. Follow the normal
Core development setup; the static build above needs neither dependency.

First generate the compiler artifacts (the compiler path is configurable):

```console
python presentations/mqsf2026/capture_programs.py --compiler build/release-clang-ipo/mlir/tools/mqt-cc/mqt-cc
```

For the application capture, install PennyLane, NumPy, OpenFermion, and Numba in
the native Core environment. Check out the
[public AFQMC example](https://github.com/amazon-braket/amazon-braket-examples/tree/16cd791da7c3ec7e104851eb3bc502d00be1c1c3/examples/hybrid_quantum_algorithms/Quantum_Monte_Carlo_Chemistry)
at commit `16cd791da7c3ec7e104851eb3bc502d00be1c1c3` outside this repository.
The capture command verifies its helper files against pinned hashes before
importing them:

```console
python presentations/mqsf2026/capture_afqmc.py --source /path/to/Quantum_Monte_Carlo_Chemistry
python presentations/mqsf2026/capture_execution.py --library /path/to/libmqt-core-qdmi-ddsim-device.so --application presentations/mqsf2026/captures/afqmc.json
```

The shared library comes from the installed package's `mqt/core/lib` directory.
Omit `--application` to capture only Shor and QPE. Defaults capture 4 shots per
Shor payload and 64 per QPE payload; `--shor-shots` and `--shots` control these
budgets. Shor capture can take several minutes per payload. Use `--help` for
other options.

The committed first capture contains one shot per structured Shor format and
four per unrolled format. The larger structured attempts exceeded the capture
limit; their failure and the retained successful runs are recorded separately.

Generate compiler artifacts, optionally generate the small AFQMC batch, then
capture execution with the DDSIM shared library from that build. The execution
command verifies payload identity, job success, and agreement between ordered
shots and independently retrieved histograms. It writes the compressed fixture
used by the static packager. Intermediate JSON files remain untracked.

Commit refreshed captures together with changes to their capture scripts. Each
capture identifies its Core revision, artifact hashes, options, target model,
and presentation assumptions. Retain an earlier downloadable bundle until its
replacement passes the offline checks.

## Demonstration boundaries

- Emerald uses the bundled 54-site, 90-edge hardware model. Reset and dynamic
  classical control are explicit presentation assumptions. Execution is ideal
  DDSIM simulation; no physical Emerald run or calibration-based noise model is
  claimed.
- The structured compiler path preserves supported loops and feedback. Indexed
  qubit operations may require specialization for routing. The unrolled path
  expands bounded loops while retaining required measurement feedback.
- Adaptive QIR and OpenQASM 3 are actual exported programs. Qiskit views expose
  current export limitations and omit drawings that are impractical to display.
- Routing operations are shown in static program order, including operations in
  conditional regions. They are not a measured execution path or an animation of
  the routing search algorithm.
- The sequence diagram identifies the API layer actually observed. Histogram
  animation uses recorded shot ordering; its playback speed is not simulator
  performance evidence.
- The AFQMC example demonstrates a small quantum shadow-collection batch.
  Classical propagation, final chemistry energies, and quantum advantage are
  outside this capture. It uses separate QDMI jobs, not a native multi-program
  job or a Slurm deployment. An explicit Givens-angle sign adaptation is checked
  against the helpers' Majorana transformation convention before execution.
- The full Benchpress comparison awaits final measurements. The presentation
  does not substitute invented values or extrapolate full-suite coverage from
  the medium-set baseline.

## Validate

```console
uv run --no-project --with pygments --with mlir-pygments --with openqasm-pygments --with pytest pytest -o addopts= -q test/python/presentation
uv run --no-project --with playwright playwright install chromium
uv run --no-project --with playwright presentations/mqsf2026/check_browser.py
```

The browser check accepts `--browser` for an existing Chromium executable and
`--screenshot` to save the opening slide. Native capture regeneration requires
the additional dependencies documented by the capture commands.

# MQSF 2026

## System Software for Quantum Computing: From the Metal to the User

A seventeen-slide keynote for a 1920 × 1080 projector and a forward/backward
clicker. The story follows the shared software stack through an H₂ QC-AFQMC
workflow, device-aware compilation, structured programs, and adaptive execution.

## Build and present

```console
uv run --no-project presentations/mqsf2026/build.py
```

Open `build/mqsf2026/index.html` directly. No server or network is needed. The
build produces two archives in `build/mqsf2026/`:

- `mqsf-2026.zip`: the standalone HTML, slide-by-slide speaker notes, a short
  `README.txt` for the presentation laptop, and combined license/asset notices.
- `mqsf-2026-evidence.zip`: complete `evidence.json.gz`, editable SVG assets,
  original licenses and provenance, this technical guide, and the compiler
  change inventory. Development commands below apply to a repository checkout.

The presentation needs only the first archive. Its HTML embeds all runtime
content and the summaries needed for display. The evidence archive preserves the
full observations and compiler outputs without duplicating the HTML. Read the
Docs hosts both ZIPs; its standard offline download is the presentation ZIP.

| Action                      | Keys                                 |
| --------------------------- | ------------------------------------ |
| Next build or slide         | Right, Page Down, Space, Enter       |
| Previous build or slide     | Left, Page Up, Backspace             |
| First / last slide          | Home / End                           |
| Jump                        | Slide number, then Enter             |
| Overview                    | G or Escape; arrows and Enter select |
| Full screen / blackout      | F / B                                |
| Show notes on the projector | P                                    |

Every narrative change waits for a click. A click may start a finite replay,
which holds its final frame. Another forward click finishes any current replay
and advances immediately. Back, direct jumps, printing, and reduced-motion mode
show completed replays. Reduced motion shows a settled frame. A fragment such as
`#9.4` opens a particular slide and build. References never require a separate
reveal. Printing creates one page per slide.

Press F before presenting from the hosted preview. Fullscreen mode hides Read
the Docs controls that would otherwise cover the slide corners.

## Scientific evidence

The bounded chemistry example is stretched H₂, STO-3G, 1.6 Å: two electrons in
four spin orbitals, six determinants. The fixed trial rotation is 0.6 radians.
Its pair-encoded preparation is checked against the original DoubleExcitation on
the vacuum and Hartree–Fock states actually used. This is a subspace identity,
not a general replacement for the four-qubit gate.

The quantum allocation submits 4,096 programs with 1,024 shots each as one
native QDMI job. Classical shadows supply determinant overlaps. A dependent
Slurm allocation propagates 128 walkers for 240 steps per trial at Δτ = 0.02
Ha⁻¹, using four ranks on four compute nodes. Both shadow-derived and
Hartree–Fock trials use the same auxiliary-field seeds. All nodes are containers
on one physical host: this validates scheduler integration and distribution, not
cluster scaling. See the
[Slurm capture guide](https://github.com/munich-quantum-toolkit/core/blob/codex/mqsf-2026/presentations/mqsf2026/slurm/README.md)
for the capture setup.

Curves are recomputed from retained local energies and weights. The band is one
pointwise walker standard error, conditional on the shared shadow data. It
excludes finite-shadow error, finite-step/projection bias, and phaseless bias.
Both curves and the full measured range remain visible. There is no population
resampling or quantum-advantage claim. Dense six-determinant contractions are an
explicit small-example simplification, not a scalable AFQMC algorithm.

Snapshot 0's exact adapter-generated OpenQASM 3 payload is the source of every
device compilation. Each batch program is compiled to QIR against the DDSIM QDMI
device before submission. Packaging binds raw-source and executable hashes to
each measured compiler invocation, then checks target stages and executed
exports. Application execution and compilation may be recorded on different Core
revisions; each retains its original provenance, tied by the exact submitted
input. Target metadata comes from retained provider snapshots:

- IQM Emerald: 54 qubits; R/CZ native operations.
- IBM Nighthawk: the official public `ibm_miami` r1 snapshot, dated April 2026.
- IonQ Forte-1: 36 all-to-all connected qubits; GPI/GPI2/RZZ with documented
  equatorial-pulse and bounded-angle constraints.

The compiler queries local SC QDMI models populated from this metadata; these
are not live vendor-device calls. Compiled payloads execute unchanged on ideal
DDSIM. No hardware jobs are submitted. Layout positions are schematic. Routing
steps, SWAPs, search parents, costs, and layouts come from compiler telemetry;
intermediate visual motion between recorded endpoints is interpolation. Logical
qubits move on the device map; the routed circuit uses physical wire labels. The
executable panel follows the target's interchange format: IQM uses QIR, IBM
shows Qiskit's own vector drawing and QPY serialization, and IonQ shows native
OpenQASM 3. The IBM QPY artifact is exported from the retained native IR; its
separate provenance records the export tools and checked round trip. Execution
checks use QIR; QPY validation covers serialization and layout preservation.
Actual native completion and indexed retrieval timestamps drive separate matrix
colors. The replay rescales preparation, compilation, execution and retrieval
for readability. The program clock continues through preparation, execution, and
propagation using timestamps from the same recorded workflow. Imaginary time
remains the scientific horizontal axis of the energy plot. The final overview
reports the complete recorded Slurm workflow duration, including scheduler and
worker startup.

Complete compiler listings remain in the downloadable evidence. Target summaries
show an entry-point excerpt with a continuation marker. The structured view
shows the complete entry point, including every gate and control-flow line, with
original line numbers. Detailed excerpts show reset, measurement feedback, and
loops. These compact examples use QC → QCO → optimization → QC → QIR, without
placement, routing, or hardware-gate synthesis. The AFQMC Python view omits
docstrings and blank lines, and removes common indentation from extracted
definitions.

Grover program sizes are actual emitted UTF-8 OpenQASM bytes for 2–22 qubits,
with preserved and fully unrolled forms. Weyl coordinates are decompositions of
actual accepted two-qubit fusion matrices, with the signed convention recorded
in the evidence. The H2 example has one accepted CX-plus-SWAP fusion,
synthesized using two CZ gates. The canonicalization example includes source,
before/after IR, and a matrix-equivalence check.

The compact syndrome example keeps reset, counted loops, and feedback through
portable adaptive QIR. The evidence bundle also retains the literature RUS
example, which implements V₃ with its required failure correction;
imported-circuit Kraus matrices verify its semantics. Scalar qubit and
scratch-bit source names are retained as optional MLIR name locations; stripping
debug locations can remove these labels. Explicit register names take priority
over inferred labels. Scratch bits are declared inside the loops and branches.
OpenQASM keeps their scope; the Qiskit export reuses storage and initializes
zero-initialized locals on each region entry. Data and ancilla declarations are
separate so broadcasting retains the intended qubit roles. The RUS construction
follows
[Paetznick and Svore (2014)](https://www.rintonpress.com/xxqic14/qic-14-1516/1277-1301.pdf).

Iterative QPE uses a four-qubit unitary and one recycled control to estimate
phase 1/3 into eight output bits. The eigenstate is |1111⟩ and
U = P(π/6)⊗⁴. The outer loop resets and reuses the control for each bit; its
nested loop repeats controlled-U 2^(7−k) times. Measurement updates the runtime
phase correction θ ← θ/2 − (π/2) result[k] for the next round. Both loops and
classical feedback survive the portable compiler path.

The complete diagram is parsed from the exact submitted OpenQASM source. Qiskit
cannot export this combination of dynamic loop bounds, indexed measurement
destinations, and runtime feedback; the evidence records its unavailability
instead of substituting an unrolled circuit. QC, QCO, optimized QC, OpenQASM,
and QIR artifacts remain available.

Four native DDSIM workers execute 8,192 shots. Device calls, JIT initialization,
and actual 64-shot batch completions share one monotonic clock. Shot indices
connect concurrent arrivals to the ordered QDMI result; worker bars and the
histogram use those observations. Instrumentation overhead is included. This is
not a throughput benchmark. The QCO branch-sharing sampler is a separate
execution path from adaptive QIR.

## Capture and validate

Native capture is separate from static packaging. Use the repository build
instructions and the machine's supported optimized preset. The recorders are:

- `capture_slurm_afqmc.py`: dependent scheduler allocations and accounting.
- `capture_afqmc.py`: chemistry checks, actual batch payloads and observations,
  distributed walker tasks, scientific reductions.
- `capture_programs.py`: compact structured examples and compiler outputs.
- `capture_devices.py`: the same submitted source across three target models.
- `capture_compiler_evidence.py`: measured Grover sizes, canonicalization,
  routing consistency, and Weyl coordinates.
- `capture_execution.py`: unchanged QDMI device payloads, native calls, and JIT
  runtime/shot observations in isolated workers.

The [compiler change inventory](compiler-fixes.md) separates reusable compiler
fixes, presentation telemetry, and source-diagram limitations. AFQMC retains its
original execution provenance when portable examples and target compiler
artifacts are recaptured; it is not relabeled as a new application run.

```console
uv run --no-sync pytest test/python/presentation
uv run --no-project presentations/mqsf2026/build.py
uv run --no-project --with playwright presentations/mqsf2026/check_browser.py --browser /usr/bin/google-chrome
```

The browser check blocks networking and verifies every build, complete control
boundaries, circuit dependencies, recorded result progression, navigation,
interrupted animations, reduced motion, and print output. Rendered Chrome
inspection remains necessary for visual judgment. Rehearse on the actual laptop,
projector, and clicker before the event.

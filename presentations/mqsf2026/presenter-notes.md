# MQSF 2026 presenter notes

## System Software for Quantum Computing: From the Metal to the User

Lukas Burgholzer · MQSC / Technical University of Munich

The story follows a useful computing workflow: connect users and devices,
complete a hybrid application, inspect how its circuit reaches different
architectures, and then retain the classical structure needed by adaptive
programs. Budget 34 minutes plus pauses. References are always available; there
are no reference-only clicks.

Forward starts a scene's playback. During a finite replay, forward finishes it;
the next press advances. During a loop, forward stops and advances immediately.
Back restores a settled build. No animation changes slides automatically. For
rehearsal, type a slide number and Enter, or use a URL fragment such as `#9.2`.
Direct jumps open a settled state. `P` shows notes on the projector; keep this
file separate for private rehearsal.

## 1–2 · The shared stack · 4 minutes

**1. Title.** Introduce MQSC, with TUM as the secondary affiliation. The opening
world connects users, classical resources and different QPU technologies. The
promise is system software from the metal to the user.

**2. Hardware scales. Software breaks.** Start with the integration problem. The
same people and machines stay in place as their bespoke connections gather
around the shared stack. Advance once to see that stack, and again to reveal its
responsibilities: programming models, resource management and scheduling,
compiler infrastructure, and device interfaces. Resource orchestration belongs
to the larger architecture; it is not all implemented inside MQT Core.

Use the published MQSS paper as the architectural reference, without reading its
URL. Transition: “Let's follow a practical application through these layers,
then open up the compiler that connects them.”

## 3–5 · A practical hybrid application · 10 minutes

**3. Quantum-assisted AFQMC.** Let the whole workflow unfold. The speaker's
subject is software integration, not a claim of application expertise. A small
LiH active-space calculation provides a concrete example: STO-3G, a frozen Li 1s
core, two active electrons in three sigma orbitals, represented by six spin
orbitals. Quantum sampling collects shadows once; classical CPUs reuse them for
trial-overlap estimates during imaginary-time propagation. No quantum call is
invented inside each walker update.

The second build shows the complete captured PennyLane quantum function and its
broadcast call. It prepares a vacuum reference and trial, rotates into a
randomized matchgate basis, then collects outcomes. The fixed trial parameters
were tuned classically at 1.6 Å and reused for a stretched 2.4 Å bond. This is a
workflow demonstration, not a quantum VQE or quantum advantage result.

**4. Many programs, one hybrid workflow.** The QDMI replay starts at the actual
submission boundary, skipping circuit preparation. There are 2,048 programs, 256
shots each and 524,288 outcomes in one native job. Each tile groups 16 indexed
programs and appears when its actual result retrieval returns. The sequence and
result count use the same recorded clock.

The second build integrates the CPU part. Four local worker processes propagate
128 walkers for 480 steps per trial. Each timeline bar is one complete walker
task. That is measured wall time; the walker-weight animation uses imaginary
time, a projection parameter. The two coordinates are separate. This captures
local process parallelism, not a run on an HPC cluster. The execution pattern
can be partitioned across classical resources.

The 15 measured determinant overlaps permit exact dense contractions in this
small space. They are not a scalable replacement for general AFQMC overlap
computation. Grid positions are schematic; weights and process intervals are
recorded observations.

**5. Energy estimate.** The stretched geometry produces visible projection from
the reused trial. Grow the energy curves alongside the walker weights and
explain the weighted reduction. The zero line is exact FCI within the stated
active space, not the full molecule. A finite projection need not reach that
line, and the phaseless approximation can introduce bias.

The blue band is one walker standard error conditional on the shared shadow
data. Shadow uncertainty, finite time step and projection time, and phaseless
bias are outside it. Both trials share auxiliary-field seeds and are correlated.
The last point is a final-time snapshot, not a tail average. Conclude with the
engineering result: real quantum measurement programs supplied data for an
independently checked classical calculation.

## 6–8 · Device-aware compilation · 10 minutes

**6. From circuit to physical target.** A simulator is only the beginning. Keep
one actual six-wire LiH measurement circuit beside Emerald. First explain
placement; then let the forward/backward refinement move the logical qubits. The
endpoints come from actual search diagnostics. Movement between them is
interpolation, not additional compiler-search observations.

Next follow the emitted routed operations and SWAPs on the architecture, then
native synthesis. The counters show introspected operations, two-qubit
operations, depth and active wires. SWAP counts during refinement refer to that
trial; the final circuit uses the selected mapping. Graph positions are
schematic, not a chip-layout drawing.

**7. Change the device ID.** Each selection changes the topology first, then
replays placement, routing and synthesis of the identical input. Compare
Emerald, the public Miami Nighthawk snapshot, and all-to-all IonQ Forte-1. The
native IonQ basis is GPI/GPI2/RZZ. Its fixed pulse constraints and bounded RZZ
angles are supplied explicitly because the QDMI metadata API does not express
those parameter constraints. MLIR/QIR use exact R pulse forms; the native export
preserves their global phase. Braket uses radians; direct IonQ API turn units
differ.

All vendor data was captured in advance. Compilation uses a local SC QDMI model,
with the recorded IonQ constraints added to its compiler target. IBM's public r1
snapshot is dated, not a claim about the latest calibration. Compiled QIR
executes unchanged on DDSIM. No hardware job is submitted.

**8. Inside the compiler.** Now explain what made those transformations
possible. Wider consecutive windows show actual source, QC, QCO, then optimized
QCO. QC has mutable quantum references; QCO makes successive quantum values and
dependencies explicit. Standard MLIR dialects represent classical computation.
Point to one representative operation in each form rather than reading the whole
program. The generated LiH programs are longer than the projection window;
original line numbers identify it and full artifacts are bundled.

## 9–10 · Preserve and execute structure · 8 minutes

**9. Structured quantum programs.** Explicitly switch to the small teaching
example: two data qubits and a syndrome ancilla, reset, measured and corrected
inside a counted loop. It is a recognizable syndrome-extraction pattern, not a
complete error-correcting code. Read the complete source and follow the
measurement-to-correction connection.

Then show the whole structure in target-native OpenQASM. Gate runs are folded
for projection; every control boundary remains visible. The next build shows the
actual QIR entry point with labelled display folds. Reset, measurement, result
reading, branch conditions and loop backedges remain visible. Display folding
does not change the executable.

Expand a bounded loop when a target needs explicit rounds. Both outputs retain
adaptive feedback. Finally show the complete repeat-until-success source: its
runtime condition cannot be statically expanded. The “Why are we unrolling?”
paper is the reference throughout.

**10. Adaptive execution.** One moderate-speed replay follows four-qubit
iterative QPE with eight output bits, phase 1/3 and 2,048 shots. API calls and
histogram increments share recorded timestamps. The non-exact binary phase
produces a distribution. Instrumented serial DDSIM execution is intentionally
slowed for inspection; it is not a simulator throughput benchmark.

## 11–12 · Return to the whole system · 2 minutes

**11. MQT Core.** Zoom out from the demonstrated paths to the library: language
integrations, quantum IR, decision diagrams, ZX, compiler infrastructure, QDMI
and execution. Stars and cumulative downloads are recorded snapshots, not a
count of unique users. Source and documentation QR codes are already visible.

**12. From the metal to the user.** Let the same Core diagram morph into the
larger architecture. Users, classical infrastructure and QPUs return around it.
Recap practical hybrid computation, portable device access, device-aware
compilation and preserved quantum program structure. Credit MQT, QDMI, MQSS,
Munich Quantum Valley and the TUM Chair for Design Automation. Leave mq.sc
visible for discussion.

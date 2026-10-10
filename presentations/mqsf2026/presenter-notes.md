# MQSF 2026 presenter notes

## System Software for Quantum Computing: From the Metal to the User

Lukas Burgholzer · MQSC / Technical University of Munich

The through-line is a useful computing workflow: connect users and devices,
complete one scientific application, then open the compiler and execution layers
that make it possible. The audience should leave with three ideas: shared
interfaces connect the stack, the compiler adapts programs to devices, and
classical structure remains part of quantum programs.

Budget 34 minutes of content and one minute for pauses. Reach the molecule at
minute 4, device-aware compilation at minute 14, structured programs at minute
23, and the closing Core overview at minute 32. If time is tight, shorten the
code commentary and repeated animation cycles; keep the explanation of what was
actually measured.

Forward reveals the next build and starts its animation. For a one-shot replay,
one forward press finishes it and another advances. For a looping replay,
forward stops it and advances in the same press. Back restores the previous
build. Nothing advances slides automatically. The normal playbacks are listed in
the [README](README.md#build-and-present); every interaction works through the
clicker. During rehearsal, type a slide number and Enter to jump, or use a URL
fragment such as `#17.0`. `P` shows notes on the projected display, so keep this
file separate for private rehearsal.

## 1–4 · Establish the shared stack · 4 minutes

**1. System Software for Quantum Computing.** Open with the title's promise:
what must exist between a quantum machine and someone trying to solve a problem?
Let the moving connections carry that question from HPC and cloud resources
through software to different QPU technologies. Introduce MQSC and TUM without
narrating every logo.

**2. Hardware scales. Software breaks.** Reveal the user groups, computing
resources, and devices. Then reveal their connections. The problem is the
integration burden when every combination requires separate engineering. The
picture explains that burden; the claim is not that every existing tool fails.

**3. A shared software stack.** Keep the same people and machines in place as
the connections gather around the middle. “We want to connect a program and a
device once, then compose the workflow.” This is the thesis of the talk, so
allow a short pause before opening the stack.

**4. From the metal to the user.** Expand the stack into frontends, resources
and scheduling, compiler infrastructure, and backends. Give each a concrete
responsibility. QDMI contributes a shared device interface; MQT contributes
compiler and program infrastructure. The published MQSS paper supplies the
architectural reference. Transition with: “Let's follow a complete application
through these layers, then inspect the compiler that connects them.”

## 5–9 · Complete a small hybrid application · 10 minutes

**5. Quantum-assisted auxiliary-field Monte Carlo.** Ask for a molecular energy
before showing a circuit. LiH is the example: STO-3G, a frozen Li 1s core, two
active electrons, and three sigma spatial orbitals represented by six spin
orbitals. The molecular and orbital pictures are conceptual. The exact reference
later is confined to this active space, not the full physical molecule.

Explain imaginary-time projection as preferentially retaining low-energy
components. State the trial's origin now: its fixed parameters were tuned
classically to a nearly exact state in this small model. The demonstration shows
the measured-shadow/classical workflow; it does not show quantum trial
optimization or quantum advantage.

**6. The quantum–classical algorithm.** Walk through the whole picture once:
prepare the trial with a vacuum reference, measure it in randomized matchgate
bases, keep the bases and outcomes, reconstruct trial overlaps, and propagate
classical walkers. Quantum measurements are collected up front and reused. There
is no quantum call hidden inside every classical walker update. The energy is a
weighted reduction over that ensemble.

**7. Many programs, one device job.** Reveal the actual 2,048 programs, 256
shots per program, and 524,288 outcomes. This is one native QDMI multi-program
job. Start the 12-second slowed replay and follow submission, waiting, and
indexed retrieval; each tile groups 16 programs. The source is the actual
PennyLane adapter. Outcomes become visible when its recorded retrieval calls
return. The duration covers the full batch call and its surrounding adapter
work, not just device execution.

**8. Parallel propagation of the walker ensemble.** A walker is a Slater
determinant, a numerical electronic state. Let the 20-second loop show the
recorded importance weights through imaginary time. Positions in the grid are
schematic. The 15 measured determinant overlaps permit exact small-space
contractions for overlap, local energy, and force bias; they do not turn general
AFQMC into a 15-number problem.

Advance to the separate 10-second CPU timeline. Its four rows come from actual
worker processes, each bar is one complete walker task, and its horizontal
coordinate is wall time. The steady processing window is zoomed; the displayed
whole-pool duration also includes startup. Both trial variants use 128 walkers
and 240 steps. Distinguish these observed task intervals from the imaginary-time
coordinate in the preceding animation.

**9. From the walker ensemble to an energy estimate.** Grow the saved curves and
explain the weighted estimator. The blue shadow-derived trial already starts
near the exact active-space reference because of the classically tuned trial. Do
not describe its flat curve as discovery or convergence from an unoptimized
state. The Hartree–Fock-trial comparison illustrates how a different trial
changes the same propagation workflow.

The band is one walker-only standard error conditional on the shared shadow
data. Shadow-sampling uncertainty, time-step error, finite projection time, and
phaseless bias are outside that band. Both curves reuse the same per-walker
auxiliary-field seeds and are correlated. The displayed last point is a
final-time snapshot, not a tail average. Close this section with the engineering
result: real measurement programs supplied data for a complete, independently
checked classical calculation.

## 10–13 · Open the device-aware compiler · 9 minutes

**10. A simulator is only the beginning.** Keep the same application and take
one of its captured six-wire shadow circuits. A physical target adds
connectivity, native operations, and control constraints. Emerald is the first
model. Its metadata was captured in advance; the compiler queries a local QDMI
SC model populated from it. The payload still executes on ideal DDSIM. This is
not a live call to an IQM provider or a hardware submission.

**11. A program through the compiler.** Follow OpenQASM into QC, QCO,
optimization, and the declared target interface. Read one meaningful operation
per representation rather than the whole excerpt. QC expresses quantum
operations within MLIR; QCO gives successive quantum values explicit use and
ownership. Classical MLIR dialects remain available for arithmetic, functions,
and control flow. End with target rotations and entangling gates, not a list of
passes.

**12. Placement, routing and native gates.** Begin with the logical-to-physical
assignment. The forward/backward refinement replay lasts 18 seconds and follows
recorded compiler traversal endpoints. Explain why looking through a program in
both directions can improve placement. The 16-second routing replay then shows
the actual emitted operations and SWAPs. The final 14-second replay shows
synthesis to the declared gate set. Interpolation makes movement readable; it is
not additional compiler search evidence. Graph positions are a schematic layout,
not chip geometry.

**13. Change the target, recompile the program.** Keep the input fixed and
switch from Emerald to IBM Nighthawk, then IonQ Forte-1. The topology and
compiled program change together. The IBM evidence is the public `ibm_miami` r1
snapshot calibrated on 17 April 2026; do not call it today's r2 calibration.
IonQ exposes an all-to-all topology. Its compiled payload uses the supported
RX/RY/RZ/CNOT QIS interface; the provider still owns physical GPI/GPI2/ZZ
synthesis. Reported calibration averages are dated source metadata, not a noise
model for the ideal DDSIM results.

## 14–17 · Preserve and execute program structure · 9 minutes

**14. Structured quantum programs.** Explicitly change from the chemistry
measurement circuit to a teaching example: two data qubits and a parity ancilla,
repeated twice with reset and measurement-dependent correction. It is a
recognizable syndrome-extraction pattern, not a complete error-correcting code.
Read the loop and conditional once; the diagram connects the measurement to the
correction.

**15. Preserve structure all the way to the executable.** Show the same program
in QCO, target QCO, OpenQASM, and adaptive QIR. Physical gates and placement
change while the required classical control remains. Then compare the structured
program with bounded loop expansion. A target can require two explicit rounds
without eliminating either round's measurement-dependent correction. Use the
linked “Why are we unrolling?” paper as the resource, not as a detour into every
compilation policy.

**16. Loops, feedback and repeat-until-success.** A known trip count differs
from a loop whose stopping condition comes from a measurement. Show the actual
source and QIR condition/back-edge. This requires adaptive execution; it cannot
be replaced with one fixed number of repetitions in general. Keep the source
explanation short enough to leave time for the runtime evidence.

**17. Adaptive execution through QDMI.** Make the example change explicit: “For
a richer output distribution, use iterative phase estimation.” A reused query
qubit and a three-qubit eigenstate register produce eight measured bits. The
phase is 1/3, between 85/256 and 86/256; nearby integer outcomes are expected.

First play the 2,048-shot capture at its measured duration. Then play the same
timeline over 18 seconds and follow the native client calls, wait interval, and
accumulating histogram. Counts change at recorded shot completions. This client
calls the DDSIM QDMI C interface directly; it is separate from the PennyLane
adapter shown with the chemistry batch.

State the timing boundary once: the local DDSIM worker executes one circuit per
shot in opt-in serial demonstration instrumentation. The measurements include
that execution mode and logging overhead. They are not normal batch throughput
or remote QPU timings. The browser only replays these saved events.

## 18–19 · Return to the whole stack · 2 minutes

**18. MQT Core.** Zoom out from the compiler and runtime to frontends and
exchange formats, program libraries, device integration, and execution tools.
Then reveal the sourced repository/download snapshot and resource codes.
Cumulative downloads are not unique users. Credit the open-source contributors
and give the audience time to find the source and documentation links.

**19. System software for quantum computing.** Return to the opening picture
with evidence behind its layers: a hybrid scientific workflow, compilation for
different targets, and retained program structure through execution. Credit the
research and ecosystem partners, then leave `mq.sc` and the documentation code
visible. Invite people to bring their programs and devices to the stack.

# MQSF 2026 presenter notes

**System Software for Quantum Computing: From the Metal to the User** Lukas
Burgholzer · MQSC / Technical University of Munich

The talk's through-line is that useful quantum work crosses several software
boundaries. Follow one small program until those boundaries are familiar, then
show an application using the same stack. The audience should leave with three
ideas: retain program structure, connect devices through shared interfaces, and
integrate quantum work into a complete workflow.

Budget 34 minutes of content and one minute of room for pauses. Suggested
checkpoints are minute 8 at the start of the compiler, minute 16 at QDMI, minute
23 at the molecule, and minute 31 at the return to HPC. The compiler is the
center of the talk; avoid spending its time on the opening ecosystem map.

Forward reveals the next build; back reverses it. The runtime replays on slides
17–18 start with one forward press. During a replay, forward finishes the build
and the following press changes slides. The slow replay lasts 18 seconds. There
is no need to touch the mouse. During rehearsal, a slide number followed by
Enter jumps directly to that slide; `#18.0` in the URL opens the slow replay at
its start. `P` puts short notes on the projected display, so use this file
separately when notes must remain private.

## 1–4 · Why the software middle matters · 4 minutes

**1. System Software for Quantum Computing.** Start with the work, not a list of
products: “We have increasingly capable quantum machines. What does it take to
make them usable inside a computing workflow?” Introduce the title and move on.

**2. Hardware scales. Software breaks.** Let the researchers, HPC systems, and
hardware appear before discussing the connections. The point is the growing
integration burden when every pair needs a special bridge. Do not imply that
every existing software product is broken.

**3. Connect once. Compose the workflow.** Reuse the same picture. The shared
stack makes it possible to connect an application and a device without building
a new end-to-end system for each combination. Introduce open interfaces as a
practical engineering choice.

**4. Open the software stack.** Reveal device access, runtime/scheduling, and
compiler infrastructure in order. Give each one sentence: discover and control
the device; manage execution; translate and adapt the program. Promise to open
the compiler first, then follow its output into execution.

## 5–7 · Make the program familiar · 4 minutes

**5. A circuit is only part of a program.** Connect briefly to last year's
“Beyond Circuits.” Measurements, decisions, repeated work, and resets matter as
much as gate operations. The compiler must carry that meaning through its
representations.

**6. Measure. Correct. Repeat.** Introduce two data qubits and one parity
ancilla. The ancilla measures a relation between the data qubits, the program
may apply a correction, and the loop repeats twice. This is a teaching example
for feedback; it is not a complete error-correction protocol. Read only the
loop, measurement, and conditional line.

**7. Follow the same three wires.** Walk the circuit with the clicker: reset,
first parity interaction, second interaction, measurement, conditional
correction, enclosing loop. Pause long enough for the audience to locate each
operation. The diagram shows the loop body once; the loop annotation carries its
repetition.

## 8–13 · Open the compiler · 8 minutes

**8. Change the representation. Keep the structure.** Use the pipeline as the
map for this section. QC is convenient for importing and exporting operations on
qubits; QCO makes qubit dataflow explicit for transformations. Target stages
introduce hardware constraints. Do not teach the entire MLIR type system.

**9. The loop survives the language change.** Advance through the actual source,
QC, and QCO excerpts. Keep attention on the same loop and one quantum operation.
The claim is structural continuity; the rest of the listing is supporting
evidence, not something the audience must read.

**10. Give every qubit a physical home.** Show the captured placement on the
Emerald coupling graph. Each next press highlights the same operands in the
circuit and graph. Explain why locality matters. This small placement may not
require an inserted SWAP; do not narrate a routing operation that is absent. The
graph's positions are a readable layout, not physical chip coordinates.

**11. Speak the device's gate language.** The same circuit now uses native R/CZ
operations. Explain decomposition with one selected operation and keep its
physical sites visible. State once that Emerald provides the model, reset and
unrestricted classical control are demonstration assumptions, and the payload
executes on ideal DDSIM.

**12. Keep the loop—or expand the bounded repetition.** Show the representation
choice and its concrete size consequence. Unrolling is about bounded repeated
work; it does not remove the need for measurement feedback. Avoid describing
either adaptive program as a static circuit merely because its loop expanded.

**13. One compiled program. Two payload formats.** Next switches from OpenQASM 3
to adaptive QIR. Point out the corresponding conditional in both. Close the
compiler section with an executable artifact, then ask what the compiler needs
to know about the device that will receive it.

## 14–18 · Turn a payload into a result · 7 minutes

**14. The compiler needs answers from the device.** Reveal native operations,
connectivity, accepted program formats, and execution constraints. Hardware
information must flow back into compilation. QDMI supports this conversation as
well as job submission.

**15. Discover. Submit. Retrieve.** Three verbs prepare the audience for the
trace. Explain that the next sequence comes from calls to the DDSIM device C
interface. It is observed execution evidence, not a hand-authored animation of
what an API ought to do.

**16. An inexact phase produces a distribution.** Explicitly change examples:
“The small parity program made compilation readable. For execution, let's use
phase estimation.” One query qubit is reused alongside a three-qubit eigenstate
register. Eight output bits approximate 1/3, which lies between 85/256 and
86/256. Establish why a distribution is expected before revealing the actual
2,048-shot histogram.

**17. First, the run at its recorded speed.** Press once and let it finish.
“That was the recorded run at its measured speed.” This is local ideal DDSIM,
with serial demonstration instrumentation. Each recorded shot executes the
circuit separately; the timing includes that mode and logging overhead. It is
not a claim about ordinary simulator batch throughput or remote QPU speed.

**18. Now slow down the same clock.** Press once. Use the 18-second replay to
follow the client code, submission, wait, and accumulating outcomes. All three
views share the recorded clock; counts change at actual shot completions. Slow
playback adds explanatory time, not invented events. If ahead of the animation,
forward completes it; then advance into the application question.

## 19–24 · Complete a small scientific workflow · 8 minutes

**19. A molecule is the question—not a circuit.** Ask for the ground-state
energy of H₂. This instance uses a 0.75 Å bond, STO-3G, two electrons, and four
spin orbitals. Its size makes every boundary checkable. The molecular picture
and potential curve are conceptual; the measured energy results arrive later.

**20. Prepare. Randomize. Measure.** Explain the quantum part: prepare a trial
state with a vacuum reference, apply a random matchgate basis, and measure.
Retain both the basis and measured bits. The displayed sample is from the
capture. The basis convention is checked explicitly against the Majorana
transformation before submitting any payload.

**21. 512 programs. One QDMI job.** Reveal the batch size, 512 shots per
program, and 262,144 outcomes. This is one native multi-program job, not a
visual group of separate jobs. The displayed duration measures the entire
PennyLane batch call, including its work around the device. Indexed programs,
ordered shots, and independently retrieved counts are checked.

**22. Quantum data guides classical walkers.** A walker is a Slater determinant,
not an electron moving through space. Trial overlaps reconstructed from shadows
enter the classical force bias and local-energy evaluation. Show the actual
saved weights of the 64 walkers as propagation proceeds for 160 steps. The grid
arrangement is schematic. The six-coefficient cache is exact in this small
space; it is not a claim that general AFQMC has become a six-number problem.

**23. A complete small chemistry calculation.** Grow the recorded energy curve,
then reveal the Hartree–Fock-trial comparison and explain the FCI reference. The
band is one pointwise walker-only standard error, conditional on these measured
shadows. It excludes shadow sampling error, time-step error, and phaseless bias.
The last point is a final-time estimate, not a tail average. The two curves
illustrate a functioning workflow; this example establishes neither quantum
advantage nor a general accuracy improvement.

**24. The useful unit is the whole workflow.** Pull back: quantum measurements
became data for overlap reconstruction and classical propagation, which became
an energy estimate. Use the recorded durations only to show where work occurs;
they are not a speedup comparison. This closes the loop promised in the title.

## 25–28 · Put the stack in context · 3 minutes

**25. Fit quantum work into existing infrastructure.** Return to the stack from
the application's viewpoint. Users already have schedulers, compute resources,
and workflows. The AWS/QDMI tutorial supplies the Slurm/cloud deployment
context; the chemistry result shown here was captured locally on DDSIM.

**26. Coverage. Circuit quality. Compilation cost.** The current rehearsal slide
has no completed full-suite benchmark result. Explain the three dimensions
briefly and state that measurements are pending. When final results arrive,
replace this slide with sourced, scoped evidence; do not substitute a smaller
subset or an extrapolated number.

**27. Built in the open. Built together.** Credit MQT contributors, QDMI/MQSS
partners, TUM's Chair for Design Automation, and Munich Quantum Valley. Keep
MQSC's engineering role and the research/community contributions visible. Give
the audience a moment to find the documentation link.

**28. Hardware scales. Put the software to work.** Land the three outcomes in
order: preserve program structure, connect the device, run the workflow. Invite
people to bring programs and devices to the stack. Leave the resource codes
visible for questions.

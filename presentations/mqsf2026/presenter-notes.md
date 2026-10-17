# MQSF rehearsal notes

Allow about 35 minutes, including animations. A click advances to the next
build; clicking during playback finishes it and advances immediately. Let
replays settle when pointing to their results.

Use this file privately. **P displays the short notes on the projector.** Right
and Left move forward and back. Type a slide number and press Enter to jump; G
opens the overview. F toggles fullscreen and B blacks out the screen. F3.2 means
slide 3, second build.

## F1 · System Software for Quantum Computing: From the Metal to the User

Introduce yourself as CTO and co-founder of MQSC, and Senior Researcher at TUM's
Chair for Design Automation. The full title names the scope: software connecting
quantum devices to the people and infrastructure that use them.

Keep the opening still and give the audience a moment to settle.

**Next:** “Let’s start with the ecosystem this software has to connect.”

## F2 · Connecting the Quantum Ecosystem

As machines and deployment options grow, software must support many routes
between users, classical resources, and quantum processors.

- **F2.1:** Start with the range of quantum processors.
- **F2.2–F2.4:** Follow a researcher from a workstation to one cloud-accessible
  processor, then another provider, then the other modalities.
- **F2.5–F2.6:** Increase the classical workload: cloud compute, then HPC with
  on-premise quantum access and connections to cloud services.
- **F2.7:** Add the other users. Let the full set of paths appear: maintaining
  bespoke software for every path will not scale.

**Next:** “We need common interfaces and a stack whose parts can be reused.”

## F3 · A Shared Software Stack

Modularity, efficiency, and extensibility apply across programming models,
devices, and deployment environments.

- **F3.1:** Let the classical resources move beneath the stack.
- **F3.2:** Follow the frontend reveal from Qiskit through PennyLane, OpenQASM,
  and the SDK extension point. Every user can enter through any frontend.
- **F3.3:** Separate programs from runtime requests. Compilation transforms the
  program; orchestration coordinates compilation, resources, and execution.
- **F3.4:** The backend interface supplies device constraints and carries jobs
  and results. Point to the published MQSS architecture for further detail.

These are architectural roles. The next view makes the current products
concrete.

**Next:** “Here is how we are building these parts at MQSC.”

## F4 · Our Approach: From the Metal to the User

Build the product story from device access to compilation and orchestration.

- **F4.1–F4.2:** Start at the backend interface. QDMI connects the IQM, Amazon
  Braket, DDSIM, and IBM implementations shown here.
- **F4.3:** The MQT Compiler Collection consumes programs and device
  constraints.
- **F4.4:** MQSC Orchestrator connects workflows to execution and scheduling,
  including Slurm. Both upper components live in MQT Core.
- **F4.5:** Finish with openness, community, freedom to change components,
  practical integration, and performance.

Credit TUM CDA as MQT's origin and the continuing MQT community. QDMI grew from
TUM CDA, TUM CAPS, and LRZ, with MQV, MQSC, and community contributors
maintaining and developing it. MQSC drives these projects into practical
deployments. Downloads are cumulative downloads, not individual users.

**Next:** “Let’s follow an application through this stack.”

## F5 · A Classical–Quantum Workflow

Introduce quantum-assisted auxiliary-field Quantum Monte Carlo as a practical
example of the software demands of a hybrid application.

**Next:** “The quantum calculation is one part of a much larger classical
workflow.”

## F6 · Quantum-Assisted Auxiliary-Field Quantum Monte Carlo

Explain where each part runs and what passes between them. The demonstration
adapts the public Amazon Braket example to H₂ in the STO-3G basis.

- **F6.1:** Classical preparation defines the molecular problem, trial state,
  and random measurement bases. Credit the paper and Amazon Braket example.
- **F6.2:** Quantum sampling collects classical shadows: 4,096 programs with
  1,024 shots each. These observations supply trial-state overlap estimates.
- **F6.3:** Classical Monte Carlo reuses the observations while propagating
  walkers. Circle area shows walker weight; the reduction gives the energy.

Keep the explanation at the workflow level. The small molecule demonstrates
software integration and does not establish quantum advantage.

**Next:** “Now let’s look at the code and execution behind those boxes.”

## F7 · Execute the Hybrid Workflow

Follow preparation, the quantum batch, and the dependent classical allocation as
one workflow.

- **F7.1:** Orient the audience in the actual preparation and batch code.
- **F7.2:** Follow basis generation, validation, PennyLane preprocessing, and
  conversion to OpenQASM. Each cell tracks one prepared program.
- **F7.3:** Let the quantum panel expand. One QDMI job contains 4,096 compiled
  QIR programs and 4,194,304 shots. Distinguish compilation, execution
  completion, and result retrieval as the counters advance.
- **F7.4:** Move to four Slurm ranks on four compute nodes, 128 walkers, and 240
  imaginary-time steps per trial. Let both trial traces finish.
- **F7.5:** Pause on the complete workflow. The quantum observations are reused
  throughout classical propagation.

The jobs and Slurm allocations are real; the four nodes are containers on one
host, and quantum execution uses ideal DDSIM. Playback rescales recorded wall
time; the energy plot uses imaginary time. Its reference is exact FCI within
this basis. The band is walker standard error conditional on the shared shadows,
not total uncertainty.

**Next:** “Moving this program toward a physical device adds constraints.”

## F8 · From Quantum Program to Executable

A device brings a native gate set, connectivity, and an accepted program format.
The compiler connects the application to those choices.

**Next:** “We’ll keep one actual AFQMC payload and follow its compilation.”

## F9 · Compiling for a Device

Keep the input and hardware-independent optimization fixed while the
target-dependent stages change.

- **F9.1–F9.2:** Show snapshot 0's submitted payload and its optimized form.
- **F9.3:** Introduce Emerald: 54 qubits, its coupling map, and R/CZ operations.
- **F9.4:** Place logical qubits on physical sites. Establish the wire labels.
- **F9.5:** Follow the captured final routing pass. Connect gate and SWAP
  highlights to movement on the device map.
- **F9.6–F9.7:** Synthesize native operations, then produce adaptive QIR. Use
  gate counts and depth to orient the audience through the changes.
- **F9.8:** Switch to IBM's Nighthawk model. Show the actual Qiskit drawing and
  QPY serialization alongside the new topology and native circuit.
- **F9.9:** Switch to IonQ's all-to-all target and GPI/GPI2/RZZ gate set. The
  output panel now shows native OpenQASM 3.
- **F9.10:** Zoom out to the frontend, output, and integration ecosystem.

The local QDMI models use captured provider metadata. DDSIM executes each
target's compiled QIR unchanged. IBM's QPY export is separately serialized and
round-trip checked, including its physical layout; no hardware job is submitted.
Its drawing hides idle wires but retains every operation. Device QR codes lead
to the corresponding QDMI implementations.

**Next:** “Device constraints are one challenge. Preserving the structure of a
larger program is another.”

## F10 · Why Are We Unrolling?

Introduce loops and classical control flow as useful program structure,
especially for repeated error-correction cycles and larger algorithms. Credit
the paper collaborators.

**Next:** “Grover’s algorithm makes the cost of expanding that structure
visible.”

## F11 · Grover Program Growth

A repeated algorithm can remain a compact program.

Read the sketch: prepare, then repeat oracle and diffusion. Let the unrolled
curve appear before the structure-preserving curve. At 22 qubits, structured
OpenQASM uses 1,249 bytes and unrolled OpenQASM uses 2,319,631 bytes: about 1.25
KB versus 2.32 MB. The logarithmic graph shows program size, not runtime or
speedup.

**Next:** “The compiler also needs to preserve loops containing measurements and
decisions.”

## F12 · Structured Programs, All the Way Down

Read one recognizable syndrome-extraction program through to executable control
flow.

- **F12.1:** Start with the complete source and circuit. The flag is reset and
  reused; its measurement controls a correction.
- **F12.2:** Reveal the complete QIR entry point. Compilation passed through QC
  and QCO, optimization, and lowering to adaptive QIR.
- **F12.3–F12.5:** Follow reset, measurement feedback, and the counted loop.
  Connect each source highlight to its QIR excerpt. Scratch bit s is loop-local.

This portable path does not perform placement, routing, or native-gate
synthesis. Credit the paper collaborators and jeff's role in exchanging
structured programs. The evidence also contains a repeat-until-success V₃
example for discussion.

**Next:** “Those loops and decisions now need to execute with their classical
calculations.”

## F13 · From Compilation to Execution

DDSIM's QIR runtime executes classical control flow and quantum operations.
Introduce the JIT as the bridge from QIR to the running program.

**Next:** “Iterative phase estimation makes that interaction visible.”

## F14 · A Runtime for Structured Quantum Programs

The runtime executes nested loops and measurement feedback in the compiled
program.

- **F14.1:** Read the diagram before starting. Four qubits carry the eigenstate;
  a fifth is the recycled control. The outer loop estimates eight bits, while
  the inner loop applies the required controlled-U power.
- **F14.2:** Follow QDMI into DDSIM and the LLVM JIT, then the four worker
  counters and accumulating histogram. Let the replay settle.

U = P(π/6)⊗⁴ acts on |1111⟩, giving phase 1/3. Eight bits cannot represent it
exactly, so the 8,192 shots show a distribution. Round k repeats controlled-U
2^(7−k) times; measurement updates θ ← θ/2 − (π/2) result[k]. Both loops and the
floating-point feedback remain in QIR.

The diagram follows the submitted OpenQASM. Its dynamic bounds and classical
indexing are currently outside the Qiskit export path. Worker counters,
histogram arrivals, and calls share recorded 64-shot completion batches. The
slowed replay illustrates execution rather than throughput.

**Next:** “These examples use capabilities available together in MQT Core.”

## F15 · Built for the Quantum Ecosystem

Return from individual examples to the reusable software behind them.

**Next:** “Let me collect the pieces you can use today.”

## F16 · MQT Core Capabilities

Connect the demonstrated features to entry points for users and developers.

- **F16.1–F16.2:** Broad OpenQASM 3.1 support and QIR 2.1 Base/Adaptive Profile
  compilation and execution. Avoid implying every OpenQASM feature is supported;
  subroutines and calibrations are examples outside the current scope.
- **F16.3–F16.4:** Qiskit's native C API integration, then PennyLane's hybrid
  workflows through QDMI. Relate these to the application.
- **F16.5:** Installation and tooling: platform wheels, supported Python
  versions, and portable LLVM distributions.
- **F16.6:** Developer resources, followed by the QOSS survey invitation. Invite
  anyone building or using quantum software. Leave the QR visible long enough to
  scan: about 20 minutes to complete, open through 29 October 2026.

Repository and documentation QR codes are available on the earlier builds.

**Next:** “All of this depends on an active software community.”

## F17 · From the Metal to the User

Bring the application, compiler, and runtime back to the product tower: QDMI,
the MQT Compiler Collection, and MQSC Orchestrator.

“Built with the Community. Driven into practice by MQSC.” Credit the people and
organizations represented here, thank the audience, and invite discussion.

Leave the closing image visible. mq.sc and the social links remain in the
footer.

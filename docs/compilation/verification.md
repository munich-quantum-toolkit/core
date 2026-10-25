# Verify compilation results

{py:func}`mqt.core.mlir.verify` compares two borrowed
{py:class}`~mqt.core.mlir.QCOProgram` objects without compiling or changing
them. It resolves concrete classical execution once, then selects eligible
simulation, construction, and alternating DD checks within a shared worker
budget.

```python
from mqt.core import mlir

# before and after are QCOProgram objects retained across compilation.
result = mlir.verify(before, after)
assert result.verdict == mlir.VerificationVerdict.EQUIVALENT, result.reason
```

The default compares all circuit data inputs, preserves global phase, and checks
terminal readout identity. A successful check applies only to the supplied
classical bindings. It does not verify all values of a classical parameter or
prove termination for other bindings.

## Choose the contract

| Contract                 | Compared inputs and outputs                                                                                                                                 |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Default `CIRCUIT_INPUTS` | Every source quantum input is free; all source quantum outputs and terminal readout identities are compared.                                                |
| `INITIALIZED_PROGRAM`    | Entry allocations start at zero; actual quantum function arguments remain free.                                                                             |
| `zero_ancillas`          | Listed source slots start at zero under either domain. Their outputs remain observable unless omitted by the validated function interface and proved clean. |
| `TERMINAL_PROBABILITIES` | Joint probabilities of selected returned classical bits for every free quantum input; unobserved quantum outputs may be discarded.                          |

`VerificationContract.num_inputs` asserts the common source wire count. Routing
workspaces do not become new free inputs. Coherent checks require private
workspaces to return to zero before omission. A `qco.sink` or `qtensor.dealloc`
alone never grants permission to discard quantum information.

The verifier decodes `mqt.layout` directly: `initial[logical]` is an initial
device position, `routing[position]` is its final position, and
`sites[position]` is its hardware ID. Final logical placement is
`routing[initial[logical]]`. Returned classical bits retain result position,
register name, and original bit index; sparse destinations are not independently
renumbered. Flattened quantum arguments and returns retain their positional wire
correspondence. The verifier does not claim equality of arbitrary MLIR function
types. When quantum returns and layout metadata are both present, their decoded
output identities must agree.

Target programs require source or layout provenance. Raw physical programs
without that metadata are unsupported; their caller must supply a known layout.

Concrete scalar arguments are bound by entry argument position:

```python
contract = mlir.VerificationContract()
contract.domain = mlir.VerificationDomain.INITIALIZED_PROGRAM
result = mlir.verify(
    before,
    after,
    contract=contract,
    left_bindings={0: 0.125, 1: 3, 2: True},
    right_bindings={0: 0.125, 1: 3, 2: True},
)
```

Both programs must be valid QCO. Nested concrete branches and bounded execution
of `scf.for`, `scf.while`, and nonrecursive calls are supported, including
loop-carried qubits and bound parameters. Terminal measurements can cross calls
and flow through returned bit registers. A measurement result used for control,
a quantum operation after measuring that wire, an executed reset, unresolved
symbolic values needed by the executed path, and nested quantum allocation are
unsupported. An unexecuted branch does not restrict the resolved execution.
Shared entry-allocation rules continue to apply; see {doc}`../mlir/index`.

## Select a relation

Let $A,B:\mathbb C^d\to\mathbb C^M$ be the aligned maps, including initialized
ancillas. The free input dimension is $d$, not the dimension including ancillas.

- `EXACT` with `PRESERVE` compares $\max_{ij}|A_{ij}-B_{ij}|$ to `tolerance`
  (default $10^{-12}$). This preserves the contract needed for a function that
  may later be controlled.
- `EXACT` with `IGNORE` allows one global phase:
  $\min_{|z|=1}\max_{ij}|A_{ij}-zB_{ij}|$. This is appropriate for a closed
  whole program when global phase is unobservable. The implementation uses a
  phase witness and conservative Frobenius bounds; undecided cases return
  `INCONCLUSIVE` rather than assuming that the overlap phase minimizes the
  maximum entry residual.
- `PROCESS_DISTANCE` uses $\sqrt{1-|\operatorname{Tr}(A^\dagger B)/d|^2}$ and
  requires `IGNORE`. For unitaries, average gate infidelity is $d/(d+1)$ times
  its square. It gives no worst-input error guarantee.
- `AMPLITUDE_DISTANCE` uses $\|A-B\|_F/\sqrt d$ and requires `PRESERVE`.

```python
options = mlir.VerificationOptions()
options.relation = mlir.VerificationRelation.PROCESS_DISTANCE
options.phase = mlir.VerificationPhase.IGNORE
options.tolerance = 1e-6
result = mlir.verify(before, after, options=options)
```

Approximate distances use residual contractions near equality to avoid
subtracting nearly equal trace magnitudes. `tolerance` is the accepted synthesis
error; `numerical_tolerance` is a separate uncertainty interval around that
threshold. The latter is an engineering guard, not a certified rounding bound.
Neither setting changes DD canonicalization. Results report the DD tolerance,
accepted threshold, input domain, and concrete bindings. Results near an
approximate threshold are inconclusive. Resolved dense gates must remain finite
and unitary within the IR's matrix tolerance of $10^{-10}$. Non-finite
comparison arithmetic and residuals outside the isometry bound are inconclusive.

For `TERMINAL_PROBABILITIES`, each returned outcome $y$ defines an effect
$E_y=A^\dagger\Pi_y A$. Construction compares every selected joint-outcome
effect entry within the numerical tolerance given by `tolerance`. The projector
$\Pi_y$ includes constant and copied bits and acts as identity on unobserved
outputs. Select labels such as `result0:result[7]` in
`contract.observed_outputs`; an empty selection uses all returned bits. Missing
or undefined labels are unsupported. Matching these effects does not establish
equality of reduced quantum states. Approximate partial equivalence and general
channel equivalence are unsupported.

## Interpret results and limits

`VerificationResult` has no Boolean conversion. Inspect its verdict:

| Verdict          | Meaning                                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `EQUIVALENT`     | A complete eligible comparison met the requested relation within its stated numerical settings.                           |
| `NOT_EQUIVALENT` | A completed comparison or conservative simulation bound refuted the relation.                                             |
| `INCONCLUSIVE`   | Numerical uncertainty, successful finite sampling, cancellation, a deadline, or resource exhaustion prevented a decision. |
| `UNSUPPORTED`    | The resolved execution, requested checker, or shared interface lies outside the supported contract.                       |

`AUTO` runs at most three useful checks with two workers by default. Simulation
uses computational basis states followed by independent samples from the six
single-qubit Pauli eigenstates, with the recorded seed. Passing those samples is
inconclusive for a free quantum input domain; no confidence percentage is
inferred. A fully initialized domain has a single input and can be decided by
simulation. Approximate average-distance and terminal-effect relations use
construction; the aligned unitary case also admits alternating checking for
coherent relations. Forced ineligible methods return `UNSUPPORTED`. An all-input
comparison that omits private workspace requires construction to prove the
workspace clean; finite simulation cannot establish that premise.

`options.limits` bounds steps, call depth, retained gates, gate/classical
allocation payload bytes, DD backing storage, qubits, joint outcomes, elapsed
time, and workers. DD byte accounting includes comparison caches and overlapping
worker packages; it excludes allocator overhead, stack storage, and the MLIR
context. Interpreter metadata is bounded by the step limit. These are not hard
process-memory limits. Cancellation is cooperative; DD recursion polls, while
individual garbage-collection and external allocation operations may delay a
stop. Existing bounded-width dense matrix construction and eigensolvers also run
to completion before the next cancellation check. Completed workers are joined
before returning, and results retain no DD handles. C++ callers may supply a
`std::stop_token`; Python callers can set `options.limits.timeout_ms`.

Concurrent calls own separate DD packages and share worker grants with sampling.
They never change global numerical settings. Do not change the legacy global DD
tolerance, or mutate or consume either borrowed program, while checking it.
Python releases the GIL and retains both program owners during the call.

C++ callers include `mqt/Compiler/Verification.h` for the borrowed-program
`mlir::verify` overload and link `MQTCompilerVerification`. For explicit QCO
functions and `DDArgumentBindings`, include
`mqt/Dialect/QCO/Utils/Verification.h`, call `mlir::qco::verify`, and link
`MLIRQCOVerification`. Invalid IR or malformed options return `FailureOr`
failure; unsupported execution is a structured result. No QCEC runtime
dependency is required.

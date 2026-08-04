# From a PennyLane circuit to any gate-based QDMI device

PennyLane is especially pleasant when an application matters more than the
details of a provider SDK: write a quantum function, choose a device, and let
the framework differentiate the result. QDMI solves the complementary problem on
the hardware side. It gives devices stable identities and one common interface
for capabilities, program submission, and results.

MQT Core connects those two layers. With the optional PennyLane integration, the
built-in decision-diagram simulator is a regular PennyLane device:

```bash
uv add "mqt-core[pennylane]"
```

```python
import pennylane as qp

device = qp.device("mqt.ddsim.default", wires=2, shots=1000)


@qp.qnode(device)
def bell_state():
    qp.Hadamard(0)
    qp.CNOT(wires=[0, 1])
    return qp.counts(wires=[0, 1])


print(bell_state())
```

No provider SDK or credentials are needed for this local example. An installed
hardware-provider package can register another PennyLane entry point under its
stable QDMI device ID and reuse the same
{py:class}`~mqt.core.plugins.pennylane.device.QDMIDevice`.

## What crosses the QDMI boundary

PennyLane first preprocesses each quantum tape. It validates wires, defers
mid-circuit measurements, splits non-commuting measurements, diagonalizes
observables, decomposes higher-level operations, expands broadcasts, and
replaces requested results with computational-basis sampling. That last step
keeps the QDMI boundary small: the device returns samples, and PennyLane
reconstructs samples, counts, probabilities, expectation values, variances,
Hamiltonian results, and shot-vector partitions.

MQT Core then selects exactly one program format:

1. OpenQASM 3 if the QDMI device advertises it.
2. OpenQASM 2 if, and only if, OpenQASM 3 is unavailable.
3. A focused error before job creation if neither format is available.

The OpenQASM 3 converter is deliberately small and capability driven. It emits
one contiguous qubit register, one classical register, bound finite numeric
parameters, provider-advertised operation names, and a final whole-register
measurement. It does not emit an include, gate definition, pragma, or gate
modifier. Semantic aliases bridge common provider spellings such as `cx` and
`cnot`, `p` and `phaseshift`, `sdg` and `si`, or `rxx` and `xx`.

When QASM3 is advertised, a conversion failure remains a QASM3 error. The
integration never hides an unsupported program by silently retrying QASM2. For a
QASM2-only device, MQT Core calls PennyLane's serializer as follows:

```python
qp.to_openqasm(
    tape,
    wires=device_wires,
    rotations=False,
    measure_all=True,
)
```

Observable rotations are already present after preprocessing, so disabling
serializer rotations prevents applying them twice.

## A finite-shot QAOA application

The checked-in
[`pennylane_qaoa.py`](https://github.com/munich-quantum-toolkit/core/blob/main/docs/_scripts/pennylane_qaoa.py)
example solves MaxCut for a fixed four-node graph. It prepares one QAOA layer
with two trainable parameters, evaluates `qp.qaoa.maxcut`, differentiates with
the parameter-shift rule, performs optimizer updates, samples the final circuit,
and extracts the best observed cut.

Run the complete local application from the repository root:

```bash
uv run python docs/_scripts/pennylane_qaoa.py
```

The script reports the initial cost and gradient, final parameters and cost,
best sampled bit string, QDMI job count, and elapsed wall-clock time. A typical
run has the following shape; finite-shot values vary:

```text
initial cost: -1.5...
initial gradient: (...)
parameters: (...)
final cost: -2....
best observed cut: 0101 (3 edges)
QDMI jobs: ...
elapsed: ... s
```

Parameter-shift is worth making visible here. A two-parameter gradient is not
one remote execution: PennyLane expands it into shifted tapes, and each tape is
submitted as a separate QDMI job. The current implementation executes those jobs
sequentially. Parallel QDMI submission could improve remote latency without
changing the device or converted-program APIs.

The circuit factory accepts any
{py:class}`~mqt.core.plugins.pennylane.device.QDMIDevice`, so a provider demo
can use the same application with another stable device ID:

```python
import pennylane as qp

device = qp.device(
    "provider.stable-device-id",
    wires=4,
    shots=200,
    # Provider-specific entry points translate familiar keyword arguments
    # into QDMI session and job parameters.
)
```

For paid remote simulators or QPUs, keep shots and optimizer steps deliberately
small until the complete circuit has passed locally.

## Direct generic construction

Provider entry points are the convenient public surface, but applications can
also open any registered gate-based device explicitly:

```python
import pennylane as qp

from mqt.core.plugins.pennylane import QDMIDevice

device = QDMIDevice(
    device_id="provider.stable-device-id",
    wires=["a", "b", "c", "d"],
    shots=[(100, 2), 500],
    session_parameters={
        "base_url": "provider-device-selector",
        "token": "...",
    },
    job_parameters={
        "custom1": "provider-job-value",
    },
)
```

Arbitrary PennyLane wire labels map deterministically to contiguous QASM
indices. The converter validates one- and two-qubit loci advertised by QDMI but
does not route a circuit. A topology-incompatible program therefore fails before
submission.

Shot vectors, batches, and parameter-shift tapes are executed in order. Analytic
execution is rejected because hardware-style QDMI jobs require finite shots.

## Supported gate-level scope

The QASM3 path covers the gate-based operations needed by the local simulator,
remote circuit simulators, QAOA, and ordinary variational applications: identity
and Pauli gates; H, S, T, SX and their supported adjoints; RX, RY, RZ, and phase
shift; controlled Pauli and phase gates; Toffoli, SWAP, and CSWAP; ISWAP, PSWAP,
ECR; and Ising XX, XY, YY, and ZZ rotations. PennyLane decomposes higher-level
operations when their decomposition reaches an advertised set.

The current implementation does not provide pulse programming, provider-specific
non-gate properties, routing, or parallel job submission. The
{py:class}`~mqt.core.plugins.pennylane.converter.ConvertedProgram` boundary
keeps payload, selected format, wire mapping, and measurement order together, so
a compiler-backed or another exchange-format converter can replace the current
text conversion later without changing user circuits.

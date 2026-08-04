---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# PennyLane interface for gate-based QDMI devices

The {py:mod}`mqt.core.plugins.pennylane` module implements PennyLane's device
interface for gate-based quantum devices exposed through QDMI. PennyLane
programs are preprocessed into executable tapes, converted to a program format
advertised by the selected QDMI device, submitted through FoMaC, and
reconstructed from finite-shot QDMI results.

Any registered gate-based QDMI device can use this integration if it advertises
OpenQASM 3 or OpenQASM 2, accepts finite-shot jobs, and returns
computational-basis samples. Specialized neutral-atom interfaces, pulse-level
control, and analytic execution are outside this contract. The examples below
use the local [DD-based simulator device](ddsim_device.md) included with MQT
Core, so they are reproducible without credentials or remote resources.

Install MQT Core with the optional PennyLane dependency into the active
environment:

```bash
uv pip install "mqt-core[pennylane]"
```

`uv pip install` installs packages into an environment. In contrast, `uv add`
modifies the dependency metadata of a uv-managed project; consult the
[`uv` command reference](https://docs.astral.sh/uv/reference/cli/) for the two
workflows.

## Finite-shot execution

The following imports and plotting configuration are shared by the executable
examples. Figures use SVG output and an explicit light background so that axes,
labels, and graph annotations remain legible with either documentation theme.

```{code-cell} ipython3
from collections import Counter
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qp

from mqt.core.plugins.pennylane import QDMIDevice

%config InlineBackend.figure_formats = ['svg']

plt.rcParams.update(
    {
        "axes.facecolor": "#f8f9fb",
        "figure.facecolor": "#f8f9fb",
        "savefig.facecolor": "#f8f9fb",
        "text.color": "#202124",
        "axes.labelcolor": "#202124",
        "axes.edgecolor": "#5f6368",
        "xtick.color": "#3c4043",
        "ytick.color": "#3c4043",
    }
)
```

A Bell-state circuit verifies stable-ID discovery, circuit conversion, and
sample reconstruction through the DDSIM QDMI device.

```{code-cell} ipython3
bell_device = qp.device("mqt.ddsim.default", wires=2, shots=1000)


@qp.qnode(bell_device)
def bell_state():
    qp.Hadamard(0)
    qp.CNOT(wires=[0, 1])
    return qp.counts(wires=[0, 1])


bell_counts = bell_state()
bell_counts
```

Only the computational-basis states $00$ and $11$ have nonzero probability, up
to finite-shot fluctuations in their relative frequencies.

## Program conversion and result reconstruction

PennyLane first preprocesses every quantum tape. The preprocessing pipeline
validates wires and shots, defers measurements, splits non-commuting
measurements, diagonalizes observables, decomposes higher-level operations,
expands broadcasted parameters, and replaces requested measurements with
computational-basis sampling. PennyLane subsequently reconstructs samples,
counts, probabilities, expectation values, variances, Hamiltonian results, and
shot-vector partitions from those raw samples.

MQT Core selects the program format in the following order:

1. OpenQASM 3 if the QDMI device advertises it.
2. OpenQASM 2 only if OpenQASM 3 is unavailable.
3. A format error before job creation if neither format is available.

The OpenQASM 3 converter is capability driven. It emits one contiguous qubit
array, one classical bit array, finite bound numeric parameters, operation names
advertised by the QDMI device, and a final whole-register measurement. It emits
no include, custom gate definition, pragma, or gate modifier. Semantic alias
groups account for equivalent gate names such as `cx` and `cnot`, `p` and
`phaseshift`, `sdg` and `si`, or `rxx` and `xx`.

If OpenQASM 3 is advertised, a conversion failure remains an OpenQASM 3 error;
the integration does not retry with OpenQASM 2. For a device that advertises
only OpenQASM 2, MQT Core invokes PennyLane's serializer with

```python
qp.to_openqasm(
    tape,
    wires=device_wires,
    rotations=False,
    measure_all=True,
)
```

Observable rotations have already been inserted during preprocessing, so
disabling serializer rotations prevents duplicate basis changes.

## Finite-shot MaxCut QAOA

Consider MaxCut on the fixed graph $G=(V,E)$ with $V=\{0,1,2,3\}$ and
$E=\{(0,1),(0,2),(1,2),(2,3)\}$. The node positions are fixed to make the
visualization deterministic.

```{code-cell} ipython3
graph = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3)])
positions = {
    0: (-1.0, 0.75),
    1: (-1.0, -0.75),
    2: (0.25, 0.0),
    3: (1.35, 0.0),
}

figure, axis = plt.subplots(figsize=(5.5, 3.3))
nx.draw_networkx(
    graph,
    pos=positions,
    ax=axis,
    node_color="#4c78a8",
    edge_color="#5f6368",
    font_color="white",
    node_size=850,
    width=2,
)
axis.set_title("Four-node MaxCut instance")
axis.set_axis_off()
figure.tight_layout()
```

PennyLane constructs the cost and mixer Hamiltonians. The ansatz applies
Hadamard gates followed by one QAOA cost layer and one mixer layer,
parameterized by $\gamma$ and $\beta$.

```{code-cell} ipython3
cost_hamiltonian, mixer_hamiltonian = qp.qaoa.maxcut(graph)


def ansatz(parameters):
    for wire in graph.nodes:
        qp.Hadamard(wire)
    qp.qaoa.cost_layer(parameters[0], cost_hamiltonian)
    qp.qaoa.mixer_layer(parameters[1], mixer_hamiltonian)


qaoa_device = qp.device("mqt.ddsim.default", wires=4, shots=1000)
assert isinstance(qaoa_device, QDMIDevice)


@qp.qnode(qaoa_device, diff_method="parameter-shift")
def cost(parameters):
    ansatz(parameters)
    return qp.expval(cost_hamiltonian)


@qp.qnode(qaoa_device)
def sample(parameters):
    ansatz(parameters)
    return qp.sample(wires=range(4))
```

The calculation below evaluates the initial parameter-shift gradient and
performs four gradient-descent updates. Each objective value is an independent
finite-shot estimate; the resulting sequence is therefore not expected to be
monotonic.

```{code-cell} ipython3
parameters = qp.numpy.array([0.5, 0.5], requires_grad=True)
optimizer = qp.GradientDescentOptimizer(stepsize=0.15)
jobs_before = qaoa_device.submitted_jobs
started = time.monotonic()

objective_values = [float(cost(parameters))]
initial_gradient = np.asarray(qp.grad(cost)(parameters), dtype=float)

for _ in range(4):
    parameters = optimizer.step(cost, parameters)
    objective_values.append(float(cost(parameters)))

samples = np.asarray(sample(parameters), dtype=np.int8)
submitted_jobs = qaoa_device.submitted_jobs - jobs_before
elapsed = time.monotonic() - started

print(f"Initial gradient: {initial_gradient}")
print(f"Final parameters: {np.asarray(parameters)}")
print(f"QDMI jobs submitted: {submitted_jobs}")
print(f"Elapsed time: {elapsed:.3f} s")
```

Parameter-shift expands one gradient evaluation into several shifted tapes. Each
executable tape is submitted as a distinct QDMI job. The current implementation
submits these jobs sequentially; parallel QDMI submission is not part of the
present device interface.

The sampled bit strings determine candidate bipartitions. The cut value is the
number of graph edges whose endpoints have different bit values.

```{code-cell} ipython3
def bitstring(sample_row):
    return "".join(str(int(bit)) for bit in sample_row)


def cut_value(candidate):
    return sum(
        candidate[first] != candidate[second] for first, second in graph.edges
    )


sample_counts = Counter(bitstring(row) for row in samples)
best_bitstring = max(
    sample_counts,
    key=lambda candidate: (cut_value(candidate), sample_counts[candidate], candidate),
)
best_cut = cut_value(best_bitstring)

print(f"Best sampled partition: {best_bitstring}")
print(f"Cut edges: {best_cut} of {graph.number_of_edges()}")
```

The following panels show the noisy objective estimates, the empirical
bit-string distribution, and the highest-cut partition observed in the final
sample. Orange edges cross that partition.

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(14, 3.8))

axes[0].plot(
    range(len(objective_values)),
    objective_values,
    marker="o",
    color="#4c78a8",
)
axes[0].set(
    xlabel="Optimizer update",
    ylabel=r"$\langle H_C \rangle$",
    title="Noisy finite-shot objective estimates",
    xticks=range(len(objective_values)),
)
axes[0].grid(alpha=0.25)

ordered_bitstrings = sorted(sample_counts)
axes[1].bar(
    ordered_bitstrings,
    [sample_counts[candidate] for candidate in ordered_bitstrings],
    color="#4c78a8",
)
axes[1].set(
    xlabel="Bit string",
    ylabel="Observed count",
    title="Final sampled distribution",
)
axes[1].tick_params(axis="x", rotation=60)

node_colors = [
    "#4c78a8" if best_bitstring[node] == "0" else "#e45756"
    for node in graph.nodes
]
edge_colors = [
    "#f28e2b"
    if best_bitstring[first] != best_bitstring[second]
    else "#9aa0a6"
    for first, second in graph.edges
]
edge_widths = [
    3.2 if best_bitstring[first] != best_bitstring[second] else 1.5
    for first, second in graph.edges
]
nx.draw_networkx(
    graph,
    pos=positions,
    ax=axes[2],
    node_color=node_colors,
    edge_color=edge_colors,
    width=edge_widths,
    font_color="white",
    node_size=850,
)
axes[2].set_title(f"Best sampled cut: {best_cut} edges")
axes[2].set_axis_off()

figure.tight_layout()
```

## Direct generic construction

Stable entry points such as `mqt.ddsim.default` provide the simplest
construction. A device integration package can register a small
{py:class}`~mqt.core.plugins.pennylane.device.QDMIDevice` subclass under the
stable ID of another QDMI device. Applications may also construct the generic
class directly:

```python
import pennylane as qp

from mqt.core.plugins.pennylane import QDMIDevice

device_id = "stable ID returned by the QDMI device registration"
device = QDMIDevice(
    device_id=device_id,
    wires=["a", "b", "c", "d"],
    shots=[(100, 2), 500],
    session_parameters={
        "base_url": "device endpoint or selector",
        "token": "...",
    },
    job_parameters={
        "custom1": "device-specific job value",
    },
)
```

Arbitrary PennyLane wire labels map deterministically to contiguous QASM
indices. The converter validates the one- and two-qubit loci advertised through
QDMI but does not route circuits. A topology-incompatible program therefore
fails before submission. Shot vectors, batches, and parameter-shift tapes are
executed in order, and every execution requires finite shots.

## Supported gate-level scope

The OpenQASM 3 path covers identity and Pauli gates; H, S, T, SX and supported
adjoints; RX, RY, RZ, and phase shift; controlled Pauli and phase gates;
Toffoli, SWAP, and CSWAP; ISWAP, PSWAP, and ECR; and Ising XX, XY, YY, and ZZ
rotations. PennyLane decomposes higher-level operations when their
decompositions reach operations advertised by the QDMI device.

The interface does not implement pulse programming, device-specific non-gate
properties, routing, analytic execution, or parallel job submission. The
{py:class}`~mqt.core.plugins.pennylane.converter.ConvertedProgram` record keeps
the payload, selected format, wire mapping, and measurement order together. This
boundary permits a compiler-backed or alternative exchange-format converter to
replace the current text conversion without changing the PennyLane device API.

---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# MQT Core DD

MQT Core provides a fully-fledged, high-performance decision diagram package for quantum computing.
The resulting library allows for the efficient representation and manipulation of quantum states and operations.
If you are not yet familiar with decision diagrams as a data structure, you might want to start with the [introduction to quantum decision diagrams](#how-do-quantum-decision-diagrams-work) below.

Throughout the MQT, this library enables many classical simulation, synthesis, or verification techniques.
While primarily developed in C++, the corresponding functionality is also exposed to Python users in the form of the {py:mod}`mqt.core.dd` module.
The following section provides an overview on how to work with decision diagrams in MQT Core from Python.

## Quickstart

In its simplest use case, the MQT Core DD package can be used as a classical circuit simulator using the
{py:func}`~mqt.core.dd.sample` function.
The underlying simulation approach supports mid-circuit measurements, reset operations, as well as classically-controlled operations. For example, the following code snippet demonstrates how to simulate the iterative quantum phase estimation algorithm shown in [the MQT Core IR Quickstart guide](mqt_core_ir).

```{code-cell} ipython3
from mqt.core.dd import sample
from mqt.core.ir import QuantumComputation
from mqt.core.ir.operations import OpType

from math import pi

theta = 3 * pi / 8
precision = 3

# Create an empty quantum computation
qc = QuantumComputation()

# Counting register
q = qc.add_qubit_register(1, "q")

# Eigenstate register
psi = qc.add_qubit_register(1, "psi")

# Classical register for the result, the estimated phase is `0.c_2 c_1 c_0 * pi`
c = qc.add_classical_register(precision, "c")

# Prepare psi in the eigenstate |1>
qc.x(psi[0])

for i in range(precision):
  # Hadamard on the working qubit
  qc.h(q[0])

  # Controlled phase gate
  qc.cp(2**(precision - i - 1) * theta, q[0], psi[0])

  # Iterative inverse QFT
  for j in range(i):
    qc.if_(op_type=OpType.p, target=q[0], control_bit=c[j], params=[-pi / 2**(i - j)])
  qc.h(q[0])

  # Measure the result
  qc.measure(q[0], c[i])

  # Reset the qubit if not finished
  if i < precision - 1:
    qc.reset(q[0])

# Run the simulation
counts = sample(qc, 1024)
```

```{code-cell} ipython3
---
tags: [remove-cell]
---
from matplotlib import pyplot as plt

def generate_plot(counts: dict[str, int], name: str, light: bool) -> None:
    if light:
        plt.style.use('default')
    else:
        plt.style.use('dark_background')

    # Create the bar plot
    fig, ax = plt.subplots()
    bars = ax.bar(counts.keys(), counts.values(), color='#0065bd')

    # Annotate counts above the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Set background to transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel("Measurement Outcome")
    plt.ylabel("Counts")

    # export to SVG
    filename = "fig-" + name + ("-light" if light else "-dark") + ".svg"
    plt.savefig(filename, format="svg")

name = 'qpe'
generate_plot(counts, name, light=True)
generate_plot(counts, name, light=False)
```

```{figure} fig-qpe-light.svg
:align: center
:width: 75%
:figclass: only-light
```

```{figure} fig-qpe-dark.svg
:align: center
:width: 75%
:figclass: only-dark
```

The {py:func}`~mqt.core.dd.sample` function is a high-level interface to the decision diagram package that does not require any knowledge of the underlying data structure.
In a similar fashion, the {py:func}`~mqt.core.dd.simulate_statevector` and {py:func}`~mqt.core.dd.build_unitary` functions can be used to perform statevector simulation or to construct the unitary matrix representation of a quantum circuit, respectively.

```{code-cell} ipython3
from mqt.core.dd import simulate_statevector

import numpy as np

qc = QuantumComputation(2)
qc.h(0)
qc.cx(0, 1)

vec = np.array(simulate_statevector(qc), copy=False)
with np.printoptions(precision=3, suppress=True):
  print(vec)
```

```{code-cell} ipython3
from mqt.core.dd import build_unitary

unitary = np.array(build_unitary(qc), copy=False)
with np.printoptions(precision=3, suppress=True):
  print(unitary)
```

Both of these functions are inherently limited in their scalability due to the exponential growth of the resulting data structures.
MQT Core also allows one to work with decision diagrams directly, which is particularly useful for larger quantum circuits.
To this end, the {py:class}`~mqt.core.dd.DDPackage` class provides a low-level interface to the decision diagram package.
An instance of this class can be used to simulate quantum circuits (see {py:func}`~mqt.core.dd.simulate`),
construct unitary matrices (see {py:func}`~mqt.core.dd.build_functionality`), or perform other operations on decision diagrams.

```{code-cell} ipython3
from mqt.core.dd import DDPackage, simulate

dd = DDPackage(qc.num_qubits)
zero_state_dd = dd.zero_state(qc.num_qubits)
out_state_dd = simulate(qc, zero_state_dd, dd)
```

If the [Graphviz](https://www.graphviz.org/) library is installed, the `graphviz` Python package can be used to visualize resulting
decision diagram via the {py:meth}`~mqt.core.dd.VectorDD.to_dot` method.
To directly, generate SVG files, the {py:meth}`~mqt.core.dd.VectorDD.to_svg` method can be used.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 20%
    align: center
---
import graphviz

graphviz.Source(out_state_dd.to_dot())
```

The DD package provides list of additional functionality when it comes to working with decision diagrams.
Check out the full API documentation of the {py:class}`~mqt.core.dd.DDPackage` class for more details.

## How do Quantum Decision Diagrams Work?

Decision diagrams were introduced in the 1980s as a data structure for the efficient representation and manipulation of Boolean functions {cite:p}`bryantGraphbasedAlgorithmsBoolean1986`.
This led to the emergence of a wide variety of decision diagrams, including BDDs, FBDDs, KFDDs, MTBDDs, and ZDDs (see, for example, {cite:p}`bryantSymbolicBooleanManipulation1992,wegenerBranchingProgramsBinary2000,gergovEfficientBooleanManipulation1994,drechslerEfficientRepresentationManipulation1994,baharAlgebraicDecisionDiagrams1993,minatoZerosuppressedBDDsSet1993`), which made them a crucial tool in the development of modern circuits and systems.
Because of their previous success, decision diagrams have been proposed for application in the realm of quantum computing {cite:p}`willeDecisionDiagramsQuantum2023,willeToolsQuantumComputing2022,millerQMDDDecisionDiagram2006,niemannQMDDsEfficientQuantum2016,zulehnerHowEfficientlyHandle2019,hongTensorNetworkBased2020,vinkhuijzenLIMDDDecisionDiagram2021`.
Particularly for design tasks like _simulation_ {cite:p}`viamontesImprovingGatelevelSimulation2003,zulehnerAdvancedSimulationQuantum2019,hillmichJustRealThing2020,burgholzerHybridSchrodingerFeynmanSimulation2021,vinkhuijzenLIMDDDecisionDiagram2021,hillmichApproximatingDecisionDiagrams2022,burgholzerSimulationPathsQuantum2022,grurlNoiseawareQuantumCircuit2023,matoMixeddimensionalQuantumCircuit2023,sanderHamiltonianSimulationDecision2023`, _synthesis_ {cite:p}`niemannEfficientSynthesisQuantum2014,abdollahiAnalysisSynthesisQuantum2006,soekenSynthesisReversibleCircuits2012,zulehnerOnepassDesignReversible2018,adarshSyReCSynthesizerMQT2022,matoMixeddimensionalQuditState2024`, and _verification_ {cite:p}`burgholzerAdvancedEquivalenceChecking2021,burgholzerRandomStimuliGeneration2021,burgholzerVerifyingResultsIBM2020,wangXQDDbasedVerificationMethod2008,smithQuantumLogicSynthesis2019,hongEquivalenceCheckingDynamic2021` of quantum circuits, they recently attracted great attention.

The following sections provide a comprehensive guide for quantum computing with decision diagrams, including the representation of quantum states and operations and the fundamental operations on decision diagrams.

### Representation of Quantum States

First, we review how quantum states are represented using decision diagrams.
To this end, we consider the simple case of a single-qubit system.
The state $\ket{\Psi}$ of such a system is described by two complex-valued, normalized amplitudes $\alpha_0$ and $\alpha_1$, that is,

```{math}
:label: ssstate
\ket{\Psi} = \alpha_0 \ket{0} + \alpha_1 \ket{1},
```

which is commonly represented as a statevector

```{math}
\ket{\Psi}\equiv \begin{bmatrix} \alpha_0 & \alpha_1	\end{bmatrix}^\top.
```

A rather simple observation and consequence of [](ssstate) is that this vector can be equally split into a contribution of the $\ket{0}$ state ($\alpha_0$) and a contribution of the $\ket{1}$ state ($\alpha_1$), that is,

```{math}
:label: splitting
\bigl(
\overbrace{
\overset{\ket{0}}{\begin{bmatrix} \alpha_0
\end{bmatrix}}
\ \ \
\overset{\ket{1}}{\begin{bmatrix} \alpha_1
\end{bmatrix}}
}^{\ket{\Psi}}
\bigr)^\top.
```

This decomposition is the core of the decision-diagram formalism.
The decision diagram representing $\ket{\Psi}$ has the structure

```{image} _static/dd-figure-01.svg
:width: 15%
:align: center
```

It consists of a single _node_ with one _incoming edge_ that represents the entry point in the decision diagram, as well as two _successors_ that represent the split shown in [](#splitting) and end in a _terminal_ node (the black box).
The state's amplitudes are annotated at the respective edges.
Edges without annotations correspond to an edge weight of 1.

````{admonition} Example _(Single-Qubit States)_
:class: tip
Consider the computational basis states $\ket{0}$ and $\ket{1}$.
Then, the corresponding decision diagrams have the structures

```{image} _static/dd-figure-02.svg
:align: center
:width: 8%
```
```{math}
\ket{0}\equiv\begin{bmatrix}1 & 0\end{bmatrix}^\top
```

and

```{image} _static/dd-figure-03.svg
:align: center
:width: 8%
```
```{math}
\ket{1}\equiv\begin{bmatrix}0 & 1\end{bmatrix}^\top
```

In each of the cases, one of the successors ends in the terminal node, while the other ends in a \emph{zero stub} (indicated by a black dot)---uncannily resembling the corresponding vector descriptions.
````

Building off the intuition of a single-qubit state, we can move to larger systems.

````{admonition} Example _(Multi-Qubit States)_
:class: tip

Consider the following statevector of a three-qubit system:
```{math}
\ket{\Psi} = \begin{bmatrix} \frac{1}{2\sqrt{2}} & \frac{1}{2\sqrt{2}} &  \frac{1}{2} & 0 & \frac{1}{2\sqrt{2}} & \frac{1}{2\sqrt{2}} & \frac{1}{2} & 0\end{bmatrix}^T
```
Then, $\ket{\Psi}$ can be recursively split into equally-sized parts similar to [](#splitting), i.e.,
```{math}
\overbrace{
\overbrace{\begin{matrix}
\overbrace{\begin{matrix}
\bigl[ \overset{\ket{000}}{
\begin{matrix} \frac{1}{2\sqrt{2}}
\end{matrix} }
& \overset{\ket{001}}{
\begin{matrix} \frac{1}{2\sqrt{2}}
\end{matrix} }
\end{matrix}}^{\ket{00q_0}}
& \overbrace{
\begin{matrix}\overset{\ket{010}}{
\begin{matrix} \frac{1}{2}
\end{matrix}}
& \overset{\ket{011}}{
\begin{matrix} 0
\end{matrix} }
\end{matrix}}^{\ket{01q_0}}
\end{matrix}}^{\ket{0q_1q_0}}
\ \
\overbrace{\begin{matrix}
\overbrace{\begin{matrix}
\overset{\ket{100}}{
\begin{matrix} \frac{1}{2\sqrt{2}}
\end{matrix} }
& \overset{\ket{101}}{
\begin{matrix} \frac{1}{2\sqrt{2}}
\end{matrix} }
\end{matrix}}^{\ket{10q_0}}
& \overbrace{
\begin{matrix} \overset{\ket{110}}{
\begin{matrix} \frac{1}{2}
\end{matrix} }
& \overset{\ket{111}}{
\begin{matrix} 0
\end{matrix} } \bigr]^\top
\end{matrix}}^{\ket{11q_0}}
\end{matrix}}^{\ket{1q_1q_0}}
}^{\ket{q_2q_1q_0}}
```
where $q_2, q_1, q_0 \in \{0, 1\}$.
This directly translates to the decision-diagram formalism:

```{image} _static/dd-figure-04.svg
:align: center
:width: 65%
```
```{math}
:label: 3qbdd
```

Each level of the decision diagram consists of decision nodes with corresponding left and right successor edges.
These successors represent the path that leads to an amplitude where the local quantum system (corresponding to the _level_ of the node, annotated here with the labels) is in the $\ket{0}$ (left successor) or the $\ket{1}$ state (right successor).
````

At this point, this has been just a one-to-one translation between the statevector and a fancy graphical representation.
The unique core feature of decision diagrams is that their graph structure allows redundant parts to be merged in the representation instead of being represented repeatedly.

````{admonition} Example _(Redundancy in Decision Diagrams)_
:class: tip
Observe how, as in the previous example, the left and right successors of the top-level node (labeled $q_2$) lead to exactly the same structure (highlighted by dashed rectangles in [](#3qbdd)).
As a result, the whole sub-diagram does not need to be represented twice, i.e.,

```{image} _static/dd-figure-05.svg
:align: center
:width: 40%
```

From a memory perspective, this reduction alone has compressed the overall memory required to represent the state by 50\%.
````

Identifying redundancies in these kinds of representations heavily depends on the use of what is referred to as a _normalization scheme_ for the decision diagram nodes {cite:p}`niemannQMDDsEfficientQuantum2016`.
Such a normalization scheme makes sure two decision diagram nodes that represent the same functionality do indeed have the same numerical structure.
In computer science, this property is called _canonicity_.

The most widely used and practically relevant normalization scheme is to normalize the outgoing edges of a node by dividing both weights by the norm of the vector containing both edge weights and adjusting the incoming edges accordingly {cite:p}`hillmichJustRealThing2020`.
This normalizes the sum of the squared magnitudes of the outgoing edge weights to $1$ and is consistent with quantum semantics, where basis states $\ket{0}$ and $\ket{1}$ are observed after measurement with probabilities that are squared magnitudes of the respective weights.
Normalization is recursively applied in a bottom-up fashion to ensure that every possible redundancy is caught.

````{admonition} Example _(Normalization of Decision Diagrams)_
:class: tip

Considering the decision diagram from the previous example, this results in the following _normalized_ and _reduced_ decision diagram:

```{image} _static/dd-figure-06.svg
:align: center
:width: 35%
```

The first two levels ($q_2$ and $q_1$) of the above diagram naturally encode that the respective qubits have a $50/50$ chance to be in $\ket{0}$ and $\ket{1}$ (since $\vert1/\sqrt{2}\vert^2 = 0.5$).
Meanwhile, the bottom level ($q_0$) encodes that the probability of $q_0$ depends on the state of $q_1$.
If $q_1$ is in the $\ket{0}$ state (following the left successor), then $q_0$ has probability $0.5$ for both $\ket{0}$ and $\ket{1}$.
If $q_1$ is in the $\ket{1}$ state (following the right successor), it is guaranteed that the remaining qubit is in the $\ket{0}$ state.
````

Overall, statevectors are represented as decision diagrams conceptionally equivalent to halving the vector in a recursive fashion until it is fully decomposed.
The key idea is to exploit the redundancies in the resulting diagrams to create a more compact representation.
Some interesting properties that are worth pointing out:

- Decision diagrams can be initialized in their compact form (as, for example, shown in the last example above). There is no need to create the maximally large decision diagram (as shown, for example, in [](#3qbdd)) at any point in a calculation.
- Determining a particular amplitude of the represented state corresponds to multiplying the edge weights along a single-path traversal from the top edge of the decision diagram (called its _root_) to a terminal node.
- The efficiency of decision diagrams is commonly measured by their _size_, that is, the number of nodes in the decision diagram---the smaller the number of nodes, the higher the compaction achieved by the data structure. Note that the terminal (node) is typically not counted towards the size of a decision diagram.
- Any product state naturally has a decision diagram consisting of a single node per site. However, a compact DD does not correlate with the state being trivial. Even entangled states such as the _GHZ state_ or the _W state_ have decision diagrams whose size (that is, the number of nodes) is linear in the number of qubits.
- DDs are not a "silver bullet." The worst-case size of decision diagrams, corresponding to states without redundancy, is still exponential in the number of qubits. More specifically, a maximally large decision diagram has $1+2^1+2^2+\dots+2^{n-1} = 2^n-1$ nodes.
- To reduce visual clutter in illustrations of decision diagrams, edge weights are commonly not explicitly annotated, but their magnitude and phase are reflected in the thickness and the color of the respective edge.
  In addition, to make the correspondence of the individual levels in a decision diagram to a system's qubits more explicit, the nodes are frequently annotated with the qubit's index as an identifier.
  See {cite:p}`willeVisualizingDecisionDiagrams2021` for further details on common techniques for visualization of decision diagrams.

### Representation of Quantum Operations

Quantum operations are fundamentally described by complex-valued matrices.
Matrix decision diagrams are a natural extension to vector decision diagrams by an additional dimension.
To this end, consider the base case of a $2\times 2$ matrix $U$, that is,

```{math}
U &= \begin{bmatrix}
U_{00} & U_{01} \\ U_{10} & U_{11}
\end{bmatrix} = U_{00} \ket{0}\!\bra{0} + U_{01} \ket{1}\!\bra{0} + U_{10} \ket{0}\!\bra{1} + U_{11} \ket{1}\!\bra{1} .
```

Then, the decision diagram representing this matrix has the structure

```{image} _static/dd-figure-07.svg
:align: center
:width: 35%
```

which again resembles the general structure of the matrix.
Note that $U_{ij}$ can be interpreted as the transformation of $\ket{j}$ to $\ket{i}$.

````{admonition} Example _(Single-Qubit Operations)_
:class: tip
The following shows decision diagram representations for selected \mbox{single-qubit} operations:

```{image} _static/dd-figure-08.svg
:align: center
:width: 65%
```

The last equivalence demonstrates how a common factor between the edge weights can be pulled out and attached to the incoming (root) edge.
````

The generalization to larger matrices works analogously to the vector case.
To construct the decision diagram representing a matrix, the matrix is recursively divided into quarters, and the four elements correspond to the four successors of the node to represent that split.
As for vector decision diagrams, a normalization scheme is applied to ensure that the resulting data structure is canonical and redundancy can be exploited.
The conventional approach is to normalize all edge weights by the weight with the highest magnitude, selecting the leftmost one if multiple weights have the same magnitude.
It is important to note that this ensures that all complex numbers within the decision diagram have a magnitude of at most $1$, which is used for optimization purposes.

````{admonition} Example _(Matrix Decision Diagrams)_
:class: tip
Consider the maximally-entangling two-qubit $R_{xx}$ rotation represented by the matrix
```{math}
R_{xx} \Bigl(\theta = \frac{\pi}{2} \Bigl) = \frac{1}{\sqrt{2}}\begin{bmatrix}
1 & 0 & 0 & -i \\
0 & 1 & -i & 0 \\
0 & -i & 1 & 0 \\
-i & 0 & 0 & 1
\end{bmatrix} .
```
This matrix is equivalent to blocks of $2 \times 2$ matrices corresponding to the identity $I$ and the Pauli-$X$ matrix, i.e.,
```{math}
:label: rxxmat
R_{xx} \Bigl(\theta = \frac{\pi}{2} \Bigl) = \frac{1}{\sqrt{2}}\begin{bmatrix}
I & -iX \\
-iX & I
\end{bmatrix}.
```
The corresponding (already reduced) decision diagram has the following structure:

```{image} _static/dd-figure-09.svg
:align: center
:width: 40%
```

Notice how the decision diagram naturally resembles the structure of the matrix.
The nodes at the bottom represent the identity and the $X$ matrix while the node at the top encodes the redundancy of the upper left quadrant and the bottom right quadrant, as well as the upper right and lower left quadrant in [](#rxxmat).
Similarly to the vector example above, exploiting redundancy has halved the overall memory requirement.
````

Again, some interesting properties to point out:

- Just as in the vector case, it is always possible to work with the reduced form of matrix decision diagrams right away, that is, without ever constructing the exponentially-sized, maximally-large diagram.
- A maximally-large matrix decision diagram for $n$ qubits has $\sum_{i=1}^n 4^{i-1} = \frac{(4^n -1)}{3}$ nodes.
- Decision diagrams are not limited to local interactions. Even long-range interactions between arbitrary qubits typically produce compact representations as decision diagrams. For example, any two-qubit interaction between arbitrary qubits can be represented as a decision diagram with at most $1+4(n-1)$ nodes---an exponential reduction.
- Decision diagrams are not limited to two-qubit interactions either. For example, controlled quantum gates with arbitrarily many controls (such as the multi-controlled Toffoli gate) give rise to decision diagrams with a linear number of nodes.

### Fundamental Operations on Decision Diagrams

Merely defining means for compactly representing any kind of state or operation does not yet allow one to perform efficient computations.
It is crucial to also define efficient means of working with or manipulating the resulting representations.
In the following, it is demonstrated how the most fundamental operations can be carried out within the decision-diagram formalism and how they scale.
The focus is mainly on how operations are realized on vectors, since the concepts extend from vectors to matrices in a straightforward fashion.

The main concept throughout all of these schemes is to recursively break the respective operations down into subcomputations.
This decomposition then naturally matches the recursive decomposition of decision diagrams.
As such, operations generally scale with the number of nodes in the involved decision diagrams.

#### Kronecker Product

The Kronecker product is necessary to create product states and to chain together local operations.
For vectors, it can be expressed as

<!-- prettier-ignore -->
```{math}
:label: kronecker
\ket{\Psi} \otimes \ket{\Phi} = \begin{bmatrix}
\Psi_{0} \ket{\Phi} \\
\Psi_{1} \ket{\Phi}
\end{bmatrix}
= \begin{bmatrix}
\Psi_{0} \begin{bmatrix} \Phi_0 \\ \Phi_1 \end{bmatrix} \\
\Psi_{1} \begin{bmatrix} \Phi_0 \\ \Phi_1 \end{bmatrix}
\end{bmatrix}.
```

In the decision-diagram formalism, this is one of the simplest operations to perform and is done by simply replacing the terminal nodes of the first decision diagram with the root node of the second decision diagram.
In case of the above example, this has the following form:

```{image} _static/dd-figure-10.svg
:align: center
:width: 60%
```

As such, its complexity is linear in the number of nodes of the first decision diagram.

#### Addition

Standard vector addition can be recursively broken down according to

<!-- prettier-ignore -->
```{math}
:label: addition
\ket{\Psi} + \ket{\Phi} = \begin{bmatrix} \Psi_0 \\ \Psi_1 \end{bmatrix} + \begin{bmatrix} \Phi_0 \\ \Phi_1 \end{bmatrix} = w \begin{bmatrix} \alpha_0  \\ \alpha_1 \end{bmatrix} + w' \begin{bmatrix} \alpha'_0 \\ \alpha'_1 \end{bmatrix} = \begin{bmatrix} w \alpha_0 + w' \alpha'_0 \\ w \alpha_1 + w' \alpha'_1 \end{bmatrix},
```

where $w$ and $w'$ are common factors of the terms in $\ket{\Psi}$ and $\ket{\Phi}$, respectively.

In the decision-diagram formalism, this corresponds to a simultaneous traversal of both decision diagrams from their roots to the terminal (multiplying edge weights along the way until the individual amplitudes are reached) and back again (accumulating the results of the recursive computations).
More precisely,

```{image} _static/dd-figure-11.svg
:align: center
:width: 70%
```

where the dashed nodes represent the respective successor decision diagrams.
Overall, this results in a complexity that is linear in the size of the larger decision diagram.

#### Matrix-Vector Multiplication

Matrix-vector multiplication can be handled in a very similar fashion as addition.
Standard matrix-vector multiplication can be expressed as

<!-- prettier-ignore -->
```{math}
:label: multiplication
U\ket{\Psi} = \begin{bmatrix} U_{00} & U_{01} \\
U_{10} & U_{11} \end{bmatrix} \begin{bmatrix} \Psi_0 \\ \Psi_1 \end{bmatrix}
 = w \begin{bmatrix} u_{00} & u_{01} \\
u_{10} & u_{11} \end{bmatrix} w' \begin{bmatrix} \alpha_0 \\ \alpha_1 \end{bmatrix} = ww' \begin{bmatrix} u_{00} \cdot \alpha_0 + u_{10} \cdot \alpha_1 \\
u_{01} \cdot \alpha_0 + u_{11} \cdot \alpha_1 \end{bmatrix} .
```

This implies that a multiplication boils down to four smaller multiplications and two additions.
In the decision-diagram formalism, this has the form

```{image} _static/dd-figure-12.svg
:align: center
:width: 90%
```

where the dashed nodes again represent the respective successor decision diagrams.
Overall, this results in a complexity that scales with the product of the size of both decision diagrams.

#### Inner Product

Computing the inner product of two vectors can be recursively broken down according to

<!-- prettier-ignore -->
```{math}
:label: innerproduct
\langle\Psi \vert \Phi\rangle = \begin{bmatrix} \Psi^*_0 & \Psi^*_1 \end{bmatrix} \begin{bmatrix} \Phi_0 \\ \Phi_1 \end{bmatrix}
= w^* \begin{bmatrix} \alpha^*_0 & \alpha^*_1 \end{bmatrix} w' \begin{bmatrix} \alpha'_0 \\ \alpha'_1 \end{bmatrix} = w^*w' (\alpha^*_0 \alpha'_0 + \alpha^*_1 \alpha'_0)
```

This implies that the inner product boils down to two smaller inner product computations and adding the results.
As with the matrix-vector multiplication, this is done recursively for each level of the decision diagram.
In the decision-diagram formalism, this has the following form

```{image} _static/dd-figure-13.svg
:align: center
:width: 70%
```

Overall, this results in a complexity that, just as in addition, scales linearly with the size of the larger decision diagram.

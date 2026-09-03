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

MQT Core provides a fully-fledged, high-performance decision diagram package for
quantum computing. The resulting library allows for the efficient representation
and manipulation of quantum states and operations. If you are not yet familiar
with decision diagrams as a data structure, you might want to start with the
[introduction to quantum decision diagrams](#how-do-quantum-decision-diagrams-work)
below.

Throughout the MQT, this library enables many classical simulation, synthesis,
or verification techniques. While primarily developed in C++, the corresponding
functionality is also exposed to Python users in the form of the
{py:mod}`mqt.core.dd` module. The following section provides an overview on how
to work with decision diagrams in MQT Core from Python.

## Quickstart

The MQT Compiler Collection uses the DD package to simulate
{py:class}`~mqt.core.mlir.QCOProgram` objects. The simulator supports
mid-circuit measurements, resets, and classically controlled operations. This
example compiles and samples a Bell-state program:

```{code-cell} ipython3
from mqt.core.mlir import sample

bell_qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] result;
h q[0];
cx q[0], q[1];
result = measure q;
"""

counts = sample(bell_qasm, shots=1024, seed=1)
print(counts)
```

The {py:func}`~mqt.core.mlir.sample`, {py:func}`~mqt.core.mlir.simulate`, and
{py:func}`~mqt.core.mlir.build_functionality` functions accept source text,
paths, Qiskit circuits, and typed compiler programs. They lower each input
directly to QCO. The corresponding {py:class}`~mqt.core.mlir.QCOProgram` methods
remain useful when the program is already compiled. Without an explicit DD
package, `simulate` and `build_functionality` manage one internally and return
NumPy arrays. A compatible `complex128` input array is read directly, and the
result is materialized without an additional copy. Compiler frontends allocate
their declared qubits, so dense simulation of source, QC, OpenQASM, or Qiskit
inputs normally starts with the scalar state `[1]`.

Pass an explicit {py:class}`~mqt.core.dd.DDPackage` to return DDs instead. These
overloads and the `QCOProgram` methods avoid constructing exponentially large
dense arrays unless the result is explicitly converted with
{py:meth}`~mqt.core.dd.VectorDD.get_vector` or
{py:meth}`~mqt.core.dd.MatrixDD.get_matrix`.

```{code-cell} ipython3
import numpy as np
from mqt.core.dd import DDPackage
from mqt.core.mlir import QCOProgram, build_functionality, simulate

unitary_program = QCOProgram.from_mlir_str("""
module {
  func.func @main() attributes {mqt.entry_point} {
    %q0 = qco.static 0 : !qco.qubit
    %q1 = qco.static 1 : !qco.qubit
    %q0_h = qco.h %q0 : !qco.qubit -> !qco.qubit
    %q0_out, %q1_out = qco.ctrl(%q0_h) targets(%target = %q1) {
      %target_out = qco.x %target : !qco.qubit -> !qco.qubit
      qco.yield %target_out : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    qco.sink %q0_out : !qco.qubit
    qco.sink %q1_out : !qco.qubit
    return
  }
}
""")

zero_state = np.zeros(4, dtype=np.complex128)
zero_state[0] = 1
vec = simulate(unitary_program, zero_state)
unitary = build_functionality(unitary_program)

dd = DDPackage(2)
zero_state_dd = dd.zero_state(2)
out_state_dd = unitary_program.simulate(zero_state_dd, dd)
vec = np.array(out_state_dd.get_vector(), copy=False)
with np.printoptions(precision=3, suppress=True):
  print(vec)

functionality_dd = unitary_program.build_functionality(dd)
unitary = np.array(functionality_dd.get_matrix(2), copy=False)
with np.printoptions(precision=3, suppress=True):
  print(unitary)
```

If the [Graphviz](https://www.graphviz.org/) library is installed, the
`graphviz` Python package can be used to visualize resulting decision diagram
with the {py:meth}`~mqt.core.dd.VectorDD.to_dot` method. Use
{py:meth}`~mqt.core.dd.VectorDD.to_svg` to generate an SVG file directly.

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

The DD package provides list of additional functionality when it comes to
working with decision diagrams. Check out the full API documentation of the
{py:class}`~mqt.core.dd.DDPackage` class for more details.

## How do Quantum Decision Diagrams Work?

Decision diagrams were introduced in the 1980s as a data structure for the
efficient representation and manipulation of Boolean functions
{cite:p}`bryantGraphbasedAlgorithmsBoolean1986`. This led to the emergence of a
wide variety of decision diagrams, including BDDs, FBDDs, KFDDs, MTBDDs, and
ZDDs (see, for example,
{cite:p}`bryantSymbolicBooleanManipulation1992,wegenerBranchingProgramsBinary2000,gergovEfficientBooleanManipulation1994,drechslerEfficientRepresentationManipulation1994,baharAlgebraicDecisionDiagrams1993,minatoZerosuppressedBDDsSet1993`),
which made them a crucial tool in the development of modern circuits and
systems. Because of their previous success, decision diagrams have been proposed
for application in the realm of quantum computing
{cite:p}`willeDecisionDiagramsQuantum2023,willeToolsQuantumComputing2022,millerQMDDDecisionDiagram2006,niemannQMDDsEfficientQuantum2016,zulehnerHowEfficientlyHandle2019,hongTensorNetworkBased2020,vinkhuijzenLIMDDDecisionDiagram2021`.
Particularly for design tasks like _simulation_
{cite:p}`viamontesImprovingGatelevelSimulation2003,zulehnerAdvancedSimulationQuantum2019,hillmichJustRealThing2020,burgholzerHybridSchrodingerFeynmanSimulation2021,vinkhuijzenLIMDDDecisionDiagram2021,hillmichApproximatingDecisionDiagrams2022,burgholzerSimulationPathsQuantum2022,grurlNoiseawareQuantumCircuit2023,matoMixeddimensionalQuantumCircuit2023,sanderHamiltonianSimulationDecision2023`,
_synthesis_
{cite:p}`niemannEfficientSynthesisQuantum2014,abdollahiAnalysisSynthesisQuantum2006,soekenSynthesisReversibleCircuits2012,zulehnerOnepassDesignReversible2018,adarshSyReCSynthesizerMQT2022,matoMixeddimensionalQuditState2024`,
and _verification_
{cite:p}`burgholzerAdvancedEquivalenceChecking2021,burgholzerRandomStimuliGeneration2021,burgholzerVerifyingResultsIBM2020,wangXQDDbasedVerificationMethod2008,smithQuantumLogicSynthesis2019,hongEquivalenceCheckingDynamic2021`
of quantum circuits, they recently attracted great attention.

The following sections provide a comprehensive guide for quantum computing with
decision diagrams, including the representation of quantum states and operations
and the fundamental operations on decision diagrams.

### Representation of Quantum States

First, we review how quantum states are represented using decision diagrams. To
this end, we consider the simple case of a single-qubit system. The state
$\ket{\Psi}$ of such a system is described by two complex-valued, normalized
amplitudes $\alpha_0$ and $\alpha_1$, that is,

```{math}
:label: ssstate
\ket{\Psi} = \alpha_0 \ket{0} + \alpha_1 \ket{1},
```

which is commonly represented as a statevector

```{math}
\ket{\Psi}\equiv \begin{bmatrix} \alpha_0 & \alpha_1	\end{bmatrix}^\top.
```

A rather simple observation and consequence of {eq}`ssstate` is that this vector
can be equally split into a contribution of the $\ket{0}$ state ($\alpha_0$) and
a contribution of the $\ket{1}$ state ($\alpha_1$), that is,

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

This decomposition is the core of the decision-diagram formalism. The decision
diagram representing $\ket{\Psi}$ has the structure

```{image} _static/dd-figure-01.svg
:width: 15%
:align: center
```

It consists of a single _node_ with one _incoming edge_ that represents the
entry point in the decision diagram, as well as two _successors_ that represent
the split shown in {eq}`splitting` and end in a _terminal_ node (the black box).
The state's amplitudes are annotated at the respective edges. Edges without
annotations correspond to an edge weight of 1.

````{admonition} Example _(Single-Qubit States)_
:class: tip

Consider the computational basis states $\ket{0}$ and $\ket{1}$. Then, the
corresponding decision diagrams have the structures

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

Building off the intuition of a single-qubit state, we can move to larger
systems.

````{admonition} Example _(Multi-Qubit States)_
:class: tip

Consider the following statevector of a three-qubit system:
```{math}
\ket{\Psi} = \begin{bmatrix} \frac{1}{2\sqrt{2}} & \frac{1}{2\sqrt{2}} &  \frac{1}{2} & 0 & \frac{1}{2\sqrt{2}} & \frac{1}{2\sqrt{2}} & \frac{1}{2} & 0\end{bmatrix}^T
```
Then, $\ket{\Psi}$ can be recursively split into equally-sized parts similar to {eq}`splitting`, i.e.,
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

At this point, this has been just a one-to-one translation between the
statevector and a fancy graphical representation. The unique core feature of
decision diagrams is that their graph structure allows redundant parts to be
merged in the representation instead of being represented repeatedly.

````{admonition} Example _(Redundancy in Decision Diagrams)_
:class: tip

Observe how, as in the previous example, the left and right successors of the
top-level node (labeled $q_2$) lead to exactly the same structure (highlighted
by dashed rectangles in {eq}`3qbdd`). As a result, the whole sub-diagram does
not need to be represented twice, i.e.,

```{image} _static/dd-figure-05.svg
:align: center
:width: 40%
```

From a memory perspective, this reduction alone has compressed the overall memory required to represent the state by 50\%.
````

Identifying redundancies in these kinds of representations heavily depends on
the use of what is referred to as a _normalization scheme_ for the decision
diagram nodes {cite:p}`niemannQMDDsEfficientQuantum2016`. Such a normalization
scheme makes sure two decision diagram nodes that represent the same
functionality do indeed have the same numerical structure. In computer science,
this property is called _canonicity_.

The most widely used and practically relevant normalization scheme is to
normalize the outgoing edges of a node by dividing both weights by the norm of
the vector containing both edge weights and adjusting the incoming edges
accordingly {cite:p}`hillmichJustRealThing2020`. This normalizes the sum of the
squared magnitudes of the outgoing edge weights to $1$ and is consistent with
quantum semantics, where basis states $\ket{0}$ and $\ket{1}$ are observed after
measurement with probabilities that are squared magnitudes of the respective
weights. Normalization is recursively applied in a bottom-up fashion to ensure
that every possible redundancy is caught.

````{admonition} Example _(Normalization of Decision Diagrams)_
:class: tip

Considering the decision diagram from the previous example, this results in the
following _normalized_ and _reduced_ decision diagram:

```{image} _static/dd-figure-06.svg
:align: center
:width: 35%
```

The first two levels ($q_2$ and $q_1$) of the above diagram naturally encode that the respective qubits have a $50/50$ chance to be in $\ket{0}$ and $\ket{1}$ (since $\vert1/\sqrt{2}\vert^2 = 0.5$).
Meanwhile, the bottom level ($q_0$) encodes that the probability of $q_0$ depends on the state of $q_1$.
If $q_1$ is in the $\ket{0}$ state (following the left successor), then $q_0$ has probability $0.5$ for both $\ket{0}$ and $\ket{1}$.
If $q_1$ is in the $\ket{1}$ state (following the right successor), it is guaranteed that the remaining qubit is in the $\ket{0}$ state.
````

Overall, statevectors are represented as decision diagrams conceptually
equivalent to halving the vector in a recursive fashion until it is fully
decomposed. The key idea is to exploit the redundancies in the resulting
diagrams to create a more compact representation. Some interesting properties
that are worth pointing out:

- Decision diagrams can be initialized in their compact form (as, for example,
  shown in the last example above). There is no need to create the maximally
  large decision diagram (as shown, for example, in {eq}`3qbdd`) at any point in
  a calculation.
- Determining a particular amplitude of the represented state corresponds to
  multiplying the edge weights along a single-path traversal from the top edge
  of the decision diagram (called its _root_) to a terminal node.
- The efficiency of decision diagrams is commonly measured by their _size_, that
  is, the number of nodes in the decision diagram---the smaller the number of
  nodes, the higher the compaction achieved by the data structure. Note that the
  terminal (node) is typically not counted towards the size of a decision
  diagram.
- Any product state naturally has a decision diagram consisting of a single node
  per site. However, a compact DD does not correlate with the state being
  trivial. Even entangled states such as the _GHZ state_ or the _W state_ have
  decision diagrams whose size (that is, the number of nodes) is linear in the
  number of qubits.
- DDs are not a "silver bullet." The worst-case size of decision diagrams,
  corresponding to states without redundancy, is still exponential in the number
  of qubits. More specifically, a maximally large decision diagram has
  $1+2^1+2^2+\dots+2^{n-1} = 2^n-1$ nodes.
- To reduce visual clutter in illustrations of decision diagrams, edge weights
  are commonly not explicitly annotated, but their magnitude and phase are
  reflected in the thickness and the color of the respective edge. In addition,
  to make the correspondence of the individual levels in a decision diagram to a
  system's qubits more explicit, the nodes are frequently annotated with the
  qubit's index as an identifier. See
  {cite:p}`willeVisualizingDecisionDiagrams2021` for further details on common
  techniques for visualization of decision diagrams.

### Representation of Quantum Operations

Quantum operations are fundamentally described by complex-valued matrices.
Matrix decision diagrams are a natural extension to vector decision diagrams by
an additional dimension. To this end, consider the base case of a $2\times 2$
matrix $U$, that is,

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

which again resembles the general structure of the matrix. Note that $U_{ij}$
can be interpreted as the transformation of $\ket{j}$ to $\ket{i}$.

````{admonition} Example _(Single-Qubit Operations)_
:class: tip

The following shows decision diagram representations for selected
\mbox{single-qubit} operations:

```{image} _static/dd-figure-08.svg
:align: center
:width: 65%
```

The last equivalence demonstrates how a common factor between the edge weights can be pulled out and attached to the incoming (root) edge.
````

The generalization to larger matrices works analogously to the vector case. To
construct the decision diagram representing a matrix, the matrix is recursively
divided into quarters, and the four elements correspond to the four successors
of the node to represent that split. As for vector decision diagrams, a
normalization scheme is applied to ensure that the resulting data structure is
canonical and redundancy can be exploited. The conventional approach is to
normalize all edge weights by the weight with the highest magnitude, selecting
the leftmost one if multiple weights have the same magnitude. It is important to
note that this ensures that all complex numbers within the decision diagram have
a magnitude of at most $1$, which is used for optimization purposes.

````{admonition} Example _(Matrix Decision Diagrams)_
:class: tip

Consider the maximally-entangling two-qubit $R_{xx}$ rotation represented by the
matrix

```{math}
R_{xx} \Bigl(\theta = \frac{\pi}{2} \Bigl) = \frac{1}{\sqrt{2}}\begin{bmatrix}
1 & 0 & 0 & -i \\
0 & 1 & -i & 0 \\
0 & -i & 1 & 0 \\
-i & 0 & 0 & 1
\end{bmatrix}.
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

- Just as in the vector case, it is always possible to work with the reduced
  form of matrix decision diagrams right away, that is, without ever
  constructing the exponentially-sized, maximally-large diagram.
- A maximally-large matrix decision diagram for $n$ qubits has
  $\sum_{i=1}^n 4^{i-1} = \frac{(4^n -1)}{3}$ nodes.
- Decision diagrams are not limited to local interactions. Even long-range
  interactions between arbitrary qubits typically produce compact
  representations as decision diagrams. For example, any two-qubit interaction
  between arbitrary qubits can be represented as a decision diagram with at most
  $1+4(n-1)$ nodes---an exponential reduction.
- Decision diagrams are not limited to two-qubit interactions either. For
  example, controlled quantum gates with arbitrarily many controls (such as the
  multi-controlled Toffoli gate) give rise to decision diagrams with a linear
  number of nodes.

### Fundamental Operations on Decision Diagrams

Merely defining means for compactly representing any kind of state or operation
does not yet allow one to perform efficient computations. It is crucial to also
define efficient means of working with or manipulating the resulting
representations. In the following, it is demonstrated how the most fundamental
operations can be carried out within the decision-diagram formalism and how they
scale. The focus is mainly on how operations are realized on vectors, since the
concepts extend from vectors to matrices in a straightforward fashion.

The main concept throughout all of these schemes is to recursively break the
respective operations down into subcomputations. This decomposition then
naturally matches the recursive decomposition of decision diagrams. As such,
operations generally scale with the number of nodes in the involved decision
diagrams.

#### Kronecker Product

The Kronecker product is necessary to create product states and to chain
together local operations. For vectors, it can be expressed as

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

In the decision-diagram formalism, this is one of the simplest operations to
perform and is done by simply replacing the terminal nodes of the first decision
diagram with the root node of the second decision diagram. In case of the above
example, this has the following form:

```{image} _static/dd-figure-10.svg
:align: center
:width: 60%
```

As such, its complexity is linear in the number of nodes of the first decision
diagram.

#### Addition

Standard vector addition can be recursively broken down according to

<!-- prettier-ignore -->
```{math}
:label: addition
\ket{\Psi} + \ket{\Phi} = \begin{bmatrix} \Psi_0 \\ \Psi_1 \end{bmatrix} + \begin{bmatrix} \Phi_0 \\ \Phi_1 \end{bmatrix} = w \begin{bmatrix} \alpha_0  \\ \alpha_1 \end{bmatrix} + w' \begin{bmatrix} \alpha'_0 \\ \alpha'_1 \end{bmatrix} = \begin{bmatrix} w \alpha_0 + w' \alpha'_0 \\ w \alpha_1 + w' \alpha'_1 \end{bmatrix},
```

where $w$ and $w'$ are common factors of the terms in $\ket{\Psi}$ and
$\ket{\Phi}$, respectively.

In the decision-diagram formalism, this corresponds to a simultaneous traversal
of both decision diagrams from their roots to the terminal (multiplying edge
weights along the way until the individual amplitudes are reached) and back
again (accumulating the results of the recursive computations). More precisely,

```{image} _static/dd-figure-11.svg
:align: center
:width: 70%
```

where the dashed nodes represent the respective successor decision diagrams.
Overall, this results in a complexity that is linear in the size of the larger
decision diagram.

#### Matrix-Vector Multiplication

Matrix-vector multiplication can be handled in a very similar fashion as
addition. Standard matrix-vector multiplication can be expressed as

```{math}
:label: multiplication
U\ket{\Psi} = \begin{bmatrix} U_{00} & U_{01} \\
U_{10} & U_{11} \end{bmatrix} \begin{bmatrix} \Psi_0 \\ \Psi_1 \end{bmatrix}
 = w \begin{bmatrix} u_{00} & u_{01} \\
u_{10} & u_{11} \end{bmatrix} w' \begin{bmatrix} \alpha_0 \\ \alpha_1 \end{bmatrix} = ww' \begin{bmatrix} u_{00} \cdot \alpha_0 + u_{10} \cdot \alpha_1 \\
u_{01} \cdot \alpha_0 + u_{11} \cdot \alpha_1 \end{bmatrix}.
```

This implies that a multiplication boils down to four smaller multiplications
and two additions. In the decision-diagram formalism, this has the form

```{image} _static/dd-figure-12.svg
:align: center
:width: 90%
```

where the dashed nodes again represent the respective successor decision
diagrams. Overall, this results in a complexity that scales with the product of
the size of both decision diagrams.

#### Inner Product

Computing the inner product of two vectors can be recursively broken down
according to

<!-- prettier-ignore -->
```{math}
:label: innerproduct
\langle\Psi \vert \Phi\rangle = \begin{bmatrix} \Psi^*_0 & \Psi^*_1 \end{bmatrix} \begin{bmatrix} \Phi_0 \\ \Phi_1 \end{bmatrix}
= w^* \begin{bmatrix} \alpha^*_0 & \alpha^*_1 \end{bmatrix} w' \begin{bmatrix} \alpha'_0 \\ \alpha'_1 \end{bmatrix} = w^*w' (\alpha^*_0 \alpha'_0 + \alpha^*_1 \alpha'_0)
```

This implies that the inner product boils down to two smaller inner product
computations and adding the results. As with the matrix-vector multiplication,
this is done recursively for each level of the decision diagram. In the
decision-diagram formalism, this has the following form

```{image} _static/dd-figure-13.svg
:align: center
:width: 70%
```

Overall, this results in a complexity that, just as in addition, scales linearly
with the size of the larger decision diagram.

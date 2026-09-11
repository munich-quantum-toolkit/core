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

MQT Core represents and manipulates quantum states and operations with decision
diagrams (DDs). The C++ library and {py:mod}`mqt.core.dd` Python module support
simulation, synthesis, and verification. Start with the quickstart for Python
usage or the introduction below for the data structure and its limits.

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
provide the DD-native interface for reusable compiled programs, custom initial
states, and dynamic simulation. The top-level `simulate` function starts a
closed program in the all-zero state. It and `build_functionality` manage the DD
package internally and materialize their results directly into NumPy arrays.

The `QCOProgram` methods avoid constructing exponentially large dense arrays
unless the result is explicitly converted with
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

vec = simulate(unitary_program)
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

If [Graphviz](https://www.graphviz.org/) is installed, use
{py:meth}`~mqt.core.dd.VectorDD.to_svg` to export a decision diagram as SVG.
IPython can display the resulting file in a notebook. DOT exports use unique
node IDs assigned in traversal order, so ordinary exports do not depend on
memory addresses. The `memory=True` option includes addresses as debugging
information.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 20%
    align: center
    alt: Bell-state DD with shared zero and one branches.
---
from IPython.display import SVG

out_state_dd.to_svg("bell_state.svg")
SVG(filename="bell_state.svg")
```

See {py:class}`~mqt.core.dd.DDPackage` for the full API.

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

The following sections explain how decision diagrams represent quantum states
and operations, and how computations act on those representations.

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

The vector in {eq}`ssstate` splits into the contributions of the $\ket{0}$ state
($\alpha_0$) and the $\ket{1}$ state ($\alpha_1$):

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
:alt: One qubit with outgoing edges weighted by its zero and one amplitudes.
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
:alt: Zero state: only the zero successor has nonzero weight.
:align: center
:width: 8%
```
```{math}
\ket{0}\equiv\begin{bmatrix}1 & 0\end{bmatrix}^\top
```

and

```{image} _static/dd-figure-03.svg
:alt: One state: only the one successor has nonzero weight.
:align: center
:width: 8%
```
```{math}
\ket{1}\equiv\begin{bmatrix}0 & 1\end{bmatrix}^\top
```

In each of the cases, one of the successors ends in the terminal node, while the other ends in a _zero stub_ (indicated by a black dot)---uncannily resembling the corresponding vector descriptions.
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
:alt: Unreduced three-qubit state with repeated branches.
:name: dd-three-qubits
:align: center
:width: 65%
```


Each level of the decision diagram consists of decision nodes with corresponding left and right successor edges.
These successors represent the path that leads to an amplitude where the local quantum system (corresponding to the _level_ of the node, annotated here with the labels) is in the $\ket{0}$ (left successor) or the $\ket{1}$ state (right successor).
````

The diagrams above represent each part of the statevector separately. Merging
redundant subgraphs makes the representation compact.

````{admonition} Example _(Redundancy in Decision Diagrams)_
:class: tip

Observe how, as in the previous example, the left and right successors of the
top-level node (labeled $q_2$) lead to exactly the same structure (highlighted
by dashed rectangles in {ref}`the unreduced diagram <dd-three-qubits>`). As a
result, the whole sub-diagram does not need to be represented twice, i.e.,

```{image} _static/dd-figure-05.svg
:alt: Equal branches of the three-qubit state merged into one subdiagram.
:align: center
:width: 40%
```

From a memory perspective, this reduction alone has compressed the overall memory required to represent the state by 50%.
````

Identifying redundancies in these kinds of representations heavily depends on
the use of what is referred to as a _normalization scheme_ for the decision
diagram nodes {cite:p}`niemannQMDDsEfficientQuantum2016`. Such a normalization
scheme makes sure two decision diagram nodes that represent the same
functionality do indeed have the same numerical structure. In computer science,
this property is called _canonicity_.

The most widely used and practically relevant normalization scheme is to
normalize the outgoing edges of a node by dividing both weights by the norm of
the vector containing both edge weights and extracting a common phase into the
incoming edge {cite:p}`hillmichJustRealThing2020`. This normalizes the sum of
the squared magnitudes of the outgoing edge weights to $1$ and is consistent
with quantum semantics, where basis states $\ket{0}$ and $\ket{1}$ are observed
after measurement with probabilities that are squared magnitudes of the
respective weights. MQT Core selects a maximum-magnitude edge (preferring the
left edge within numerical tolerance) and makes its normalized weight real and
nonnegative. The incoming edge retains its complex phase. Normalization proceeds
bottom-up; complex-number comparisons use the package tolerance.

````{admonition} Example _(Normalization of Decision Diagrams)_
:class: tip

Considering the decision diagram from the previous example, this results in the
following _normalized_ and _reduced_ decision diagram:

```{image} _static/dd-figure-06.svg
:alt: Normalized state with shared subdiagrams and conditional amplitudes.
:align: center
:width: 35%
```

The first two levels ($q_2$ and $q_1$) of the above diagram naturally encode that the respective qubits have a $50/50$ chance to be in $\ket{0}$ and $\ket{1}$ (since $\vert1/\sqrt{2}\vert^2 = 0.5$).
Meanwhile, the bottom level ($q_0$) encodes that the probability of $q_0$ depends on the state of $q_1$.
If $q_1$ is in the $\ket{0}$ state (following the left successor), then $q_0$ has probability $0.5$ for both $\ket{0}$ and $\ket{1}$.
If $q_1$ is in the $\ket{1}$ state (following the right successor), it is guaranteed that the remaining qubit is in the $\ket{0}$ state.
````

A statevector DD recursively halves the vector and shares redundant subgraphs.
This representation has the following properties:

- Decision diagrams can be initialized in their compact form (as, for example,
  shown in the last example above). There is no need to create the maximally
  large decision diagram (as shown, for example, in
  {ref}`the unreduced diagram <dd-three-qubits>`) at any point in a calculation.
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
- The worst-case size, for states without redundancy, is exponential in the
  number of qubits. More specifically, a maximally large decision diagram has
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
\end{bmatrix} = U_{00} \ket{0}\!\bra{0} + U_{01} \ket{0}\!\bra{1} + U_{10} \ket{1}\!\bra{0} + U_{11} \ket{1}\!\bra{1} .
```

Then, the decision diagram representing this matrix has the structure

```{image} _static/dd-figure-07.svg
:alt: Matrix DD with four successors in row-major order.
:align: center
:width: 35%
```

which again resembles the general structure of the matrix. Note that $U_{ij}$
can be interpreted as the transformation of $\ket{j}$ to $\ket{i}$.

````{admonition} Example _(Single-Qubit Operations)_
:class: tip

The following shows decision diagram representations for selected single-qubit
operations:

```{image} _static/dd-figure-08.svg
:alt: DDs for single-qubit gates with common factors on the root edge.
:align: center
:width: 65%
```

The last equivalence demonstrates how a common factor between the edge weights can be pulled out and attached to the incoming (root) edge.
````

The generalization to larger matrices works analogously to the vector case. To
construct the decision diagram representing a matrix, the matrix is recursively
divided into quarters, and the four elements correspond to the four successors
of the node to represent that split. As for vector decision diagrams, a
normalization scheme makes the representation canonical so equivalent subgraphs
can be shared. Each node's outgoing edge weights are divided by the weight with
the highest magnitude, selecting the leftmost one in a tie. The normalized
outgoing weights have magnitude at most $1$; the extracted factor moves to the
incoming edge.

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
:alt: Rxx rotation sharing identity and Pauli-X submatrices.
:align: center
:width: 40%
```

Notice how the decision diagram naturally resembles the structure of the matrix.
The nodes at the bottom represent the identity and the $X$ matrix while the node at the top encodes the redundancy of the upper left quadrant and the bottom right quadrant, as well as the upper right and lower left quadrant in {eq}`rxxmat`.
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

DD operations recursively split computations along the graph structure and cache
shared subproblems. Their cost depends on the distinct subproblems visited and
the size of the result, as described below. The examples use vectors; the same
recursive approach extends to matrices.

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

The DD Kronecker product replaces the nonzero terminal edges of the first
diagram with the root edge of the second, multiplying their weights. For the
example above:

```{image} _static/dd-figure-10.svg
:alt: Kronecker product replacing terminal edges with the second DD.
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
:alt: Addition recursively combining corresponding weighted successors.
:align: center
:width: 70%
```

where the dashed nodes represent the respective successor decision diagrams. The
cost depends on the distinct weighted subproblems and the resulting DD. Even two
compact inputs can produce an exponentially large sum; input node counts alone
do not give a linear time bound.

#### Matrix-Vector Multiplication

Matrix-vector multiplication can be handled in a very similar fashion as
addition. Standard matrix-vector multiplication can be expressed as

```{math}
:label: multiplication
U\ket{\Psi} = \begin{bmatrix} U_{00} & U_{01} \\
U_{10} & U_{11} \end{bmatrix} \begin{bmatrix} \Psi_0 \\ \Psi_1 \end{bmatrix}
 = w \begin{bmatrix} u_{00} & u_{01} \\
u_{10} & u_{11} \end{bmatrix} w' \begin{bmatrix} \alpha_0 \\ \alpha_1 \end{bmatrix} = ww' \begin{bmatrix} u_{00} \cdot \alpha_0 + u_{01} \cdot \alpha_1 \\
u_{10} \cdot \alpha_0 + u_{11} \cdot \alpha_1 \end{bmatrix}.
```

This implies that a multiplication boils down to four smaller multiplications
and two additions. In the decision-diagram formalism, this has the form

```{image} _static/dd-figure-12.svg
:alt: Matrix-vector product combining each matrix row with the vector.
:align: center
:width: 90%
```

where the dashed nodes again represent the respective successor decision
diagrams. Runtime depends on the distinct weighted subproblems, intermediate
additions, and output size. Cache reuse can reduce repeated work; compact input
DDs alone do not guarantee a compact result.

#### Inner Product

Computing the inner product of two vectors can be recursively broken down
according to

<!-- prettier-ignore -->
```{math}
:label: innerproduct
\langle\Psi \vert \Phi\rangle = \begin{bmatrix} \Psi^*_0 & \Psi^*_1 \end{bmatrix} \begin{bmatrix} \Phi_0 \\ \Phi_1 \end{bmatrix}
= w^* \begin{bmatrix} \alpha^*_0 & \alpha^*_1 \end{bmatrix} w' \begin{bmatrix} \alpha'_0 \\ \alpha'_1 \end{bmatrix} = w^*w' (\alpha^*_0 \alpha'_0 + \alpha^*_1 \alpha'_1)
```

This implies that the inner product boils down to two smaller inner product
computations and adding the results. As with the matrix-vector multiplication,
this is done recursively for each level of the decision diagram. In the
decision-diagram formalism, this has the following form

```{image} _static/dd-figure-13.svg
:alt: Inner product conjugating the first vector and summing paired branches.
:align: center
:width: 70%
```

The recursion visits pairs of subdiagrams and reuses cached results. Its cost
depends on the pairs visited and cache reuse, rather than only the size of the
larger input.

### Check the algebra with complex amplitudes

The same operations can be compared directly with NumPy. A nonsymmetric matrix
makes row/column mistakes visible, while complex amplitudes exercise conjugation
and phase handling.

```{code-cell} ipython3
matrix = np.array([[0.6, -0.8], [0.8, 0.6]], dtype=complex)
state = np.array([1, 1j], dtype=complex) / np.sqrt(2)
other = np.array([0, 1], dtype=complex)
package = DDPackage(1)
state_dd = package.from_vector(state)
other_dd = package.from_vector(other)
matrix_dd = package.from_matrix(matrix)
product = package.matrix_vector_multiply(matrix_dd, state_dd)
np.testing.assert_allclose(product.get_vector(), matrix @ state)
np.testing.assert_allclose(package.inner_product(state_dd, other_dd), np.vdot(state, other))
np.testing.assert_allclose(package.vector_add(state_dd, other_dd).get_vector(), state + other)
with np.printoptions(precision=3, suppress=True):
    print("Matrix-vector product:", np.asarray(product.get_vector()))
    print("Inner product:", package.inner_product(state_dd, other_dd))
```

### Compact inputs can have a large sum

Both inputs below are product states with one nonterminal node per qubit. Their
sum needs many more nodes. The public `size()` includes the terminal; subtract
one when reporting nonterminal nodes.

```{code-cell} ipython3
print("Qubits | Left nodes | Right nodes | Sum nodes")
for n in (4, 6, 8):
    package = DDPackage(n)
    left = np.ones(1, dtype=complex)
    right = left.copy()
    for j in range(n):
        left = np.kron(left, [1, 1]) / np.sqrt(2)
        angle = 0.2 + 0.031 * j
        right = np.kron(right, [np.cos(angle), np.sin(angle)])
    left_dd = package.from_vector(left)
    right_dd = package.from_vector(right)
    result_dd = package.vector_add(left_dd, right_dd)
    np.testing.assert_allclose(result_dd.get_vector(), left + right, atol=1e-10)
    print(n, left_dd.size() - 1, right_dd.size() - 1, result_dd.size() - 1, sep=" | ")
```

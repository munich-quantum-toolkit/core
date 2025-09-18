# MQT Core ZX

MQT Core provides a minimal library for creating ZX-diagrams and rewriting them using the ZX-calculus.
The main use of this library within the Munich Quantum Toolkit is as a tool for equivalence checking of quantum circuits, especially with a focus on symbolic quantum circuits.
Other uses for the ZX-calculus and ZX-diagrams include compilation and optimization.
Furthermore, other diagrammatic rewriting systems exist for reasoning about different aspects of quantum circuits such as the ZH-calculus.
If you are looking for general-purpose libraries that encompass these functionalities, we recommend [PyZX](https://pyzx.readthedocs.io/en/latest/) or [QuiZX](https://github.com/zxcalc/quizx).
For an introduction to the ZX-calculus, we recommend the excellent [ZX-calculus for the working quantum computer scientist](https://arxiv.org/abs/2012.13966).

## Quickstart

There are two ways of obtaining a ZX-diagram with MQT Core.
The simplest way is from a quantum circuit, or rather, an MQT Core `QuantumComputation`:

```cpp
#include "zx/ZXDiagram.hpp"
#include "ir/QuantumComputation.hpp"
#include "zx/FunctionalityConstruction"

// Create GHZ state preparation circuit
ir::QuantumComputation qc{3};
qc.h(0);
qc.cx(1, 0);
qc.cx(2, 0);

// Create ZX-diagram from circuit
auto diag = zx::FunctionalityConstruction.buildFunctionality(&qc);
```

This yields the following ZX-diagram.

```{image} _static/ghz.svg
:width: 40%
:align: center
```

The other way is to manipulate the diagram directly.
Let's create the ZX-diagram for the GHZ circuit above directly.

We start off by creating an empty ZX-diagram.

```cpp
#include "zx/ZXDiagram.hpp"
#include "zx/ZXDefinitions.hpp"

zx::ZXDiagram diag{};
```

This is a diagram without any vertices.
Next, we add qubits to the diagram.

```cpp
diag.addQubits(3);
```

Some vertices in a ZX-diagram are special because they represent inputs and outputs of the diagram.
All vertices in a ZX-diagram are either of type `zx::VertexType::Z`, `zx::VertexType::X` or `zx::VertexType::Boundary`.
Boundary vertices denote the inputs.
Therefore, at this point the diagram has 6 vertices (3 input vertices, 3 output vertices) and 3 edges connecting the respective inputs and outputs.

Next, we add the Hadamard gate on qubit 0.
While Hadamard gates have an Euler decomposition in terms of X- and Z-spiders, they are so common that it is helpful to have a notation for them.
To this end, there are two types of edges in the diagram.
An edge is either `zx::EdgeType::Simple` or `zx::EdgeType::Hadamard`, where Hadamard edges represent wires with a Hadamard gate on them.

To represent a Hadamard gate we, therefore, have to add a new Vertex to the diagram, connect it to the input of qubit 0 with a Hadamard edge.

```cpp
auto in0 = diag.getInputs()[0];
auto newVertex = diag.addVertex(0, 0, zx::PiExpression(), zx::VertexType::Z);
diag.addEdge(in0, newVertex, zx::EdgeType::Hadamard);
```

We see that adding a new vertex requires 4 parameters.
These are the x- and y-coordinates of the vertex, the phase of the vertex and the vertextype.
The coordinates are there to identify where vertices lie if we would arrange the diagram on a grid.
They have no other special semantics.

The phase of the diagram is of type `zx::PiExpression` which is a class representing symbolic sums of monomials.
We will talk a bit more about these further below.
For now, we simply need to know that `zx::PiExpression()` represents a phase of 0.

Next we need to add two CNOTs to the diagram.
A CNOT in the ZX-calculus is represented by a Z-vertex (the control) and an X-vertex (the target), connected by a single non-Hadamard wire.

```cpp
auto in1 = diag.getInputs()[1];
auto in2 = diag.getInputs()[2];

auto ctrl1 = diag.addVertex(1, 0, zx::PiExpression(), zx::VertexType::Z);
auto ctrl2 = diag.addVertex(1, 0, zx::PiExpression(), zx::VertexType::Z);
auto trgt1 = diag.addVertex(1, 0, zx::PiExpression(), zx::VertexType::X);
auto trgt2 = diag.addVertex(2, 0, zx::PiExpression(), zx::VertexType::X);

// connect vertices to their respective qubit lines.
diag.addEdge(newVertex, ctrl1); // omitting the edge type adds a non-Hadamard edge
diag.addEdge(ctrl1, ctrl2);
diag.addEdge(in1, trgt1);
diag.addEdge(in2, trgt2);

// add edges for CNOTs
diag.addEdge(ctrl1, trgt1);
diag.addEdge(ctrl2, trgt2);

// complete diagram by connecting last vertices to the outputs
diag.addEdge(ctrl2, diag.getOutputs()[0]);
diag.addEdge(trgt1, diag.getOutputs()[1]);
diag.addEdge(trgt2, diag.getOutputs()[2]);
```

Let us return to the matter of symbolic phases and phases in general.
Phases of vertices in a ZX-diagram are all of type `zx::PiExpression` even if the phases are variable-free.
In the variable-free case, the `zx::PiExpression` consists only of a constant of type `zx::PiRational`.
The `zx::PiRational` class represents angles in the half-open interval $(-\pi, \pi]$ as a fraction of $\pi$.
For example, the number $\pi$ itself would be represented by the fraction $\frac{1}{1}$, the number $-\pi / 2$ would be $\frac{-1}{2}$.
This is because phases in terms of fractions of $\pi$ appear frequently in the ZX-calculus.
For more on symbolic expressions we refer to the code documentation.

## Rewriting

The true power of ZX-diagrams lies in the ZX-calculus which allows for manipulating ZX-diagrams.
The MQT Core ZX-calculus library provides some rewriting rules for ZX-diagrams in the header `Rules.hpp`.
The simplification routines are provided in the header `Simplify.hpp`.

For example, the previous diagram has multiple connected Z-vertices on qubit 0.
According to the axioms of the ZX-calculus, these can be merged via spider fusion.
We can perform this simplification on the diagram as follows.

```cpp
#include "zx/Simplify.hpp"

auto n_simplifications = spiderSimp(diag);
```

This results in the following diagram.

```{image} _static/ghz_simp.svg
:width: 40%
:align: center
```

`n_simplifications` will be two when executing this code since two spiders can be fused.
The diagrams are manipulated inplace for performance reasons.

For an overview on simplifications, we refer to the code documentation.

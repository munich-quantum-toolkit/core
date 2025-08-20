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

# MQT Core IR

The central interface for working with quantum computations throughout the Munich Quantum Toolkit is the {py:class}`~mqt.core.ir.QuantumComputation` class.
It effectively represents quantum computations as sequential lists of operation, similar to Qiskit's {py:class}`~qiskit.circuit.QuantumCircuit` class.

The following will demonstrate how to work with the {py:class}`~mqt.core.ir.QuantumComputation` class in Python.

:::{note}
MQT Core is primarily designed in C++ with a thin Python wrapper.
Historically, the C++ part of MQT Core was the focus and the Python interface was added later.
As the standards we hold ourselves to have evolved, the Python interface is much better documented than the C++ interface.
Contributions to the C++ documentation are welcome. See the [contribution guidelines](contributing.md) for more information.
:::

## Quickstart

The following code snippet demonstrates how to construct a quantum computation for an
instance of the Iterative Quantum Phase Estimation algorithm that aims to estimate
the phase of a unitary operator $U=p(3\pi/8)$ using 3 bits of precision.

```{code-cell} ipython3
---
mystnb:
  text_lexer: 'qasm3'
---
from mqt.core.ir import QuantumComputation

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
    qc.classic_controlled(op="p", target=q[0], cbit=c[j], params=[-pi / 2**(i - j)])
  qc.h(q[0])

  # Measure the result
  qc.measure(q[0], c[i])

  # Reset the qubit if not finished
  if i < precision - 1:
    qc.reset(q[0])
```

The circuit class provides lots of flexibility when it comes to the kind of gates that can be applied.
Check out the full API documentation of the {py:class}`~mqt.core.ir.QuantumComputation` class for more details.

## Visualizing Circuits

Circuits can be printed in a human-readable, text-based format.
The output is to be read from top to bottom and left to right.
Each line represents a single operation in the circuit.

:::{note}
The first and last lines have a special meaning: the first line contains the initial layout information, while the last line contains the output permutation. This is explained in more detail in the [Layout Information](#layout-information) section.
:::

```{code-cell} ipython3
print(qc)
```

Circuits can also easily be exported to OpenQASM 3 using the {py:meth}`~mqt.core.ir.QuantumComputation.qasm3_str` method.

```{code-cell} ipython3
---
mystnb:
  text_lexer: 'qasm3'
---
print(qc.qasm3_str())
```

## Layout Information

When compiling a quantum circuit for a specific quantum device, it is necessary to map the qubits of the circuit to the qubits of the device.
In addition, SWAP operations might be necessary to ensure that gates are only applied to qubits connected on the device.
These SWAP operations permute the assignment of circuit qubits to device qubits.
At the end of the computation, the values of the circuit qubits are measured at specific device qubits.
This kind of _layout information_ is important for reasoning about the functionality of the compiled circuit.
As such, preserving this information is essential for verification and debugging purposes.

:::{note}
In the literature, the qubits used in the circuit are often referred to as _logical qubits_ or _virtual qubits_, while the qubits of the device are also called _physical qubits_.
Within the MQT, we try to avoid the terms _logical_ and _physical_ qubits, as they can be misleading due to the connection to error correction.
Instead, we use the terms _circuit qubits_ and _device qubits_.
:::

To this end, the {py:class}`~mqt.core.ir.QuantumComputation` class contains two members, {py:attr}`~mqt.core.ir.QuantumComputation.initial_layout` and {py:attr}`~mqt.core.ir.QuantumComputation.output_permutation`, which are instances of the {py:class}`~mqt.core.ir.Permutation` class.
The initial layout tracks the mapping of circuit qubits to device qubits at the beginning of the computation, while the output permutation tracks where a particular circuit qubit is measured at the end of the computation.
While the output permutation can generally be inferred from the measurements in the circuit (using {py:meth}`~mqt.core.ir.QuantumComputation.initialize_io_mapping`), the initial layout is not always clear.
OpenQASM, for example, lacks a way to express the initial layout of a circuit and preserve this information.
Therefore, MQT Core will output the layout information as comments in the first two lines of the QASM string using the following format:

- `// i Q_0, Q_1, ..., Q_n`, meaning circuit qubit $i$ is mapped to device qubit $Q_i$.
- `// o Q_0, Q_1, ..., Q_n` meaning the value of circuit qubit $i$ (assumed to be stored in classical bit $c[i]$) is measured at device qubit $Q_i$.

An example illustrates the idea:

```{code-cell} ipython3
---
mystnb:
  text_lexer: 'qasm3'
---
# 3 qubits, 3 classical bits
qc = QuantumComputation(3, 3)

qc.h(0)
qc.x(1)
qc.s(2)

# c[0] is measured at device qubit 1
qc.measure(1, 0)
# c[1] is measured at device qubit 2
qc.measure(2, 1)
# c[2] is measured at device qubit 0
qc.measure(0, 2)

# determine permutation from measurement
qc.initialize_io_mapping()

print(qc.qasm3_str())
```

In the example above, the initial layout is not explicitly specified.
A trivial layout is thus assumed, where the circuit qubits are mapped to the device qubits in order.
The output permutation is determined from the measurements and is printed as comments in the QASM string.

:::{note}
This layout information is not part of the OpenQASM 3 standard.
It is a feature of MQT Core to help with debugging and verification.
MQT Core's QASM export will always include this layout information in the first two lines of the QASM string.
MQT Core's QASM import will parse these lines and set the initial layout and output permutation accordingly.
:::

## Operations

The operations in a {py:class}`~mqt.core.ir.QuantumComputation` object are of type {py:class}`~mqt.core.ir.operations.Operation`.
Every type of operation in `mqt-core` is derived from this class.
Operations can also be explicitly constructed.
Each {py:class}`~mqt.core.ir.operations.Operation` has a type in the form of an {py:class}`~mqt.core.ir.operations.OpType`.

### `StandardOperation`

A {py:class}`~mqt.core.ir.operations.StandardOperation` is used to represent basic unitary gates.
These can also be declared with arbitrarily many controls.

```{code-cell} ipython3
from mqt.core.ir.operations import OpType, StandardOperation, Control

# u3 gate on qubit 0
u_gate = StandardOperation(target=0, params=[pi / 4, pi, -pi / 2], op_type=OpType.u)

# controlled x-rotation
crx = StandardOperation(control=Control(0), target=1, params=[pi], op_type=OpType.rx)

# multi-controlled x-gate
mcx = StandardOperation(controls={Control(0), Control(1)}, target=2, op_type=OpType.x)

# add operations to a quantum computation
qc = QuantumComputation(3)
qc.append(u_gate)
qc.append(crx)
qc.append(mcx)

print(qc)
```

### `NonUnitaryOperation`

A {py:class}`~mqt.core.ir.operations.NonUnitaryOperation` is used to represent operations involving measurements or resets.

```{code-cell} ipython3
from mqt.core.ir.operations import NonUnitaryOperation

nqubits = 2
qc = QuantumComputation(nqubits, nqubits)
qc.h(0)

# measure qubit 0 on classical bit 0
meas_0 = NonUnitaryOperation(target=0, classic=0)

# reset all qubits
reset = NonUnitaryOperation(targets=[0, 1], op_type=OpType.reset)

qc.append(meas_0)
qc.append(reset)

print(qc)
```

### `SymbolicOperation`

A {py:class}`~mqt.core.ir.operations.SymbolicOperation` can represent all gates of a {py:class}`~mqt.core.ir.operations.StandardOperation` but the gate parameters can be symbolic.
Symbolic expressions are represented in MQT using the {py:class}`~mqt.core.ir.symbolic.Expression` type, which represent linear combinations of symbolic {py:class}`~mqt.core.ir.symbolic.Term` objects over some set of {py:class}`~mqt.core.ir.symbolic.Variable` objects.

```{code-cell} ipython3
from mqt.core.ir.operations import SymbolicOperation
from mqt.core.ir.symbolic import Expression, Term, Variable

x = Variable("x")
y = Variable("y")
sym = Expression([Term(x, 2), Term(y, 3)])
print(sym)

sym += 1
print(sym)

# Create symbolic gate
u1_symb = SymbolicOperation(target=0, params=[sym], op_type=OpType.p)

# Mixed symbolic and instantiated parameters
u2_symb = SymbolicOperation(target=0, params=[sym, 2.0], op_type=OpType.u2)
```

### `CompoundOperation`

A {py:class}`~mqt.core.ir.operations.CompoundOperation` bundles multiple {py:class}`~mqt.core.ir.operations.Operation` objects together.

```{code-cell} ipython3
from mqt.core.ir.operations import CompoundOperation

comp_op = CompoundOperation()

# create bell pair circuit
comp_op.append(StandardOperation(0, op_type=OpType.h))
comp_op.append(StandardOperation(target=0, control=Control(1), op_type=OpType.x))

qc = QuantumComputation(2)
qc.append(comp_op)

print(qc)
```

Circuits can be conveniently turned into operations which allows to create nested circuits:

```{code-cell} ipython3
nqubits = 2
comp = QuantumComputation(nqubits)
comp.h(0)
comp.cx(0, 1)

qc = QuantumComputation(nqubits)
qc.append(comp.to_operation())

print(qc)
```

### `IfElseOperation`

A {py:class}`~mqt.core.ir.operations.IfElseOperation` is an operation controlled by a classical bit or a classical register.
If a given condition is met, the {py:attr}`~mqt.core.ir.operations.IfElseOperation.then_operation` is applied.
If the condition is not met, the {py:attr}`~mqt.core.ir.operations.IfElseOperation.else_operation` is applied.

```{code-cell} ipython3
from mqt.core.ir.operations import IfElseOperation

qc = QuantumComputation(1, 1)

qc.h(0)
qc.measure(0, 0)
qc.if_else(
    then_operation=StandardOperation(target=0, op_type=OpType.x),
    else_operation=StandardOperation(target=0, op_type=OpType.y),
    control_bit=0,
)

print(qc)
```

## Interfacing with other SDKs and Formats

### OpenQASM

OpenQASM is a widely used format for representing quantum circuits.
Its latest version, [OpenQASM 3](https://openqasm.com/index.html), is a powerful language that can express a wide range of quantum circuits.
MQT Core supports the full functionality of OpenQASM 2.0 (including classically controlled operations) and a growing subset of OpenQASM 3.

```{code-cell} ipython3
from mqt.core.ir import QuantumComputation

qasm_str = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
cx q[0], q[1];
cx q[0], q[2];
"""

qc = QuantumComputation.from_qasm_str(qasm_str)

print(qc)
```

### Qiskit

In addition to OpenQASM, `mqt-core` can natively import [Qiskit](https://qiskit.org/) {py:class}`~qiskit.circuit.QuantumCircuit` objects.

```{code-cell} ipython3
from qiskit import QuantumCircuit

# GHZ circuit in qiskit
qiskit_qc = QuantumCircuit(3)
qiskit_qc.h(0)
qiskit_qc.cx(0, 1)
qiskit_qc.cx(0, 2)

qiskit_qc.draw(output="mpl", style="iqp")
```

```{code-cell} ipython3
from mqt.core.plugins.qiskit import qiskit_to_mqt

mqt_qc = qiskit_to_mqt(qiskit_qc)
print(mqt_qc)
```

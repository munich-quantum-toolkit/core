# Getting Started

The MQT Compiler Collection adopts a quantum-classical approach to quantum
compilation. Following an open-source philosophy, the project welcomes external
contributions. However, because quantum compilers are complex programs by
nature, it can sometimes be difficult to know where and how to start.

This page guides you through the key concepts and provides a solid understanding
of the project's quantum-compilation infrastructure and underlying design
decisions.

## Setup

Before you start, visit the [installation](../installation.md) page. There you
will find detailed instructions on how to download the project and install and
set up the MLIR framework correctly. Once this is done, you can compile the
project as follows:

Run the following commands from the repository root.

```console
cmake --preset release
cmake --build --preset release --target mqt-cc
```

If everything worked correctly, the following command should print a usage
message.

```console
./build/release/mlir/tools/mqt-cc/mqt-cc --help
```

## Fundamentals

To keep this guide self-contained, this section reviews the fundamentals of
quantum computing and the key concepts of MLIR. If you are familiar with both
quantum computing and MLIR, you may skip this section.

### Quantum Computing

_Qubits_ are the fundamental unit of quantum computing. Whereas a classical bit
is either in the state $0$ or $1$, a qubit can exist in a superposition of both
states. Mathematically, we denote a qubit as follows.

```{math}
:label: qubit_equation
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle,
```

where $\alpha$ and $\beta$ are complex numbers such that
$|\alpha|^2 + |\beta|^2 = 1$.

These complex numbers determine the probabilities of the outcome of a
_measurement_. A measurement collapses the qubit's state to $|0\rangle$ with
probability $|\alpha|^2$ and to $|1\rangle$ with probability $|\beta|^2$, and
returns the respective classical outcome ($0$ or $1$).

Quantum _gates_ --- mathematically described by unitary matrices --- modify a
qubit's state. For instance, the Hadamard gate H creates an equal superposition
state. Measuring that state yields $|0\rangle$ or $|1\rangle$ with probability
$0.5$ each.

```{math}
:label: hadamard_gate_ket0
\begin{aligned}
H|0\rangle &= \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle\\
H|1\rangle &= \frac{1}{\sqrt{2}}|0\rangle - \frac{1}{\sqrt{2}}|1\rangle
\end{aligned}
```

Another example of a one-qubit gate is the X gate, which simply flips the
qubit's state.

```{math}
:label: x_gate
\begin{aligned}
X|0\rangle &= |1\rangle\\
X|1\rangle &= |0\rangle
\end{aligned}
```

A quantum gate may target multiple qubits. The controlled-X gate acts on two
qubits and applies an X gate to the target qubit when the control (first) qubit
is in the $|1\rangle$ state:

```{math}
:label: cx_gate_action
\begin{aligned}
{CX}|00\rangle &= |00\rangle \\
{CX}|01\rangle &= |01\rangle \\
{CX}|10\rangle &= |11\rangle \\
{CX}|11\rangle &= |10\rangle
\end{aligned},
```

Quantum _circuits_ describe a quantum computation graphically:

```{figure} ../_static/mlir/bell-circuit.svg
:align: center
:width: 50%
:figclass: only-light
:name: fig:mlir-ir-structure

A circuit constructing the first Bell state.
```

```{figure} ../_static/mlir/bell-circuit-dark.svg
:align: center
:width: 50%
:figclass: only-dark
:name: fig:mlir-ir-structure

A circuit constructing the first Bell state.
```

Read from left to right, the quantum circuit above prepares and measures the
[Bell state](https://en.wikipedia.org/wiki/Bell_state) $|\Phi^{+}\rangle$:

1. Initialize both qubits in the $|0\rangle$ state.
2. Apply a Hadamard gate H to the upper qubit. Consequently, this qubit is now
   in an equal superposition state.
3. Apply a controlled-X gate to both qubits. The black dot $\bullet{}$ and the
   $\oplus{}$ represent the control and target qubits, respectively. The
   resulting two-qubit state is $|\Phi^{+}\rangle$.
4. Measure both qubits and receive two classical output bits.

With that, we have already covered the most fundamental building blocks of
quantum computing: qubits, measurements, gates, and circuits. In the next
section, you will learn about the Multi-Level Intermediate Representation (MLIR)
framework, bringing us one step closer to our goal of understanding and building
quantum compilers.

### Multi-Level Intermediate Representation (MLIR)

The Multi-Level Intermediate Representation (MLIR) project is an extensive
framework for building compilers for heterogeneous hardware. Key to its success
is the ability to represent programs at multiple levels of abstraction, as well
as the ability to lower them from higher to lower levels.

The core concept in MLIR is the _dialect_, which groups operations, types, and
attributes under a common namespace. A single program may combine multiple
dialects, which facilitates code reuse. For example, the `arith` dialect defines
integer and floating-point operations, while the `func` dialect lets you define
and call functions.

The following snippet contains a function `@main` that defines and adds two
32-bit integers.

```mlir
func.func @main() {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.addi %c1, %c2 : i32

  return
}
```

The `func` and `arith` prefixes specify the dialect names. For example,
`arith.constant` represents the `constant` operation from the `arith` dialect.
Moreover, the `%` prefix marks variables that store values, similar to other
programming languages. However, variables in MLIR adhere to the
_Static Single-Assignment_ (SSA) principle, where each variable is assigned
exactly once and never reassigned. For instance, the first `arith.constant`
operation produces the `%c1` SSA variable. Analogous to functional programming,
the `arith.addi` operation produces a new SSA variable `%c3` representing the
sum of its operands. Lastly, the `: i32` annotations specify the type, where
`i32` represents a 32-bit integer.

MLIR also provides dialects for structured control flow, such as loops and
conditionals, in the `scf` dialect. To demonstrate this, the snippet below sums
the numbers from 0 to 100 using the `scf.for` operation.

```{code-block} mlir
func.func @main() {
    %lb = arith.constant 0 : index
    %ub = arith.constant 100 : index
    %step = arith.constant 1 : index

    %sum_0 = arith.constant 0 : i32
    %sum = scf.for %iv = %lb to %ub step %step iter_args(%sum_iter = %sum_0) -> (i32) {
        %1 = arith.index_cast %iv : index to i32
        %2 = arith.addi %sum_iter, %1 : i32
        scf.yield %2 : i32
    }

    return
}
```

We already know what the first three operations do: they define constants of
type `index` using the `arith.constant` operation. Semantically, the `index`
type is much like `std::size_t` in C++. Namely, both represent a value that is
large enough to index any memory location on the target architecture.

Usually, we would implement the loop as follows:

```text
acc = 0
for iv = 0; iv < 100; ++iv {
  acc = acc + iv
}
```

However, because variables in MLIR adhere to the SSA principle, the
`acc = acc + iv` statement is invalid. To represent this loop in MLIR, we must
utilize _loop-carried variables_, which we specify via the `iter_args`
construct. These variables are passed from one iteration to another, maintaining
SSA form. In the example above, the `scf.yield` operation carries the SSA value
`%2` to the next iteration and returns it after the final iteration, where the
final result is stored in `%sum`.

Because MLIR uses a strongly-typed system, we must specify the type of the
loop-carried variable using the `->` symbol. Thus, in summary, the
`iter_args(%sum_iter = %sum_0) -> (i32)` specifies the loop-carried variable
`%sum_iter` with data type `i32`. Moreover, because the `%iv` variable has data
type `index`, a cast to `i32` using the `arith.index_cast` operation is required
for the subsequent addition operation.

Earlier, we stated that we can represent programs at multiple levels of
abstraction in MLIR. So, how exactly is this achieved? Even though the `scf`
dialect provides a `scf.if` operation, the following snippet uses the `cf`
(control flow) dialect to implement a conditional on a lower abstraction level.
In particular, the function `@select` returns either `%a` or `%b` depending on
the condition `%cond`.

```{code-block} mlir
func.func @select(%a: i32, %b: i32, %cond: i1) -> i32 {
    cf.cond_br %cond, ^exit(%a: i32), ^exit(%b: i32)
^exit(%v : i32):
    return %v : i32
}
```

The `^exit` label defines a _block_ that takes a 32-bit integer as an argument
and contains the `return` operation. There is another implicit block hidden in
the snippet above. Internally, the `@select` function is represented as follows.

```{code-block} mlir
func.func @select(i32, i32, i1) -> i32 {
^entry(%a: i32, %b: i32, %cond: i1):
    cf.cond_br %cond, ^exit(%a: i32), ^exit(%b: i32)
^exit(%v : i32):
    return %v : i32
}
```

The _terminator_ --- the last operation inside a block --- determines the
control flow. For instance, the `cf.cond_br` terminator jumps to the exit block
with `%v = %a` if `%cond` is true. Otherwise, it uses `%v = %b`. The `return`
operation is another example of a terminator that returns control to the caller
of the function. Generally, a block consists of multiple operations, with the
final one being the terminator.

Note that the `@select` function body consists of two blocks. The respective
enveloping structure in MLIR is a _region_. A region is always attached to an
operation (its parent), encompasses one or more blocks, and is usually indicated
by curly braces. The following figure illustrates the interplay of operations,
blocks, and regions graphically.

```{figure} ../_static/mlir/mlir-ir-structure.svg
:align: center
:width: 70%
:figclass: only-light
:name: fig:mlir-ir-structure

The nested substructures of an IR.
```

```{figure} ../_static/mlir/mlir-ir-structure-dark.svg
:align: center
:width: 70%
:figclass: only-dark
:name: fig:mlir-ir-structure

The nested substructures of an IR.
```

For each operation in the `scf` dialect, there is an equivalent sequence of
operations in the `cf` dialect. This transformation to a lower abstraction level
is referred to as _lowering_. For instance, the following program is
semantically equivalent to the previous `scf.for` example but uses the `cf`
dialect.

```mlir
func.func @main() {
    %lb = arith.constant 0 : index
    %ub = arith.constant 100 : index
    %step = arith.constant 1 : index

    %sum_0 = arith.constant 0 : i32

    cf.br ^bb1(%lb, %sum_0 : index, i32)
  ^bb1(%iv: index, %sum_iter: i32):  // 2 preds: ^bb0, ^bb2
    %cond = arith.cmpi slt, %iv, %ub : index
    cf.cond_br %cond, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %1 = arith.index_cast %iv : index to i32
    %sum_next = arith.addi %sum_iter, %1 : i32
    %iv_next = arith.addi %iv, %step : index
    cf.br ^bb1(%iv_next, %sum_next : index, i32)
  ^bb3:  // pred: ^bb1
    return
  }
```

While higher-level abstractions are extremely useful for optimizations, it is
much easier to generate actual machine instructions from lower-level dialects.

Luckily, we don't have to perform this _conversion_ --- the transformation from
one dialect to another --- by hand. The MLIR framework already implements this
and many other conversions between the built-in dialects. Furthermore, we can
develop additional custom conversions using the conversion framework, which
defines exactly how a transformation must look and under what circumstances the
resulting IR is considered valid.

A conversion is a specific class of transformation within the
_pass infrastructure_. Simply put, a pass traverses the nested substructures of
a program and optionally rewrites some parts of it. Moreover, multiple passes
can be orchestrated into a _pass pipeline_, which executes a series of
configurable passes sequentially.

That's it! Now that we've covered the fundamentals, we can move on and explore
how the MQT Compiler Collection utilizes MLIR to build a quantum-classical
compilation framework.

## The MQT Compiler Collection

The MQT Compiler Collection provides tools to optimize and transpile quantum
programs. This section outlines how we use the MLIR framework and its
compilation infrastructure to implement these tasks.

### Quantum Dialects

The MQT Compiler Collection defines two dialects in MLIR, each with a distinct
purpose. While the Quantum Circuit (QC) dialect is tailored to exchanging with
other formats (such as OpenQASM), the Quantum Circuit Optimization (QCO) dialect
is --- as the name suggests --- specifically designed for optimizations. Let's
explore their differences.

The following snippets allocate and subsequently deallocate a dynamic qubit
using the `alloc` operation of each dialect. In the QC dialect, we can
deallocate dynamic qubits using the `dealloc` operation, whereas in the QCO
dialect we mark the end of a qubit's lifespan with the `sink` operation.

::::{grid} 2

:::{grid-item}

```{code-block} mlir
//          QC
%q0 = qc.alloc : !qc.qubit
qc.dealloc %q0 : !qc.qubit
```

:::

:::{grid-item}

```mlir
//            QCO
%q0_0 = qco.alloc : !qco.qubit
qco.sink %q0_0 : !qco.qubit
```

:::

::::

To target specific hardware qubits, we use the `static` operation. While static
qubits in the QCO dialect still require a `sink` operation, a `dealloc`
operation is not required in the QC dialect. There is a sound rationale behind
this seemingly obscure design choice: The QCO dialect enforces "linear typing",
where each qubit SSA value is used _exactly_ once. If there were no `sink`
operation for static qubits in the QCO dialect, this property would be violated.

::::{grid} 2

:::{grid-item}

```{code-block} mlir
//          QC
%q0 = qc.static 1 : !qc.qubit
```

:::

:::{grid-item}

```mlir
//            QCO
%q0_0 = qco.static 1 : !qco.qubit
qco.sink %q0_0 : !qco.qubit
```

:::

::::

Let's apply a Hadamard gate to a qubit next:

::::{grid} 2

:::{grid-item}

```{code-block} mlir
//          QC
%q0 = qc.alloc : !qc.qubit
qc.h %q0 : !qc.qubit
qc.dealloc %q0 : !qc.qubit
```

:::

:::{grid-item}

```mlir
//            QCO
%q0_0 = qco.alloc : !qco.qubit
%q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit
qco.sink %q0_1 : !qco.qubit
```

:::

::::

Notice how the Hadamard operation in the QCO dialect consumes and produces SSA
values, while the operation in the QC dialect operates on a qubit in place. We
say that the QC dialect uses "reference semantics" whereas the QCO dialect uses
"value semantics". Semantically, the operations in the QCO dialect return the
new state of a qubit after modifying it.

Instead of a Hadamard gate, we can also achieve the same transformation with X
and Y rotations using parameterized gates as follows:

::::{grid} 2

:::{grid-item}

```{code-block} mlir
//          QC
%theta = arith.constant 1.570796 : f64
%phi = arith.constant 3.141593 : f64

%q0 = qc.alloc : !qc.qubit
qc.ry(%theta) %q0 : !qc.qubit
qc.rx(%phi) %q0 : !qc.qubit
qc.dealloc %q0 : !qc.qubit
```

:::

:::{grid-item}

```mlir
//            QCO
%theta = arith.constant 1.570796 : f64
%phi = arith.constant 3.141593 : f64

%q0_0 = qco.alloc : !qco.qubit
%q0_1 = qco.ry(%theta) %q0_0 : !qco.qubit -> !qco.qubit
%q0_2 = qco.rx(%phi) %q0_1 : !qco.qubit -> !qco.qubit
qco.sink %q0_2 : !qco.qubit
```

:::

::::

To measure qubits, we use the `measure` operation. In the QCO dialect, the
measurement operation returns not only the classical measurement outcome but
also the state after measurement.

::::{grid} 2

:::{grid-item}

```{code-block} mlir
//          QC
%q0 = qc.alloc : !qc.qubit
qc.h %q0 : !qc.qubit
%c0 = qc.measure %q0 : !qc.qubit -> i1
qc.dealloc %q0 : !qc.qubit
```

:::

:::{grid-item}

```mlir
//            QCO
%q0_0 = qco.alloc : !qco.qubit
%q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit
%q0_2, %c0 = qco.measure %q0_1 : !qco.qubit
qco.sink %q0_2 : !qco.qubit
```

:::

::::

Moving on from one-qubit gates, let us apply a controlled-X operation. To that
end, we allocate a second qubit and use the `ctrl` modifier operation of the
respective dialect to implement the controlled-X. By using modifiers, arbitrary
(multi-)controlled gates can be represented without having to explicitly define
them.

::::{grid} 2

:::{grid-item}

```{code-block} mlir
//          QC
%q0 = qc.alloc : !qc.qubit
%q1 = qc.alloc : !qc.qubit

qc.h %q0 : !qc.qubit
qc.ctrl(%q0) targets(%arg0 = %q1) {
    qc.x %arg0 : !qc.qubit

} : {!qc.qubit}, {!qc.qubit}

%c0 = qc.measure %q0 : !qc.qubit -> i1
%c1 = qc.measure %q1 : !qc.qubit -> i1

qc.dealloc %q0 : !qc.qubit
qc.dealloc %q1 : !qc.qubit
```

:::

:::{grid-item}

```mlir
//            QCO
%q0_0 = qco.alloc : !qco.qubit
%q1_0 = qco.alloc : !qco.qubit

%q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit
%q0_2, %q1_1 = qco.ctrl(%q0_1) targets (%arg0 = %q1_0) {
    %q0_2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
    qco.yield %q0_2
} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

%q0_3, %c0 = qco.measure %q0_2 : !qco.qubit
%q1_2, %c1 = qco.measure %q1_1 : !qco.qubit

qco.sink %q0_3 : !qco.qubit
qco.sink %q1_2 : !qco.qubit
```

:::

::::

The figure below illustrates the data-flow graph of the textual QCO intermediate
representation above.

```{figure} ../_static/mlir/data-flow.svg
:align: center
:width: 75%
:figclass: only-light

The data-flow graph of the QCO IR shown above.
```

```{figure} ../_static/mlir/data-flow-dark.svg
:align: center
:width: 75%
:figclass: only-dark

The data-flow graph of the QCO IR shown above.
```

The dependencies between operations are naturally expressed because the QCO
dialect models quantum computations as directed acyclic "data-flow" graphs
(DAGs). For instance, the controlled-X operation depends on the application of
the Hadamard operation. This is especially useful for gate cancellation: if one
gate is the inverse of another, cancel both. Consequently, the expressive
data-flow representation is what makes the QCO dialect so powerful for
optimization and, more generally, algorithms.

The following figure depicts the data-flow of the `ctrl` modifier.

```{figure} ../_static/mlir/ctrl-modifier.svg
:align: center
:width: 75%
:figclass: only-light

The data-flow of the `ctrl` modifier.
```

```{figure} ../_static/mlir/ctrl-modifier-dark.svg
:align: center
:width: 75%
:figclass: only-dark

The data-flow of the `ctrl` modifier.
```

The `qco.ctrl` operation also supports multiple targets. The following two
snippets apply a CXX gate.

::::{grid} 2

:::{grid-item}

```{code-block} mlir
//          QC
%q0 = qc.alloc : !qc.qubit
%q1 = qc.alloc : !qc.qubit
%q2 = qc.alloc : !qc.qubit

qc.h %q0 : !qc.qubit
qc.ctrl(%q0) targets(%arg0 = %q1, %arg1 = %q2) {
    qc.x %arg0 : !qc.qubit
    qc.x %arg1 : !qc.qubit
    qc.yield
} : {!qc.qubit}, {!qc.qubit, !qc.qubit}

%c0 = qc.measure %q0 : !qc.qubit -> i1
%c1 = qc.measure %q1 : !qc.qubit -> i1
%c2 = qc.measure %q2 : !qc.qubit -> i1

qc.dealloc %q0 : !qc.qubit
qc.dealloc %q1 : !qc.qubit
qc.dealloc %q2 : !qc.qubit
```

:::

:::{grid-item}

```mlir
//            QCO
%q0_0 = qco.alloc : !qco.qubit
%q1_0 = qco.alloc : !qco.qubit
%q2_0 = qco.alloc : !qco.qubit

%q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit
%q0_2, %q1_1, %q2_1 = qco.ctrl(%q0_1) targets (%arg0 = %q1_0, %arg1 = %q2_0) {
    %q1_1 = qco.x %arg0 : !qco.qubit -> !qco.qubit
    %q2_1 = qco.x %arg1 : !qco.qubit -> !qco.qubit
    qco.yield %q1_1, %q2_1
} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

%q0_3, %c0 = qco.measure %q0_2 : !qco.qubit
%q1_2, %c1 = qco.measure %q1_1 : !qco.qubit
%q2_2, %c2 = qco.measure %q2_1 : !qco.qubit

qco.sink %q0_3 : !qco.qubit
qco.sink %q1_2 : !qco.qubit
qco.sink %q2_2 : !qco.qubit
```

:::

::::

In many front-end quantum languages, there is a concept of a register, that is,
a collection of qubits. The QC and QCO dialects use the `memref` and `qtensor`
dialects to describe these constructs, respectively, where the latter is part of
the MQT Compiler Collection. The following snippets construct the three-qubit
[GHZ](https://en.wikipedia.org/wiki/Greenberger–Horne–Zeilinger_state) state in
the QC and QCO dialects. In many front-end quantum languages, there is a concept
of a register, that is, a collection of qubits. The QC and QCO dialects use the
`memref` and `qtensor` dialects to describe these constructs, respectively,
where the latter is part of the MQT Compiler Collection. The following snippets
construct the three-qubit
[GHZ](https://en.wikipedia.org/wiki/Greenberger–Horne–Zeilinger_state) state in
the QC and QCO dialects.

::::{grid} 2

:::{grid-item}

```{code-block} mlir
//          QC
%i0 = arith.constant 0 : index
%i1 = arith.constant 1 : index
%i2 = arith.constant 2 : index


%r0 = memref.alloc() : memref<3x!qc.qubit>

%q0 = memref.load %r0[%i0] : memref<3x!qc.qubit>
%q1 = memref.load %r0[%i1] : memref<3x!qc.qubit>
%q2 = memref.load %r0[%i2] : memref<3x!qc.qubit>

qc.h %q0 : !qc.qubit

qc.ctrl(%q0) targets(%arg0 = %q1) {
    qc.x %arg0 : !qc.qubit

} : {!qc.qubit}, {!qc.qubit}

qc.ctrl(%q0) targets(%arg0 = %q2){
    qc.x %arg0 : !qc.qubit

} : {!qc.qubit}, {!qc.qubit}





memref.dealloc %r0 : memref<3x!qc.qubit>
```

:::

:::{grid-item}

```mlir
//            QCO
%i0 = arith.constant 0 : index
%i1 = arith.constant 1 : index
%i2 = arith.constant 2 : index
%N = arith.constant 3 : index

%r0_0 = qtensor.alloc(%N) : tensor<3x!qco.qubit>

%r0_1, %q0_0 = qtensor.extract %r0_0[%i0] : tensor<3x!qco.qubit>
%r0_2, %q1_0 = qtensor.extract %r0_1[%i1] : tensor<3x!qco.qubit>
%r0_3, %q2_0 = qtensor.extract %r0_2[%i2] : tensor<3x!qco.qubit>

%q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit

%q0_2, %q1_1 = qco.ctrl(%q0_1) targets (%arg0 = %q1_0) {
  %q1_1 = qco.x %arg0 : !qco.qubit -> !qco.qubit
  qco.yield %q1_1
} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

%q0_3, %q2_1 = qco.ctrl(%q0_2) targets (%arg0 = %q2_0) {
  %q2_1 = qco.x %arg0 : !qco.qubit -> !qco.qubit
  qco.yield %q2_1
} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

%r0_4 = qtensor.insert %q0_3 into %r0_3[%i0] : tensor<3x!qco.qubit>
%r0_5 = qtensor.insert %q1_1 into %r0_4[%i1] : tensor<3x!qco.qubit>
%r0_6 = qtensor.insert %q2_1 into %r0_5[%i2] : tensor<3x!qco.qubit>

qtensor.dealloc %r0_6 : tensor<3x!qco.qubit>
```

:::

::::

Similarly to the earlier argument, the QCO dialect requires `insert` operations
for qubit SSA values to satisfy linear typing. Moreover, the QCO dialect also
enforces this property for registers (`%r0_1`, `%r0_2`, etc.). Consequently, in
the QCO dialect, the SSA value of a register represents its state.

What if we wanted to construct a four-qubit GHZ state? A straightforward
solution would be to simply increase the size of the register and add another
controlled-X. However, there is a more expressive and elegant solution using
structured control flow (SCF) operations; in particular, a loop. To do so, we
leverage MLIR's built-in `scf` dialect.

::::{grid} 2

:::{grid-item}

```mlir
//          QC
%N = arith.constant 4 : index

%r0 = memref.alloc(%N) : memref<?x!qc.qubit>

%i0 = arith.constant 0 : index
%q0 = memref.load %r0[%i0] : memref<?x!qc.qubit>
qc.h %q0 : !qc.qubit

%c1 = arith.constant 1 : index
scf.for %iv = %c1 to %N step %c1 {
  %qi = memref.load %r0[%iv] : memref<?x!qc.qubit>

  qc.ctrl(%q0) targets(%arg0 = %qi) {
    qc.x %arg0 : !qc.qubit

  } : {!qc.qubit}, {!qc.qubit}




}



memref.dealloc %r0 : memref<?x!qc.qubit>
```

:::

:::{grid-item}

```mlir
//            QCO
%N = arith.constant 4 : index

%r0_0 = qtensor.alloc(%N) : tensor<?x!qco.qubit>

%i0 = arith.constant 0 : index
%r0_1, %q0_0 = qtensor.extract %r0_0[%i0] : tensor<?x!qco.qubit>
%q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit

%c1 = arith.constant 1 : index
%r0_3, %q0_2 = scf.for %iv = %c1 to %N step %c1 iter_args(%ri_0 = %r0_1, %ctrl_0 = %q0_1) -> (tensor<?x!qco.qubit>, !qco.qubit) {
  %ri_1, %qi_0 = qtensor.extract %ri_0[%iv] : tensor<?x!qco.qubit>

  %ctrl_1, %qi_1 = qco.ctrl(%ctrl_0) targets (%arg3 = %qi_0) {
    %6 = qco.x %arg3 : !qco.qubit -> !qco.qubit
    qco.yield %6 : !qco.qubit
  } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

  %ri_2 = qtensor.insert %qi_1 into %ri_1[%iv] : tensor<?x!qco.qubit>

  scf.yield %ri_2, %ctrl_1 : tensor<?x!qco.qubit>, !qco.qubit
}

%r0_4 = qtensor.insert %q0_2 into %r0_3[%i0] : tensor<?x!qco.qubit>

qtensor.dealloc %r0_4 : tensor<?x!qco.qubit>
```

:::

::::

What if we wanted to construct a generic N-qubit GHZ state? Naturally, a
function taking N as a parameter seems like an appropriate choice.

::::{grid} 2

:::{grid-item}

```mlir
//          QC
func.func @ghz(%N: index) {
  %r0 = memref.alloc(%N) : memref<?x!qc.qubit>

  %i0 = arith.constant 0 : index
  %q0 = memref.load %r0[%i0] : memref<?x!qc.qubit>
  qc.h %q0 : !qc.qubit

  %c1 = arith.constant 1 : index
  scf.for %iv = %c1 to %N step %c1 {
    %qi = memref.load %r0[%iv] : memref<?x!qc.qubit>

    qc.ctrl(%q0) targets(%arg0 = %qi) {
      qc.x %arg0 : !qc.qubit

    } : {!qc.qubit}, {!qc.qubit}



  }




  memref.dealloc %r0 : memref<?x!qc.qubit>

  return
}
```

:::

:::{grid-item}

```mlir
//            QCO
 func.func @ghz(%N: index) {
    %r0_0 = qtensor.alloc(%N) : tensor<?x!qco.qubit>

    %i0 = arith.constant 0 : index
    %r0_1, %q0_0 = qtensor.extract %r0_0[%i0] : tensor<?x!qco.qubit>
    %q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit

    %c1 = arith.constant 1 : index
    %r0_3, %q0_2 = scf.for %iv = %c1 to %N step %c1 iter_args(%ri_0 = %r0_1, %ctrl_0 = %q0_1) -> (tensor<?x!qco.qubit>, !qco.qubit) {
      %ri_1, %qi_0 = qtensor.extract %ri_0[%iv] : tensor<?x!qco.qubit>

      %ctrl_1, %qi_1 = qco.ctrl(%ctrl_0) targets (%arg3 = %qi_0) {
        %6 = qco.x %arg3 : !qco.qubit -> !qco.qubit
        qco.yield %6 : !qco.qubit
      } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

      %ri_2 = qtensor.insert %qi_1 into %ri_1[%iv] : tensor<?x!qco.qubit>

      scf.yield %ri_2, %ctrl_1 : tensor<?x!qco.qubit>, !qco.qubit
    }

    %r0_4 = qtensor.insert %q0_2 into %r0_3[%i0] : tensor<?x!qco.qubit>

    qtensor.dealloc %r0_4 : tensor<?x!qco.qubit>

    return
  }
```

:::

::::

Maybe you have already noticed that the dialects in the MQT Compiler Collection
can represent quantum computations far beyond simple quantum circuits.
Ultimately, our goal is to define dialects expressive enough to represent a
broad range of hybrid quantum-classical computations.

### Building Programs Programmatically

Sometimes it can be useful to create an in-memory representation of a quantum
program. For instance, a unit test might want to build a quantum program,
perform some actions, and then verify that the resulting quantum program matches
the expected outcome.

To create quantum programs in the MLIR representation, the MQT Compiler
Collection provides the `QCProgramBuilder` and `QCOProgramBuilder` classes,
respectively. The following code snippets illustrate their usage.

::::{grid} 2

:::{grid-item}

```{code-block} cpp
qc::QCProgramBuilder builder(/*MLIRContext*/ &ctx);
builder.initialize();

auto q0 = builder.allocQubit();
auto q1 = builder.allocQubit();

builder.h(q0);
builder.ctrl(q0, [&] { builder.x(q1); });






builder.barrier({q0, q1});





auto c0 = builder.measure(q0);
auto c1 = builder.measure(q1);

builder.dealloc(q0);
builder.dealloc(q1);

auto program = builder.finalize();
```

:::

:::{grid-item}

```{code-block} cpp
qco::QCOProgramBuilder builder(&ctx);
builder.initialize();

auto q0 = builder.allocQubit();
auto q1 = builder.allocQubit();

q0 = builder.h(q0);
const auto [ctrlsOut, targOut] = builder.ctrl(q0, q1,
  [&](ValueRange targets) -> SmallVector<Value> {
   return {builder.x(targets[0])};
});
q0 = ctrlsOut[0];
q1 = targOut[0];

auto barrierOut = builder.barrier({q0, q1});
q0 = barrierOut[0];
q1 = barrierOut[1];

Value c0;
Value c1;
std::tie(q0, c0) = builder.measure(q0);
std::tie(q1, c1) = builder.measure(q1);

builder.sink(q0);
builder.sink(q1);

auto program = builder.finalize();
```

:::

::::

### Compilation Flow

The goal of any compiler is to take a (quantum) program and transform it into a
more efficient, executable one. The MQT Compiler Collection achieves this using
the following compilation pipeline:

- First, a program in a front-end quantum language (e.g., OpenQASM) is
  translated to the QC dialect.
- Next, the compiler converts the program to the QCO dialect and subsequently
  applies hardware-agnostic and optionally hardware-dependent passes. Finally,
  the program is converted back to the QC dialect.
- Optionally, the optimized and transpiled program can be translated into a
  target representation such as LLVM using the Quantum Intermediate
  Representation (QIR) extension.

The figure below illustrates the compilation flow graphically.

```{figure} ../_static/mlir/compiler-collection-pipeline.svg
:align: center
:width: 70%
:figclass: only-light

The compilation pipeline of the MQT Compiler Collection.
```

```{figure} ../_static/mlir/compiler-collection-pipeline-dark.svg
:align: center
:width: 70%
:figclass: only-dark

The compilation pipeline of the MQT Compiler Collection.
```

This has an important consequence: Theoretically, any front-end quantum language
(today's and tomorrow's!) that translates its abstract syntax tree to the QC
dialect can leverage all passes developed and maintained within the MQT Compiler
Collection. This approach has been a success story in the classical compiler
world, where, instead of relying on proprietary compilation stacks, programming
languages such as C++, Rust, and Go use LLVM as an optimization driver. The MQT
Compiler Collection attempts to establish this design philosophy in the quantum
world.

## Writing Your First Optimization Pass

Optimizations are a core part of the MQT Compiler Collection, as they are for
any compiler. In this section, you will implement a simple optimization pass and
see how it fits into the MQT Compiler Collection workflow. So where should this
pass live in the codebase?

### Directory Layout

To show where this pass belongs, let’s look at the directory layout.

---

**`mlir/include/mlir/`**

This folder contains `.h` header files and TableGen `.td` specifications. It
consists of the following subdirectories:

| Directory     | Description                                                                  |
| ------------- | ---------------------------------------------------------------------------- |
| `Compiler/`   | Defines the compiler pipeline.                                               |
| `Conversion/` | Defines conversions from or to other MLIR dialects.                          |
| `Dialect/`    | Defines (among others) the QC and QCO dialects. Contains the TableGen files. |
| `Support/`    | Defines utilities.                                                           |

Each dialect follows a consistent structure:

| Directory     | Description                                             |
| ------------- | ------------------------------------------------------- |
| `Builder/`    | Defines the program builder.                            |
| `IR/`         | Defines the dialect, operations, and types in TableGen. |
| `Transforms/` | Defines transformations on the dialect.                 |
| `Utils/`      | Defines utilities.                                      |

**`mlir/lib/`**

The accompanying `.cpp` files for the headers. It follows the same folder
structure as the include directory.

**`mlir/tools/`**

This folder contains the entry-point function for the `mqt-cc` executable.

**`mlir/unittests/`**

This folder contains unit-tests for the MQT Compiler Collection.

---

With that in place, let’s move on to the pass implementation.

### Consecutive Hadamard Cancellation Pass

Because the Hadamard gate corresponds to a Hermitian matrix, two consecutive
Hadamards cancel each other.

```{math}
:label: hh=i
HH^{\dagger} = HH = I
```

We can use this equality to simplify a given quantum program. Towards that end,
we want to implement an optimization pass within the MQT Compiler Collection. By
doing so, everyone using the MQT Compiler Collection as a backend benefits from
the work we put into the optimization pass.

First, we need to define the pass in the QCO's TableGen file responsible for
transformations.

```{code-block} cpp
// file: mlir/include/mlir/Dialect/QCO/Transforms/Passes.td

def CancelConsecutiveHadamards : Pass<"cancel-consecutive-hadamards", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect"];
  let summary = "Cancel two consecutive Hadamard gates";
  let description = [{
    Cancels two consecutive Hadamard gates acting on the same qubit,
    reducing circuit depth & gate count.
  }];
}
```

Let's dissect the TableGen definition:

- The summary and description are self-explanatory.
- To load and initialize `QCODialect` automatically in the MLIR context, we list
  it as a dependent dialect.
- The pass entry operation is `mlir::ModuleOp`. Since a module is the top-level
  container for `func.func` operations, attaching the pass to the module applies
  it to all nested operations, including functions.
- The pass is available from the command line via
  `--cancel-consecutive-hadamards`.

Thanks to TableGen, we do not have to worry about boilerplate code and can
instead let the tool generate it. To generate these files, build the
`MLIRQCOTransforms` target. We use the auto-generated
`CancelConsecutiveHadamardsBase` class to define the actual pass in C++ as
follows.

```{code-block} cpp
// file: mlir/lib/Dialect/QCO/Transforms/Optimizations/CancelConsecutiveHadamards.cpp

#include "mlir/Dialect/QCO/Transforms/Passes.h"

namespace mlir::qco {

#define GEN_PASS_DEF_CANCELCONSECUTIVEHADAMARDS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc" // -> CancelConsecutiveHadamardsBase

/**
 * @brief Cancel two consecutive Hadamard gates.
 */
struct CancelConsecutiveHadamards final
    : impl::CancelConsecutiveHadamardsBase<CancelConsecutiveHadamards> {

  // Inherit the constructor of the base class.
  using CancelConsecutiveHadamardsBase::CancelConsecutiveHadamardsBase;

protected:

  // This method is invoked for every entry point operation. Here: mlir::ModuleOp.
  void runOnOperation() override {
    mlir::ModuleOp entryPoint = getOperation();

    // TODO
  }
};
} // namespace mlir::qco
```

To implement the logic of the pass, we utilize MLIR's rewrite patterns.
Particularly, we add a class `CancelConsecutiveHadamardsPattern` which inherits
from `OpRewritePattern<HOp>`, where the template variable `HOp` specifies that
the pattern should match Hadamard operations. The overridden `matchAndRewrite`
method is called for each matched Hadamard operation.

```cpp
// file: mlir/lib/Dialect/QCO/Transforms/Optimizations/CancelConsecutiveHadamards.cpp

#include "mlir/Dialect/QCO/IR/QCOOps.h" // Newly added.

struct CancelConsecutiveHadamardsPattern final : OpRewritePattern<HOp> {

  explicit CancelConsecutiveHadamardsPattern(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(HOp op,
                                PatternRewriter& rewriter) const override {

    // Check if the successor is also a Hadamard operation.
    auto nextOp = dyn_cast<HOp>(*op.getOutputQubit(0).user_begin());
    if (!nextOp) {
      return failure();
    }

    // Replace the output of 'nextOp' with the input of 'op', essentially
    // unlinking both operations.
    //
    //           ┌───┐    ┌───┐
    // ───%in───▶│ H |───▶│ H |───%out───▶  =>  ───%out = %in───▶
    //           └───┘    └───┘
    //

    rewriter.replaceAllUsesWith(
      /*from= */nextOp.getOutputQubit(0),
      /*to= */op.getInputQubit(0));
    rewriter.eraseOp(nextOp);
    rewriter.eraseOp(op);
  }
};
```

Finally, we implement the `runOnOperation` method by initializing a pattern set
with `CancelConsecutiveHadamardsPattern` and calling `applyPatternsGreedily`,
which repeatedly applies the pattern until no matches remain.

```{code-block} cpp
// file: mlir/lib/Dialect/QCO/Transforms/Optimizations/CancelConsecutiveHadamards.cpp

void runOnOperation() override {
  ModuleOp entryPoint = getOperation();

  // Create the pattern set.
  RewritePatternSet patterns(&getContext());
  patterns.add<CancelConsecutiveHadamardsPattern>();

  // Apply patterns in an iterative and greedy manner.
  if (failed(applyPatternsGreedily(entryPoint, std::move(patterns)))) {
    signalPassFailure();
  }
}
```

Congratulations, you have written your first optimization pass in MLIR! 🎉

The MQT Compiler Collection implements a generic version of this pass not only
for the Hadamard gate but for many hermitian gates at once. If you are curious
about how this is achieved using C++ templates, see
[QCOUtils.h](https://github.com/munich-quantum-toolkit/core/blob/main/mlir/include/mlir/Dialect/QCO/QCOUtils.h).
To see the pass in action, continue with the next section.

## Using the MQT Compiler Collection Tool

Finally, to run your optimization, this section shows you how to use the MQT
Compiler Collection tool (`mqt-cc`).

### Optimizing an OpenQASM Program

Let's say you want to optimize the following OpenQASM program. Create a `.qasm`
file and name it `ghz.qasm`:

```{code-block}
:lineno-start: 0
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

h q[0];
h q[0];
h q[0];

cx q[0], q[1];
cx q[1], q[2];

barrier q[0], q[1], q[2];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
```

The program defines four qubits but uses only three. Next, execute the MQT
Compiler Collection tool:

```console
mqt-cc ghz.qasm
```

The MQT Compiler Collection tool will print the following IR:

```{code-block} mlir
:lineno-start: 0
module {
  func.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
    %c0_i64 = arith.constant 0 : i64
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index

    %alloc = memref.alloc() : memref<3x!qc.qubit>
    %0 = memref.load %alloc[%c0] : memref<3x!qc.qubit>
    %1 = memref.load %alloc[%c1] : memref<3x!qc.qubit>
    %2 = memref.load %alloc[%c2] : memref<3x!qc.qubit>

    qc.h %0 : !qc.qubit

    qc.ctrl(%0) targets (%arg0 = %1) {
      qc.x %arg0 : !qc.qubit
    } : {!qc.qubit}, {!qc.qubit}
    qc.ctrl(%1) targets (%arg0 = %2) {
      qc.x %arg0 : !qc.qubit
    } : {!qc.qubit}, {!qc.qubit}

    qc.barrier %0, %1, %2 : !qc.qubit, !qc.qubit, !qc.qubit
    %3 = qc.measure("c", 3, 0) %0 : !qc.qubit -> i1
    %4 = qc.measure("c", 3, 1) %1 : !qc.qubit -> i1
    %5 = qc.measure("c", 3, 2) %2 : !qc.qubit -> i1

    memref.dealloc %alloc : memref<3x!qc.qubit>
    return %c0_i64 : i64
  }
}
```

Note that instead of three consecutive only one hadamard is applied. Thus, the
optimizer has successfully applied the $HH^{\dagger} = HH = I$ equality.

### Emitting Quantum Intermediate Representation (QIR)

Now that your quantum program is optimized, you may want to simulate it using a
classical simulator via the MQT QIR Runtime. To transform the program into QIR,
you can supply the `--emit-qir-base` (or `--emit-qir-adaptive`) command-line
option:

```console
mqt-cc ghz.qasm --emit-qir-base > ghz.ll
```

Next, refer to the [QIR Runtime Guide](../qir/index.md) on how to run the
program with a classical simulator.

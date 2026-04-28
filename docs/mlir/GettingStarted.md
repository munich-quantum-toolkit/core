# Getting Started

## Setup

Before we actually get started, make sure to visit the [installation](../installation.md) page.
There you will find detailed instructions on how to download the project as well as install and set up the Multi-Level Intermediate Representation (MLIR) framework correctly.
Once this is done, you can compile the project as follows:

```console
$ cmake -B build \
    -DMLIR_DIR=<path to MLIR> \
    -DLLVM_DIR=<path to LLVM> \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_MQT_CORE_MLIR=ON
$ cd build && cmake --build . --target mqt-cc
```

If everything worked correctly, the following command should print a usage message.

```console
$ ./mlir/tools/mqt-cc/mqt-cc --help
```

## Fundamentals

To keep this tutorial self-contained, this section reviews the fundamentals of quantum computing and the key concepts of MLIR.
If you are familiar with both quantum computing and MLIR, you may skip this section.

### Quantum Computing

Qubits are the fundamental computational unit of quantum computing.
Whereas a classical bit is either in the state $0$ or $1$, a qubit can exist in a superposition of both states.
Mathematically, we denote a qubit as follows.

```{math}
:label: qubit_equation
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle
```

where $\alpha$ and $\beta$ are complex numbers such that $|\alpha|^2 + |\beta|^2 = 1$.

These complex numbers determine the probabilities of the outcome of a measurement.
A measurement collapses the qubit's state to $|0\rangle$ with probability $|\alpha|^2$ and to $|1\rangle$ with probability $|\beta|^2$ and returns the respective classical outcome ($0$ or $1$).

Quantum gates --- mathematically described by unitary matrices --- modify a qubit's state.
For instance, the Hadamard gate H creates the equal superposition state for which a measurement collapses the state to $|0\rangle$ or $|1\rangle$ with probability $0.5$.

```{math}
:label: hadamard_gate_ket0
\begin{aligned}
H|0\rangle &= \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle\\
H|1\rangle &= \frac{1}{\sqrt{2}}|0\rangle - \frac{1}{\sqrt{2}}|1\rangle
\end{aligned}
```

Another example of a one-qubit gate is the X gate, which simply flips the qubit's state.

```{math}
:label: x_gate
\begin{aligned}
X|0\rangle &= |1\rangle\\
X|1\rangle &= |0\rangle
\end{aligned}
```

A quantum gate may target multiple qubits: The controlled-X gate acts on two qubits and applies an X gate on the target qubit, if the control qubit is in the $|1\rangle$ state:

```{math}
:label: cx_gate_action
\begin{aligned}
{CX}|00\rangle &= |00\rangle \\
{CX}|01\rangle &= |01\rangle \\
{CX}|10\rangle &= |11\rangle \\
{CX}|11\rangle &= |10\rangle
\end{aligned}
```

where the first qubit acts as the control qubit.

Quantum circuits describe a quantum computation graphically:

```{image} ../_static/bell-circuit.svg
:width: 50%
:align: center
```

Read from left to right, the above quantum circuit computes and measures the [Bell state](https://en.wikipedia.org/wiki/Bell_state) $|\Phi^{+}\rangle$:

1. Initialize both qubits in the $|0\rangle$ state.
2. Apply a Hadamard gate H to the upper qubit. Consequently, this qubit is now in an equal superposition state.
3. Apply a controlled-X gate to both qubits. The black dot $\bullet{}$ and the $\oplus{}$ represent the control and target qubits, respectively. The resulting two-qubit state is $|\Phi^{+}\rangle$.
4. Measure both qubits and receive two classical output bits.

With that, we've already covered the most fundamental building blocks of quantum computing: qubits, measurements, gates, and finally circuits.
In the next section, you will learn about the Multi-Level Intermediate Representation (MLIR) framework, bringing us one step closer to our goal of understanding and building quantum compilers.

### Multi-Level Intermediate Representation (MLIR)

The Multi-Level Intermediate Representation (MLIR) project is an extensive framework to build compilers for heterogeneous hardware.
Key to its success is the ability to represent programs at multiple levels of abstraction, as well as the capacity to lower them from higher to lower levels.

The core concept in MLIR is a dialect.
A dialect groups operations, types, and attributes under a common namespace.
A single program may combine multiple dialects, which facilitates code reuse.
For example, the structured control flow (SCF) dialect provides functionality for control flow constructs, while the `arith` dialect defines integer and floating-point operations.
Another essential dialect is the `func` dialect, which lets us define and call functions.

The following snippet combines the three dialects into a single program which sums up the numbers from 0 to 100.

```mlir
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

- The `func`, `arith`, and `scf` prefixes specify the dialect's name. For example, `arith.constant` represents the `constant` operation from the `arith` dialect.
- The percentage symbol `%` prefixes static single-assignment (SSA) values --- a principle, where each variable is assigned exactly once but never reassigned. For instance, the first `arith.constant` operation produces the `%lb` SSA value.
- The `: index` and `: i32` specifies the type, where `i32` represents an 32-bit integer while `index` is a special type for loop bounds. The `arith.index_cast` operation casts the `%iv` variable with the type `index` to `i32`.
- To return the final values after loop termination, we define loop-carried variables via the `iter_args` construct and the return type with the `->` symbol. Inside the loop body, we specify the value for the next iteration via the `scf.yield` operation.

Moving beyond, the function below returns either `%a` or `%b` depending on the conditional `%cond`.

```mlir
func.func @select(i32, i32, i1) -> i32 {
^entry(%a: i32, %b: i32, %cond: i1):
    cf.cond_br %cond, ^exit(%a: i32), ^exit(%b: i32)
^exit(%v : i32):
    return %v : i32
}
```

- The `^entry` and `^exit` define a block, respectively. In MLIR, a block is a list of operations. Moreover, blocks take a list of arguments in an intuitive, function-like, way.
- The terminator, the last operation inside the block, determines the control flow. For instance, the `cf.cond_br` terminator jumps to the exit block with variable `%a`, if the `%cond` is `1`. Otherwise, it uses variable `%b`. The `return` operation is another example of a terminator which returns the control flow to the caller of the function.
- A region combines multiple blocks and is indicated by curly brackets.

The following figure illustrates the interplay of operations, blocks, and regions graphically.

```{image} ../_static/mlir-regions-blocks-ops.svg
:width: 70%
:align: center
```

The control flow dialect (`cf`) is the lower-level equivalent of the structured control flow dialect (`scf`).
For each IR that uses the SCF dialect there is an equivalent one in the CF dialect.
For example, the IR below is semantically equivalent to the one above that sums up the numbers from 0 to 100.
However, it uses the CF instead of the SCF dialect.

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

Luckily, we don't have to perform this conversion --- the transformation from one dialect to another --- per hand.
The MLIR framework already implements this and many other conversions between the built-in dialects.
Furthermore, we can develop custom conversions using the conversion framework which defines exactly how a transformation must look like and under what circumstances the resulting IR is considered valid.

Conversions are a specific instance of transformation passes. More generally, a pass in MLIR traverses the IR and optionally modifies it. An example of a non-rewriting pass are analyses passes, which simply collect statistics of the IR. Moreover, multiple passes can be combined into a pass pipeline which executes a series of passes sequentially.

That's it! Now that we've also got all the fundamentals covered, we can move on and explore how the MQT Compiler Collection utilizes MLIR to build a compiler for quantum computing.

## The MQT Compiler Collection

The MQT Compiler Collection (`mqt-cc`) provides tools to optimize and transpile quantum programs.
This section outlines how we utilize the MLIR framework as well as its compilation infrastructure to implement these tasks.

### Quantum Dialects

The MQT Compiler Collection defines two dialects in MLIR, each with a distinctive purpose.
While the Quantum Circuit (QC) dialect is great for exchanging with other formats (such as OpenQASM), the Quantum Circuit Optimization (QCO) dialect is --- as the name suggests --- specifically designed for optimizations.
Let's explore their differences.

The following snippet allocates and subsequently deallocates a dynamic qubit using the `alloc` operation of the respective dialect.
In the QC dialect, we can dealloc dynamic qubits using the `dealloc` operation, whereas in the QCO dialect we define end of a qubit's lifespan with the `sink` operation.

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

To target specific hardware qubits, we use the `static` operation.
While static qubits in the QCO dialect still require a `sink` operation, the `dealloc` is omitted for the QC dialect.
There is a sound rationale behind this seemingly obscure design choice: The QCO dialect enforces "linear typing", where each (qubit) SSA value is used _exactly_ once.
If there wasn't a `sink` operation for static qubits in the QCO dialect, this property would be violated.

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

Let's apply an Hadamard gate to a qubit next:

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

Notice how the Hadamard operation in the QCO dialect consumes and produces SSA values, while the operation in the QC dialect simply references the targeted qubit.
We say that the QC dialect uses "reference semantics" whereas the QCO dialect uses "value semantics".
Semantically, the unitary operations in the QCO dialect return the new state after modifying it.

Instead of the Hadamard, we can also apply the identical transformation as X and Y rotations with parameterized gates as follows:

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

To measure qubits, we use the `measure` operation.
In the QCO dialect, the measurement operation returns not only the classical measurement outcome but also the state after measurement.

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

Moving on from one-qubit gates, let us apply a controlled-X operation.
Towards that end, we allocate a second qubit and use the `ctrl` modifier operation of the respective dialect to implement the controlled-X.
By using modifiers, arbitrary (multi-)controlled gates can be represented without having to explicitly define them.

::::{grid} 2
:::{grid-item}

```{code-block} mlir
//          QC
%q0 = qc.alloc : !qc.qubit
%q1 = qc.alloc : !qc.qubit

qc.h %q0 : !qc.qubit
qc.ctrl(%q0) {
    qc.x %q1 : !qc.qubit

} : !qc.qubit

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

The figure below illustrates the data-flow graph of the above textual intermediate representation graphically.

```{figure} ../_static/mlir/data-flow.svg
:width: 75%
:align: center
:name: fig:data-flow

The data-flow graph of the IR shown above.
```

The dependencies between operations are naturally express because the QCO dialect models quantum computations as directed acyclic "data-flow" graphs (DAG).
For instance, the controlled-X operation depends on the application of the Hadamard operation.
This is, for example, very useful for gate cancellation: The dependency of one gate is the inverse of it? Cancel the two!
Consequently, the expressive dataflow representation is what makes the QCO dialect so powerful for optimization and algorithms more generally.

However, with that expressiveness complexity increases also. This is best seen for the `qco.ctrl` operation:

- The input target qubit must be explicitly specified and is aliased to the block argument `%arg0`.
- The result of the `qco.x` operation needs to be passed to the outer block. Thus, similarly to the operations in the SCF dialect, we use `qco.yield` to return the control flow to the outer scope.
- Analogously to the other unitary operations in the QCO dialect, the `qco.ctrl` modifier returns the modified state of the input qubits.

The following figure depicts the data-flow of the `ctrl` modifier.

```{figure} ../_static/mlir/ctrl-modifier.svg
:width: 75%
:align: center
:name: fig:ctrl-modifier

The data-flow of the `ctrl` modifier.
```

In many front-end quantum languages, there is a concept describing a register (a collection) of qubits.
The QC and QCO dialect use the `memref` and `qtensor` dialects to describe these constructs, respectively, where the latter is part of the MQT Compiler Collection.
The following snippets construct the [GHZ](https://en.wikipedia.org/wiki/Greenberger–Horne–Zeilinger_state) state in the QC and QCO dialect.

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
qc.h %q1 : !qc.qubit
qc.h %q2 : !qc.qubit

qc.ctrl(%q0) {
    qc.x %q1 : !qc.qubit

} : !qc.qubit

qc.ctrl(%q0) {
    qc.x %q2 : !qc.qubit

} : !qc.qubit





memref.dealloc %r0 : memref<3x!qc.qubit>
```

:::

:::{grid-item}

```mlir
//            QCO
%i0 = arith.constant 0 : index
%i1 = arith.constant 1 : index
%i2 = arith.constant 2 : index

%sz = arith.constant 3 : index
%r0_0 = qtensor.alloc(%sz) : tensor<3x!qco.qubit>

%r0_1, %q0_0 = qtensor.extract %r0_0[%i0] : tensor<3x!qco.qubit>
%r0_2, %q1_0 = qtensor.extract %r0_1[%c1] : tensor<3x!qco.qubit>
%r0_3, %q2_0 = qtensor.extract %r0_2[%c2] : tensor<3x!qco.qubit>

%q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit
%q1_1 = qco.h %q1_0 : !qco.qubit -> !qco.qubit
%q2_1 = qco.h %q2_0 : !qco.qubit -> !qco.qubit

%q0_2, %q1_2 = qco.ctrl(%q0_1) targets (%arg0 = %q1_1) {
  %q1_2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
  qco.yield %q1_2
} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

%q0_3, %q2_2 = qco.ctrl(%q0_2) targets (%arg0 = %q2_1) {
  %q2_2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
  qco.yield %q2_2
} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

%r0_4 = qtensor.insert %targets_out into %r0_3[%c1] : tensor<3x!qco.qubit>
%r0_5 = qtensor.insert %targets_out_5 into %r0_4[%c2] : tensor<3x!qco.qubit>
%r0_6 = qtensor.insert %controls_out_4 into %r0_5[%c0] : tensor<3x!qco.qubit>

qtensor.dealloc %r0_6 : tensor<3x!qco.qubit>
```

:::
::::

Similarly to the argument above, to satisfy linear typing, the QCO dialect requires `insert` operations for the qubit SSA values.
Moreover, the QCO dialect also enforces this property for registers (`%r0_1`, `%r0_2`, etc.).
Consequently, in the QCO dialect the SSA value of a register represents its state.

### Compilation Flow

The goal of any compiler is to take a (quantum) program and transform into a more efficient and executable one. The MQT Compiler Collection achieves this using the following compilation pipeline:

- First, a program in an input quantum language (e.g. OpenQASM) is translated to the QC dialect.
- Next, the compiler transforms the program to QCO dialect. Subsequently, we apply optimizations, optionally perform transpilation for a target quantum architecture, and finally transform the program back to the QC dialect.
- Optionally, the optimized and transpiled program can be transformed into an exit dialect such as LLVM using the Quantum Intermediate Representation (QIR) extension.

The figure below illustrates the compilation flow graphically.

```{figure} ../_static/mlir/compiler-collection-pipeline.svg
:width: 70%
:align: center
:name: fig:compiler-collection-pipeline

The compilation pipeline of the MQT Compiler Collection.
```

## Writing Your First Optimization Pass

### Directory Layout

Wrapping one's head around the folder structure of MLIR projects can be quite confusing in the beginning.
To help you navigate the project, the following paragraphs provide a brief introduction to its directory layout.

**`core/mlir/include/mlir/`**

This folder contains `.h` header files and TableGen `.td` specifications.
It consists of the following sub-directories:

| Directory     | Description                                                                  |
| ------------- | ---------------------------------------------------------------------------- |
| `Compiler/`   | Defines the compiler pipeline.                                               |
| `Conversion/` | Defines conversions from or to other MLIR dialects.                          |
| `Dialect/`    | Defines (among others) the QC and QCO dialects. Contains the TableGen files. |
| `Support/`    | Defines utilities.                                                           |

Each of the dialects follows a consistent structure:

| Directory     | Description                                             |
| ------------- | ------------------------------------------------------- |
| `Builder/`    | Defines the program builder.                            |
| `IR/`         | Defines the dialect, operations, and types in TableGen. |
| `Transforms/` | Defines transformations on the dialect.                 |
| `Utils/`      | Defines utilities.                                      |

**`core/mlir/lib/`**

The accompanying `.cpp` files for the headers. Follows the same folder structure as the include directory.

**`core/mlir/tools/`**

In this folder resides the entry point function for the `mqt-cc` executable.

**`core/mlir/unittests/`**

This folder contains unit-tests for the MQT Compiler Collection.

## Using the MQT Compiler Collection Tool

This section shows you how to use the MQT Compiler Collection Tool (`mqt-cc`).

### Optimizing an OpenQASM Program

Lets say you want to optimize the following OpenQASM program. Create a `.qasm` file and name it `ghz.qasm`:

```{code-block} OpenQASM
:lineno-start: 0
OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[4];

h q[0];
cx q[0], q[1];
cx q[1], q[2];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
```

The program defines four qubits, but uses only three. Next, execute the MQT Compiler Collection Tool:

```console
% mqt-cc ghz.qasm
```

The MQT Compiler Collection Tool will print the following IR:

```{code-block} mlir
:lineno-start: 0
module {
  func.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
    %c0_i64 = arith.constant 0 : i64
    %0 = qc.alloc("q", 4, 0) : !qc.qubit
    %1 = qc.alloc("q", 4, 1) : !qc.qubit
    %2 = qc.alloc("q", 4, 2) : !qc.qubit
    qc.h %0 : !qc.qubit
    qc.ctrl(%0) {
      qc.x %1 : !qc.qubit
    } : !qc.qubit
    qc.ctrl(%1) {
      qc.x %2 : !qc.qubit
    } : !qc.qubit
    %3 = qc.measure("c", 4, 0) %0 : !qc.qubit -> i1
    %4 = qc.measure("c", 4, 1) %1 : !qc.qubit -> i1
    %5 = qc.measure("c", 4, 2) %2 : !qc.qubit -> i1
    qc.dealloc %0 : !qc.qubit
    qc.dealloc %1 : !qc.qubit
    qc.dealloc %2 : !qc.qubit
    return %c0_i64 : i64
  }
}
```

Note that only three instead of four qubits are allocated.
Thus, the optimizer has successfully removed the unused qubit.

### Emitting Quantum Intermediate Representation (QIR)

Now that your quantum program is optimized, you want to simulate it using a classical simulator via the MQT QIR Runtime.
To transform the program to the QIR, you can supply the `--emit-qir` command-line option:

```console
% mqt-cc ghz.qasm --emit-qir
```

Using the `mlir-translate` tool, store the file as LLVM file (`.ll`) as follows.

```console
% mqt-cc ghz.qasm --emit-qir | mlir-translate --mlir-to-llvmir > ghz.ll
```

Next, refer to the [QIR Runtime Guide](../qir/index.md) on how to run the program with a classical simulator.

## Conclusion

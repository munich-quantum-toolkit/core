# Getting Started With the MQT Compiler Collection

## Installing And Building

Make sure to visit the [installation](../installation.md) page which describes how to download the project as well as install and setup MLIR correctly. Once you have done that you can build the project via your favourite IDE or using the following commands:

```console
% cmake -Bbuild \
    -DMLIR_DIR=<path to MLIR> \
    -DLLVM_DIR=<path to LLVM> \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_MQT_CORE_MLIR=ON
% cd build && cmake --build . --target mqt-cc
```

You can verify your build by running the following command. If everything worked correctly, it should print an usage message.

```console
% ./mlir/tools/mqt-cc/mqt-cc --help
```

## Fundamentals

This section covers the basics of quantum computing and provides a soft introduction to the Multi-Level Intermediate Representation (MLIR) framework. If you are familiar with both quantum computing and MLIR, you may skip this section.

### Quantum Computing

### Multi-Level Intermediate Representation (MLIR)

The Multi-Level Intermediate Representation (MLIR) project is an extensive framework to build compilers for heterogeneous hardware. Key to its success is the ability to represent programs at multiple levels of abstraction, as well as the capacity to _lower_ them from higher to lower levels.

The core concept in MLIR is a _dialect_. A dialect groups operations, types, and attributes under a common namespace. A single program may combine multiple dialects, which facilitates code reuse. For example, the structured control flow (SCF) dialect provides functionality for control flow constructs, while the arith dialect defines integer and floating-point operations. Another essential dialect is the Func dialect, which lets us define and call functions.

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

- The `func.`, `arith.`, `scf.` specifies the dialect's name. For example, `arith.constant` represents the `constant` operation from the arith dialect.
- The percentage symbol `%` prefixes static single-assignment (SSA) values - a principle, where each variable is assigned exactly once but never reassigned. For instance, the first `arith.constant` operation produces the `%lb` SSA value.
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

- The `^entry` and `^exit` define a _block_, respectively. In MLIR, a block is a list of operations. Moreover, blocks take a list of arguments in an intuitive, function-like, way.
- The _terminator_, the last operation inside the block, determines the control flow. For instance, the `cf.cond_br` terminator jumps to the exit block with variable `%a`, if the `%cond` is `1`. Otherwise, it uses variable `%b`. The `return` operation is another example of a terminator which returns the control flow to the caller of the function.
- A _region_ combines multiple blocks and is indicated by curly brackets.

The following figure illustrates the interplay of operations, blocks, and regions graphically.

```{image} ../_static/mlir-regions-blocks-ops.svg
:width: 70%
:align: center
```

Now that we've got all the fundamentals covered, we can move on and explore how the MQT Compiler Collection works.

## The MQT Compiler Collection

The MQT Compiler Collection provides tools to optimize and transpile quantum programs. This section outlines how we utilize the MLIR framework as well as its compilation infrastructure to implement these tasks.

### Quantum Dialects

The MQT Compiler Collection defines two dialects in MLIR, each with a distinctive purpose. While the Quantum Circuit (QC) dialect is great for exchanging with other formats (such as OpenQASM), the Quantum Circuit Optimization (QCO) dialect is - as the name suggests - specifically designed for optimizations. Let's explore their differences.

The following snippet allocates and subsequently deallocates one qubit using the `alloc` and `dealloc` operations, respectively.

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
qco.dealloc %q0_0 : !qco.qubit
```

:::
::::

Alternatively, we can target specific hardware qubits by using the `static` operation.

::::{grid} 2
:::{grid-item}

```{code-block} mlir
//          QC
%q0 = qc.static 1 : !qc.qubit
qc.dealloc %q0 : !qc.qubit
```

:::

:::{grid-item}

```mlir
//            QCO
%q0_0 = qco.static 1 : !qco.qubit
qco.dealloc %q0_0 : !qco.qubit
```

:::
::::

So far so good. No visible differences. Next, we want to apply the single-qubit Hadamard gate.

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
qco.dealloc %q0_1 : !qco.qubit
```

:::
::::

Notice how the Hadamard operation in the QCO dialect consumes and produces SSA values, while the operation in the QC dialect simply references the targeted qubit. We say that the QC dialect uses _reference semantics_ whereas the QCO dialect uses _value semantics_. Semantically, the unitary operations in the QCO dialect return the new state after modifying it.

Instead of using the Hadamard directly, we can also apply the transformation in terms of X and Y rotations with parameterized gates.

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
qco.dealloc %q0_2 : !qco.qubit
```

:::
::::

To measure qubits, use the `measure` operation. In the QCO dialect, the measurement operation returns not only the classical measurement outcome but also the state after measurement.

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
qco.dealloc %q0_2 : !qco.qubit
```

:::
::::

Moving on from one-qubit gates, let us apply a controlled-X operation. Towards that end, we allocate another qubit and subsequently use the `ctrl` _modifier_ operation of the respective dialect to implement the controlled-X. Thanks to modifiers, we can represent arbitrary (multi-)controlled gates without having to explicitly define them.

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

qco.dealloc %q0_3 : !qco.qubit
qco.dealloc %q1_2 : !qco.qubit
```

:::
::::

The `qco.ctrl` operation adds a bit of complexity:

- The input target qubit must be explicitly specified and is aliased to the block argument `%arg0`.
- The result of the `qco.x` operation needs to be passed to the outer block. Thus, similarly to the operations in the SCF dialect, we use `qco.yield`.
- Analogously to the other unitary operations in the QCO dialect, the `qco.ctrl` modifier returns the modified state of the input qubits.

```{image} ../_static/qco-dataflow.svg
:width: 85%
:align: center
```

### Compilation Flow

## Examples Using the MQT Compiler Collection Tool

This section contains examples showing you how to use the MQT Compiler Collection Tool.

### Optimizing an OpenQASM Program

Let's say you want to optimize the following OpenQASM program. Create a `.qasm` file and name it `ghz.qasm`:

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

The MQT Compiler Collection Tool will output the following IR on stdout:

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

Note how in the IR only three instead of four qubits are allocated. Consequently, the optimizer has successfully removed the unused qubit.

### Emitting Quantum Intermediate Representation (QIR)

Now that your quantum program is optimized, you want to simulate it using a classical simulator via the MQT QIR Runtime. To transform the program to the QIR, you can supply the `--emit-qir` command-line option:

```console
% mqt-cc ghz.qasm --emit-qir
```

Using the `mlir-translate` tool, store the file as LLVM file (`.ll`) as follows.

```console
% mqt-cc ghz.qasm --emit-qir | mlir-translate --mlir-to-llvmir > ghz.ll
```

Next, refer to the [QIR Runtime Guide](../qir/index.md) on how to run the program with a classical simulator.

## Conclusion

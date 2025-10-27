// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef FluxOPS
#define FluxOPS

include "mlir/Dialect/Flux/IR/FluxInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def FluxDialect : Dialect {
    let name = "flux";

    let summary = "The Flux (value semantics) dialect for quantum computing.";

    let description = [{
        The Flux dialect uses **value semantics** where quantum operations
        consume input qubits and produce new output values, following the
        functional programming and SSA paradigm. This model enables:

        - Powerful compiler optimizations through clear dataflow
        - Safe reordering and parallelization analysis
        - Advanced transformation passes
        - Explicit dependency tracking

        The name "Flux" captures the flowing, transformative nature of
        value-based representations—quantum states flow through operations,
        each transformation producing new values like a river flowing through
        a landscape.

        Example:
        ```mlir
        %q_out = flux.h %q_in                        // Consumes %q_in, produces %q_out
        %q0_out, %q1_out = flux.swap %q0_in, %q1_in  // Consumes inputs, produces outputs
        ```
    }];

    let cppNamespace = "::mlir::flux";

    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Flux Type Definitions
//===----------------------------------------------------------------------===//

class FluxType<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<FluxDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : FluxType<"Qubit", "qubit"> {
    let summary = "Flux qubit value type";
    let description = [{
        The `!flux.qubit` type represents an SSA value holding a quantum bit
        in the Flux dialect. Operations using this type consume input qubits
        and produce new output qubits following value semantics and the SSA
        paradigm, enabling powerful dataflow analysis and optimization.

        Example:
        ```mlir
        %q0 = flux.alloc : !flux.qubit
        %q1 = flux.h %q0 : !flux.qubit -> !flux.qubit
        %q2 = flux.x %q1 : !flux.qubit -> !flux.qubit
        ```
    }];
}

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

class FluxOp<string mnemonic, list<Trait> traits = []> :
    Op<FluxDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Resource Operations
//===----------------------------------------------------------------------===//

def AllocOp : FluxOp<"alloc", [MemoryEffects<[MemAlloc]>]> {
    let summary = "Allocate a qubit dynamically";
    let description = [{
        Allocates a new qubit dynamically and returns an SSA value representing it.
        The qubit is initialized to the |0⟩ state.

        Optionally, the qubit can be part of a register by specifying:
        - `register_name`: The name of the register this qubit belongs to
        - `register_size`: The total size of the register
        - `register_index`: The index of this qubit within the register

        Example (single qubit):
        ```mlir
        %q = flux.alloc : !flux.qubit
        ```

        Example (qubits in a register):
        ```mlir
        %q0 = flux.alloc("q", 3, 0) : !flux.qubit
        %q1 = flux.alloc("q", 3, 1) : !flux.qubit
        %q2 = flux.alloc("q", 3, 2) : !flux.qubit
        ```
    }];

    let arguments = (ins OptionalAttr<StrAttr>:$register_name,
                     OptionalAttr<ConfinedAttr<I64Attr, [IntPositive]>>:$register_size,
                     OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$register_index);
    let results = (outs QubitType:$result);
    let assemblyFormat = [{
        (`(` $register_name^ `,` $register_size `,` $register_index `)`)?
        attr-dict `:` type($result)
    }];

    let builders = [
        OpBuilder<(ins), [{
            build($_builder, $_state, QubitType::get($_builder.getContext()), nullptr, nullptr, nullptr);
        }]>,
        OpBuilder<(ins "::mlir::StringAttr":$register_name,
                       "::mlir::IntegerAttr":$register_size,
                       "::mlir::IntegerAttr":$register_index), [{
            build($_builder, $_state, QubitType::get($_builder.getContext()),
                  register_name, register_size, register_index);
        }]>
    ];

    let hasVerifier = 1;
}

def DeallocOp : FluxOp<"dealloc", [MemoryEffects<[MemFree]>]> {
    let summary = "Deallocate a qubit";
    let description = [{
        Deallocates a qubit, releasing its resources.

        Example:
        ```mlir
        flux.dealloc %q : !flux.qubit
        ```
    }];

    let arguments = (ins QubitType:$qubit);
    let assemblyFormat = "$qubit attr-dict `:` type($qubit)";
    let hasCanonicalizer = 1;
}

def StaticOp : FluxOp<"static", [Pure]> {
    let summary = "Retrieve a static qubit by index";
    let description = [{
        The `flux.static` operation produces an SSA value representing a qubit
        identified by a static index. This is useful for referring to fixed
        qubits in a quantum program or to hardware-mapped qubits.

        Example:
        ```mlir
        %q = flux.static 0 : !flux.qubit
        ```
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntNonNegative]>:$index);
    let results = (outs QubitType:$qubit);
    let assemblyFormat = "$index attr-dict `:` type($qubit)";
}

//===----------------------------------------------------------------------===//
// Measurement and Reset Operations
//===----------------------------------------------------------------------===//

def MeasureOp : FluxOp<"measure"> {
    let summary = "Measure a qubit in the computational basis";
    let description = [{
        Measures a qubit in the computational (Z) basis, collapsing the state
        and returning both the output qubit and a classical bit result.

        Optionally, the measurement can be recorded to an output register by
        specifying:
        - `register_name`: Name of the classical register (e.g., "c")
        - `register_size`: Total size of the register
        - `register_index`: Index within the register for this measurement

        Example (simple measurement):
        ```mlir
        %q_out, %result = flux.measure %q_in : !flux.qubit
        ```

        Example (measurement with output recording):
        ```mlir
        %q_out, %result = flux.measure("c", 2, 0) %q_in : !flux.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the qubit to measure", [MemRead]>:$qubit_in,
                     OptionalAttr<StrAttr>:$register_name,
                     OptionalAttr<ConfinedAttr<I64Attr, [IntPositive]>>:$register_size,
                     OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$register_index);
    let results = (outs QubitType:$qubit_out, I1:$result);
    let assemblyFormat = [{
        (`(` $register_name^ `,` $register_size `,` $register_index `)`)?
        $qubit_in `:` type($qubit_in) attr-dict
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in), [{
            build($_builder, $_state, QubitType::get($_builder.getContext()), $_builder.getI1Type(),
                  qubit_in, nullptr, nullptr, nullptr);
        }]>
    ];

    let hasVerifier = 1;
}

def ResetOp : FluxOp<"reset", [Idempotent, SameOperandsAndResultType]> {
    let summary = "Reset a qubit to |0⟩ state";
    let description = [{
        Resets a qubit to the |0⟩ state, regardless of its current state,
        and returns the reset qubit.

        Example:
        ```mlir
        %q_out = flux.reset %q_in : !flux.qubit -> !flux.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the qubit to reset", [MemRead]>:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";
    let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Unitray Operations
//===----------------------------------------------------------------------===//

def XOp : FluxOp<"x", traits = [Flux_UnitaryOpInterface]> {
    let arguments = (ins QubitType:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        size_t getNumQubits() { return getNumTargets() + getNumControls(); }
        size_t getNumTargets() const { return 1; }
        size_t getNumControls() { return getNumPosControls() + getNumNegControls(); }
        size_t getNumPosControls() const;
        size_t getNumNegControls() const;
        ::mlir::Value getQubit(size_t i) const;
        ::mlir::Value getTarget(size_t i);
        ::mlir::Value getPosControl(size_t i) const;
        ::mlir::Value getNegControl(size_t i) const;
        ::mlir::Value getInput(size_t i);
        ::mlir::Value getOutput(size_t i);
        ::mlir::Value getInputForOutput(::mlir::Value output);
        ::mlir::Value getOutputForInput(::mlir::Value input);
        size_t getNumParams() const { return 0; }
        ::mlir::flux::ParameterDescriptor getParameter(size_t i) const;
        bool hasStaticUnitary() const { return true; }
        ::mlir::DenseElementsAttr tryGetStaticMatrix();
        ::llvm::StringRef getBaseSymbol() const { return "x"; }
        ::mlir::flux::CanonicalDescriptor getCanonicalDescriptor() const;
    }];
}

#endif // FluxOPS

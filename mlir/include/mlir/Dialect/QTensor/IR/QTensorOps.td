// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef QTENSOROPS
#define QTENSOROPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def QTensorDialect : Dialect {
  let name = "qtensor";

  let summary = "The QTensor dialect for one-dimensional tensors of qubits with linear typing.";

  let description = [{
      The Qtensor dialect is an adjusted variant of the standard tensor dialect of MLIR that supports linear typing for one-dimensional tensors of qubits.
      In order to support linear typing, the extract operations of this dialect are modified so that they also return the updated tensor.
      In addition, alloc/dealloc operations are added to the dialect to support the bulk allocation and deallocation of qubit tensors with linear types.
  }];

  let dependentDialects = ["::mlir::qco::QCODialect"];

  let cppNamespace = "::mlir::qtensor";
}

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

class QTensorOp<string mnemonic, list<Trait> traits = []>
    : Op<QTensorDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

def QubitType :
  Type<CPred<"llvm::isa<::mlir::qco::QubitType>($_self)">,
       "Elements must be of qubit type">;

def Static1DQubitTensor : Type<
  And<[
    1DTensorOf<[QubitType]>.predicate,
    HasStaticShapePred
  ]>,
  "Tensor must be statically shaped and elements must be of qubit type">;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def QTensor_AllocOp : QTensorOp<"alloc", [
    MemoryEffects<[MemAlloc]>]> {
    let summary = "Tensor alloc operation";
    let description = [{
        Allocates a qubit tensor with the given size and returns the allocated tensor.
        The qubits are initialized to the |0> state.

        Example:
        ```mlir
        %qtensor = qtensor.alloc(3) : tensor<3x!qco.qubit>
        ```
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntPositive]>:$size);
    let results = (outs Static1DQubitTensor:$result);
    let assemblyFormat = "`(`$size`)` attr-dict `:` type($result)";

    let builders = [
        OpBuilder<(ins "int64_t":$size)>
    ];

    let hasVerifier = 1;
}

def QTensor_DeallocOp : QTensorOp<"dealloc", [MemoryEffects<[MemFree]>]> {
    let summary = "Deallocate a tensor";
    let description = [{
        This operation deallocates the tensor and the values it holds, releasing its resources.
        Qubits that were extracted but not inserted again need to be deallocated separately.

        Example:
        ```mlir
        qtensor.dealloc %tensor : tensor<3x!qco.qubit>
        ```
    }];

    let arguments = (ins 1DTensorOf<[QubitType]>:$tensor);
    let assemblyFormat = "$tensor attr-dict `:` type($tensor)";

    let hasCanonicalizer = 1;
}

def QTensor_FromElementsOp : QTensorOp<"from_elements", [
    Pure,
    TypesMatchWith<"operand types match result element type",
                   "result", "elements", "SmallVector<Type, 2>("
                   "::llvm::cast<RankedTensorType>($_self).getNumElements(), "
                   "::llvm::cast<RankedTensorType>($_self).getElementType())">
  ]> {
    let summary = "Create a tensor from a range of values";
    let description = [{
        The `qtensor.from_elements` operation is a wrapper operation of the `tensor.from_elements` operation.
        This operation creates a one-dimensional qubit tensor using the operands as its input. The also causes
        the input qubits to be consumed.

        Example:
        ```mlir
        %tensor = qtensor.from_elements %q0, %q1, %q2 : tensor<3x!qco.qubit>
        ```
    }];

    let arguments = (ins Variadic<QubitType>:$elements);
    let results = (outs Static1DQubitTensor:$result);
    let assemblyFormat = "$elements attr-dict `:` type($result)";

    let builders = [
        // Special case builder for when `elements` has size >=1.
        OpBuilder<(ins "ValueRange":$elements)>
    ];
}

def QTensor_ExtractOp : QTensorOp<"extract", [
    Pure,
    TypesMatchWith<"result type matches element type of tensor",
                   "tensor", "result",
                   "::llvm::cast<TensorType>($_self).getElementType()">,
                   TypesMatchWith<
      "returned tensor type matches input tensor", "tensor", "out_tensor", "$_self">]> {
    let summary = "Extract element from tensor";
    let description = [{
        The `qtensor.extract` operation is the modified version of the standard `tensor.extract`
        operation of the tensor dialect. It reads a ranked tensor and returns one qubit from the input tensor
        at the given index. In addition, it also returns the updated input tensor as a result.

        Example:
        ```mlir
        %outTensor, %q0 = qtensor.extract %tensor[%c0] : tensor<4x!qco.qubit>
        ```
    }];

    let arguments = (ins 1DTensorOf<[QubitType]>:$tensor, Index:$index);
    let results = (outs 1DTensorOf<[QubitType]>:$out_tensor, QubitType:$result);
    let assemblyFormat = "$tensor `[` $index `]` attr-dict `:` type($tensor)";

    let hasFolder = 1;
    let hasVerifier = 1;
}

def QTensor_ExtractSliceOp : QTensorOp<"extract_slice", [
    Pure,
    TypesMatchWith<"returned tensor type matches input tensor",
    "tensor", "out_tensor", "$_self">
  ]> {
    let summary = "Extract slice from tensor";
    let description = [{
        The `qtensor.extract_slice` operation is the modified version of the standard `tensor.extract_slice`
        operation of the tensor dialect. It reads a one-dimensional qubit and returns the extracted tensor of qubits specified by the
        offset and size argument. In addition, it also returns the updated input tensor as result.

        The extract_slice operation supports the following arguments:

        - tensor: the "base" tensor from which to extract a slice.
        - offset: the starting position in the base tensor from which the slice is
                  extracted.
        - size: the length of the slice to extract from the base tensor.

        Example:
        ```mlir
        %outTensor, %extractedSlice = qtensor.extract_slice %tensor[%c0][%c2] : tensor<3x!qco.qubit> to tensor<2x!qco.qubit>
        ```
    }];

    let arguments = (ins
        1DTensorOf<[QubitType]>:$tensor,
        Index:$offset,
        Index:$size
    );
    let results = (outs 1DTensorOf<[QubitType]>:$out_tensor, 1DTensorOf<[QubitType]>:$result);

    let assemblyFormat = [{
        $tensor `[`$offset `]` `[`$size`]`
        attr-dict `:` type($tensor) `to` type($result)
    }];

    let builders = [
        OpBuilder<(ins "Value":$tensor, "Value":$offset,
          "Value":$size,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
      ];

    let hasFolder = 1;
    let hasVerifier = 1;
}

def QTensor_InsertOp : QTensorOp<"insert", [
    Pure,
    TypesMatchWith<"result type matches type of dest",
                   "dest", "result",
                   "$_self">,
    TypesMatchWith<"scalar type matches element type of dest",
                   "dest", "scalar",
                   "::llvm::cast<TensorType>($_self).getElementType()">]> {
    let summary = "Insert element into tensor";
    let description = [{
        The `qtensor.insert` operation is a wrapper operation for the `tensor.insert` operation of the tensor dialect.
        This operation inserts a qubit scalar into a ranked tensor as specified by the operation's index. The insertion
        consumes the input qubit.

        Example:
        ```mlir
        %newTensor = qtensor.insert %q0 into %tensor[%c0] : tensor<3x!qco.qubit>
        ```
    }];

    let arguments = (ins QubitType:$scalar,
                        1DTensorOf<[QubitType]>:$dest,
                        Index:$index);
    let results = (outs 1DTensorOf<[QubitType]>:$result);
    let assemblyFormat = [{
        $scalar `into` $dest `[` $index `]` attr-dict `:` type($dest)
    }];

    let hasFolder = 1;
    let hasVerifier = 1;
}

def QTensor_InsertSliceOp : QTensorOp<"insert_slice", [
    Pure,
    TypesMatchWith<"expected result type to match dest type",
                   "dest", "result", "$_self">
  ]> {
    let summary = "Insert slice into tensor";
    let description = [{
        The `qtensor.insert_slice` operation is a modified version of the `tensor.insert_slice` operation
        of the tensor dialect. The operation inserts a tensor `source` into another tensor `dest` as specified by the
        operation's offset and size arguments. This insertion consumes `source` tensor of qubits.

        The insert_slice operation supports the following arguments:

          - source: the tensor that is inserted. The source is consumed after the insertion.
          - dest: the tensor into which the source tensor is inserted.
          - offset: the starting index in the `dest` tensor where the slice is inserted.
          - size: the number of elements in the slice, which must match the size of the
                  source tensor type.

        Example:
        ```mlir
        %outTensor = qtensor.insert_slice %slicedTensor into %tensor[%c0][%c2] : tensor<2x!qco.qubit> into tensor<3x!qco.qubit>
        ```
    }];

    let arguments = (ins
        1DTensorOf<[QubitType]>:$source,
        1DTensorOf<[QubitType]>:$dest,
        Index:$offset,
        Index:$size
    );
    let results = (outs 1DTensorOf<[QubitType]>:$result);
    let assemblyFormat = [{
        $source `into` $dest `[`$offset `]` `[`$size`]`
        attr-dict `:` type($source) `into` type($dest)
    }];

    let hasFolder = 1;
    let hasVerifier = 1;
}

#endif // QTENSOROPS

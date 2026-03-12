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
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ParallelCombiningOpInterface.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def QTensorDialect : Dialect {
  let name = "qtensor";

  let summary = "The QTensor dialect for tensors with linear typing.";

  let description = [{
      The Qtensor dialect is an adjusted variant of the standard tensor dialect of MLIR that supports linear typing for tensor types.
      The extract operations of this dialect are modified so that they also return the updated tensor.
      In addition, alloc/dealloc operations are added to the dialect to support the bulk allocation and deallocation of tensors with linear types.
  }];

  let cppNamespace = "::mlir::qtensor";
}

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

class QTensorOp<string mnemonic, list<Trait> traits = []>
    : Op<QTensorDialect, mnemonic, traits>;

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
    let results = (outs AnyStaticShapeTensor:$result);
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

        Example:
        ```mlir
        qtensor.dealloc %tensor : tensor<3x!qco.qubit>
        ```
    }];

    let arguments = (ins AnyRankedTensor:$tensor);
    let assemblyFormat = "$tensor attr-dict `:` type($tensor)";

    let hasVerifier = 1;
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
        This operation creates a tensor using the operands as its values.

        Example:

    }];

    let arguments = (ins Variadic<AnyType>:$elements);
    let results = (outs AnyStaticShapeTensor:$result);
    let assemblyFormat = "$elements attr-dict `:` type($result)";

    let builders = [
        // Special case builder for when `elements` has size >=1.
        OpBuilder<(ins "ValueRange":$elements)>
    ];

    let hasVerifier = 1;
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
        operation of the tensor dialect. It reads a ranked tensor and returns one element from the input tensor
        at the given indices. In addition, it also returns the updated input tensor as a result.
        The indices must be of `index` type.

        Example:
        ```mlir
        %q0, %outTensor = qtensor.extract %tensor[%c0] : tensor<4x!qco.qubit>
        ```
    }];

    let arguments = (ins AnyRankedTensor:$tensor, Index:$index);
    let results = (outs AnyType:$result, AnyRankedTensor:$out_tensor);
    let assemblyFormat = "$tensor `[` $index `]` attr-dict `:` type($tensor)";

    let hasFolder = 1;
    let hasVerifier = 1;
}

def QTensor_ExtractSliceOp : QTensorOp<"extract_slice", [
    Pure,
    TypesMatchWith<"returned tensor type matches input tensor",
    "source", "out_source", "$_self">
  ]> {
    let summary = "Extract slice from tensor";
    let description = [{
        The `qtensor.extract_slice` operation is the modified version of the standard `tensor.extract_slice`
        operation of the tensor dialect. It reads a ranked tensor and returns the extracted tensor specified by the
        offset and size argument. In addition, it also returns the updated input tensor as result.

        The extract_slice operation supports the following arguments:

        * source: the "base" tensor from which to extract a slice.
        * offset: the starting position in the base tensor from which the slice is
                  extracted.
        * size: the length of the slice to extract from the base tensor.

        Example:
        ```mlir
        %extractedSlice, %outSourceTensor = qtensor.extract_slice %sourceTensor[%c0][%c2] : tensor<3x!qco.qubit> to tensor<2x!qco.qubit>
        ```
    }];

    let arguments = (ins
        AnyRankedTensor:$source,
        Index:$offset,
        Index:$size
    );
    let results = (outs AnyRankedTensor:$result, AnyRankedTensor:$out_source);

    let assemblyFormat = [{
        $source `[`$offset `]` `[`$size`]`
        attr-dict `:` type($source) `to` type($result)
    }];

    let builders = [
        OpBuilder<(ins "Value":$source, "Value":$offset,
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
        This operation inserts a scalar into a ranked tensor as specified by the operation`s index.

        Example:

        ```mlir
        %newTensor = qtensor.insert %q0 into %tensor[%c0] : tensor<3x!qco.qubit>
        ```
    }];

    let arguments = (ins AnyType:$scalar,
                        AnyRankedTensor:$dest,
                        Index:$index);
    let results = (outs AnyRankedTensor:$result);
    let assemblyFormat = [{
        $scalar `into` $dest `[` $index `]` attr-dict `:` type($dest)
    }];

    let hasVerifier = 1;
    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def QTensor_InsertSliceOp : QTensorOp<"insert_slice", [
    Pure,
    TypesMatchWith<"expected result type to match dest type",
                   "dest", "result", "$_self">
  ]> {
    let summary = "Insert slice into tensor";
    let description = [{
        The `qtensor.insert_slice` operation is a minimally adjusted version of the `tensor.insert_slice` operation
        of the tensor dialect. The operation inserts a tensor `source` into another tensor `dest` as specified by the
        operation`s offset and size argument.

        The insert_slice operation supports the following arguments:

          * source: the tensor that is inserted.
          * dest: the tensor into which the source tensor is inserted.
          * offset: the starting index in the `dest` tensor where the slice is inserted.
          * size: the number of elements in the slice, which must match the size of the
                  source tensor type.

        Example:
        ```mlir
        %outTensor = qtensor.insert_slice %slicedTensor into %tensor[%c0][%c2] : tensor<2x!qco.qubit> into tensor<3x!qco.qubit>
        ```
    }];

    let arguments = (ins
        AnyRankedTensor:$source,
        AnyRankedTensor:$dest,
        Index:$offset,
        Index:$size
    );
    let results = (outs AnyRankedTensor:$result);
    let assemblyFormat = [{
        $source `into` $dest `[`$offset `]` `[`$size`]`
        attr-dict `:` type($source) `into` type($dest)
    }];

    let builders = [
        OpBuilder<(ins "Value":$source,
          "Value":$offset, "Value":$size,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
    ];

    let hasCanonicalizer = 1;
    let hasFolder = 1;
    let hasVerifier = 1;
}

#endif // QTENSOROPS

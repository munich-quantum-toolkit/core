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

  let hasCanonicalizer = 1;

  let cppNamespace = "::mlir::qtensor";
}

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

class QTensorOp<string mnemonic, list<Trait> traits = []>
    : Op<QTensorDialect, mnemonic, traits>;

// Base class for ops with static/dynamic offset, sizes and strides
// attributes/arguments.
class QTensorOpWithOffsetSizesAndStrides<string mnemonic,
                                         list<Trait> traits = []>
    : QTensorOp<mnemonic, traits> {
  code extraBaseClassDeclaration = [{
    /// Return the type of the base tensor operand.
    ::mlir::RankedTensorType getSourceType() {
      return ::llvm::cast<RankedTensorType>(getSource().getType());
    }

    /// Return the type of the result tensor.
    ::mlir::RankedTensorType getResultType() {
      return ::llvm::cast<RankedTensorType>(getResult().getType());
    }

    /// Return the dynamic sizes for this subview operation if specified.
    ::mlir::Operation::operand_range getDynamicSizes() { return getSizes(); }

  }];
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def QTensor_AllocOp : QTensorOp<"alloc", [
    MemoryEffects<[MemAlloc]>]> {
    let summary = "Tensor alloc operation";
    let description = [{
        Allocates a qubit tensor with the given size and returns the allocated tensor.
        The qubits are initialized to the |0> state.

        Example :
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
    let summary = "tensor from elements operation";
    let description = [{
        The `qtensor.from_elements` is a wrapper operation for the `tensor.from_elements` operation
        of the tensor dialect. This operation creates a tensor using the operands as its values.

        Example:

        ```mlir
        %tensor = qtensor.from_elements %q0, %q1, %q2 :  tensor<3x!qco.qubit>
        ```
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

    let arguments = (ins AnyRankedTensor:$tensor, Variadic<Index>:$indices);
    let results = (outs AnyType:$result, AnyRankedTensor:$out_tensor);
    let assemblyFormat = "$tensor `[` $indices `]` attr-dict `:` type($tensor)";

    let hasFolder = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def QTensor_ExtractSliceOp : QTensorOpWithOffsetSizesAndStrides<"extract_slice", [
    AttrSizedOperandSegments,
    Pure,
    OffsetSizeAndStrideOpInterface,
    TypesMatchWith<"returned tensor type matches input tensor",
    "source", "out_source", "$_self">
  ]> {
    let summary = "Extract slice from tensor";
    let description = [{
        The `qtensor.extract_slice` operation is the modified version of the standard `tensor.extract_slice`
        operation of the tensor dialect. It reads a ranked tensor and returns the extracted tensor specified by the
        offsets, sizes and strides arguments. If the strides arguments are omitted, they are set to default values of 1.
        In addition, it also returns the updated input tensor as result.


        The extract_slice operation supports the following arguments:

        * source: the "base" tensor from which to extract a slice.
        * offsets: tensor-rank number of offsets into the "base" tensor from which
                   to extract the slice.
        * sizes: tensor-rank number of sizes which specify the sizes of the result
                 tensor type.
        * strides: tensor-rank number of strides specifying subsampling in each
                   dimension.

        The representation based on offsets, sizes and strides support a
        partially-static specification via attributes specified through the
        `static_offsets`, `static_sizes` and `static_strides` arguments. A special
        sentinel value ShapedType::kDynamic encodes that the corresponding entry has
        a dynamic value.

        Example:
        ```mlir
        %extractedSlice, %outSourceTensor = qtensor.extract_slice %sourceTensor[%c0][%c2][%c1] : tensor<3x!qco.qubit> to tensor<2x!qco.qubit>
        ```
    }];

    let arguments = (ins
        AnyRankedTensor:$source,
        Variadic<Index>:$offsets,
        Variadic<Index>:$sizes,
        Variadic<Index>:$strides,
        DenseI64ArrayAttr:$static_offsets,
        DenseI64ArrayAttr:$static_sizes,
        DenseI64ArrayAttr:$static_strides
    );
    let results = (outs AnyRankedTensor:$result, AnyRankedTensor:$out_source);

    let assemblyFormat = [{
        $source ``
        custom<DynamicIndexList>($offsets, $static_offsets)
        custom<DynamicIndexList>($sizes, $static_sizes)
        custom<DynamicIndexList>($strides, $static_strides)
        attr-dict `:` type($source) `to` type($result)
    }];

    let builders = [
        // Build an ExtractSliceOp with mixed static and dynamic entries and
        // inferred result type.
        OpBuilder<(ins "Value":$source, "ArrayRef<OpFoldResult>":$offsets,
          "ArrayRef<OpFoldResult>":$sizes, CArg<"ArrayRef<OpFoldResult>", "{}">:$strides,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
        // Build an ExtractSliceOp with mixed static and dynamic entries and custom
        // result type. If the type passed is nullptr, it is inferred.
        OpBuilder<(ins "RankedTensorType":$resultType, "RankedTensorType":$outSourceType, "Value":$source,
          "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
          CArg<"ArrayRef<OpFoldResult>", "{}">:$strides,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
        OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
          "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
          CArg<"ArrayRef<OpFoldResult>", "{}">:$strides,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
        // Build an ExtractSliceOp with dynamic entries and custom result type. If
        // the type passed is nullptr, it is inferred.
        OpBuilder<(ins "Value":$source, "ValueRange":$offsets,
          "ValueRange":$sizes, CArg<"ValueRange", "{}">:$strides,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
        // Build an ExtractSliceOp with dynamic entries and inferred result type.
        OpBuilder<(ins "RankedTensorType":$resultType, "RankedTensorType":$outSourceType, "Value":$source,
          "ValueRange":$offsets, "ValueRange":$sizes, CArg<"ValueRange", "{}">:$strides,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
        OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
          "ValueRange":$offsets, "ValueRange":$sizes, CArg<"ValueRange", "{}">:$strides,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
        // Build an ExtractSliceOp with mixed static and dynamic entries packed in
        // a Range vector.
        OpBuilder<(ins "Value":$source, "ArrayRef<Range>":$ranges,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
        // Build an ExtractSliceOp with mixed static and dynamic sizes, inferred
        // result type, offsets set to 0 and strides set to 1.
        OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
          "ArrayRef<OpFoldResult>":$sizes,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
      ];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        /// The result of an extract_slice is always a tensor.
        RankedTensorType getType() {
          return getResultType();
        }

        /// Compute the rank-reduction mask that can be applied to map the source
        /// tensor type to the result tensor type by dropping unit dims.
        std::optional<llvm::SmallDenseSet<unsigned>>
        computeRankReductionMask() {
          return ::mlir::computeRankReductionMask(getSourceType().getShape(),
                                                  getType().getShape());
        };

        /// An extract_slice result type can be inferred, when it is not
        /// rank-reduced, from the source type and the static representation of
        /// sizes. Special sentinels encode the dynamic case.
        static RankedTensorType inferResultType(
          RankedTensorType sourceTensorType,
          ArrayRef<int64_t> staticSizes);
        static RankedTensorType inferResultType(
          RankedTensorType sourceTensorType,
          ArrayRef<OpFoldResult> staticSizes);

        /// If the rank is reduced (i.e. the desiredResultRank is smaller than the
        /// number of sizes), drop as many size 1 as needed to produce an inferred type
        /// with the desired rank.
        ///
        /// Note that there may be multiple ways to compute this rank-reduced type:
        ///   e.g. 1x6x1 can rank-reduce to either 1x6 or 6x1 2-D tensors.
        ///
        /// To disambiguate, this function always drops the first 1 sizes occurrences.
        static RankedTensorType inferCanonicalRankReducedResultType(
          unsigned resultRank,
          RankedTensorType sourceRankedTensorType,
          ArrayRef<int64_t> staticSizes);
        static RankedTensorType inferCanonicalRankReducedResultType(
          unsigned resultRank,
          RankedTensorType sourceRankedTensorType,
          ArrayRef<OpFoldResult> staticSizes);

        /// Return the expected rank of each of the`static_offsets`, `static_sizes`
        /// and `static_strides` attributes.
        std::array<unsigned, 3> getArrayAttrMaxRanks() {
          unsigned rank = getSourceType().getRank();
          return {rank, rank, rank};
        }

        /// Return the number of leading operands before the `offsets`, `sizes` and
        /// and `strides` operands.
        static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

        /// Return the dimensions of the source that are dropped in the
        /// result when the result is rank-reduced.
        llvm::SmallBitVector getDroppedDims();

        /// Given a `value`, asserted to be of RankedTensorType, build an
        /// ExtractSliceOp that results in a rank-reducing extract to the desired
        /// tensor shape and return the new value created.
        /// If the shape of `value` is already the `desiredShape`, just return
        /// `value`.
        /// If the shape of `value` cannot be rank-reduced to `desiredShape`, fail.
        static FailureOr<Value> rankReduceIfNeeded(
          OpBuilder &b, Location loc, Value value, ArrayRef<int64_t> desiredShape);
    }];

    let hasCanonicalizer = 1;
    let hasFolder = 1;
    let hasVerifier = 1;
}

def QTensor_InsertOp : QTensorOp<"insert", [
    DestinationStyleOpInterface,
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
        This operation inserts a scalar into a ranked tensor as specified by the operation`s indices.

        Example:

        ```mlir
        %newTensor = qtensor.insert %q0 into %tensor[%c0] : tensor<3x!qco.qubit>
        ```
    }];

    let arguments = (ins AnyType:$scalar,
                        AnyRankedTensor:$dest,
                        Variadic<Index>:$indices);
    let results = (outs AnyRankedTensor:$result);
    let assemblyFormat = [{
        $scalar `into` $dest `[` $indices `]` attr-dict `:` type($dest)
    }];

    let extraClassDeclaration = [{
        MutableOperandRange getDpsInitsMutable() { return getDestMutable(); }
    }];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def QTensor_InsertSliceOp : QTensorOpWithOffsetSizesAndStrides<"insert_slice", [
    AttrSizedOperandSegments,
    DestinationStyleOpInterface,
    Pure,
    OffsetSizeAndStrideOpInterface,
    TypesMatchWith<"expected result type to match dest type",
                   "dest", "result", "$_self">
  ]> {
    let summary = "Insert slice into tensor";
    let description = [{
        The `qtensor.insert_slice` operation is a minimally adjusted version of the `tensor.insert_slice` operation
        of the tensor dialect. The operation inserts a tensor `source` into another tensor `dest` as specified by the
        operation`s offset, sizes and strides arguments. If the strides arguments are omitted, they are set to default values of 1.

        The insert_slice operation supports the following arguments:

          * source: the tensor that is inserted.
          * dest: the tensor into which the source tensor is inserted.
          * offsets: tensor-rank number of offsets into the `dest` tensor into which
                    the slice is inserted.
          * sizes: tensor-rank number of sizes which specify the sizes of the source
                  tensor type.
          * strides: tensor-rank number of strides that specify subsampling in each
                    dimension.

        The representation based on offsets, sizes and strides support a
        partially-static specification via attributes specified through the
        `static_offsets`, `static_sizes` and `static_strides` arguments. A special
        sentinel value ShapedType::kDynamic encodes that the corresponding entry has
        a dynamic value.

        Example:
        ```mlir
        %outTensor = qtensor.insert_slice %slicedTensor into %tensor[%c0][%c2][%c1] : tensor<2x!qco.qubit> into tensor<3x!qco.qubit>
        ```
    }];

    let arguments = (ins
        AnyRankedTensor:$source,
        AnyRankedTensor:$dest,
        Variadic<Index>:$offsets,
        Variadic<Index>:$sizes,
        Variadic<Index>:$strides,
        DenseI64ArrayAttr:$static_offsets,
        DenseI64ArrayAttr:$static_sizes,
        DenseI64ArrayAttr:$static_strides
    );
    let results = (outs AnyRankedTensor:$result);
    let assemblyFormat = [{
        $source `into` $dest ``
        custom<DynamicIndexList>($offsets, $static_offsets)
        custom<DynamicIndexList>($sizes, $static_sizes)
        custom<DynamicIndexList>($strides, $static_strides)
        attr-dict `:` type($source) `into` type($dest)
    }];

    let builders = [
        // Build a InsertSliceOp with mixed static and dynamic entries and inferred
        // result type.
        OpBuilder<(ins "Value":$source, "Value":$dest,
          "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
          CArg<"ArrayRef<OpFoldResult>", "{}">:$strides,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
        // Build a InsertSliceOp with dynamic entries and inferred result type.
        OpBuilder<(ins "Value":$source, "Value":$dest,
          "ValueRange":$offsets, "ValueRange":$sizes, CArg<"ValueRange", "{}">:$strides,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
        // Build an InsertSliceOp with mixed static and dynamic entries packed in
        // a Range vector and inferred result type.
        OpBuilder<(ins "Value":$source, "Value":$dest,
          "ArrayRef<Range>":$ranges,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
    ];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        /// The result of a insert_slice is always a tensor.
        RankedTensorType getType() {
          return getResultType();
        }

        /// The `dest` type is the same as the result type.
        RankedTensorType getDestType() {
          return getResultType();
        }

        /// Return the expected rank of each of the`static_offsets`, `static_sizes`
        /// and `static_strides` attributes.
        std::array<unsigned, 3> getArrayAttrMaxRanks() {
          unsigned rank = getResultType().getRank();
          return {rank, rank, rank};
        }

        /// Return the dimensions of the dest that are omitted to insert a source
        /// when the result is rank-extended.
        llvm::SmallBitVector getDroppedDims();

        /// Return the number of leading operands before the `offsets`, `sizes` and
        /// and `strides` operands.
        static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }

        MutableOperandRange getDpsInitsMutable() { return getDestMutable(); }
    }];

    let hasCanonicalizer = 1;
    let hasFolder = 1;
    let hasVerifier = 1;
}

#endif // QTENSOROPS

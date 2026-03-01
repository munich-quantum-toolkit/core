// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

//===- QTensorOps.td - QTensor op definitions ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QTENSOROPS
#define QTENSOROPS

include "mlir/Dialect/QTensor/IR/QTensorBase.td"
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
def QTensor_ExtractOp : QTensorOp<"extract", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure,
    TypesMatchWith<"result type matches element type of tensor",
                   "tensor", "result",
                   "::llvm::cast<TensorType>($_self).getElementType()">,
                   TypesMatchWith<
      "returned tensor type matches input tensor",
      "tensor", "out_tensor", "$_self">]> {
  let summary = "element extraction operation";
  let description = [{
    The `tensor.extract` op reads a ranked tensor and returns one element as
    specified by the given indices. The result of the op is a value with the
    same type as the elements of the tensor. The arity of indices must match
    the rank of the accessed value. All indices should all be of `index` type.

    Example:

    ```mlir
    %4 = tensor.extract %t[%1, %2] : tensor<4x4xi32>
    %5 = tensor.extract %rt[%1, %2] : tensor<?x?xi32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$tensor, Variadic<Index>:$indices);
  let results = (outs AnyType:$result, AnyRankedTensor:$out_tensor);
  let assemblyFormat = "$tensor `[` $indices `]` attr-dict `:` type($tensor)";

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}


//===----------------------------------------------------------------------===//
// ExtractSliceOp
//===----------------------------------------------------------------------===//

def QTensor_ExtractSliceOp : QTensorOpWithOffsetSizesAndStrides<"extract_slice", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
    AttrSizedOperandSegments,
    Pure,
    OffsetSizeAndStrideOpInterface,
    TypesMatchWith<"returned tensor type matches input tensor",
    "source", "out_source", "$_self">
  ]> {
  let summary = "extract slice operation";
  let description = [{

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
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build an ExtractSliceOp with mixed static and dynamic entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins "RankedTensorType":$resultType, "RankedTensorType":$outSourceType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build an ExtractSliceOp with dynamic entries and custom result type. If
    // the type passed is nullptr, it is inferred.
    OpBuilder<(ins "Value":$source, "ValueRange":$offsets,
      "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build an ExtractSliceOp with dynamic entries and inferred result type.
    OpBuilder<(ins "RankedTensorType":$resultType, "RankedTensorType":$outSourceType, "Value":$source,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
      OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build an ExtractSliceOp with mixed static and dynamic entries packed in
    // a Range vector.
    OpBuilder<(ins "Value":$source, "ArrayRef<Range>":$ranges,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// The result of an extract_slice is always a tensor.
    // TODO: deprecate
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
    /// offsets, sizes and strides. Special sentinels encode the dynamic case.
    static RankedTensorType inferResultType(
      RankedTensorType sourceTensorType,
      ArrayRef<int64_t> staticOffsets,
      ArrayRef<int64_t> staticSizes,
      ArrayRef<int64_t> staticStrides);
    static RankedTensorType inferResultType(
      RankedTensorType sourceTensorType,
      ArrayRef<OpFoldResult> staticOffsets,
      ArrayRef<OpFoldResult> staticSizes,
      ArrayRef<OpFoldResult> staticStrides);

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
      ArrayRef<int64_t> staticOffsets,
      ArrayRef<int64_t> staticSizes,
      ArrayRef<int64_t> staticStrides);
    static RankedTensorType inferCanonicalRankReducedResultType(
      unsigned resultRank,
      RankedTensorType sourceRankedTensorType,
      ArrayRef<OpFoldResult> staticOffsets,
      ArrayRef<OpFoldResult> staticSizes,
      ArrayRef<OpFoldResult> staticStrides);

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

def QTensor_FromElementsOp : QTensorOp<"from_elements", [
    Pure,
    TypesMatchWith<"operand types match result element type",
                   "result", "elements", "SmallVector<Type, 2>("
                   "::llvm::cast<RankedTensorType>($_self).getNumElements(), "
                   "::llvm::cast<RankedTensorType>($_self).getElementType())">
  ]> {
  let summary = "tensor from elements operation.";
  let description = [{
    Create a N-D tensor from a range of same-type arguments. The number of
    provided `elements` should equal to the number of the elements in the
    result type. The `elements` correspond to a flattened tensor.

    Example:

    ```mlir
    tensor.from_elements %a, %b, %c, %d, %e, %f :  tensor<2x3xindex>
    ```

    will result in a tensor

    [[%a, %b, %c]
     [%d, %e, %f]]
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs AnyStaticShapeTensor:$result);

  let assemblyFormat = "$elements attr-dict `:` type($result)";

  let builders = [
    // Special case builder for when `elements` has size >=1.
    OpBuilder<(ins "ValueRange":$elements)>
  ];

  let hasCanonicalizer = 1;
}

def QTensor_InsertOp : QTensorOp<"insert", [
    Pure,
    TypesMatchWith<"result type matches type of dest",
                   "dest", "result",
                   "$_self">,
    TypesMatchWith<"scalar type matches element type of dest",
                   "dest", "scalar",
                   "::llvm::cast<TensorType>($_self).getElementType()">]> {
  let summary = "element insertion operation";
  let description = [{
  }];

  let arguments = (ins AnyType:$scalar,
                       AnyRankedTensor:$dest,
                       Variadic<Index>:$indices);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $scalar `into` $dest `[` $indices `]` attr-dict `:` type($dest)
  }];

  let hasCanonicalizer = 1;
}

def QTensor_InsertSliceOp : QTensorOp<"insert_slice", [
    AttrSizedOperandSegments,
    Pure,
    TypesMatchWith<"expected result type to match dest type",
                   "dest", "result", "$_self">
  ]> {
  let summary = "insert_slice operation";
  let description = [{
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
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a InsertSliceOp with dynamic entries and inferred result type.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build an InsertSliceOp with mixed static and dynamic entries packed in
    // a Range vector and inferred result type.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ArrayRef<Range>":$ranges,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build an InsertSliceOp with mixed static and dynamic sizes, offsets set
    // to 0, strides set to 1 and inferred result type.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ArrayRef<OpFoldResult>":$sizes,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let hasCanonicalizer = 1;
}

def QTensor_DeallocOp : QTensorOp<"dealloc", [MemoryEffects<[MemFree]>]> {
    let summary = "Deallocate a tensor";
    let description = [{
    }];

    let arguments = (ins AnyRankedTensor:$tensor);
    let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}
#endif // TENSOR_OPS

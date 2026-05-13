/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/QIR/Utils/QIRMetadata.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {

//===----------------------------------------------------------------------===//
// Allocation mode
//===----------------------------------------------------------------------===//

enum class AllocationMode : std::uint8_t { Unset, Static, Dynamic };

//===----------------------------------------------------------------------===//
// LoweringState
//===----------------------------------------------------------------------===//

struct LoweringState : qir::QIRMetadata {
  llvm::DenseMap<int64_t, Value> staticQubits;
  llvm::DenseMap<Value, Value> memrefSizes;
  llvm::StringMap<Value> resultArrays;
  llvm::DenseMap<std::pair<StringRef, int64_t>, Value> loadedResults;
  llvm::DenseMap<int64_t, Value> resultPtrs;

  int64_t inCtrlOp = 0;
  llvm::DenseMap<int64_t, SmallVector<Value>> controls;

  llvm::BumpPtrAllocator allocator;
  llvm::StringSaver stringSaver{allocator};

  Block* entryBlock = nullptr;
  Block* measurementsBlock = nullptr;
  Block* outputBlock = nullptr;

  AllocationMode allocationMode = AllocationMode::Unset;

  [[nodiscard]] LogicalResult ensureAllocationMode(AllocationMode requested,
                                                   Operation* op);
};

//===----------------------------------------------------------------------===//
// QCToQIRTypeConverter
//===----------------------------------------------------------------------===//

struct QCToQIRTypeConverter final : LLVMTypeConverter {
  explicit QCToQIRTypeConverter(MLIRContext* ctx);
};

//===----------------------------------------------------------------------===//
// StatefulOpConversionPattern
//===----------------------------------------------------------------------===//

template <typename OpType>
class StatefulOpConversionPattern : public OpConversionPattern<OpType> {
public:
  StatefulOpConversionPattern(TypeConverter& tc, MLIRContext* ctx,
                              LoweringState* state)
      : OpConversionPattern<OpType>(tc, ctx), state_(state) {}

  [[nodiscard]] LoweringState& getState() const { return *state_; }

private:
  LoweringState* state_;
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void addInitialize(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                   LoweringState& state);

void populateQCToQIRPatterns(RewritePatternSet& patterns,
                             QCToQIRTypeConverter& typeConverter,
                             MLIRContext* ctx, LoweringState& state);

void addOutputRecording(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                        LoweringState& state);

} // namespace mlir

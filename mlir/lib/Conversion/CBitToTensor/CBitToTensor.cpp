/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/CBitToTensor/CBitToTensor.h"

#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstdint>

namespace mlir::cbit {

Value CBitToTensorState::resolveRegister(Value regOrAlias) const {
  const auto it = registerAliases.find(regOrAlias);
  return it != registerAliases.end() ? it->second : regOrAlias;
}

Value CBitToTensorState::findRegister(Value value) const {
  value = resolveRegister(value);
  return isa<RegisterType>(value.getType()) ? value : Value{};
}

Value CBitToTensorState::getCurrentRegister(Value reg,
                                            Operation* anchor) const {
  for (auto* region = anchor->getParentRegion(); region != nullptr;
       region = region->getParentRegion()) {
    const auto regionIt = registerTensors.find(region);
    if (regionIt == registerTensors.end()) {
      continue;
    }
    if (const auto valueIt = regionIt->second.find(reg);
        valueIt != regionIt->second.end()) {
      return valueIt->second;
    }
  }
  return nullptr;
}

void CBitToTensorState::setCurrentRegister(Value reg, Value tensor,
                                           Operation* anchor) {
  registerTensors[anchor->getParentRegion()][reg] = tensor;
}

void CBitToTensorState::setCurrentRegister(Value reg, Value tensor,
                                           Region* region) {
  registerTensors[region][reg] = tensor;
}

void CBitToTensorState::addRegisterAlias(Value tensor, Value reg) {
  registerAliases[tensor] = reg;
}

Value CBitToTensorState::getRegisterForAlias(Value tensor) const {
  const auto it = registerAliases.find(tensor);
  return it != registerAliases.end() ? it->second : Value{};
}

DenseMap<Value, Value>* CBitToTensorState::getRegionRegisters(Region* region) {
  const auto it = registerTensors.find(region);
  return it != registerTensors.end() ? &it->second : nullptr;
}

void CBitToTensorState::recordRegisterUses(Operation* root) {
  root->walk([&](Operation* operation) {
    if (isa<LoadOp>(operation)) {
      operationRegisters[operation] = operation->getOperand(0);
    } else if (isa<StoreOp>(operation)) {
      operationRegisters[operation] = operation->getOperand(1);
    }
  });
}

Value CBitToTensorState::getRecordedRegister(Operation* operation) const {
  const auto it = operationRegisters.find(operation);
  return it != operationRegisters.end() ? it->second : Value{};
}

void addCBitToTensorTypeConversion(TypeConverter& typeConverter) {
  typeConverter.addConversion([](RegisterType type) -> Type {
    return RankedTensorType::get({type.getWidth()},
                                 IntegerType::get(type.getContext(), 1));
  });
}

namespace {

template <typename OpType>
class CBitToTensorPattern : public OpConversionPattern<OpType> {
public:
  CBitToTensorPattern(TypeConverter& typeConverter, MLIRContext* context,
                      CBitToTensorState* state)
      : OpConversionPattern<OpType>(typeConverter, context), state(state) {}

protected:
  [[nodiscard]] CBitToTensorState& getState() const { return *state; }

private:
  CBitToTensorState* state;
};

struct ConvertAllocOp final : CBitToTensorPattern<AllocOp> {
  using CBitToTensorPattern::CBitToTensorPattern;

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto registerType = op.getResult().getType();
    const auto tensorType =
        RankedTensorType::get({registerType.getWidth()}, rewriter.getI1Type());
    auto tensor =
        tensor::EmptyOp::create(rewriter, op.getLoc(), tensorType, ValueRange{})
            .getResult();
    if (op.getInitialization() == Initialization::Zero) {
      auto zero = arith::ConstantOp::create(rewriter, op.getLoc(),
                                            rewriter.getBoolAttr(false));
      for (int64_t index = 0; index < registerType.getWidth(); ++index) {
        auto indexValue =
            arith::ConstantIndexOp::create(rewriter, op.getLoc(), index);
        tensor = tensor::InsertOp::create(rewriter, op.getLoc(), zero, tensor,
                                          indexValue.getResult())
                     .getResult();
      }
    }

    auto& state = getState();
    state.setCurrentRegister(op.getResult(), tensor, op);
    state.addRegisterAlias(tensor, op.getResult());
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

struct ConvertStoreOp final : CBitToTensorPattern<StoreOp> {
  using CBitToTensorPattern::CBitToTensorPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto reg = state.getRecordedRegister(op);
    if (!reg) {
      reg = state.resolveRegister(op->getOperand(1));
    }
    auto tensor = state.getCurrentRegister(reg, op);
    if (!tensor) {
      return rewriter.notifyMatchFailure(op, "unknown CBit register");
    }
    tensor = rewriter.getRemappedValue(tensor);
    auto updated = tensor::InsertOp::create(
        rewriter, op.getLoc(), adaptor.getValue(), tensor, adaptor.getIndex());
    state.setCurrentRegister(reg, updated, op);
    state.addRegisterAlias(updated, reg);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertLoadOp final : CBitToTensorPattern<LoadOp> {
  using CBitToTensorPattern::CBitToTensorPattern;

  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto reg = state.getRecordedRegister(op);
    if (!reg) {
      reg = state.resolveRegister(op->getOperand(0));
    }
    auto tensor = state.getCurrentRegister(reg, op);
    if (!tensor) {
      return rewriter.notifyMatchFailure(op, "unknown CBit register");
    }
    tensor = rewriter.getRemappedValue(tensor);
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, tensor,
                                                   adaptor.getIndex());
    return success();
  }
};

} // namespace

void populateCBitToTensorConversionPatterns(TypeConverter& typeConverter,
                                            RewritePatternSet& patterns,
                                            CBitToTensorState& state) {
  patterns.add<ConvertAllocOp, ConvertLoadOp, ConvertStoreOp>(
      typeConverter, patterns.getContext(), &state);
}

} // namespace mlir::cbit

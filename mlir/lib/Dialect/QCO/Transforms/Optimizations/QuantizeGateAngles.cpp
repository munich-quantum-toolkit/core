/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/Utils/AngleConversion.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <cmath>
#include <cstdint>
#include <optional>

namespace mlir::qco {

#define GEN_PASS_DEF_QUANTIZEGATEANGLES
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

[[nodiscard]] static std::optional<uint32_t>
precisionWidth(const IntegerAttr precision) {
  const auto& value = precision.getValue();
  if (value.isZero() || value.ugt(mqt::angle::MACHINE_WIDTH)) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(value.getZExtValue());
}

namespace {

struct QuantizeGateAngles final
    : impl::QuantizeGateAnglesBase<QuantizeGateAngles> {
  using QuantizeGateAnglesBase::QuantizeGateAnglesBase;

protected:
  void runOnOperation() override {
    if (!mqt::angle::isSupportedWidth(precisionBits)) {
      getOperation().emitError() << "precision-bits must be between 1 and "
                                 << mqt::angle::MACHINE_WIDTH;
      signalPassFailure();
      return;
    }

    if (const auto existing =
            getOperation()->getAttr(mqt::angle::FINAL_QUANTIZATION_ATTR)) {
      const auto integer = dyn_cast<IntegerAttr>(existing);
      if (!integer || !precisionWidth(integer)) {
        getOperation().emitError()
            << "invalid final gate-angle precision metadata";
        signalPassFailure();
        return;
      }
    }

    const auto preflight = getOperation().walk([&](UnitaryOpInterface unitary) {
      if (isa<PowOp>(unitary.getOperation())) {
        return WalkResult::advance();
      }
      for (const auto parameter : unitary.getParameters()) {
        const auto constant = utils::valueToConstantDouble(parameter);
        if (constant && !std::isfinite(*constant)) {
          unitary.emitError() << "cannot quantize a non-finite gate parameter";
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (preflight.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    DenseMap<Block*, DenseMap<Value, Value>> blockConversions;
    getOperation().walk([&](UnitaryOpInterface unitary) {
      if (isa<PowOp>(unitary.getOperation())) {
        return;
      }
      const auto parameters = unitary.getParameters();
      if (parameters.empty()) {
        return;
      }
      rewriter.setInsertionPoint(unitary);
      const auto firstParameter = parameters.getBeginOperandIndex();
      for (const auto [index, parameter] : llvm::enumerate(parameters)) {
        const auto existing = mqt::angle::matchQuantizedRadians(parameter);
        if (existing && existing->bitWidth == precisionBits) {
          continue;
        }
        auto& conversions = blockConversions[unitary->getBlock()];
        Value quantized;
        if (const auto found = conversions.find(parameter);
            found != conversions.end()) {
          quantized = found->second;
        } else {
          quantized = mqt::angle::buildQuantizedRadians(
              rewriter, unitary.getLoc(), parameter, precisionBits);
          conversions.try_emplace(parameter, quantized);
        }
        unitary->setOperand(firstParameter + index, quantized);
      }
    });
    getOperation()->setAttr(
        mqt::angle::FINAL_QUANTIZATION_ATTR,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(precisionBits)));
  }
};

} // namespace

} // namespace mlir::qco

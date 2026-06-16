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

#include <mlir/IR/Operation.h>
#ifndef MQT_CORE_CLASSICALARITHOPERATION_H
#define MQT_CORE_CLASSICALARITHOPERATION_H

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"

#include <mlir/Dialect/Arith/IR/Arith.h>

#include <cmath>
#include <stdexcept>

/**
 * This file provides information of available arith operations. It calculates
 * the result of valid arith operations. Operations are only valid with one to
 * two operands, not if they are applied to sequences.
 */
inline int64_t getArithOpResult(mlir::Operation* operation, int64_t value1,
                                int64_t value2 = 0, int64_t value3 = 0) {

  for (mlir::Value operand : operation->getOperands()) {
    if (isa<mlir::VectorType>(operand.getType())) {
      throw std::runtime_error(
          "Constant propagation does not support vectors as classical types.");
    }
  }

  return mlir::TypeSwitch<mlir::Operation*, int64_t>(operation)
      .Case<mlir::arith::AddIOp>([&](auto) { return value1 + value2; })
      .Case<mlir::arith::AndIOp>([&](auto) { return value1 & value2; })
      .Case<mlir::arith::CeilDivSIOp>([&](auto) {
        // Division that rounds to positive infinity
        return ceil(1.0 * value1 / value2);
      })
      .Case<mlir::arith::DivSIOp>([&](auto) {
        // Division that rounds towards zero
        return value1 / value2;
      })
      .Case<mlir::arith::FloorDivSIOp>([&](auto) {
        // Division that rounds to negative infinity
        return floor(1.0 * value1 / value2);
      })
      .Case<mlir::arith::MaxSIOp>(
          [&](auto) { return value1 > value2 ? value1 : value2; })
      .Case<mlir::arith::MinSIOp>(
          [&](auto) { return value1 < value2 ? value1 : value2; })
      .Case<mlir::arith::MulIOp>([&](auto) { return value1 * value2; })
      .Case<mlir::arith::OrIOp>([&](auto) { return value1 | value2; })
      .Case<mlir::arith::RemSIOp>(
          [&](auto) { return remainder(value1, value2); })
      .Case<mlir::arith::ShLIOp>([&](auto) { return value1 << value2; })
      .Case<mlir::arith::ShRSIOp>([&](auto) { return value1 >> value2; })
      .Case<mlir::arith::SubIOp>([&](auto) { return value1 - value2; })
      .Case<mlir::arith::XOrIOp>([&](auto) { return value1 ^ value2; })
      .Case<mlir::arith::SelectOp>(
          [&](auto) { return value3 == 0 ? value3 : value2; })
      .Default([&](auto) -> int64_t {
        throw std::runtime_error("Unsupported integer operation in "
                                 "mlir::qco::classicalarithoperation");
      });
}

inline double getArithOpResult(mlir::Operation* operation, double value1,
                               double value2 = 0.0) {

  for (mlir::Value operand : operation->getOperands()) {
    if (isa<mlir::VectorType>(operand.getType())) {
      throw std::runtime_error(
          "Constant propagation does not support vectors as classical types.");
    }
  }

  return mlir::TypeSwitch<mlir::Operation*, double>(operation)
      .Case<mlir::arith::AddFOp>([&](auto) { return value1 + value2; })
      .Case<mlir::arith::DivFOp>([&](auto) { return value1 / value2; })
      .Case<mlir::arith::MaximumFOp>(
          [&](auto) { return value1 > value2 ? value1 : value2; })
      .Case<mlir::arith::MaxNumFOp>(
          [&](auto) { return value1 > value2 ? value1 : value2; })
      .Case<mlir::arith::MinimumFOp>(
          [&](auto) { return value1 < value2 ? value1 : value2; })
      .Case<mlir::arith::MinNumFOp>(
          [&](auto) { return value1 < value2 ? value1 : value2; })
      .Case<mlir::arith::MulFOp>([&](auto) { return value1 * value2; })
      .Case<mlir::arith::NegFOp>([&](auto) { return -value1; })
      .Case<mlir::arith::RemFOp>(
          [&](auto) { return remainder(value1, value2); })
      .Case<mlir::arith::SubFOp>([&](auto) { return value1 - value2; })
      .Default([&](auto) -> double {
        throw std::runtime_error("Unsupported floating-point operation in "
                                 "mlir::qco::classicalarithoperation");
      });
}

#endif // MQT_CORE_MQT_CORE_CLASSICALARITHOPERATION_H

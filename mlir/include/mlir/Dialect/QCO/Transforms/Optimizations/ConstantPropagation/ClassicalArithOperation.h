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

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <cstdint>

/**
 * This file provides information of available arith operations. It calculates
 * the result of valid arith operations. Operations are only valid with one to
 * two operands, not if they are applied to sequences.
 */
inline int64_t getArithIntegerOpResult(mlir::Operation* operation,
                                       const int64_t value1,
                                       const int64_t value2 = 0,
                                       const int64_t value3 = 0) {

  for (auto operand : operation->getOperands()) {
    if (mlir::isa<mlir::VectorType>(operand.getType())) {
      llvm::report_fatal_error(
          "Constant propagation does not support vectors as classical types.");
    }
  }

  const auto intTy =
      dyn_cast<mlir::IntegerType>(operation->getResult(0).getType());
  if (!intTy) {
    llvm::report_fatal_error(
        "IntegerType is needed to apply arith::integer operation.");
  }

  const unsigned width = intTy.getWidth();
  if (width > 64) {
    llvm::report_fatal_error(
        "Result of an arith operation cannot be safely stored in a 64-bit "
        "integer, which is required by constant propagation.");
  }

  // APInt respects the signedness of the underlying MLIR type.
  const bool isSigned = intTy.isSigned();
  const llvm::APInt a(width, static_cast<uint64_t>(value1), isSigned);
  const llvm::APInt b(width, static_cast<uint64_t>(value2), isSigned);
  const llvm::APInt c(width, static_cast<uint64_t>(value3), isSigned);

  return mlir::TypeSwitch<mlir::Operation*, int64_t>(operation)
      .Case<mlir::arith::AddIOp>([&](auto) { return (a + b).getSExtValue(); })
      .Case<mlir::arith::AndIOp>([&](auto) { return (a & b).getSExtValue(); })
      .Case<mlir::arith::CmpIOp>([&](mlir::arith::CmpIOp cmp) {
        using Pred = mlir::arith::CmpIPredicate;
        bool result = false;
        switch (cmp.getPredicate()) {
        case Pred::eq:
          result = (a == b);
          break;
        case Pred::ne:
          result = (a != b);
          break;
        case Pred::slt:
          result = a.slt(b);
          break;
        case Pred::sle:
          result = a.sle(b);
          break;
        case Pred::sgt:
          result = a.sgt(b);
          break;
        case Pred::sge:
          result = a.sge(b);
          break;
        case Pred::ult:
          result = a.ult(b);
          break;
        case Pred::ule:
          result = a.ule(b);
          break;
        case Pred::ugt:
          result = a.ugt(b);
          break;
        case Pred::uge:
          result = a.uge(b);
          break;
        }

        return result ? int64_t{1} : int64_t{0};
      })
      .Case<mlir::arith::DivSIOp>([&](auto) {
        if (a.isZero()) {
          llvm::report_fatal_error(
              "Handling of division by zero in mlir::arith::DivSIOp is not "
              "supported by constant propagation.");
        }
        if (a.isAllOnes() && b.isMinSignedValue()) {
          llvm::report_fatal_error(
              "Handling of INT_MIN / -1 in mlir::arith::DivSIOp is not "
              "supported by constant propagation.");
        }
        return a.sdiv(b).getSExtValue();
      })
      .Case<mlir::arith::MaxSIOp>([&](auto) {
        return a.getSExtValue() > b.getSExtValue() ? a.getSExtValue()
                                                   : b.getSExtValue();
      })
      .Case<mlir::arith::MinSIOp>([&](auto) {
        return a.getSExtValue() < b.getSExtValue() ? a.getSExtValue()
                                                   : b.getSExtValue();
      })
      .Case<mlir::arith::MulIOp>([&](auto) { return (a * b).getSExtValue(); })
      .Case<mlir::arith::OrIOp>([&](auto) { return (a | b).getSExtValue(); })
      .Case<mlir::arith::SubIOp>([&](auto) { return (a - b).getSExtValue(); })
      .Case<mlir::arith::XOrIOp>([&](auto) { return (a ^ b).getSExtValue(); })
      .Case<mlir::arith::SelectOp>([&](auto) {
        // SelectOp: first operand is the i1 condition.
        // In our helper `value1` is the condition (0 == false).
        return (a != llvm::APInt(width, 0, true)) ? b.getSExtValue()
                                                  : c.getSExtValue();
      })
      .Default([](auto*) {
        llvm::report_fatal_error("Unsupported integer operation in "
                                 "mlir::qco::classicalarithoperation");
        return 0;
      });
}

inline double getArithDoubleOpResult(mlir::Operation* operation,
                                     const double value1,
                                     const double value2 = 0.0) {
  for (mlir::Value operand : operation->getOperands()) {
    if (mlir::isa<mlir::VectorType>(operand.getType())) {
      llvm::report_fatal_error(
          "Constant propagation does not support vectors as classical types.");
    }
  }

  const auto floatTy =
      dyn_cast<mlir::FloatType>(operation->getResult(0).getType());
  if (!floatTy) {
    llvm::report_fatal_error("Expected floating-point result type.");
  }

  const llvm::fltSemantics& sem = floatTy.getFloatSemantics();
  constexpr auto rm = llvm::APFloat::rmNearestTiesToEven;
  bool losesInfo = false;

  llvm::APFloat lhs(value1);
  llvm::APFloat rhs(value2);
  lhs.convert(sem, rm, &losesInfo);
  if (losesInfo) {
    llvm::report_fatal_error("value1 cannot be represented safely.");
  }

  losesInfo = false;
  rhs.convert(sem, rm, &losesInfo);
  if (losesInfo) {
    llvm::report_fatal_error("value2 cannot be represented safely.");
  }

  llvm::APFloat result = lhs;

  const bool supported =
      mlir::TypeSwitch<mlir::Operation*, bool>(operation)
          .Case<mlir::arith::AddFOp>([&](auto) {
            result = lhs;
            auto status = result.add(rhs, rm);
            return (status & llvm::APFloat::opInvalidOp) == 0;
          })
          .Case<mlir::arith::DivFOp>([&](auto) {
            result = lhs;
            auto status = result.divide(rhs, rm);
            return (status & llvm::APFloat::opInvalidOp) == 0;
          })
          .Case<mlir::arith::MaximumFOp>([&](auto) {
            result = llvm::maximum(lhs, rhs);
            return true;
          })
          .Case<mlir::arith::MaxNumFOp>([&](auto) {
            result = llvm::maxnum(lhs, rhs);
            return true;
          })
          .Case<mlir::arith::MinimumFOp>([&](auto) {
            result = llvm::minimum(lhs, rhs);
            return true;
          })
          .Case<mlir::arith::MinNumFOp>([&](auto) {
            result = llvm::minnum(lhs, rhs);
            return true;
          })
          .Case<mlir::arith::MulFOp>([&](auto) {
            result = lhs;
            auto status = result.multiply(rhs, rm);
            return (status & llvm::APFloat::opInvalidOp) == 0;
          })
          .Case<mlir::arith::NegFOp>([&](auto) {
            result = lhs;
            result.changeSign();
            return true;
          })
          .Case<mlir::arith::RemFOp>([&](auto) {
            result = lhs;
            auto status = result.remainder(rhs);
            return (status & llvm::APFloat::opInvalidOp) == 0;
          })
          .Case<mlir::arith::SubFOp>([&](auto) {
            result = lhs;
            return result.subtract(rhs, rm) == llvm::APFloat::opOK;
          })
          .Default([](auto) { return false; });

  if (!supported) {
    llvm::report_fatal_error("Unsupported floating-point operation in "
                             "mlir::qco::classicalarithoperation");
  }

  llvm::APFloat resultInDouble = result;
  bool loseInfoResult = false;
  result.convert(llvm::APFloat::IEEEdouble(), rm, &loseInfoResult);
  if (loseInfoResult) {
    llvm::report_fatal_error(
        "Floating-point fold result cannot be represented safely as double.");
  }
  const double folded = result.convertToDouble();

  return folded;
}

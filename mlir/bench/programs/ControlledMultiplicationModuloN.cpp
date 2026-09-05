/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/ControlledMultiplicationModuloN.hpp"

#include "Programs.h"
#include "QFTAdderUtils.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>
#include <string>

namespace mqt::bench {

using namespace mlir;

[[nodiscard]] static Value
unsignedIntegerConstant(qc::QCProgramBuilder& builder, StringRef bits) {
  const auto width = static_cast<unsigned>(bits.size() + 1U);
  auto type = builder.getIntegerType(width);
  auto value = llvm::APInt(width, bits, 2);
  return arith::ConstantOp::create(builder,
                                   builder.getIntegerAttr(type, value));
}

[[nodiscard]] static Value
unsignedIntegerConstant(qc::QCProgramBuilder& builder, IntegerType type,
                        uint64_t value) {
  return arith::ConstantOp::create(builder,
                                   builder.getIntegerAttr(type, value));
}

static void phaseAdd(qc::QCProgramBuilder& builder, Value accumulator,
                     int64_t width, Value addend, ValueRange controls,
                     bool inverse) {
  auto integerType = cast<IntegerType>(addend.getType());
  auto zero = builder.indexConstant(0);
  auto upper = builder.indexConstant(width);
  auto one = builder.indexConstant(1);
  auto zeroAngle = builder.floatConstant(0.);
  auto half = builder.floatConstant(0.5);
  auto pi = builder.floatConstant(std::numbers::pi);
  auto negativeOne = builder.floatConstant(-1.);
  auto zeroBit = unsignedIntegerConstant(builder, integerType, 0);
  auto oneBit = unsignedIntegerConstant(builder, integerType, 1);

  auto loop = scf::ForOp::create(builder, zero, upper, one,
                                 ValueRange{zeroAngle, addend});
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());

  auto target = loop.getInductionVar();
  auto remaining = loop.getRegionIterArg(1);
  auto bit = arith::AndIOp::create(builder, remaining, oneBit).getResult();
  auto hasBit =
      arith::CmpIOp::create(builder, arith::CmpIPredicate::ne, bit, zeroBit)
          .getResult();
  auto previous = loop.getRegionIterArg(0);
  auto decayed = arith::MulFOp::create(builder, previous, half).getResult();
  auto selectAngle =
      scf::IfOp::create(builder, builder.getF64Type(), hasBit, true);
  {
    OpBuilder::InsertionGuard selectGuard(builder);
    auto& thenBlock = selectAngle.getThenRegion().front();
    if (!thenBlock.empty()) {
      thenBlock.back().erase();
    }
    builder.setInsertionPointToEnd(&thenBlock);
    auto angle = arith::AddFOp::create(builder, decayed, pi).getResult();
    scf::YieldOp::create(builder, ValueRange{angle});

    auto& elseBlock = selectAngle.getElseRegion().front();
    if (!elseBlock.empty()) {
      elseBlock.back().erase();
    }
    builder.setInsertionPointToEnd(&elseBlock);
    scf::YieldOp::create(builder, ValueRange{decayed});
  }
  auto angle = selectAngle.getResult(0);
  auto gateAngle =
      inverse ? arith::MulFOp::create(builder, angle, negativeOne).getResult()
              : angle;
  auto qubit = builder.loadQubit(accumulator, target);
  if (controls.empty()) {
    builder.p(gateAngle, qubit);
  } else if (controls.size() == 1U) {
    builder.cp(gateAngle, controls.front(), qubit);
  } else {
    builder.mcp(gateAngle, controls, qubit);
  }
  auto next = arith::ShRUIOp::create(builder, remaining, oneBit).getResult();
  scf::YieldOp::create(builder, ValueRange{angle, next});
}

static void modularAdd(qc::QCProgramBuilder& builder, Value accumulator,
                       int64_t width, Value addend, Value modulus,
                       ValueRange controls, Value work) {
  auto overflowIndex = builder.indexConstant(width - 1);

  phaseAdd(builder, accumulator, width, addend, controls, false);
  phaseAdd(builder, accumulator, width, modulus, {}, true);

  detail::inverseQFT(builder, accumulator, width);
  builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
  detail::forwardQFT(builder, accumulator, width);

  phaseAdd(builder, accumulator, width, modulus, work, false);
  phaseAdd(builder, accumulator, width, addend, controls, true);

  detail::inverseQFT(builder, accumulator, width);
  builder.x(builder.loadQubit(accumulator, overflowIndex));
  builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
  builder.x(builder.loadQubit(accumulator, overflowIndex));
  detail::forwardQFT(builder, accumulator, width);

  phaseAdd(builder, accumulator, width, addend, controls, false);
}

SmallVector<Value> controlledMultiplicationModuloN(
    qc::QCProgramBuilder& builder,
    const ControlledMultiplicationModuloN& benchmark) {
  const auto& options = benchmark.options();
  const auto bits = static_cast<int64_t>(options.modulus.size());
  const auto width = bits + 1;

  auto control = builder.allocQubit();
  auto multiplicand = builder.allocQubitRegisterStorage(bits, "multiplicand");
  auto accumulator = builder.allocQubitRegisterStorage(width, "accumulator");
  auto work = builder.allocQubit();
  auto result = builder.allocClassicalBitRegister(
      static_cast<int64_t>(benchmark.output().width), benchmark.output().name);

  builder.h(control);
  builder.scfFor(0, bits, 1, [&](Value index) {
    builder.h(builder.loadQubit(multiplicand, index));
  });

  detail::forwardQFT(builder, accumulator, width);

  auto addend = unsignedIntegerConstant(builder, options.multiplier);
  auto modulus = unsignedIntegerConstant(builder, options.modulus);
  auto integerType = cast<IntegerType>(addend.getType());
  auto zero = builder.indexConstant(0);
  auto upper = builder.indexConstant(bits);
  auto one = builder.indexConstant(1);
  auto multiplierLoop =
      scf::ForOp::create(builder, zero, upper, one, ValueRange{addend});
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(multiplierLoop.getBody());
    auto index = multiplierLoop.getInductionVar();
    auto currentAddend = multiplierLoop.getRegionIterArg(0);
    SmallVector<Value, 2> controls{control,
                                   builder.loadQubit(multiplicand, index)};
    modularAdd(builder, accumulator, width, currentAddend, modulus, controls,
               work);

    auto oneBit = unsignedIntegerConstant(builder, integerType, 1);
    auto doubled =
        arith::ShLIOp::create(builder, currentAddend, oneBit).getResult();
    auto next = arith::RemUIOp::create(builder, doubled, modulus).getResult();
    scf::YieldOp::create(builder, ValueRange{next});
  }

  detail::inverseQFT(builder, accumulator, width);

  builder.measureQubitRegister(accumulator, result, width);
  auto multiplicandOffset = builder.indexConstant(width);
  builder.scfFor(0, bits, 1, [&](Value index) {
    auto resultIndex =
        arith::AddIOp::create(builder, multiplicandOffset, index).getResult();
    builder.measure(builder.loadQubit(multiplicand, index), result,
                    resultIndex);
  });
  builder.measure(control, result, 2 * bits + 1);
  return {result};
}

} // namespace mqt::bench

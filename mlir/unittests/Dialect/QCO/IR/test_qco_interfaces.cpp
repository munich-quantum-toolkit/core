/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/LinearityUtils.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <iterator>

using namespace mlir;

namespace {

TEST(QCOWireInterface, PreservesWiresAcrossUnitaryAndNonunitaryOperations) {
  MLIRContext context;
  context.loadDialect<qco::QCODialect, func::FuncDialect>();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%condition: i1, %index: index,
                      %q0: !qco.qubit, %q1: !qco.qubit,
                      %register: tensor<2x!qco.qubit>)
          -> (i1, !qco.qubit, !qco.qubit, tensor<2x!qco.qubit>) {
        %a, %b = qco.swap %q0, %q1 : !qco.qubit, !qco.qubit
        %c, %bit = qco.measure %a : !qco.qubit
        %d = qco.reset %c : !qco.qubit
        %selected:4 = qco.if %condition
            args(%x = %d, %y = %b, %r = %register)
            -> (i1, !qco.qubit, !qco.qubit, tensor<2x!qco.qubit>) {
          qco.yield %bit, %x, %y, %r
              : i1, !qco.qubit, !qco.qubit, tensor<2x!qco.qubit>
        } else args(%x = %d, %y = %b, %r = %register) {
          qco.yield %condition, %x, %y, %r
              : i1, !qco.qubit, !qco.qubit, tensor<2x!qco.qubit>
        }
        %out:4 = qco.index_switch %index
            -> (i1, !qco.qubit, !qco.qubit, tensor<2x!qco.qubit>)
        case 0 args(%x = %selected#1, %y = %selected#2, %r = %selected#3) {
          qco.yield %selected#0, %x, %y, %r
              : i1, !qco.qubit, !qco.qubit, tensor<2x!qco.qubit>
        }
        default args(%x = %selected#1, %y = %selected#2, %r = %selected#3) {
          qco.yield %condition, %x, %y, %r
              : i1, !qco.qubit, !qco.qubit, tensor<2x!qco.qubit>
        }
        return %out#0, %out#1, %out#2, %out#3
            : i1, !qco.qubit, !qco.qubit, tensor<2x!qco.qubit>
      }
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));

  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  for (auto& operation : function.getBody().front().without_terminator()) {
    auto wires = dyn_cast<qco::WireOpInterface>(&operation);
    ASSERT_TRUE(wires);
    EXPECT_EQ(wires.getInputQubits().size(), wires.getOutputQubits().size());
    for (auto [index, input] : llvm::enumerate(wires.getInputQubits())) {
      auto output = wires.getOutputQubit(index);
      EXPECT_EQ(wires.getInputQubit(index), input);
      EXPECT_EQ(wires.getOutputForInput(input), output);
      EXPECT_EQ(wires.getInputForOutput(output), input);
    }
    EXPECT_EQ(isa<qco::UnitaryOpInterface>(&operation),
              isa<qco::SWAPOp>(&operation));
  }

  auto swap = *function.getOps<qco::SWAPOp>().begin();
  auto unitary = cast<qco::UnitaryOpInterface>(swap.getOperation());
  EXPECT_EQ(unitary.getInputQubit(0), function.getArgument(2));
  EXPECT_EQ(unitary.getInputForOutput(unitary.getOutputQubit(1)),
            function.getArgument(3));

  auto measure = *function.getOps<qco::MeasureOp>().begin();
  EXPECT_EQ(measure.getOutputQubits().front(), measure.getQubitOut());
  EXPECT_EQ(measure.getOutputQubits().size(), 1U);
  auto conditional = *function.getOps<qco::IfOp>().begin();
  EXPECT_EQ(conditional.getInputQubits().getBeginOperandIndex(), 1U);
  EXPECT_EQ(conditional.getOutputQubit(0), conditional.getResult(1));

  DenseMap<Value, Value> origins;
  ASSERT_TRUE(succeeded(qco::collectWireOrigins(*moduleOp, origins)));
  auto returned = cast<func::ReturnOp>(function.getBody().front().back());
  for (auto [output, input] :
       llvm::zip_equal(returned.getOperands().drop_front(),
                       function.getArguments().drop_front(2))) {
    EXPECT_EQ(origins.lookup(output), input);
  }

  auto reset = *function.getOps<qco::ResetOp>().begin();
  auto indexSwitch = *function.getOps<qco::IndexSwitchOp>().begin();
  const std::array chains{
      SmallVector<Value>{
          function.getArgument(2),
          swap.getOutputQubit(0),
          measure.getQubitOut(),
          reset.getQubitOut(),
          conditional.getOutputQubit(0),
          indexSwitch.getOutputQubit(0),
      },
      SmallVector<Value>{
          function.getArgument(3),
          swap.getOutputQubit(1),
          conditional.getOutputQubit(1),
          indexSwitch.getOutputQubit(1),
      },
  };
  for (const auto& chain : chains) {
    qco::WireIterator forward(chain.front());
    for (Value expected : chain) {
      ASSERT_NE(forward, std::default_sentinel);
      EXPECT_EQ(forward.qubit(), expected);
      ++forward;
    }
    EXPECT_EQ(forward.operation(), returned.getOperation());
    ++forward;
    EXPECT_EQ(forward, std::default_sentinel);

    qco::WireIterator backward(chain.back());
    for (Value expected : llvm::reverse(chain)) {
      ASSERT_NE(backward, std::default_sentinel);
      EXPECT_EQ(backward.qubit(), expected);
      --backward;
    }
    EXPECT_EQ(backward, std::default_sentinel);
  }
}

} // namespace

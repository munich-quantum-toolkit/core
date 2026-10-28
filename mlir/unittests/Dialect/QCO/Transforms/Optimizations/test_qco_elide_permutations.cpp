/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Package.hpp"
#include "mqt/Conversion/QCOToQC/QCOToQC.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

namespace {

class ElidePermutationsTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<qtensor::QTensorDialect, mlir::mqt::MQTDialect,
                         arith::ArithDialect, cf::ControlFlowDialect,
                         func::FuncDialect, scf::SCFDialect>();
  }

  static Value swapTensor(Value tensor, OpBuilder& builder) {
    auto location = tensor.getLoc();
    auto zero = arith::ConstantIndexOp::create(builder, location, 0);
    auto one = arith::ConstantIndexOp::create(builder, location, 1);
    auto first = qtensor::ExtractOp::create(builder, location, tensor, zero);
    auto second = qtensor::ExtractOp::create(builder, location,
                                             first.getOutTensor(), one);
    auto swap = SWAPOp::create(builder, location, first.getResult(),
                               second.getResult());
    auto inserted = qtensor::InsertOp::create(
        builder, location, swap.getQubit0Out(), second.getOutTensor(), zero);
    return qtensor::InsertOp::create(builder, location, swap.getQubit1Out(),
                                     inserted.getResult(), one)
        .getResult();
  }

  static void rewrite(ModuleOp program,
                      const ElidePermutationsOptions& options = {}) {
    PassManager pm(program.getContext());
    pm.addPass(createElidePermutations(options));
    ASSERT_TRUE(succeeded(pm.run(program)));
    EXPECT_TRUE(succeeded(verify(program)));
    EXPECT_TRUE(succeeded(verifyLinearity(program)));
  }

  static void checkUnitary(ModuleOp program, size_t maximumSwaps,
                           ArrayRef<int64_t> arguments = {},
                           size_t numQubits = 2, bool convertToQC = true,
                           const ElidePermutationsOptions& options = {}) {
    ASSERT_TRUE(succeeded(verify(program)));
    ASSERT_TRUE(succeeded(verifyLinearity(program)));
    OwningOpRef<ModuleOp> original(program.clone());
    ASSERT_NO_FATAL_FAILURE(rewrite(program, options));
    size_t swaps = 0;
    program.walk([&](SWAPOp) { ++swaps; });
    EXPECT_LE(swaps, maximumSwaps);
    if (options.terminalOnly) {
      // Without output tracking, the outer SWAP also retains physical order.
      EXPECT_EQ(swaps, maximumSwaps);
    }

    const auto matrix = [&](ModuleOp source, dd::Package& package) {
      auto function = mlir::mqt::getEntryPoint(source);
      OpBuilder builder(source.getContext());
      DDArgumentBindings bindings;
      for (auto [argument, value] :
           llvm::zip_equal(function.getArguments(), arguments)) {
        bindings[argument] = builder.getIntegerAttr(argument.getType(), value);
      }
      return buildFunctionality(function, package, bindings);
    };
    auto package = llvm::cantFail(dd::Package::create(numQubits));
    auto expected = matrix(*original, *package);
    auto actual = matrix(program, *package);
    ASSERT_TRUE(succeeded(expected));
    ASSERT_TRUE(succeeded(actual));
    const auto expectedMatrix =
        llvm::cantFail(dd::getMatrix(*expected, numQubits));
    const auto actualMatrix = llvm::cantFail(dd::getMatrix(*actual, numQubits));
    for (size_t row = 0; row < expectedMatrix.size(); ++row) {
      for (size_t column = 0; column < expectedMatrix[row].size(); ++column) {
        EXPECT_LE(
            std::abs(expectedMatrix[row][column] - actualMatrix[row][column]),
            1e-12);
      }
    }
    package->decRef(*expected);
    package->decRef(*actual);
    if (convertToQC) {
      PassManager conversion(program.getContext());
      conversion.addPass(createQCOToQC());
      EXPECT_TRUE(succeeded(conversion.run(program)));
    }
  }
};

TEST_F(ElidePermutationsTest,
       PreservesTerminalSamplingWithoutClassicalResults) {
  for (const bool structured : {false, true}) {
    SCOPED_TRACE(testing::Message() << "structured=" << structured);
    const std::string prefix = R"mlir(module {
      func.func @main() attributes {mqt.entry_point} {
        %a = qco.alloc : !qco.qubit
        %b = qco.alloc : !qco.qubit
        %x = qco.x %a : !qco.qubit
        %sa, %sb = qco.swap %x, %b : !qco.qubit, !qco.qubit
    )mlir";
    const auto* measurements = structured ? R"mlir(
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %out:2 = scf.for %i = %zero to %one step %one
            iter_args(%qa = %sa, %qb = %sb) -> (!qco.qubit, !qco.qubit) {
          %oa, %ba = qco.measure %qa : !qco.qubit
          %ob, %bb = qco.measure %qb : !qco.qubit
          scf.yield %oa, %ob : !qco.qubit, !qco.qubit
        }
        qco.sink %out#0 : !qco.qubit
        qco.sink %out#1 : !qco.qubit
    )mlir"
                                          : R"mlir(
        %qa, %ba = qco.measure %sa : !qco.qubit
        %qb, %bb = qco.measure %sb : !qco.qubit
        qco.sink %qa : !qco.qubit
        qco.sink %qb : !qco.qubit
    )mlir";
    auto program = parseSourceString<ModuleOp>(
        prefix + measurements + "return } }", &context_);
    ASSERT_TRUE(program);
    ASSERT_NO_FATAL_FAILURE(rewrite(*program));
    PassManager canonicalizer(&context_);
    canonicalizer.addPass(createCanonicalizerPass());
    ASSERT_TRUE(succeeded(canonicalizer.run(*program)));
    auto function = mlir::mqt::getEntryPoint(*program);
    auto counts = sample(function, 1, 42);
    ASSERT_TRUE(succeeded(counts));
    ASSERT_EQ(counts->size(), 1U);
    EXPECT_EQ(counts->begin()->first, "10");
    auto package = llvm::cantFail(dd::Package::create(2));
    auto state = simulateStatevector(function, *package);
    ASSERT_TRUE(succeeded(state));
    package->decRef(*state);
    PassManager conversion(&context_);
    conversion.addPass(createQCOToQC());
    EXPECT_TRUE(succeeded(conversion.run(*program)));
  }
}

TEST_F(ElidePermutationsTest, TracksReinsertedTensorSlots) {
  for (const auto& [allocation, convertToQC] : {
           std::pair{R"mlir(
      %two = arith.constant 2 : index
      %tensor = qtensor.alloc(%two) : tensor<2x!qco.qubit>
    )mlir",
                     true},
           std::pair{R"mlir(
      %left = qco.alloc : !qco.qubit
      %right = qco.alloc : !qco.qubit
      %tensor = qtensor.from_elements %left, %right : tensor<2x!qco.qubit>
    )mlir",
                     false},
       }) {
    SCOPED_TRACE(allocation);
    auto program = parseSourceString<ModuleOp>(std::string(R"mlir(module {
    func.func @main() attributes {mqt.entry_point} {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
    )mlir") + allocation + R"mlir(
      %t0, %a = qtensor.extract %tensor[%zero] : tensor<2x!qco.qubit>
      %t1, %b = qtensor.extract %t0[%one] : tensor<2x!qco.qubit>
      %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit
      %h = qco.h %sa : !qco.qubit
      %t2 = qtensor.insert %h into %t1[%zero] : tensor<2x!qco.qubit>
      %t3 = qtensor.insert %sb into %t2[%one] : tensor<2x!qco.qubit>
      %t4, %againA = qtensor.extract %t3[%zero] : tensor<2x!qco.qubit>
      %t5, %againB = qtensor.extract %t4[%one] : tensor<2x!qco.qubit>
      %outA, %outB = qco.swap %againA, %againB : !qco.qubit, !qco.qubit
      %t6 = qtensor.insert %outA into %t5[%zero] : tensor<2x!qco.qubit>
      %t7 = qtensor.insert %outB into %t6[%one] : tensor<2x!qco.qubit>
      qtensor.dealloc %t7 : tensor<2x!qco.qubit>
      return
    }
  })mlir",
                                               &context_);
    ASSERT_TRUE(program);
    // Tensors packed from owned scalars require placement before QCO-to-QC.
    ASSERT_NO_FATAL_FAILURE(checkUnitary(*program, 0, {}, 2, convertToQC));
  }
}

TEST_F(ElidePermutationsTest, PropagatesCommonBranchPermutations) {
  for (const bool common : {false, true}) {
    for (const int64_t condition : {0, 1}) {
      SCOPED_TRACE(testing::Message()
                   << "common=" << common << ", condition=" << condition);
      auto program = parseSourceString<ModuleOp>(std::string(R"mlir(module {
        func.func @main(%condition: i1) attributes {mqt.entry_point} {
          %a = qco.alloc : !qco.qubit
          %b = qco.alloc : !qco.qubit
          %left, %right = qco.if %condition args(%ta = %a, %tb = %b)
              -> (!qco.qubit, !qco.qubit) {
            %sa, %sb = qco.swap %ta, %tb : !qco.qubit, !qco.qubit
            %h = qco.h %sa : !qco.qubit
            qco.yield %h, %sb : !qco.qubit, !qco.qubit
          } else args(%ea = %a, %eb = %b) {
      )mlir") +
                                                     (common ? R"mlir(
            %sa, %sb = qco.swap %ea, %eb : !qco.qubit, !qco.qubit
            %x = qco.x %sb : !qco.qubit
            qco.yield %sa, %x : !qco.qubit, !qco.qubit
      )mlir"
                                                             : R"mlir(
            %x = qco.x %eb : !qco.qubit
            qco.yield %ea, %x : !qco.qubit, !qco.qubit
      )mlir") +
                                                     R"mlir(
          }
          %outA, %outB = qco.swap %left, %right : !qco.qubit, !qco.qubit
          qco.sink %outA : !qco.qubit
          qco.sink %outB : !qco.qubit
          return
        }
      })mlir",
                                                 &context_);
      ASSERT_TRUE(program);
      for (const bool terminalOnly : {false, true}) {
        SCOPED_TRACE(terminalOnly);
        OwningOpRef<ModuleOp> instance(program->clone());
        ElidePermutationsOptions options;
        options.terminalOnly = terminalOnly;
        ASSERT_NO_FATAL_FAILURE(
            checkUnitary(*instance, common ? (terminalOnly ? 3 : 0) : 2,
                         {condition}, 2, true, options));
      }
    }
  }
}

TEST_F(ElidePermutationsTest, TerminalModePreservesEarlierRouting) {
  for (const auto* boundary : {
           R"mlir(%next = qco.h %sa : !qco.qubit)mlir",
           R"mlir(%measured, %bit = qco.measure %sa : !qco.qubit
                  %next = qco.h %measured : !qco.qubit)mlir",
           R"mlir(%next = qco.reset %sa : !qco.qubit)mlir",
           R"mlir(%next = qco.if %condition args(%q = %sa) -> (!qco.qubit) {
                    qco.yield %q : !qco.qubit
                  } else args(%q = %sa) {
                    qco.yield %q : !qco.qubit
                  })mlir",
       }) {
    SCOPED_TRACE(boundary);
    auto program = parseSourceString<ModuleOp>(
        std::string(
            R"mlir(module attributes {mqt.source_qubit_count = 3 : i64} {
          func.func @main(%condition: i1) attributes {mqt.entry_point} {
            %a = qco.alloc {mqt.source_qubit_indices = array<i64: 0>} : !qco.qubit
            %b = qco.alloc {mqt.source_qubit_indices = array<i64: 1>} : !qco.qubit
            %c = qco.alloc {mqt.source_qubit_indices = array<i64: 2>} : !qco.qubit
            %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit
        )mlir") +
            boundary + R"mlir(
            %outB, %outC = qco.swap %sb, %c : !qco.qubit, !qco.qubit
            qco.sink %next : !qco.qubit
            qco.sink %outB : !qco.qubit
            qco.sink %outC : !qco.qubit
            return
          }
        })mlir",
        &context_);
    ASSERT_TRUE(program);
    ASSERT_TRUE(succeeded(verifyLinearity(*program)));
    ElidePermutationsOptions options;
    options.terminalOnly = true;
    options.trackOutputPermutation = true;
    ASSERT_NO_FATAL_FAILURE(rewrite(*program, options));
    size_t swaps = 0;
    program->walk([&](SWAPOp) { ++swaps; });
    // A single-qubit gate may follow a virtual SWAP; the other boundaries
    // retain the original exchange instead of synthesizing a new permutation.
    EXPECT_EQ(swaps,
              StringRef(boundary).starts_with("%next = qco.h") ? 0U : 1U);
  }
}

TEST_F(ElidePermutationsTest, SummarizesTensorBranchesBySlot) {
  for (const bool incomingPermutation : {false, true}) {
    SCOPED_TRACE(incomingPermutation);
    for (const int64_t condition : {0, 1}) {
      SCOPED_TRACE(condition);
      auto program = parseSourceString<ModuleOp>(R"mlir(module {
      func.func @main(%condition: i1) attributes {mqt.entry_point} {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %tensor = qtensor.alloc(%two) : tensor<2x!qco.qubit>
        %joined = qco.if %condition args(%t = %tensor) -> (tensor<2x!qco.qubit>) {
          %t0, %a = qtensor.extract %t[%zero] : tensor<2x!qco.qubit>
          %t1, %b = qtensor.extract %t0[%one] : tensor<2x!qco.qubit>
          %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit
          %h = qco.h %sa : !qco.qubit
          %t2 = qtensor.insert %h into %t1[%zero] : tensor<2x!qco.qubit>
          %t3 = qtensor.insert %sb into %t2[%one] : tensor<2x!qco.qubit>
          qco.yield %t3 : tensor<2x!qco.qubit>
        } else args(%t = %tensor) {
          %t0, %b = qtensor.extract %t[%one] : tensor<2x!qco.qubit>
          %t1, %a = qtensor.extract %t0[%zero] : tensor<2x!qco.qubit>
          %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit
          %x = qco.x %sb : !qco.qubit
          %t2 = qtensor.insert %x into %t1[%one] : tensor<2x!qco.qubit>
          %t3 = qtensor.insert %sa into %t2[%zero] : tensor<2x!qco.qubit>
          qco.yield %t3 : tensor<2x!qco.qubit>
        }
        %t0, %a = qtensor.extract %joined[%zero] : tensor<2x!qco.qubit>
        %t1, %b = qtensor.extract %t0[%one] : tensor<2x!qco.qubit>
        %outA, %outB = qco.swap %a, %b : !qco.qubit, !qco.qubit
        %t2 = qtensor.insert %outA into %t1[%zero] : tensor<2x!qco.qubit>
        %t3 = qtensor.insert %outB into %t2[%one] : tensor<2x!qco.qubit>
        qtensor.dealloc %t3 : tensor<2x!qco.qubit>
        return
      }
    })mlir",
                                                 &context_);
      ASSERT_TRUE(program);
      if (incomingPermutation) {
        auto branch =
            *mlir::mqt::getEntryPoint(*program).getOps<IfOp>().begin();
        OpBuilder builder(branch);
        auto tensor = branch.getQubits().front();
        branch.getQubitsMutable().assign(swapTensor(tensor, builder));
        // Put the common permutation before the branch instead of inside it.
        for (auto& region : branch->getRegions()) {
          auto swap = *region.front().getOps<SWAPOp>().begin();
          swap.getQubit0Out().replaceAllUsesWith(swap.getQubit0In());
          swap.getQubit1Out().replaceAllUsesWith(swap.getQubit1In());
          swap.erase();
        }
      }
      ASSERT_NO_FATAL_FAILURE(checkUnitary(*program, 0, {condition}));
    }
  }
}

TEST_F(ElidePermutationsTest, ReconcilesIndexSwitchPermutations) {
  for (const bool common : {false, true}) {
    for (const int64_t selector : {2, 7, 9}) {
      SCOPED_TRACE(testing::Message()
                   << "common=" << common << ", selector=" << selector);
      auto program = parseSourceString<ModuleOp>(std::string(R"mlir(module {
        func.func @main(%selector: index) attributes {mqt.entry_point} {
          %a = qco.alloc : !qco.qubit
          %b = qco.alloc : !qco.qubit
          %left, %right = qco.index_switch %selector -> (!qco.qubit, !qco.qubit)
          case 2 args(%qa = %a, %qb = %b) {
            %sa, %sb = qco.swap %qa, %qb : !qco.qubit, !qco.qubit
            %h = qco.h %sa : !qco.qubit
            qco.yield %h, %sb : !qco.qubit, !qco.qubit
          }
          case 7 args(%qa = %a, %qb = %b) {
            %sa, %sb = qco.swap %qa, %qb : !qco.qubit, !qco.qubit
            %t = qco.t %sb : !qco.qubit
            qco.yield %sa, %t : !qco.qubit, !qco.qubit
          }
          default args(%qa = %a, %qb = %b) {
      )mlir") +
                                                     (common ? R"mlir(
            %sa, %sb = qco.swap %qa, %qb : !qco.qubit, !qco.qubit
            %x = qco.x %sa : !qco.qubit
            qco.yield %x, %sb : !qco.qubit, !qco.qubit
      )mlir"
                                                             : R"mlir(
            %x = qco.x %qa : !qco.qubit
            qco.yield %x, %qb : !qco.qubit, !qco.qubit
      )mlir") +
                                                     R"mlir(
          }
          %outA, %outB = qco.swap %left, %right : !qco.qubit, !qco.qubit
          qco.sink %outA : !qco.qubit
          qco.sink %outB : !qco.qubit
          return
        }
      })mlir",
                                                 &context_);
      ASSERT_TRUE(program);
      ASSERT_NO_FATAL_FAILURE(
          checkUnitary(*program, common ? 0 : 3, {selector}));
    }
  }
}

TEST_F(ElidePermutationsTest, PreservesLoopBackedgeOrder) {
  for (const auto& [body, maximumSwaps] : {
           std::pair{R"mlir(
        %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit
        %loop:2 = scf.for %i = %zero to %limit step %one
            iter_args(%qa = %sa, %qb = %sb) -> (!qco.qubit, !qco.qubit) {
          %h = qco.h %qa : !qco.qubit
          scf.yield %h, %qb : !qco.qubit, !qco.qubit
        }
        %out:2 = qco.swap %loop#0, %loop#1 : !qco.qubit, !qco.qubit
      )mlir",
                     0U},
           std::pair{R"mlir(
        %out:3 = scf.while (%qa = %a, %qb = %b, %i = %zero)
            : (!qco.qubit, !qco.qubit, index) -> (!qco.qubit, !qco.qubit, index) {
          %continue = arith.cmpi ult, %i, %limit : index
          scf.condition(%continue) %qa, %qb, %i : !qco.qubit, !qco.qubit, index
        } do {
        ^bb0(%qa: !qco.qubit, %qb: !qco.qubit, %i: index):
          %sa, %sb = qco.swap %qa, %qb : !qco.qubit, !qco.qubit
          %h = qco.h %sa : !qco.qubit
          %ra, %rb = qco.swap %h, %sb : !qco.qubit, !qco.qubit
          %x = qco.x %rb : !qco.qubit
          %ta, %tb = qco.swap %ra, %x : !qco.qubit, !qco.qubit
          %next = arith.addi %i, %one : index
          scf.yield %ta, %tb, %next : !qco.qubit, !qco.qubit, index
        }
      )mlir",
                     1U},
           std::pair{R"mlir(
        %loop:3 = scf.while (%qa = %a, %qb = %b, %i = %zero)
            : (!qco.qubit, !qco.qubit, index) -> (!qco.qubit, !qco.qubit, index) {
          %sa, %sb = qco.swap %qa, %qb : !qco.qubit, !qco.qubit
          %h = qco.h %sa : !qco.qubit
          %continue = arith.cmpi ult, %i, %limit : index
          scf.condition(%continue) %h, %sb, %i : !qco.qubit, !qco.qubit, index
        } do {
        ^bb0(%qa: !qco.qubit, %qb: !qco.qubit, %i: index):
          %sa, %sb = qco.swap %qa, %qb : !qco.qubit, !qco.qubit
          %x = qco.x %sb : !qco.qubit
          %next = arith.addi %i, %one : index
          scf.yield %sa, %x, %next : !qco.qubit, !qco.qubit, index
        }
        %out:2 = qco.swap %loop#0, %loop#1 : !qco.qubit, !qco.qubit
      )mlir",
                     0U},
       }) {
    SCOPED_TRACE(body);
    auto program = parseSourceString<ModuleOp>(std::string(R"mlir(module {
      func.func @main(%limit: index) attributes {mqt.entry_point} {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %a = qco.alloc : !qco.qubit
        %b = qco.alloc : !qco.qubit
    )mlir") + body + R"mlir(
        qco.sink %out#0 : !qco.qubit
        qco.sink %out#1 : !qco.qubit
        return
      }
    })mlir",
                                               &context_);
    ASSERT_TRUE(program);
    for (const int64_t limit : {0, 1, 2, 3}) {
      SCOPED_TRACE(limit);
      OwningOpRef<ModuleOp> instance(program->clone());
      ASSERT_NO_FATAL_FAILURE(checkUnitary(*instance, maximumSwaps, {limit}));
    }
  }
  for (const bool incomingPermutation : {false, true}) {
    SCOPED_TRACE(incomingPermutation);
    auto tensorLoop = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main(%limit: index) attributes {mqt.entry_point} {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      %tensor = qtensor.alloc(%two) : tensor<2x!qco.qubit>
      %out = scf.for %i = %zero to %limit step %one
          iter_args(%t = %tensor) -> (tensor<2x!qco.qubit>) {
        %t0, %a = qtensor.extract %t[%zero] : tensor<2x!qco.qubit>
        %t1, %b = qtensor.extract %t0[%one] : tensor<2x!qco.qubit>
        %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit
        %h = qco.h %sa : !qco.qubit
        %ra, %rb = qco.swap %h, %sb : !qco.qubit, !qco.qubit
        %t2 = qtensor.insert %ra into %t1[%zero] : tensor<2x!qco.qubit>
        %t3 = qtensor.insert %rb into %t2[%one] : tensor<2x!qco.qubit>
        scf.yield %t3 : tensor<2x!qco.qubit>
      }
      qtensor.dealloc %out : tensor<2x!qco.qubit>
      return
    }
  })mlir",
                                                  &context_);
    ASSERT_TRUE(tensorLoop);
    if (incomingPermutation) {
      auto function = mlir::mqt::getEntryPoint(*tensorLoop);
      auto loop = *function.getOps<scf::ForOp>().begin();
      OpBuilder builder(loop);
      loop.getInitArgsMutable().assign(
          swapTensor(loop.getInitArgs().front(), builder));
      auto dealloc = *function.getOps<qtensor::DeallocOp>().begin();
      builder.setInsertionPoint(dealloc);
      dealloc.getTensorMutable().assign(
          swapTensor(dealloc.getTensor(), builder));
    }
    for (const int64_t limit : {0, 1, 2, 3}) {
      SCOPED_TRACE(limit);
      OwningOpRef<ModuleOp> instance(tensorLoop->clone());
      ASSERT_NO_FATAL_FAILURE(checkUnitary(*instance, 0, {limit}));
    }
  }
}

TEST_F(ElidePermutationsTest, PreservesStructuredAndDynamicPacking) {
  for (const auto& [tail, dynamic] : {
           std::pair{R"mlir(
             %outA, %outB = scf.for %i = %zero to %index step %one
                 iter_args(%qa = %sa, %qb = %sb) -> (!qco.qubit, !qco.qubit) {
               %x, %y = qco.swap %qa, %qb : !qco.qubit, !qco.qubit
               scf.yield %x, %y : !qco.qubit, !qco.qubit
             }
             %packed0 = qtensor.insert %outA into %rest1[%zero] : tensor<2x!qco.qubit>
             %packed1 = qtensor.insert %outB into %packed0[%one] : tensor<2x!qco.qubit>
           )mlir",
                     false},
           std::pair{R"mlir(
             %packed0 = qtensor.insert %sa into %rest1[%index] : tensor<2x!qco.qubit>
             %packed1 = qtensor.insert %sb into %packed0[%other] : tensor<2x!qco.qubit>
           )mlir",
                     true},
       }) {
    SCOPED_TRACE(tail);
    auto program = parseSourceString<ModuleOp>(std::string(R"mlir(module {
          func.func @main(%index: index) attributes {mqt.entry_point} {
            %zero = arith.constant 0 : index
            %one = arith.constant 1 : index
            %two = arith.constant 2 : index
            %other = arith.subi %one, %index : index
            %tensor = qtensor.alloc(%two) : tensor<2x!qco.qubit>
        )mlir") +
                                                   (dynamic ? R"mlir(
            %rest0, %a = qtensor.extract %tensor[%index] : tensor<2x!qco.qubit>
            %rest1, %b = qtensor.extract %rest0[%other] : tensor<2x!qco.qubit>
        )mlir"
                                                            : R"mlir(
            %rest0, %a = qtensor.extract %tensor[%zero] : tensor<2x!qco.qubit>
            %rest1, %b = qtensor.extract %rest0[%one] : tensor<2x!qco.qubit>
        )mlir") +
                                                   R"mlir(
            %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit
        )mlir" + tail + R"mlir(
            %rest2, %again = qtensor.extract %packed1[%zero] : tensor<2x!qco.qubit>
            %h = qco.h %again : !qco.qubit
            %packed2 = qtensor.insert %h into %rest2[%zero] : tensor<2x!qco.qubit>
            qtensor.dealloc %packed2 : tensor<2x!qco.qubit>
            return
          }
        })mlir",
                                               &context_);
    ASSERT_TRUE(program);
    for (const int64_t index : {0, 1}) {
      SCOPED_TRACE(index);
      OwningOpRef<ModuleOp> instance(program->clone());
      ASSERT_NO_FATAL_FAILURE(checkUnitary(*instance, 2, {index}));
    }
  }
}

TEST_F(ElidePermutationsTest, PreservesNestedLoopPermutations) {
  auto program = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main() attributes {mqt.entry_point} {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %a = qco.alloc : !qco.qubit
      %b = qco.alloc : !qco.qubit
      %c = qco.alloc : !qco.qubit
      %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit
      %out:3 = scf.for %i = %zero to %one step %one
          iter_args(%x = %sa, %y = %sb, %z = %c)
          -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %inner:3 = scf.for %j = %zero to %one step %one
            iter_args(%p = %x, %q = %y, %r = %z)
            -> (!qco.qubit, !qco.qubit, !qco.qubit) {
          %h = qco.h %q : !qco.qubit
          %s0, %s1 = qco.swap %p, %h : !qco.qubit, !qco.qubit
          %t1, %t2 = qco.swap %s1, %r : !qco.qubit, !qco.qubit
          scf.yield %s0, %t1, %t2 : !qco.qubit, !qco.qubit, !qco.qubit
        }
        scf.yield %inner#0, %inner#1, %inner#2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
      %t = qco.t %out#0 : !qco.qubit
      %ra, %rc = qco.swap %t, %out#2 : !qco.qubit, !qco.qubit
      qco.sink %ra : !qco.qubit
      qco.sink %out#1 : !qco.qubit
      qco.sink %rc : !qco.qubit
      return
    }
  })mlir",
                                             &context_);
  ASSERT_TRUE(program);
  ASSERT_NO_FATAL_FAILURE(checkUnitary(*program, 4, {}, 3));
}

TEST_F(ElidePermutationsTest, ResumesAfterDynamicTensorAccess) {
  for (const int64_t index : {0, 1}) {
    SCOPED_TRACE(index);
    auto program = parseSourceString<ModuleOp>(R"mlir(module {
      func.func @main(%index: index) attributes {mqt.entry_point} {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %tensor = qtensor.alloc(%two) : tensor<2x!qco.qubit>
        %t0, %a = qtensor.extract %tensor[%zero] : tensor<2x!qco.qubit>
        %t1, %b = qtensor.extract %t0[%one] : tensor<2x!qco.qubit>
        %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit
        %t2 = qtensor.insert %sa into %t1[%zero] : tensor<2x!qco.qubit>
        %t3 = qtensor.insert %sb into %t2[%one] : tensor<2x!qco.qubit>
        %t4, %dynamic = qtensor.extract %t3[%index] : tensor<2x!qco.qubit>
        %h = qco.h %dynamic : !qco.qubit
        %t5 = qtensor.insert %h into %t4[%index] : tensor<2x!qco.qubit>
        %t6, %againA = qtensor.extract %t5[%zero] : tensor<2x!qco.qubit>
        %t7, %againB = qtensor.extract %t6[%one] : tensor<2x!qco.qubit>
        %ra, %rb = qco.swap %againA, %againB : !qco.qubit, !qco.qubit
        %x = qco.x %ra : !qco.qubit
        %outA, %outB = qco.swap %x, %rb : !qco.qubit, !qco.qubit
        %t8 = qtensor.insert %outA into %t7[%zero] : tensor<2x!qco.qubit>
        %t9 = qtensor.insert %outB into %t8[%one] : tensor<2x!qco.qubit>
        qtensor.dealloc %t9 : tensor<2x!qco.qubit>
        return
      }
    })mlir",
                                               &context_);
    ASSERT_TRUE(program);
    ASSERT_NO_FATAL_FAILURE(checkUnitary(*program, 1, {index}));
  }
}

TEST_F(ElidePermutationsTest, PreservesCallsModifiersAndResourceOwners) {
  auto program = parseSourceString<ModuleOp>(R"mlir(module {
    func.func private @gate(%a: !qco.qubit, %b: !qco.qubit)
        -> (!qco.qubit, !qco.qubit) attributes {mqt.unitary} {
      %out:2 = qco.inv(%qa = %a, %qb = %b) {
        %sa, %sb = qco.swap %qa, %qb : !qco.qubit, !qco.qubit
        %t = qco.t %sa : !qco.qubit
        %ra, %rb = qco.swap %t, %sb : !qco.qubit, !qco.qubit
        %h = qco.h %rb : !qco.qubit
        %ta, %tb = qco.swap %ra, %h : !qco.qubit, !qco.qubit
        qco.yield %ta, %tb : !qco.qubit, !qco.qubit
      } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
      return %out#0, %out#1 : !qco.qubit, !qco.qubit
    }
    func.func @main() attributes {mqt.entry_point} {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %left = qtensor.alloc(%one) : tensor<1x!qco.qubit>
      %right = qtensor.alloc(%one) : tensor<1x!qco.qubit>
      %l0, %a = qtensor.extract %left[%zero] : tensor<1x!qco.qubit>
      %r0, %b = qtensor.extract %right[%zero] : tensor<1x!qco.qubit>
      %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit
      %l1 = qtensor.insert %sa into %l0[%zero] : tensor<1x!qco.qubit>
      %r1 = qtensor.insert %sb into %r0[%zero] : tensor<1x!qco.qubit>
      %l2, %againA = qtensor.extract %l1[%zero] : tensor<1x!qco.qubit>
      %r2, %againB = qtensor.extract %r1[%zero] : tensor<1x!qco.qubit>
      %call:2 = qco.call @gate(%againA, %againB) : (!qco.qubit, !qco.qubit) -> (!qco.qubit, !qco.qubit)
      %outA, %outB = qco.swap %call#0, %call#1 : !qco.qubit, !qco.qubit
      %l3 = qtensor.insert %outA into %l2[%zero] : tensor<1x!qco.qubit>
      %r3 = qtensor.insert %outB into %r2[%zero] : tensor<1x!qco.qubit>
      qtensor.dealloc %l3 : tensor<1x!qco.qubit>
      qtensor.dealloc %r3 : tensor<1x!qco.qubit>
      return
    }
  })mlir",
                                             &context_);
  ASSERT_TRUE(program);
  ASSERT_NO_FATAL_FAILURE(checkUnitary(*program, 1));

  auto fixed = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main() attributes {mqt.entry_point} {
      %a = qco.static 0 : !qco.qubit
      %b = qco.static 1 : !qco.qubit
      %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit
      %h = qco.h %sa : !qco.qubit
      %outA, %outB = qco.swap %h, %sb : !qco.qubit, !qco.qubit
      qco.sink %outA : !qco.qubit
      qco.sink %outB : !qco.qubit
      return
    }
  })mlir",
                                           &context_);
  ASSERT_TRUE(fixed);
  ASSERT_NO_FATAL_FAILURE(checkUnitary(*fixed, 2));
  EXPECT_EQ(
      llvm::range_size(mlir::mqt::getEntryPoint(*fixed).getOps<qc::SWAPOp>()),
      2U);
}

TEST_F(ElidePermutationsTest, ReconcilesCyclesAcrossTensorRegionOwners) {
  auto program = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main() attributes {mqt.entry_point} {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      %left = qtensor.alloc(%two) : tensor<2x!qco.qubit>
      %right = qtensor.alloc(%one) : tensor<1x!qco.qubit>
      %l0, %a0 = qtensor.extract %left[%zero] : tensor<2x!qco.qubit>
      %l1, %a1 = qtensor.extract %l0[%one] : tensor<2x!qco.qubit>
      %r0, %b0 = qtensor.extract %right[%zero] : tensor<1x!qco.qubit>
      %sa0, %sa1 = qco.swap %a0, %a1 : !qco.qubit, !qco.qubit
      %ta1, %tb0 = qco.swap %sa1, %b0 : !qco.qubit, !qco.qubit
      %l2 = qtensor.insert %sa0 into %l1[%zero] : tensor<2x!qco.qubit>
      %l3 = qtensor.insert %ta1 into %l2[%one] : tensor<2x!qco.qubit>
      %r1 = qtensor.insert %tb0 into %r0[%zero] : tensor<1x!qco.qubit>
      %out = scf.for %i = %zero to %two step %one
          iter_args(%tensor = %l3) -> (tensor<2x!qco.qubit>) {
        scf.yield %tensor : tensor<2x!qco.qubit>
      }
      qtensor.dealloc %out : tensor<2x!qco.qubit>
      qtensor.dealloc %r1 : tensor<1x!qco.qubit>
      return
    }
  })mlir",
                                             &context_);
  ASSERT_TRUE(program);
  ASSERT_NO_FATAL_FAILURE(checkUnitary(*program, 2, {}, 3));
}

TEST_F(ElidePermutationsTest, PreservesOpaqueRegionCapturesAndTensorOwnership) {
  auto program = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main() attributes {mqt.entry_point} {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      %tensor = qtensor.alloc(%two) : tensor<2x!qco.qubit>
      %t0, %a = qtensor.extract %tensor[%zero] : tensor<2x!qco.qubit>
      %t1, %b = qtensor.extract %t0[%one] : tensor<2x!qco.qubit>
      %x = qco.x %a : !qco.qubit
      %sa, %sb = qco.swap %x, %b : !qco.qubit, !qco.qubit
      %opaque:2 = scf.execute_region -> (!qco.qubit, !qco.qubit) {
        %inside = qco.x %sa : !qco.qubit
        scf.yield %inside, %sb : !qco.qubit, !qco.qubit
      }
      %t2 = qtensor.insert %opaque#0 into %t1[%zero] : tensor<2x!qco.qubit>
      %t3 = qtensor.insert %opaque#1 into %t2[%one] : tensor<2x!qco.qubit>
      qtensor.dealloc %t3 : tensor<2x!qco.qubit>
      return
    }
  })mlir",
                                             &context_);
  ASSERT_TRUE(program);
  ASSERT_NO_FATAL_FAILURE(rewrite(*program));
  PassManager canonicalizer(&context_);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*program)));
  auto counts = sample(mlir::mqt::getEntryPoint(*program), 1, 42);
  ASSERT_TRUE(succeeded(counts));
  ASSERT_EQ(counts->size(), 1U);
  EXPECT_EQ(counts->begin()->first, "11");
  PassManager conversion(&context_);
  conversion.addPass(createQCOToQC());
  EXPECT_TRUE(succeeded(conversion.run(*program)));

  for (const bool tensor : {false, true}) {
    SCOPED_TRACE(testing::Message() << "tensor=" << tensor);
    const auto* const capture = tensor ? R"mlir(
      %packed = qtensor.from_elements %s0, %s1 : tensor<2x!qco.qubit>
      %opaque = scf.execute_region -> tensor<2x!qco.qubit> {
        scf.yield %packed : tensor<2x!qco.qubit>
      }
      %t0, %a = qtensor.extract %opaque[%zero] : tensor<2x!qco.qubit>
      %t1, %b = qtensor.extract %t0[%one] : tensor<2x!qco.qubit>
    )mlir"
                                       : R"mlir(
      %a, %b = scf.execute_region -> (!qco.qubit, !qco.qubit) {
        scf.yield %s0, %s1 : !qco.qubit, !qco.qubit
      }
    )mlir";
    const auto* const release = tensor ? R"mlir(
      %t2 = qtensor.insert %outA into %t1[%zero] : tensor<2x!qco.qubit>
      %t3 = qtensor.insert %outB into %t2[%one] : tensor<2x!qco.qubit>
      qtensor.dealloc %t3 : tensor<2x!qco.qubit>
    )mlir"
                                       : R"mlir(
      qco.sink %outA : !qco.qubit
      qco.sink %outB : !qco.qubit
    )mlir";
    auto fixed = parseSourceString<ModuleOp>(std::string(R"mlir(module {
    func.func @main() attributes {mqt.entry_point} {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %s0 = qco.static 0 : !qco.qubit
      %s1 = qco.static 1 : !qco.qubit
    )mlir") + capture + R"mlir(
      %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit
      %h = qco.h %sa : !qco.qubit
      %outA, %outB = qco.swap %h, %sb : !qco.qubit, !qco.qubit
    )mlir" + release + "return } }",
                                             &context_);
    ASSERT_TRUE(fixed);
    ASSERT_NO_FATAL_FAILURE(rewrite(*fixed));
    EXPECT_EQ(
        llvm::range_size(mlir::mqt::getEntryPoint(*fixed).getOps<SWAPOp>()),
        2U);
  }
}

} // namespace

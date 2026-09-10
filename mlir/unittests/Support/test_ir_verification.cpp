/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"

#include "Support/IRVerification.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"

#include <array>
#include <string>

using namespace mlir;

static void expectComparison(llvm::StringRef lhsSource,
                             llvm::StringRef rhsSource, bool equivalent = false,
                             bool structurallyEquivalent = false) {
  DialectRegistry registry;
  registry
      .insert<arith::ArithDialect, func::FuncDialect, cf::ControlFlowDialect,
              qc::QCDialect, qco::QCODialect, cbit::CBitDialect,
              qtensor::QTensorDialect, LLVM::LLVMDialect>();
  MLIRContext context(registry);
  auto lhs = parseSourceString<ModuleOp>(lhsSource, &context);
  auto rhs = parseSourceString<ModuleOp>(rhsSource, &context);
  ASSERT_TRUE(lhs);
  ASSERT_TRUE(rhs);
  ASSERT_TRUE(succeeded(verify(*lhs)));
  ASSERT_TRUE(succeeded(verify(*rhs)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*lhs)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*rhs)));
  EXPECT_TRUE(areModulesStructurallyEquivalent(*lhs, *lhs));
  EXPECT_TRUE(areModulesStructurallyEquivalent(*rhs, *rhs));
  EXPECT_EQ(areModulesStructurallyEquivalent(*lhs, *rhs),
            structurallyEquivalent);
  EXPECT_EQ(areModulesStructurallyEquivalent(*rhs, *lhs),
            structurallyEquivalent);
  EXPECT_TRUE(areModulesEquivalentWithPermutations(*lhs, *lhs));
  EXPECT_TRUE(areModulesEquivalentWithPermutations(*rhs, *rhs));
  EXPECT_EQ(areModulesEquivalentWithPermutations(*lhs, *rhs), equivalent);
  EXPECT_EQ(areModulesEquivalentWithPermutations(*rhs, *lhs), equivalent);
}

TEST(IRVerificationTest, DenseConstantValueMatters) {
  expectComparison(R"(func.func @f() -> tensor<1xi64> {
    %x = arith.constant dense<1> : tensor<1xi64>
    return %x : tensor<1xi64>
  })",
                   R"(func.func @f() -> tensor<1xi64> {
    %x = arith.constant dense<2> : tensor<1xi64>
    return %x : tensor<1xi64>
  })");
}

TEST(IRVerificationTest, FunctionArgumentTypeMatters) {
  expectComparison("func.func @f(%x: i32) { return }",
                   "func.func @f(%x: i64) { return }");
}

TEST(IRVerificationTest, MissingInterfaceAttributeMatters) {
  expectComparison(
      "func.func @f() attributes {llvm.emit_c_interface} { return }",
      "func.func @f() { return }");
}

TEST(IRVerificationTest, QCNoncommutingGateOrderMatters) {
  expectComparison(R"(func.func @f(%q: !qc.qubit) {
    qc.h %q : !qc.qubit
    qc.x %q : !qc.qubit
    return
  })",
                   R"(func.func @f(%q: !qc.qubit) {
    qc.x %q : !qc.qubit
    qc.h %q : !qc.qubit
    return
  })");
}

TEST(IRVerificationTest, QCOConditionMatters) {
  const std::string lhs =
      R"(func.func @f(%a: i1, %b: i1, %q: !qco.qubit) -> !qco.qubit {
    %r = qco.if %a args(%t = %q) -> !qco.qubit {
      %h = qco.h %t : !qco.qubit -> !qco.qubit
      qco.yield %h : !qco.qubit
    } else args(%t = %q) {
      %x = qco.x %t : !qco.qubit -> !qco.qubit
      qco.yield %x : !qco.qubit
    }
    return %r : !qco.qubit
  })";
  auto rhs = lhs;
  rhs.replace(rhs.find("qco.if %a"), std::string("qco.if %a").size(),
              "qco.if %b");
  expectComparison(lhs, rhs);
}

TEST(IRVerificationTest, QCOSwitchSelectorMatters) {
  const std::string lhs =
      R"(func.func @f(%a: index, %b: index, %q: !qco.qubit) -> !qco.qubit {
    %r = qco.index_switch %a -> !qco.qubit
    case 0 args(%t = %q) {
      %h = qco.h %t : !qco.qubit -> !qco.qubit
      qco.yield %h : !qco.qubit
    }
    default args(%t = %q) {
      %x = qco.x %t : !qco.qubit -> !qco.qubit
      qco.yield %x : !qco.qubit
    }
    return %r : !qco.qubit
  })";
  auto rhs = lhs;
  rhs.replace(rhs.find("qco.index_switch %a"),
              std::string("qco.index_switch %a").size(), "qco.index_switch %b");
  expectComparison(lhs, rhs);
}

TEST(IRVerificationTest, BranchDestinationMatters) {
  const std::string lhs = R"(func.func @f(%c: i1) -> i64 {
    cf.cond_br %c, ^yes, ^no
  ^yes:
    %one = arith.constant 1 : i64
    return %one : i64
  ^no:
    %zero = arith.constant 0 : i64
    return %zero : i64
  })";
  auto rhs = lhs;
  rhs.replace(rhs.find("^yes, ^no"), std::string("^yes, ^no").size(),
              "^no, ^yes");
  expectComparison(lhs, rhs);
}

TEST(IRVerificationTest, ClassicalFloatingConstantsCompareExactly) {
  const std::string lhs = R"(func.func @f(%threshold: f64) -> i1 {
    %value = arith.constant 0.0 : f64
    %equal = arith.cmpf oeq, %value, %threshold : f64
    return %equal : i1
  })";
  auto rhs = lhs;
  rhs.replace(rhs.find("0.0"), 3, "1.0e-16");
  expectComparison(lhs, rhs);
}

TEST(IRVerificationTest, ModuleAttributesMatter) {
  expectComparison("module attributes {test.value = 1 : i32} {}",
                   "module attributes {test.value = 2 : i32} {}");
}

TEST(IRVerificationTest, IndependentQCOGatesCanBeReordered) {
  expectComparison(
      R"(func.func @f(%a: !qco.qubit, %b: !qco.qubit) -> (!qco.qubit, !qco.qubit) {
    %h = qco.h %a : !qco.qubit -> !qco.qubit
    %x = qco.x %b : !qco.qubit -> !qco.qubit
    return %h, %x : !qco.qubit, !qco.qubit
  })",
      R"(func.func @f(%a: !qco.qubit, %b: !qco.qubit) -> (!qco.qubit, !qco.qubit) {
    %x = qco.x %b : !qco.qubit -> !qco.qubit
    %h = qco.h %a : !qco.qubit -> !qco.qubit
    return %h, %x : !qco.qubit, !qco.qubit
  })",
      true);
}

TEST(IRVerificationTest, IndependentLinearQubitDisposalCanBeReordered) {
  expectComparison(R"(func.func @f(%a: !qco.qubit, %b: !qco.qubit) {
    qco.sink %a : !qco.qubit
    qco.sink %b : !qco.qubit
    return
  })",
                   R"(func.func @f(%a: !qco.qubit, %b: !qco.qubit) {
    qco.sink %b : !qco.qubit
    qco.sink %a : !qco.qubit
    return
  })",
                   true);
}

TEST(IRVerificationTest, ModuleSymbolOrderCanDiffer) {
  expectComparison("func.func private @a() func.func private @b()",
                   "func.func private @b() func.func private @a()", true);
}

TEST(IRVerificationTest, IndependentClassicalAllocationCanMove) {
  expectComparison(R"(func.func @f(%q: !qc.qubit) -> !cbit.reg<1> {
    %c = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
    qc.h %q : !qc.qubit
    return %c : !cbit.reg<1>
  })",
                   R"(func.func @f(%q: !qc.qubit) -> !cbit.reg<1> {
    qc.h %q : !qc.qubit
    %c = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
    return %c : !cbit.reg<1>
  })",
                   true);
}

TEST(IRVerificationTest, ConsecutiveQCDisposalsCanCommute) {
  expectComparison(R"(func.func @f(%a: !qc.qubit, %b: !qc.qubit) {
    qc.dealloc %a : !qc.qubit
    qc.dealloc %b : !qc.qubit
    return
  })",
                   R"(func.func @f(%a: !qc.qubit, %b: !qc.qubit) {
    qc.dealloc %b : !qc.qubit
    qc.dealloc %a : !qc.qubit
    return
  })",
                   true);
}

TEST(IRVerificationTest, DisposalCannotCrossAnotherEffect) {
  expectComparison(R"(func.func @f(%a: !qc.qubit, %b: !qc.qubit) {
    qc.dealloc %a : !qc.qubit
    qc.h %b : !qc.qubit
    qc.dealloc %b : !qc.qubit
    return
  })",
                   R"(func.func @f(%a: !qc.qubit, %b: !qc.qubit) {
    qc.h %b : !qc.qubit
    qc.dealloc %a : !qc.qubit
    qc.dealloc %b : !qc.qubit
    return
  })");
}

TEST(IRVerificationTest, UnknownCallOrderMatters) {
  expectComparison(R"(func.func private @a() func.func private @b()
    func.func @f() { func.call @a() : () -> ()
      func.call @b() : () -> () return })",
                   R"(func.func private @a() func.func private @b()
    func.func @f() { func.call @b() : () -> ()
      func.call @a() : () -> () return })");
}

TEST(IRVerificationTest, QIRReleaseCallOrderIsPreserved) {
  expectComparison(R"(llvm.func @__quantum__rt__result_release(!llvm.ptr)
    llvm.func @f(%a: !llvm.ptr, %b: !llvm.ptr) {
      llvm.call @__quantum__rt__result_release(%a) : (!llvm.ptr) -> ()
      llvm.call @__quantum__rt__result_release(%b) : (!llvm.ptr) -> ()
      llvm.return })",
                   R"(llvm.func @__quantum__rt__result_release(!llvm.ptr)
    llvm.func @f(%a: !llvm.ptr, %b: !llvm.ptr) {
      llvm.call @__quantum__rt__result_release(%b) : (!llvm.ptr) -> ()
      llvm.call @__quantum__rt__result_release(%a) : (!llvm.ptr) -> ()
      llvm.return })");
}

TEST(IRVerificationTest, ForwardSSADefinitionsMustAgreeWithTheirUses) {
  const std::string source = R"(func.func @f() -> i32 {
    cf.br ^definition
  ^use:
    return %first : i32
  ^definition:
    %first = arith.constant 42 : i32
    %second = arith.constant 7 : i32
    cf.br ^use
  })";
  expectComparison(source, source, true, true);
  auto otherUse = source;
  otherUse.replace(otherUse.find("return %first"),
                   std::string("return %first").size(), "return %second");
  expectComparison(source, otherUse);
  auto otherDefinition = source;
  otherDefinition.replace(otherDefinition.find("42"), 2, "43");
  expectComparison(source, otherDefinition);
}

TEST(IRVerificationTest, TensorWritesToTheSameSlotKeepTheirOrder) {
  const std::string source =
      R"(func.func @f(%a: !qco.qubit, %b: !qco.qubit, %index: index)
      -> tensor<1x!qco.qubit> {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %tensor = qtensor.alloc(%one) : tensor<1x!qco.qubit>
    %first = qtensor.insert %a into %tensor[%zero] : tensor<1x!qco.qubit>
    %second = qtensor.insert %b into %first[%zero] : tensor<1x!qco.qubit>
    return %second : tensor<1x!qco.qubit>
  })";
  auto reversed = source;
  reversed.replace(reversed.find("insert %a"), 9, "insert %b");
  reversed.replace(reversed.find("insert %b into %first"), 9, "insert %a");
  expectComparison(source, reversed);
  auto dynamic = source;
  while (dynamic.find("[%zero]") != std::string::npos) {
    dynamic.replace(dynamic.find("[%zero]"), 7, "[%index]");
  }
  reversed = dynamic;
  reversed.replace(reversed.find("insert %a"), 9, "insert %b");
  reversed.replace(reversed.find("insert %b into %first"), 9, "insert %a");
  expectComparison(dynamic, reversed);
}

TEST(IRVerificationTest, TensorWritesToDistinctConstantSlotsCanCommute) {
  const std::string source = R"(func.func @f(%a: !qco.qubit, %b: !qco.qubit)
      -> tensor<2x!qco.qubit> {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %tensor = qtensor.alloc(%two) : tensor<2x!qco.qubit>
    %first = qtensor.insert %a into %tensor[%zero] : tensor<2x!qco.qubit>
    %second = qtensor.insert %b into %first[%one] : tensor<2x!qco.qubit>
    return %second : tensor<2x!qco.qubit>
  })";
  auto reversed = source;
  reversed.replace(reversed.find("%a into %tensor[%zero]"),
                   std::string("%a into %tensor[%zero]").size(),
                   "%b into %tensor[%one]");
  reversed.replace(reversed.find("%b into %first[%one]"),
                   std::string("%b into %first[%one]").size(),
                   "%a into %first[%zero]");
  expectComparison(source, reversed, true);
}

TEST(IRVerificationTest, TensorComparisonWaitsForItsAllocation) {
  const std::string source = R"(func.func @f(%size: index) -> !qco.qubit {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %length = arith.addi %size, %one : index
    %tensor = qtensor.alloc(%length) : tensor<?x!qco.qubit>
    %rest, %q = qtensor.extract %tensor[%zero] : tensor<?x!qco.qubit>
    %h = qco.h %q : !qco.qubit -> !qco.qubit
    qtensor.dealloc %rest : tensor<?x!qco.qubit>
    return %h : !qco.qubit
  })";
  auto different = source;
  different.replace(different.find("qco.h"), 5, "qco.x");
  expectComparison(source, different);
}

TEST(IRVerificationTest, QCOYieldOrderMattersInControlFlowRegions) {
  const std::array bodies = {
      R"(
      %r:2 = qco.if %flag args(%u = %a, %v = %b) -> (!qco.qubit, !qco.qubit) {
        qco.yield %v, %u : !qco.qubit, !qco.qubit
      } else args(%u = %a, %v = %b) {
        qco.yield %u, %v : !qco.qubit, !qco.qubit
      }
    )",
      R"(
      %r:2 = qco.index_switch %index -> (!qco.qubit, !qco.qubit)
      case 0 args(%u = %a, %v = %b) {
        qco.yield %v, %u : !qco.qubit, !qco.qubit
      }
      default args(%u = %a, %v = %b) {
        qco.yield %u, %v : !qco.qubit, !qco.qubit
      }
    )",
  };
  for (const auto* body : bodies) {
    SCOPED_TRACE(body);
    const std::string source = std::string(R"(
      func.func @f(%flag: i1, %index: index, %a: !qco.qubit, %b: !qco.qubit)
          -> (!qco.qubit, !qco.qubit) {
    )") + body + R"(
      return %r#0, %r#1 : !qco.qubit, !qco.qubit
    })";
    auto swapped = source;
    swapped.replace(swapped.find("qco.yield %v, %u"),
                    std::string("qco.yield %v, %u").size(), "qco.yield %u, %v");
    expectComparison(source, swapped);
  }
}

TEST(IRVerificationTest, QCOYieldFollowsConsistentlyPermutedParentResults) {
  expectComparison(R"(func.func @f(%c: i1, %a: !qco.qubit, %b: !qco.qubit)
      -> (!qco.qubit, !qco.qubit) {
    %r:2 = qco.if %c args(%u = %a, %v = %b) -> (!qco.qubit, !qco.qubit) {
      qco.yield %v, %u : !qco.qubit, !qco.qubit
    } else args(%u = %a, %v = %b) {
      qco.yield %u, %v : !qco.qubit, !qco.qubit
    }
    return %r#0, %r#1 : !qco.qubit, !qco.qubit
  })",
                   R"(func.func @f(%c: i1, %a: !qco.qubit, %b: !qco.qubit)
      -> (!qco.qubit, !qco.qubit) {
    %r:2 = qco.if %c args(%v = %b, %u = %a) -> (!qco.qubit, !qco.qubit) {
      qco.yield %u, %v : !qco.qubit, !qco.qubit
    } else args(%v = %b, %u = %a) {
      qco.yield %v, %u : !qco.qubit, !qco.qubit
    }
    return %r#1, %r#0 : !qco.qubit, !qco.qubit
  })",
                   true);
}

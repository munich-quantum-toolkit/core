/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "OpenQASMTestUtils.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Target/OpenQASM/Frontend.h"
#include "qasm_programs.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace mlir;
using namespace mlir::oq3::test;

namespace {

TEST(OpenQASMTargetTest, EmitsVerifiedQCDirectly) {
  MLIRContext context;
  auto module = qc::translateQASM3ToQC(BROADCAST_PROGRAM, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t gates = 0;
  module->walk([&](qc::HOp) { ++gates; });
  EXPECT_EQ(gates, 2);
}

TEST(OpenQASMTargetTest, ProductionTranslationUsesTheStagedPipeline) {
  MLIRContext context;
  auto module = qc::translateQASM3ToQC(BROADCAST_PROGRAM, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));

  bool hasQuantumOperation = false;
  module->walk([&](Operation* operation) {
    hasQuantumOperation |= isa<qc::HOp>(operation);
  });
  EXPECT_TRUE(hasQuantumOperation);
}

TEST(OpenQASMTargetTest, EmitsTypedMixedNumericGateExpressions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
gate shifted(theta) q {
  rx(theta + 1) q;
}
qubit q;
shifted(0.5) q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t numericCasts = 0;
  module->walk([&](Operation* operation) {
    numericCasts += isa<arith::SIToFPOp, arith::UIToFPOp>(operation);
  });
  EXPECT_EQ(numericCasts, 1);
}

TEST(OpenQASMTargetTest, EmitsScalarMathFunctions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
float value = 0.5;
qubit q;
rx(sin(value) + cos(value) + tan(value) + exp(value) + log(value) +
   sqrt(value)) q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t functions = 0;
  module->walk([&](Operation* operation) {
    functions += isa<math::SinOp, math::CosOp, math::TanOp, math::ExpOp,
                     math::LogOp, math::SqrtOp>(operation);
  });
  EXPECT_EQ(functions, 6);
}

TEST(OpenQASMTargetTest, EmitsInverseTrigFunctionsWithNumericConversions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
uint unsigned_value = 1;
int signed_value = 1;
float float_value = 0.5;
qubit q;
rx(arccos(unsigned_value) + arcsin(signed_value) + arctan(float_value)) q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t arcCosines = 0;
  size_t arcSines = 0;
  size_t arcTangents = 0;
  size_t unsignedConversions = 0;
  size_t signedConversions = 0;
  module->walk([&](Operation* operation) {
    arcCosines += isa<math::AcosOp>(operation);
    arcSines += isa<math::AsinOp>(operation);
    arcTangents += isa<math::AtanOp>(operation);
    unsignedConversions += isa<arith::UIToFPOp>(operation);
    signedConversions += isa<arith::SIToFPOp>(operation);
  });
  EXPECT_EQ(arcCosines, 1);
  EXPECT_EQ(arcSines, 1);
  EXPECT_EQ(arcTangents, 1);
  EXPECT_EQ(unsignedConversions, 1);
  EXPECT_EQ(signedConversions, 1);
}

TEST(OpenQASMTargetTest, EmitsRuntimeScalarConversionMatrix) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
float source_float = 2.5;
uint uint_from_float = source_float;
int int_from_float = source_float;
bool source_bool = true;
float float_from_bool = source_bool;
int int_from_bool = source_bool;
uint source_uint = 2;
float float_from_uint = source_uint;
int source_int = 2;
float float_from_int = source_int;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t floatToUnsigned = 0;
  size_t floatToSigned = 0;
  size_t boolToInteger = 0;
  size_t unsignedToFloat = 0;
  size_t signedToFloat = 0;
  module->walk([&](Operation* operation) {
    floatToUnsigned += isa<arith::FPToUIOp>(operation);
    floatToSigned += isa<arith::FPToSIOp>(operation);
    boolToInteger += isa<arith::ExtUIOp>(operation);
    unsignedToFloat += isa<arith::UIToFPOp>(operation);
    signedToFloat += isa<arith::SIToFPOp>(operation);
  });
  EXPECT_EQ(floatToUnsigned, 1);
  EXPECT_EQ(floatToSigned, 1);
  EXPECT_EQ(boolToInteger, 1);
  EXPECT_EQ(unsignedToFloat, 2);
  EXPECT_EQ(signedToFloat, 1);
}

TEST(OpenQASMTargetTest, EmitsEveryRuntimeComparisonPredicate) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
float float_left = 1.0;
float float_right = 2.0;
bool float_equal = float_left == float_right;
bool float_not_equal = float_left != float_right;
bool float_less = float_left < float_right;
bool float_less_equal = float_left <= float_right;
bool float_greater = float_left > float_right;
bool float_greater_equal = float_left >= float_right;
int signed_left = 1;
int signed_right = 2;
bool signed_equal = signed_left == signed_right;
bool signed_not_equal = signed_left != signed_right;
bool signed_less = signed_left < signed_right;
bool signed_less_equal = signed_left <= signed_right;
bool signed_greater = signed_left > signed_right;
bool signed_greater_equal = signed_left >= signed_right;
uint unsigned_left = 1;
uint unsigned_right = 2;
bool unsigned_equal = unsigned_left == unsigned_right;
bool unsigned_not_equal = unsigned_left != unsigned_right;
bool unsigned_less = unsigned_left < unsigned_right;
bool unsigned_less_equal = unsigned_left <= unsigned_right;
bool unsigned_greater = unsigned_left > unsigned_right;
bool unsigned_greater_equal = unsigned_left >= unsigned_right;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  std::array<size_t, 6> floatPredicates{};
  std::array<size_t, 10> integerPredicates{};
  module->walk([&](Operation* operation) {
    if (auto comparison = dyn_cast<arith::CmpFOp>(operation)) {
      switch (comparison.getPredicate()) {
      case arith::CmpFPredicate::OEQ:
        ++floatPredicates[0];
        break;
      case arith::CmpFPredicate::UNE:
        ++floatPredicates[1];
        break;
      case arith::CmpFPredicate::OLT:
        ++floatPredicates[2];
        break;
      case arith::CmpFPredicate::OLE:
        ++floatPredicates[3];
        break;
      case arith::CmpFPredicate::OGT:
        ++floatPredicates[4];
        break;
      case arith::CmpFPredicate::OGE:
        ++floatPredicates[5];
        break;
      default:
        break;
      }
    }
    if (auto comparison = dyn_cast<arith::CmpIOp>(operation)) {
      switch (comparison.getPredicate()) {
      case arith::CmpIPredicate::eq:
        ++integerPredicates[0];
        break;
      case arith::CmpIPredicate::ne:
        ++integerPredicates[1];
        break;
      case arith::CmpIPredicate::slt:
        ++integerPredicates[2];
        break;
      case arith::CmpIPredicate::sle:
        ++integerPredicates[3];
        break;
      case arith::CmpIPredicate::sgt:
        ++integerPredicates[4];
        break;
      case arith::CmpIPredicate::sge:
        ++integerPredicates[5];
        break;
      case arith::CmpIPredicate::ult:
        ++integerPredicates[6];
        break;
      case arith::CmpIPredicate::ule:
        ++integerPredicates[7];
        break;
      case arith::CmpIPredicate::ugt:
        ++integerPredicates[8];
        break;
      case arith::CmpIPredicate::uge:
        ++integerPredicates[9];
        break;
      default:
        break;
      }
    }
  });
  EXPECT_EQ(floatPredicates, (std::array<size_t, 6>{1, 1, 1, 1, 1, 1}));
  EXPECT_EQ(integerPredicates,
            (std::array<size_t, 10>{2, 2, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(OpenQASMTargetTest, FoldsAndEmitsCeilingAndFloor) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
gate rounded(theta) q {
  rx(sin(theta) + cos(theta)) q;
}
float runtime_value = 0.5;
qubit q;
rx(ceiling(1.25) + floor(-1.25)) q;
rx(ceiling(runtime_value) + floor(runtime_value)) q;
rounded(0.5) q;
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  const auto& constantApplication = std::get<oq3::frontend::GateApplication>(
      analyzed.program->statements[analyzed.program->body[2]].data);
  const auto& parameter =
      analyzed.program->expressions.at(constantApplication.parameters.front());
  ASSERT_EQ(parameter.kind, oq3::frontend::ExpressionKind::Cast);
  EXPECT_EQ(parameter.type, oq3::frontend::ScalarType::Angle);
  const auto& constant = analyzed.program->expressions.at(parameter.lhs);
  ASSERT_EQ(constant.kind, oq3::frontend::ExpressionKind::Constant);
  EXPECT_DOUBLE_EQ(std::get<double>(constant.constant), 0.0);

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t ceilings = 0;
  size_t floors = 0;
  module->walk([&](Operation* operation) {
    ceilings += isa<math::CeilOp>(operation);
    floors += isa<math::FloorOp>(operation);
  });
  EXPECT_EQ(ceilings, 1);
  EXPECT_EQ(floors, 1);
}

TEST(OpenQASMTargetTest, NestsAlternatingControlsAndFlipsPolarityOutside) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
qubit[5] q;
ctrl(2) @ negctrl @ inv @ ctrl @ x q[0], q[1], q[2], q[3], q[4];
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);

  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<size_t> controlArities;
  size_t outerPolarityFlips = 0;
  module->walk([&](Operation* operation) {
    if (auto control = dyn_cast<qc::CtrlOp>(operation)) {
      controlArities.push_back(control.getNumControls());
    }
    if (isa<qc::XOp>(operation) &&
        operation->getParentOfType<qc::CtrlOp>() == nullptr &&
        operation->getParentOfType<qc::InvOp>() == nullptr) {
      ++outerPolarityFlips;
    }
  });
  llvm::sort(controlArities);
  EXPECT_EQ(controlArities, (SmallVector<size_t>{1, 1, 2}));
  EXPECT_EQ(outerPolarityFlips, 2);
}

TEST(OpenQASMTargetTest, EmitsOneControlRegionForNegCtrlArity) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[4] q;
negctrl(3) @ x q[0], q[1], q[2], q[3];
)qasm";
  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t controlRegions = 0;
  size_t polarityFlips = 0;
  module->walk([&](qc::CtrlOp control) {
    ++controlRegions;
    EXPECT_EQ(control.getNumControls(), 3);
  });
  module->walk([&](qc::XOp operation) {
    polarityFlips += operation->getParentOfType<qc::CtrlOp>() == nullptr;
  });
  EXPECT_EQ(controlRegions, 1);
  EXPECT_EQ(polarityFlips, 6);
}

TEST(OpenQASMTargetTest, LowersDynamicPowerModifiersToQC) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
float exponent = 0.5;
qubit q;
pow(exponent) @ x q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<qc::PowOp> powers;
  module->walk([&](qc::PowOp op) { powers.push_back(op); });
  ASSERT_EQ(powers.size(), 1U);
  ASSERT_TRUE(powers.front().getExponentValue().has_value());
  EXPECT_DOUBLE_EQ(*powers.front().getExponentValue(), 0.5);
}

TEST(OpenQASMTargetTest, GuardsRuntimeIntegerPowerModifierExactness) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      R"qasm(
OPENQASM 3.1;
qubit q;
uint exponent = 9007199254740993;
pow(exponent) @ x q;
)qasm",
      R"qasm(
OPENQASM 3.1;
qubit q;
int exponent = 9007199254740992;
bit choose = measure q;
if (choose) { exponent = 9007199254740993; }
pow(exponent) @ x q;
)qasm",
      R"qasm(
OPENQASM 3.1;
qubit q;
uint exponent = 9007199254740993;
bit repeat = measure q;
while (repeat) {
  exponent -= 1;
  repeat = measure q;
}
pow(exponent) @ x q;
)qasm",
  });

  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    MLIRContext context;
    auto module = qc::translateQASM3ToQC(source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));

    SmallVector<qc::PowOp> powers;
    size_t exactnessAssertions = 0;
    module->walk([&](Operation* operation) {
      if (auto power = dyn_cast<qc::PowOp>(operation)) {
        powers.push_back(power);
      }
      if (auto assertion = dyn_cast<cf::AssertOp>(operation);
          assertion &&
          assertion.getMsg().contains(
              "power modifier exponent cannot be represented exactly")) {
        ++exactnessAssertions;
      }
    });
    ASSERT_EQ(powers.size(), 1U);
    EXPECT_FALSE(powers.front().getExponentValue().has_value());
    EXPECT_EQ(exactnessAssertions, 1U);
  }
}

TEST(OpenQASMTargetTest,
     LowersCustomGatesConditionalsAndQuantumRuntimeOperations) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate pair(theta) left, right {
  rx(theta) left;
  cx left, right;
}
qubit[2] q;
bit c = measure q[0];
if (!c) {
  pair(0.5) q[0], q[1];
} else {
  reset q[1];
}
barrier q;
output bit[2] out;
out = measure q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t conditionals = 0;
  module->walk([&](Operation* operation) {
    conditionals += operation->getName().getStringRef() == "scf.if";
  });
  EXPECT_EQ(conditionals, 1);

  ASSERT_TRUE(succeeded(verify(*module)));

  size_t resets = 0;
  size_t barriers = 0;
  module->walk([&](Operation* operation) {
    auto name = operation->getName().getStringRef();
    resets += name == "qc.reset";
    barriers += name == "qc.barrier";
  });
  EXPECT_EQ(resets, 1);
  EXPECT_EQ(barriers, 1);
}

TEST(OpenQASMTargetTest, ResolvesManyCustomGateDefinitionsThroughTheIndex) {
  constexpr size_t definitionCount = 2048;
  std::string source = "OPENQASM 3.1;\nqubit q;\n";
  source.reserve(definitionCount * 40);
  for (size_t index = 0; index < definitionCount; ++index) {
    source += "gate g" + std::to_string(index) + " target { x target; }\n";
  }
  for (size_t index = 0; index < definitionCount; ++index) {
    source += "g" + std::to_string(index) + " q;\n";
  }

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t xGates = 0;
  module->walk(
      [&](Operation* operation) { xGates += isa<qc::XOp>(operation); });
  EXPECT_EQ(xGates, definitionCount);
}

TEST(OpenQASMTargetTest, LowersOpenQASM2ControlledGateCompatibilityPrefixes) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
cccx q[0], q[1], q[2], q[3];
measure q -> c;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t controls = 0;
  module->walk(
      [&](Operation* operation) { controls += isa<qc::CtrlOp>(operation); });
  EXPECT_EQ(controls, 1);
}

TEST(OpenQASMTargetTest, LowersLanguageBuiltinsOnHardwareQubits) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
gphase(pi / 2);
x $3;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t globalPhases = 0;
  size_t xGates = 0;
  module->walk([&](Operation* operation) {
    globalPhases += isa<qc::GPhaseOp>(operation);
    xGates += isa<qc::XOp>(operation);
  });
  EXPECT_EQ(globalPhases, 1);
  EXPECT_EQ(xGates, 1);
}

TEST(OpenQASMTargetTest, RejectsMixedQubitAllocationAtTheQCTarget) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
x q;
x $0;
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;

  MLIRContext context;
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    return success();
  });
  auto module = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(module);
  EXPECT_NE(diagnostic.find("mixing physical and declared qubits"),
            std::string::npos)
      << diagnostic;
  EXPECT_NE(diagnostic.find("QC target"), std::string::npos) << diagnostic;
}

TEST(OpenQASMTargetTest, AllocatesClassicalStorageOnlyForBitOutputs) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
bit local = measure q;
if (local) {
  x q;
}
output bit result;
result = measure q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t classicalAllocations = 0;
  size_t classicalLoads = 0;
  size_t classicalStores = 0;
  module->walk([&](Operation* operation) {
    if (auto allocation = dyn_cast<memref::AllocOp>(operation);
        allocation && allocation.getType().getElementType().isInteger(1)) {
      ++classicalAllocations;
    }
    if (auto load = dyn_cast<memref::LoadOp>(operation);
        load && load.getMemRefType().getElementType().isInteger(1)) {
      ++classicalLoads;
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation);
        store && store.getMemRefType().getElementType().isInteger(1)) {
      ++classicalStores;
    }
  });
  EXPECT_EQ(classicalAllocations, 1);
  EXPECT_EQ(classicalLoads, 0);
  EXPECT_EQ(classicalStores, 1);
}

TEST(OpenQASMTargetTest, PreservesOrderedScalarAndRegisterOutputs) {
  constexpr llvm::StringLiteral implicitSource = R"qasm(
OPENQASM 3.1;
int count = 1;
bit[2] bits;
bits[0] = true;
bits[1] = false;
float ratio = 2.0;
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(implicitSource);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->outputs.size(), 3);
  EXPECT_EQ(analyzed.program->outputs[0].kind,
            oq3::frontend::OutputKind::Scalar);
  EXPECT_EQ(analyzed.program->scalars[analyzed.program->outputs[0].symbol].name,
            "count");
  EXPECT_EQ(analyzed.program->outputs[1].kind,
            oq3::frontend::OutputKind::BitRegister);
  EXPECT_EQ(
      analyzed.program->registers[analyzed.program->outputs[1].symbol].name,
      "bits");
  EXPECT_EQ(analyzed.program->outputs[2].kind,
            oq3::frontend::OutputKind::Scalar);
  EXPECT_EQ(analyzed.program->scalars[analyzed.program->outputs[2].symbol].name,
            "ratio");

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(implicitSource, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  func::ReturnOp result;
  module->walk([&](func::ReturnOp operation) { result = operation; });
  ASSERT_TRUE(result);
  ASSERT_EQ(result.getNumOperands(), 3);
  EXPECT_TRUE(result.getOperand(0).getType().isInteger(64));
  const auto bitType = dyn_cast<MemRefType>(result.getOperand(1).getType());
  ASSERT_TRUE(bitType);
  EXPECT_EQ(bitType.getShape(), ArrayRef<int64_t>{2});
  EXPECT_TRUE(bitType.getElementType().isInteger(1));
  EXPECT_TRUE(result.getOperand(2).getType().isF64());

  constexpr llvm::StringLiteral explicitSource = R"qasm(
OPENQASM 3.1;
int internal = 1;
output int result;
result = internal + 1;
)qasm";
  auto explicitAnalysis = oq3::frontend::analyzeOpenQASM(explicitSource);
  ASSERT_TRUE(explicitAnalysis) << explicitAnalysis.diagnostics.front().message;
  ASSERT_EQ(explicitAnalysis.program->outputs.size(), 1);
  EXPECT_EQ(explicitAnalysis.program->outputs.front().kind,
            oq3::frontend::OutputKind::Scalar);
  EXPECT_EQ(explicitAnalysis.program
                ->scalars[explicitAnalysis.program->outputs.front().symbol]
                .name,
            "result");
}

TEST(OpenQASMTargetTest, RejectsExcessiveDynamicDispatch) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[317] q;
qubit[317] aux;
int i = 0;
  int j = 1;
cx q[i], aux[j];
)qasm";

  MLIRContext context;
  std::string diagnostic;
  Location location = UnknownLoc::get(&context);
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    location = value.getLocation();
    return success();
  });
  auto module = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(module);
  const auto fileLocation = dyn_cast<FileLineColLoc>(location);
  ASSERT_TRUE(fileLocation);
  EXPECT_EQ(fileLocation.getFilename(), "<input>");
  EXPECT_EQ(fileLocation.getLine(), 8);
  EXPECT_EQ(fileLocation.getColumn(), 1);
  EXPECT_NE(diagnostic.find("projected emitted operation count"),
            std::string::npos);
}

TEST(OpenQASMTargetTest, RejectsExcessiveCustomGateExpansion) {
  std::string source = "OPENQASM 3.1;\n"
                       "include \"stdgates.inc\";\n"
                       "gate g0 q { x q; }\n";
  for (size_t level = 1; level <= 17; ++level) {
    source += "gate g" + std::to_string(level) + " q { g" +
              std::to_string(level - 1) + " q; g" + std::to_string(level - 1) +
              " q; }\n";
  }
  source += "qubit q;\ng17 q;\n";

  MLIRContext context;
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    return success();
  });
  auto module = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(module);
  EXPECT_NE(diagnostic.find("projected emitted operation count"),
            std::string::npos);
}

TEST(OpenQASMTargetTest, ComposesDispatchAndCustomGateExpansionBudgets) {
  std::string source = "OPENQASM 3.1;\n"
                       "include \"stdgates.inc\";\n"
                       "gate expanded a, b {\n";
  for (size_t operation = 0; operation < 25; ++operation) {
    source += operation % 2 == 0 ? "  x a;\n" : "  x b;\n";
  }
  source += "}\n"
            "qubit[64] q;\n"
            "qubit[64] aux;\n"
            "int i = 0;\n"
            "int j = 1;\n"
            "expanded q[i], aux[j];\n";

  MLIRContext context;
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    return success();
  });
  auto module = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(module);
  EXPECT_NE(diagnostic.find("projected emitted operation count"),
            std::string::npos);
}

TEST(OpenQASMTargetTest, RejectsWideConstructionBeforeEmittingOperations) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1;\nqubit[50001] q;\n",
      "OPENQASM 3.1;\nbit[40000] c;\nint i = 0;\n"
      "output bit value;\nvalue = false;\n"
      "c[i] = value;\n",
  });
  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    MLIRContext context;
    std::string diagnostic;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
      diagnostic = value.str();
      return success();
    });
    auto module = qc::translateQASM3ToQC(source, &context);
    EXPECT_FALSE(module);
    EXPECT_NE(diagnostic.find("projected emitted operation count"),
              std::string::npos)
        << diagnostic;
  }
}

TEST(OpenQASMTargetTest, BudgetsScalarAndStructuredOperationConstruction) {
  std::string expressionSource =
      "OPENQASM 3.1;\nint operand = 1;\nint result = ";
  std::vector<std::string> expressions(16384, "operand");
  while (expressions.size() > 1) {
    std::vector<std::string> next;
    next.reserve(expressions.size() / 2);
    for (size_t expression = 0; expression < expressions.size();
         expression += 2) {
      next.push_back("(" + expressions[expression] + " + " +
                     expressions[expression + 1] + ")");
    }
    expressions = std::move(next);
  }
  expressionSource += expressions.front();
  expressionSource += ";\n";

  std::string controlFlowSource = "OPENQASM 3.1;\nint operand = 1;\n";
  constexpr size_t conditionals = 12000;
  for (size_t conditional = 0; conditional < conditionals; ++conditional) {
    controlFlowSource += "if (operand > 0) {}\n";
  }

  std::string phaseSource = "OPENQASM 3.1;\nqubit q;\n";
  // Setup costs eight operations and each OpenQASM 3 U application costs
  // three parameter constants plus four phase-aware lowering operations.
  constexpr size_t phaseGates = ((100000 - 8) / 7) + 1;
  static_assert(8 + ((phaseGates - 1) * 7) <= 100000);
  static_assert(8 + (phaseGates * 7) > 100000);
  for (size_t gate = 0; gate < phaseGates; ++gate) {
    phaseSource += "U(0.1, 0.2, 0.3) q;\n";
  }

  std::string powerSource = "OPENQASM 3.1;\nqubit q;\nint exponent = 1;\n";
  constexpr size_t powerGates = 6000;
  for (size_t gate = 0; gate < powerGates; ++gate) {
    powerSource += "pow(exponent) @ x q;\n";
  }

  for (const auto* source :
       {&expressionSource, &controlFlowSource, &phaseSource, &powerSource}) {
    MLIRContext context;
    std::string diagnostic;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
      diagnostic = value.str();
      return success();
    });
    auto module = qc::translateQASM3ToQC(*source, &context);
    EXPECT_FALSE(module);
    EXPECT_NE(diagnostic.find("projected emitted operation count"),
              std::string::npos)
        << diagnostic;
  }
}

TEST(OpenQASMTargetTest, BudgetsLinearBitVectorPackingWork) {
  constexpr size_t width = 12501;
  std::string source =
      "OPENQASM 3.1;\noutput bit[" + std::to_string(width) + "] value;\n";
  for (size_t bit = 0; bit < width; ++bit) {
    source += "value[" + std::to_string(bit) + "] = false;\n";
  }
  source += "int distance = 1;\nvalue = rotl(value, distance);\n";

  MLIRContext context;
  ScopedDiagnosticHandler diagnostics(&context,
                                      [](Diagnostic&) { return success(); });
  auto module = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(module);
}

TEST(OpenQASMTargetTest, LowersGateBodyLoopsAndBuiltinConstants) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate repeated(theta) q {
  for int i in [0:2] { rx(theta + pi + i) q; }
  while (false) { x q; }
}
qubit q;
repeated(0.5) q;
bit result = measure q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t forLoops = 0;
  size_t whileLoops = 0;
  module->walk([&](Operation* operation) {
    forLoops += isa<scf::ForOp>(operation);
    whileLoops += isa<scf::WhileOp>(operation);
  });
  EXPECT_EQ(forLoops, 1);
  EXPECT_EQ(whileLoops, 1);

  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OpenQASMTargetTest, GateDefinitionsCaptureGlobalConstants) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
const float theta = pi / 2;
gate g q { rx(theta) q; }
qubit q;
g q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  qc::RXOp rotation;
  module->walk([&](qc::RXOp application) { rotation = application; });
  ASSERT_TRUE(rotation);
  FloatAttr angle;
  EXPECT_TRUE(matchPattern(rotation.getParameter(0), m_Constant(&angle)));
  EXPECT_DOUBLE_EQ(angle.getValueAsDouble(), std::numbers::pi / 2);
}

TEST(OpenQASMTargetTest, SupportsWholeBitRegisterAssignment) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
bit[2] source = measure q;
output bit[2] target;
target = source;
if (target[0] || target[1]) { x q[0]; }
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  const oq3::frontend::BitVectorAssignmentStatement* assignment = nullptr;
  for (const auto& statement : analyzed.program->statements) {
    if (const auto* current =
            std::get_if<oq3::frontend::BitVectorAssignmentStatement>(
                &statement.data)) {
      ASSERT_EQ(assignment, nullptr);
      assignment = current;
    }
  }
  ASSERT_NE(assignment, nullptr);
  ASSERT_EQ(analyzed.program->registers.size(), 3);
  EXPECT_EQ(analyzed.program->registers[assignment->target].name, "target");
  const auto& value =
      analyzed.program->bitVectorExpressions.at(assignment->value);
  EXPECT_EQ(value.kind, oq3::frontend::BitVectorExpressionKind::Register);
  EXPECT_EQ(analyzed.program->registers[value.reg].name, "source");
  EXPECT_EQ(value.width, 2);

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<Value> measured;
  module->walk([&](qc::MeasureOp measurement) {
    measured.push_back(measurement.getResult());
  });
  const auto returned = returnedBitValues(*module);
  ASSERT_EQ(measured.size(), 2);
  ASSERT_EQ(returned.size(), 2);
  EXPECT_EQ(returned[0], measured[0]);
  EXPECT_EQ(returned[1], measured[1]);
}

TEST(OpenQASMTargetTest, LowersTypedBitVectorBuiltins) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
output bit[5] value;
value[0] = true;
value[1] = false;
value[2] = true;
value[3] = false;
value[4] = true;
uint count = popcount(value);
value = rotl(value, 0);
value = rotr(value, -7);
qubit q;
if (count == 3) { x q; }
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_FALSE(analyzed.program->bitVectorExpressions.empty());
  EXPECT_TRUE(
      llvm::any_of(analyzed.program->expressions, [](const auto& expression) {
        return expression.kind == oq3::frontend::ExpressionKind::PopCount &&
               expression.type == oq3::frontend::ScalarType::Uint;
      }));

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t populationCounts = 0;
  size_t funnelShifts = 0;
  module->walk([&](Operation* operation) {
    populationCounts += isa<math::CtPopOp>(operation);
    funnelShifts += isa<LLVM::FshlOp, LLVM::FshrOp>(operation);
  });
  EXPECT_EQ(populationCounts, 1);
  // Both rotation distances are constant and therefore only permute SSA values.
  EXPECT_EQ(funnelShifts, 0);
}

TEST(OpenQASMTargetTest, ReusesPackedNestedDynamicRotations) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
bit[5] value;
value[0] = true;
value[1] = false;
value[2] = true;
value[3] = false;
value[4] = true;
int distance = -7;
uint count = popcount(rotl(rotr(value, distance), 1));
qubit q;
if (count == 3) { x q; }
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t leftShifts = 0;
  size_t rightShifts = 0;
  size_t populationCounts = 0;
  size_t unpackingTruncations = 0;
  module->walk([&](Operation* operation) {
    leftShifts += isa<LLVM::FshlOp>(operation);
    rightShifts += isa<LLVM::FshrOp>(operation);
    populationCounts += isa<math::CtPopOp>(operation);
    if (auto truncation = dyn_cast<arith::TruncIOp>(operation);
        truncation && truncation.getOut().getType().isInteger(1)) {
      ++unpackingTruncations;
    }
  });
  EXPECT_EQ(leftShifts, 1);
  EXPECT_EQ(rightShifts, 1);
  EXPECT_EQ(populationCounts, 1);
  // The nested packed value reaches popcount without an unpack/repack cycle.
  EXPECT_EQ(unpackingTruncations, 0);
}

TEST(OpenQASMTargetTest, CarriesAtomicRotationsThroughControlFlow) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
output bit[5] value;
value[0] = true;
value[1] = false;
value[2] = true;
value[3] = false;
value[4] = true;
qubit q;
bit condition = measure q;
if (condition) {
  value = rotl(value, 1);
}
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  bool carriedWholeRegister = false;
  module->walk([&](scf::IfOp conditional) {
    carriedWholeRegister |= conditional.getNumResults() == 5;
  });
  EXPECT_TRUE(carriedWholeRegister);
}

TEST(OpenQASMTargetTest, SelfRotationSnapshotsTheWholeRegister) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[5] q;
output bit[5] result;
result = measure q;
result = rotl(result, 2);
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<Value> measured;
  module->walk([&](qc::MeasureOp measurement) {
    measured.push_back(measurement.getResult());
  });
  const auto returned = returnedBitValues(*module);
  ASSERT_EQ(measured.size(), 5);
  ASSERT_EQ(returned.size(), 5);
  EXPECT_EQ(returned[0], measured[3]);
  EXPECT_EQ(returned[1], measured[4]);
  EXPECT_EQ(returned[2], measured[0]);
  EXPECT_EQ(returned[3], measured[1]);
  EXPECT_EQ(returned[4], measured[2]);
}

TEST(OpenQASMTargetTest, SupportsWidthOneBitVectorBuiltins) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
output bit[1] value;
value[0] = measure q;
int distance = -3;
value = rotl(value, distance);
value = rotr(value, 4);
uint count = popcount(value);
rx(count) q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t populationCounts = 0;
  size_t leftShifts = 0;
  size_t rightShifts = 0;
  module->walk([&](Operation* operation) {
    populationCounts += isa<math::CtPopOp>(operation);
    leftShifts += isa<LLVM::FshlOp>(operation);
    rightShifts += isa<LLVM::FshrOp>(operation);
  });
  EXPECT_EQ(populationCounts, 1);
  EXPECT_EQ(leftShifts, 1);
  EXPECT_EQ(rightShifts, 0);
}

TEST(OpenQASMTargetTest, SupportsWideBitVectorBuiltins) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[65] q;
bit[65] value = measure q;
int distance = 3;
value = rotl(value, distance);
uint count = popcount(value);
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t wideFunnelShifts = 0;
  size_t populationCounts = 0;
  size_t narrowedCounts = 0;
  module->walk([&](Operation* operation) {
    if (auto shift = dyn_cast<LLVM::FshlOp>(operation)) {
      wideFunnelShifts += shift.getResult().getType().isInteger(65);
    }
    if (auto count = dyn_cast<math::CtPopOp>(operation)) {
      populationCounts += count.getResult().getType().isInteger(65);
    }
    if (auto truncation = dyn_cast<arith::TruncIOp>(operation)) {
      narrowedCounts += truncation.getIn().getType().isInteger(65) &&
                        truncation.getOut().getType().isInteger(64);
    }
  });
  EXPECT_EQ(wideFunnelShifts, 1);
  EXPECT_EQ(populationCounts, 1);
  EXPECT_EQ(narrowedCounts, 1);
}

TEST(OpenQASMTargetTest, RotationsProduceSpecifiedBitResults) {
  constexpr std::array input{true, false, true, true, false};
  constexpr std::array<int64_t, 5> distances{0, 2, -2, 7, -7};
  std::string source = "OPENQASM 3.1;\n";
  std::vector<std::vector<bool>> expectedResults;
  size_t resultIndex = 0;
  for (const bool runtime : {false, true}) {
    for (const auto distance : distances) {
      for (const bool left : {true, false}) {
        const auto resultName = "result" + std::to_string(resultIndex);
        source += "output bit[5] " + resultName + ";\n";
        for (size_t bit = 0; bit < input.size(); ++bit) {
          source += resultName + "[" + std::to_string(bit) +
                    "] = " + (input[bit] ? "true;\n" : "false;\n");
        }
        std::string distanceExpression = std::to_string(distance);
        if (runtime) {
          const auto distanceName = "distance" + std::to_string(resultIndex);
          source.append("int ")
              .append(distanceName)
              .append(" = ")
              .append(distanceExpression)
              .append(";\n");
          distanceExpression = distanceName;
        }
        source.append(resultName)
            .append(" = ")
            .append(left ? "rotl(" : "rotr(")
            .append(resultName)
            .append(", ")
            .append(distanceExpression)
            .append(");\n");
        expectedResults.push_back(rotateBits(input, distance, left));
        ++resultIndex;
      }
    }
  }

  const auto outputs = canonicalizedBitOutputs(source);
  ASSERT_EQ(outputs.size(), expectedResults.size() * input.size());
  std::vector<std::vector<bool>> actualResults;
  actualResults.reserve(expectedResults.size());
  for (size_t result = 0; result < expectedResults.size(); ++result) {
    const auto begin =
        outputs.begin() + static_cast<ptrdiff_t>(result * input.size());
    actualResults.emplace_back(begin, begin + input.size());
    EXPECT_EQ(actualResults.back(), expectedResults[result])
        << "rotation result " << result;
  }

  for (size_t runtime = 0; runtime < 2; ++runtime) {
    for (size_t distance = 0; distance < distances.size(); ++distance) {
      const auto opposite = std::ranges::find(distances, -distances[distance]);
      ASSERT_NE(opposite, distances.end());
      const auto oppositeIndex =
          static_cast<size_t>(opposite - distances.begin());
      const auto left = ((runtime * distances.size()) + distance) * 2;
      const auto oppositeRight =
          (((runtime * distances.size()) + oppositeIndex) * 2) + 1;
      EXPECT_EQ(actualResults[left], actualResults[oppositeRight])
          << "rotl(a, n) differs from rotr(a, -n) for n = "
          << distances[distance];
    }
  }
}

TEST(OpenQASMTargetTest, PopcountProducesSpecifiedResult) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
bit[5] source;
source[0] = true;
source[1] = false;
source[2] = true;
source[3] = true;
source[4] = false;
output bit[6] result;
result[0] = false;
result[1] = false;
result[2] = false;
result[3] = false;
result[4] = false;
result[5] = false;
result[popcount(source)] = true;
)qasm";

  EXPECT_EQ(canonicalizedBitOutputs(source),
            (std::vector<bool>{false, false, false, true, false, false}));
}

TEST(OpenQASMTargetTest, SupportsOpenQASM2RegisterConditions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
measure q -> c;
if (c == 1) x q[0];
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t conditionals = 0;
  module->walk([&](scf::IfOp) { ++conditionals; });
  // The register equality and the source-level branch each short-circuit
  // through their own structured conditional.
  EXPECT_EQ(conditionals, 2);
}

TEST(OpenQASMTargetTest, SelectsFloatingPowForNegativeSignedExponent) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
int base = 4;
float result = pow(base, -2);
qubit q;
if (result == 0.0625) { x q; }
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  bool foundResult = false;
  module->walk([&](arith::ConstantFloatOp constant) {
    foundResult |= constant.value().convertToDouble() == 0.0625;
  });
  EXPECT_TRUE(foundResult);
}

TEST(OpenQASMTargetTest, PreservesSignedPowResultWithUnsignedExponent) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
int base = -2;
uint exponent = 3;
output int result;
result = pow(base, exponent);
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  const oq3::frontend::ScalarAssignmentStatement* assignment = nullptr;
  for (const auto& statement : analyzed.program->statements) {
    if (const auto* current =
            std::get_if<oq3::frontend::ScalarAssignmentStatement>(
                &statement.data);
        current != nullptr &&
        analyzed.program->scalars[current->scalar].name == "result") {
      assignment = current;
    }
  }
  ASSERT_NE(assignment, nullptr);
  ASSERT_TRUE(assignment->value);
  const auto& power = analyzed.program->expressions[*assignment->value];
  EXPECT_EQ(power.kind, oq3::frontend::ExpressionKind::Power);
  EXPECT_EQ(power.type, oq3::frontend::ScalarType::Int);
  EXPECT_EQ(analyzed.program->expressions[power.lhs].type,
            oq3::frontend::ScalarType::Int);
  EXPECT_EQ(analyzed.program->expressions[power.rhs].type,
            oq3::frontend::ScalarType::Uint);

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t integerPowers = 0;
  size_t floatConversions = 0;
  module->walk([&](scf::WhileOp) { ++integerPowers; });
  module->walk([&](Operation* operation) {
    floatConversions += isa<arith::SIToFPOp, arith::UIToFPOp>(operation);
  });
  EXPECT_EQ(integerPowers, 1);
  EXPECT_EQ(floatConversions, 0);
}

TEST(OpenQASMTargetTest, SupportsBitMeasurementReassignment) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
bit measured;
measured = measure q;
if (measured) { x q; }
measured = measure q;
if (!measured) { h q; }
)qasm";
  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t measurements = 0;
  module->walk([&](qc::MeasureOp) { ++measurements; });
  EXPECT_EQ(measurements, 2);
}

TEST(OpenQASMTargetTest, EmitsStructuredDiagnosticsWithIncludeStacks) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "OPENQASM 3.1;\ninclude \"stdgates.inc\";\n"
                                   "include \"outer.inc\";\n",
                                   "main.qasm"),
                               llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "include \"nested.inc\";\n", "outer.inc"),
                               llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "qubit q;\npow(9007199254740993) @ x q;\n", "nested.inc"),
      llvm::SMLoc());

  MLIRContext context;
  std::string message;
  Location location = UnknownLoc::get(&context);
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    message = diagnostic.str();
    location = diagnostic.getLocation();
    return success();
  });
  auto module = qc::translateQASM3ToQC(sourceMgr, &context);
  EXPECT_FALSE(module);
  EXPECT_NE(message.find("cannot be represented exactly"), std::string::npos);
  const auto mainCall = dyn_cast<CallSiteLoc>(location);
  ASSERT_TRUE(mainCall);
  const auto mainLocation = dyn_cast<FileLineColLoc>(mainCall.getCaller());
  ASSERT_TRUE(mainLocation);
  EXPECT_EQ(mainLocation.getFilename(), "main.qasm");
  EXPECT_EQ(mainLocation.getLine(), 3);
  const auto outerCall = dyn_cast<CallSiteLoc>(mainCall.getCallee());
  ASSERT_TRUE(outerCall);
  const auto outerLocation = dyn_cast<FileLineColLoc>(outerCall.getCaller());
  ASSERT_TRUE(outerLocation);
  EXPECT_EQ(outerLocation.getFilename(), "outer.inc");
  EXPECT_EQ(outerLocation.getLine(), 1);
  const auto nestedLocation = dyn_cast<FileLineColLoc>(outerCall.getCallee());
  ASSERT_TRUE(nestedLocation);
  EXPECT_EQ(nestedLocation.getFilename(), "nested.inc");
  EXPECT_EQ(nestedLocation.getLine(), 2);
}

TEST(OpenQASMTargetTest, NormalizesNegativeIndicesAndChecksDynamicAliases) {
  constexpr llvm::StringLiteral indexSource = R"qasm(
OPENQASM 3.1;
qubit[3] q;
x q[-1];
bit[3] c = measure q;
if (c[-1]) { h q[-1]; }
int i = -1;
x q[i];
c[i] = measure q[i];
if (c[i]) { x q[0]; }
output bit[3] result;
result = measure q;
)qasm";
  MLIRContext indexContext;
  auto indexed = qc::translateQASM3ToQC(indexSource, &indexContext);
  ASSERT_TRUE(indexed);
  ASSERT_TRUE(succeeded(verify(*indexed)));
  size_t indexSelections = 0;
  indexed->walk([&](arith::SelectOp) { ++indexSelections; });
  EXPECT_GE(indexSelections, 3);

  constexpr llvm::StringLiteral aliasSource = R"qasm(
OPENQASM 3.1;
qubit[2] q;
int i = 0;
cx q[i], q[i];
bit[2] result = measure q;
)qasm";
  MLIRContext aliasContext;
  auto aliased = qc::translateQASM3ToQC(aliasSource, &aliasContext);
  ASSERT_TRUE(aliased);
  ASSERT_TRUE(succeeded(verify(*aliased)));
  size_t aliasAssertions = 0;
  aliased->walk([&](cf::AssertOp) { ++aliasAssertions; });
  EXPECT_GE(aliasAssertions, 3);
}

TEST(OpenQASMTargetTest, DispatchesDynamicQubitGatesWithStructuredControlFlow) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
x q[i];
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t switches = 0;
  size_t conditionals = 0;
  module->walk([&](scf::IndexSwitchOp switchOp) {
    ++switches;
    EXPECT_EQ(switchOp.getNumCases(), 1);
    EXPECT_EQ(switchOp.getNumResults(), 0);
  });
  module->walk([&](scf::IfOp) { ++conditionals; });
  EXPECT_EQ(switches, 1);
  EXPECT_EQ(conditionals, 0);
}

TEST(OpenQASMTargetTest,
     DispatchesDynamicQubitMeasurementsWithStructuredControlFlow) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[2] q;
bit c;
int i = 0;
c = measure q[i];
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t switches = 0;
  size_t conditionals = 0;
  module->walk([&](scf::IndexSwitchOp switchOp) {
    ++switches;
    EXPECT_EQ(switchOp.getNumCases(), 1);
    ASSERT_EQ(switchOp.getNumResults(), 1);
    EXPECT_TRUE(switchOp.getResult(0).getType().isInteger(1));
  });
  module->walk([&](scf::IfOp) { ++conditionals; });
  EXPECT_EQ(switches, 1);
  EXPECT_EQ(conditionals, 0);
}

TEST(OpenQASMTargetTest, HandlesWidthOneAndNestedDynamicQubitDispatch) {
  constexpr llvm::StringLiteral widthOneSource = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[1] q;
int i = 0;
x q[i];
)qasm";
  MLIRContext widthOneContext;
  auto widthOneModule =
      qc::translateQASM3ToQC(widthOneSource, &widthOneContext);
  ASSERT_TRUE(widthOneModule);
  widthOneModule->walk([&](scf::IndexSwitchOp switchOp) {
    EXPECT_EQ(switchOp.getNumCases(), 0);
  });

  constexpr llvm::StringLiteral nestedSource = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] left;
qubit[2] right;
int i = 0;
int j = 1;
cx left[i], right[j];
)qasm";
  MLIRContext nestedContext;
  auto nestedModule = qc::translateQASM3ToQC(nestedSource, &nestedContext);
  ASSERT_TRUE(nestedModule);
  ASSERT_TRUE(succeeded(verify(*nestedModule)));
  size_t switches = 0;
  size_t controls = 0;
  nestedModule->walk([&](scf::IndexSwitchOp) { ++switches; });
  nestedModule->walk([&](qc::CtrlOp) { ++controls; });
  EXPECT_EQ(switches, 3);
  EXPECT_EQ(controls, 4);
}

TEST(OpenQASMTargetTest, SupportsOrdinaryBitInitializationAndAssignment) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
bit enabled = false;
enabled = true;
bit[2] flags;
flags[0] = enabled;
flags[1] = !enabled;
if (flags[0] && !flags[1]) { x q; }
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t xGates = 0;
  module->walk([&](qc::XOp) { ++xGates; });
  EXPECT_EQ(xGates, 1);
}

TEST(OpenQASMTargetTest, SupportsTargetlessMeasurements) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
measure q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t measurements = 0;
  module->walk([&](qc::MeasureOp) { ++measurements; });
  EXPECT_EQ(measurements, 1);
}

TEST(OpenQASMTargetTest, PromotesMixedRangeEndpointsBeforeIteration) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
const uint start = 0;
const int stop = -1;
qubit q;
for uint i in [start:-1:stop] { x q; }
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*module)));
  size_t loops = 0;
  size_t xGates = 0;
  module->walk([&](Operation* operation) {
    loops += isa<scf::ForOp>(operation);
    xGates += isa<qc::XOp>(operation);
  });
  EXPECT_EQ(loops, 0);
  EXPECT_EQ(xGates, 0);
}

TEST(OpenQASMTargetTest, ThreadsGateParametersIntoWhileConditions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate conditional(theta) q {
  while (theta > 0.0) { x q; }
}
qubit q;
conditional(0.0) q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t loops = 0;
  module->walk([&](scf::WhileOp) { ++loops; });
  EXPECT_EQ(loops, 1);
}

TEST(OpenQASMTargetTest, RejectsModifiersOnStructuredCustomGatesAtQCTarget) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate looped(theta) q {
  for int i in [0:0] { p(theta) q; }
}
qubit q;
inv @ looped(pi / 2) q;
)qasm";

  MLIRContext context;
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    return success();
  });
  auto module = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(module);
  EXPECT_NE(diagnostic.find("structured control flow"), std::string::npos);
}

TEST(OpenQASMTargetTest,
     RejectsModifiersOnTransitivelyStructuredCustomGatesAtQCTarget) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate looped q { for int i in [0:0] { x q; } }
gate wrapper q { looped q; }
qubit q;
inv @ wrapper q;
)qasm";

  MLIRContext context;
  std::string diagnostic;
  Location location = UnknownLoc::get(&context);
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    location = value.getLocation();
    return success();
  });
  auto module = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(module);
  const auto fileLocation = dyn_cast<FileLineColLoc>(location);
  ASSERT_TRUE(fileLocation);
  EXPECT_EQ(fileLocation.getFilename(), "<input>");
  EXPECT_EQ(fileLocation.getLine(), 7);
  EXPECT_EQ(fileLocation.getColumn(), 1);
  EXPECT_NE(diagnostic.find("structured control flow"), std::string::npos);
}

TEST(OpenQASMTargetTest, IgnoresUnreachableStructuredCustomGates) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate looped q { for int i in [0:0] { x q; } }
gate wrapper q { looped q; }
qubit q;
x q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OpenQASMTargetTest, LowersRuntimeDynamicIndicesWithBoundsChecks) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
bit choose = measure q[0];
if (choose) { i = 1; }
x q[i];
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t assertions = 0;
  module->walk([&](cf::AssertOp) { ++assertions; });
  EXPECT_GE(assertions, 1);
}

TEST(OpenQASMTargetTest, LowersRuntimeIndicesAcrossStatementKinds) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      R"qasm(OPENQASM 3.1; include "stdgates.inc"; qubit[2] q;
bit[2] c = measure q;
int i = 0; bit choose = measure q[0]; if (choose) { i = 1; }
if (c[i]) { x q[0]; })qasm",
      "OPENQASM 3.1; qubit[2] q; bit[2] c = measure q; int i = 0; "
      "bit choose = measure q[0]; if (choose) { i = 1; } bool value = c[i];",
      "OPENQASM 3.1; qubit[2] q; bit[2] c = measure q; int i = 0; "
      "bool value = false; "
      "bit choose = measure q[0]; if (choose) { i = 1; } value = c[i];",
      "OPENQASM 3.1; qubit[2] q; bit[2] c = measure q; int i = 0; "
      "bit choose = measure q[0]; if (choose) { i = 1; } c[i] = true;",
      "OPENQASM 3.1; qubit[2] q; bit[2] c = measure q; int i = 0; "
      "bit choose = measure q[0]; if (choose) { i = 1; } "
      "c[i] = measure q[0];",
      "OPENQASM 3.1; qubit[2] q; int i = 0; "
      "bit choose = measure q[0]; if (choose) { i = 1; } reset q[i];",
      "OPENQASM 3.1; qubit[2] q; int i = 0; "
      "bit choose = measure q[0]; if (choose) { i = 1; } barrier q[i];",
  });

  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    MLIRContext context;
    auto module = qc::translateQASM3ToQC(source, &context);
    ASSERT_TRUE(module);
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(OpenQASMTargetTest, LowersLoopVariantDynamicIndicesAtQCTarget) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
bit repeat = measure q[0];
while (repeat) { x q[i]; i = 1; repeat = measure q[0]; }
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OpenQASMTargetTest, LowersMultiIterationInductionIndicesAtQCTarget) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[4] q;
for uint i in [0:2] { int x = i + 1; h q[x]; }
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OpenQASMTargetTest, LowersCheckedIntegerArithmeticAtQCTarget) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
int turns = 0;
for int i in [0:2] { turns += 1; }
rx(turns) q;
)qasm",
      "OPENQASM 3.1; int value = 1; int derived = value + 1;",
      R"qasm(OPENQASM 3.1; include "stdgates.inc"; qubit q; int value = 1;
if (value + 1 > 0) { x q; })qasm",
      "OPENQASM 3.1; include \"stdgates.inc\"; qubit q; int value = 1; "
      "rx(value + 1) q;",
      "OPENQASM 3.1; int value = 1; bool result = value + 1 > 0;",
      "OPENQASM 3.1; int value = 1; bit result; result = value + 1 > 0;",
  });

  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    MLIRContext context;
    auto module = qc::translateQASM3ToQC(source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));
    size_t assertions = 0;
    module->walk([&](cf::AssertOp) { ++assertions; });
    EXPECT_GE(assertions, 1);
  }
}

TEST(OpenQASMTargetTest, LowersRuntimeSignedAndUnsignedIntegerOperators) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
int signedValue = 7;
int signedOperand = 2;
signedValue = -signedValue;
signedValue = signedValue + signedOperand;
signedValue = signedValue - signedOperand;
signedValue = signedValue * signedOperand;
signedValue = signedValue / signedOperand;
signedValue = signedValue % signedOperand;
signedValue = signedValue ** signedOperand;
uint unsignedValue = 7;
uint unsignedOperand = 2;
unsignedValue = -unsignedValue;
unsignedValue = unsignedValue + unsignedOperand;
unsignedValue = unsignedValue - unsignedOperand;
unsignedValue = unsignedValue * unsignedOperand;
unsignedValue = unsignedValue / unsignedOperand;
unsignedValue = unsignedValue % unsignedOperand;
unsignedValue = unsignedValue ** unsignedOperand;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t assertions = 0;
  size_t powerLoops = 0;
  module->walk([&](cf::AssertOp) { ++assertions; });
  module->walk([&](scf::WhileOp) { ++powerLoops; });
  EXPECT_GE(assertions, 7);
  EXPECT_EQ(powerLoops, 2);
}

TEST(OpenQASMTargetTest, UsesConstantBoundsForStaticInclusiveRanges) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1; qubit q; for int i in [0:1:2] { x q; }",
      "OPENQASM 3.1; qubit q; for int i in [2:-1:0] { x q; }",
      "OPENQASM 3.1; qubit q; for int i in [3:1:0] { x q; }",
      "OPENQASM 3.1; qubit q; for int i in [7:1:7] { x q; }",
      "OPENQASM 3.1; qubit q; for int i in [0:2:3] { x q; }",
      R"qasm(OPENQASM 3.1; qubit q;
for int i in [9223372036854775806:1:9223372036854775807] { x q; })qasm",
  });
  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    MLIRContext context;
    auto module = qc::translateQASM3ToQC(source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));
    size_t forLoops = 0;
    size_t whileLoops = 0;
    size_t divisions = 0;
    module->walk([&](scf::ForOp) { ++forLoops; });
    module->walk([&](scf::WhileOp) { ++whileLoops; });
    module->walk([&](arith::DivUIOp) { ++divisions; });
    EXPECT_EQ(forLoops, 1);
    EXPECT_EQ(whileLoops, 0);
    EXPECT_EQ(divisions, 0);
  }
}

TEST(OpenQASMTargetTest, UsesComparisonDrivenDynamicInclusiveRanges) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
int start = 0;
int step = 1;
int stop = 2;
bit choose = measure q;
if (choose) { start = 1; }
for int i in [start:step:stop] { x q; }
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t whileLoops = 0;
  size_t divisions = 0;
  size_t assertions = 0;
  module->walk([&](scf::WhileOp) { ++whileLoops; });
  module->walk([&](arith::DivUIOp) { ++divisions; });
  module->walk([&](cf::AssertOp) { ++assertions; });
  EXPECT_EQ(whileLoops, 1);
  EXPECT_EQ(divisions, 0);
  EXPECT_GE(assertions, 1);
}

TEST(OpenQASMTargetTest, PreservesStaticallySelectedIndexState) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
if (false) { i = 1; pow(2) @ x q[1]; }
x q[i];
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t conditionals = 0;
  size_t indexSwitches = 0;
  size_t xGates = 0;
  size_t powers = 0;
  module->walk([&](scf::IfOp) { ++conditionals; });
  module->walk([&](scf::IndexSwitchOp) { ++indexSwitches; });
  module->walk([&](qc::XOp) { ++xGates; });
  module->walk([&](qc::PowOp) { ++powers; });
  EXPECT_EQ(conditionals, 0);
  EXPECT_EQ(indexSwitches, 1);
  EXPECT_EQ(xGates, 2);
  EXPECT_EQ(powers, 0);
}

TEST(OpenQASMTargetTest, PreservesEqualConstantIndexJoins) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
bit choose = measure q[0];
if (choose) { i = 1; } else { i = 1; }
x q[i];
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OpenQASMTargetTest, LowersShortCircuitBooleanEvaluation) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[2] q;
bit[2] measured = measure q;
float negative = -1.0;
float notANumber = sqrt(negative);
if ((measured[0] && measured[1]) || notANumber != notANumber) { x q[0]; }
output bit[2] result;
result = measure q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<int64_t> firstMeasuredIndices;
  bool sawUnorderedInequality = false;
  size_t shortCircuitOperations = 0;
  size_t eagerLogicalOperations = 0;
  module->walk([&](Operation* operation) {
    if (auto comparison = dyn_cast<arith::CmpFOp>(operation)) {
      sawUnorderedInequality |=
          comparison.getPredicate() == arith::CmpFPredicate::UNE;
    }
    if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
      shortCircuitOperations += conditional.getNumResults() == 1;
    }
    eagerLogicalOperations += isa<arith::AndIOp, arith::OrIOp>(operation);
    auto measurement = dyn_cast<qc::MeasureOp>(operation);
    if (!measurement || firstMeasuredIndices.size() == 2) {
      return;
    }
    auto load = measurement.getQubit().getDefiningOp<memref::LoadOp>();
    if (!load || load.getIndices().empty()) {
      return;
    }
    APInt index;
    if (matchPattern(load.getIndices().front(), m_ConstantInt(&index))) {
      firstMeasuredIndices.push_back(index.getSExtValue());
    }
  });
  EXPECT_EQ(firstMeasuredIndices, (SmallVector<int64_t>{0, 1}));
  EXPECT_EQ(shortCircuitOperations, 2);
  EXPECT_EQ(eagerLogicalOperations, 0);
  EXPECT_TRUE(sawUnorderedInequality);
}

TEST(OpenQASMTargetTest, EmitsStructuredLoopsWithCarriedMutableState) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
float theta = 0.0;
for int i in [0:2] {
  theta += 0.125;
  h q;
}
bit repeat = measure q;
while (repeat) {
  theta += 0.25;
  rx(theta) q;
  repeat = measure q;
}
rx(theta) q;
bit result = measure q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  scf::ForOp forLoop;
  scf::WhileOp whileLoop;
  module->walk([&](Operation* operation) {
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      forLoop = loop;
    }
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      whileLoop = loop;
    }
  });
  ASSERT_TRUE(forLoop);
  ASSERT_TRUE(whileLoop);
  EXPECT_EQ(forLoop.getInitArgs().size(), 1);
  EXPECT_EQ(forLoop.getNumResults(), 1);
  EXPECT_EQ(forLoop.getBody()->getTerminator()->getNumOperands(), 1);
  EXPECT_EQ(whileLoop.getInits().size(), 2);
  EXPECT_EQ(whileLoop.getNumResults(), 2);
  EXPECT_EQ(whileLoop.getBeforeBody()->getTerminator()->getNumOperands(), 3);
  EXPECT_EQ(whileLoop.getAfterBody()->getTerminator()->getNumOperands(), 2);
  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*module)));
  forLoop = {};
  module->walk([&](scf::ForOp loop) { forLoop = loop; });
  ASSERT_TRUE(forLoop);
  APInt lower;
  APInt upper;
  APInt step;
  ASSERT_TRUE(matchPattern(forLoop.getLowerBound(), m_ConstantInt(&lower)));
  ASSERT_TRUE(matchPattern(forLoop.getUpperBound(), m_ConstantInt(&upper)));
  ASSERT_TRUE(matchPattern(forLoop.getStep(), m_ConstantInt(&step)));
  EXPECT_EQ(lower.getSExtValue(), 0);
  EXPECT_EQ(upper.getSExtValue(), 3);
  EXPECT_EQ(step.getSExtValue(), 1);

  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OpenQASMTargetTest, PreservesBranchAndWhileCarriedClassicalBits) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
bool choose = true;
bit branch;
if (choose) {
  branch = measure q[0];
} else {
  branch = measure q[1];
}
while (branch) {
  h q[0];
  branch = measure q[0];
}
if (branch) { x q[1]; }
output bit[2] result;
result = measure q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t resultBearingConditionals = 0;
  size_t resultBearingWhiles = 0;
  module->walk([&](Operation* operation) {
    if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
      resultBearingConditionals += conditional.getNumResults() != 0;
    }
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      resultBearingWhiles += loop.getNumResults() != 0;
    }
  });
  EXPECT_EQ(resultBearingConditionals, 1);
  EXPECT_EQ(resultBearingWhiles, 1);
  module->walk([&](scf::IfOp conditional) {
    if (conditional.getNumResults() == 0) {
      return;
    }
    EXPECT_EQ(conditional.getNumResults(), 1);
    EXPECT_TRUE(conditional.getResult(0).getType().isInteger(1));
    EXPECT_EQ(
        conditional.getThenRegion().front().getTerminator()->getNumOperands(),
        1);
    EXPECT_EQ(
        conditional.getElseRegion().front().getTerminator()->getNumOperands(),
        1);
  });
  module->walk([&](scf::WhileOp loop) {
    EXPECT_EQ(loop.getInits().size(), 1);
    EXPECT_EQ(loop.getNumResults(), 1);
    EXPECT_EQ(loop.getBeforeBody()->getTerminator()->getNumOperands(), 2);
    EXPECT_EQ(loop.getAfterBody()->getTerminator()->getNumOperands(), 1);
  });
}

TEST(OpenQASMTargetTest, CarriesEveryPotentiallyMutatedDynamicBit) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
uint index = 0;
bit[2] state;
state[0] = false;
state[1] = false;
bool choose = true;
if (choose) {
  state[index] = measure q;
} else {
  state[index] = true;
}
if (state[0]) { x q; }
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  scf::IfOp stateUpdate;
  module->walk([&](scf::IfOp conditional) {
    if (conditional.getNumResults() == 2) {
      stateUpdate = conditional;
    }
  });
  ASSERT_TRUE(stateUpdate);
  EXPECT_TRUE(llvm::all_of(stateUpdate.getResultTypes(),
                           [](Type type) { return type.isInteger(1); }));
  EXPECT_EQ(
      stateUpdate.getThenRegion().front().getTerminator()->getNumOperands(), 2);
  EXPECT_EQ(
      stateUpdate.getElseRegion().front().getTerminator()->getNumOperands(), 2);
}

TEST(OpenQASMTargetTest, HandlesTheMaximumUnsignedSingletonRange) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
const uint maximum = 18446744073709551615;
qubit q;
for uint i in [maximum:maximum] { if (i == maximum) { x q; } }
)qasm";
  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  scf::ForOp loop;
  module->walk([&](scf::ForOp current) { loop = current; });
  ASSERT_TRUE(loop);
  APInt lower;
  APInt step;
  ASSERT_TRUE(matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)));
  ASSERT_TRUE(matchPattern(loop.getStep(), m_ConstantInt(&step)));
  EXPECT_EQ(lower.getSExtValue(), 0);
  EXPECT_EQ(step.getSExtValue(), 1);

  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*module)));
  size_t remainingLoops = 0;
  size_t xApplications = 0;
  module->walk([&](Operation* operation) {
    remainingLoops += isa<scf::ForOp>(operation);
    xApplications += isa<qc::XOp>(operation);
  });
  EXPECT_EQ(remainingLoops, 0);
  EXPECT_EQ(xApplications, 1);
}

TEST(OpenQASMTargetTest, ExpandsAnOperandlessBarrierToAllDeclaredQubits) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[3] q;
barrier;
bit[3] result = measure q;
)qasm";
  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t barriers = 0;
  module->walk([&](qc::BarrierOp barrier) {
    ++barriers;
    EXPECT_EQ(barrier.getNumQubits(), 3);
  });
  EXPECT_EQ(barriers, 1);
}

TEST(OpenQASMTargetTest, CanonicalizesVariadicCompatibilityGates) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[10] q;
mcx q[0], q[1], q[2], q[3];
mcphase(0.5) q[0], q[1], q[2];
mcx_vchain q[0], q[1], q[2], q[3], q[4], q[8], q[9];
mcx_recursive q[0], q[1], q[2], q[3], q[4], q[9];
)qasm";
  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t controls = 0;
  size_t xGates = 0;
  size_t phaseGates = 0;
  module->walk([&](Operation* operation) {
    controls += isa<qc::CtrlOp>(operation);
    xGates += isa<qc::XOp>(operation);
    phaseGates += isa<qc::POp>(operation);
  });
  EXPECT_EQ(controls, 4);
  EXPECT_EQ(xGates, 3);
  EXPECT_EQ(phaseGates, 1);
}

TEST(OpenQASMTargetTest, BroadcastsRegistersAlongsideScalarQubits) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[3] controls;
qubit target;
cx controls, target;
bit[3] left = measure controls;
bit right = measure target;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  size_t controls = 0;
  module->walk([&](qc::CtrlOp) { ++controls; });
  EXPECT_EQ(controls, 3);
}

TEST(OpenQASMTargetTest, PreservesImportedWhileBehavior) {
  struct OperationCounts {
    size_t h;
    size_t x;
    size_t measurements;
    size_t controls;
  };
  struct ConditionalCounts {
    size_t semantic;
    size_t dispatch;
    size_t whileMeasurements;
  };
  struct Fixture {
    llvm::StringRef name;
    llvm::StringRef source;
    SmallVector<int64_t> tripCounts;
    size_t whileLoops;
    OperationCounts operations;
    ConditionalCounts conditionals;
  };
  const auto fixtures = std::to_array<Fixture>({
      {.name = "simple-while",
       .source = qasm::simpleWhileReset,
       .tripCounts = {},
       .whileLoops = 1,
       .operations = {.h = 2, .x = 0, .measurements = 3, .controls = 0},
       .conditionals = {.semantic = 0, .dispatch = 0, .whileMeasurements = 0}},
      {.name = "condition-while-and",
       .source = qasm::conditionWhileAnd,
       .tripCounts = {},
       .whileLoops = 1,
       .operations = {.h = 3, .x = 0, .measurements = 6, .controls = 0},
       .conditionals = {.semantic = 1, .dispatch = 0, .whileMeasurements = 0}},
  });

  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(fixture.name.str());
    MLIRContext context;
    auto module = qc::translateQASM3ToQC(fixture.source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));

    PassManager canonicalizer(&context);
    canonicalizer.addPass(createCanonicalizerPass());
    ASSERT_TRUE(succeeded(canonicalizer.run(*module)));

    SmallVector<scf::ForOp> forLoops;
    size_t whileLoops = 0;
    bool hasQubitSelect = false;
    module->walk([&](Operation* operation) {
      if (auto loop = dyn_cast<scf::ForOp>(operation)) {
        forLoops.push_back(loop);
      }
      whileLoops += isa<scf::WhileOp>(operation);
      if (auto select = dyn_cast<arith::SelectOp>(operation)) {
        hasQubitSelect |= isa<qc::QubitType>(select.getType());
      }
    });
    ASSERT_EQ(forLoops.size(), fixture.tripCounts.size());
    EXPECT_EQ(whileLoops, fixture.whileLoops);
    EXPECT_FALSE(hasQubitSelect);
    for (const auto [loop, expectedCount] :
         llvm::zip_equal(forLoops, fixture.tripCounts)) {
      APInt lower;
      APInt upper;
      APInt step;
      ASSERT_TRUE(matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)));
      ASSERT_TRUE(matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)));
      ASSERT_TRUE(matchPattern(loop.getStep(), m_ConstantInt(&step)));
      EXPECT_EQ(lower.getSExtValue(), 0);
      EXPECT_EQ(upper.getSExtValue(), expectedCount);
      EXPECT_EQ(step.getSExtValue(), 1);
    }

    ASSERT_TRUE(succeeded(verify(*module)));
    size_t hGates = 0;
    size_t xGates = 0;
    size_t measurements = 0;
    size_t controls = 0;
    size_t semanticConditionals = 0;
    size_t dispatchConditionals = 0;
    SmallVector<scf::ForOp> loweredForLoops;
    SmallVector<scf::WhileOp> loweredWhileLoops;
    module->walk([&](Operation* operation) {
      hGates += isa<qc::HOp>(operation);
      xGates += isa<qc::XOp>(operation);
      measurements += isa<qc::MeasureOp>(operation);
      controls += isa<qc::CtrlOp>(operation);
      if (auto control = dyn_cast<qc::CtrlOp>(operation)) {
        size_t controlledXGates = 0;
        control->walk([&](qc::XOp) { ++controlledXGates; });
        EXPECT_EQ(controlledXGates, 1)
            << "each imported controlled-X must retain its controlled body";
      }
      if (auto loop = dyn_cast<scf::ForOp>(operation)) {
        loweredForLoops.push_back(loop);
      }
      if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
        loweredWhileLoops.push_back(loop);
      }
      auto conditional = dyn_cast<scf::IfOp>(operation);
      if (!conditional) {
        return;
      }
      auto comparison =
          conditional.getCondition().getDefiningOp<arith::CmpIOp>();
      if (!comparison ||
          comparison.getPredicate() != arith::CmpIPredicate::eq) {
        ++semanticConditionals;
        return;
      }

      APInt candidate;
      const bool lhsCandidate =
          matchPattern(comparison.getLhs(), m_ConstantInt(&candidate));
      const bool rhsCandidate =
          matchPattern(comparison.getRhs(), m_ConstantInt(&candidate));
      if (lhsCandidate == rhsCandidate) {
        ++semanticConditionals;
        return;
      }
      ++dispatchConditionals;
      size_t dispatchedQuantumOperations = 0;
      conditional->walk([&](Operation* nested) {
        dispatchedQuantumOperations +=
            isa<qc::HOp, qc::XOp, qc::MeasureOp, qc::CtrlOp>(nested);
      });
      EXPECT_GT(dispatchedQuantumOperations, 0)
          << "each dynamic-index dispatch must retain quantum behavior";
    });
    EXPECT_EQ(hGates, fixture.operations.h);
    EXPECT_EQ(xGates, fixture.operations.x);
    EXPECT_EQ(measurements, fixture.operations.measurements);
    EXPECT_EQ(controls, fixture.operations.controls);
    EXPECT_EQ(semanticConditionals, fixture.conditionals.semantic);
    EXPECT_EQ(dispatchConditionals, fixture.conditionals.dispatch);

    ASSERT_EQ(loweredForLoops.size(), fixture.tripCounts.size());
    for (auto loop : loweredForLoops) {
      size_t bodyGates = 0;
      loop.getRegion().walk([&](Operation* nested) {
        bodyGates += isa<qc::HOp, qc::XOp>(nested);
      });
      EXPECT_GT(bodyGates, 0)
          << "each imported for-loop body must retain its gate behavior";
    }

    ASSERT_EQ(loweredWhileLoops.size(), fixture.whileLoops);
    for (auto loop : loweredWhileLoops) {
      size_t conditionMeasurements = 0;
      size_t bodyGates = 0;
      loop.getBefore().walk([&](qc::MeasureOp) { ++conditionMeasurements; });
      loop.getAfter().walk([&](Operation* nested) {
        bodyGates += isa<qc::HOp, qc::XOp>(nested);
      });
      EXPECT_EQ(conditionMeasurements, fixture.conditionals.whileMeasurements);
      EXPECT_GT(bodyGates, 0)
          << "each imported while-loop body must retain its gate behavior";
    }
  }
}

} // namespace

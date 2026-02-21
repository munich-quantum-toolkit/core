/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "mlir/Conversion/JeffToQC/JeffToQC.h"
#include "mlir/Conversion/QCToJeff/QCToJeff.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "qc_programs.h"

#include <gtest/gtest.h>
#include <jeff/IR/JeffDialect.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <ostream>
#include <string>

using namespace mlir;

struct JeffRoundTripTestCase {
  std::string name;
  mqt::test::NamedBuilder<qc::QCProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<qc::QCProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const JeffRoundTripTestCase& info);
};

class JeffRoundTripTest : public testing::TestWithParam<JeffRoundTripTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, func::FuncDialect, jeff::JeffDialect,
                    mlir::qc::QCDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

std::ostream& operator<<(std::ostream& os, const JeffRoundTripTestCase& info) {
  return os << "JeffRoundTrip{" << info.name
            << ", original=" << mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << mqt::test::displayName(info.referenceBuilder.name) << "}";
}

static LogicalResult runJeffRoundTrip(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCToJeff());
  pm.addPass(createJeffToQC());
  return pm.run(module);
}

TEST_P(JeffRoundTripTest, ProgramEquivalence) {
  const auto& [nameStr, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + nameStr + ")";
  mqt::test::DeferredPrinter printer;

  auto program = qc::QCProgramBuilder::build(context.get(), programBuilder.fn);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printer.record(program.get(), "Canonicalized QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runJeffRoundTrip(program.get())));
  printer.record(program.get(), "Converted QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printer.record(program.get(), "Canonicalized Converted QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference =
      qc::QCProgramBuilder::build(context.get(), referenceBuilder.fn);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QC IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  runCanonicalizationPasses(reference.get());
  printer.record(reference.get(), "Canonicalized Reference QC IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/// \name JeffRoundTrip/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCGPhaseOpTest, JeffRoundTripTest,
                         testing::Values(JeffRoundTripTestCase{
                             "GlobalPhase", MQT_NAMED_BUILDER(qc::globalPhase),
                             MQT_NAMED_BUILDER(qc::globalPhase)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCHOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"H", MQT_NAMED_BUILDER(qc::h),
                              MQT_NAMED_BUILDER(qc::h)},
        JeffRoundTripTestCase{"SingleControlledH",
                              MQT_NAMED_BUILDER(qc::singleControlledH),
                              MQT_NAMED_BUILDER(qc::singleControlledH)},
        JeffRoundTripTestCase{"MultipleControlledH",
                              MQT_NAMED_BUILDER(qc::multipleControlledH),
                              MQT_NAMED_BUILDER(qc::multipleControlledH)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCIdOpTest, JeffRoundTripTest,
                         testing::Values(JeffRoundTripTestCase{
                             "Identity", MQT_NAMED_BUILDER(qc::identity),
                             MQT_NAMED_BUILDER(qc::identity)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCPOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"P", MQT_NAMED_BUILDER(qc::p),
                              MQT_NAMED_BUILDER(qc::p)},
        JeffRoundTripTestCase{"SingleControlledP",
                              MQT_NAMED_BUILDER(qc::singleControlledP),
                              MQT_NAMED_BUILDER(qc::singleControlledP)},
        JeffRoundTripTestCase{"MultipleControlledP",
                              MQT_NAMED_BUILDER(qc::multipleControlledP),
                              MQT_NAMED_BUILDER(qc::multipleControlledP)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RX", MQT_NAMED_BUILDER(qc::rx),
                              MQT_NAMED_BUILDER(qc::rx)},
        JeffRoundTripTestCase{"SingleControlledRX",
                              MQT_NAMED_BUILDER(qc::singleControlledRx),
                              MQT_NAMED_BUILDER(qc::singleControlledRx)},
        JeffRoundTripTestCase{"MultipleControlledRX",
                              MQT_NAMED_BUILDER(qc::multipleControlledRx),
                              MQT_NAMED_BUILDER(qc::multipleControlledRx)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRYOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RY", MQT_NAMED_BUILDER(qc::ry),
                              MQT_NAMED_BUILDER(qc::ry)},
        JeffRoundTripTestCase{"SingleControlledRY",
                              MQT_NAMED_BUILDER(qc::singleControlledRy),
                              MQT_NAMED_BUILDER(qc::singleControlledRy)},
        JeffRoundTripTestCase{"MultipleControlledRY",
                              MQT_NAMED_BUILDER(qc::multipleControlledRy),
                              MQT_NAMED_BUILDER(qc::multipleControlledRy)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRZOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RZ", MQT_NAMED_BUILDER(qc::rz),
                              MQT_NAMED_BUILDER(qc::rz)},
        JeffRoundTripTestCase{"SingleControlledRZ",
                              MQT_NAMED_BUILDER(qc::singleControlledRz),
                              MQT_NAMED_BUILDER(qc::singleControlledRz)},
        JeffRoundTripTestCase{"MultipleControlledRZ",
                              MQT_NAMED_BUILDER(qc::multipleControlledRz),
                              MQT_NAMED_BUILDER(qc::multipleControlledRz)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"S", MQT_NAMED_BUILDER(qc::s),
                              MQT_NAMED_BUILDER(qc::s)},
        JeffRoundTripTestCase{"SingleControlledS",
                              MQT_NAMED_BUILDER(qc::singleControlledS),
                              MQT_NAMED_BUILDER(qc::singleControlledS)},
        JeffRoundTripTestCase{"MultipleControlledS",
                              MQT_NAMED_BUILDER(qc::multipleControlledS),
                              MQT_NAMED_BUILDER(qc::multipleControlledS)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSdgOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Sdg", MQT_NAMED_BUILDER(qc::sdg),
                              MQT_NAMED_BUILDER(qc::sdg)},
        JeffRoundTripTestCase{"SingleControlledSdg",
                              MQT_NAMED_BUILDER(qc::singleControlledSdg),
                              MQT_NAMED_BUILDER(qc::singleControlledSdg)},
        JeffRoundTripTestCase{"MultipleControlledSdg",
                              MQT_NAMED_BUILDER(qc::multipleControlledSdg),
                              MQT_NAMED_BUILDER(qc::multipleControlledSdg)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSWAPOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"SWAP", MQT_NAMED_BUILDER(qc::swap),
                              MQT_NAMED_BUILDER(qc::swap)},
        JeffRoundTripTestCase{"SingleControlledSWAP",
                              MQT_NAMED_BUILDER(qc::singleControlledSwap),
                              MQT_NAMED_BUILDER(qc::singleControlledSwap)},
        JeffRoundTripTestCase{"MultipleControlledSWAP",
                              MQT_NAMED_BUILDER(qc::multipleControlledSwap),
                              MQT_NAMED_BUILDER(qc::multipleControlledSwap)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCTOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"T", MQT_NAMED_BUILDER(qc::t_),
                              MQT_NAMED_BUILDER(qc::t_)},
        JeffRoundTripTestCase{"SingleControlledT",
                              MQT_NAMED_BUILDER(qc::singleControlledT),
                              MQT_NAMED_BUILDER(qc::singleControlledT)},
        JeffRoundTripTestCase{"MultipleControlledT",
                              MQT_NAMED_BUILDER(qc::multipleControlledT),
                              MQT_NAMED_BUILDER(qc::multipleControlledT)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCTdgOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Tdg", MQT_NAMED_BUILDER(qc::tdg),
                              MQT_NAMED_BUILDER(qc::tdg)},
        JeffRoundTripTestCase{"SingleControlledTdg",
                              MQT_NAMED_BUILDER(qc::singleControlledTdg),
                              MQT_NAMED_BUILDER(qc::singleControlledTdg)},
        JeffRoundTripTestCase{"MultipleControlledTdg",
                              MQT_NAMED_BUILDER(qc::multipleControlledTdg),
                              MQT_NAMED_BUILDER(qc::multipleControlledTdg)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCUOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"U", MQT_NAMED_BUILDER(qc::u),
                              MQT_NAMED_BUILDER(qc::u)},
        JeffRoundTripTestCase{"SingleControlledU",
                              MQT_NAMED_BUILDER(qc::singleControlledU),
                              MQT_NAMED_BUILDER(qc::singleControlledU)},
        JeffRoundTripTestCase{"MultipleControlledU",
                              MQT_NAMED_BUILDER(qc::multipleControlledU),
                              MQT_NAMED_BUILDER(qc::multipleControlledU)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"X", MQT_NAMED_BUILDER(qc::x),
                              MQT_NAMED_BUILDER(qc::x)},
        JeffRoundTripTestCase{"SingleControlledX",
                              MQT_NAMED_BUILDER(qc::singleControlledX),
                              MQT_NAMED_BUILDER(qc::singleControlledX)},
        JeffRoundTripTestCase{"MultipleControlledX",
                              MQT_NAMED_BUILDER(qc::multipleControlledX),
                              MQT_NAMED_BUILDER(qc::multipleControlledX)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCYOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Y", MQT_NAMED_BUILDER(qc::y),
                              MQT_NAMED_BUILDER(qc::y)},
        JeffRoundTripTestCase{"SingleControlledY",
                              MQT_NAMED_BUILDER(qc::singleControlledY),
                              MQT_NAMED_BUILDER(qc::singleControlledY)},
        JeffRoundTripTestCase{"MultipleControlledY",
                              MQT_NAMED_BUILDER(qc::multipleControlledY),
                              MQT_NAMED_BUILDER(qc::multipleControlledY)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCZOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Z", MQT_NAMED_BUILDER(qc::z),
                              MQT_NAMED_BUILDER(qc::z)},
        JeffRoundTripTestCase{"SingleControlledZ",
                              MQT_NAMED_BUILDER(qc::singleControlledZ),
                              MQT_NAMED_BUILDER(qc::singleControlledZ)},
        JeffRoundTripTestCase{"MultipleControlledZ",
                              MQT_NAMED_BUILDER(qc::multipleControlledZ),
                              MQT_NAMED_BUILDER(qc::multipleControlledZ)}));
/// @}

/// \name JeffRoundTrip/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCMeasureOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{
            "SingleMeasurementToSingleBit",
            MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit),
            MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit)},
        JeffRoundTripTestCase{
            "RepeatedMeasurementToSameBit",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit),
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit)},
        JeffRoundTripTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits)},
        JeffRoundTripTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements)}));
/// @}

/// \name JeffRoundTrip/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCResetOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"ResetQubitAfterSingleOp",
                              MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp),
                              MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp)},
        JeffRoundTripTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp)}));
/// @}

// class QCToJeffConversionTest : public ::testing::Test {
// protected:
//   std::unique_ptr<mlir::MLIRContext> context;
//   void SetUp() override {
//     // Register all dialects needed for the full compilation pipeline
//     DialectRegistry registry;
//     registry.insert<arith::ArithDialect, func::FuncDialect,
//     jeff::JeffDialect,
//                     mlir::qc::QCDialect>();

//     context = std::make_unique<MLIRContext>();
//     context->appendDialectRegistry(registry);
//     context->loadAllAvailableDialects();
//   }

//   [[nodiscard]] OwningOpRef<ModuleOp> buildQCIR(
//       const std::function<void(mlir::qc::QCProgramBuilder&)>& buildFunc)
//       const {
//     mlir::qc::QCProgramBuilder builder(context.get());
//     builder.initialize();
//     buildFunc(builder);
//     auto module = builder.finalize();
//     return module;
//   }

//   static std::string getOutputString(mlir::OwningOpRef<mlir::ModuleOp>& mod)
//   {
//     std::string outputString;
//     llvm::raw_string_ostream outputStream(outputString);
//     mod->print(outputStream);
//     outputStream.flush();
//     return outputString;
//   }
// };

// TEST_F(QCToJeffConversionTest, Measure) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.measure(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.qubit_measure_nd"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, Reset) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.reset(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.qubit_reset"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, GPhase) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     b.gphase(0.5);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.gphase"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, CGPhase) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.cgphase(0.5, q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.gphase"), std::string::npos);
//   ASSERT_NE(outputString.find("num_ctrls = 1"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, Id) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.id(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.i"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, X) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.x(q);
//     b.x(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.x"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, MCX) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(3, "q");
//     auto q0 = reg[0];
//     auto q1 = reg[1];
//     auto q2 = reg[2];
//     b.mcx({q1, q2}, q0);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.x"), std::string::npos);
//   ASSERT_NE(outputString.find("num_ctrls = 2"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, Y) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.y(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.y"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, Z) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.z(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.z"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, H) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.h(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.h"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, S) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.s(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.s"), std::string::npos);
//   ASSERT_NE(outputString.find("is_adjoint = false"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, Sdg) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.sdg(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.s"), std::string::npos);
//   ASSERT_NE(outputString.find("is_adjoint = true"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, T) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.t(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.t"), std::string::npos);
//   ASSERT_NE(outputString.find("is_adjoint = false"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, Tdg) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.tdg(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.t"), std::string::npos);
//   ASSERT_NE(outputString.find("is_adjoint = true"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, RX) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.rx(0.5, q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.rx"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, CRX) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(2, "q");
//     auto q0 = reg[0];
//     auto q1 = reg[1];
//     b.crx(0.5, q0, q1);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.rx"), std::string::npos);
//   ASSERT_NE(outputString.find("num_ctrls = 1"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, RY) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.ry(0.5, q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.ry"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, RZ) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.rz(0.5, q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.rz"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, P) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.p(0.5, q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.r1"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, U) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.u(0.1, 0.2, 0.3, q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.u"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, CU) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(2, "q");
//     auto q0 = reg[0];
//     auto q1 = reg[1];
//     b.cu(0.1, 0.2, 0.3, q1, q0);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.u"), std::string::npos);
//   ASSERT_NE(outputString.find("num_ctrls = 1"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, SWAP) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(2, "q");
//     auto q0 = reg[0];
//     auto q1 = reg[1];
//     b.swap(q0, q1);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.swap"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, CSWAP) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(3, "q");
//     auto q0 = reg[0];
//     auto q1 = reg[1];
//     auto q2 = reg[2];
//     b.cswap(q0, q1, q2);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.swap"), std::string::npos);
//   ASSERT_NE(outputString.find("num_ctrls = 1"), std::string::npos);
// }

// TEST_F(QCToJeffConversionTest, Bell) {
//   auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(2, "q");
//     auto q0 = reg[0];
//     auto q1 = reg[1];
//     b.h(q0);
//     b.cx(q0, q1);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QC-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.h"), std::string::npos);
//   ASSERT_NE(outputString.find("jeff.x"), std::string::npos);
//   ASSERT_NE(outputString.find("num_ctrls = 1"), std::string::npos);
// }

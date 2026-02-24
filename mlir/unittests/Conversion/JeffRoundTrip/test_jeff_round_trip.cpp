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
#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"
#include "mlir/Conversion/QCOToJeff/QCOToJeff.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "qco_programs.h"

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
  mqt::test::NamedBuilder<qco::QCOProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<qco::QCOProgramBuilder> referenceBuilder;

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
                    mlir::qco::QCODialect>();
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
  pm.addPass(createQCOToJeff());
  pm.addPass(createJeffToQCO());
  return pm.run(module);
}

TEST_P(JeffRoundTripTest, ProgramEquivalence) {
  const auto& [nameStr, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + nameStr + ")";
  mqt::test::DeferredPrinter printer;

  auto program =
      qco::QCOProgramBuilder::build(context.get(), programBuilder.fn);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printer.record(program.get(), "Canonicalized QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runJeffRoundTrip(program.get())));
  printer.record(program.get(), "Converted QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printer.record(program.get(), "Canonicalized Converted QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference =
      qco::QCOProgramBuilder::build(context.get(), referenceBuilder.fn);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QCO IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  runCanonicalizationPasses(reference.get());
  printer.record(reference.get(), "Canonicalized Reference QCO IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/// \name JeffRoundTrip/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCOGPhaseOpTest, JeffRoundTripTest,
                         testing::Values(JeffRoundTripTestCase{
                             "GlobalPhase", MQT_NAMED_BUILDER(qco::globalPhase),
                             MQT_NAMED_BUILDER(qco::globalPhase)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOHOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"H", MQT_NAMED_BUILDER(qco::h),
                              MQT_NAMED_BUILDER(qco::h)},
        JeffRoundTripTestCase{"SingleControlledH",
                              MQT_NAMED_BUILDER(qco::singleControlledH),
                              MQT_NAMED_BUILDER(qco::singleControlledH)},
        JeffRoundTripTestCase{"MultipleControlledH",
                              MQT_NAMED_BUILDER(qco::multipleControlledH),
                              MQT_NAMED_BUILDER(qco::multipleControlledH)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCOIdOpTest, JeffRoundTripTest,
                         testing::Values(JeffRoundTripTestCase{
                             "Identity", MQT_NAMED_BUILDER(qco::identity),
                             MQT_NAMED_BUILDER(qco::identity)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOPOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"P", MQT_NAMED_BUILDER(qco::p),
                              MQT_NAMED_BUILDER(qco::p)},
        JeffRoundTripTestCase{"SingleControlledP",
                              MQT_NAMED_BUILDER(qco::singleControlledP),
                              MQT_NAMED_BUILDER(qco::singleControlledP)},
        JeffRoundTripTestCase{"MultipleControlledP",
                              MQT_NAMED_BUILDER(qco::multipleControlledP),
                              MQT_NAMED_BUILDER(qco::multipleControlledP)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RX", MQT_NAMED_BUILDER(qco::rx),
                              MQT_NAMED_BUILDER(qco::rx)},
        JeffRoundTripTestCase{"SingleControlledRX",
                              MQT_NAMED_BUILDER(qco::singleControlledRx),
                              MQT_NAMED_BUILDER(qco::singleControlledRx)},
        JeffRoundTripTestCase{"MultipleControlledRX",
                              MQT_NAMED_BUILDER(qco::multipleControlledRx),
                              MQT_NAMED_BUILDER(qco::multipleControlledRx)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORYOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RY", MQT_NAMED_BUILDER(qco::ry),
                              MQT_NAMED_BUILDER(qco::ry)},
        JeffRoundTripTestCase{"SingleControlledRY",
                              MQT_NAMED_BUILDER(qco::singleControlledRy),
                              MQT_NAMED_BUILDER(qco::singleControlledRy)},
        JeffRoundTripTestCase{"MultipleControlledRY",
                              MQT_NAMED_BUILDER(qco::multipleControlledRy),
                              MQT_NAMED_BUILDER(qco::multipleControlledRy)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RZ", MQT_NAMED_BUILDER(qco::rz),
                              MQT_NAMED_BUILDER(qco::rz)},
        JeffRoundTripTestCase{"SingleControlledRZ",
                              MQT_NAMED_BUILDER(qco::singleControlledRz),
                              MQT_NAMED_BUILDER(qco::singleControlledRz)},
        JeffRoundTripTestCase{"MultipleControlledRZ",
                              MQT_NAMED_BUILDER(qco::multipleControlledRz),
                              MQT_NAMED_BUILDER(qco::multipleControlledRz)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"S", MQT_NAMED_BUILDER(qco::s),
                              MQT_NAMED_BUILDER(qco::s)},
        JeffRoundTripTestCase{"SingleControlledS",
                              MQT_NAMED_BUILDER(qco::singleControlledS),
                              MQT_NAMED_BUILDER(qco::singleControlledS)},
        JeffRoundTripTestCase{"MultipleControlledS",
                              MQT_NAMED_BUILDER(qco::multipleControlledS),
                              MQT_NAMED_BUILDER(qco::multipleControlledS)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSdgOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Sdg", MQT_NAMED_BUILDER(qco::sdg),
                              MQT_NAMED_BUILDER(qco::sdg)},
        JeffRoundTripTestCase{"SingleControlledSdg",
                              MQT_NAMED_BUILDER(qco::singleControlledSdg),
                              MQT_NAMED_BUILDER(qco::singleControlledSdg)},
        JeffRoundTripTestCase{"MultipleControlledSdg",
                              MQT_NAMED_BUILDER(qco::multipleControlledSdg),
                              MQT_NAMED_BUILDER(qco::multipleControlledSdg)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSWAPOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"SWAP", MQT_NAMED_BUILDER(qco::swap),
                              MQT_NAMED_BUILDER(qco::swap)},
        JeffRoundTripTestCase{"SingleControlledSWAP",
                              MQT_NAMED_BUILDER(qco::singleControlledSwap),
                              MQT_NAMED_BUILDER(qco::singleControlledSwap)},
        JeffRoundTripTestCase{"MultipleControlledSWAP",
                              MQT_NAMED_BUILDER(qco::multipleControlledSwap),
                              MQT_NAMED_BUILDER(qco::multipleControlledSwap)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"T", MQT_NAMED_BUILDER(qco::t_),
                              MQT_NAMED_BUILDER(qco::t_)},
        JeffRoundTripTestCase{"SingleControlledT",
                              MQT_NAMED_BUILDER(qco::singleControlledT),
                              MQT_NAMED_BUILDER(qco::singleControlledT)},
        JeffRoundTripTestCase{"MultipleControlledT",
                              MQT_NAMED_BUILDER(qco::multipleControlledT),
                              MQT_NAMED_BUILDER(qco::multipleControlledT)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTdgOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Tdg", MQT_NAMED_BUILDER(qco::tdg),
                              MQT_NAMED_BUILDER(qco::tdg)},
        JeffRoundTripTestCase{"SingleControlledTdg",
                              MQT_NAMED_BUILDER(qco::singleControlledTdg),
                              MQT_NAMED_BUILDER(qco::singleControlledTdg)},
        JeffRoundTripTestCase{"MultipleControlledTdg",
                              MQT_NAMED_BUILDER(qco::multipleControlledTdg),
                              MQT_NAMED_BUILDER(qco::multipleControlledTdg)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOUOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"U", MQT_NAMED_BUILDER(qco::u),
                              MQT_NAMED_BUILDER(qco::u)},
        JeffRoundTripTestCase{"SingleControlledU",
                              MQT_NAMED_BUILDER(qco::singleControlledU),
                              MQT_NAMED_BUILDER(qco::singleControlledU)},
        JeffRoundTripTestCase{"MultipleControlledU",
                              MQT_NAMED_BUILDER(qco::multipleControlledU),
                              MQT_NAMED_BUILDER(qco::multipleControlledU)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"X", MQT_NAMED_BUILDER(qco::x),
                              MQT_NAMED_BUILDER(qco::x)},
        JeffRoundTripTestCase{"SingleControlledX",
                              MQT_NAMED_BUILDER(qco::singleControlledX),
                              MQT_NAMED_BUILDER(qco::singleControlledX)},
        JeffRoundTripTestCase{"MultipleControlledX",
                              MQT_NAMED_BUILDER(qco::multipleControlledX),
                              MQT_NAMED_BUILDER(qco::multipleControlledX)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOYOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Y", MQT_NAMED_BUILDER(qco::y),
                              MQT_NAMED_BUILDER(qco::y)},
        JeffRoundTripTestCase{"SingleControlledY",
                              MQT_NAMED_BUILDER(qco::singleControlledY),
                              MQT_NAMED_BUILDER(qco::singleControlledY)},
        JeffRoundTripTestCase{"MultipleControlledY",
                              MQT_NAMED_BUILDER(qco::multipleControlledY),
                              MQT_NAMED_BUILDER(qco::multipleControlledY)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOZOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Z", MQT_NAMED_BUILDER(qco::z),
                              MQT_NAMED_BUILDER(qco::z)},
        JeffRoundTripTestCase{"SingleControlledZ",
                              MQT_NAMED_BUILDER(qco::singleControlledZ),
                              MQT_NAMED_BUILDER(qco::singleControlledZ)},
        JeffRoundTripTestCase{"MultipleControlledZ",
                              MQT_NAMED_BUILDER(qco::multipleControlledZ),
                              MQT_NAMED_BUILDER(qco::multipleControlledZ)}));
/// @}

/// \name JeffRoundTrip/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOMeasureOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{
            "SingleMeasurementToSingleBit",
            MQT_NAMED_BUILDER(qco::singleMeasurementToSingleBit),
            MQT_NAMED_BUILDER(qco::singleMeasurementToSingleBit)},
        JeffRoundTripTestCase{
            "RepeatedMeasurementToSameBit",
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToSameBit),
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToSameBit)},
        JeffRoundTripTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToDifferentBits)},
        JeffRoundTripTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qco::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(
                qco::multipleClassicalRegistersAndMeasurements)}));
/// @}

/// \name JeffRoundTrip/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOResetOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"ResetQubitAfterSingleOp",
                              MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp),
                              MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp)},
        JeffRoundTripTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qco::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(qco::resetMultipleQubitsAfterSingleOp)}));
/// @}

// class QCOToJeffConversionTest : public ::testing::Test {
// protected:
//   std::unique_ptr<mlir::MLIRContext> context;
//   void SetUp() override {
//     // Register all dialects needed for the full compilation pipeline
//     DialectRegistry registry;
//     registry.insert<arith::ArithDialect, func::FuncDialect,
//     jeff::JeffDialect,
//                     mlir::qco::QCODialect>();

//     context = std::make_unique<MLIRContext>();
//     context->appendDialectRegistry(registry);
//     context->loadAllAvailableDialects();
//   }

//   [[nodiscard]] OwningOpRef<ModuleOp> buildQCOIR(
//       const std::function<void(mlir::qco::QCOProgramBuilder&)>& buildFunc)
//       const {
//     mlir::qco::QCOProgramBuilder builder(context.get());
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

// TEST_F(QCOToJeffConversionTest, Measure) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.measure(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.qubit_measure_nd"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, Reset) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.reset(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.qubit_reset"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, GPhase) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     b.gphase(0.5);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.gphase"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, CGPhase) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.cgphase(0.5, q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.gphase"), std::string::npos);
//   ASSERT_NE(outputString.find("num_ctrls = 1"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, Id) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.id(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.i"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, X) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.x(q);
//     b.x(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.x"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, MCX) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(3, "q");
//     auto q0 = reg[0];
//     auto q1 = reg[1];
//     auto q2 = reg[2];
//     b.mcx({q1, q2}, q0);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.x"), std::string::npos);
//   ASSERT_NE(outputString.find("num_ctrls = 2"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, Y) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.y(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.y"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, Z) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.z(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.z"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, H) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.h(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.h"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, S) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.s(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.s"), std::string::npos);
//   ASSERT_NE(outputString.find("is_adjoint = false"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, Sdg) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.sdg(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.s"), std::string::npos);
//   ASSERT_NE(outputString.find("is_adjoint = true"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, T) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.t(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.t"), std::string::npos);
//   ASSERT_NE(outputString.find("is_adjoint = false"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, Tdg) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.tdg(q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.t"), std::string::npos);
//   ASSERT_NE(outputString.find("is_adjoint = true"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, RX) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.rx(0.5, q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.rx"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, CRX) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(2, "q");
//     auto q0 = reg[0];
//     auto q1 = reg[1];
//     b.crx(0.5, q0, q1);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.rx"), std::string::npos);
//   ASSERT_NE(outputString.find("num_ctrls = 1"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, RY) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.ry(0.5, q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.ry"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, RZ) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.rz(0.5, q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.rz"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, P) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.p(0.5, q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.r1"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, U) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(1, "q");
//     auto q = reg[0];
//     b.u(0.1, 0.2, 0.3, q);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.u"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, CU) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(2, "q");
//     auto q0 = reg[0];
//     auto q1 = reg[1];
//     b.cu(0.1, 0.2, 0.3, q1, q0);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.u"), std::string::npos);
//   ASSERT_NE(outputString.find("num_ctrls = 1"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, SWAP) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(2, "q");
//     auto q0 = reg[0];
//     auto q1 = reg[1];
//     b.swap(q0, q1);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.swap"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, CSWAP) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(3, "q");
//     auto q0 = reg[0];
//     auto q1 = reg[1];
//     auto q2 = reg[2];
//     b.cswap(q0, q1, q2);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.swap"), std::string::npos);
//   ASSERT_NE(outputString.find("num_ctrls = 1"), std::string::npos);
// }

// TEST_F(QCOToJeffConversionTest, Bell) {
//   auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
//     auto reg = b.allocQubitRegister(2, "q");
//     auto q0 = reg[0];
//     auto q1 = reg[1];
//     b.h(q0);
//     b.cx(q0, q1);
//   });

//   PassManager pm(context.get());
//   pm.addPass(createQCOToJeff());
//   if (failed(pm.run(input.get()))) {
//     FAIL() << "Error during QCO-to-Jeff conversion";
//   }

//   const auto outputString = getOutputString(input);

//   // ASSERT_EQ(outputString, "test");

//   ASSERT_NE(outputString.find("jeff.h"), std::string::npos);
//   ASSERT_NE(outputString.find("jeff.x"), std::string::npos);
//   ASSERT_NE(outputString.find("num_ctrls = 1"), std::string::npos);
// }

/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/Programs.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>

namespace mqt::test::compiler {
namespace {

constexpr auto BELL_QUAKE = R"mlir(
module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__bell = "__nvqpp__mlirgen__bell_PyKernelEntryPointRewrite"}} {
  func.func @__nvqpp__mlirgen__bell() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %q = quake.alloca !quake.veq<2>
    %q0 = quake.extract_ref %q[0] : (!quake.veq<2>) -> !quake.ref
    %q1 = quake.extract_ref %q[1] : (!quake.veq<2>) -> !quake.ref
    quake.h %q0 : (!quake.ref) -> ()
    quake.x [%q0] %q1 : (!quake.ref, !quake.ref) -> ()
    %m = quake.mz %q name "result" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
    return
  }
})mlir";

constexpr auto STRUCTURED_QUAKE = R"mlir(
module {
  func.func @structured() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %q = quake.alloca !quake.ref
    %theta = arith.constant 5.000000e-01 : f64
    quake.rx (%theta) %q : (f64, !quake.ref) -> ()
    %m = quake.mz %q name "feedback" : (!quake.ref) -> !cc.measure_handle
    %bit = quake.discriminate %m : (!cc.measure_handle) -> i1
    cc.if (%bit) {
      quake.x %q : (!quake.ref) -> ()
    }
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %result = cc.loop while ((%i = %c0) -> (i64)) {
      %active = arith.cmpi slt, %i, %c2 : i64
      cc.condition %active(%i : i64)
    } do {
    ^bb0(%i : i64):
      quake.h %q : (!quake.ref) -> ()
      %next = arith.addi %i, %c1 : i64
      cc.continue %next : i64
    }
    return
  }
})mlir";

TEST(QuakeProgramTest, ImportsCudaQReferenceFormAndPreservesMeasurements) {
  auto quake = mlir::QuakeProgram::fromMLIRString(BELL_QUAKE);
  ASSERT_TRUE(quake);
  auto copied = quake->copy();
  auto qc = std::move(copied).intoQC();
  ASSERT_TRUE(qc);
  const auto text = qc->str();
  EXPECT_NE(text.find("qc.h"), std::string::npos);
  EXPECT_NE(text.find("qc.ctrl"), std::string::npos);
  EXPECT_NE(text.find("register_name = \"result\""), std::string::npos);
  EXPECT_TRUE(quake->isValid());
}

TEST(QuakeProgramTest, ExportsQCAndRoundTripsBackToQC) {
  auto quake = mlir::QuakeProgram::fromMLIRString(BELL_QUAKE);
  ASSERT_TRUE(quake);
  auto qc = std::move(*quake).intoQC();
  ASSERT_TRUE(qc);
  mlir::QuakeExportOptions options;
  options.entryPointName = "round_trip";
  auto exported = std::move(qc->copy()).intoQuake(options);
  ASSERT_TRUE(exported);
  EXPECT_NE(exported->str().find("@__nvqpp__mlirgen__round_trip"),
            std::string::npos);
  EXPECT_NE(exported->str().find(
                "__nvqpp__mlirgen__round_trip = "
                "\"__nvqpp__mlirgen__round_trip_PyKernelEntryPointRewrite\""),
            std::string::npos);
  EXPECT_EQ(exported->str().find("__nvqpp__mlirgen__bell"), std::string::npos);
  EXPECT_NE(exported->str().find("quake.x ["), std::string::npos);
  auto roundTripped = std::move(*exported).intoQC();
  ASSERT_TRUE(roundTripped);
  EXPECT_NE(roundTripped->str().find("qc.measure"), std::string::npos);
}

TEST(QuakeProgramTest, ImportsStructuredFeedbackAndBoundedLoop) {
  auto quake = mlir::QuakeProgram::fromMLIRString(STRUCTURED_QUAKE);
  ASSERT_TRUE(quake);
  auto qc = std::move(*quake).intoQC();
  ASSERT_TRUE(qc);
  const auto text = qc->str();
  EXPECT_NE(text.find("qc.rx"), std::string::npos);
  EXPECT_NE(text.find("scf.if"), std::string::npos);
  EXPECT_NE(text.find("scf.while"), std::string::npos);
}

TEST(QuakeProgramTest, PropagatesFailuresFromConditionalRegions) {
  constexpr auto source = R"mlir(
module {
  func.func @broken() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %q = quake.alloca !quake.ref
    %condition = arith.constant true
    cc.if (%condition) {
      quake.apply @missing %q : (!quake.ref) -> ()
    }
    return
  }
})mlir";
  auto quake = mlir::QuakeProgram::fromMLIRString(source);
  ASSERT_TRUE(quake);
  EXPECT_FALSE(std::move(*quake).intoQC());
}

TEST(QuakeProgramTest, ImportsMeasurementsReturnedFromAppliedKernels) {
  constexpr auto source = R"mlir(
module {
  func.func @measure(%q: !quake.ref) -> !cc.measure_handle
      attributes {"cudaq-kernel"} {
    %measurement = quake.mz %q name "applied" : (!quake.ref) -> !cc.measure_handle
    return %measurement : !cc.measure_handle
  }
  func.func @entry() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %q = quake.alloca !quake.ref
    %measurement = quake.apply @measure %q : (!quake.ref) -> !cc.measure_handle
    %condition = quake.discriminate %measurement : (!cc.measure_handle) -> i1
    cc.if (%condition) {
      quake.x %q : (!quake.ref) -> ()
    }
    return
  }
})mlir";
  auto quake = mlir::QuakeProgram::fromMLIRString(source);
  ASSERT_TRUE(quake);
  auto qc = std::move(*quake).intoQC();
  ASSERT_TRUE(qc);
  const auto text = qc->str();
  EXPECT_NE(text.find("register_name = \"applied\""), std::string::npos);
  EXPECT_NE(text.find("scf.if"), std::string::npos);
  EXPECT_NE(text.find("memref<1xi1>"), std::string::npos);
}

TEST(QuakeProgramTest, ImportsComputeActionLambdas) {
  constexpr auto source = R"mlir(
module {
  func.func @compute_action() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %q = quake.alloca !quake.ref
    %compute = cc.create_lambda {
      cc.scope {
        quake.s %q : (!quake.ref) -> ()
      }
      cc.return
    } : !cc.callable<() -> ()>
    %action = cc.create_lambda {
      cc.scope {
        quake.x %q : (!quake.ref) -> ()
      }
      cc.return
    } : !cc.callable<() -> ()>
    quake.compute_action %compute, %action : !cc.callable<() -> ()>, !cc.callable<() -> ()>
    return
  }
})mlir";
  auto quake = mlir::QuakeProgram::fromMLIRString(source);
  ASSERT_TRUE(quake);
  auto qc = std::move(*quake).intoQC();
  ASSERT_TRUE(qc);
  const auto text = qc->str();
  EXPECT_NE(text.find("qc.s"), std::string::npos);
  EXPECT_NE(text.find("qc.x"), std::string::npos);
  EXPECT_NE(text.find("qc.inv"), std::string::npos);
}

TEST(QuakeProgramTest, ExportsMeasurementFeedbackThroughClassicalRegisters) {
  constexpr auto source = R"mlir(
module {
  func.func @main() -> memref<2xi1>
      attributes {passthrough = ["entry_point"]} {
    %q = qc.alloc : !qc.qubit
    %bits = memref.alloc() : memref<2xi1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %first = qc.measure %q : !qc.qubit -> i1 {register_name = "first"}
    memref.store %first, %bits[%c0] : memref<2xi1>
    %condition = memref.load %bits[%c0] : memref<2xi1>
    scf.if %condition {
      qc.x %q : !qc.qubit
    }
    %second = qc.measure %q : !qc.qubit -> i1 {register_name = "second"}
    memref.store %second, %bits[%c1] : memref<2xi1>
    qc.dealloc %q : !qc.qubit
    return %bits : memref<2xi1>
  }
})mlir";
  auto qc = mlir::QCProgram::fromMLIRString(source);
  ASSERT_TRUE(qc);
  auto quake = std::move(*qc).intoQuake();
  ASSERT_TRUE(quake);
  const auto text = quake->str();
  EXPECT_NE(text.find("quake.discriminate"), std::string::npos);
  EXPECT_NE(text.find("cc.if"), std::string::npos);
  auto roundTripped = std::move(*quake).intoQC();
  ASSERT_TRUE(roundTripped);
  EXPECT_NE(roundTripped->str().find("memref<2xi1>"), std::string::npos);
}

TEST(QuakeProgramTest, AppendsQuakeToTheProgramFormatEnum) {
  EXPECT_EQ(static_cast<uint8_t>(mlir::ProgramFormat::Jeff), 5U);
  EXPECT_EQ(static_cast<uint8_t>(mlir::ProgramFormat::QIRBase), 6U);
  EXPECT_EQ(static_cast<uint8_t>(mlir::ProgramFormat::QIRAdaptive), 7U);
  EXPECT_EQ(static_cast<uint8_t>(mlir::ProgramFormat::Quake), 8U);
}

TEST(QuakeProgramTest, RejectsSSIAllocation) {
  constexpr auto source = R"mlir(
module {
  func.func @ssi() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %q = quake.alloca !quake.wire
    return
  }
})mlir";
  auto quake = mlir::QuakeProgram::fromMLIRString(source);
  ASSERT_TRUE(quake);
  EXPECT_FALSE(std::move(*quake).intoQC());
}

TEST(QuakeProgramTest, RejectsNonzeroGlobalPhaseUnlessExplicitlyIgnored) {
  constexpr auto source = R"mlir(
module {
  func.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
    %phase = arith.constant 2.500000e-01 : f64
    qc.gphase(%phase)
    %zero = arith.constant 0 : i64
    return %zero : i64
  }
})mlir";
  auto qc = mlir::QCProgram::fromMLIRString(source);
  ASSERT_TRUE(qc);
  EXPECT_FALSE(std::move(qc->copy()).intoQuake());
  mlir::QuakeExportOptions options;
  options.ignoreGlobalPhase = true;
  EXPECT_TRUE(std::move(*qc).intoQuake(options));
}

} // namespace
} // namespace mqt::test::compiler

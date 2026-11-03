/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Target.h"
#include "mqt/Target/IQM/Export.h"

#include "gtest/gtest.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <clocale>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace {

class IQMExportTest : public testing::Test {
protected:
  llvm::LLVMContext context;

  static CompilerTarget
  target(std::vector<CompilerTarget::Site> sites = {
             llvm::cantFail(CompilerTarget::Site::create(0, "QBzero")),
             llvm::cantFail(CompilerTarget::Site::create(7, "QBseven")),
             llvm::cantFail(CompilerTarget::Site::create(42, "QBfortyTwo")),
         }) {
    using Capability = CompilerTarget::OperationCapability;
    return llvm::cantFail(CompilerTarget::create(
        std::move(sites), CompilerTarget::Connectivity::allToAll(),
        CompilerTarget::NativeOperations::fromOperations({
            llvm::cantFail(Capability::create("r", 1, 2)),
            llvm::cantFail(Capability::create("cz", 2, 0)),
            llvm::cantFail(Capability::create("measure", 1, 0)),
        })));
  }

  std::unique_ptr<llvm::Module> parse(StringRef body) {
    const auto source = R"qir(
      @register = internal constant [9 x i8] c"read_out\00"
      @unterminated = internal constant [3 x i8] c"bad"
      declare void @__quantum__rt__initialize(ptr)
      declare void @__quantum__qis__prx__body(double, double, ptr)
      declare void @__quantum__qis__cz__body(ptr, ptr)
      declare void @__quantum__qis__mz__body(ptr, ptr)
      declare void @__quantum__rt__array_record_output(i64, ptr)
      declare void @__quantum__rt__result_record_output(ptr, ptr)
      declare void @unknown()
      define i64 @circuit() #0 {
    )qir" + body.str() + R"qir(
      }
      attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
    )qir";
    llvm::SMDiagnostic diagnostic;
    auto module = llvm::parseAssemblyString(source, diagnostic, context);
    if (!module) {
      diagnostic.print("IQMExportTest", llvm::errs());
    }
    return module;
  }

  static constexpr auto MEASURE_AND_RECORD = R"qir(
    call void @__quantum__qis__mz__body(ptr null, ptr null)
    call void @__quantum__rt__result_record_output(ptr null, ptr null)
    ret i64 0
  )qir";
};

TEST_F(IQMExportTest, PreservesNativeParametersAndRecordedOutputOrder) {
  auto module = parse(R"qir(
    call void @__quantum__rt__initialize(ptr null)
    br label %gates
  gates:
    call void @__quantum__qis__prx__body(double 1.25, double -0.75,
                                      ptr inttoptr (i64 7 to ptr))
    call void @__quantum__qis__cz__body(ptr inttoptr (i64 7 to ptr),
                                     ptr inttoptr (i64 42 to ptr))
    br label %measurements
  measurements:
    call void @__quantum__qis__mz__body(ptr inttoptr (i64 7 to ptr), ptr null)
    call void @__quantum__qis__mz__body(ptr inttoptr (i64 42 to ptr),
                                     ptr inttoptr (i64 3 to ptr))
    call void @__quantum__qis__mz__body(ptr null, ptr inttoptr (i64 8 to ptr))
    br label %output
  output:
    call void @__quantum__rt__array_record_output(i64 2, ptr @register)
    call void @__quantum__rt__result_record_output(ptr inttoptr (i64 3 to ptr), ptr null)
    call void @__quantum__rt__result_record_output(ptr null, ptr null)
    call void @__quantum__rt__result_record_output(ptr inttoptr (i64 8 to ptr), ptr null)
    ret i64 0
  )qir");
  ASSERT_TRUE(module);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream, nullptr);
  auto exported = qir::exportIQMJSON(*module, target());
  ASSERT_TRUE(static_cast<bool>(exported))
      << llvm::toString(exported.takeError());
  const auto circuit = llvm::cantFail(llvm::json::parse(*exported));
  EXPECT_EQ(circuit.getAsObject()->getString("name"), "circuit");
  EXPECT_EQ(*circuit.getAsObject()->get("instructions"),
            llvm::cantFail(llvm::json::parse(R"json([
    {"name":"prx","locus":["QBseven"],"args":{"angle":1.25,"phase":-0.75}},
    {"name":"cz","locus":["QBseven","QBfortyTwo"],"args":{}},
    {"name":"measure","locus":["QBfortyTwo"],"args":{"key":"read_out_2_0_0"}},
    {"name":"measure","locus":["QBseven"],"args":{"key":"read_out_2_0_1"}},
    {"name":"measure","locus":["QBzero"],"args":{"key":"c1_1_1_0"}}
  ])json")));
  auto repeated = qir::exportIQMJSON(*module, target());
  ASSERT_TRUE(static_cast<bool>(repeated))
      << llvm::toString(repeated.takeError());
  EXPECT_EQ(*repeated, *exported);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream, nullptr);
  EXPECT_EQ(before, after);
}

TEST_F(IQMExportTest, RejectsUnsupportedCallsParametersAndControlFlow) {
  for (const auto* prefix : {
           "call void @unknown()\n",
           "call void @__quantum__qis__prx__body(double 0x7FF0000000000000, "
           "double 0.0, ptr null)\n",
           "call void @__quantum__qis__prx__body(double 0.5, double 0.0, ptr "
           "inttoptr (i64 1 to ptr))\n",
           "call void @__quantum__qis__cz__body(ptr null, ptr null)\n",
           "call void @__quantum__qis__prx__body(double 1.0, double 0.0, ptr "
           "null)\ncall void @__quantum__rt__initialize(ptr null)\n",
           "br i1 true, label %next, label %next\nnext:\n",
       }) {
    SCOPED_TRACE(prefix);
    auto module = parse(std::string(prefix) + MEASURE_AND_RECORD);
    ASSERT_TRUE(module);
    auto exported = qir::exportIQMJSON(*module, target());
    ASSERT_FALSE(static_cast<bool>(exported));
    llvm::consumeError(exported.takeError());
  }
  for (const auto* body :
       {"br label %loop\nloop: br label %loop", "ret i64 1"}) {
    auto module = parse(body);
    ASSERT_TRUE(module);
    auto exported = qir::exportIQMJSON(*module, target());
    ASSERT_FALSE(static_cast<bool>(exported));
    llvm::consumeError(exported.takeError());
  }
}

TEST_F(IQMExportTest, RequiresTerminalReadoutRecordedExactlyOnce) {
  for (const auto* body : {
           "ret i64 0",
           "call void @__quantum__qis__mz__body(ptr null, ptr null)\nret i64 0",
           "call void @__quantum__rt__result_record_output(ptr null, ptr "
           "null)\nret i64 0",
           "call void @__quantum__qis__mz__body(ptr null, ptr null)\n"
           "call void @__quantum__qis__prx__body(double 0.5, double 0.0, ptr "
           "inttoptr (i64 7 to ptr))\n"
           "call void @__quantum__rt__result_record_output(ptr null, ptr "
           "null)\nret i64 0",
           "call void @__quantum__qis__mz__body(ptr null, ptr null)\n"
           "call void @__quantum__qis__mz__body(ptr null, ptr inttoptr (i64 1 "
           "to ptr))\nret i64 0",
           "call void @__quantum__qis__mz__body(ptr null, ptr null)\n"
           "call void @__quantum__qis__mz__body(ptr inttoptr (i64 7 to ptr), "
           "ptr null)\nret i64 0",
           "call void @__quantum__qis__mz__body(ptr null, ptr null)\n"
           "call void @__quantum__rt__result_record_output(ptr null, ptr "
           "null)\n"
           "call void @__quantum__rt__result_record_output(ptr null, ptr "
           "null)\nret i64 0",
           "call void @__quantum__qis__mz__body(ptr null, ptr null)\n"
           "call void @__quantum__rt__array_record_output(i64 2, ptr null)\n"
           "call void @__quantum__rt__result_record_output(ptr null, ptr "
           "null)\nret i64 0",
       }) {
    SCOPED_TRACE(body);
    auto module = parse(body);
    ASSERT_TRUE(module);
    auto exported = qir::exportIQMJSON(*module, target());
    ASSERT_FALSE(static_cast<bool>(exported));
    llvm::consumeError(exported.takeError());
  }
}

TEST_F(IQMExportTest, RequiresConstantNullTerminatedOutputLabels) {
  for (const auto* label : {"@unterminated", "inttoptr (i64 7 to ptr)"}) {
    auto module = parse(
        std::string(
            "call void @__quantum__qis__mz__body(ptr null, ptr null)\n") +
        "call void @__quantum__rt__array_record_output(i64 1, ptr " + label +
        ")\ncall void @__quantum__rt__result_record_output(ptr null, ptr "
        "null)\n"
        "ret i64 0");
    ASSERT_TRUE(module);
    auto exported = qir::exportIQMJSON(*module, target());
    ASSERT_FALSE(static_cast<bool>(exported));
    llvm::consumeError(exported.takeError());
  }
}

TEST_F(IQMExportTest, RequiresBaseEntryPointAndNamedTarget) {
  auto module = parse(MEASURE_AND_RECORD);
  ASSERT_TRUE(module);
  for (const auto& sites : {
           std::vector{
               llvm::cantFail(CompilerTarget::Site::create(0)),
               llvm::cantFail(CompilerTarget::Site::create(7)),
           },
           std::vector{
               llvm::cantFail(CompilerTarget::Site::create(0, "same")),
               llvm::cantFail(CompilerTarget::Site::create(7, "same")),
           },
       }) {
    auto exported = qir::exportIQMJSON(*module, target(sites));
    ASSERT_FALSE(static_cast<bool>(exported));
    llvm::consumeError(exported.takeError());
  }
  auto* entry = module->getFunction("circuit");
  entry->removeFnAttr("qir_profiles");
  auto exported = qir::exportIQMJSON(*module, target());
  ASSERT_FALSE(static_cast<bool>(exported));
  llvm::consumeError(exported.takeError());
  entry->addFnAttr("qir_profiles", "adaptive_profile");
  exported = qir::exportIQMJSON(*module, target());
  ASSERT_FALSE(static_cast<bool>(exported));
  llvm::consumeError(exported.takeError());
}

} // namespace

TEST_F(IQMExportTest, FormatsParametersIndependentlyOfNumericLocale) {
  auto module =
      parse(std::string("call void @__quantum__qis__prx__body(double 0.25, "
                        "double 0.1, ptr null)\n") +
            MEASURE_AND_RECORD);
  ASSERT_TRUE(module);
  const auto originalLocale = std::string(std::setlocale(LC_NUMERIC, nullptr));
  const auto restoreLocale = llvm::scope_exit(
      [&] { std::setlocale(LC_NUMERIC, originalLocale.c_str()); });
  if (std::setlocale(LC_NUMERIC, "de_DE.UTF-8") == nullptr &&
      std::setlocale(LC_NUMERIC, "de_DE.utf8") == nullptr) {
    GTEST_SKIP() << "A comma-decimal numeric locale is unavailable";
  }
  auto exported = qir::exportIQMJSON(*module, target());
  ASSERT_TRUE(static_cast<bool>(exported))
      << llvm::toString(exported.takeError());
  std::setlocale(LC_NUMERIC, "C");
  auto circuit = llvm::json::parse(*exported);
  ASSERT_TRUE(static_cast<bool>(circuit))
      << llvm::toString(circuit.takeError());
  const auto* args = circuit->getAsObject()
                         ->getArray("instructions")
                         ->front()
                         .getAsObject()
                         ->getObject("args");
  EXPECT_EQ(args->getNumber("angle"), 0.25);
  EXPECT_EQ(args->getNumber("phase"), 0.1);
}

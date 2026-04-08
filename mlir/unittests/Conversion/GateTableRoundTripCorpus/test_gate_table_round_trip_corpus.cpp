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
#include "mlir/Conversion/GateTable.h"
#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"
#include "mlir/Conversion/QCOToJeff/QCOToJeff.h"
#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Conversion/QCToQIR/QCToQIR.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"

#include <gtest/gtest.h>
#include <jeff/IR/JeffDialect.h>
#include <jeff/IR/JeffOps.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/QC/IR/QCDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace mlir;

namespace {

enum class CorpusVariant : uint8_t {
  Base,
  Inverted,
  Ctrl1,
  Ctrl2,
  Ctrl1Inverted,
  Ctrl2Inverted,
};

struct GateCase {
  const char* key = nullptr;
  size_t numTargets = 0;
  size_t numParams = 0;
  CorpusVariant variant = CorpusVariant::Base;
};

} // namespace

static std::string toString(const GateCase& c) {
  std::string s = c.key != nullptr ? c.key : "<null>";
  s += "_t" + std::to_string(c.numTargets);
  s += "_p" + std::to_string(c.numParams);
  s += "_";
  switch (c.variant) {
  case CorpusVariant::Base:
    return s + "base";
  case CorpusVariant::Inverted:
    return s + "inv";
  case CorpusVariant::Ctrl1:
    return s + "c1";
  case CorpusVariant::Ctrl2:
    return s + "c2";
  case CorpusVariant::Ctrl1Inverted:
    return s + "c1inv";
  case CorpusVariant::Ctrl2Inverted:
    return s + "c2inv";
  }
  return s + "unknown";
}

static LogicalResult convertQCOToJeff(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCOToJeff());
  return pm.run(module);
}

static LogicalResult convertJeffToQCO(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createJeffToQCO());
  return pm.run(module);
}

static LogicalResult convertQCToQCO(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCToQCO());
  return pm.run(module);
}

static LogicalResult convertQCOToQC(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCOToQC());
  return pm.run(module);
}

static LogicalResult convertQCToQIR(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCToQIR());
  return pm.run(module);
}

static func::FuncOp createMainFunc(OpBuilder& b, ModuleOp module) {
  auto funcType = b.getFunctionType({}, {b.getI64Type()});
  auto mainFunc = func::FuncOp::create(b.getUnknownLoc(), "main", funcType);
  auto entryPointAttr = b.getStringAttr("entry_point");
  mainFunc->setAttr("passthrough", b.getArrayAttr({entryPointAttr}));
  module.push_back(mainFunc);
  auto& entryBlock = *mainFunc.addEntryBlock();
  b.setInsertionPointToStart(&entryBlock);
  return mainFunc;
}

static std::variant<double, Value> paramValue(size_t i) {
  // Deterministic, distinct values; avoids special angles like 0 or pi.
  return 0.123 + (static_cast<double>(i) * 0.456);
}

template <typename QCOOpType, size_t NumTargets, size_t NumParams,
          size_t... TargetI, size_t... ParamI>
static llvm::SmallVector<Value>
emitGateImpl(OpBuilder& b, Location loc, llvm::ArrayRef<Value> targets,
             std::index_sequence<TargetI...> /*tgt*/,
             std::index_sequence<ParamI...> /*par*/) {
  auto gate =
      QCOOpType::create(b, loc, targets[TargetI]..., paramValue(ParamI)...);
  llvm::SmallVector<Value> out;
  out.reserve(NumTargets);
  for (size_t i = 0; i < NumTargets; ++i) {
    out.push_back(gate->getResult(i));
  }
  return out;
}

template <typename QCOOpType, size_t NumTargets, size_t NumParams>
static llvm::SmallVector<Value> emitGate(OpBuilder& b, Location loc,
                                         llvm::ArrayRef<Value> targets) {
  return emitGateImpl<QCOOpType, NumTargets, NumParams>(
      b, loc, targets, std::make_index_sequence<NumTargets>{},
      std::make_index_sequence<NumParams>{});
}

template <typename QCOOpType, size_t NumTargets, size_t NumParams>
static void buildGateCaseBody(OpBuilder& b, Location loc, const GateCase& tc) {
  constexpr size_t maxCtrls = 2;
  llvm::SmallVector<Value> controls;
  llvm::SmallVector<Value> targets;
  controls.reserve(maxCtrls);
  targets.reserve(NumTargets);

  const int64_t totalQubits =
      static_cast<int64_t>(maxCtrls) + static_cast<int64_t>(NumTargets);
  for (int64_t i = 0; i < totalQubits; ++i) {
    auto q = qco::AllocOp::create(b, loc).getResult();
    if (std::cmp_less(i, maxCtrls)) {
      controls.push_back(q);
    } else {
      targets.push_back(q);
    }
  }

  auto applyBase = [&](ValueRange inTargets) -> llvm::SmallVector<Value> {
    llvm::SmallVector<Value> in;
    in.append(inTargets.begin(), inTargets.end());
    return emitGate<QCOOpType, NumTargets, NumParams>(b, loc, in);
  };

  auto applyInv = [&](ValueRange inTargets) -> llvm::SmallVector<Value> {
    auto invOut =
        qco::InvOp::create(b, loc, inTargets, [&](ValueRange invArgs) {
          return applyBase(invArgs);
        });
    llvm::SmallVector<Value> out;
    out.append(invOut.getQubitsOut().begin(), invOut.getQubitsOut().end());
    return out;
  };

  auto applyCtrl = [&](size_t nCtrls, bool inverted) {
    auto ctrlIns = ValueRange(controls).take_front(nCtrls);
    auto tgtIns = ValueRange(targets);
    auto ctrlOp = qco::CtrlOp::create(
        b, loc, ctrlIns, tgtIns,
        [&](ValueRange bodyTargets) -> llvm::SmallVector<Value> {
          return inverted ? applyInv(bodyTargets) : applyBase(bodyTargets);
        });

    for (size_t i = 0; i < nCtrls; ++i) {
      controls[i] = ctrlOp.getControlsOut()[i];
    }
    for (size_t i = 0; i < NumTargets; ++i) {
      targets[i] = ctrlOp.getTargetsOut()[i];
    }
  };

  switch (tc.variant) {
  case CorpusVariant::Base:
    targets = emitGate<QCOOpType, NumTargets, NumParams>(b, loc, targets);
    break;
  case CorpusVariant::Inverted:
    targets = applyInv(targets);
    break;
  case CorpusVariant::Ctrl1:
    applyCtrl(1, false);
    break;
  case CorpusVariant::Ctrl2:
    applyCtrl(2, false);
    break;
  case CorpusVariant::Ctrl1Inverted:
    applyCtrl(1, true);
    break;
  case CorpusVariant::Ctrl2Inverted:
    applyCtrl(2, true);
    break;
  }

  for (auto q : controls) {
    qco::SinkOp::create(b, loc, q);
  }
  for (auto q : targets) {
    qco::SinkOp::create(b, loc, q);
  }

  auto zero = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(0));
  func::ReturnOp::create(b, loc, zero.getResult());
}

template <typename QCOOpType, size_t NumTargets, size_t NumParams>
static OwningOpRef<ModuleOp> buildGateCase(MLIRContext* ctx,
                                           const GateCase& tc) {
  OpBuilder b(ctx);
  auto loc = FileLineColLoc::get(ctx, "<gate-table-round-trip-corpus>", 1, 1);
  auto module = ModuleOp::create(loc);
  (void)createMainFunc(b, module);
  buildGateCaseBody<QCOOpType, NumTargets, NumParams>(b, loc, tc);
  return {module};
}

template <typename QCOpType, size_t NumTargets, size_t NumParams,
          size_t... TargetI, size_t... ParamI>
static void emitQCGateImpl(OpBuilder& b, Location loc,
                           llvm::ArrayRef<Value> targets,
                           std::index_sequence<TargetI...> /*tgt*/,
                           std::index_sequence<ParamI...> /*par*/) {
  if constexpr (std::is_same_v<QCOpType, qc::U2Op>) {
    static_assert(NumTargets == 1);
    static_assert(NumParams == 2);
    qc::U2Op::create(b, loc, targets[0], paramValue(0), paramValue(1));
  } else {
    QCOpType::create(b, loc, targets[TargetI]..., paramValue(ParamI)...);
  }
}

template <typename QCOpType, size_t NumTargets, size_t NumParams,
          size_t... TargetI, size_t... ParamI>
static void emitQCGateImplReference(OpBuilder& b, Location loc,
                                    llvm::ArrayRef<Value> targets,
                                    std::index_sequence<TargetI...> /*tgt*/,
                                    std::index_sequence<ParamI...> /*par*/) {
  if constexpr (std::is_same_v<QCOpType, qc::U2Op>) {
    static_assert(NumTargets == 1);
    static_assert(NumParams == 2);
    constexpr double piOver2 = 1.5707963267948966;
    // QC→QCO→Jeff lowering encodes U2 as U(π/2, φ, λ).
    qc::UOp::create(b, loc, targets[0], piOver2, paramValue(0), paramValue(1));
  } else {
    QCOpType::create(b, loc, targets[TargetI]..., paramValue(ParamI)...);
  }
}

template <typename QCOpType, size_t NumTargets, size_t NumParams>
static void emitQCGate(OpBuilder& b, Location loc,
                       llvm::ArrayRef<Value> targets) {
  emitQCGateImpl<QCOpType, NumTargets, NumParams>(
      b, loc, targets, std::make_index_sequence<NumTargets>{},
      std::make_index_sequence<NumParams>{});
}

template <typename QCOpType, size_t NumTargets, size_t NumParams>
static void emitQCGateReference(OpBuilder& b, Location loc,
                                llvm::ArrayRef<Value> targets) {
  emitQCGateImplReference<QCOpType, NumTargets, NumParams>(
      b, loc, targets, std::make_index_sequence<NumTargets>{},
      std::make_index_sequence<NumParams>{});
}

template <typename QCOpType, size_t NumTargets, size_t NumParams>
static OwningOpRef<ModuleOp> buildQCGateCase(MLIRContext* ctx,
                                             const GateCase& tc) {
  qc::QCProgramBuilder builder(ctx);
  builder.initialize();

  constexpr size_t maxCtrls = 2;
  const int64_t totalQubits =
      static_cast<int64_t>(maxCtrls) + static_cast<int64_t>(NumTargets);
  auto qubits = builder.allocQubitRegister(totalQubits);

  llvm::SmallVector<Value> controls;
  llvm::SmallVector<Value> targets;
  controls.reserve(maxCtrls);
  targets.reserve(NumTargets);
  for (size_t i = 0; i < maxCtrls; ++i) {
    controls.push_back(qubits[i]);
  }
  for (size_t i = 0; i < NumTargets; ++i) {
    targets.push_back(qubits[maxCtrls + i]);
  }

  auto& b = static_cast<OpBuilder&>(builder);
  auto loc = builder.getLoc();

  auto applyBase = [&]() {
    emitQCGate<QCOpType, NumTargets, NumParams>(b, loc, targets);
  };
  auto applyInv = [&]() { builder.inv([&]() { applyBase(); }); };
  auto applyCtrl = [&](size_t nCtrls, bool inverted) {
    builder.ctrl(ValueRange(controls).take_front(nCtrls), [&]() {
      if (inverted) {
        applyInv();
      } else {
        applyBase();
      }
    });
  };

  switch (tc.variant) {
  case CorpusVariant::Base:
    applyBase();
    break;
  case CorpusVariant::Inverted:
    applyInv();
    break;
  case CorpusVariant::Ctrl1:
    applyCtrl(1, false);
    break;
  case CorpusVariant::Ctrl2:
    applyCtrl(2, false);
    break;
  case CorpusVariant::Ctrl1Inverted:
    applyCtrl(1, true);
    break;
  case CorpusVariant::Ctrl2Inverted:
    applyCtrl(2, true);
    break;
  }

  return builder.finalize();
}

template <typename QCOpType, size_t NumTargets, size_t NumParams>
static OwningOpRef<ModuleOp> buildQCGateCaseReference(MLIRContext* ctx,
                                                      const GateCase& tc) {
  qc::QCProgramBuilder builder(ctx);
  builder.initialize();

  constexpr size_t maxCtrls = 2;
  const int64_t totalQubits =
      static_cast<int64_t>(maxCtrls) + static_cast<int64_t>(NumTargets);
  auto qubits = builder.allocQubitRegister(totalQubits);

  llvm::SmallVector<Value> controls;
  llvm::SmallVector<Value> targets;
  controls.reserve(maxCtrls);
  targets.reserve(NumTargets);
  for (size_t i = 0; i < maxCtrls; ++i) {
    controls.push_back(qubits[i]);
  }
  for (size_t i = 0; i < NumTargets; ++i) {
    targets.push_back(qubits[maxCtrls + i]);
  }

  auto& b = static_cast<OpBuilder&>(builder);
  auto loc = builder.getLoc();

  auto applyBase = [&]() {
    if constexpr (std::is_same_v<QCOpType, qc::U2Op>) {
      static_assert(NumTargets == 1);
      static_assert(NumParams == 2);
      constexpr double pi = std::numbers::pi;
      constexpr double piOver2 = 1.5707963267948966;
      const auto phi = std::get<double>(paramValue(0));
      const auto lambda = std::get<double>(paramValue(1));
      // QC→QCO→Jeff lowering encodes U2 as U(π/2, φ, λ).
      qc::UOp::create(b, loc, targets[0], piOver2, phi, lambda);
    } else {
      emitQCGateReference<QCOpType, NumTargets, NumParams>(b, loc, targets);
    }
  };

  auto applyInv = [&]() {
    if constexpr (std::is_same_v<QCOpType, qc::U2Op>) {
      static_assert(NumTargets == 1);
      static_assert(NumParams == 2);
      constexpr double pi = std::numbers::pi;
      constexpr double piOver2 = 1.5707963267948966;
      const auto phi = std::get<double>(paramValue(0));
      const auto lambda = std::get<double>(paramValue(1));
      // Match the canonical form produced by the QC↔QCO↔Jeff round-trip for
      // `qc.inv { qc.u2(phi, lambda) }`.
      qc::UOp::create(b, loc, targets[0], piOver2, -(pi + lambda), (pi - phi));
    } else {
      builder.inv([&]() { applyBase(); });
    }
  };
  auto applyCtrl = [&](size_t nCtrls, bool inverted) {
    builder.ctrl(ValueRange(controls).take_front(nCtrls), [&]() {
      if (inverted) {
        applyInv();
      } else {
        applyBase();
      }
    });
  };

  switch (tc.variant) {
  case CorpusVariant::Base:
    applyBase();
    break;
  case CorpusVariant::Inverted:
    applyInv();
    break;
  case CorpusVariant::Ctrl1:
    applyCtrl(1, false);
    break;
  case CorpusVariant::Ctrl2:
    applyCtrl(2, false);
    break;
  case CorpusVariant::Ctrl1Inverted:
    applyCtrl(1, true);
    break;
  case CorpusVariant::Ctrl2Inverted:
    applyCtrl(2, true);
    break;
  }

  return builder.finalize();
}

static std::vector<GateCase> makeCases() {
  std::vector<GateCase> cases;

#define MQT_ADD_CASES(KEY, TARGETS, PARAMS, QCO_OP, QC_OP, JEFF_KIND, JEFF_OP, \
                      JEFF_BASE_ADJOINT, JEFF_CUSTOM_NAME, JEFF_PPR, QIR_KIND, \
                      QIR_FN)                                                  \
  do {                                                                         \
    constexpr size_t t = (TARGETS);                                            \
    constexpr size_t p = (PARAMS);                                             \
    cases.push_back(GateCase{#KEY, t, p, CorpusVariant::Base});                \
    cases.push_back(GateCase{#KEY, t, p, CorpusVariant::Inverted});            \
    cases.push_back(GateCase{#KEY, t, p, CorpusVariant::Ctrl1});               \
    cases.push_back(GateCase{#KEY, t, p, CorpusVariant::Ctrl2});               \
    cases.push_back(GateCase{#KEY, t, p, CorpusVariant::Ctrl1Inverted});       \
    cases.push_back(GateCase{#KEY, t, p, CorpusVariant::Ctrl2Inverted});       \
  } while (false);

  MQT_GATE_TABLE(MQT_ADD_CASES)

#undef MQT_ADD_CASES

  return cases;
}

template <typename QCOOpType, size_t Targets, size_t Params>
static void runGateCase(MLIRContext* ctx, const GateCase& tc) {
  ::mqt::test::DeferredPrinter printer;
  auto program = buildGateCase<QCOOpType, Targets, Params>(ctx, tc);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QCO IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*program).succeeded());

  ASSERT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized QCO IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*program).succeeded());

  ASSERT_TRUE(succeeded(convertQCOToJeff(program.get())));
  printer.record(program.get(), "Converted Jeff IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*program).succeeded());

  ASSERT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(),
                 "Canonicalized Converted Jeff IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*program).succeeded());

  ASSERT_TRUE(succeeded(convertJeffToQCO(program.get())));
  printer.record(program.get(), "Converted QCO IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*program).succeeded());

  ASSERT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(),
                 "Canonicalized Converted QCO IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*program).succeeded());

  auto reference = buildGateCase<QCOOpType, Targets, Params>(ctx, tc);
  ASSERT_TRUE(reference);
  ASSERT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  printer.record(reference.get(), "Reference QCO IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*reference).succeeded());

  ASSERT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

template <typename CtxT>
static bool dispatchGateCase(CtxT* ctx, const GateCase& tc) {
#define MQT_DISPATCH(KEY, TARGETS, PARAMS, QCO_OP, QC_OP, JEFF_KIND, JEFF_OP,  \
                     JEFF_BASE_ADJOINT, JEFF_CUSTOM_NAME, JEFF_PPR, QIR_KIND,  \
                     QIR_FN)                                                   \
  do {                                                                         \
    if (tc.numTargets == (TARGETS) && tc.numParams == (PARAMS) &&              \
        std::string_view(tc.key) == std::string_view(#KEY)) {                  \
      runGateCase<QCO_OP, (TARGETS), (PARAMS)>(ctx, tc);                       \
      return true;                                                             \
    }                                                                          \
  } while (false);

  MQT_GATE_TABLE(MQT_DISPATCH)

#undef MQT_DISPATCH
  return false;
}

TEST(GateTableRoundTripCorpus, QCOJeffQCORoundTripAllGates) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, jeff::JeffDialect,
                  qco::QCODialect, qtensor::QTensorDialect>();
  auto ctx = std::make_unique<MLIRContext>();
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();

  const auto cases = makeCases();
  for (const auto& tc : cases) {
    SCOPED_TRACE(toString(tc));
    ASSERT_TRUE(dispatchGateCase(ctx.get(), tc));
  }
}

template <typename QCOpType, size_t Targets, size_t Params>
static void runGateCaseQCChain(MLIRContext* ctx, const GateCase& tc) {
  ::mqt::test::DeferredPrinter printer;
  auto program = buildQCGateCase<QCOpType, Targets, Params>(ctx, tc);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QC IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*program).succeeded());

  ASSERT_TRUE(runQCCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized QC IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*program).succeeded());

  ASSERT_TRUE(succeeded(convertQCToQCO(program.get())));
  printer.record(program.get(), "Converted QCO IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*program).succeeded());

  ASSERT_TRUE(succeeded(convertQCOToJeff(program.get())));
  printer.record(program.get(), "Converted Jeff IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*program).succeeded());

  ASSERT_TRUE(succeeded(convertJeffToQCO(program.get())));
  printer.record(program.get(), "Converted QCO IR 2 (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*program).succeeded());

  ASSERT_TRUE(succeeded(convertQCOToQC(program.get())));
  printer.record(program.get(), "Converted QC IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*program).succeeded());

  ASSERT_TRUE(runQCCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(),
                 "Canonicalized Converted QC IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*program).succeeded());

  OwningOpRef<ModuleOp> reference;
  if constexpr (std::is_same_v<QCOpType, qc::U2Op>) {
    reference = buildQCGateCaseReference<QCOpType, Targets, Params>(ctx, tc);
  } else {
    reference = buildQCGateCase<QCOpType, Targets, Params>(ctx, tc);
  }
  ASSERT_TRUE(reference);
  ASSERT_TRUE(runQCCleanupPipeline(reference.get()).succeeded());
  printer.record(reference.get(), "Reference QC IR (" + toString(tc) + ")");
  ASSERT_TRUE(verify(*reference).succeeded());

  ASSERT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

template <typename CtxT>
static bool dispatchGateCaseQCChain(CtxT* ctx, const GateCase& tc) {
#define MQT_DISPATCH_QC(KEY, TARGETS, PARAMS, QCO_OP, QC_OP, JEFF_KIND,        \
                        JEFF_OP, JEFF_BASE_ADJOINT, JEFF_CUSTOM_NAME,          \
                        JEFF_PPR, QIR_KIND, QIR_FN)                            \
  do {                                                                         \
    if (tc.numTargets == (TARGETS) && tc.numParams == (PARAMS) &&              \
        std::string_view(tc.key) == std::string_view(#KEY)) {                  \
      runGateCaseQCChain<QC_OP, (TARGETS), (PARAMS)>(ctx, tc);                 \
      return true;                                                             \
    }                                                                          \
  } while (false);

  MQT_GATE_TABLE(MQT_DISPATCH_QC)

#undef MQT_DISPATCH_QC
  return false;
}

TEST(GateTableRoundTripCorpus, QCQCOJeffQCOQCRoundTripAllGates) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, jeff::JeffDialect,
                  memref::MemRefDialect, qco::QCODialect, qc::QCDialect,
                  qtensor::QTensorDialect>();
  auto ctx = std::make_unique<MLIRContext>();
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();

  const auto cases = makeCases();
  for (const auto& tc : cases) {
    SCOPED_TRACE(toString(tc));
    ASSERT_TRUE(dispatchGateCaseQCChain(ctx.get(), tc));
  }
}

template <typename QCOpType, size_t Targets, size_t Params>
static OwningOpRef<ModuleOp> buildQCGateCaseForQIR(MLIRContext* ctx,
                                                   const GateCase& tc) {
  qc::QCProgramBuilder builder(ctx);
  builder.initialize();

  constexpr size_t maxCtrls = 2;
  const int64_t totalQubits =
      static_cast<int64_t>(maxCtrls) + static_cast<int64_t>(Targets);
  auto qubits = builder.allocQubitRegister(totalQubits);

  llvm::SmallVector<Value> controls;
  llvm::SmallVector<Value> targets;
  controls.reserve(maxCtrls);
  targets.reserve(Targets);
  for (size_t i = 0; i < maxCtrls; ++i) {
    controls.push_back(qubits[i]);
  }
  for (size_t i = 0; i < Targets; ++i) {
    targets.push_back(qubits[maxCtrls + i]);
  }

  auto& b = static_cast<OpBuilder&>(builder);
  auto loc = builder.getLoc();

  auto applyBase = [&]() {
    if constexpr (std::is_same_v<QCOpType, qc::U2Op>) {
      qc::U2Op::create(b, loc, targets[0], paramValue(0), paramValue(1));
    } else {
      emitQCGateImpl<QCOpType, Targets, Params>(
          b, loc, targets, std::make_index_sequence<Targets>{},
          std::make_index_sequence<Params>{});
    }
  };

  auto applyCtrl = [&](size_t nCtrls) {
    builder.ctrl(ValueRange(controls).take_front(nCtrls),
                 [&]() { applyBase(); });
  };

  switch (tc.variant) {
  case CorpusVariant::Base:
    applyBase();
    break;
  case CorpusVariant::Ctrl1:
    applyCtrl(1);
    break;
  case CorpusVariant::Ctrl2:
    applyCtrl(2);
    break;
  default:
    // We intentionally do not build inverted variants here because QC→QIR
    // currently doesn't support qc.inv.
    break;
  }

  return builder.finalize();
}

template <typename QCOpType, size_t Targets, size_t Params,
          mlir::mqt::gates::QIRKind QIRKindValue, auto QIRFnNameSelector>
static void runGateCaseQCToQIR(MLIRContext* ctx, const GateCase& tc) {
  if constexpr (QIRKindValue != mlir::mqt::gates::QIRKind::Unitary) {
    return;
  } else {
    // QC→QIR does not (currently) support qc.inv; only test the base and ctrl
    // shapes from the GateTable.
    if ((tc.variant == CorpusVariant::Inverted) ||
        (tc.variant == CorpusVariant::Ctrl1Inverted) ||
        (tc.variant == CorpusVariant::Ctrl2Inverted)) {
      return;
    }

    size_t numCtrls = 0;
    switch (tc.variant) {
    case CorpusVariant::Base:
      numCtrls = 0;
      break;
    case CorpusVariant::Ctrl1:
      numCtrls = 1;
      break;
    case CorpusVariant::Ctrl2:
      numCtrls = 2;
      break;
    default:
      return;
    }

    ::mqt::test::DeferredPrinter printer;
    auto program = buildQCGateCaseForQIR<QCOpType, Targets, Params>(ctx, tc);
    ASSERT_TRUE(program);
    printer.record(program.get(), "Original QC IR (" + toString(tc) + ")");
    ASSERT_TRUE(verify(*program).succeeded());

    ASSERT_TRUE(succeeded(convertQCToQIR(program.get())));
    printer.record(program.get(), "Converted QIR IR (" + toString(tc) + ")");
    ASSERT_TRUE(verify(*program).succeeded());

    // Check that the expected runtime function name appears in the IR.
    const auto expected = QIRFnNameSelector(numCtrls).str();
    std::string printed;
    {
      llvm::raw_string_ostream os(printed);
      program.get().print(os);
    }
    ASSERT_NE(printed.find(expected), std::string::npos)
        << "Expected QIR function name substring not found: " << expected;
  }
}

template <typename CtxT>
static bool dispatchGateCaseQCToQIR(CtxT* ctx, const GateCase& tc) {
#define MQT_DISPATCH_QC_TO_QIR(KEY, TARGETS, PARAMS, QCO_OP, QC_OP, JEFF_KIND, \
                               JEFF_OP, JEFF_BASE_ADJOINT, JEFF_CUSTOM_NAME,   \
                               JEFF_PPR, QIR_KIND, QIR_FN)                     \
  do {                                                                         \
    if (tc.numTargets == (TARGETS) && tc.numParams == (PARAMS) &&              \
        std::string_view(tc.key) == std::string_view(#KEY)) {                  \
      runGateCaseQCToQIR<QC_OP, (TARGETS), (PARAMS), (QIR_KIND), &(QIR_FN)>(   \
          ctx, tc);                                                            \
      return true;                                                             \
    }                                                                          \
  } while (false);

  MQT_GATE_TABLE(MQT_DISPATCH_QC_TO_QIR)

#undef MQT_DISPATCH_QC_TO_QIR
  return false;
}

TEST(GateTableRoundTripCorpus, QCToQIRGateTableUnitaryGates) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, LLVM::LLVMDialect, arith::ArithDialect,
                  func::FuncDialect, memref::MemRefDialect>();
  auto ctx = std::make_unique<MLIRContext>();
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();

  const auto cases = makeCases();
  for (const auto& tc : cases) {
    SCOPED_TRACE(toString(tc));
    ASSERT_TRUE(dispatchGateCaseQCToQIR(ctx.get(), tc));
  }
}

static OwningOpRef<ModuleOp> buildMalformedJeffCustom(MLIRContext* ctx,
                                                      StringRef name,
                                                      ValueRange targets,
                                                      ValueRange params) {
  OpBuilder b(ctx);
  auto loc = b.getUnknownLoc();
  auto module = ModuleOp::create(loc);

  // JeffToQCO expects certain Jeff module-level attributes.
  auto ui16 = b.getIntegerType(16, /*isSigned=*/false);
  module->setAttr("jeff.entrypoint", b.getIntegerAttr(ui16, 0));
  module->setAttr("jeff.strings", b.getArrayAttr({b.getStringAttr("main")}));
  module->setAttr("jeff.tool", b.getStringAttr("mqt-core-tests"));
  module->setAttr("jeff.toolVersion", b.getStringAttr("0"));
  module->setAttr("jeff.version", b.getIntegerAttr(ui16, 0));
  module->setAttr("jeff.versionMinor", b.getIntegerAttr(ui16, 0));
  module->setAttr("jeff.versionPatch", b.getIntegerAttr(ui16, 0));
  auto func = func::FuncOp::create(loc, "main", b.getFunctionType({}, {}));
  auto& entryBlock = *func.addEntryBlock();
  b.setInsertionPointToStart(&entryBlock);

  // Create a custom op with no controls, power=1.
  (void)jeff::CustomOp::create(
      b, loc, /*in_target_qubits=*/targets, /*in_ctrl_qubits=*/ValueRange{},
      /*params=*/params,
      /*num_ctrls=*/0,
      /*is_adjoint=*/false,
      /*power=*/1,
      /*name=*/name,
      /*num_targets=*/static_cast<int32_t>(targets.size()),
      /*num_params=*/static_cast<int32_t>(params.size()));

  func::ReturnOp::create(b, loc);
  module.push_back(func);
  return {module};
}

TEST(GateTableRoundTripCorpus, JeffToQCORejectsMalformedCustomArity) {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, jeff::JeffDialect, qco::QCODialect>();
  auto ctx = std::make_unique<MLIRContext>();
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();

  OpBuilder b(ctx.get());
  auto loc = b.getUnknownLoc();

  auto c2 = b.create<jeff::IntConst32Op>(loc, 2);
  auto qureg = b.create<jeff::QuregAllocOp>(loc, c2.getResult());
  auto idx0 = b.create<jeff::IntConst32Op>(loc, 0);
  auto idx1 = b.create<jeff::IntConst32Op>(loc, 1);
  auto ex0 = b.create<jeff::QuregExtractIndexOp>(loc, idx0.getResult(), qureg);
  auto ex1 = b.create<jeff::QuregExtractIndexOp>(loc, idx1.getResult(),
                                                 ex0.getOutQreg());

  auto target0 = ex0.getOutQubit();
  auto target1 = ex1.getOutQubit();

  auto p0 =
      b.create<jeff::FloatConst64Op>(loc, b.getF64FloatAttr(0.123)).getResult();
  auto p1 =
      b.create<jeff::FloatConst64Op>(loc, b.getF64FloatAttr(0.579)).getResult();

  // "r" expects (TARGETS=1, PARAMS=2) in the gate table.
  {
    auto module =
        buildMalformedJeffCustom(ctx.get(), "r",
                                 /*targets=*/ValueRange{target0, target1},
                                 /*params=*/ValueRange{p0, p1});
    ASSERT_TRUE(module);
    EXPECT_TRUE(failed(convertJeffToQCO(*module)));
  }
  {
    auto module = buildMalformedJeffCustom(ctx.get(), "r",
                                           /*targets=*/ValueRange{target0},
                                           /*params=*/ValueRange{p0});
    ASSERT_TRUE(module);
    EXPECT_TRUE(failed(convertJeffToQCO(*module)));
  }
}

TEST(GateTableRoundTripCorpus, JeffToQCORejectsUnknownCustomGateName) {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, jeff::JeffDialect, qco::QCODialect>();
  auto ctx = std::make_unique<MLIRContext>();
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();

  OpBuilder b(ctx.get());
  auto loc = b.getUnknownLoc();
  auto c1 = b.create<jeff::IntConst32Op>(loc, 1);
  auto qureg = b.create<jeff::QuregAllocOp>(loc, c1.getResult());
  auto idx0 = b.create<jeff::IntConst32Op>(loc, 0);
  auto ex0 = b.create<jeff::QuregExtractIndexOp>(loc, idx0.getResult(), qureg);
  auto target0 = ex0.getOutQubit();

  auto module =
      buildMalformedJeffCustom(ctx.get(), "definitely_unsupported_gate",
                               /*targets=*/ValueRange{target0},
                               /*params=*/ValueRange{});
  ASSERT_TRUE(module);
  EXPECT_TRUE(failed(convertJeffToQCO(*module)));
}

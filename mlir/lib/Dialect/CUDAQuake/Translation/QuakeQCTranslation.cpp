/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/CUDAQuake/Translation/QuakeQCTranslation.h"

#include "mlir/Dialect/CUDAQuake/IR/CUDAQuakeCompat.h"
#include "mlir/Dialect/CUDAQuake/IR/CUDAQuakeCompatOps.h" // IWYU pragma: keep
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h" // IWYU pragma: keep

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::cudaq_compat {
namespace {
// Clang-tidy's LLVM profile prefers static free functions, while its general
// readability profile rejects redundant static declarations in this namespace.
// NOLINTBEGIN(llvm-prefer-static-over-anonymous-namespace)

using QubitMap = DenseMap<Value, SmallVector<Value>>;
using ClassicalRegisterMap = DenseMap<Value, SmallVector<Value>>;

struct ImportState {
  IRMapping values;
  QubitMap qubits;
  QubitMap measurements;
  SmallVector<Value> allocatedQubits;
  SmallVector<Value> measurementResults;
};

[[nodiscard]] SmallVector<Value> lookupQubits(const ImportState& state,
                                              const Value value) {
  if (const auto found = state.qubits.find(value);
      found != state.qubits.end()) {
    return found->second;
  }
  return {};
}

[[nodiscard]] SmallVector<Value> flattenQubits(const ImportState& state,
                                               const ValueRange values) {
  SmallVector<Value> result;
  for (const auto value : values) {
    llvm::append_range(result, lookupQubits(state, value));
  }
  return result;
}

[[nodiscard]] func::FuncOp findQuakeEntry(ModuleOp input) {
  func::FuncOp soleFunction;
  size_t functionCount = 0;
  for (auto function : input.getOps<func::FuncOp>()) {
    ++functionCount;
    soleFunction = function;
    if (function->hasAttr("cudaq-entrypoint")) {
      return function;
    }
  }
  return functionCount == 1 ? soleFunction : func::FuncOp{};
}

[[nodiscard]] LogicalResult
emitQCGate(OpBuilder& builder, const Location loc, const StringRef name,
           const ValueRange parameters, const ValueRange explicitControls,
           const llvm::ArrayRef<bool> negativeControls,
           const ValueRange inheritedControls, const ValueRange targets,
           const bool adjoint) {
  SmallVector<Value> controls(inheritedControls);
  llvm::append_range(controls, explicitControls);

  if (!negativeControls.empty() &&
      negativeControls.size() != explicitControls.size()) {
    return emitError(loc, "Quake negative-control mask has the wrong size");
  }
  for (const auto [index, negative] : llvm::enumerate(negativeControls)) {
    if (negative) {
      qc::XOp::create(builder, loc, explicitControls[index]);
    }
  }
  const auto restoreNegativeControls = llvm::make_scope_exit([&] {
    for (const auto [index, negative] : llvm::enumerate(negativeControls)) {
      if (negative) {
        qc::XOp::create(builder, loc, explicitControls[index]);
      }
    }
  });

  auto qcName = name;
  if (name == "r1") {
    qcName = "p";
  } else if (name == "u3") {
    qcName = "u";
  } else if (name == "phased_rx") {
    qcName = "r";
  }
  const auto emitBase = [&](const ValueRange gateTargets) {
    OperationState operationState(loc, ("qc." + qcName).str());
    operationState.addOperands(gateTargets);
    operationState.addOperands(parameters);
    builder.create(operationState);
  };
  const auto emitAdjoint = [&](const ValueRange gateTargets) {
    if (!adjoint) {
      emitBase(gateTargets);
      return;
    }
    qc::InvOp::create(builder, loc, gateTargets,
                      [&](const ValueRange args) { emitBase(args); });
  };
  const auto emitControlled = [&](const ValueRange gateTargets) {
    if (controls.empty()) {
      emitAdjoint(gateTargets);
      return;
    }
    qc::CtrlOp::create(builder, loc, controls, gateTargets,
                       [&](const ValueRange args) { emitAdjoint(args); });
  };

  if (qcName == "swap") {
    if (targets.size() != 2) {
      return emitError(loc, "quake.swap requires exactly two scalar targets");
    }
    emitControlled(targets);
    return success();
  }
  for (const auto target : targets) {
    emitControlled(ValueRange{target});
  }
  return success();
}

class QuakeImporter final {
public:
  explicit QuakeImporter(ModuleOp input)
      : input(input), context(input.getContext()) {}

  [[nodiscard]] FailureOr<OwningOpRef<ModuleOp>> run() {
    auto entry = findQuakeEntry(input);
    if (!entry) {
      return input.emitError(
          "expected exactly one function or a 'cudaq-entrypoint' function");
    }
    if (!entry.getArgumentTypes().empty()) {
      return entry.emitError("Quake import requires a specialized kernel with "
                             "no runtime arguments; use cudaq.synthesize");
    }

    auto result = ModuleOp::create(input.getLoc());
    result->setAttrs(input->getAttrs());
    OpBuilder builder(context);
    builder.setInsertionPointToStart(result.getBody());
    const auto functionType = builder.getFunctionType({}, {});
    auto main =
        func::FuncOp::create(builder, entry.getLoc(), "main", functionType);
    main->setAttr("passthrough",
                  builder.getArrayAttr({builder.getStringAttr("entry_point")}));
    auto& destination = main.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&destination);
    ImportState state;
    if (failed(translateBlock(entry.getBody().front(), builder, state, {},
                              false))) {
      return failure();
    }
    for (const auto qubit : state.allocatedQubits) {
      qc::DeallocOp::create(builder, entry.getLoc(), qubit);
    }
    if (state.measurementResults.empty()) {
      auto zero = arith::ConstantIntOp::create(builder, entry.getLoc(), 0, 64);
      main.setType(builder.getFunctionType({}, {builder.getI64Type()}));
      func::ReturnOp::create(builder, entry.getLoc(), zero.getResult());
    } else {
      const auto registerType = MemRefType::get(
          {static_cast<int64_t>(state.measurementResults.size())},
          builder.getI1Type());
      auto resultRegister =
          memref::AllocOp::create(builder, entry.getLoc(), registerType);
      for (const auto [index, measurement] :
           llvm::enumerate(state.measurementResults)) {
        auto constant = arith::ConstantIndexOp::create(
            builder, entry.getLoc(), static_cast<int64_t>(index));
        memref::StoreOp::create(builder, entry.getLoc(), measurement,
                                resultRegister, ValueRange{constant});
      }
      main.setType(builder.getFunctionType({}, {registerType}));
      func::ReturnOp::create(builder, entry.getLoc(),
                             resultRegister.getResult());
    }
    if (failed(verify(result))) {
      return result.emitError("Quake import produced invalid QC IR");
    }
    return OwningOpRef<ModuleOp>(result);
  }

private:
  [[nodiscard]] LogicalResult translateBlock(Block& source, OpBuilder& builder,
                                             ImportState& state,
                                             const ValueRange inheritedControls,
                                             const bool inheritedAdjoint) {
    for (Operation& operation : source.without_terminator()) {
      if (failed(translateOperation(operation, builder, state,
                                    inheritedControls, inheritedAdjoint))) {
        return failure();
      }
    }
    return success();
  }

  [[nodiscard]] LogicalResult
  translateOperation(Operation& operation, OpBuilder& builder,
                     ImportState& state, const ValueRange inheritedControls,
                     const bool inheritedAdjoint) {
    const auto loc = operation.getLoc();
    if (auto alloca = dyn_cast<quake::QuakeAllocaOp>(operation)) {
      if (alloca.getSize()) {
        return alloca.emitOpError("dynamic Quake allocations are unsupported; "
                                  "specialize kernel arguments first");
      }
      SmallVector<Value> allocated;
      if (auto vectorType =
              dyn_cast<quake::VeqType>(alloca.getReference().getType())) {
        if (!vectorType.hasSpecifiedSize()) {
          return alloca.emitOpError("unsized Quake allocation is unsupported");
        }
        allocated.reserve(vectorType.getSize());
        for (uint64_t i = 0; i < vectorType.getSize(); ++i) {
          allocated.push_back(qc::AllocOp::create(builder, loc).getResult());
        }
      } else if (isa<quake::RefType>(alloca.getReference().getType())) {
        allocated.push_back(qc::AllocOp::create(builder, loc).getResult());
      } else {
        return alloca.emitOpError("unsupported Quake allocation type ")
               << alloca.getReference().getType();
      }
      state.qubits[alloca.getReference()] = std::move(allocated);
      llvm::append_range(state.allocatedQubits,
                         state.qubits[alloca.getReference()]);
      return success();
    }
    if (auto extract = dyn_cast<quake::QuakeExtractRefOp>(operation)) {
      const auto vector = lookupQubits(state, extract.getVeq());
      uint64_t index = extract.getRawIndex();
      if (index == quake::QuakeExtractRefOp::kDynamicIndex) {
        const auto mapped = state.values.lookupOrNull(extract.getIndex());
        auto constant = mapped ? mapped.getDefiningOp<arith::ConstantIntOp>()
                               : arith::ConstantIntOp{};
        if (!constant || constant.value() < 0) {
          return extract.emitOpError("dynamic vector access is unsupported");
        }
        index = static_cast<uint64_t>(constant.value());
      }
      if (index >= vector.size()) {
        return extract.emitOpError("qubit vector index is out of bounds");
      }
      state.qubits[extract.getReference()] = {vector[index]};
      return success();
    }
    if (auto relax = dyn_cast<quake::QuakeRelaxSizeOp>(operation)) {
      state.qubits[relax.getResult()] = lookupQubits(state, relax.getInput());
      return success();
    }
    if (auto concat = dyn_cast<quake::QuakeConcatOp>(operation)) {
      state.qubits[concat.getResult()] =
          flattenQubits(state, concat.getInputs());
      return success();
    }
    if (auto dealloc = dyn_cast<quake::QuakeDeallocOp>(operation)) {
      for (const auto qubit : lookupQubits(state, dealloc.getReference())) {
        qc::DeallocOp::create(builder, loc, qubit);
        llvm::erase(state.allocatedQubits, qubit);
      }
      return success();
    }
    if (auto reset = dyn_cast<quake::QuakeResetOp>(operation)) {
      for (const auto qubit : flattenQubits(state, reset.getTargets())) {
        qc::ResetOp::create(builder, loc, qubit);
      }
      return success();
    }
    const auto isSSIType = [](const Type type) {
      return isa<quake::WireType, quake::CableType>(type);
    };
    if (llvm::any_of(operation.getOperandTypes(), isSSIType) ||
        llvm::any_of(operation.getResultTypes(), isSSIType)) {
      return operation.emitOpError(
          "SSI wire/cable Quake programs are unsupported; synthesize "
          "reference-semantics Quake");
    }
    if (auto apply = dyn_cast<quake::QuakeApplyOp>(operation)) {
      auto callee = input.lookupSymbol<func::FuncOp>(apply.getCallee());
      if (!callee) {
        return apply.emitOpError("cannot resolve Quake callee ")
               << apply.getCallee();
      }
      if (callee.getNumArguments() != apply.getActuals().size()) {
        return apply.emitOpError("callee argument count mismatch");
      }
      ImportState child = state;
      for (const auto [formal, actual] :
           llvm::zip(callee.getArguments(), apply.getActuals())) {
        if (isa<quake::RefType, quake::VeqType>(formal.getType())) {
          child.qubits[formal] = lookupQubits(state, actual);
        } else if (const auto mapped = state.values.lookupOrNull(actual)) {
          child.values.map(formal, mapped);
        } else {
          return apply.emitOpError("unsupported unmapped classical argument");
        }
      }
      SmallVector<Value> controls(inheritedControls);
      llvm::append_range(controls, flattenQubits(state, apply.getControls()));
      if (failed(translateBlock(callee.getBody().front(), builder, child,
                                controls,
                                inheritedAdjoint != apply.getIsAdj()))) {
        return failure();
      }
      auto returned =
          dyn_cast<func::ReturnOp>(callee.getBody().front().getTerminator());
      if (!returned || returned.getNumOperands() != apply.getNumResults()) {
        return apply.emitOpError("callee result count mismatch");
      }
      for (const auto [sourceResult, returnedValue] :
           llvm::zip(apply.getResults(), returned.getOperands())) {
        if (isa<quake::RefType, quake::VeqType>(sourceResult.getType())) {
          const auto qubits = lookupQubits(child, returnedValue);
          if (qubits.empty()) {
            return apply.emitOpError("callee returned an unmapped qubit value");
          }
          state.qubits[sourceResult] = qubits;
          continue;
        }
        if (const auto measurement = child.measurements.find(returnedValue);
            measurement != child.measurements.end()) {
          state.measurements[sourceResult] = measurement->second;
          continue;
        }
        const auto mapped = child.values.lookupOrNull(returnedValue);
        if (!mapped) {
          return apply.emitOpError(
              "callee returned an unmapped classical value");
        }
        state.values.map(sourceResult, mapped);
      }
      state.allocatedQubits = std::move(child.allocatedQubits);
      state.measurementResults = std::move(child.measurementResults);
      return success();
    }
    if (isa<cc::CCCreateLambdaOp>(operation)) {
      return success();
    }
    if (auto scope = dyn_cast<cc::CCScopeOp>(operation)) {
      if (scope.getNumResults() != 0 || !scope.getBody().hasOneBlock()) {
        return scope.emitOpError(
            "only single-block, result-free cc.scope is supported");
      }
      return translateBlock(scope.getBody().front(), builder, state,
                            inheritedControls, inheritedAdjoint);
    }
    if (auto computeAction = dyn_cast<quake::QuakeComputeActionOp>(operation)) {
      const auto translateCallable = [&](const Value callable,
                                         const bool adjoint) -> LogicalResult {
        auto lambda = callable.getDefiningOp<cc::CCCreateLambdaOp>();
        if (!lambda || !lambda.getBody().hasOneBlock()) {
          return computeAction.emitOpError(
              "compute/action operands must be local single-block lambdas");
        }
        const auto signature =
            cast<cc::CallableType>(callable.getType()).getSignature();
        if (signature.getNumInputs() != 0 || signature.getNumResults() != 0) {
          return computeAction.emitOpError(
              "only argument-free, result-free compute/action lambdas are "
              "supported");
        }
        return translateBlock(lambda.getBody().front(), builder, state,
                              inheritedControls, adjoint);
      };
      if (failed(translateCallable(computeAction.getCompute(),
                                   computeAction.getIsDagger())) ||
          failed(
              translateCallable(computeAction.getAction(), inheritedAdjoint)) ||
          failed(translateCallable(computeAction.getCompute(),
                                   !computeAction.getIsDagger()))) {
        return failure();
      }
      return success();
    }

    if (auto measurement = dyn_cast<quake::QuakeMzOp>(operation)) {
      return translateMeasurement(measurement, builder, state, "z");
    }
    if (auto measurement = dyn_cast<quake::QuakeMxOp>(operation)) {
      return translateMeasurement(measurement, builder, state, "x");
    }
    if (auto measurement = dyn_cast<quake::QuakeMyOp>(operation)) {
      return translateMeasurement(measurement, builder, state, "y");
    }
    if (auto discriminate = dyn_cast<quake::QuakeDiscriminateOp>(operation)) {
      const auto found = state.measurements.find(discriminate.getMeasurement());
      if (found == state.measurements.end() || found->second.size() != 1) {
        return discriminate.emitOpError(
            "only scalar measurement discrimination is supported");
      }
      state.values.map(discriminate.getResult(), found->second.front());
      return success();
    }
    if (auto conditional = dyn_cast<cc::CCIfOp>(operation)) {
      if (conditional.getNumResults() != 0) {
        return conditional.emitOpError("result-carrying cc.if is unsupported");
      }
      const auto condition =
          state.values.lookupOrNull(conditional.getCondition());
      if (!condition) {
        return conditional.emitOpError("condition is not a mapped i1 value");
      }
      bool regionFailed = false;
      auto thenBuilder = [&](OpBuilder& nested, Location) {
        ImportState child = state;
        if (failed(translateBlock(conditional.getThenRegion().front(), nested,
                                  child, inheritedControls,
                                  inheritedAdjoint))) {
          regionFailed = true;
          conditional.emitOpError("failed to translate then region");
          return;
        }
        scf::YieldOp::create(nested, loc);
      };
      if (conditional.getElseRegion().empty()) {
        scf::IfOp::create(builder, loc, condition, thenBuilder);
      } else {
        auto elseBuilder = [&](OpBuilder& nested, Location) {
          ImportState child = state;
          if (failed(translateBlock(conditional.getElseRegion().front(), nested,
                                    child, inheritedControls,
                                    inheritedAdjoint))) {
            regionFailed = true;
            conditional.emitOpError("failed to translate else region");
            return;
          }
          scf::YieldOp::create(nested, loc);
        };
        scf::IfOp::create(builder, loc, condition, thenBuilder, elseBuilder);
      }
      if (regionFailed) {
        return failure();
      }
      return success();
    }
    if (auto loop = dyn_cast<cc::CCLoopOp>(operation)) {
      if (loop.getPostCondition() || !loop.getElseRegion().empty()) {
        return loop.emitOpError(
            "post-condition and loop-else forms are unsupported");
      }
      SmallVector<Value> initialValues;
      for (const auto initial : loop.getInitialArgs()) {
        const auto mapped = state.values.lookupOrNull(initial);
        if (!mapped) {
          return loop.emitOpError("unmapped classical loop initializer");
        }
        initialValues.push_back(mapped);
      }
      bool regionFailed = false;
      auto whileOp = scf::WhileOp::create(
          builder, loc, ValueRange(initialValues).getTypes(), initialValues,
          [&](OpBuilder& nested, Location nestedLoc, ValueRange arguments) {
            ImportState child = state;
            for (const auto [sourceArgument, destinationArgument] : llvm::zip(
                     loop.getWhileRegion().front().getArguments(), arguments)) {
              child.values.map(sourceArgument, destinationArgument);
            }
            if (failed(translateBlock(loop.getWhileRegion().front(), nested,
                                      child, inheritedControls,
                                      inheritedAdjoint))) {
              regionFailed = true;
              return;
            }
            auto condition = dyn_cast<cc::CCConditionOp>(
                loop.getWhileRegion().front().getTerminator());
            if (!condition) {
              loop.emitOpError("while region must end in cc.condition");
              regionFailed = true;
              return;
            }
            SmallVector<Value> forwarded;
            for (const auto value : condition.getResults()) {
              forwarded.push_back(child.values.lookup(value));
            }
            scf::ConditionOp::create(
                nested, nestedLoc,
                child.values.lookup(condition.getCondition()), forwarded);
          },
          [&](OpBuilder& nested, Location nestedLoc, ValueRange arguments) {
            ImportState child = state;
            for (const auto [sourceArgument, destinationArgument] : llvm::zip(
                     loop.getBodyRegion().front().getArguments(), arguments)) {
              child.values.map(sourceArgument, destinationArgument);
            }
            if (failed(translateBlock(loop.getBodyRegion().front(), nested,
                                      child, inheritedControls,
                                      inheritedAdjoint))) {
              regionFailed = true;
              return;
            }
            auto continuation = dyn_cast<cc::CCContinueOp>(
                loop.getBodyRegion().front().getTerminator());
            SmallVector<Value> yielded;
            if (continuation) {
              for (const auto value : continuation.getValues()) {
                yielded.push_back(child.values.lookup(value));
              }
            }
            if (!loop.getStepRegion().empty()) {
              ImportState stepState = child;
              for (const auto [sourceArgument, destinationArgument] : llvm::zip(
                       loop.getStepRegion().front().getArguments(), yielded)) {
                stepState.values.map(sourceArgument, destinationArgument);
              }
              if (failed(translateBlock(loop.getStepRegion().front(), nested,
                                        stepState, inheritedControls,
                                        inheritedAdjoint))) {
                regionFailed = true;
                return;
              }
              if (auto stepContinuation = dyn_cast<cc::CCContinueOp>(
                      loop.getStepRegion().front().getTerminator())) {
                yielded.clear();
                for (const auto value : stepContinuation.getValues()) {
                  yielded.push_back(stepState.values.lookup(value));
                }
              }
            }
            scf::YieldOp::create(nested, nestedLoc, yielded);
          });
      if (regionFailed) {
        return failure();
      }
      for (const auto [sourceResult, destinationResult] :
           llvm::zip(loop.getResults(), whileOp.getResults())) {
        state.values.map(sourceResult, destinationResult);
      }
      return success();
    }

    if (succeeded(translateGate(operation, builder, state, inheritedControls,
                                inheritedAdjoint))) {
      return success();
    }
    if (operation.getName().getDialectNamespace() == "quake") {
      return operation.emitOpError("unsupported Quake operation '")
             << operation.getName() << "'";
    }
    if (operation.getName().getDialectNamespace() == "cc") {
      return operation.emitOpError("unsupported CUDA-Q CC operation '")
             << operation.getName() << "'";
    }

    builder.clone(operation, state.values);
    return success();
  }

  template <class MeasurementOp>
  [[nodiscard]] LogicalResult
  translateMeasurement(MeasurementOp measurement, OpBuilder& builder,
                       ImportState& state, const StringRef basis) {
    SmallVector<Value> results;
    for (const auto qubit : flattenQubits(state, measurement.getTargets())) {
      if (basis == "x") {
        qc::HOp::create(builder, measurement.getLoc(), qubit);
      } else if (basis == "y") {
        qc::SdgOp::create(builder, measurement.getLoc(), qubit);
        qc::HOp::create(builder, measurement.getLoc(), qubit);
      }
      auto measured =
          qc::MeasureOp::create(builder, measurement.getLoc(), qubit);
      if (const auto name = measurement.getRegisterNameAttr()) {
        measured->setAttr("register_name", name);
      }
      results.push_back(measured.getResult());
      state.measurementResults.push_back(measured.getResult());
    }
    state.measurements[measurement.getMeasurement()] = results;
    return success();
  }

  [[nodiscard]] static LogicalResult
  translateGate(Operation& operation, OpBuilder& builder, ImportState& state,
                const ValueRange inheritedControls,
                const bool inheritedAdjoint) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(&operation)
        .Case<quake::QuakeHOp, quake::QuakePhasedRxOp, quake::QuakeR1Op,
              quake::QuakeRxOp, quake::QuakeRyOp, quake::QuakeRzOp,
              quake::QuakeSOp, quake::QuakeSwapOp, quake::QuakeTOp,
              quake::QuakeU2Op, quake::QuakeU3Op, quake::QuakeXOp,
              quake::QuakeYOp, quake::QuakeZOp>(
            [&](auto gate) -> LogicalResult {
              SmallVector<Value> parameters;
              for (const auto parameter : gate.getParameters()) {
                const auto mapped = state.values.lookupOrNull(parameter);
                if (!mapped) {
                  gate.emitOpError("unmapped gate parameter");
                  return failure();
                }
                parameters.push_back(mapped);
              }
              const auto controls = flattenQubits(state, gate.getControls());
              const auto targets = flattenQubits(state, gate.getTargets());
              const auto negative = gate.getNegatedQubitControls();
              return emitQCGate(
                  builder, gate.getLoc(), gate->getName().stripDialect(),
                  parameters, controls, negative ? *negative : ArrayRef<bool>{},
                  inheritedControls, targets,
                  inheritedAdjoint != gate.getIsAdj());
            })
        .Default([](Operation*) { return failure(); });
  }

  ModuleOp input;
  MLIRContext* context;
};

[[nodiscard]] std::optional<double> constantFloat(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantFloatOp>()) {
    return constant.value().convertToDouble();
  }
  return std::nullopt;
}

class QCExporter final {
public:
  QCExporter(ModuleOp input, QuakeExportOptions options)
      : input(input), options(std::move(options)), context(input.getContext()) {
  }

  [[nodiscard]] FailureOr<OwningOpRef<ModuleOp>> run() {
    func::FuncOp entry;
    for (auto function : input.getOps<func::FuncOp>()) {
      if (function->hasAttr("passthrough")) {
        entry = function;
        break;
      }
      if (!entry) {
        entry = function;
      }
    }
    if (!entry || !entry.getArgumentTypes().empty()) {
      return input.emitError(
          "QC export requires an argument-free entry function");
    }
    if (failed(checkGlobalPhase(entry))) {
      return failure();
    }

    auto result = ModuleOp::create(input.getLoc());
    result->setAttrs(input->getAttrs());
    OpBuilder builder(context);
    constexpr llvm::StringLiteral nvqppPrefix = "__nvqpp__mlirgen__";
    auto entryPointSymbol = options.entryPointName;
    if (!llvm::StringRef(entryPointSymbol).starts_with(nvqppPrefix)) {
      entryPointSymbol = (nvqppPrefix + entryPointSymbol).str();
    }
    result->setAttr("quake.mangled_name_map",
                    builder.getDictionaryAttr({builder.getNamedAttr(
                        entryPointSymbol,
                        builder.getStringAttr(entryPointSymbol +
                                              "_PyKernelEntryPointRewrite"))}));
    builder.setInsertionPointToStart(result.getBody());
    auto function =
        func::FuncOp::create(builder, entry.getLoc(), entryPointSymbol,
                             builder.getFunctionType({}, {}));
    function->setAttr("cudaq-entrypoint", builder.getUnitAttr());
    function->setAttr("cudaq-kernel", builder.getUnitAttr());
    auto& destination = function.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&destination);
    IRMapping values;
    DenseMap<Value, Value> qregValues;
    ClassicalRegisterMap classicalRegisters;
    if (failed(translateBlock(entry.getBody().front(), builder, values,
                              qregValues, classicalRegisters, {}, false))) {
      return failure();
    }
    func::ReturnOp::create(builder, entry.getLoc());
    if (failed(verify(result))) {
      return result.emitError("QC export produced invalid Quake IR");
    }
    return OwningOpRef<ModuleOp>(result);
  }

private:
  [[nodiscard]] LogicalResult checkGlobalPhase(func::FuncOp entry) const {
    auto result = success();
    entry.walk([&](qc::GPhaseOp phase) {
      const auto value = constantFloat(phase.getTheta());
      if (!value || *value != 0.0) {
        if (!options.ignoreGlobalPhase) {
          phase.emitOpError("nonzero QC global phase cannot be represented by "
                            "CUDA-Q 0.15 reference-form Quake; set "
                            "ignoreGlobalPhase to drop it explicitly");
          result = failure();
        }
      }
    });
    return result;
  }

  [[nodiscard]] LogicalResult
  translateBlock(Block& source, OpBuilder& builder, IRMapping& values,
                 DenseMap<Value, Value>& qregValues,
                 ClassicalRegisterMap& classicalRegisters,
                 const ValueRange inheritedControls,
                 const bool inheritedAdjoint) {
    for (Operation& operation : source.without_terminator()) {
      const auto loc = operation.getLoc();
      if (auto alloc = dyn_cast<qc::AllocOp>(operation)) {
        auto quakeAlloc = quake::QuakeAllocaOp::create(
            builder, loc, quake::RefType::get(context), Value{});
        values.map(alloc.getResult(), quakeAlloc.getReference());
        continue;
      }
      if (auto alloc = dyn_cast<memref::AllocOp>(operation)) {
        const auto type = dyn_cast<MemRefType>(alloc.getType());
        if (type && type.hasStaticShape() && type.getRank() == 1 &&
            isa<qc::QubitType>(type.getElementType())) {
          auto quakeAlloc = quake::QuakeAllocaOp::create(
              builder, loc,
              quake::VeqType::get(context,
                                  static_cast<uint64_t>(type.getDimSize(0))),
              Value{});
          qregValues[alloc.getResult()] = quakeAlloc.getReference();
          continue;
        }
        if (type && type.hasStaticShape() && type.getRank() == 1 &&
            type.getElementType().isInteger(1)) {
          classicalRegisters[alloc.getResult()].resize(
              static_cast<size_t>(type.getDimSize(0)));
          continue;
        }
      }
      if (auto load = dyn_cast<memref::LoadOp>(operation)) {
        const auto found = qregValues.find(load.getMemref());
        if (found != qregValues.end()) {
          auto index =
              load.getIndices().front().getDefiningOp<arith::ConstantIndexOp>();
          if (!index || index.value() < 0) {
            return load.emitOpError(
                "dynamic qubit-register access is unsupported");
          }
          auto extracted = quake::QuakeExtractRefOp::create(
              builder, loc, quake::RefType::get(context), found->second,
              Value{}, builder.getI64IntegerAttr(index.value()));
          values.map(load.getResult(), extracted.getReference());
          continue;
        }
        if (const auto classical = classicalRegisters.find(load.getMemref());
            classical != classicalRegisters.end()) {
          auto index =
              load.getIndices().front().getDefiningOp<arith::ConstantIndexOp>();
          if (!index || index.value() < 0 ||
              static_cast<size_t>(index.value()) >= classical->second.size()) {
            return load.emitOpError(
                "classical register access requires an in-bounds constant "
                "index");
          }
          const auto stored =
              classical->second[static_cast<size_t>(index.value())];
          if (!stored) {
            return load.emitOpError("classical register element is undefined");
          }
          values.map(load.getResult(), stored);
          continue;
        }
        if (const auto type = dyn_cast<MemRefType>(load.getMemref().getType());
            type && type.getElementType().isInteger(1)) {
          return load.emitOpError("unsupported classical register source");
        }
      }
      if (auto store = dyn_cast<memref::StoreOp>(operation)) {
        if (const auto type = dyn_cast<MemRefType>(store.getMemref().getType());
            type && type.getElementType().isInteger(1)) {
          const auto classical = classicalRegisters.find(store.getMemref());
          auto index = store.getIndices()
                           .front()
                           .getDefiningOp<arith::ConstantIndexOp>();
          if (classical == classicalRegisters.end() || !index ||
              index.value() < 0 ||
              static_cast<size_t>(index.value()) >= classical->second.size()) {
            return store.emitOpError(
                "classical register store requires an in-bounds constant "
                "index");
          }
          const auto stored = values.lookupOrNull(store.getValue());
          if (!stored) {
            return store.emitOpError("stored classical value is not mapped");
          }
          classical->second[static_cast<size_t>(index.value())] = stored;
          continue;
        }
      }
      if (auto dealloc = dyn_cast<qc::DeallocOp>(operation)) {
        quake::QuakeDeallocOp::create(builder, loc,
                                      values.lookup(dealloc.getQubit()));
        continue;
      }
      if (auto dealloc = dyn_cast<memref::DeallocOp>(operation)) {
        if (const auto found = qregValues.find(dealloc.getMemref());
            found != qregValues.end()) {
          quake::QuakeDeallocOp::create(builder, loc, found->second);
          continue;
        }
        if (classicalRegisters.contains(dealloc.getMemref())) {
          continue;
        }
      }
      if (auto reset = dyn_cast<qc::ResetOp>(operation)) {
        quake::QuakeResetOp::create(builder, loc, TypeRange{},
                                    values.lookup(reset.getQubit()));
        continue;
      }
      if (auto measurement = dyn_cast<qc::MeasureOp>(operation)) {
        NamedAttrList attrs;
        if (const auto name =
                measurement->getAttrOfType<StringAttr>("register_name")) {
          attrs.set("registerName", name);
        }
        OperationState state(loc, "quake.mz");
        state.addOperands(values.lookup(measurement.getQubit()));
        state.addTypes(cc::MeasureHandleType::get(context));
        state.addAttributes(attrs);
        auto* measured = builder.create(state);
        OperationState discriminate(loc, "quake.discriminate");
        discriminate.addOperands(measured->getResult(0));
        discriminate.addTypes(builder.getI1Type());
        auto* bit = builder.create(discriminate);
        values.map(measurement.getResult(), bit->getResult(0));
        continue;
      }
      if (isa<qc::GPhaseOp>(operation)) {
        continue;
      }
      if (auto ctrl = dyn_cast<qc::CtrlOp>(operation)) {
        SmallVector<Value> controls(inheritedControls);
        for (const auto control : ctrl.getControls()) {
          controls.push_back(values.lookup(control));
        }
        IRMapping child = values;
        for (const auto [argument, target] : llvm::zip(
                 ctrl.getRegion().front().getArguments(), ctrl.getTargets())) {
          child.map(argument, values.lookup(target));
        }
        if (failed(translateBlock(ctrl.getRegion().front(), builder, child,
                                  qregValues, classicalRegisters, controls,
                                  inheritedAdjoint))) {
          return failure();
        }
        continue;
      }
      if (auto inv = dyn_cast<qc::InvOp>(operation)) {
        IRMapping child = values;
        for (const auto [argument, qubit] : llvm::zip(
                 inv.getRegion().front().getArguments(), inv.getQubits())) {
          child.map(argument, values.lookup(qubit));
        }
        if (failed(translateBlock(inv.getRegion().front(), builder, child,
                                  qregValues, classicalRegisters,
                                  inheritedControls, !inheritedAdjoint))) {
          return failure();
        }
        continue;
      }
      if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
        if (conditional.getNumResults() != 0) {
          return conditional.emitOpError(
              "result-carrying scf.if is unsupported for Quake export");
        }
        OperationState state(loc, "cc.if");
        state.addOperands(values.lookup(conditional.getCondition()));
        state.addRegion();
        state.addRegion();
        auto* emitted = builder.create(state);
        const auto emitRegion =
            [&](Region& sourceRegion,
                Region& destinationRegion) -> LogicalResult {
          if (sourceRegion.empty()) {
            return success();
          }
          auto& block = destinationRegion.emplaceBlock();
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&block);
          IRMapping child = values;
          ClassicalRegisterMap childRegisters = classicalRegisters;
          if (failed(translateBlock(sourceRegion.front(), builder, child,
                                    qregValues, childRegisters,
                                    inheritedControls, inheritedAdjoint))) {
            return failure();
          }
          cc::CCContinueOp::create(builder, loc, ValueRange{});
          return success();
        };
        if (failed(emitRegion(conditional.getThenRegion(),
                              emitted->getRegion(0))) ||
            failed(emitRegion(conditional.getElseRegion(),
                              emitted->getRegion(1)))) {
          return failure();
        }
        continue;
      }
      if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
        SmallVector<Value> initialValues;
        for (const auto operand : loop.getInits()) {
          initialValues.push_back(values.lookup(operand));
        }
        OperationState state(loc, "cc.loop");
        state.addOperands(initialValues);
        state.addTypes(ValueRange(initialValues).getTypes());
        state.addAttribute("post_condition", builder.getBoolAttr(false));
        for (size_t i = 0; i < 4; ++i) {
          state.addRegion();
        }
        auto* emitted = builder.create(state);
        {
          OpBuilder::InsertionGuard guard(builder);
          auto& block = emitted->getRegion(0).emplaceBlock();
          for (const auto argument : loop.getBefore().front().getArguments()) {
            block.addArgument(argument.getType(), argument.getLoc());
          }
          builder.setInsertionPointToStart(&block);
          IRMapping child = values;
          for (const auto [sourceArgument, destinationArgument] :
               llvm::zip(loop.getBefore().front().getArguments(),
                         block.getArguments())) {
            child.map(sourceArgument, destinationArgument);
          }
          if (failed(translateBlock(loop.getBefore().front(), builder, child,
                                    qregValues, classicalRegisters,
                                    inheritedControls, inheritedAdjoint))) {
            return failure();
          }
          auto condition =
              cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
          SmallVector<Value> forwarded;
          for (const auto value : condition.getArgs()) {
            forwarded.push_back(child.lookup(value));
          }
          cc::CCConditionOp::create(
              builder, loc, child.lookup(condition.getCondition()), forwarded);
        }
        {
          OpBuilder::InsertionGuard guard(builder);
          auto& block = emitted->getRegion(1).emplaceBlock();
          for (const auto argument : loop.getAfter().front().getArguments()) {
            block.addArgument(argument.getType(), argument.getLoc());
          }
          builder.setInsertionPointToStart(&block);
          IRMapping child = values;
          for (const auto [sourceArgument, destinationArgument] :
               llvm::zip(loop.getAfter().front().getArguments(),
                         block.getArguments())) {
            child.map(sourceArgument, destinationArgument);
          }
          if (failed(translateBlock(loop.getAfter().front(), builder, child,
                                    qregValues, classicalRegisters,
                                    inheritedControls, inheritedAdjoint))) {
            return failure();
          }
          auto yielded =
              cast<scf::YieldOp>(loop.getAfter().front().getTerminator());
          SmallVector<Value> forwarded;
          for (const auto value : yielded.getResults()) {
            forwarded.push_back(child.lookup(value));
          }
          cc::CCContinueOp::create(builder, loc, forwarded);
        }
        for (const auto [sourceResult, destinationResult] :
             llvm::zip(loop.getResults(), emitted->getResults())) {
          values.map(sourceResult, destinationResult);
        }
        continue;
      }
      if (auto unitary = dyn_cast<qc::UnitaryOpInterface>(operation)) {
        SmallVector<Value> parameters;
        for (const auto parameter : unitary.getParameters()) {
          parameters.push_back(values.lookup(parameter));
        }
        SmallVector<Value> targets;
        for (const auto target : unitary.getTargets()) {
          targets.push_back(values.lookup(target));
        }
        SmallVector<Value> controls(inheritedControls);
        for (const auto control : unitary.getControls()) {
          controls.push_back(values.lookup(control));
        }
        auto name = unitary.getBaseSymbol();
        if (name == "p") {
          name = "r1";
        } else if (name == "u") {
          name = "u3";
        }
        OperationState state(loc, ("quake." + name).str());
        state.addOperands(parameters);
        state.addOperands(controls);
        state.addOperands(targets);
        state.addAttribute("operand_segment_sizes",
                           builder.getDenseI32ArrayAttr(
                               {static_cast<int32_t>(parameters.size()),
                                static_cast<int32_t>(controls.size()),
                                static_cast<int32_t>(targets.size())}));
        if (inheritedAdjoint) {
          state.addAttribute("is_adj", builder.getUnitAttr());
        }
        builder.create(state);
        continue;
      }
      if (const auto* dialect = operation.getDialect();
          dialect != nullptr && dialect->getNamespace() == "qc") {
        return operation.emitOpError(
            "unsupported QC operation for Quake export");
      }
      if (llvm::any_of(operation.getOperands(), [&](const Value operand) {
            return !values.contains(operand);
          })) {
        return operation.emitOpError(
            "cannot export operation with an unmapped operand");
      }
      builder.clone(operation, values);
    }
    return success();
  }

  ModuleOp input;
  QuakeExportOptions options;
  MLIRContext* context;
};

// NOLINTEND(llvm-prefer-static-over-anonymous-namespace)
} // namespace

FailureOr<OwningOpRef<ModuleOp>> translateQuakeToQC(ModuleOp input) {
  return QuakeImporter(input).run();
}

FailureOr<OwningOpRef<ModuleOp>>
translateQCToQuake(ModuleOp input, const QuakeExportOptions& options) {
  return QCExporter(input, options).run();
}

} // namespace mlir::cudaq_compat

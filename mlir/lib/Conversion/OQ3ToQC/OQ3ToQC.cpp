/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/OQ3ToQC/OQ3ToQC.h"

#include "mlir/Dialect/OQ3/IR/GateCatalog.h"
#include "mlir/Dialect/OQ3/IR/OQ3Ops.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::oq3 {
#define GEN_PASS_DEF_OQ3TOQC
#include "mlir/Conversion/OQ3ToQC/OQ3ToQC.h.inc"

namespace {

class OQ3ToQCPass final : public impl::OQ3ToQCBase<OQ3ToQCPass> {
public:
  void runOnOperation() override {
    auto configureTarget = [&](ConversionTarget& target) {
      target.addIllegalDialect<OQ3Dialect>();
      target.addLegalOp<GateOp, GateDeclOp, YieldOp>();
      target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    };

    llvm::SmallVector<GateOp> gates;
    if (failed(collectReachableGates(gates))) {
      signalPassFailure();
      return;
    }
    llvm::DenseSet<Operation*> reachable;
    for (auto gate : gates) {
      reachable.insert(gate.getOperation());
    }
    llvm::SmallVector<GateOp> unreachableGates;
    for (auto gate : getOperation().getOps<GateOp>()) {
      if (!reachable.contains(gate.getOperation())) {
        unreachableGates.push_back(gate);
      }
    }
    for (auto gate : unreachableGates) {
      gate.erase();
    }
    for (auto gate : gates) {
      llvm::SmallVector<Operation*> bodyOperations;
      for (auto& operation : gate.getBody().getOps()) {
        bodyOperations.push_back(&operation);
      }
      ConversionTarget gateTarget(getContext());
      configureTarget(gateTarget);
      RewritePatternSet gatePatterns(&getContext());
      gatePatterns.add<ApplyGateOpConversion>(&getContext(), *this);
      if (failed(applyFullConversion(bodyOperations, gateTarget,
                                     std::move(gatePatterns)))) {
        signalPassFailure();
        return;
      }
    }

    ConversionTarget target(getContext());
    configureTarget(target);

    RewritePatternSet patterns(&getContext());
    patterns.add<ApplyGateOpConversion>(&getContext(), *this);
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<Operation*> declarations;
    getOperation().walk([&](Operation* op) {
      if (isa<GateOp, GateDeclOp>(op)) {
        declarations.push_back(op);
      }
    });
    for (Operation* declaration : declarations) {
      declaration->erase();
    }

    ConversionTarget finalTarget(getContext());
    finalTarget.addIllegalDialect<OQ3Dialect>();
    finalTarget.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    if (failed(applyFullConversion(getOperation(), finalTarget, {}))) {
      signalPassFailure();
    }
  }

private:
  enum class VisitState : std::uint8_t { Unvisited, Active, Complete };

  LogicalResult collectReachableGates(SmallVectorImpl<GateOp>& postorder) {
    llvm::DenseMap<Operation*, VisitState> states;
    llvm::DenseMap<Operation*, std::size_t> expansionCosts;
    constexpr std::size_t expansionLimit = 100000;
    std::size_t totalExpansionCost = 0;

    const auto visit = [&](auto&& self, GateOp gate) -> LogicalResult {
      auto& state = states[gate.getOperation()];
      if (state == VisitState::Complete) {
        return success();
      }
      if (state == VisitState::Active) {
        return gate.emitError("recursive custom gates cannot be lowered");
      }
      state = VisitState::Active;
      std::size_t expansionCost = 1;
      WalkResult result = gate.getBody().walk([&](ApplyGateOp application) {
        auto callee =
            dyn_cast_or_null<GateOp>(SymbolTable::lookupNearestSymbolFrom(
                application.getOperation(), application.getCalleeAttr()));
        if (!callee) {
          if (++expansionCost > expansionLimit) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        }
        if (failed(self(self, callee))) {
          return WalkResult::interrupt();
        }
        const auto dependencyCost =
            expansionCosts.lookup(callee.getOperation());
        if (dependencyCost > expansionLimit - expansionCost) {
          expansionCost = expansionLimit + 1;
          return WalkResult::interrupt();
        }
        expansionCost += dependencyCost;
        return WalkResult::advance();
      });
      if (result.wasInterrupted()) {
        if (states[gate.getOperation()] == VisitState::Active &&
            expansionCost <= expansionLimit) {
          return failure();
        }
        return gate.emitError(
            "custom-gate expansion exceeds the safe lowering limit");
      }
      state = VisitState::Complete;
      expansionCosts[gate.getOperation()] = expansionCost;
      postorder.push_back(gate);
      return success();
    };

    LogicalResult result = success();
    getOperation().walk([&](ApplyGateOp application) {
      if (application->getParentOfType<GateOp>()) {
        return WalkResult::advance();
      }
      auto gate = dyn_cast_or_null<GateOp>(SymbolTable::lookupNearestSymbolFrom(
          application.getOperation(), application.getCalleeAttr()));
      if (gate) {
        if (failed(visit(visit, gate))) {
          result = failure();
          return WalkResult::interrupt();
        }
        const auto rootCost = expansionCosts.lookup(gate.getOperation());
        if (rootCost > expansionLimit - totalExpansionCost) {
          (void)application.emitError(
              "module custom-gate expansion exceeds the safe lowering limit");
          result = failure();
          return WalkResult::interrupt();
        }
        totalExpansionCost += rootCost;
      }
      return WalkResult::advance();
    });
    return result;
  }

  class ApplyGateOpConversion final : public OpConversionPattern<ApplyGateOp> {
  public:
    ApplyGateOpConversion(MLIRContext* context, OQ3ToQCPass& pass)
        : OpConversionPattern(context), pass(pass) {}

    LogicalResult
    matchAndRewrite(ApplyGateOp application, OpAdaptor /*adaptor*/,
                    ConversionPatternRewriter& rewriter) const override {
      return pass.lowerGateApplication(application, rewriter);
    }

  private:
    OQ3ToQCPass& pass;
  };

  static LogicalResult emitPrimitive(OpBuilder& builder, Location loc,
                                     StringRef name, ValueRange parameters,
                                     ValueRange qubits) {
    auto operationName =
        llvm::StringSwitch<StringRef>(name)
            .Case("gphase", qc::GPhaseOp::getOperationName())
            .Case("id", qc::IdOp::getOperationName())
            .Case("x", qc::XOp::getOperationName())
            .Case("y", qc::YOp::getOperationName())
            .Case("z", qc::ZOp::getOperationName())
            .Case("h", qc::HOp::getOperationName())
            .Case("s", qc::SOp::getOperationName())
            .Case("sdg", qc::SdgOp::getOperationName())
            .Case("t", qc::TOp::getOperationName())
            .Case("tdg", qc::TdgOp::getOperationName())
            .Case("sx", qc::SXOp::getOperationName())
            .Case("sxdg", qc::SXdgOp::getOperationName())
            .Case("p", qc::POp::getOperationName())
            .Case("rx", qc::RXOp::getOperationName())
            .Case("ry", qc::RYOp::getOperationName())
            .Case("rz", qc::RZOp::getOperationName())
            .Case("r", qc::ROp::getOperationName())
            .Case("u2", qc::U2Op::getOperationName())
            .Case("U", qc::UOp::getOperationName())
            .Case("swap", qc::SWAPOp::getOperationName())
            .Case("iswap", qc::iSWAPOp::getOperationName())
            .Case("dcx", qc::DCXOp::getOperationName())
            .Case("ecr", qc::ECROp::getOperationName())
            .Case("rxx", qc::RXXOp::getOperationName())
            .Case("ryy", qc::RYYOp::getOperationName())
            .Case("rzx", qc::RZXOp::getOperationName())
            .Case("rzz", qc::RZZOp::getOperationName())
            .Case("xx_plus_yy", qc::XXPlusYYOp::getOperationName())
            .Case("xx_minus_yy", qc::XXMinusYYOp::getOperationName())
            .Default({});
    if (operationName.empty()) {
      return failure();
    }

    OperationState state(loc, operationName);
    if (name == "gphase") {
      state.addOperands(parameters);
    } else {
      state.addOperands(qubits);
      state.addOperands(parameters);
    }
    builder.create(state);
    return success();
  }

  LogicalResult emitResolvedGate(OpBuilder& builder, ApplyGateOp application,
                                 Operation* declaration, ValueRange parameters,
                                 ValueRange qubits) const {
    if (auto gate = dyn_cast<GateOp>(declaration)) {
      IRMapping mapping;
      llvm::SmallVector<Value> arguments(parameters.begin(), parameters.end());
      arguments.append(qubits.begin(), qubits.end());
      if (arguments.size() != gate.getBody().front().getNumArguments()) {
        return application.emitError(
            "custom-gate operands do not match its verified declaration");
      }
      mapping.map(gate.getBody().front().getArguments(), arguments);
      for (Operation& operation : gate.getBody().front().without_terminator()) {
        builder.clone(operation, mapping);
      }
      return success();
    }

    auto resolvedName = application.getCallee();
    const GateCatalogEntry* catalogEntry = lookupGate(resolvedName);
    if (!catalogEntry) {
      return application.emitError() << "gate '" << resolvedName
                                     << "' has no canonical QC lowering entry";
    }
    if (qubits.size() < catalogEntry->targetCount) {
      return application.emitError(
          "gate has fewer qubit operands than its target count");
    }
    const size_t controls = catalogEntry->variadicControls
                                ? qubits.size() - catalogEntry->targetCount
                                : catalogEntry->controlCount;
    if (qubits.size() < controls + catalogEntry->targetCount) {
      return application.emitError(
          "implicit-control count exceeds gate operands");
    }
    auto primitive = catalogEntry->primitive;
    auto emitCatalogPrimitive = [&](ValueRange primitiveQubits) {
      if (!catalogEntry->inverse) {
        return emitPrimitive(builder, application.getLoc(), primitive,
                             parameters, primitiveQubits);
      }
      LogicalResult result = success();
      qc::InvOp::create(builder, application.getLoc(), primitiveQubits,
                        [&](ValueRange aliases) {
                          result =
                              emitPrimitive(builder, application.getLoc(),
                                            primitive, parameters, aliases);
                        });
      return result;
    };
    if (controls == 0) {
      if (failed(emitCatalogPrimitive(qubits))) {
        return application.emitError()
               << "gate '" << resolvedName
               << "' has no QC lowering for the selected target";
      }
      return success();
    }

    auto controlValues = qubits.take_front(controls);
    auto targets = qubits.drop_front(controls);
    ValueRange primitiveParameters = parameters;
    if (resolvedName == "cu") {
      if (controls != 1 || controlValues.size() != 1 || targets.size() != 1 ||
          parameters.size() != 4) {
        return application.emitError(
            "cu operands do not match its verified standard signature");
      }
      // OpenQASM's four-parameter cu applies p(gamma) to the control before a
      // controlled U(theta, phi, lambda). Keep the relative phase instead of
      // silently treating cu as the three-parameter cu3 alias.
      qc::POp::create(builder, application.getLoc(), controlValues.front(),
                      parameters.back());
      primitiveParameters = parameters.drop_back();
    }
    qc::CtrlOp::create(
        builder, application.getLoc(), controlValues, targets,
        [&](ValueRange aliases) {
          if (catalogEntry->inverse) {
            qc::InvOp::create(builder, application.getLoc(), aliases,
                              [&](ValueRange inverseAliases) {
                                (void)emitPrimitive(
                                    builder, application.getLoc(), primitive,
                                    primitiveParameters, inverseAliases);
                              });
          } else {
            (void)emitPrimitive(builder, application.getLoc(), primitive,
                                primitiveParameters, aliases);
          }
        });
    return success();
  }

  LogicalResult
  lowerGateApplication(ApplyGateOp application,
                       ConversionPatternRewriter& rewriter) const {
    Operation* declaration = SymbolTable::lookupNearestSymbolFrom(
        application.getOperation(), application.getCalleeAttr());
    if (declaration == nullptr) {
      return application.emitError("cannot lower an unresolved gate symbol");
    }

    if (auto gate = dyn_cast<GateOp>(declaration);
        gate &&
        llvm::any_of(application.getModifierKinds(), [](const auto raw) {
          const auto kind = static_cast<GateModifierKind>(raw);
          return kind == GateModifierKind::inv ||
                 kind == GateModifierKind::ctrl ||
                 kind == GateModifierKind::negctrl;
        })) {
      const auto containsStructuredControlFlow =
          gate.walk([&](Operation* nested) {
                if (nested->getName().getDialectNamespace() == "scf" &&
                    nested->getNumRegions() > 0) {
                  return WalkResult::interrupt();
                }
                return WalkResult::advance();
              })
              .wasInterrupted();
      if (containsStructuredControlFlow) {
        return application.emitError(
            "modifiers on custom gates with structured control flow cannot "
            "be represented by the QC target");
      }
    }

    llvm::SmallVector<int64_t> controlCounts(
        application.getModifierKinds().size(), 0);
    llvm::SmallVector<Value> negativeControls;
    size_t controlOffset = 0;
    for (const auto [position, rawKind] :
         llvm::enumerate(application.getModifierKinds())) {
      const auto kind = static_cast<GateModifierKind>(rawKind);
      if (kind == GateModifierKind::pow) {
        return application.emitError(
            "pow gate modifiers are preserved in OQ3 until QC power support "
            "is available");
      }
      if (kind != GateModifierKind::ctrl && kind != GateModifierKind::negctrl) {
        continue;
      }
      const int32_t operandIndex =
          application.getModifierOperandIndices()[position];
      if (operandIndex < 0) {
        controlCounts[position] = 1;
      } else {
        auto constant = application.getModifierOperands()[operandIndex]
                            .getDefiningOp<arith::ConstantIntOp>();
        if (!constant || constant.value() <= 0) {
          return application.emitError(
              "dynamic control counts cannot be lowered to the selected "
              "target");
        }
        controlCounts[position] = constant.value();
      }

      const size_t controlCount = controlCounts[position];
      if (application.getQubits().size() < controlOffset + controlCount) {
        return application.emitError(
            "modifier control count exceeds the available gate operands");
      }
      if (kind == GateModifierKind::negctrl) {
        auto controls =
            application.getQubits().slice(controlOffset, controlCount);
        negativeControls.append(controls.begin(), controls.end());
      }
      controlOffset += controlCount;
    }

    rewriter.setInsertionPoint(application);
    OpBuilder& builder = rewriter;
    for (auto control : negativeControls) {
      qc::XOp::create(builder, application.getLoc(), control);
    }
    auto result = emitModifiers(builder, application, declaration,
                                controlCounts, 0, application.getQubits());
    for (auto control : negativeControls) {
      qc::XOp::create(builder, application.getLoc(), control);
    }
    if (failed(result)) {
      return failure();
    }
    rewriter.eraseOp(application);
    return success();
  }

  LogicalResult emitModifiers(OpBuilder& builder, ApplyGateOp application,
                              Operation* declaration,
                              ArrayRef<int64_t> controlCounts,
                              const size_t position, ValueRange qubits) const {
    if (position == application.getModifierKinds().size()) {
      return emitResolvedGate(builder, application, declaration,
                              application.getParameters(), qubits);
    }
    const auto kind =
        static_cast<GateModifierKind>(application.getModifierKinds()[position]);
    if (kind == GateModifierKind::inv) {
      LogicalResult result = success();
      qc::InvOp::create(
          builder, application.getLoc(), qubits, [&](ValueRange aliases) {
            result = emitModifiers(builder, application, declaration,
                                   controlCounts, position + 1, aliases);
          });
      return result;
    }

    return emitControls(builder, application, declaration, controlCounts,
                        position + 1, controlCounts[position], qubits);
  }

  LogicalResult emitControls(OpBuilder& builder, ApplyGateOp application,
                             Operation* declaration,
                             ArrayRef<int64_t> controlCounts,
                             const size_t nextPosition,
                             const size_t remainingControls,
                             ValueRange qubits) const {
    if (remainingControls == 0) {
      return emitModifiers(builder, application, declaration, controlCounts,
                           nextPosition, qubits);
    }

    LogicalResult result = success();
    qc::CtrlOp::create(builder, application.getLoc(), qubits.take_front(1),
                       qubits.drop_front(1), [&](ValueRange aliases) {
                         result = emitControls(
                             builder, application, declaration, controlCounts,
                             nextPosition, remainingControls - 1, aliases);
                       });
    return result;
  }
};

} // namespace

std::unique_ptr<Pass> createOQ3ToQCPass() {
  return std::make_unique<OQ3ToQCPass>();
}

} // namespace mlir::oq3

/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/OQ3/IR/GateCatalog.h"
#include "mlir/Dialect/OQ3/IR/OQ3Ops.h"
#include "mlir/Dialect/OQ3/Transforms/Passes.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>

namespace mlir::oq3 {
#define GEN_PASS_DEF_LOWEROQ3TOQC
#include "mlir/Dialect/OQ3/Transforms/Passes.h.inc"

namespace {

class LowerOQ3ToQCPass final : public impl::LowerOQ3ToQCBase<LowerOQ3ToQCPass> {
public:
  explicit LowerOQ3ToQCPass(const OpenQASMLoweringOptions /*options*/) {}

  void runOnOperation() override {
    llvm::SmallVector<ForOp> loops;
    const WalkResult ranges = getOperation().walk([&](ForOp op) {
      auto constant = op.getStep().getDefiningOp<arith::ConstantIntOp>();
      if (!constant) {
        op.emitError("dynamic range step cannot be proven nonzero for the "
                     "selected target");
        return WalkResult::interrupt();
      }
      if (constant.value() == 0) {
        op.emitError("OpenQASM range step cannot be zero");
        return WalkResult::interrupt();
      }
      loops.push_back(op);
      return WalkResult::advance();
    });
    if (ranges.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<ApplyGateOp> applications;
    getOperation().walk([&](ApplyGateOp op) { applications.push_back(op); });
    for (ApplyGateOp application : applications) {
      if (failed(lowerGateApplication(application))) {
        signalPassFailure();
        return;
      }
    }

    for (ForOp loop : llvm::reverse(loops)) {
      if (failed(lowerInclusiveRange(loop))) {
        signalPassFailure();
        return;
      }
    }

    lowerBitInterfaces();

    llvm::SmallVector<Operation*> declarations;
    getOperation().walk([&](Operation* op) {
      if (isa<GateOp, GateDeclOp>(op)) {
        declarations.push_back(op);
      }
    });
    for (Operation* declaration : declarations) {
      declaration->erase();
    }
  }

private:
  void lowerBitInterfaces() {
    getOperation().walk([&](func::FuncOp function) {
      llvm::SmallVector<Type> inputTypes;
      inputTypes.reserve(function.getNumArguments());
      for (BlockArgument argument : function.getArguments()) {
        Type type = argument.getType();
        if (const auto bit = dyn_cast<BitType>(type)) {
          type = IntegerType::get(function.getContext(), bit.getWidth());
          argument.setType(type);
        }
        inputTypes.push_back(type);
      }
      llvm::SmallVector<Type> resultTypes;
      for (Type type : function.getResultTypes()) {
        if (const auto bit = dyn_cast<BitType>(type)) {
          type = IntegerType::get(function.getContext(), bit.getWidth());
        }
        resultTypes.push_back(type);
      }
      function.setType(
          FunctionType::get(function.getContext(), inputTypes, resultTypes));
    });

    llvm::SmallVector<UnpackBitOp> unpackOperations;
    getOperation().walk(
        [&](UnpackBitOp operation) { unpackOperations.push_back(operation); });
    for (UnpackBitOp operation : unpackOperations) {
      OpBuilder builder(operation);
      Value value = operation->getOperand(0);
      const auto type = cast<IntegerType>(value.getType());
      if (operation.getIndex() != 0) {
        const Value shift = arith::ConstantIntOp::create(
            builder, operation.getLoc(), operation.getIndex(), type.getWidth());
        value =
            arith::ShRUIOp::create(builder, operation.getLoc(), value, shift);
      }
      if (type.getWidth() != 1) {
        value = arith::TruncIOp::create(builder, operation.getLoc(),
                                        builder.getI1Type(), value);
      }
      operation.replaceAllUsesWith(value);
      operation.erase();
    }

    llvm::SmallVector<PackBitsOp> packOperations;
    getOperation().walk(
        [&](PackBitsOp operation) { packOperations.push_back(operation); });
    for (PackBitsOp operation : packOperations) {
      OpBuilder builder(operation);
      const unsigned width = operation.getResult().getType().getWidth();
      const auto type = IntegerType::get(operation.getContext(), width);
      Value packed =
          arith::ConstantIntOp::create(builder, operation.getLoc(), 0, width);
      for (const auto [index, bit] : llvm::enumerate(operation.getBits())) {
        Value extended = bit;
        if (width != 1) {
          extended =
              arith::ExtUIOp::create(builder, operation.getLoc(), type, bit);
        }
        if (index != 0) {
          const Value shift = arith::ConstantIntOp::create(
              builder, operation.getLoc(), index, width);
          extended = arith::ShLIOp::create(builder, operation.getLoc(),
                                           extended, shift);
        }
        packed =
            arith::OrIOp::create(builder, operation.getLoc(), packed, extended);
      }
      operation.replaceAllUsesWith(packed);
      operation.erase();
    }
  }

  static LogicalResult lowerInclusiveRange(ForOp loop) {
    auto sourceType = cast<IntegerType>(loop.getStart().getType());
    if (sourceType.getWidth() == IntegerType::kMaxWidth) {
      return loop.emitError(
          "range induction width cannot be widened without exceeding MLIR's "
          "integer-width limit");
    }
    auto step = loop.getStep().getDefiningOp<arith::ConstantIntOp>();
    if (!step) {
      return loop.emitError("dynamic range step cannot be proven nonzero for "
                            "the selected target");
    }

    OpBuilder builder(loop);
    const Location loc = loop.getLoc();
    const auto wideType =
        IntegerType::get(loop.getContext(), sourceType.getWidth() + 1,
                         sourceType.getSignedness());
    auto extend = [&](const Value value) -> Value {
      if (sourceType.isUnsigned()) {
        return arith::ExtUIOp::create(builder, loc, wideType, value);
      }
      return arith::ExtSIOp::create(builder, loc, wideType, value);
    };
    const Value start = extend(loop.getStart());
    const Value stop = extend(loop.getStop());
    const Value wideStep = extend(loop.getStep());

    auto whileOp = scf::WhileOp::create(builder, loc, TypeRange{wideType},
                                        ValueRange{start});
    Block& conditionBlock = whileOp.getBefore().emplaceBlock();
    conditionBlock.addArgument(wideType, loc);
    builder.setInsertionPointToStart(&conditionBlock);
    const bool descending = step.value() < 0;
    const arith::CmpIPredicate predicate =
        sourceType.isUnsigned() ? arith::CmpIPredicate::ule
                                : (descending ? arith::CmpIPredicate::sge
                                              : arith::CmpIPredicate::sle);
    const Value condition = arith::CmpIOp::create(
        builder, loc, predicate, conditionBlock.getArgument(0), stop);
    scf::ConditionOp::create(builder, loc, condition,
                             conditionBlock.getArguments());

    Block& bodyBlock = whileOp.getAfter().emplaceBlock();
    bodyBlock.addArgument(wideType, loc);
    builder.setInsertionPointToStart(&bodyBlock);
    const Value visibleInduction = arith::TruncIOp::create(
        builder, loc, sourceType, bodyBlock.getArgument(0));
    IRMapping mapping;
    mapping.map(loop.getBody().front().getArgument(0), visibleInduction);
    for (Operation& operation : loop.getBody().front().without_terminator()) {
      builder.clone(operation, mapping);
    }
    const Value next =
        arith::AddIOp::create(builder, loc, bodyBlock.getArgument(0), wideStep);
    scf::YieldOp::create(builder, loc, next);
    loop.erase();
    return success();
  }

  static LogicalResult emitPrimitive(OpBuilder& builder, const Location loc,
                                     const StringRef name,
                                     const ValueRange parameters,
                                     const ValueRange qubits) {
    const StringRef operationName =
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
                                 Operation* declaration,
                                 const ValueRange parameters,
                                 const ValueRange qubits) const {
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

    const StringRef resolvedName = application.getCallee();
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
    const StringRef primitive = catalogEntry->primitive;
    const auto emitCatalogPrimitive = [&](const ValueRange primitiveQubits) {
      if (!catalogEntry->inverse) {
        return emitPrimitive(builder, application.getLoc(), primitive,
                             parameters, primitiveQubits);
      }
      LogicalResult result = success();
      qc::InvOp::create(builder, application.getLoc(), primitiveQubits,
                        [&](const ValueRange aliases) {
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

    const ValueRange controlValues = qubits.take_front(controls);
    const ValueRange targets = qubits.drop_front(controls);
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
        [&](const ValueRange aliases) {
          if (catalogEntry->inverse) {
            qc::InvOp::create(builder, application.getLoc(), aliases,
                              [&](const ValueRange inverseAliases) {
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

  LogicalResult lowerGateApplication(ApplyGateOp application) const {
    Operation* declaration = SymbolTable::lookupNearestSymbolFrom(
        application.getOperation(), application.getCalleeAttr());
    if (declaration == nullptr) {
      return application.emitError("cannot lower an unresolved gate symbol");
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
        const auto controls =
            application.getQubits().slice(controlOffset, controlCount);
        negativeControls.append(controls.begin(), controls.end());
      }
      controlOffset += controlCount;
    }

    OpBuilder builder(application);
    for (const Value control : negativeControls) {
      qc::XOp::create(builder, application.getLoc(), control);
    }
    const LogicalResult result =
        emitModifiers(builder, application, declaration, controlCounts, 0,
                      application.getQubits());
    for (const Value control : negativeControls) {
      qc::XOp::create(builder, application.getLoc(), control);
    }
    if (failed(result)) {
      return failure();
    }
    application.erase();
    return success();
  }

  LogicalResult emitModifiers(OpBuilder& builder, ApplyGateOp application,
                              Operation* declaration,
                              const ArrayRef<int64_t> controlCounts,
                              const size_t position,
                              const ValueRange qubits) const {
    if (position == application.getModifierKinds().size()) {
      return emitResolvedGate(builder, application, declaration,
                              application.getParameters(), qubits);
    }
    const auto kind =
        static_cast<GateModifierKind>(application.getModifierKinds()[position]);
    if (kind == GateModifierKind::inv) {
      LogicalResult result = success();
      qc::InvOp::create(
          builder, application.getLoc(), qubits, [&](const ValueRange aliases) {
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
                             const ArrayRef<int64_t> controlCounts,
                             const size_t nextPosition,
                             const size_t remainingControls,
                             const ValueRange qubits) const {
    if (remainingControls == 0) {
      return emitModifiers(builder, application, declaration, controlCounts,
                           nextPosition, qubits);
    }

    LogicalResult result = success();
    qc::CtrlOp::create(builder, application.getLoc(), qubits.take_front(1),
                       qubits.drop_front(1), [&](const ValueRange aliases) {
                         result = emitControls(
                             builder, application, declaration, controlCounts,
                             nextPosition, remainingControls - 1, aliases);
                       });
    return result;
  }
};

} // namespace

std::unique_ptr<Pass>
createLowerOQ3ToQCPass(const OpenQASMLoweringOptions options) {
  return std::make_unique<LowerOQ3ToQCPass>(options);
}

} // namespace mlir::oq3

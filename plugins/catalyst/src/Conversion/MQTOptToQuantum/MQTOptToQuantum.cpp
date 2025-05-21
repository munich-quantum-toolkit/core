/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/MQTOptToQuantum/MQTOptToQuantum.h"

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

#include <Quantum/IR/QuantumDialect.h>
#include <Quantum/IR/QuantumOps.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::mqt::ir::conversions {

#define GEN_PASS_DEF_MQTOPTTOCATALYSTQUANTUM
#include "mlir/Conversion/MQTOptToQuantum/MQTOptToQuantum.h.inc"

using namespace mlir;

class MQTOptToQuantumTypeConverter : public TypeConverter {
public:
  explicit MQTOptToQuantumTypeConverter(MLIRContext* ctx) {
    // Identity conversion: Allow all types to pass through unmodified if
    // needed.
    addConversion([](Type type) { return type; });

    // Convert source QubitRegisterType to target QuregType
    addConversion([ctx](::mqt::ir::opt::QubitRegisterType /*type*/) -> Type {
      return catalyst::quantum::QuregType::get(ctx);
    });

    // Convert source QubitType to target QubitType
    addConversion([ctx](::mqt::ir::opt::QubitType /*type*/) -> Type {
      return catalyst::quantum::QubitType::get(ctx);
    });
  }
};

struct ConvertMQTOptAlloc
    : public OpConversionPattern<::mqt::ir::opt::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Prepare the result type(s)
    auto resultType = catalyst::quantum::QuregType::get(rewriter.getContext());

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::AllocOp>(
        op.getLoc(), resultType, adaptor.getSize(), adaptor.getSizeAttrAttr());

    // Get the result of the new operation, which represents the qubit register
    auto trgtQreg = catalystOp->getResult(0);

    // Collect the users of the original operation to update their operands
    std::vector<mlir::Operation*> users(op->getUsers().begin(),
                                        op->getUsers().end());

    // Iterate over the users in reverse order
    for (auto* user : llvm::reverse(users)) {
      // Registers should only be used in Extract, Insert or Dealloc operations
      if (mlir::isa<::mqt::ir::opt::ExtractOp>(user) ||
          mlir::isa<::mqt::ir::opt::InsertOp>(user) ||
          mlir::isa<::mqt::ir::opt::DeallocOp>(user)) {
        // Update the operand of the user operation to the new qubit register
        user->setOperand(0, trgtQreg);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertMQTOptDealloc
    : public OpConversionPattern<::mqt::ir::opt::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::DeallocOp>(
        op.getLoc(), ::mlir::TypeRange({}), adaptor.getQreg());

    // Replace the original with the new operation
    rewriter.replaceOp(op, catalystOp);
    return success();
  }
};

struct ConvertMQTOptMeasure
    : public OpConversionPattern<::mqt::ir::opt::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s)
    auto inQubit = adaptor.getInQubits()[0];

    // Prepare the result type(s)
    auto qubitType = catalyst::quantum::QubitType::get(rewriter.getContext());
    auto bitType = rewriter.getI1Type();

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::MeasureOp>(
        op.getLoc(), bitType, qubitType, inQubit,
        /*optional::mlir::IntegerAttr postselect=*/nullptr);

    // Because the results (bit and qubit) have change order, we need to
    // manually update their uses
    auto mqtQubit = op->getResult(0);
    auto catalystQubit = catalystOp->getResult(1);

    auto mqtMeasure = op->getResult(1);
    auto catalystMeasure = catalystOp->getResult(0);

    // Collect the users of the original input qubit register to update their
    // operands
    std::vector<mlir::Operation*> qubitUsers(mqtQubit.getUsers().begin(),
                                             mqtQubit.getUsers().end());

    // Iterate over users in reverse order to update their operands properly
    for (auto* user : llvm::reverse(qubitUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(catalystOp) && user != catalystOp &&
          user != op) {
        // Update operands in the user operation {
        user->replaceUsesOfWith(mqtQubit, catalystQubit);
      }
    }

    std::vector<mlir::Operation*> measureUsers(mqtMeasure.getUsers().begin(),
                                               mqtMeasure.getUsers().end());

    // Iterate over users in reverse order to update their operands properly
    for (auto* user : llvm::reverse(measureUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(catalystOp) && user != catalystOp &&
          user != op) {
        // Update operands in the user operation {
        user->replaceUsesOfWith(mqtMeasure, catalystMeasure);
      }
    }

    // Erase the old operation
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertMQTOptExtract
    : public OpConversionPattern<::mqt::ir::opt::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Prepare the result type(s)
    auto resultType = catalyst::quantum::QubitType::get(rewriter.getContext());

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::ExtractOp>(
        op.getLoc(), resultType, adaptor.getInQreg(), adaptor.getIndex(),
        adaptor.getIndexAttrAttr());

    auto mqtQreg = op->getResult(0);
    auto catalystQreg = catalystOp.getOperand(0);

    // Collect the users of the original input qubit register to update their
    // operands
    std::vector<mlir::Operation*> users(mqtQreg.getUsers().begin(),
                                        mqtQreg.getUsers().end());

    // Iterate over users in reverse order to update their operands properly
    for (auto* user : llvm::reverse(users)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(catalystOp) && user != catalystOp &&
          user != op) {
        // Update operands in the user operation
        if (mlir::isa<::mqt::ir::opt::ExtractOp>(user) ||
            mlir::isa<::mqt::ir::opt::InsertOp>(user) ||
            mlir::isa<::mqt::ir::opt::DeallocOp>(user)) {
          user->setOperand(0, catalystQreg);
        }
      }
    }

    // Collect the users of the original output qubit
    auto oldQubit = op->getResult(1);
    auto newQubit = catalystOp->getResult(0);

    std::vector<mlir::Operation*> qubitUsers(oldQubit.getUsers().begin(),
                                             oldQubit.getUsers().end());

    // Iterate over qubit users in reverse order
    for (auto* user : llvm::reverse(qubitUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(catalystOp) && user != catalystOp &&
          user != op) {

        auto operandIdx = 0;
        for (auto operand : user->getOperands()) {
          if (operand == oldQubit) {
            user->setOperand(operandIdx, newQubit);
          }
          operandIdx++;
        }
      }
    }

    // Erase the old operation
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertMQTOptInsert
    : public OpConversionPattern<::mqt::ir::opt::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s) and attribute(s)
    auto inQregValue = adaptor.getInQreg();
    auto qubitValue = adaptor.getInQubit();
    auto idxValue = adaptor.getIndex();
    auto idxIntegerAttr = adaptor.getIndexAttrAttr();

    // Prepare the result type(s)
    auto resultType = catalyst::quantum::QuregType::get(rewriter.getContext());

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::InsertOp>(
        op.getLoc(), resultType, inQregValue, idxValue, idxIntegerAttr,
        qubitValue);

    // Replace the original with the new operation
    rewriter.replaceOp(op, catalystOp);
    return success();
  }
};

template <typename MQTGateOp>
struct ConvertMQTOptSimpleGate : public OpConversionPattern<MQTGateOp> {
  using OpConversionPattern<MQTGateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateOp op, typename MQTGateOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto inQubitsValues = adaptor.getInQubits(); // excl. controls
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    llvm::SmallVector<mlir::Value> inCtrlQubits;
    inCtrlQubits.append(posCtrlQubitsValues.begin(), posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type
    mlir::Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // Merge inQubitsValues and inCtrlQubits to form the full qubit list
    auto allQubitsValues = llvm::SmallVector<mlir::Value>();
    allQubitsValues.append(inQubitsValues.begin(), inQubitsValues.end());
    allQubitsValues.append(inCtrlQubits.begin(), inCtrlQubits.end());
    auto inQubits = mlir::ValueRange(allQubitsValues);

    // Determine gate name depending on control count
    llvm::StringRef gateName = getGateName(inCtrlQubits.size());
    if (gateName.empty()) {
      llvm::errs() << "Unsupported controlled gate for op: " << op->getName()
                   << "\n";
      return failure();
    }

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*out_qubits=*/outQubitTypes,
        /*out_ctrl_qubits=*/mlir::TypeRange({}),
        /*params=*/adaptor.getParams(),
        /*in_qubits=*/inQubits,
        /*gate_name=*/gateName,
        /*adjoint=*/nullptr,
        /*in_ctrl_qubits=*/mlir::ValueRange({}),
        /*in_ctrl_values=*/mlir::ValueRange());

    // Replace the original with the new operation
    rewriter.replaceOp(op, catalystOp);
    return success();
  }

private:
  // Is specialized for each gate type
  static llvm::StringRef getGateName(std::size_t numControls);
};

// -- XOp (PauliX, CNOT, Toffoli)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::XOp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "PauliX";
  if (numControls == 1)
    return "CNOT";
  if (numControls == 2)
    return "Toffoli";
  return "";
}

// -- YOp (PauliY, CY)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::YOp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "PauliY";
  if (numControls == 1)
    return "CY";
  return "";
}

// -- ZOp (PauliZ, CZ)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::ZOp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "PauliZ";
  if (numControls == 1)
    return "CZ";
  return "";
}

// -- HOp (Hadamard)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::HOp>::getGateName(
    std::size_t numControls) {
  return "Hadamard";
}

// -- SWAPOp (SWAP)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::SWAPOp>::getGateName(
    std::size_t numControls) {
  return "SWAP";
}

// -- RXOp (RX, CRX)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::RXOp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "RX";
  if (numControls == 1)
    return "CRX";
  return "";
}

// -- RYOp (RY, CRY)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::RYOp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "RY";
  if (numControls == 1)
    return "CRY";
  return "";
}

// -- RZOp (RZ, CRZ)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::RZOp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "RZ";
  if (numControls == 1)
    return "CRZ";
  return "";
}

// -- POp (PhaseShift, ControlledPhaseShift)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::POp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "PhaseShift";
  if (numControls == 1)
    return "ControlledPhaseShift";
  return "";
}

struct MQTOptToQuantum : impl::MQTOptToQuantumBase<MQTOptToQuantum> {
  using MQTOptToQuantumBase::MQTOptToQuantumBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<catalyst::quantum::QuantumDialect>();
    target.addIllegalDialect<::mqt::ir::opt::MQTOptDialect>();

    // Mark operations legal, that have no equivalent in the target dialect
    target.addLegalOp<
        ::mqt::ir::opt::IOp, ::mqt::ir::opt::GPhaseOp,
        ::mqt::ir::opt::BarrierOp, ::mqt::ir::opt::SOp, ::mqt::ir::opt::SdgOp,
        ::mqt::ir::opt::TOp, ::mqt::ir::opt::TdgOp, ::mqt::ir::opt::VOp,
        ::mqt::ir::opt::VdgOp, ::mqt::ir::opt::UOp, ::mqt::ir::opt::U2Op,
        ::mqt::ir::opt::SXOp, ::mqt::ir::opt::SXdgOp, ::mqt::ir::opt::iSWAPOp,
        ::mqt::ir::opt::iSWAPdgOp, ::mqt::ir::opt::PeresOp,
        ::mqt::ir::opt::PeresdgOp, ::mqt::ir::opt::DCXOp, ::mqt::ir::opt::ECROp,
        ::mqt::ir::opt::RXXOp, ::mqt::ir::opt::RYYOp, ::mqt::ir::opt::RZZOp,
        ::mqt::ir::opt::RZXOp, ::mqt::ir::opt::XXminusYY,
        ::mqt::ir::opt::XXplusYY>();

    RewritePatternSet patterns(context);
    MQTOptToQuantumTypeConverter typeConverter(context);

    patterns.add<ConvertMQTOptAlloc, ConvertMQTOptDealloc, ConvertMQTOptExtract,
                 ConvertMQTOptMeasure, ConvertMQTOptInsert>(typeConverter,
                                                            context);

    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::XOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::YOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::ZOp>>(typeConverter,
                                                               context);

    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::RXOp>>(typeConverter,
                                                                context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::RYOp>>(typeConverter,
                                                                context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::RZOp>>(typeConverter,
                                                                context);

    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::HOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::SWAPOp>>(typeConverter,
                                                                  context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::POp>>(typeConverter,
                                                               context);

    // Boilerplate code to prevent "unresolved materialization" errors when the
    // IR contains ops with signature or operand/result types not yet rewritten:
    // https://www.jeremykun.com/2023/10/23/mlir-dialect-conversion

    // Rewrites func.func signatures to use the converted types.
    // Needed so that the converted argument/result types match expectations
    // in callers, bodies, and return ops.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    // Mark func.func as dynamically legal if:
    // - the signature types are legal under the type converter
    // - all block arguments in the function body are type-converted
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    // Converts return ops (func.return) to match the new function result types.
    // Required when the function result types are changed by the converter.
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    // Legal only if the return operand types match the converted function
    // result types.
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

    // Rewrites call sites (func.call) to use the converted argument and result
    // types. Needed so that calls into rewritten functions pass/receive correct
    // types.
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    // Legal only if operand/result types are all type-converted correctly.
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    // Rewrites control-flow ops like cf.br, cf.cond_br, etc.
    // Ensures block argument types are consistent after conversion.
    // Required for any dialects or passes that use CFG-based control flow.
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    // Fallback: mark any unhandled op as dynamically legal if:
    // - it's not a return or branch-like op (i.e., doesn't require special
    // handling), or
    // - it passes the legality checks for branch ops or return ops
    // This is crucial to avoid blocking conversion for unknown ops that don't
    // require specific operand type handling.
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::mqt::ir::conversions

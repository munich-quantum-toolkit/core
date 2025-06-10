/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/Catalyst/MQTOptToCatalystQuantum/MQTOptToCatalystQuantum.h"

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
#include "mlir/Conversion/Catalyst/MQTOptToCatalystQuantum/MQTOptToCatalystQuantum.h.inc"

using namespace mlir;

class MQTOptToCatalystQuantumTypeConverter : public TypeConverter {
public:
  explicit MQTOptToCatalystQuantumTypeConverter(MLIRContext* ctx) {
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
    // BarrierOp has no semantic effect on the circuit. Therefore, we erase it.
    if (std::is_same_v<MQTGateOp, ::mqt::ir::opt::BarrierOp>) {
      rewriter.eraseOp(op);
      return success();
    }

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

template <typename MQTGateOp>
struct ConvertMQTOptAdjointGate : public OpConversionPattern<MQTGateOp> {
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

    // Get the base gate name and whether it's an adjoint version
    std::pair<llvm::StringRef, bool> gateInfo = getGateInfo<MQTGateOp>();

    // Create the gate
    auto catalystOp = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*out_qubits=*/outQubitTypes,
        /*out_ctrl_qubits=*/mlir::TypeRange{},
        /*params=*/adaptor.getParams(),
        /*in_qubits=*/inQubits,
        /*gate_name=*/gateInfo.first,
        /*adjoint=*/gateInfo.second ? rewriter.getBoolAttr(true) : nullptr,
        /*in_ctrl_qubits=*/ValueRange{},
        /*in_ctrl_values=*/ValueRange{});

    rewriter.replaceOp(op, catalystOp);
    return success();
  }

private:
  template <typename T> static std::pair<llvm::StringRef, bool> getGateInfo() {
    if constexpr (std::is_same_v<T, ::mqt::ir::opt::SdgOp>)
      return {"S", true};
    else if constexpr (std::is_same_v<T, ::mqt::ir::opt::TdgOp>)
      return {"T", true};
    else if constexpr (std::is_same_v<T, ::mqt::ir::opt::iSWAPdgOp>)
      return {"ISWAP", true};
    // Default case
    return {"", false};
  }
};

// Conversions of unsupported gates which need decomposition
template <>
struct ConvertMQTOptSimpleGate<::mqt::ir::opt::VOp>
    : public OpConversionPattern<::mqt::ir::opt::VOp> {
  using OpConversionPattern<::mqt::ir::opt::VOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::VOp op,
                  typename ::mqt::ir::opt::VOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    llvm::SmallVector<mlir::Value> inCtrlQubits;
    inCtrlQubits.append(posCtrlQubitsValues.begin(), posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    mlir::Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // V = RZ(π/2) RY(π/2) RZ(-π/2)
    auto pi_2 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(M_PI_2));
    auto neg_pi_2 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(-M_PI_2));

    // Create the decomposed operations
    auto rz1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{pi_2},
        inQubitsValues, "RZ", nullptr, inCtrlQubits, ValueRange{});

    auto ry = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{pi_2},
        rz1.getResults(), "RY", nullptr, inCtrlQubits, ValueRange{});

    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{neg_pi_2},
        ry.getResults(), "RZ", nullptr, inCtrlQubits, ValueRange{});

    // Replace the original operation with the decomposition
    rewriter.replaceOp(op, rz2.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<::mqt::ir::opt::VdgOp>
    : public OpConversionPattern<::mqt::ir::opt::VdgOp> {
  using OpConversionPattern<::mqt::ir::opt::VdgOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::VdgOp op,
                  typename ::mqt::ir::opt::VdgOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    llvm::SmallVector<mlir::Value> inCtrlQubits;
    inCtrlQubits.append(posCtrlQubitsValues.begin(), posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    mlir::Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // V = RZ(π/2) RY(-π/2) RZ(-π/2)
    auto pi_2 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(M_PI_2));
    auto neg_pi_2 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(-M_PI_2));

    // Create the decomposed operations
    auto rz1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{pi_2},
        inQubitsValues, "RZ", nullptr, inCtrlQubits, ValueRange{});

    auto ry = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{neg_pi_2},
        rz1.getResults(), "RY", nullptr, inCtrlQubits, ValueRange{});

    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{neg_pi_2},
        ry.getResults(), "RZ", nullptr, inCtrlQubits, ValueRange{});

    // Replace the original operation with the decomposition
    rewriter.replaceOp(op, rz2.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<::mqt::ir::opt::DCXOp>
    : public OpConversionPattern<::mqt::ir::opt::DCXOp> {
  using OpConversionPattern<::mqt::ir::opt::DCXOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::DCXOp op,
                  typename ::mqt::ir::opt::DCXOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    llvm::SmallVector<mlir::Value> inCtrlQubits;
    inCtrlQubits.append(posCtrlQubitsValues.begin(), posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    mlir::Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // DCX = CNOT(q2,q1) CNOT(q1,q2)
    auto cnot1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{},
        inQubitsValues, "CNOT", nullptr, inCtrlQubits, ValueRange{});

    auto cnot2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{},
        cnot1.getResults(), "CNOT", nullptr, inCtrlQubits, ValueRange{});

    rewriter.replaceOp(op, cnot2.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<::mqt::ir::opt::RZXOp>
    : public OpConversionPattern<::mqt::ir::opt::RZXOp> {
  using OpConversionPattern<::mqt::ir::opt::RZXOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::RZXOp op,
                  typename ::mqt::ir::opt::RZXOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();
    auto theta = adaptor.getParams()[0];

    llvm::SmallVector<mlir::Value> inCtrlQubits;
    inCtrlQubits.append(posCtrlQubitsValues.begin(), posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    mlir::Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // RZX(θ) = H(q2) CNOT(q1,q2) RZ(θ)(q2) CNOT(q1,q2) H(q2)
    auto h1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{},
        inQubitsValues, "H", nullptr, inCtrlQubits, ValueRange{});

    auto cnot1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{},
        h1.getResults(), "CNOT", nullptr, inCtrlQubits, ValueRange{});

    auto rz = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{theta},
        cnot1.getResults(), "RZ", nullptr, inCtrlQubits, ValueRange{});

    auto cnot2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{},
        rz.getResults(), "CNOT", nullptr, inCtrlQubits, ValueRange{});

    auto h2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{},
        cnot2.getResults(), "H", nullptr, inCtrlQubits, ValueRange{});

    rewriter.replaceOp(op, h2.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<::mqt::ir::opt::XXminusYY>
    : public OpConversionPattern<::mqt::ir::opt::XXminusYY> {
  using OpConversionPattern<::mqt::ir::opt::XXminusYY>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::XXminusYY op,
                  typename ::mqt::ir::opt::XXminusYY::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();
    auto theta = adaptor.getParams()[0];
    auto beta = adaptor.getParams()[1];

    llvm::SmallVector<mlir::Value> inCtrlQubits;
    inCtrlQubits.append(posCtrlQubitsValues.begin(), posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    mlir::Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // XXminusYY(θ,β) = RX(π/2)(q1) RY(π/2)(q2) CNOT(q1,q2) RZ(θ)(q2)
    // CNOT(q1,q2) RZ(β)(q1) RZ(β)(q2) RX(-π/2)(q1) RY(-π/2)(q2)
    auto pi_2 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(M_PI_2));
    auto neg_pi_2 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(-M_PI_2));

    auto rx1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{pi_2},
        inQubitsValues, "RX", nullptr, inCtrlQubits, ValueRange{});

    auto ry1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{pi_2},
        rx1.getResults(), "RY", nullptr, inCtrlQubits, ValueRange{});

    auto cnot1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{},
        ry1.getResults(), "CNOT", nullptr, inCtrlQubits, ValueRange{});

    auto rz1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{theta},
        cnot1.getResults(), "RZ", nullptr, inCtrlQubits, ValueRange{});

    auto cnot2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{},
        rz1.getResults(), "CNOT", nullptr, inCtrlQubits, ValueRange{});

    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{beta},
        cnot2.getResults(), "RZ", nullptr, inCtrlQubits, ValueRange{});

    auto rz3 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{beta},
        rz2.getResults(), "RZ", nullptr, inCtrlQubits, ValueRange{});

    auto rx2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{neg_pi_2},
        rz3.getResults(), "RX", nullptr, inCtrlQubits, ValueRange{});

    auto ry2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{neg_pi_2},
        rx2.getResults(), "RY", nullptr, inCtrlQubits, ValueRange{});

    rewriter.replaceOp(op, ry2.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<::mqt::ir::opt::UOp>
    : public OpConversionPattern<::mqt::ir::opt::UOp> {
  using OpConversionPattern<::mqt::ir::opt::UOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::UOp op,
                  typename ::mqt::ir::opt::UOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    // Extract parameters
    llvm::SmallVector<mlir::Value> paramValues;
    auto dynamicParams = adaptor.getParams();
    auto staticParams = op.getStaticParams();
    auto paramMask = op.getParamsMask();

    // There must be exactly 3 parameters
    constexpr size_t numParams = 3;
    for (size_t i = 0, dynIdx = 0, statIdx = 0; i < numParams; ++i) {
      if (paramMask.has_value()) {
        if ((*paramMask)[i]) {
          // Static parameter
          auto attr = (*staticParams)[statIdx++];
          auto constOp = rewriter.create<arith::ConstantOp>(
              op.getLoc(), attr.cast<mlir::FloatAttr>());
          paramValues.push_back(constOp);
        } else {
          // Dynamic parameter
          paramValues.push_back(dynamicParams[dynIdx++]);
        }
      } else if (staticParams.has_value()) {
        // All static
        auto attr = (*staticParams)[i];
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), attr.cast<mlir::FloatAttr>());
        paramValues.push_back(constOp);
      } else {
        // All dynamic
        paramValues.push_back(dynamicParams[i]);
      }
    }
    // Now paramValues[0] = θ, [1] = φ, [2] = λ
    auto theta = paramValues[0];
    auto phi = paramValues[1];
    auto lambda = paramValues[2];

    llvm::SmallVector<mlir::Value> inCtrlQubits;
    inCtrlQubits.append(posCtrlQubitsValues.begin(), posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    mlir::Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // Based on
    // https://docs.quantum.ibm.com/api/qiskit/0.24/qiskit.circuit.library.UGate
    // U(θ, φ, λ) = RZ(φ − π⁄2) ⋅ RX(π⁄2) ⋅ RZ(π − θ) ⋅ RX(π⁄2) ⋅ RZ(λ − π⁄2)
    // Note: The MQT UOp uses U(θ/2, φ, λ)
    auto pi = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(M_PI));
    auto pi_2 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(M_PI_2));

    // Compute φ - π/2
    auto phiMinusPi2 = rewriter.create<arith::SubFOp>(op.getLoc(), phi, pi_2);
    // Compute π - θ/2
    auto piMinusTheta2 =
        rewriter.create<arith::SubFOp>(op.getLoc(), pi, theta / 2);
    // Compute λ - π/2
    auto lambdaMinusPi2 =
        rewriter.create<arith::SubFOp>(op.getLoc(), lambda, pi_2);

    // RZ(λ − π/2)
    auto rz1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{},
        ValueRange{lambdaMinusPi2}, inQubitsValues, "RZ", nullptr, inCtrlQubits,
        ValueRange{});

    // RX(π/2)
    auto rx1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{pi_2},
        rz1.getResults(), "RX", nullptr, inCtrlQubits, ValueRange{});

    // RZ(π − θ)
    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{},
        ValueRange{piMinusTheta2}, rx1.getResults(), "RZ", nullptr,
        inCtrlQubits, ValueRange{});

    // RX(π/2)
    auto rx2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{pi_2},
        rz2.getResults(), "RX", nullptr, inCtrlQubits, ValueRange{});

    // RZ(φ − π/2)
    auto rz3 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{phiMinusPi2},
        rx2.getResults(), "RZ", nullptr, inCtrlQubits, ValueRange{});

    // Replace the original U gate with the decomposed sequence
    rewriter.replaceOp(op, rz3.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<::mqt::ir::opt::U2Op>
    : public OpConversionPattern<::mqt::ir::opt::U2Op> {
  using OpConversionPattern<::mqt::ir::opt::U2Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::U2Op op,
                  typename ::mqt::ir::opt::U2Op::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    // Extract parameters
    llvm::SmallVector<mlir::Value> paramValues;
    auto dynamicParams = adaptor.getParams();
    auto staticParams = op.getStaticParams();
    auto paramMask = op.getParamsMask();

    // There must be exactly 2 parameters
    constexpr size_t numParams = 2;
    for (size_t i = 0, dynIdx = 0, statIdx = 0; i < numParams; ++i) {
      if (paramMask.has_value()) {
        if ((*paramMask)[i]) {
          // Static parameter
          auto attr = (*staticParams)[statIdx++];
          auto constOp = rewriter.create<arith::ConstantOp>(
              op.getLoc(), attr.cast<mlir::FloatAttr>());
          paramValues.push_back(constOp);
        } else {
          // Dynamic parameter
          paramValues.push_back(dynamicParams[dynIdx++]);
        }
      } else if (staticParams.has_value()) {
        // All static
        auto attr = (*staticParams)[i];
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), attr.cast<mlir::FloatAttr>());
        paramValues.push_back(constOp);
      } else {
        // All dynamic
        paramValues.push_back(dynamicParams[i]);
      }
    }
    // Now paramValues [0] = φ, [1] = λ
    auto phi = paramValues[0];
    auto lambda = paramValues[1];

    llvm::SmallVector<mlir::Value> inCtrlQubits;
    inCtrlQubits.append(posCtrlQubitsValues.begin(), posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    mlir::Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // U2(φ, λ) = U(π/2, φ, λ) = RZ(φ − π⁄2) ⋅ RX(π⁄2) ⋅ RZ(3/4 π) ⋅ RX(π⁄2) ⋅
    // RZ(λ − π⁄2)
    auto pi = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(M_PI));
    auto pi_2 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(M_PI_2));
    auto pi_4 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(M_PI_4));
    auto pi_3_4 = rewriter.create<arith::MulFOp>(op.getLoc(), pi_4,
                                                 rewriter.getF64FloatAttr(3.0));

    // Compute φ - π/2
    auto phiMinusPi2 = rewriter.create<arith::SubFOp>(op.getLoc(), phi, pi_2);
    // Compute λ - π/2
    auto lambdaMinusPi2 =
        rewriter.create<arith::SubFOp>(op.getLoc(), lambda, pi_2);

    // RZ(λ − π/2)
    auto rz1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{},
        ValueRange{lambdaMinusPi2}, inQubitsValues, "RZ", nullptr, inCtrlQubits,
        ValueRange{});

    // RX(π/2)
    auto rx1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{pi_2},
        rz1.getResults(), "RX", nullptr, inCtrlQubits, ValueRange{});

    // RZ(3/4 π)
    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{pi_3_4},
        rx1.getResults(), "RZ", nullptr, inCtrlQubits, ValueRange{});

    // RX(π/2)
    auto rx2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{pi_2},
        rz2.getResults(), "RX", nullptr, inCtrlQubits, ValueRange{});

    // RZ(φ − π/2)
    auto rz3 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{phiMinusPi2},
        rx2.getResults(), "RZ", nullptr, inCtrlQubits, ValueRange{});

    // Replace the original U gate with the decomposed sequence
    rewriter.replaceOp(op, rz3.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<::mqt::ir::opt::SXOp>
    : public OpConversionPattern<::mqt::ir::opt::SXOp> {
  using OpConversionPattern<::mqt::ir::opt::SXOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::SXOp op,
                  typename ::mqt::ir::opt::SXOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    llvm::SmallVector<mlir::Value> inCtrlQubits;
    inCtrlQubits.append(posCtrlQubitsValues.begin(), posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    mlir::Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // SX = H S H = H R(π/2) H
    auto pi_2 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(M_PI_2));

    // Create the decomposed operations
    auto h1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{},
        inQubitsValues, "Hadamard", nullptr, inCtrlQubits, ValueRange{});

    auto r = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{pi_2},
        h1.getResults(), "PhaseShift", nullptr, inCtrlQubits, ValueRange{});

    auto h2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{},
        r.getResults(), "Hadamard", nullptr, inCtrlQubits, ValueRange{});

    // Replace the original operation with the decomposition
    rewriter.replaceOp(op, h2.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<::mqt::ir::opt::SXdgOp>
    : public OpConversionPattern<::mqt::ir::opt::SXdgOp> {
  using OpConversionPattern<::mqt::ir::opt::SXdgOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::SXdgOp op,
                  typename ::mqt::ir::opt::SXdgOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    llvm::SmallVector<mlir::Value> inCtrlQubits;
    inCtrlQubits.append(posCtrlQubitsValues.begin(), posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    mlir::Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // SX_dagger = H S H = H R(-π/2) H
    auto neg_pi_2 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(-M_PI_2));

    // Create the decomposed operations
    auto h1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{},
        inQubitsValues, "Hadamard", nullptr, inCtrlQubits, ValueRange{});

    auto r = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{neg_pi_2},
        h1.getResults(), "PhaseShift", nullptr, inCtrlQubits, ValueRange{});

    auto h2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{},
        r.getResults(), "Hadamard", nullptr, inCtrlQubits, ValueRange{});

    // Replace the original operation with the decomposition
    rewriter.replaceOp(op, h2.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<::mqt::ir::opt::ECROp>
    : public OpConversionPattern<::mqt::ir::opt::ECROp> {
  using OpConversionPattern<::mqt::ir::opt::ECROp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::ECROp op,
                  typename ::mqt::ir::opt::ECROp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operands
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    llvm::SmallVector<mlir::Value> inCtrlQubits;
    inCtrlQubits.append(posCtrlQubitsValues.begin(), posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    mlir::Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // Constants
    auto pi = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(llvm::numbers::pi));
    auto pi_4 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(llvm::numbers::pi / 4));
    auto neg_pi_4 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(-llvm::numbers::pi / 4));

    // RZX(π/4)
    auto rzx1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{pi_4},
        inQubitsValues, "RZX", nullptr, inCtrlQubits, ValueRange{});

    // RX(π) on the first qubit (q_0)
    mlir::Value rxQubit = rzx1.getResult(0); // q_0 after RZX
    auto rx = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{pi},
        ValueRange{rxQubit}, "RX", nullptr, inCtrlQubits, ValueRange{});

    // Combine updated q_0 with unchanged q_1
    llvm::SmallVector<mlir::Value> rzx2Inputs{rx.getResult(0),
                                              rzx1.getResult(1)};

    // RZX(-π/4)
    auto rzx2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, mlir::TypeRange{}, ValueRange{neg_pi_4},
        rzx2Inputs, "RZX", nullptr, inCtrlQubits, ValueRange{});

    // Replace the original ECR with the result of final RZX
    rewriter.replaceOp(op, rzx2.getResults());
    return success();
  }
};

// -- GPhaseOp (gphase)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::GPhaseOp>::getGateName(
    std::size_t numControls) {
  return "gphase";
}

// -- IOp (Identity)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::IOp>::getGateName(
    std::size_t numControls) {
  return "Identity";
}

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

// -- SOP (S)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::SOp>::getGateName(
    std::size_t numControls) {
  return "S";
}

// -- TOP (T)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::TOp>::getGateName(
    std::size_t numControls) {
  return "T";
}

// -- SWAPOp (SWAP)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::SWAPOp>::getGateName(
    std::size_t numControls) {
  return "SWAP";
}

// -- iSWAPOp (iSWAP)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::iSWAPOp>::getGateName(
    std::size_t numControls) {
  return "ISWAP";
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

// -- RXXOp (IsingXX)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::RXXOp>::getGateName(
    std::size_t numControls) {
  return "IsingXX";
}

// -- RYYOp (IsingYY)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::RYYOp>::getGateName(
    std::size_t numControls) {
  return "IsingYY";
}

// -- RZZ (IsingZZ)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::RZZOp>::getGateName(
    std::size_t numControls) {
  return "IsingZZ";
}

// -- XXplusYY (IsingXY)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::XXplusYY>::getGateName(
    std::size_t numControls) {
  return "IsingXY";
}

struct MQTOptToCatalystQuantum
    : impl::MQTOptToCatalystQuantumBase<MQTOptToCatalystQuantum> {
  using MQTOptToCatalystQuantumBase::MQTOptToCatalystQuantumBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<catalyst::quantum::QuantumDialect>();
    target.addIllegalDialect<::mqt::ir::opt::MQTOptDialect>();

    // Mark operations legal, that have no equivalent in the target dialect
    target.addLegalOp<::mqt::ir::opt::PeresOp, ::mqt::ir::opt::PeresdgOp>();

    RewritePatternSet patterns(context);
    MQTOptToCatalystQuantumTypeConverter typeConverter(context);

    patterns.add<ConvertMQTOptAlloc, ConvertMQTOptDealloc, ConvertMQTOptExtract,
                 ConvertMQTOptMeasure, ConvertMQTOptInsert>(typeConverter,
                                                            context);

    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::BarrierOp>>(
        typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::GPhaseOp>>(
        typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::IOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::XOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::YOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::ZOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::SOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::TOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::VOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::VdgOp>>(typeConverter,
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
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::iSWAPOp>>(
        typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::POp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::DCXOp>>(typeConverter,
                                                                 context);

    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::RXXOp>>(typeConverter,
                                                                 context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::RYYOp>>(typeConverter,
                                                                 context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::RZZOp>>(typeConverter,
                                                                 context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::RZXOp>>(typeConverter,
                                                                 context);

    patterns.add<ConvertMQTOptAdjointGate<::mqt::ir::opt::SdgOp>>(typeConverter,
                                                                  context);
    patterns.add<ConvertMQTOptAdjointGate<::mqt::ir::opt::TdgOp>>(typeConverter,
                                                                  context);
    patterns.add<ConvertMQTOptAdjointGate<::mqt::ir::opt::iSWAPdgOp>>(
        typeConverter, context);

    patterns.add<ConvertMQTOptAdjointGate<::mqt::ir::opt::XXminusYY>>(
        typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::XXplusYY>>(
        typeConverter, context);

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

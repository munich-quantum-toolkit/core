/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/MQTOptToCatalystQuantum/MQTOptToCatalystQuantum.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

#include <Quantum/IR/QuantumDialect.h>
#include <Quantum/IR/QuantumOps.h>
#include <cstddef>
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
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mqt::ir::conversions {

#define GEN_PASS_DEF_MQTOPTTOCATALYSTQUANTUM
#include "mlir/Conversion/MQTOptToCatalystQuantum/MQTOptToCatalystQuantum.h.inc"

using namespace mlir;
using namespace mlir::arith;

class MQTOptToCatalystQuantumTypeConverter final : public TypeConverter {
public:
  explicit MQTOptToCatalystQuantumTypeConverter(MLIRContext* ctx) {
    // Identity conversion: Allow all types to pass through unmodified if
    // needed.
    addConversion([](const Type type) { return type; });

    // Convert source QubitRegisterType to target QuregType
    addConversion([ctx](opt::QubitRegisterType /*type*/) -> Type {
      return catalyst::quantum::QuregType::get(ctx);
    });

    // Convert source QubitType to target QubitType
    addConversion([ctx](opt::QubitType /*type*/) -> Type {
      return catalyst::quantum::QubitType::get(ctx);
    });
  }
};

struct ConvertMQTOptAlloc final : OpConversionPattern<opt::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Prepare the result type(s)
    const auto resultType =
        catalyst::quantum::QuregType::get(rewriter.getContext());

    // Create the new operation
    const auto catalystOp = rewriter.create<catalyst::quantum::AllocOp>(
        op.getLoc(), resultType, adaptor.getSize(), adaptor.getSizeAttrAttr());

    // Get the result of the new operation, which represents the qubit register
    const auto targetQreg = catalystOp->getResult(0);

    // Iterate over the users to update their operands
    for (auto* user : op->getUsers()) {
      // Registers should only be used in Extract, Insert or Dealloc operations
      assert(mlir::isa<opt::ExtractOp>(user) ||
             mlir::isa<opt::InsertOp>(user) || mlir::isa<opt::DeallocOp>(user));
      // Update the operand of the user operation to the new qubit register
      user->setOperand(0, targetQreg);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertMQTOptDealloc final : OpConversionPattern<opt::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Create the new operation
    const auto catalystOp = rewriter.create<catalyst::quantum::DeallocOp>(
        op.getLoc(), TypeRange({}), adaptor.getQreg());

    // Replace the original with the new operation
    rewriter.replaceOp(op, catalystOp);
    return success();
  }
};

struct ConvertMQTOptMeasure final : OpConversionPattern<opt::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s)
    auto inQubit = adaptor.getInQubits()[0];

    // Prepare the result type(s)
    auto qubitType = catalyst::quantum::QubitType::get(rewriter.getContext());
    auto bitType = rewriter.getI1Type();

    // Create the new operation
    const auto catalystOp = rewriter.create<catalyst::quantum::MeasureOp>(
        op.getLoc(), bitType, qubitType, inQubit,
        /*optional::mlir::IntegerAttr postselect=*/nullptr);

    // Replace all uses of both results and then erase the operation
    const auto catalystMeasure = catalystOp->getResult(0);
    const auto catalystQubit = catalystOp->getResult(1);
    rewriter.replaceOp(op, ValueRange{catalystQubit, catalystMeasure});
    return success();
  }
};

struct ConvertMQTOptExtract final : OpConversionPattern<opt::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Prepare the result type(s)
    auto resultType = catalyst::quantum::QubitType::get(rewriter.getContext());

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::ExtractOp>(
        op.getLoc(), resultType, adaptor.getInQreg(), adaptor.getIndex(),
        adaptor.getIndexAttrAttr());

    const auto mqtQreg = op->getResult(0);
    const auto catalystQreg = catalystOp.getOperand(0);

    // Collect the users of the original input qubit register to update their
    // operands and iterate over users to update their operands
    for (auto* user : mqtQreg.getUsers()) {
      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(catalystOp) && user != catalystOp &&
          user != op) {
        assert(mlir::isa<opt::ExtractOp>(user) ||
               mlir::isa<opt::InsertOp>(user) ||
               mlir::isa<opt::DeallocOp>(user));
        // Update operands in the user operation
        user->setOperand(0, catalystQreg);
      }
    }

    // Collect the users of the original output qubit
    const auto oldQubit = op->getResult(1);
    const auto newQubit = catalystOp->getResult(0);
    // Iterate over qubit users to update their operands
    for (auto* user : oldQubit.getUsers()) {
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

struct ConvertMQTOptInsert final : OpConversionPattern<opt::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s) and attribute(s)
    auto inQregValue = adaptor.getInQreg();
    auto qubitValue = adaptor.getInQubit();
    auto idxValue = adaptor.getIndex();
    auto idxIntegerAttr = adaptor.getIndexAttrAttr();

    // Prepare the result type(s)
    auto resultType = catalyst::quantum::QuregType::get(rewriter.getContext());

    // Create the new operation
    const auto catalystOp = rewriter.create<catalyst::quantum::InsertOp>(
        op.getLoc(), resultType, inQregValue, idxValue, idxIntegerAttr,
        qubitValue);

    // Replace the original with the new operation
    rewriter.replaceOp(op, catalystOp);
    return success();
  }
};

template <typename MQTGateOp>
struct ConvertMQTOptSimpleGate final : OpConversionPattern<MQTGateOp> {
  using OpConversionPattern<MQTGateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateOp op, typename MQTGateOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // BarrierOp has no semantic effect on the circuit. Therefore, we erase it.
    if (std::is_same_v<MQTGateOp, opt::BarrierOp>) {
      rewriter.eraseOp(op);
      return success();
    }

    // Extract operand(s) and attribute(s)
    auto inQubitsValues = adaptor.getInQubits(); // excl. controls
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    SmallVector<Value> inCtrlQubits(posCtrlQubitsValues.begin(),
                                    posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type
    const Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    const SmallVector<Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    const auto outQubitTypes = TypeRange(qubitTypes);

    // Merge inQubitsValues and inCtrlQubits to form the full qubit list
    auto allQubitsValues = inCtrlQubits;
    allQubitsValues.append(inQubitsValues.begin(), inQubitsValues.end());
    const auto inQubits = ValueRange(allQubitsValues);

    // Determine gate name depending on control count
    const StringRef gateName = getGateName(inCtrlQubits.size());
    if (gateName.empty()) {
      llvm::errs() << "Unsupported controlled gate for op: " << op->getName()
                   << "\n";
      return failure();
    }

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*out_qubits=*/outQubitTypes,
        /*out_ctrl_qubits=*/TypeRange({}),
        /*params=*/adaptor.getParams(),
        /*in_qubits=*/inQubits,
        /*gate_name=*/gateName,
        /*adjoint=*/false,
        /*in_ctrl_qubits=*/ValueRange({}),
        /*in_ctrl_values=*/ValueRange());

    // Replace the original with the new operation
    rewriter.replaceOp(op, catalystOp);
    return success();
  }

private:
  // Is specialized for each gate type
  static StringRef getGateName(std::size_t numControls);
};

template <typename MQTGateOp>
struct ConvertMQTOptAdjointGate final : OpConversionPattern<MQTGateOp> {
  using OpConversionPattern<MQTGateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateOp op, typename MQTGateOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto inQubitsValues = adaptor.getInQubits(); // excl. controls
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    SmallVector<Value> inCtrlQubits(posCtrlQubitsValues.begin(),
                                    posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type
    const Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    const SmallVector<Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    const auto outQubitTypes = TypeRange(qubitTypes);

    // Merge inQubitsValues and inCtrlQubits to form the full qubit list
    auto allQubitsValues =
        SmallVector<Value>(inCtrlQubits.begin(), inCtrlQubits.end());
    allQubitsValues.append(inQubitsValues.begin(), inQubitsValues.end());
    const auto inQubits = ValueRange(allQubitsValues);

    // Get the base gate name and whether it is an adjoint version
    const auto& [gateName, adjoint] = getGateInfo<MQTGateOp>();

    // Create the gate
    auto catalystOp = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*out_qubits=*/outQubitTypes,
        /*out_ctrl_qubits=*/TypeRange{},
        /*params=*/adaptor.getParams(),
        /*in_qubits=*/inQubits,
        /*gate_name=*/gateName,
        /*adjoint=*/adjoint,
        /*in_ctrl_qubits=*/ValueRange{},
        /*in_ctrl_values=*/ValueRange{});

    rewriter.replaceOp(op, catalystOp);
    return success();
  }

private:
  template <typename T> static std::pair<StringRef, bool> getGateInfo() {
    if constexpr (std::is_same_v<T, opt::SdgOp>) {
      return {"S", true};
    } else if constexpr (std::is_same_v<T, opt::TdgOp>) {
      return {"T", true};
    } else if constexpr (std::is_same_v<T, opt::iSWAPdgOp>) {
      return {"ISWAP", true};
    } else if constexpr (std::is_same_v<T, opt::SXdgOp>) {
      return {"SX", true};
    }
    // Default case
    return {"", false};
  }
};

// Conversions of unsupported gates, which need decomposition
template <>
struct ConvertMQTOptSimpleGate<opt::VOp> final : OpConversionPattern<opt::VOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::VOp op, opt::VOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto inQubitsValues = adaptor.getInQubits();
    const auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    const auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    SmallVector<Value> inCtrlQubits(posCtrlQubitsValues.begin(),
                                    posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    const Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    const SmallVector<Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = TypeRange(qubitTypes);

    // V = RZ(π/2) RY(π/2) RZ(-π/2)
    auto pi2 = rewriter.create<ConstantOp>(op.getLoc(),
                                           rewriter.getF64FloatAttr(M_PI_2));

    // Create the decomposed operations
    auto rz1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{pi2},
        inQubitsValues, "RZ", false, inCtrlQubits, ValueRange{});

    auto ry = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{pi2},
        rz1.getResults(), "RY", false, inCtrlQubits, ValueRange{});

    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{pi2},
        ry.getResults(), "RZ", true, inCtrlQubits, ValueRange{});

    // Replace the original operation with the decomposition
    rewriter.replaceOp(op, rz2.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<opt::VdgOp> final
    : OpConversionPattern<opt::VdgOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::VdgOp op, opt::VdgOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto inQubitsValues = adaptor.getInQubits();
    const auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    const auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    SmallVector<Value> inCtrlQubits(posCtrlQubitsValues.begin(),
                                    posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    const Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    const SmallVector<Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = TypeRange(qubitTypes);

    // V = RZ(π/2) RY(-π/2) RZ(-π/2)
    auto negPi2 = rewriter.create<ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(-M_PI_2));

    // Create the decomposed operations
    auto rz1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{negPi2},
        inQubitsValues, "RZ", true, inCtrlQubits, ValueRange{});

    auto ry = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{negPi2},
        rz1.getResults(), "RY", false, inCtrlQubits, ValueRange{});

    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{negPi2},
        ry.getResults(), "RZ", false, inCtrlQubits, ValueRange{});

    // Replace the original operation with the decomposition
    rewriter.replaceOp(op, rz2.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<opt::DCXOp> final
    : OpConversionPattern<opt::DCXOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::DCXOp op, opt::DCXOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto inQubitsValues = adaptor.getInQubits();
    const auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    const auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    SmallVector<Value> inCtrlQubits(posCtrlQubitsValues.begin(),
                                    posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    const Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    const SmallVector<Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = TypeRange(qubitTypes);

    // DCX = CNOT(q2,q1) CNOT(q1,q2)
    auto cnot1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{}, inQubitsValues,
        "CNOT", false, inCtrlQubits, ValueRange{});

    auto cnot2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{},
        cnot1.getResults(), "CNOT", false, inCtrlQubits, ValueRange{});

    rewriter.replaceOp(op, cnot2.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<opt::RZXOp> final
    : OpConversionPattern<opt::RZXOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::RZXOp op, opt::RZXOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto inQubitsValues = adaptor.getInQubits();
    const auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    const auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();
    const auto theta = adaptor.getParams()[0];

    SmallVector<Value> inCtrlQubits(posCtrlQubitsValues.begin(),
                                    posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    const Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    const SmallVector<Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = TypeRange(qubitTypes);

    // RZX(θ) = H(q2) CNOT(q1,q2) RZ(θ)(q2) CNOT(q1,q2) H(q2)
    auto h1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{}, inQubitsValues,
        "H", false, inCtrlQubits, ValueRange{});

    auto cnot1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{}, h1.getResults(),
        "CNOT", false, inCtrlQubits, ValueRange{});

    auto rz = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{theta},
        cnot1.getResults(), "RZ", false, inCtrlQubits, ValueRange{});

    auto cnot2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{}, rz.getResults(),
        "CNOT", false, inCtrlQubits, ValueRange{});

    auto h2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{},
        cnot2.getResults(), "H", false, inCtrlQubits, ValueRange{});

    rewriter.replaceOp(op, h2.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<opt::XXminusYY> final
    : OpConversionPattern<opt::XXminusYY> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::XXminusYY op, opt::XXminusYY::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();
    auto theta = adaptor.getParams()[0];
    auto beta = adaptor.getParams()[1];

    SmallVector<Value> inCtrlQubits(posCtrlQubitsValues.begin(),
                                    posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    const Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    const SmallVector<Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = TypeRange(qubitTypes);

    // XXminusYY(θ,β) = RX(π/2)(q1) RY(π/2)(q2) CNOT(q1,q2) RZ(θ)(q2)
    // CNOT(q1,q2) RZ(β)(q1) RZ(β)(q2) RX(-π/2)(q1) RY(-π/2)(q2)
    auto pi2 = rewriter.create<ConstantOp>(op.getLoc(),
                                           rewriter.getF64FloatAttr(M_PI_2));
    auto negPi2 = rewriter.create<ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(-M_PI_2));

    auto rx1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{pi2},
        inQubitsValues, "RX", false, inCtrlQubits, ValueRange{});

    auto ry1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{pi2},
        rx1.getResults(), "RY", false, inCtrlQubits, ValueRange{});

    auto cnot1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{}, ry1.getResults(),
        "CNOT", false, inCtrlQubits, ValueRange{});

    auto rz1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{theta},
        cnot1.getResults(), "RZ", false, inCtrlQubits, ValueRange{});

    auto cnot2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{}, rz1.getResults(),
        "CNOT", false, inCtrlQubits, ValueRange{});

    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{beta},
        cnot2.getResults(), "RZ", false, inCtrlQubits, ValueRange{});

    auto rz3 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{beta},
        rz2.getResults(), "RZ", false, inCtrlQubits, ValueRange{});

    auto rx2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{negPi2},
        rz3.getResults(), "RX", false, inCtrlQubits, ValueRange{});

    auto ry2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{negPi2},
        rx2.getResults(), "RY", false, inCtrlQubits, ValueRange{});

    rewriter.replaceOp(op, ry2.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<opt::UOp> final : OpConversionPattern<opt::UOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::UOp op, opt::UOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    // Extract parameters
    SmallVector<Value> paramValues;
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
          auto floatAttr = rewriter.getF64FloatAttr(attr);
          auto constOp = rewriter.create<ConstantOp>(op.getLoc(), floatAttr);
          paramValues.push_back(constOp);
        } else {
          // Dynamic parameter
          paramValues.push_back(dynamicParams[dynIdx++]);
        }
      } else if (staticParams.has_value()) {
        // All static
        auto attr = (*staticParams)[i];
        auto floatAttr = rewriter.getF64FloatAttr(attr);
        auto constOp = rewriter.create<ConstantOp>(op.getLoc(), floatAttr);
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

    SmallVector<Value> inCtrlQubits(posCtrlQubitsValues.begin(),
                                    posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    const Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    const SmallVector<Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = TypeRange(qubitTypes);

    // Based on
    // https://docs.quantum.ibm.com/api/qiskit/0.24/qiskit.circuit.library.UGate
    // U(θ, φ, λ) = RZ(φ − π⁄2) ⋅ RX(π⁄2) ⋅ RZ(π − θ) ⋅ RX(π⁄2) ⋅ RZ(λ − π⁄2)
    // Note: The MQT UOp uses U(θ/2, φ, λ)
    auto pi = rewriter.create<ConstantOp>(op.getLoc(),
                                          rewriter.getF64FloatAttr(M_PI));
    auto pi2 = rewriter.create<ConstantOp>(op.getLoc(),
                                           rewriter.getF64FloatAttr(M_PI_2));

    // Compute φ - π/2
    auto phiMinusPi2 = rewriter.create<SubFOp>(op.getLoc(), phi, pi2);
    // Compute π - θ/2
    auto two =
        rewriter.create<ConstantOp>(op.getLoc(), rewriter.getF64FloatAttr(2.0));
    auto theta2 = rewriter.create<DivFOp>(op.getLoc(), theta, two);
    auto piMinusTheta2 = rewriter.create<SubFOp>(op.getLoc(), pi, theta2);
    // Compute λ - π/2
    auto lambdaMinusPi2 = rewriter.create<SubFOp>(op.getLoc(), lambda, pi2);

    // RZ(λ − π/2)
    auto rz1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{lambdaMinusPi2},
        inQubitsValues, "RZ", false, inCtrlQubits, ValueRange{});

    // RX(π/2)
    auto rx1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{pi2},
        rz1.getResults(), "RX", false, inCtrlQubits, ValueRange{});

    // RZ(π − θ)
    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{piMinusTheta2},
        rx1.getResults(), "RZ", false, inCtrlQubits, ValueRange{});

    // RX(π/2)
    auto rx2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{pi2},
        rz2.getResults(), "RX", false, inCtrlQubits, ValueRange{});

    // RZ(φ − π/2)
    auto rz3 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{phiMinusPi2},
        rx2.getResults(), "RZ", false, inCtrlQubits, ValueRange{});

    // Replace the original U gate with the decomposed sequence
    rewriter.replaceOp(op, rz3.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<opt::U2Op> final
    : OpConversionPattern<opt::U2Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::U2Op op, opt::U2Op::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    // Extract parameters
    SmallVector<Value> paramValues;
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
          auto floatAttr = rewriter.getF64FloatAttr(attr);
          auto constOp = rewriter.create<ConstantOp>(op.getLoc(), floatAttr);
          paramValues.push_back(constOp);
        } else {
          // Dynamic parameter
          paramValues.push_back(dynamicParams[dynIdx++]);
        }
      } else if (staticParams.has_value()) {
        // All static
        auto attr = (*staticParams)[i];
        auto floatAttr = rewriter.getF64FloatAttr(attr);
        auto constOp = rewriter.create<ConstantOp>(op.getLoc(), floatAttr);
        paramValues.push_back(constOp);
      } else {
        // All dynamic
        paramValues.push_back(dynamicParams[i]);
      }
    }
    // Now paramValues [0] = φ, [1] = λ
    auto phi = paramValues[0];
    auto lambda = paramValues[1];

    SmallVector<Value> inCtrlQubits(posCtrlQubitsValues.begin(),
                                    posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    const Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    const SmallVector<Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = TypeRange(qubitTypes);

    // U2(φ, λ) = U(π/2, φ, λ) = RZ(φ − π⁄2) ⋅ RX(π⁄2) ⋅ RZ(3/4 π) ⋅ RX(π⁄2) ⋅
    // RZ(λ − π⁄2)
    auto pi2 = rewriter.create<ConstantOp>(op.getLoc(),
                                           rewriter.getF64FloatAttr(M_PI_2));
    auto pi4 = rewriter.create<ConstantOp>(op.getLoc(),
                                           rewriter.getF64FloatAttr(M_PI_4));
    auto three =
        rewriter.create<ConstantOp>(op.getLoc(), rewriter.getF64FloatAttr(3.0));
    auto pi34 = rewriter.create<MulFOp>(op.getLoc(), pi4, three);

    // Compute φ - π/2
    auto phiMinusPi2 = rewriter.create<SubFOp>(op.getLoc(), phi, pi2);
    // Compute λ - π/2
    auto lambdaMinusPi2 = rewriter.create<SubFOp>(op.getLoc(), lambda, pi2);

    // RZ(λ − π/2)
    auto rz1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{lambdaMinusPi2},
        inQubitsValues, "RZ", false, inCtrlQubits, ValueRange{});

    // RX(π/2)
    auto rx1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{pi2},
        rz1.getResults(), "RX", false, inCtrlQubits, ValueRange{});

    // RZ(3/4 π)
    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{pi34},
        rx1.getResults(), "RZ", false, inCtrlQubits, ValueRange{});

    // RX(π/2)
    auto rx2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{pi2},
        rz2.getResults(), "RX", false, inCtrlQubits, ValueRange{});

    // RZ(φ − π/2)
    auto rz3 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{phiMinusPi2},
        rx2.getResults(), "RZ", false, inCtrlQubits, ValueRange{});

    // Replace the original U gate with the decomposed sequence
    rewriter.replaceOp(op, rz3.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<opt::PeresOp> final
    : OpConversionPattern<opt::PeresOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::PeresOp op, opt::PeresOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operands
    auto inQubitsValues = adaptor.getInQubits();
    const auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    const auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    SmallVector<Value> inCtrlQubits(posCtrlQubitsValues.begin(),
                                    posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    const Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    const SmallVector<Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = TypeRange(qubitTypes);

    // Peres = Toffoli(q2,q1,q0) CNOT(q2,q1)
    auto ccnot1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{}, inQubitsValues,
        "Toffoli", false, inCtrlQubits, ValueRange{});

    auto cnot2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{},
        ccnot1.getResults(), "CNOT", false, inCtrlQubits, ValueRange{});

    rewriter.replaceOp(op, cnot2.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<opt::PeresdgOp> final
    : OpConversionPattern<opt::PeresdgOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::PeresdgOp op, opt::PeresdgOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operands
    auto inQubitsValues = adaptor.getInQubits();
    const auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    const auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    SmallVector<Value> inCtrlQubits(posCtrlQubitsValues.begin(),
                                    posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type setup
    const Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    const SmallVector<Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = TypeRange(qubitTypes);

    // Peresdg = CNOT(q2,q1) Toffoli(q2,q1,q0)
    auto cnot1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{}, inQubitsValues,
        "CNOT", false, inCtrlQubits, ValueRange{});

    auto ccnot2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, TypeRange{}, ValueRange{},
        cnot1.getResults(), "Toffoli", false, inCtrlQubits, ValueRange{});

    rewriter.replaceOp(op, ccnot2.getResults());
    return success();
  }
};

// -- GPhaseOp (gphase)
template <>
StringRef ConvertMQTOptSimpleGate<opt::GPhaseOp>::getGateName(
    [[maybe_unused]] std::size_t numControls) {
  return "gphase";
}

// -- IOp (Identity)
template <>
StringRef ConvertMQTOptSimpleGate<opt::IOp>::getGateName(
    [[maybe_unused]] std::size_t numControls) {
  return "Identity";
}

// -- XOp (PauliX, CNOT, Toffoli)
template <>
StringRef
ConvertMQTOptSimpleGate<opt::XOp>::getGateName(const std::size_t numControls) {
  if (numControls == 0) {
    return "PauliX";
  }
  if (numControls == 1) {
    return "CNOT";
  }
  if (numControls == 2) {
    return "Toffoli";
  }
  return "";
}

// -- YOp (PauliY, CY)
template <>
StringRef
ConvertMQTOptSimpleGate<opt::YOp>::getGateName(const std::size_t numControls) {
  if (numControls == 0) {
    return "PauliY";
  }
  if (numControls == 1) {
    return "CY";
  }
  return "";
}

// -- ZOp (PauliZ, CZ)
template <>
StringRef
ConvertMQTOptSimpleGate<opt::ZOp>::getGateName(const std::size_t numControls) {
  if (numControls == 0) {
    return "PauliZ";
  }
  if (numControls == 1) {
    return "CZ";
  }
  return "";
}

// -- HOp (Hadamard)
template <>
StringRef ConvertMQTOptSimpleGate<opt::HOp>::getGateName(
    [[maybe_unused]] std::size_t numControls) {
  return "Hadamard";
}

// -- SOP (S)
template <>
StringRef ConvertMQTOptSimpleGate<opt::SOp>::getGateName(
    [[maybe_unused]] std::size_t numControls) {
  return "S";
}

// -- TOP (T)
template <>
StringRef ConvertMQTOptSimpleGate<opt::TOp>::getGateName(
    [[maybe_unused]] std::size_t numControls) {
  return "T";
}

// -- ECROp (ECR)
template <>
StringRef ConvertMQTOptSimpleGate<opt::ECROp>::getGateName(
    [[maybe_unused]] std::size_t numControls) {
  return "ECR";
}

// -- SWAPOp (SWAP)
template <>
StringRef ConvertMQTOptSimpleGate<opt::SWAPOp>::getGateName(
    const std::size_t numControls) {
  if (numControls == 0) {
    return "SWAP";
  }
  if (numControls == 1) {
    return "CSWAP";
  }
  return "";
}

// -- iSWAPOp (iSWAP)
template <>
StringRef ConvertMQTOptSimpleGate<opt::iSWAPOp>::getGateName(
    [[maybe_unused]] std::size_t numControls) {
  return "ISWAP";
}

// -- RXOp (RX, CRX)
template <>
StringRef
ConvertMQTOptSimpleGate<opt::RXOp>::getGateName(const std::size_t numControls) {
  if (numControls == 0) {
    return "RX";
  }
  if (numControls == 1) {
    return "CRX";
  }
  return "";
}

// -- RYOp (RY, CRY)
template <>
StringRef
ConvertMQTOptSimpleGate<opt::RYOp>::getGateName(const std::size_t numControls) {
  if (numControls == 0) {
    return "RY";
  }
  if (numControls == 1) {
    return "CRY";
  }
  return "";
}

// -- RZOp (RZ, CRZ)
template <>
StringRef
ConvertMQTOptSimpleGate<opt::RZOp>::getGateName(const std::size_t numControls) {
  if (numControls == 0) {
    return "RZ";
  }
  if (numControls == 1) {
    return "CRZ";
  }
  return "";
}

// -- POp (PhaseShift, ControlledPhaseShift)
template <>
StringRef
ConvertMQTOptSimpleGate<opt::POp>::getGateName(const std::size_t numControls) {
  if (numControls == 0) {
    return "PhaseShift";
  }
  if (numControls == 1) {
    return "ControlledPhaseShift";
  }
  return "";
}

// -- RXXOp (IsingXX)
template <>
StringRef ConvertMQTOptSimpleGate<opt::RXXOp>::getGateName(
    [[maybe_unused]] std::size_t numControls) {
  return "IsingXX";
}

// -- RYYOp (IsingYY)
template <>
StringRef ConvertMQTOptSimpleGate<opt::RYYOp>::getGateName(
    [[maybe_unused]] std::size_t numControls) {
  return "IsingYY";
}

// -- RZZ (IsingZZ)
template <>
StringRef ConvertMQTOptSimpleGate<opt::RZZOp>::getGateName(
    [[maybe_unused]] std::size_t numControls) {
  return "IsingZZ";
}

// -- XXplusYY (IsingXY)
template <>
StringRef ConvertMQTOptSimpleGate<opt::XXplusYY>::getGateName(
    [[maybe_unused]] std::size_t numControls) {
  return "IsingXY";
}

struct MQTOptToCatalystQuantum final
    : impl::MQTOptToCatalystQuantumBase<MQTOptToCatalystQuantum> {
  using MQTOptToCatalystQuantumBase::MQTOptToCatalystQuantumBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<catalyst::quantum::QuantumDialect>();
    target.addIllegalDialect<opt::MQTOptDialect>();

    RewritePatternSet patterns(context);
    MQTOptToCatalystQuantumTypeConverter typeConverter(context);

    patterns.add<ConvertMQTOptAlloc, ConvertMQTOptDealloc, ConvertMQTOptExtract,
                 ConvertMQTOptMeasure, ConvertMQTOptInsert>(typeConverter,
                                                            context);

    patterns.add<ConvertMQTOptSimpleGate<opt::BarrierOp>>(typeConverter,
                                                          context);
    patterns.add<ConvertMQTOptSimpleGate<opt::GPhaseOp>>(typeConverter,
                                                         context);
    patterns.add<ConvertMQTOptSimpleGate<opt::IOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::XOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::YOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::ZOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::SOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::TOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::VOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::VdgOp>>(typeConverter, context);

    patterns.add<ConvertMQTOptSimpleGate<opt::RXOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::RYOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::RZOp>>(typeConverter, context);

    patterns.add<ConvertMQTOptSimpleGate<opt::HOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::SWAPOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::iSWAPOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::POp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::DCXOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::ECROp>>(typeConverter, context);

    patterns.add<ConvertMQTOptSimpleGate<opt::RXXOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::RYYOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::RZZOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::RZXOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::UOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::U2Op>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::PeresOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptSimpleGate<opt::PeresdgOp>>(typeConverter,
                                                          context);

    patterns.add<ConvertMQTOptAdjointGate<opt::SXOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptAdjointGate<opt::SdgOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptAdjointGate<opt::TdgOp>>(typeConverter, context);
    patterns.add<ConvertMQTOptAdjointGate<opt::iSWAPdgOp>>(typeConverter,
                                                           context);

    patterns.add<ConvertMQTOptSimpleGate<opt::XXminusYY>>(typeConverter,
                                                          context);
    patterns.add<ConvertMQTOptSimpleGate<opt::XXplusYY>>(typeConverter,
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
        [&](const func::ReturnOp op) { return typeConverter.isLegal(op); });

    // Rewrites call sites (func.call) to use the converted argument and result
    // types. Needed so that calls into rewritten functions pass/receive correct
    // types.
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    // Legal only if operand/result types are all type-converted correctly.
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](const func::CallOp op) { return typeConverter.isLegal(op); });

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

} // namespace mqt::ir::conversions

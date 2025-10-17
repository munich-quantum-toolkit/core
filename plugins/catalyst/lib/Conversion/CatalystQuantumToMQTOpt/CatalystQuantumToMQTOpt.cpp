/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h"

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

#include <Quantum/IR/QuantumDialect.h>
#include <Quantum/IR/QuantumOps.h>
#include <cassert>
#include <cstddef>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mqt::ir::conversions {

#define GEN_PASS_DEF_CATALYSTQUANTUMTOMQTOPT
#include "mlir/Conversion/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h.inc"

using namespace mlir;
using namespace mlir::arith;

class CatalystQuantumToMQTOptTypeConverter final : public TypeConverter {
public:
  explicit CatalystQuantumToMQTOptTypeConverter(MLIRContext* ctx) {
    // Identity conversion: Allow all types to pass through unmodified if needed
    addConversion([](const Type type) { return type; });

    // Convert Catalyst QubitType to MQTOpt QubitType
    addConversion([ctx](catalyst::quantum::QubitType /*type*/) -> Type {
      return opt::QubitType::get(ctx);
    });

    // Convert Catalyst QuregType to dynamic memref as placeholder
    // The actual static memref types will flow through from alloc operations
    addConversion([ctx](catalyst::quantum::QuregType /*type*/) -> Type {
      auto qubitType = opt::QubitType::get(ctx);
      return MemRefType::get({ShapedType::kDynamic}, qubitType);
    });

    // Target materialization: converts values during pattern application
    // Just returns the input - the actual memref is already created by alloc
    addTargetMaterialization([](OpBuilder& builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() == 1) {
        return inputs[0];
      }
      return nullptr;
    });
  }
};

struct ConvertQuantumAlloc final
    : OpConversionPattern<catalyst::quantum::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // Get the number of qubits from the attribute
    auto nqubitsAttr = op.getNqubitsAttrAttr();
    if (!nqubitsAttr) {
      return op.emitError("AllocOp missing nqubits_attr");
    }

    const auto nqubits = nqubitsAttr.getValue().getZExtValue();

    // Prepare the result type(s)
    const auto qubitType = opt::QubitType::get(rewriter.getContext());
    const auto memrefType =
        MemRefType::get({static_cast<int64_t>(nqubits)}, qubitType);

    // Create the new operation
    auto allocOp = rewriter.create<memref::AllocOp>(op.getLoc(), memrefType);

    // Replace the original with the new operation
    rewriter.replaceOp(op, allocOp.getResult());
    return success();
  }
};

struct ConvertQuantumDealloc final
    : OpConversionPattern<catalyst::quantum::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s)
    Value memref = adaptor.getQreg();

    // Unwrap unrealized_conversion_cast if present
    if (auto castOp = memref.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (!castOp.getInputs().empty()) {
        memref = castOp.getInputs()[0];
      }
    }

    // Create the new operation
    rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, memref);
    return success();
  }
};

struct ConvertQuantumMeasure final
    : OpConversionPattern<catalyst::quantum::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s)
    const auto inQubit = adaptor.getInQubit();

    // Prepare the result type(s)
    const auto qubitType = opt::QubitType::get(rewriter.getContext());
    const auto bitType = rewriter.getI1Type();

    // Create the new operation
    // Note: quantum.measure returns (i1, !quantum.bit)
    //       mqtopt.measure returns (!mqtopt.Qubit, i1)
    const auto mqtoptOp = rewriter.create<opt::MeasureOp>(
        op.getLoc(), qubitType, bitType, inQubit);

    // Replace with results in the correct order
    rewriter.replaceOp(op, {mqtoptOp.getResult(1), mqtoptOp.getResult(0)});
    return success();
  }
};

struct ConvertQuantumExtract final
    : OpConversionPattern<catalyst::quantum::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Prepare the result type(s)
    const auto qubitType = opt::QubitType::get(rewriter.getContext());

    // Get index (either from attribute or operand)
    Value indexValue;
    auto idxAttr = op.getIdxAttrAttr();

    if (idxAttr) {
      // Compile-time constant index from attribute
      const auto idx = idxAttr.getValue().getZExtValue();
      indexValue = rewriter.create<ConstantIndexOp>(op.getLoc(), idx);
    } else {
      // Runtime dynamic index from operand
      auto idxOperand = adaptor.getIdx();
      if (!idxOperand) {
        return op.emitError("ExtractOp missing both idx_attr and idx operand");
      }

      // Convert i64 to index type if needed
      if (isa<IntegerType>(idxOperand.getType())) {
        indexValue = rewriter.create<IndexCastOp>(
            op.getLoc(), rewriter.getIndexType(), idxOperand);
      } else {
        indexValue = idxOperand;
      }
    }

    // Extract operand(s)
    Value memref = adaptor.getQreg();

    // Unwrap unrealized_conversion_cast if present
    if (auto castOp = memref.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (!castOp.getInputs().empty()) {
        memref = castOp.getInputs()[0];
      }
    }

    // Verify we got a static memref type as expected
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType || !memrefType.hasStaticShape()) {
      return op.emitError("Expected static memref type from alloc, got: ")
             << memref.getType();
    }

    // Create the new operation
    auto loadOp = rewriter.create<memref::LoadOp>(
        op.getLoc(), qubitType, memref, ValueRange{indexValue});

    // Replace the extract operation with the loaded qubit
    rewriter.replaceOp(op, loadOp.getResult());
    return success();
  }
};

struct ConvertQuantumInsert final
    : OpConversionPattern<catalyst::quantum::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Get index (either from attribute or operand)
    Value indexValue;
    auto idxAttr = op.getIdxAttrAttr();

    if (idxAttr) {
      // Compile-time constant index from attribute
      const auto idx = idxAttr.getValue().getZExtValue();
      indexValue = rewriter.create<ConstantIndexOp>(op.getLoc(), idx);
    } else {
      // Runtime dynamic index from operand
      auto idxOperand = adaptor.getIdx();
      if (!idxOperand) {
        return op.emitError("InsertOp missing both idx_attr and idx operand");
      }

      // Convert i64 to index type if needed
      if (isa<IntegerType>(idxOperand.getType())) {
        indexValue = rewriter.create<IndexCastOp>(
            op.getLoc(), rewriter.getIndexType(), idxOperand);
      } else {
        indexValue = idxOperand;
      }
    }

    // Extract operand(s)
    Value memref = adaptor.getInQreg();

    // Unwrap unrealized_conversion_cast if present
    if (auto castOp = memref.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (!castOp.getInputs().empty()) {
        memref = castOp.getInputs()[0];
      }
    }

    // Create the new operation
    rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.getQubit(), memref,
                                     ValueRange{indexValue});

    // In the memref model, the register is modified in-place
    rewriter.replaceOp(op, memref);
    return success();
  }
};

struct ConvertQuantumGlobalPhase final
    : OpConversionPattern<catalyst::quantum::GlobalPhaseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::GlobalPhaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    const auto param = adaptor.getParams();
    const auto inCtrlQubits = adaptor.getInCtrlQubits();
    const auto inCtrlValues = adaptor.getInCtrlValues();

    // Separate positive and negative control qubits
    SmallVector<Value> inPosCtrlQubitsVec;
    SmallVector<Value> inNegCtrlQubitsVec;

    for (size_t i = 0; i < inCtrlQubits.size(); ++i) {
      if (inCtrlValues[i]) {
        inPosCtrlQubitsVec.emplace_back(inCtrlQubits[i]);
      } else {
        inNegCtrlQubitsVec.emplace_back(inCtrlQubits[i]);
      }
    }

    // Create the parameter attributes
    SmallVector<double> staticParamsVec;
    SmallVector<bool> paramsMaskVec;
    SmallVector<Value> finalParamValues;

    // Check if parameter is a compile-time constant
    if (auto constOp = param.getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
        staticParamsVec.push_back(floatAttr.getValueAsDouble());
        paramsMaskVec.push_back(true);
      } else {
        finalParamValues.push_back(param);
        paramsMaskVec.push_back(false);
      }
    } else {
      finalParamValues.push_back(param);
      paramsMaskVec.push_back(false);
    }

    const auto staticParams =
        DenseF64ArrayAttr::get(rewriter.getContext(), staticParamsVec);
    const auto paramsMask =
        DenseBoolArrayAttr::get(rewriter.getContext(), paramsMaskVec);

    // Create the new operation
    auto mqtoptOp = rewriter.create<opt::GPhaseOp>(
        op.getLoc(), TypeRange{},                   // out_qubits
        ValueRange(inPosCtrlQubitsVec).getTypes(),  // pos_ctrl_out_qubits
        ValueRange(inNegCtrlQubitsVec).getTypes(),  // neg_ctrl_out_qubits
        staticParams, paramsMask, finalParamValues, // params
        ValueRange{},                               // in_qubits
        ValueRange(inPosCtrlQubitsVec),             // pos_ctrl_in_qubits
        ValueRange(inNegCtrlQubitsVec));            // neg_ctrl_in_qubits

    // Replace the original with the new operation
    rewriter.replaceOp(op, mqtoptOp);
    return success();
  }
};

struct ConvertQuantumCustomOp final
    : OpConversionPattern<catalyst::quantum::CustomOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    const auto gateName = op.getGateName();
    const auto paramsValues = adaptor.getParams();
    const auto inQubits = adaptor.getInQubits();
    const auto inCtrlQubits = adaptor.getInCtrlQubits();
    const auto inCtrlValues = adaptor.getInCtrlValues();

    // Separate positive and negative control qubits
    SmallVector<Value> inPosCtrlQubitsVec;
    SmallVector<Value> inNegCtrlQubitsVec;

    for (size_t i = 0; i < inCtrlQubits.size(); ++i) {
      if (inCtrlValues[i]) {
        inPosCtrlQubitsVec.emplace_back(inCtrlQubits[i]);
      } else {
        inNegCtrlQubitsVec.emplace_back(inCtrlQubits[i]);
      }
    }

    // Process parameters (static vs dynamic)
    SmallVector<bool> paramsMaskVec;
    SmallVector<double> staticParamsVec;
    SmallVector<Value> finalParamValues;

    auto maskAttr = op->getAttrOfType<DenseBoolArrayAttr>("params_mask");
    auto staticParamsAttr =
        op->getAttrOfType<DenseF64ArrayAttr>("static_params");

    size_t totalParams = 0;
    if (maskAttr) {
      totalParams = maskAttr.size();
    } else {
      totalParams = staticParamsAttr
                        ? staticParamsAttr.size() + paramsValues.size()
                        : paramsValues.size();
    }

    size_t staticIdx = 0;
    size_t dynamicIdx = 0;

    for (size_t i = 0; i < totalParams; ++i) {
      const bool isStatic = (maskAttr ? maskAttr[i] : false);

      paramsMaskVec.emplace_back(isStatic);

      if (isStatic) {
        assert(staticParamsAttr && "Missing static_params for static mask");
        staticParamsVec.emplace_back(staticParamsAttr[staticIdx++]);
      } else {
        assert(dynamicIdx < paramsValues.size() &&
               "Too few dynamic parameters");
        finalParamValues.emplace_back(paramsValues[dynamicIdx++]);
      }
    }

    const auto staticParams =
        DenseF64ArrayAttr::get(rewriter.getContext(), staticParamsVec);
    const auto paramsMask =
        DenseBoolArrayAttr::get(rewriter.getContext(), paramsMaskVec);

    // Create the new operation
    Operation* mqtoptOp = nullptr;

#define CREATE_GATE_OP(GATE_TYPE)                                              \
  rewriter.create<opt::GATE_TYPE##Op>(                                         \
      op.getLoc(), inQubits.getTypes(),                                        \
      ValueRange(inPosCtrlQubitsVec).getTypes(),                               \
      ValueRange(inNegCtrlQubitsVec).getTypes(), staticParams, paramsMask,     \
      finalParamValues, inQubits, inPosCtrlQubitsVec, inNegCtrlQubitsVec)

    if (gateName == "Hadamard") {
      mqtoptOp = CREATE_GATE_OP(H);
    } else if (gateName == "Identity") {
      mqtoptOp = CREATE_GATE_OP(I);
    } else if (gateName == "PauliX") {
      mqtoptOp = CREATE_GATE_OP(X);
    } else if (gateName == "PauliY") {
      mqtoptOp = CREATE_GATE_OP(Y);
    } else if (gateName == "PauliZ") {
      mqtoptOp = CREATE_GATE_OP(Z);
    } else if (gateName == "S") {
      mqtoptOp = CREATE_GATE_OP(S);
    } else if (gateName == "T") {
      mqtoptOp = CREATE_GATE_OP(T);
    } else if (gateName == "SX") {
      mqtoptOp = CREATE_GATE_OP(SX);
    } else if (gateName == "ECR") {
      mqtoptOp = CREATE_GATE_OP(ECR);
    } else if (gateName == "SWAP") {
      mqtoptOp = CREATE_GATE_OP(SWAP);
    } else if (gateName == "ISWAP") {
      if (op.getAdjoint()) {
        mqtoptOp = CREATE_GATE_OP(iSWAPdg);
      } else {
        mqtoptOp = CREATE_GATE_OP(iSWAP);
      }
    } else if (gateName == "RX") {
      mqtoptOp = CREATE_GATE_OP(RX);
    } else if (gateName == "RY") {
      mqtoptOp = CREATE_GATE_OP(RY);
    } else if (gateName == "RZ") {
      mqtoptOp = CREATE_GATE_OP(RZ);
    } else if (gateName == "PhaseShift") {
      mqtoptOp = CREATE_GATE_OP(P);
    } else if (gateName == "CRX") {
      // CRX gate: 1 control qubit + 1 target qubit
      // inQubits[0] is control, inQubits[1] is target
      inPosCtrlQubitsVec.emplace_back(inQubits[0]);
      mqtoptOp = rewriter.create<opt::RXOp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(inPosCtrlQubitsVec).getTypes(),
          ValueRange(inNegCtrlQubitsVec).getTypes(), staticParams, paramsMask,
          finalParamValues, inQubits[1], inPosCtrlQubitsVec,
          inNegCtrlQubitsVec);
    } else if (gateName == "CRY") {
      // CRY gate: 1 control qubit + 1 target qubit
      // inQubits[0] is control, inQubits[1] is target
      inPosCtrlQubitsVec.emplace_back(inQubits[0]);
      mqtoptOp = rewriter.create<opt::RYOp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(inPosCtrlQubitsVec).getTypes(),
          ValueRange(inNegCtrlQubitsVec).getTypes(), staticParams, paramsMask,
          finalParamValues, inQubits[1], inPosCtrlQubitsVec,
          inNegCtrlQubitsVec);
    } else if (gateName == "CRZ") {
      // CRZ gate: 1 control qubit + 1 target qubit
      // inQubits[0] is control, inQubits[1] is target
      inPosCtrlQubitsVec.emplace_back(inQubits[0]);
      mqtoptOp = rewriter.create<opt::RZOp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(inPosCtrlQubitsVec).getTypes(),
          ValueRange(inNegCtrlQubitsVec).getTypes(), staticParams, paramsMask,
          finalParamValues, inQubits[1], inPosCtrlQubitsVec,
          inNegCtrlQubitsVec);
    } else if (gateName == "ControlledPhaseShift") {
      // ControlledPhaseShift gate: 1 control qubit + 1 target qubit
      // inQubits[0] is control, inQubits[1] is target
      inPosCtrlQubitsVec.emplace_back(inQubits[0]);
      mqtoptOp = rewriter.create<opt::POp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(inPosCtrlQubitsVec).getTypes(),
          ValueRange(inNegCtrlQubitsVec).getTypes(), staticParams, paramsMask,
          finalParamValues, inQubits[1], inPosCtrlQubitsVec,
          inNegCtrlQubitsVec);
    } else if (gateName == "IsingXY") {
      // PennyLane IsingXY has 1 parameter (phi), OpenQASM XXPlusYY needs 2
      // (theta, beta) Relationship: IsingXY(phi) = XXPlusYY(phi, pi) Add pi as
      // second parameter
      SmallVector<Value> isingxyParams(finalParamValues.begin(),
                                       finalParamValues.end());
      isingxyParams.push_back(rewriter
                                  .create<arith::ConstantFloatOp>(
                                      op.getLoc(), APFloat(3.141592653589793),
                                      rewriter.getF64Type())
                                  .getResult());

      SmallVector<double> isingxyStaticParams(staticParamsVec.begin(),
                                              staticParamsVec.end());
      SmallVector<bool> isingxyParamsMask(paramsMaskVec.begin(),
                                          paramsMaskVec.end());
      isingxyParamsMask.push_back(false); // pi is a dynamic constant

      auto isingxyStaticParamsAttr =
          DenseF64ArrayAttr::get(rewriter.getContext(), isingxyStaticParams);
      auto isingxyParamsMaskAttr =
          DenseBoolArrayAttr::get(rewriter.getContext(), isingxyParamsMask);

      mqtoptOp = rewriter.create<opt::XXplusYYOp>(
          op.getLoc(), inQubits.getTypes(),
          ValueRange(inPosCtrlQubitsVec).getTypes(),
          ValueRange(inNegCtrlQubitsVec).getTypes(), isingxyStaticParamsAttr,
          isingxyParamsMaskAttr, isingxyParams, inQubits, inPosCtrlQubitsVec,
          inNegCtrlQubitsVec);
    } else if (gateName == "IsingXX") {
      mqtoptOp = CREATE_GATE_OP(RXX);
    } else if (gateName == "IsingYY") {
      mqtoptOp = CREATE_GATE_OP(RYY);
    } else if (gateName == "IsingZZ") {
      mqtoptOp = CREATE_GATE_OP(RZZ);
    } else if (gateName == "CNOT") {
      inPosCtrlQubitsVec.emplace_back(inQubits[0]);
      mqtoptOp = rewriter.create<opt::XOp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(inPosCtrlQubitsVec).getTypes(),
          ValueRange(inNegCtrlQubitsVec).getTypes(), staticParams, paramsMask,
          finalParamValues, inQubits[1], inPosCtrlQubitsVec,
          inNegCtrlQubitsVec);
    } else if (gateName == "CY") {
      inPosCtrlQubitsVec.emplace_back(inQubits[0]);
      mqtoptOp = rewriter.create<opt::YOp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(inPosCtrlQubitsVec).getTypes(),
          ValueRange(inNegCtrlQubitsVec).getTypes(), staticParams, paramsMask,
          finalParamValues, inQubits[1], inPosCtrlQubitsVec,
          inNegCtrlQubitsVec);
    } else if (gateName == "CZ") {
      inPosCtrlQubitsVec.emplace_back(inQubits[0]);
      mqtoptOp = rewriter.create<opt::ZOp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(inPosCtrlQubitsVec).getTypes(),
          ValueRange(inNegCtrlQubitsVec).getTypes(), staticParams, paramsMask,
          finalParamValues, inQubits[1], inPosCtrlQubitsVec,
          inNegCtrlQubitsVec);
    } else if (gateName == "Toffoli") {
      // Toffoli gate: 2 control qubits + 1 target qubit
      // inQubits[0] and inQubits[1] are controls, inQubits[2] is target
      inPosCtrlQubitsVec.emplace_back(inQubits[0]);
      inPosCtrlQubitsVec.emplace_back(inQubits[1]);
      mqtoptOp = rewriter.create<opt::XOp>(
          op.getLoc(), inQubits[2].getType(),
          ValueRange(inPosCtrlQubitsVec).getTypes(),
          ValueRange(inNegCtrlQubitsVec).getTypes(), staticParams, paramsMask,
          finalParamValues, inQubits[2], inPosCtrlQubitsVec,
          inNegCtrlQubitsVec);
    } else if (gateName == "CSWAP") {
      // CSWAP gate: 1 control qubit + 2 target qubits
      // inQubits[0] is control, inQubits[1] and inQubits[2] are targets
      inPosCtrlQubitsVec.emplace_back(inQubits[0]);
      mqtoptOp = rewriter.create<opt::SWAPOp>(
          op.getLoc(), ValueRange{inQubits[1], inQubits[2]},
          ValueRange(inPosCtrlQubitsVec).getTypes(),
          ValueRange(inNegCtrlQubitsVec).getTypes(), staticParams, paramsMask,
          finalParamValues, ValueRange{inQubits[1], inQubits[2]},
          inPosCtrlQubitsVec, inNegCtrlQubitsVec);
    } else {
      llvm::errs() << "Unsupported gate: " << gateName << "\n";
      return failure();
    }

#undef CREATE_GATE_OP

    // Replace the original with the new operation
    rewriter.replaceOp(op, mqtoptOp);
    return success();
  }
};

struct CatalystQuantumToMQTOpt final
    : impl::CatalystQuantumToMQTOptBase<CatalystQuantumToMQTOpt> {
  using CatalystQuantumToMQTOptBase::CatalystQuantumToMQTOptBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<opt::MQTOptDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addIllegalDialect<catalyst::quantum::QuantumDialect>();

    // Mark operations legal that have no equivalent in the target dialect
    target.addLegalOp<
        catalyst::quantum::DeviceInitOp, catalyst::quantum::DeviceReleaseOp,
        catalyst::quantum::NamedObsOp, catalyst::quantum::ExpvalOp,
        catalyst::quantum::FinalizeOp, catalyst::quantum::ComputationalBasisOp,
        catalyst::quantum::StateOp, catalyst::quantum::InitializeOp>();

    RewritePatternSet patterns(context);
    const CatalystQuantumToMQTOptTypeConverter typeConverter(context);

    patterns
        .add<ConvertQuantumAlloc, ConvertQuantumDealloc, ConvertQuantumMeasure,
             ConvertQuantumExtract, ConvertQuantumInsert,
             ConvertQuantumGlobalPhase, ConvertQuantumCustomOp>(typeConverter,
                                                                context);

    // Type conversion boilerplate to handle function signatures and control
    // flow See: https://www.jeremykun.com/2023/10/23/mlir-dialect-conversion

    // Convert func.func signatures to use the converted types
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    // Mark func.func as legal only if signature and body types are converted
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    // Convert return ops to match the new function result types
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    // Mark func.return as legal only if operand types match converted types
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](const func::ReturnOp op) { return typeConverter.isLegal(op); });

    // Convert call sites to use the converted argument and result types
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    // Mark func.call as legal only if operand and result types are converted
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](const func::CallOp op) { return typeConverter.isLegal(op); });

    // Convert control-flow ops (cf.br, cf.cond_br, etc.)
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    // Mark unknown ops as legal if they don't require type conversion
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

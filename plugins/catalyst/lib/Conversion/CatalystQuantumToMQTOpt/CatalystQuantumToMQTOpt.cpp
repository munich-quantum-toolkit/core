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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

#include <Quantum/IR/QuantumDialect.h>
#include <Quantum/IR/QuantumOps.h>
#include <cassert>
#include <cstddef>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/BuiltinAttributes.h>
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

#define GEN_PASS_DEF_CATALYSTQUANTUMTOMQTOPT
#include "mlir/Conversion/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h.inc"

using namespace mlir;

class CatalystQuantumToMQTOptTypeConverter final : public TypeConverter {
public:
  explicit CatalystQuantumToMQTOptTypeConverter(MLIRContext* ctx) {
    // Identity conversion: Allow all types to pass through unmodified if
    // needed.
    addConversion([](const Type type) { return type; });

    // Convert source QubitType to target QubitType
    addConversion([ctx](catalyst::quantum::QubitType /*type*/) -> Type {
      return opt::QubitType::get(ctx);
    });

    // Convert QuregType to static memref.
    // This signals to the adaptor how to map QuregType operands.
    // The actual static memref types will flow through from the alloc
    // operation.
    addConversion([ctx](catalyst::quantum::QuregType /*type*/) -> Type {
      // We return a "signature" showing QuregType maps to memref, but
      // the actual size will come from the alloc operation.
      // Using dynamic shape here as a placeholder - the actual static memref
      // types will flow through without materialization.
      auto qubitType = opt::QubitType::get(ctx);
      return MemRefType::get({ShapedType::kDynamic}, qubitType);
    });

    // Add target materialization: this is called when the framework needs to
    // convert a QuregType value to a memref. The input is the actual static
    // memref created by the alloc pattern. We just return it directly - no cast
    // needed.
    addTargetMaterialization([](OpBuilder& builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      // Just return the input value - it is already the memref we want
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

    auto nqubits = nqubitsAttr.getValue().getZExtValue();

    // Create the static memref type directly
    auto qubitType = opt::QubitType::get(rewriter.getContext());
    auto staticMemrefType =
        MemRefType::get({static_cast<int64_t>(nqubits)}, qubitType);

    // Create the allocation with the static size
    auto allocOp =
        rewriter.create<memref::AllocOp>(op.getLoc(), staticMemrefType);

    // Replace the original operation with the alloc result directly
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
    // Get the memref from adaptor
    Value memref = adaptor.getQreg();

    // If we got an unrealized_conversion_cast wrapping the static memref,
    // unwrap it to get the actual static memref
    if (auto castOp = memref.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (castOp.getInputs().size() == 1) {
        memref = castOp.getInputs()[0];
      }
    }

    // Replace with memref.dealloc using the static memref
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
    // Prepare the result type(s)
    const auto qubitType = opt::QubitType::get(rewriter.getContext());
    const auto bitType = rewriter.getI1Type();

    // Create the new operation
    // Note: quantum.measure returns (i1, !quantum.bit)
    //       mqtopt.measure returns (!mqtopt.Qubit, i1)
    // So we need to swap the result ordering when replacing
    auto measureOp = rewriter.create<opt::MeasureOp>(
        op.getLoc(), qubitType, bitType, adaptor.getInQubit());

    // Replace with results in the correct order:
    // op.getResult(0) is i1 -> maps to measureOp.getResult(1) (i1)
    // op.getResult(1) is !quantum.bit -> maps to measureOp.getResult(0)
    // (!mqtopt.Qubit)
    rewriter.replaceOp(op, {measureOp.getResult(1), measureOp.getResult(0)});

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
    auto qubitType = opt::QubitType::get(rewriter.getContext());

    // Get the index - either from attribute (compile-time) or operand (runtime)
    Value indexValue;
    auto idxAttr = op.getIdxAttrAttr();

    if (idxAttr) {
      // Compile-time constant index from attribute
      auto idx = idxAttr.getValue().getZExtValue();
      indexValue = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), idx);
    } else {
      // Runtime dynamic index from operand
      // The index is the second operand (first is qreg)
      auto idxOperand = adaptor.getIdx();
      if (!idxOperand) {
        return op.emitError("ExtractOp missing both idx_attr and idx operand");
      }

      // Convert i64 to index type if needed
      if (isa<IntegerType>(idxOperand.getType())) {
        indexValue = rewriter.create<arith::IndexCastOp>(
            op.getLoc(), rewriter.getIndexType(), idxOperand);
      } else {
        indexValue = idxOperand;
      }
    }

    // Get the memref from adaptor
    Value memref = adaptor.getQreg();

    // If we got an unrealized_conversion_cast wrapping the static memref,
    // unwrap it to get the actual static memref
    if (auto castOp = memref.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (castOp.getInputs().size() == 1) {
        memref = castOp.getInputs()[0];
      }
    }

    // Verify we got a static memref type as expected
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType || !memrefType.hasStaticShape()) {
      return op.emitError("Expected static memref type from alloc, got: ")
             << memref.getType();
    }

    // Create the new operation directly using the actual memref type
    auto loadOp = rewriter.create<memref::LoadOp>(
        op.getLoc(), qubitType, memref, ValueRange{indexValue});

    // quantum.extract only returns the extracted qubit, not a modified register
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
    // Get the index - either from attribute (compile-time) or operand (runtime)
    Value indexValue;
    auto idxAttr = op.getIdxAttrAttr();

    if (idxAttr) {
      // Compile-time constant index from attribute
      auto idx = idxAttr.getValue().getZExtValue();
      indexValue = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), idx);
    } else {
      // Runtime dynamic index from operand
      auto idxOperand = adaptor.getIdx();
      if (!idxOperand) {
        return op.emitError("InsertOp missing both idx_attr and idx operand");
      }

      // Convert i64 to index type if needed
      if (isa<IntegerType>(idxOperand.getType())) {
        indexValue = rewriter.create<arith::IndexCastOp>(
            op.getLoc(), rewriter.getIndexType(), idxOperand);
      } else {
        indexValue = idxOperand;
      }
    }

    // Get the memref from adaptor
    Value memref = adaptor.getInQreg();

    // If we got an unrealized_conversion_cast wrapping the static memref,
    // unwrap it to get the actual static memref
    if (auto castOp = memref.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (castOp.getInputs().size() == 1) {
        memref = castOp.getInputs()[0];
      }
    }

    // Create the new operation directly
    rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.getQubit(), memref,
                                     ValueRange{indexValue});

    // In the memref model, the quantum register is modified in-place,
    // so we replace the result with the memref (unchanged)
    rewriter.replaceOp(op, memref);
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
    auto gateName = op.getGateName();
    auto paramsValues = adaptor.getParams();
    auto inQubits = adaptor.getInQubits();
    auto inCtrlQubits = adaptor.getInCtrlQubits();
    auto inCtrlValues = adaptor.getInCtrlValues();

    // Convert to SmallVector for manipulation
    SmallVector<Value> inPosCtrlQubitsVec;
    SmallVector<Value> inNegCtrlQubitsVec;

    // Derive positive and negative control qubits from existing control qubits
    for (size_t i = 0; i < inCtrlQubits.size(); ++i) {
      if (inCtrlValues[i]) {
        inPosCtrlQubitsVec.emplace_back(inCtrlQubits[i]);
      } else {
        inNegCtrlQubitsVec.emplace_back(inCtrlQubits[i]);
      }
    }

    SmallVector<bool> paramsMaskVec;
    SmallVector<double> staticParamsVec;
    SmallVector<Value> finalParamValues;

    // Read attributes
    auto maskAttr = op->getAttrOfType<DenseBoolArrayAttr>("params_mask");
    auto staticParamsAttr =
        op->getAttrOfType<DenseF64ArrayAttr>("static_params");

    // Total length of combined parameter list
    size_t totalParams = 0;
    if (maskAttr) {
      totalParams = maskAttr.size();
    } else {
      totalParams = staticParamsAttr
                        ? staticParamsAttr.size() + paramsValues.size()
                        : paramsValues.size();
    }

    // Pointers to step through static/dynamic values
    size_t staticIdx = 0;
    size_t dynamicIdx = 0;

    // Build final mask + values in order
    for (size_t i = 0; i < totalParams; ++i) {
      bool const isStatic = (maskAttr ? maskAttr[i] : false);

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

    auto staticParams =
        DenseF64ArrayAttr::get(rewriter.getContext(), staticParamsVec);
    auto paramsMask =
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
    } else if (gateName == "RX" || gateName == "CRX") {
      mqtoptOp = CREATE_GATE_OP(RX);
    } else if (gateName == "RY" || gateName == "CRY") {
      mqtoptOp = CREATE_GATE_OP(RY);
    } else if (gateName == "RZ" || gateName == "CRZ") {
      mqtoptOp = CREATE_GATE_OP(RZ);
    } else if (gateName == "PhaseShift" || gateName == "ControlledPhaseShift") {
      mqtoptOp = CREATE_GATE_OP(P);
    } else if (gateName == "IsingXY") {
      mqtoptOp = CREATE_GATE_OP(XXplusYY);
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

    // Mark operations legal, that have no equivalent in the target dialect
    // TODO: how to handle them properly?
    target.addLegalOp<
        catalyst::quantum::DeviceInitOp, catalyst::quantum::DeviceReleaseOp,
        catalyst::quantum::NamedObsOp, catalyst::quantum::ExpvalOp,
        catalyst::quantum::FinalizeOp, catalyst::quantum::ComputationalBasisOp,
        catalyst::quantum::StateOp, catalyst::quantum::InitializeOp>();

    RewritePatternSet patterns(context);
    const CatalystQuantumToMQTOptTypeConverter typeConverter(context);

    patterns.add<ConvertQuantumAlloc, ConvertQuantumDealloc,
                 ConvertQuantumExtract, ConvertQuantumMeasure,
                 ConvertQuantumInsert, ConvertQuantumCustomOp>(typeConverter,
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

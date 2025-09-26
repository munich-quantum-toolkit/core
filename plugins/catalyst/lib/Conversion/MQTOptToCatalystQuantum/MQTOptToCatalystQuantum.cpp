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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

#include <Quantum/IR/QuantumDialect.h>
#include <Quantum/IR/QuantumOps.h>
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

#define GEN_PASS_DEF_MQTOPTTOCATALYSTQUANTUM
#include "mlir/Conversion/MQTOptToCatalystQuantum/MQTOptToCatalystQuantum.h.inc"

using namespace mlir;
using namespace mlir::arith;

// Helper functions to reduce code duplication
namespace {

/// Helper struct to hold control qubit information
struct ControlInfo {
  SmallVector<Value> ctrlQubits;
  SmallVector<Value> ctrlValues;
};

/// Extract and concatenate control qubits and create corresponding control
/// values
ControlInfo extractControlInfo(ValueRange posCtrlQubits,
                               ValueRange negCtrlQubits,
                               ConversionPatternRewriter& rewriter,
                               Location loc) {
  ControlInfo info;

  // Concatenate controls: [pos..., neg...]  (preserve this order consistently)
  info.ctrlQubits.reserve(posCtrlQubits.size() + negCtrlQubits.size());
  info.ctrlQubits.append(posCtrlQubits.begin(), posCtrlQubits.end());
  info.ctrlQubits.append(negCtrlQubits.begin(), negCtrlQubits.end());

  if (info.ctrlQubits.empty()) {
    return info;
  }

  // Create control values: 1 for positive controls, 0 for negative controls
  Value one = rewriter.create<mlir::arith::ConstantIntOp>(loc, /*value=*/1,
                                                          /*width=*/1);
  Value zero = rewriter.create<mlir::arith::ConstantIntOp>(loc, /*value=*/0,
                                                           /*width=*/1);

  info.ctrlValues.reserve(info.ctrlQubits.size());
  info.ctrlValues.append(posCtrlQubits.size(), one);  // +controls => 1
  info.ctrlValues.append(negCtrlQubits.size(), zero); // -controls => 0

  return info;
}

/// Helper function to extract operands and control info - for more complex
/// cases
template <typename OpAdaptor> struct ExtractedOperands {
  ValueRange inQubits;
  ControlInfo ctrlInfo;
};

template <typename OpAdaptor>
ExtractedOperands<OpAdaptor>
extractOperands(OpAdaptor adaptor, ConversionPatternRewriter& rewriter,
                Location loc) {
  ValueRange inQubits = adaptor.getInQubits();
  ValueRange posCtrlQubits = adaptor.getPosCtrlInQubits();
  ValueRange negCtrlQubits = adaptor.getNegCtrlInQubits();

  ControlInfo ctrlInfo =
      extractControlInfo(posCtrlQubits, negCtrlQubits, rewriter, loc);

  return {inQubits, ctrlInfo};
}

} // anonymous namespace

class MQTOptToCatalystQuantumTypeConverter final : public TypeConverter {
public:
  explicit MQTOptToCatalystQuantumTypeConverter(MLIRContext* ctx) {
    // Identity conversion: Allow all types to pass through unmodified if
    // needed.
    addConversion([](const Type type) { return type; });

    // Convert source MemRefType<QubitType> to target QuregType
    addConversion([ctx](MemRefType memrefType) -> Type {
      if (auto qubitType =
              dyn_cast<opt::QubitType>(memrefType.getElementType())) {
        return catalyst::quantum::QuregType::get(ctx);
      }
      return memrefType; // Pass through non-qubit memrefs
    });

    // Convert source QubitType to target QubitType
    addConversion([ctx](opt::QubitType /*type*/) -> Type {
      return catalyst::quantum::QubitType::get(ctx);
    });
  }
};

struct ConvertMQTOptAlloc final : OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Only convert memrefs of qubit type
    auto memrefType = cast<MemRefType>(op.getType());
    if (!isa<opt::QubitType>(memrefType.getElementType())) {
      return failure();
    }

    // Prepare the result type(s)
    const auto resultType =
        catalyst::quantum::QuregType::get(rewriter.getContext());

    // Get the size from memref type or dynamic operands
    Value size = nullptr;

    // Check if this is a statically shaped memref
    if (memrefType.hasStaticShape() && memrefType.getNumElements() >= 0) {
      // For static memref like memref<2x!mqtopt.Qubit>, create constant size
      auto numQubits = memrefType.getNumElements();
      auto sizeAttr = rewriter.getI64IntegerAttr(numQubits);
      size = rewriter.create<arith::ConstantOp>(op.getLoc(), sizeAttr);
    } else {
      // For dynamic memref, get size from dynamic operands
      auto dynamicOperands = adaptor.getDynamicSizes();
      size = dynamicOperands.empty() ? nullptr : dynamicOperands[0];
    }

    // Replace with quantum alloc operation
    rewriter.replaceOpWithNewOp<catalyst::quantum::AllocOp>(op, resultType,
                                                            size, nullptr);

    return success();
  }
};

struct ConvertMQTOptDealloc final : OpConversionPattern<memref::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Only convert memrefs of qubit type
    auto memrefType = cast<MemRefType>(op.getMemref().getType());
    if (!isa<opt::QubitType>(memrefType.getElementType())) {
      return failure();
    }

    // Create the new operation
    const auto catalystOp = rewriter.create<catalyst::quantum::DeallocOp>(
        op.getLoc(), TypeRange({}), adaptor.getMemref());

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
    auto inQubit = adaptor.getInQubit();

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

struct ConvertMQTOptLoad final : OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Only convert loads of qubit type
    if (!isa<opt::QubitType>(op.getType())) {
      return failure();
    }

    // Prepare the result type(s)
    auto resultType = catalyst::quantum::QubitType::get(rewriter.getContext());

    // Get index (assuming single index for 1D memref)
    auto indices = adaptor.getIndices();
    Value index = indices.empty() ? nullptr : indices[0];

    // Convert index type to i64 if needed
    if (index && index.getType().isa<IndexType>()) {
      index = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                                  rewriter.getI64Type(), index);
    }

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::ExtractOp>(
        op.getLoc(), resultType, adaptor.getMemref(), index, nullptr);

    // Replace the load operation with the extracted qubit
    rewriter.replaceOp(op, catalystOp.getResult());
    return success();
  }
};

struct ConvertMQTOptStore final : OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Only convert stores to memrefs with qubit element type
    auto memrefType = cast<MemRefType>(op.getMemRef().getType());
    if (!isa<opt::QubitType>(memrefType.getElementType())) {
      return failure();
    }

    // Get indices (assuming single index for 1D memref)
    auto indices = adaptor.getIndices();
    Value index = indices.empty() ? nullptr : indices[0];

    // Convert index type to i64 if needed
    if (index && index.getType().isa<IndexType>()) {
      index = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                                  rewriter.getI64Type(), index);
    }

    // Prepare the result type(s)
    auto resultType = catalyst::quantum::QuregType::get(rewriter.getContext());

    // Create the new operation
    rewriter.create<catalyst::quantum::InsertOp>(op.getLoc(), resultType,
                                                 adaptor.getMemref(), index,
                                                 nullptr, adaptor.getValue());

    // Erase the original store operation (store has no results to replace)
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename MQTGateOp>
struct ConvertMQTOptSimpleGate final : OpConversionPattern<MQTGateOp> {
  using OpConversionPattern<MQTGateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateOp op, typename MQTGateOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // BarrierOp has no semantic effect.
    if (std::is_same_v<MQTGateOp, opt::BarrierOp>) {
      rewriter.eraseOp(op);
      return success();
    }

    // Extract operands and control information using helper function
    auto extracted = extractOperands(adaptor, rewriter, op.getLoc());

    // Gate name may depend on number of controls
    const StringRef gateName =
        getGateName(extracted.ctrlInfo.ctrlQubits.size());
    if (gateName.empty()) {
      op->emitError() << "Unsupported controlled gate for op: "
                      << op->getName();
      return failure();
    }

    // Sanity: lengths must match, or the op verifier will complain.
    if (extracted.ctrlInfo.ctrlQubits.size() !=
        extracted.ctrlInfo.ctrlValues.size()) {
      op->emitError() << "control qubits and control values size mismatch";
      return failure();
    }

    // Create CustomOp
    auto custom = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/gateName,
        /*in_qubits=*/extracted.inQubits,
        /*in_ctrl_qubits=*/extracted.ctrlInfo.ctrlQubits,
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/adaptor.getParams(),
        /*adjoint=*/false);

    // ---- Replace: CustomOp results are (out_qubits, out_ctrl_qubits) ----
    SmallVector<Value> replacements;
    replacements.append(custom.getOutQubits().begin(),
                        custom.getOutQubits().end());
    replacements.append(custom.getOutCtrlQubits().begin(),
                        custom.getOutCtrlQubits().end());

    rewriter.replaceOp(op, replacements);
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
    // Get the base gate name and whether it is an adjoint version
    const auto& [gateName, adjoint] = getGateInfo<MQTGateOp>();

    // Extract control information
    ControlInfo ctrlInfo =
        extractControlInfo(adaptor.getPosCtrlInQubits(),
                           adaptor.getNegCtrlInQubits(), rewriter, op.getLoc());

    // Create CustomOp with adjoint flag
    auto catalystOp = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/gateName,
        /*in_qubits=*/adaptor.getInQubits(),
        /*in_ctrl_qubits=*/ctrlInfo.ctrlQubits,
        /*in_ctrl_values=*/ctrlInfo.ctrlValues,
        /*params=*/adaptor.getParams(),
        /*adjoint=*/adjoint);

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
    // Extract operands and control information using helper function
    auto extracted = extractOperands(adaptor, rewriter, op.getLoc());

    // V = RZ(π/2) RY(π/2) RZ(-π/2)
    auto pi2 = rewriter.create<ConstantOp>(op.getLoc(),
                                           rewriter.getF64FloatAttr(M_PI_2));

    // Create the decomposed operations
    auto rz1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate_name=*/"RZ",
        /*in_qubits=*/extracted.inQubits,
        /*in_ctrl_qubits=*/extracted.ctrlInfo.ctrlQubits,
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{pi2},
        /*adjoint=*/false);

    auto ry = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate_name=*/"RY",
        /*in_qubits=*/rz1.getOutQubits(),
        /*in_ctrl_qubits=*/rz1.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{pi2},
        /*adjoint=*/false);

    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate_name=*/"RZ",
        /*in_qubits=*/ry.getOutQubits(),
        /*in_ctrl_qubits=*/ry.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{pi2},
        /*adjoint=*/true);

    // ---- Replace: CustomOp results are (out_qubits, out_ctrl_qubits) ----
    SmallVector<Value> replacements;
    replacements.append(rz2.getOutQubits().begin(), rz2.getOutQubits().end());
    replacements.append(rz2.getOutCtrlQubits().begin(),
                        rz2.getOutCtrlQubits().end());

    rewriter.replaceOp(op, replacements);
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
    // Extract operands and control information using helper function
    auto extracted = extractOperands(adaptor, rewriter, op.getLoc());

    // V† = RZ(π/2) RY(-π/2) RZ(-π/2)
    auto negPi2 = rewriter.create<ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(-M_PI_2));

    // Create the decomposed operations
    auto rz1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"RZ",
        /*in_qubits=*/extracted.inQubits,
        /*in_ctrl_qubits=*/extracted.ctrlInfo.ctrlQubits,
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{negPi2},
        /*adjoint=*/true);

    auto ry = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"RY",
        /*in_qubits=*/rz1.getOutQubits(),
        /*in_ctrl_qubits=*/rz1.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{negPi2},
        /*adjoint=*/false);

    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"RZ",
        /*in_qubits=*/ry.getOutQubits(),
        /*in_ctrl_qubits=*/ry.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{negPi2},
        /*adjoint=*/false);

    // ---- Replace: CustomOp results are (out_qubits, out_ctrl_qubits) ----
    SmallVector<Value> replacements;
    replacements.append(rz2.getOutQubits().begin(), rz2.getOutQubits().end());
    replacements.append(rz2.getOutCtrlQubits().begin(),
                        rz2.getOutCtrlQubits().end());

    rewriter.replaceOp(op, replacements);
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
    // Extract operands and control information using helper function
    auto extracted = extractOperands(adaptor, rewriter, op.getLoc());

    // DCX = CNOT(q2,q1) CNOT(q1,q2)
    auto cnot1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"CNOT",
        /*in_qubits=*/extracted.inQubits,
        /*in_ctrl_qubits=*/extracted.ctrlInfo.ctrlQubits,
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{},
        /*adjoint=*/false);

    auto cnot2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"CNOT",
        /*in_qubits=*/
        ValueRange{cnot1.getOutQubits()[1], cnot1.getOutQubits()[0]},
        /*in_ctrl_qubits=*/cnot1.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{},
        /*adjoint=*/false);

    // ---- Replace: CustomOp results are (out_qubits, out_ctrl_qubits) ----
    SmallVector<Value> replacements;
    replacements.append(cnot2.getOutQubits().begin(),
                        cnot2.getOutQubits().end());
    replacements.append(cnot2.getOutCtrlQubits().begin(),
                        cnot2.getOutCtrlQubits().end());

    rewriter.replaceOp(op, replacements);
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
    // Extract operands and control information using helper function
    auto extracted = extractOperands(adaptor, rewriter, op.getLoc());

    // RZX(q0, q1; θ) = H(q1) · RZZ(q0, q1; θ) · H(q1)
    // H gates can stay uncontrolled, as they cancel if control on RZZ is not
    // active

    // H on q1
    auto h1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"Hadamard",
        /*in_qubits=*/ValueRange{extracted.inQubits[1]},
        /*in_ctrl_qubits=*/extracted.ctrlInfo.ctrlQubits,
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{},
        /*adjoint=*/false);

    // RZZ on (q0, q1')
    auto rzz = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"IsingZZ",
        /*in_qubits=*/ValueRange{extracted.inQubits[0], h1.getOutQubits()[0]},
        /*in_ctrl_qubits=*/h1.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/adaptor.getParams()[0],
        /*adjoint=*/false);

    // H on q1''
    auto h2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"Hadamard",
        /*in_qubits=*/ValueRange{rzz.getOutQubits()[1]},
        /*in_ctrl_qubits=*/rzz.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{},
        /*adjoint=*/false);

    // Final results in mqt.opt ordering (targets..., controls...)
    SmallVector<Value> finalResults;
    finalResults.push_back(rzz.getOutQubits()[0]); // target0 final
    finalResults.push_back(h2.getOutQubits()[0]);  // target1 final
    finalResults.append(h2.getOutCtrlQubits().begin(),
                        h2.getOutCtrlQubits().end()); // controls

    rewriter.replaceOp(op, finalResults);
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<opt::GPhaseOp> final
    : OpConversionPattern<opt::GPhaseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::GPhaseOp op, opt::GPhaseOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract control information using helper function (no input qubits for
    // GPhase)
    auto ctrlInfo =
        extractControlInfo(adaptor.getPosCtrlInQubits(),
                           adaptor.getNegCtrlInQubits(), rewriter, op.getLoc());
    const auto params = adaptor.getParams();

    // Create output types for GlobalPhaseOp (control qubits only)
    const Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    SmallVector<Type> outCtrlTypes(ctrlInfo.ctrlQubits.size(), qubitType);

    auto gphase = rewriter.create<catalyst::quantum::GlobalPhaseOp>(
        op.getLoc(), TypeRange(outCtrlTypes), params[0], false,
        ctrlInfo.ctrlQubits, ctrlInfo.ctrlValues);

    // Replace the original operation with the decomposition
    rewriter.replaceOp(op, gphase.getResults());
    return success();
  }
};

template <>
struct ConvertMQTOptSimpleGate<opt::UOp> final : OpConversionPattern<opt::UOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::UOp op, opt::UOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operands and control information using helper function
    auto extracted = extractOperands(adaptor, rewriter, op.getLoc());

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
        op.getLoc(),
        /*gate=*/"RZ",
        /*in_qubits=*/extracted.inQubits,
        /*in_ctrl_qubits=*/extracted.ctrlInfo.ctrlQubits,
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{lambdaMinusPi2},
        /*adjoint=*/false);

    // RX(π/2)
    auto rx1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"RX",
        /*in_qubits=*/rz1.getOutQubits(),
        /*in_ctrl_qubits=*/rz1.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{pi2},
        /*adjoint=*/false);

    // RZ(π − θ)
    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"RZ",
        /*in_qubits=*/rx1.getOutQubits(),
        /*in_ctrl_qubits=*/rx1.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{piMinusTheta2},
        /*adjoint=*/false);

    // RX(π/2)
    auto rx2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"RX",
        /*in_qubits=*/rz2.getOutQubits(),
        /*in_ctrl_qubits=*/rz2.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{pi2},
        /*adjoint=*/false);

    // RZ(φ − π/2)
    auto rz3 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"RZ",
        /*in_qubits=*/rx2.getOutQubits(),
        /*in_ctrl_qubits=*/rx2.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{phiMinusPi2},
        /*adjoint=*/false);

    // ---- Replace: CustomOp results are (out_qubits, out_ctrl_qubits) ----
    SmallVector<Value> replacements;
    replacements.append(rz3.getOutQubits().begin(), rz3.getOutQubits().end());
    replacements.append(rz3.getOutCtrlQubits().begin(),
                        rz3.getOutCtrlQubits().end());

    // Replace the original U gate with the decomposed sequence
    rewriter.replaceOp(op, replacements);
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
    // Extract operands and control information using helper function
    auto extracted = extractOperands(adaptor, rewriter, op.getLoc());

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
        op.getLoc(),
        /*gate=*/"RZ",
        /*in_qubits=*/extracted.inQubits,
        /*in_ctrl_qubits=*/extracted.ctrlInfo.ctrlQubits,
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{lambdaMinusPi2},
        /*adjoint=*/false);

    // RX(π/2)
    auto rx1 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"RX",
        /*in_qubits=*/rz1.getOutQubits(),
        /*in_ctrl_qubits=*/rz1.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{pi2},
        /*adjoint=*/false);

    // RZ(3/4 π)
    auto rz2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"RZ",
        /*in_qubits=*/rx1.getOutQubits(),
        /*in_ctrl_qubits=*/rx1.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{pi34},
        /*adjoint=*/false);

    // RX(π/2)
    auto rx2 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"RX",
        /*in_qubits=*/rz2.getOutQubits(),
        /*in_ctrl_qubits=*/rz2.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{pi2},
        /*adjoint=*/false);

    // RZ(φ − π/2)
    auto rz3 = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"RZ",
        /*in_qubits=*/rx2.getOutQubits(),
        /*in_ctrl_qubits=*/rx2.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{phiMinusPi2},
        /*adjoint=*/false);

    // ---- Replace: CustomOp results are (out_qubits, out_ctrl_qubits) ----
    SmallVector<Value> replacements;
    replacements.append(rz3.getOutQubits().begin(), rz3.getOutQubits().end());
    replacements.append(rz3.getOutCtrlQubits().begin(),
                        rz3.getOutCtrlQubits().end());

    // Replace the original U gate with the decomposed sequence
    rewriter.replaceOp(op, replacements);
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
    // Extract operands and control information using helper function
    auto extracted = extractOperands(adaptor, rewriter, op.getLoc());

    // Peres = CNOT(q0, q1) ; X(q0)

    // CNOT(q0, q1)
    auto cnot = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"CNOT",
        /*in_qubits=*/ValueRange{extracted.inQubits[0], extracted.inQubits[1]},
        /*in_ctrl_qubits=*/extracted.ctrlInfo.ctrlQubits,
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{},
        /*adjoint=*/false);

    Value q0_after_cnot = cnot.getOutQubits()[0];
    Value q1_after_cnot = cnot.getOutQubits()[1];

    // X(q0')
    auto x = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"PauliX",
        /*in_qubits=*/ValueRange{q0_after_cnot},
        /*in_ctrl_qubits=*/cnot.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{},
        /*adjoint=*/false);

    Value q0_final = x.getOutQubits()[0]; // target0
    Value q1_final = q1_after_cnot;       // target1

    // Final: (targets..., controls...)
    SmallVector<Value> finalResults;
    finalResults.push_back(q0_final);
    finalResults.push_back(q1_final);
    finalResults.append(x.getOutCtrlQubits().begin(),
                        x.getOutCtrlQubits().end());

    rewriter.replaceOp(op, finalResults);
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
    // Extract operands and control information using helper function
    auto extracted = extractOperands(adaptor, rewriter, op.getLoc());

    // Peres† = X(q0) ; CNOT(q0, q1)

    // X(q0)
    auto x = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"PauliX",
        /*in_qubits=*/ValueRange{extracted.inQubits[0]},
        /*in_ctrl_qubits=*/extracted.ctrlInfo.ctrlQubits,
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{},
        /*adjoint=*/false);

    Value q0_after_x = x.getOutQubits()[0];

    // CNOT(q0', q1)
    auto cnot = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(),
        /*gate=*/"CNOT",
        /*in_qubits=*/ValueRange{q0_after_x, extracted.inQubits[1]},
        /*in_ctrl_qubits=*/x.getOutCtrlQubits(),
        /*in_ctrl_values=*/extracted.ctrlInfo.ctrlValues,
        /*params=*/ValueRange{},
        /*adjoint=*/false);

    // Final: (targets..., controls...)
    SmallVector<Value> finalResults;
    finalResults.push_back(cnot.getOutQubits()[0]); // q0_final
    finalResults.push_back(cnot.getOutQubits()[1]); // q1_final
    finalResults.append(cnot.getOutCtrlQubits().begin(),
                        cnot.getOutCtrlQubits().end()); // controls

    rewriter.replaceOp(op, finalResults);
    return success();
  }
};

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
StringRef ConvertMQTOptSimpleGate<opt::XXplusYYOp>::getGateName(
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
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addIllegalDialect<opt::MQTOptDialect>();

    // Mark memref operations on qubits as illegal to trigger conversion
    target.addDynamicallyLegalOp<memref::AllocOp>([](memref::AllocOp op) {
      auto memrefType = dyn_cast<MemRefType>(op.getType());
      if (!memrefType)
        return true;
      auto elementType = memrefType.getElementType();
      return !isa<opt::QubitType>(elementType);
    });

    target.addDynamicallyLegalOp<memref::DeallocOp>([](memref::DeallocOp op) {
      auto memrefType = dyn_cast<MemRefType>(op.getMemref().getType());
      if (!memrefType)
        return true;
      auto elementType = memrefType.getElementType();
      return !isa<opt::QubitType>(elementType);
    });

    target.addDynamicallyLegalOp<memref::LoadOp>([](memref::LoadOp op) {
      auto memrefType = dyn_cast<MemRefType>(op.getMemRef().getType());
      if (!memrefType)
        return true;
      auto elementType = memrefType.getElementType();
      return !isa<opt::QubitType>(elementType);
    });

    target.addDynamicallyLegalOp<memref::StoreOp>([](memref::StoreOp op) {
      auto memrefType = dyn_cast<MemRefType>(op.getMemRef().getType());
      if (!memrefType)
        return true;
      auto elementType = memrefType.getElementType();
      return !isa<opt::QubitType>(elementType);
    });

    RewritePatternSet patterns(context);
    MQTOptToCatalystQuantumTypeConverter typeConverter(context);

    patterns.add<ConvertMQTOptAlloc, ConvertMQTOptDealloc, ConvertMQTOptLoad,
                 ConvertMQTOptMeasure, ConvertMQTOptStore>(typeConverter,
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
    patterns.add<ConvertMQTOptSimpleGate<opt::XXplusYYOp>>(typeConverter,
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

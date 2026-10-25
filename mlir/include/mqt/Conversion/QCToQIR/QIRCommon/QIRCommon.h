/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/QIR/Utils/QIRUtils.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace mlir {

/// Register dialects and bufferization interfaces for constant tensor reads.
void registerQIRClassicalTensorDialects(DialectRegistry& registry);

/// Qubit allocation mode
enum class AllocationMode : std::uint8_t {
  Unset,   //!< No allocation mode has been established yet.
  Static,  //!< The module uses static qubit allocation.
  Dynamic, //!< The module uses dynamic qubit allocation.
};

/// State object for tracking lowering information during QIR conversion
struct LoweringState {
  std::optional<CompilerTarget> nativeTarget;

  /// Select the advertised name using converted physical qubits and parameters.
  [[nodiscard]] CompilerTarget::OperationMatch
  nativeOperation(StringRef name, ValueRange qubits, ValueRange parameters,
                  size_t numControls = 0) const;

  /// Retain a native QIS name until profile cleanup has finished.
  [[nodiscard]] LogicalResult nameNativeCall(LLVM::CallOp call, StringRef name,
                                             ValueRange qubits) const;

  /// Result-array pointers to be deallocated at the end of the program
  llvm::SmallSetVector<Value, 4> resultArrays;

  /// CBit read operations whose register is backed by a result array.
  DenseSet<Operation*> returnedCBitReads;

  /// Cache static qubit pointers for reuse
  DenseMap<int64_t, Value> staticQubits;

  /// Canonical Base-profile pointers for constant qubit-register elements.
  DenseMap<std::pair<Value, int64_t>, Value> staticRegisterQubits;

  /// Cache qubit register sizes for reuse
  DenseMap<Value, Value> qregSizes;

  /// Classical registers owned by the lowering state.
  SmallVector<qir::ClassicalRegister> cregs;

  /// Map from `cbit::AllocOp` to its index in `cregs`.
  DenseMap<Operation*, size_t> cregIndices;

  /// Resource definitions or scalar return indices in output order.
  SmallVector<std::variant<Operation*, int64_t>> returnedOutputs;

  /// Result pointers for returned measurements, resolved during conversion.
  DenseMap<Operation*, qir::OutputRecord> measuredResults;

  /// Destination register index and bit index of each stored measurement.
  DenseMap<Operation*, std::pair<size_t, Value>> cregMeasurements;

  /// Measurement stores kept as SSA users across structural conversion.
  DenseMap<Operation*, Operation*> deferredMeasurementStores;

  /// Owned scalar results in allocation order; array elements are owned
  /// separately.
  SmallVector<Value> scalarResults;

  /// Reusable Adaptive measurement result, owned by `scalarResults`.
  Value scratchResult;

  /// Converted controls associated with their specific body unitary.
  DenseMap<Operation*, SmallVector<Value>> controlledGates;

  /// Block information
  Block* entryBlock{};
  Block* measurementsBlock{};
  Block* outputBlock{};

  /// The qubit allocation mode used in the module
  AllocationMode allocationMode = AllocationMode::Unset;

  /// Sets or validates the allocation mode, or emits an error if it conflicts.
  [[nodiscard]] LogicalResult ensureAllocationMode(AllocationMode requestedMode,
                                                   Operation* op);
};

/// Lower remaining classical dialects and reconcile casts for either profile.
[[nodiscard]] LogicalResult
finalizeQIRConversion(ModuleOp moduleOp, ConversionTarget& target,
                      LLVMTypeConverter& typeConverter);

struct QCToQIRTypeConverter final : LLVMTypeConverter {
  explicit QCToQIRTypeConverter(MLIRContext* ctx);
};

/// Base class for conversion patterns that need access to lowering state
///
/// Extends OpConversionPattern to provide access to a shared LoweringState
/// object, which tracks qubit/result counts and caches values across multiple
/// pattern applications.
///
/// @tparam OpType The operation type to convert
template <typename OpType>
class StatefulOpConversionPattern : public OpConversionPattern<OpType> {
public:
  StatefulOpConversionPattern(TypeConverter& tc, MLIRContext* ctx,
                              LoweringState* state)
      : OpConversionPattern<OpType>(tc, ctx), state_(state) {}

  [[nodiscard]] LoweringState& getState() const { return *state_; }

private:
  LoweringState* state_;
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

/// Adds QIR initialization call to the entry block
///
/// This QIR runtime function initializes the quantum execution environment.
///
/// @param main The main LLVM function
/// @param ctx The MLIR context
/// @param state The lowering state
void addInitialize(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                   LoweringState& state);

/// Populates common conversion patterns for QC-to-QIR lowering.
///
/// Centralizes pattern registration so adding a new QC gate typically only
/// requires adding a new `ConvertQCUnitaryOpQIR<...>` specialization.
void populateQCToQIRPatterns(RewritePatternSet& patterns,
                             QCToQIRTypeConverter& typeConverter,
                             MLIRContext* ctx, LoweringState& state);

/// Record entry-function outputs in return order, independently of ownership.
///
/// Measurement registers use result-array recording; computed registers use
/// an array record followed by Boolean records. Scalar results and repeated
/// references retain their positions in the output stream.
void addOutputRecording(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                        LoweringState& state);

/// Prepares classical result registers before func-to-LLVM conversion.
///
/// Requires a single entry-function return. On failure, discard \p state.
///
/// For measurement-only returned registers, the store and measurement must
/// share a block. The index must be available at measurement or become
/// available by moving pure, speculatable operations without regions from that
/// block before it. Removes fused stores so QIR measurements write directly to
/// their destinations. Intervening operations must be effect-free, affect only
/// quantum resources, or store to a provably distinct constant index of the
/// same register.
///
/// With \p allowComputedOutputs, returned registers use Boolean storage when
/// they contain computed bits or fusion cannot preserve their stores and
/// returned measurement snapshots. Otherwise these cases fail.
/// Boolean scalar outputs remain function results until output recording.
/// Local CBit stores retain ordinary semantics.
///
/// Call while `func::ReturnOp`, `qc::MeasureOp`, and `cbit::StoreOp` remain in
/// the IR. Adaptive lowering defers fused store removal until SCF and function
/// conversion have rewritten their index operands.
[[nodiscard]] LogicalResult
prepareClassicalResults(Operation* moduleOp, LoweringState& state,
                        bool allowComputedOutputs = false,
                        bool deferMeasurementStores = false);

/// Refresh measurement indices from surviving stores and consume those stores.
void finalizeClassicalResults(LoweringState& state);

/// Returns a result pointer for a measurement that does not write into a
/// returned classical bit register
Value getResultPtr(LoweringState& state, Operation* op,
                   ConversionPatternRewriter& rewriter, bool dynamic);

} // namespace mlir

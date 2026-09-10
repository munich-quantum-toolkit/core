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

#include "mqt/Dialect/QIR/Utils/QIRUtils.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace mlir {

/// Qubit allocation mode
enum class AllocationMode : std::uint8_t {
  Unset,   //!< No allocation mode has been established yet.
  Static,  //!< The module uses static qubit allocation.
  Dynamic, //!< The module uses dynamic qubit allocation.
};

/// State object for tracking lowering information during QIR conversion
struct LoweringState {
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

  /// Returned classical-register indices in function-result order.
  SmallVector<size_t> returnedCregs;

  /// Destination register index and bit index of each stored measurement.
  DenseMap<Operation*, std::pair<size_t, Value>> cregMeasurements;

  /// Indexed scalar results, dynamically allocated in Adaptive and static in
  /// Base.
  DenseMap<int64_t, qir::StaticResult> scalarResults;

  /// Metadata for returned scalar measurement results. Each entry is a defining
  /// `qc::MeasureOp`
  DenseSet<Operation*> returnedScalarResults;

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

/// Adds output recording calls to the output block
///
/// Generates output recording calls in the output block based on the
/// measurements tracked during conversion. Follows the QIR specification for
/// labeled output schema.
///
/// Results that are part of registers are recorded via
/// `__quantum__rt__result_array_record_output`.
///
/// Results that are not part of registers (i.e., measurements without register
/// info) are grouped under a default `__unnamed__` label recorded via
/// `__quantum__rt__result_record_output`.
///
/// @param main The main LLVM function
/// @param ctx The MLIR context
/// @param state The lowering state whose returned register records are consumed
void addOutputRecording(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                        LoweringState& state);

/// Prepares classical result registers for QIR conversion
///
/// Requires a single entry-function return. Inventories classical result
/// registers and validates output stores before rewriting returns or stores.
/// A returned-register store must share a block with its measurement and use
/// an index available there (or a constant). Intervening operations must be
/// effect-free, affect only quantum resources, or store to a provably distinct
/// constant index of the same register. The QIR measurement can then write
/// directly to the destination without changing observable order or control
/// flow. Other stores to returned registers are rejected; local CBit stores
/// retain their ordinary semantics.
///
/// This must be called **before** func-to-LLVM conversion, while
/// `func::ReturnOp`, `qc::MeasureOp`, and `cbit::StoreOp` are still in the IR.
///
/// @param moduleOp The top-level module operation to walk
/// @param state The lowering state populated for profile-specific conversion
[[nodiscard]] LogicalResult prepareClassicalResults(Operation* moduleOp,
                                                    LoweringState& state);

/// Returns a result pointer for a measurement that does not write into a
/// returned classical bit register
Value getResultPtr(LoweringState& state, Operation* op,
                   ConversionPatternRewriter& rewriter, bool dynamic);

} // namespace mlir

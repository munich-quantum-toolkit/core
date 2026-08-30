/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCToQCO/QCToQCO.h"

#include "mlir/Conversion/ConversionUtils.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/RegionUtils.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir {

using namespace qco;
using namespace qc;

#define GEN_PASS_DEF_QCTOQCO
#include "mlir/Conversion/QCToQCO/QCToQCO.h.inc"

namespace {

using RegisterId = std::size_t;

/**
 * @brief Provenance for a register-backed QC qubit reference
 */
struct RegisterAccess {
  /// Stable identifier of the register the qubit belongs to
  RegisterId reg;
  /// Index of the qubit within its register
  Value index;
};

/** @brief Indices already used for one register by a quantum operation. */
struct SeenRegisterIndices {
  DenseMap<int64_t, Value> constants;
  llvm::SmallDenseSet<Value, 4> dynamicValues;
};

/** @brief Qubit allocation mode */
enum class AllocationMode : std::uint8_t {
  Unset,  //!< No allocation mode has been established yet.
  Static, //!< The module uses static qubit allocation.
  Dynamic //!< The module uses dynamic qubit allocation.
};

/**
 * @brief State object for tracking qubit value flow during conversion
 *
 * @details
 * This struct maintains the mapping between QC dialect qubits (which use
 * reference semantics) and their corresponding QCO dialect qubit values
 * (which use value semantics). As the conversion progresses, each QC
 * qubit reference is mapped to its latest QCO SSA value.
 *
 * The key insight is that QC operations modify qubits in-place:
 * ```mlir
 * %q = qc.alloc : !qc.qubit
 * qc.h %q : !qc.qubit        // modifies %q in-place
 * qc.x %q : !qc.qubit        // modifies %q in-place
 * ```
 *
 * While QCO operations consume inputs and produce new outputs:
 * ```mlir
 * %q0 = qco.alloc : !qco.qubit
 * %q1 = qco.h %q0 : !qco.qubit -> !qco.qubit   // %q0 consumed, %q1 produced
 * %q2 = qco.x %q1 : !qco.qubit -> !qco.qubit   // %q1 consumed, %q2 produced
 * ```
 *
 * The qubitMap tracks that the QC qubit %q corresponds to:
 * - %q0 after allocation
 * - %q1 after the H gate
 * - %q2 after the X gate
 */
struct LoweringState {
  struct StructuredValues {
    SmallVector<Value> qubits;
    SmallVector<RegisterId> registers;
  };

  /// Per-region map from original QC qubit reference to its latest QCO SSA
  /// value.
  ///
  /// @details Keys are `Operation::getParentRegion()` for ops being converted
  /// (typically a `func.func` body or a modifier region).
  DenseMap<Region*, DenseMap<Value, Value>> qubitMap;

  /// Per-region map from stable register identifiers to their latest QTensor
  /// SSA values.
  DenseMap<Region*, DenseMap<RegisterId, Value>> tensorMap;

  /// Transient map from source QC register values to stable identifiers.
  DenseMap<Value, RegisterId> registerIds;

  /// Transient provenance for register-backed QC qubit references.
  DenseMap<Value, RegisterAccess> registerAccesses;

  /// Transient canonical keys for modifier block arguments replaced by
  /// dialect-conversion signature conversion.
  DenseMap<Value, Value> convertedQubitAliases;

  /// Map from an operation to its used QC qubits inside its regions
  DenseMap<Operation*, SetVector<Value>> regionQubitMap;

  /// Map from an operation to the registers used inside its regions.
  DenseMap<Operation*, SetVector<RegisterId>> regionRegisterMap;

  /// Original QC value order for already-converted structured operations.
  /// Kept separate from the legality maps above so converted SCF operations
  /// remain legal.
  DenseMap<Operation*, StructuredValues> structuredValues;

  /// Qubit keys yielded by converted modifier regions, in yield order.
  DenseMap<Region*, SmallVector<Value>> modifierRegionQubits;

  /// The qubit allocation mode used in the module
  AllocationMode allocationMode = AllocationMode::Unset;

  /// Sets or validates the allocation mode, or emits an error if it conflicts.
  [[nodiscard]] LogicalResult ensureAllocationMode(AllocationMode requestedMode,
                                                   Operation* op) {
    if (allocationMode == AllocationMode::Unset) {
      allocationMode = requestedMode;
      return success();
    }
    if (allocationMode == requestedMode) {
      return success();
    }
    return op->emitOpError(
        "cannot mix static and dynamic qubit allocation modes in QC program");
  }
};

/**
 * @brief Base class for conversion patterns that need access to lowering state
 *
 * @details
 * Extends OpConversionPattern to provide access to a shared LoweringState
 * object, which tracks the mapping from reference-semantics QC qubits
 * to value-semantics QCO qubits across multiple pattern applications.
 *
 * This stateful approach is necessary because the conversion needs to:
 * 1. Track which QCO value corresponds to each QC qubit reference
 * 2. Update these mappings as operations transform qubits
 * 3. Share this information across different conversion patterns
 *
 * @tparam OpType The QC operation type to convert
 */
template <typename OpType>
class StatefulOpConversionPattern : public OpConversionPattern<OpType> {

public:
  StatefulOpConversionPattern(TypeConverter& typeConverter,
                              MLIRContext* context, LoweringState* state)
      : OpConversionPattern<OpType>(typeConverter, context), state_(state) {}

  /// Returns the shared lowering state object
  [[nodiscard]] LoweringState& getState() const { return *state_; }

private:
  LoweringState* state_;
};
} // namespace

/** @brief Returns whether a type is ranked or unranked QC qubit storage. */
[[nodiscard]] static bool isQubitMemrefType(const Type type) {
  const auto memref = dyn_cast<BaseMemRefType>(type);
  return memref && isa<qc::QubitType>(memref.getElementType());
}

/** @brief Resolves the stable identifier for a source QC register value. */
[[nodiscard]] static RegisterId lookupRegisterId(const LoweringState& state,
                                                 Value memref) {
  const auto it = state.registerIds.find(memref);
  assert(it != state.registerIds.end() && "QC register not found");
  return it->second;
}

/**
 * @brief Finds the nearest region-local map containing @p reference and
 * returns the pair containing the map and a mutable reference to the value in
 * the map.
 */
template <typename Key>
[[nodiscard]] static std::pair<DenseMap<Key, Value>*, Value*>
findRegionLocalMap(DenseMap<Region*, DenseMap<Key, Value>>& map,
                   Operation* anchor, Key reference) {
  for (auto* current = anchor->getParentRegion(); current != nullptr;
       current = current->getParentRegion()) {
    if (auto it = map.find(current); it != map.end()) {
      auto& regionMap = it->second;
      if (auto valueIt = regionMap.find(reference);
          valueIt != regionMap.end()) {
        return {&regionMap, &valueIt->second};
      }
      return {&regionMap, nullptr};
    }
  }
  return {nullptr, nullptr};
}

/** @brief Canonicalizes a source qubit key after block signature conversion. */
[[nodiscard]] static Value canonicalQubitKey(const LoweringState& state,
                                             Value qubit) {
  for (auto alias = state.convertedQubitAliases.find(qubit);
       alias != state.convertedQubitAliases.end();
       alias = state.convertedQubitAliases.find(qubit)) {
    qubit = alias->second;
  }
  return qubit;
}

/** @brief Resolves the latest QCO SSA value for a QC qubit reference. */
[[nodiscard]] static Value lookupMappedQubit(LoweringState& state,
                                             Operation* anchor, Value qcQubit) {
  qcQubit = canonicalQubitKey(state, qcQubit);
  const auto& [qubitMap, qubitValue] =
      findRegionLocalMap(state.qubitMap, anchor, qcQubit);
  assert(qubitMap != nullptr && qubitValue != nullptr && "QC qubit not found");
  return *qubitValue;
}

/** @brief Resolves the latest QTensor SSA value for a QC register. */
[[nodiscard]] static Value lookupMappedTensor(LoweringState& state,
                                              Operation* anchor,
                                              const RegisterId reg) {
  const auto& [tensorMap, tensorValue] =
      findRegionLocalMap(state.tensorMap, anchor, reg);
  assert(tensorMap != nullptr && tensorValue != nullptr &&
         "QC register not found");
  return *tensorValue;
}

/** @brief Updates the latest QCO SSA value for a QC qubit reference. */
static void assignMappedQubit(LoweringState& state, Operation* anchor,
                              Value qcQubit, Value qcoQubit) {
  qcQubit = canonicalQubitKey(state, qcQubit);
  auto [qubitMap, qubitValue] =
      findRegionLocalMap(state.qubitMap, anchor, qcQubit);
  if (qubitValue != nullptr) {
    *qubitValue = qcoQubit;
    return;
  }
  if (qubitMap != nullptr) {
    (*qubitMap)[qcQubit] = qcoQubit;
    return;
  }

  state.qubitMap[anchor->getParentRegion()][qcQubit] = qcoQubit;
}

/** @brief Updates the latest QTensor SSA value for a QC register. */
static void assignMappedTensor(LoweringState& state, Operation* anchor,
                               const RegisterId reg, Value tensor) {
  auto [tensorMap, tensorValue] =
      findRegionLocalMap(state.tensorMap, anchor, reg);

  if (tensorValue != nullptr) {
    *tensorValue = tensor;
    return;
  }
  if (tensorMap != nullptr) {
    (*tensorMap)[reg] = tensor;
    return;
  }
  state.tensorMap[anchor->getParentRegion()][reg] = tensor;
}

/** @brief Resolves a range of QC qubits to their latest QCO values. */
template <typename Range>
[[nodiscard]] static SmallVector<Value>
resolveMappedQubits(LoweringState& state, Operation* anchor,
                    const Range& qcQubits) {
  return llvm::to_vector(llvm::map_range(qcQubits, [&](Value qcQubit) {
    return lookupMappedQubit(state, anchor, qcQubit);
  }));
}

/** @brief Resolves a range of QC memrefs to their latest QTensor values. */
template <typename Range>
[[nodiscard]] static SmallVector<Value>
resolveMappedTensors(LoweringState& state, Operation* anchor,
                     const Range& registers) {
  return llvm::to_vector(llvm::map_range(registers, [&](RegisterId reg) {
    return lookupMappedTensor(state, anchor, reg);
  }));
}

/** @brief Updates mappings for matching QC and QCO qubit ranges. */
template <typename QcRange, typename QcoRange>
static void assignMappedQubits(LoweringState& state, Operation* anchor,
                               const QcRange& qcQubits, QcoRange qcoQubits) {
  for (auto [qcQubit, qcoQubit] : llvm::zip_equal(qcQubits, qcoQubits)) {
    assignMappedQubit(state, anchor, qcQubit, qcoQubit);
  }
}

/** @brief Updates mappings for matching QC memref and QTensor ranges. */
template <typename QcRange, typename QcoRange>
static void assignMappedTensors(LoweringState& state, Operation* anchor,
                                const QcRange& registers, QcoRange tensors) {
  for (auto [reg, tensor] : llvm::zip_equal(registers, tensors)) {
    assignMappedTensor(state, anchor, reg, tensor);
  }
}

/** @brief Returns the structured parent whose quantum values a terminator
 * yields. */
[[nodiscard]] static Operation* structuredValueOwner(Operation* operation) {
  if (isa<scf::YieldOp, scf::ConditionOp>(operation)) {
    return operation->getParentOp();
  }
  return operation;
}

/** @brief Seeds region-local QCO mappings for structured-control-flow block
 * arguments. */
static void seedRegionMappings(LoweringState& state, Region& region,
                               ValueRange qcQubits,
                               ArrayRef<RegisterId> registers,
                               ValueRange qcoQubits, ValueRange tensors) {
  auto& qubitMap = state.qubitMap[&region];
  for (auto [qcQubit, qcoQubit] : llvm::zip_equal(qcQubits, qcoQubits)) {
    qubitMap[canonicalQubitKey(state, qcQubit)] = qcoQubit;
  }
  auto& tensorMap = state.tensorMap[&region];
  for (auto [reg, tensor] : llvm::zip_equal(registers, tensors)) {
    tensorMap[reg] = tensor;
  }
}

/** @brief QCO operands and register provenance materialized for a QC op. */
namespace {
struct MaterializedQubits {
  SmallVector<Value> values;
  SmallVector<std::optional<RegisterAccess>> accesses;
};
} // namespace

/**
 * @brief Materializes register-backed qubits immediately before a quantum op.
 */
[[nodiscard]] static MaterializedQubits
materializeQubits(LoweringState& state, Operation* anchor, ValueRange qcQubits,
                  PatternRewriter& rewriter) {
  MaterializedQubits materialized;
  materialized.values.reserve(qcQubits.size());
  materialized.accesses.reserve(qcQubits.size());

  for (auto qcQubit : qcQubits) {
    const auto accessIt = state.registerAccesses.find(qcQubit);
    if (accessIt == state.registerAccesses.end()) {
      materialized.values.push_back(lookupMappedQubit(state, anchor, qcQubit));
      materialized.accesses.emplace_back();
      continue;
    }

    const auto access = accessIt->second;
    auto tensor = lookupMappedTensor(state, anchor, access.reg);
    auto extract = qtensor::ExtractOp::create(rewriter, anchor->getLoc(),
                                              tensor, access.index);
    assignMappedTensor(state, anchor, access.reg, extract.getOutTensor());
    materialized.values.push_back(extract.getResult());
    materialized.accesses.emplace_back(access);
  }

  return materialized;
}

/**
 * @brief Commits quantum-operation results to standalone mappings or QTensor.
 */
static void commitQubits(LoweringState& state, Operation* anchor,
                         ValueRange qcQubits, ValueRange qcoQubits,
                         const MaterializedQubits& materialized,
                         PatternRewriter& rewriter) {
  assert(qcQubits.size() == qcoQubits.size());
  assert(qcQubits.size() == materialized.accesses.size());

  for (std::size_t i = qcQubits.size(); i > 0; --i) {
    const auto position = i - 1;
    const auto& access = materialized.accesses[position];
    if (!access) {
      assignMappedQubit(state, anchor, qcQubits[position], qcoQubits[position]);
      continue;
    }

    auto tensor = lookupMappedTensor(state, anchor, access->reg);
    auto insert = qtensor::InsertOp::create(
        rewriter, anchor->getLoc(), qcoQubits[position], tensor, access->index);
    assignMappedTensor(state, anchor, access->reg, insert.getResult());
  }
}

/** @brief Resolves all structured QC state to QCO and QTensor values. */
[[nodiscard]] static SmallVector<Value> resolveAllValues(LoweringState& state,
                                                         Operation* anchor) {
  SmallVector<RegisterId> registers;
  SmallVector<Value> qcQubits;
  auto* owner = structuredValueOwner(anchor);
  if (const auto it = state.structuredValues.find(owner);
      it != state.structuredValues.end()) {
    llvm::append_range(registers, it->second.registers);
    llvm::append_range(qcQubits, it->second.qubits);
  } else {
    llvm::append_range(registers, state.regionRegisterMap[owner]);
    llvm::append_range(qcQubits, state.regionQubitMap[owner]);
  }

  SmallVector<Value> qcoTargets;
  qcoTargets.reserve(registers.size() + qcQubits.size());
  llvm::append_range(qcoTargets,
                     resolveMappedTensors(state, anchor, registers));
  llvm::append_range(qcoTargets, resolveMappedQubits(state, anchor, qcQubits));
  return qcoTargets;
}

/** @brief Rejects quantum SSA sources unsupported by the lowering state. */
[[nodiscard]] static LogicalResult
validateQuantumValueSources(Operation* root) {
  const auto result = root->walk([&](Operation* operation) {
    const bool isModifier = isa<qc::InvOp, qc::CtrlOp, qc::PowOp>(operation);
    for (Region& region : operation->getRegions()) {
      for (Block& block : region) {
        for (auto argument : block.getArguments()) {
          const bool isQubit = isa<qc::QubitType>(argument.getType());
          if ((!isQubit && !isQubitMemrefType(argument.getType())) ||
              (isModifier && isQubit)) {
            continue;
          }

          operation->emitOpError(
              "cannot convert arbitrary qubit or qubit-register block "
              "arguments; only QC modifier qubit arguments are supported");
          return WalkResult::interrupt();
        }
      }
    }

    for (auto value : operation->getResults()) {
      if (isQubitMemrefType(value.getType())) {
        auto allocation = dyn_cast<memref::AllocOp>(operation);
        if (!allocation) {
          operation->emitOpError(
              "requires a directly allocated qubit register");
          return WalkResult::interrupt();
        }
        if (allocation.getType().getRank() != 1) {
          operation->emitOpError(
              "requires one-dimensional qubit register storage");
          return WalkResult::interrupt();
        }
        continue;
      }

      if (isa<qc::QubitType>(value.getType()) &&
          !isa<qc::AllocOp, qc::StaticOp, memref::LoadOp>(operation)) {
        operation->emitOpError(
            "produces an unsupported qubit reference; use qc.alloc, "
            "qc.static, a qubit-register load, or a QC modifier argument");
        return WalkResult::interrupt();
      }
    }

    const bool supportsQuantumCaptures =
        isModifier ||
        isa<scf::ForOp, scf::WhileOp, scf::IfOp, scf::IndexSwitchOp>(operation);
    if (!supportsQuantumCaptures && operation->getNumRegions() != 0) {
      SetVector<Value> captures;
      getUsedValuesDefinedAbove(operation->getRegions(), captures);
      if (llvm::any_of(captures, [](Value value) {
            return isa<qc::QubitType>(value.getType()) ||
                   isQubitMemrefType(value.getType());
          })) {
        operation->emitOpError(
            "cannot capture quantum values in an unsupported region-bearing "
            "operation; use scf.for, scf.while, scf.if, or scf.index_switch");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

/** @brief Collects stable register identifiers and load provenance. */
[[nodiscard]] static LogicalResult
collectRegisterAccesses(Operation* root, LoweringState& state) {
  root->walk([&](memref::AllocOp op) {
    if (isa<qc::QubitType>(op.getType().getElementType())) {
      const auto reg = state.registerIds.size();
      state.registerIds.try_emplace(op.getResult(), reg);
    }
  });

  const auto result = root->walk([&](memref::LoadOp op) {
    if (!isa<qc::QubitType>(op.getMemRefType().getElementType())) {
      return WalkResult::advance();
    }

    if (op.getMemRefType().getRank() != 1 || op.getIndices().size() != 1) {
      op.emitOpError("requires one-dimensional qubit register storage");
      return WalkResult::interrupt();
    }

    const auto regIt = state.registerIds.find(op.getMemref());
    if (regIt == state.registerIds.end()) {
      op.emitOpError("requires a directly allocated qubit register");
      return WalkResult::interrupt();
    }

    state.registerAccesses.try_emplace(
        op.getResult(),
        RegisterAccess{.reg = regIt->second, .index = op.getIndices().front()});

    for (Operation* user : op.getResult().getUsers()) {
      if (isa<qc::UnitaryOpInterface, qc::MeasureOp, qc::ResetOp>(user)) {
        continue;
      }
      user->emitOpError(
          "cannot consume a register-backed qubit reference; only QC quantum "
          "operations support register-backed qubits");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }

  const auto distinctResult = root->walk([&](Operation* operation) {
    auto unitary = dyn_cast<qc::UnitaryOpInterface>(operation);
    if (!unitary || unitary.getNumQubits() < 2) {
      return WalkResult::advance();
    }

    llvm::SmallDenseSet<Value, 4> qubits;
    DenseMap<RegisterId, SeenRegisterIndices> registerIndices;
    for (auto qubit : unitary.getQubits()) {
      if (!qubits.insert(qubit).second) {
        operation->emitOpError("requires distinct qubit operands");
        return WalkResult::interrupt();
      }

      const auto access = state.registerAccesses.find(qubit);
      if (access == state.registerAccesses.end()) {
        continue;
      }

      auto& seen = registerIndices[access->second.reg];
      if (const auto constant = getConstantIntValue(access->second.index)) {
        const auto [it, inserted] =
            seen.constants.try_emplace(*constant, access->second.index);
        if (!inserted &&
            isEqualConstantIntOrValue(it->second, access->second.index)) {
          operation->emitOpError(
              "requires distinct qubit operands; register-backed operands "
              "have the same constant index");
          return WalkResult::interrupt();
        }
        continue;
      }

      if (!seen.dynamicValues.insert(access->second.index).second) {
        operation->emitOpError(
            "requires distinct qubit operands; register-backed operands use "
            "the same dynamic index");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  return success(!distinctResult.wasInterrupted());
}

/** @brief Rejects unsupported operations and qubit captures in QC modifiers. */
[[nodiscard]] static LogicalResult validateModifierBodies(Operation* root) {
  const auto result = root->walk([&](Operation* operation) {
    if (isa<qc::InvOp, qc::CtrlOp, qc::PowOp>(operation)) {
      SetVector<Value> captures;
      getUsedValuesDefinedAbove(operation->getRegions(), captures);
      if (llvm::any_of(captures, [](Value value) {
            return isa<qc::QubitType>(value.getType());
          })) {
        operation->emitOpError(
            "body must not capture qubits from above; use only its aliased "
            "block arguments");
        return WalkResult::interrupt();
      }
    }

    if (!isa<cbit::AllocOp, cbit::LoadOp, cbit::StoreOp, qc::AllocOp,
             qc::DeallocOp, qc::MeasureOp, qc::ResetOp, memref::LoadOp,
             memref::StoreOp>(operation)) {
      return WalkResult::advance();
    }

    for (auto* parent = operation->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      if (!isa<qc::InvOp, qc::CtrlOp, qc::PowOp>(parent)) {
        continue;
      }
      parent->emitOpError(
          "body must not contain non-unitary operations or access registers");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

/** @brief Collects values captured by supported structured control flow. */
static void collectStructuredCaptures(Operation* root, LoweringState& state) {
  root->walk([&](Operation* operation) {
    if (!isa<scf::ForOp, scf::WhileOp, scf::IfOp, scf::IndexSwitchOp>(
            operation)) {
      return;
    }

    SetVector<Value> captures;
    getUsedValuesDefinedAbove(operation->getRegions(), captures);

    auto& qubits = state.regionQubitMap[operation];
    auto& registers = state.regionRegisterMap[operation];
    qubits.clear();
    registers.clear();
    for (auto value : captures) {
      if (const auto access = state.registerAccesses.find(value);
          access != state.registerAccesses.end()) {
        registers.insert(access->second.reg);
        continue;
      }
      if (isa<qc::QubitType>(value.getType())) {
        qubits.insert(value);
        continue;
      }
      if (isQubitMemrefType(value.getType())) {
        registers.insert(lookupRegisterId(state, value));
      }
    }
  });
}

/**
 * @brief Canonicalizes preserved SCF capture keys after signature conversion.
 */
static void remapStructuredCaptures(Operation* root, LoweringState& state) {
  root->walk([&](Operation* operation) {
    if (!isa<scf::ForOp, scf::WhileOp, scf::IfOp, scf::IndexSwitchOp>(
            operation)) {
      return;
    }

    const auto captures = state.regionQubitMap.find(operation);
    if (captures == state.regionQubitMap.end()) {
      return;
    }

    SetVector<Value> remapped;
    for (auto qubit : captures->second) {
      remapped.insert(canonicalQubitKey(state, qubit));
    }
    captures->second = std::move(remapped);
  });
}

/** @brief Seeds region-owned modifier state after signature conversion. */
static void initializeModifierRegionState(Operation* modifier,
                                          ValueRange sourceArguments,
                                          LoweringState& state) {
  auto& region = modifier->getRegion(0);
  const auto convertedArguments = region.front().getArguments();
  for (auto [source, converted] :
       llvm::zip_equal(sourceArguments, convertedArguments)) {
    state.convertedQubitAliases[source] = converted;
  }
  state.modifierRegionQubits[&region] =
      SmallVector<Value>(convertedArguments.begin(), convertedArguments.end());
  seedRegionMappings(state, region, convertedArguments, {}, convertedArguments,
                     {});

  // Signature conversion replaces modifier block arguments. Refresh nested
  // structured captures so they refer to the converted arguments owned by the
  // moved region rather than source conversion keys.
  remapStructuredCaptures(modifier, state);
}

namespace {

/**
 * @brief Converts func.return and sinks remaining live qubits.
 *
 * @details
 * QC uses reference semantics and does not enforce linear typing for qubits.
 * After conversion, QCO requires that every qubit SSA value is consumed
 * exactly once. For allocations (including static qubits), the sink is
 * `qco.sink`. This pattern inserts `qco.sink` operations for all
 * still-live qubits tracked in the lowering state right before the return.
 */
struct ConvertFuncReturnOp final : StatefulOpConversionPattern<func::ReturnOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* funcRegion = op->getParentRegion();
    auto& map = state.qubitMap[funcRegion];

    // Build return values from qubitMap and collect live qubit information.
    // A qubit from the current scope is considered alive if it is returned from
    // the function. Otherwise, it is considered dead.
    SmallVector<Value> returnValues;
    returnValues.reserve(op.getNumOperands());
    DenseSet<Value> liveQubits;
    for (auto [qcOperand, adaptorOperand] :
         llvm::zip_equal(op.getOperands(), adaptor.getOperands())) {
      if (auto it = map.find(qcOperand); it != map.end()) {
        auto latest = it->second;
        returnValues.emplace_back(latest);
        liveQubits.insert(latest);
      } else {
        returnValues.emplace_back(adaptorOperand);
      }
    }

    // Deallocate dead qubit values
    for (auto qcoQubit : llvm::make_second_range(map)) {
      if (!liveQubits.contains(qcoQubit)) {
        SinkOp::create(rewriter, op.getLoc(), qcoQubit);
      }
    }
    state.qubitMap.erase(funcRegion);

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, returnValues);
    return success();
  }
};

/**
 * @brief Type converter for QC-to-QCO conversion
 *
 * @details
 * Handles type conversion between the QC and QCO dialects.
 * The primary conversion is from !qc.qubit to !qco.qubit, which
 * represents the semantic shift from reference types to value types.
 *
 * Other types (integers, booleans, etc.) pass through unchanged via
 * the identity conversion.
 */
class QCToQCOTypeConverter final : public TypeConverter {
public:
  explicit QCToQCOTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    // Convert QC qubit references to QCO qubit values
    addConversion([ctx](qc::QubitType /*type*/) -> Type {
      return qco::QubitType::get(ctx);
    });
  }
};

/**
 * @brief Converts memref.alloc to qtensor.alloc
 *
 * @par Example:
 * ```mlir
 * %memref = memref.alloc(%c3) : memref<3x!qc.qubit>
 * ```
 * is converted to
 * ```mlir
 * %tensor = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
 * ```
 */
struct ConvertMemRefAllocOp final
    : StatefulOpConversionPattern<memref::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (!isa<qc::QubitType>(op.getType().getElementType())) {
      return failure();
    }

    auto shape = op.getType().getShape();
    if (shape.size() != 1) {
      return failure();
    }
    if (failed(getState().ensureAllocationMode(AllocationMode::Dynamic,
                                               op.getOperation()))) {
      return failure();
    }

    qtensor::AllocOp alloc;
    if (shape[0] == ShapedType::kDynamic) {
      alloc = qtensor::AllocOp::create(rewriter, op.getLoc(),
                                       adaptor.getDynamicSizes()[0]);
    } else {
      auto size =
          arith::ConstantIndexOp::create(rewriter, op.getLoc(), shape[0]);
      alloc = qtensor::AllocOp::create(rewriter, op.getLoc(), size.getResult());
    }
    alloc->setDiscardableAttrs(op->getDiscardableAttrDictionary());

    auto& state = getState();
    auto memref = op.getResult();
    assignMappedTensor(state, alloc, lookupRegisterId(state, memref),
                       alloc.getResult());
    rewriter.replaceOp(op, alloc.getResult());

    return success();
  }
};

/**
 * @brief Erases a qubit memref.load after recording its converted index
 *
 * @par Example:
 * ```mlir
 * %q = memref.load %memref[%c0] : memref<3x!qc.qubit>
 * ```
 * The consuming quantum operation materializes and commits the referenced
 * qubit locally.
 */
struct ConvertMemRefLoadOp final : StatefulOpConversionPattern<memref::LoadOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (!isa<qc::QubitType>(op.getMemRefType().getElementType())) {
      return failure();
    }

    auto& state = getState();
    const auto accessIt = state.registerAccesses.find(op.getResult());
    assert(accessIt != state.registerAccesses.end() &&
           "qubit load provenance not found");
    accessIt->second.index = adaptor.getIndices().front();

    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts memref.dealloc to qtensor.dealloc
 *
 * @par Example:
 * ```mlir
 * memref.dealloc %memref : memref<3x!qc.qubit>
 * ```
 * is converted to
 * ```mlir
 * qtensor.dealloc %tensor : tensor<3x!qco.qubit>
 * ```
 */
struct ConvertMemRefDeallocOp final
    : StatefulOpConversionPattern<memref::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto memref = op.getMemref();
    if (!isa<qc::QubitType>(memref.getType().getElementType())) {
      return failure();
    }

    auto& state = getState();
    auto& tensorMap = state.tensorMap[op->getParentRegion()];
    const auto reg = lookupRegisterId(state, memref);
    auto qtensor = lookupMappedTensor(state, op.getOperation(), reg);
    tensorMap.erase(reg);

    rewriter.replaceOpWithNewOp<qtensor::DeallocOp>(op, qtensor);
    return success();
  }
};

/**
 * @brief Converts qc.alloc to qco.alloc
 *
 * @par Example:
 * ```mlir
 * %q = qc.alloc : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q = qco.alloc : !qco.qubit
 * ```
 */
struct ConvertQCAllocOp final : StatefulOpConversionPattern<qc::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    if (failed(state.ensureAllocationMode(AllocationMode::Dynamic,
                                          op.getOperation()))) {
      return failure();
    }
    auto qcQubit = op.getResult();

    // Create the qco.alloc operation
    auto qcoOp = rewriter.replaceOpWithNewOp<qco::AllocOp>(op);

    auto qcoQubit = qcoOp.getResult();
    assignMappedQubit(state, qcoOp, qcQubit, qcoQubit);

    return success();
  }
};

/**
 * @brief Converts qc.dealloc to qco.sink
 *
 * @details
 * Deallocates a qubit by looking up its latest QCO value and creating
 * a corresponding qco.sink operation. The mapping is removed from
 * the state as the qubit is no longer in use.
 *
 * Example transformation:
 * ```mlir
 * qc.dealloc %q : !qc.qubit
 * // becomes (where %q maps to %q_final):
 * qco.sink %q_final : !qco.qubit
 * ```
 */
struct ConvertQCDeallocOp final : StatefulOpConversionPattern<DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto& qubitMap = state.qubitMap[op->getParentRegion()];
    auto* operation = op.getOperation();

    auto qcQubit = op.getQubit();
    auto qcoQubit = lookupMappedQubit(state, operation, qcQubit);

    // Create the sink operation
    rewriter.replaceOpWithNewOp<SinkOp>(op, qcoQubit);

    // Remove from state as qubit is no longer in use
    qubitMap.erase(qcQubit);

    return success();
  }
};

/**
 * @brief Converts qc.static to qco.static
 *
 * @details
 * Static qubits represent references to hardware-mapped or fixed-position
 * qubits identified by an index. This conversion creates the corresponding
 * qco.static operation and establishes the mapping.
 *
 * Example transformation:
 * ```mlir
 * %q = qc.static 0 : !qc.qubit
 * // becomes:
 * %q0 = qco.static 0 : !qco.qubit
 * ```
 */
struct ConvertQCStaticOp final : StatefulOpConversionPattern<qc::StaticOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::StaticOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    if (failed(state.ensureAllocationMode(AllocationMode::Static,
                                          op.getOperation()))) {
      return failure();
    }
    auto qcQubit = op.getQubit();

    auto qcoOp = rewriter.replaceOpWithNewOp<qco::StaticOp>(op, op.getIndex());
    assignMappedQubit(state, qcoOp, qcQubit, qcoOp.getQubit());

    return success();
  }
};

/**
 * @brief Converts qc.measure to qco.measure
 *
 * @details
 * Measurement is a key operation where the semantic difference is visible:
 * - QC: Measures in-place, returning only the classical bit
 * - QCO: Consumes input qubit, returns both output qubit and classical bit
 *
 * The conversion looks up the latest QCO value for the QC qubit,
 * performs the measurement, updates the mapping with the output qubit,
 * and returns the classical bit result.
 *
 * @par Example:
 * ```mlir
 * %c = qc.measure %q : !qc.qubit -> i1
 * ```
 * is converted to
 * ```mlir
 * %q_out, %c = qco.measure %q_in : !qco.qubit
 * ```
 */
struct ConvertQCMeasureOp final : StatefulOpConversionPattern<qc::MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::MeasureOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto qcQubit = op.getQubit();
    const SmallVector<Value, 1> qcQubits{qcQubit};
    auto materialized = materializeQubits(state, operation, qcQubits, rewriter);

    // Create qco.measure (returns both output qubit and bit result)
    auto qcoOp =
        qco::MeasureOp::create(rewriter, op.getLoc(), materialized.values[0]);

    const SmallVector<Value, 1> qcoQubits{qcoOp.getQubitOut()};
    commitQubits(state, operation, qcQubits, qcoQubits, materialized, rewriter);

    // Replace the QC operation's bit result with the QCO bit result
    rewriter.replaceOp(op, qcoOp.getResult());

    return success();
  }
};

/**
 * @brief Converts qc.reset to qco.reset
 *
 * @details
 * Reset operations force a qubit to the |0⟩ state. The semantic difference:
 * - QC: Resets in-place (no result value)
 * - QCO: Consumes input qubit, returns reset output qubit
 *
 * The conversion looks up the latest QCO value, performs the reset,
 * and updates the mapping with the output qubit. The QC operation
 * is erased as it has no results to replace.
 *
 * Example transformation:
 * ```mlir
 * qc.reset %q : !qc.qubit
 * // becomes (where %q maps to %q_in):
 * %q_out = qco.reset %q_in : !qco.qubit -> !qco.qubit
 * // state updated: %q now maps to %q_out
 * ```
 */
struct ConvertQCResetOp final : StatefulOpConversionPattern<qc::ResetOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::ResetOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto qcQubit = op.getQubit();
    const SmallVector<Value, 1> qcQubits{qcQubit};
    auto materialized = materializeQubits(state, operation, qcQubits, rewriter);

    // Create qco.reset (consumes input, produces output)
    auto qcoOp =
        qco::ResetOp::create(rewriter, op.getLoc(), materialized.values[0]);

    const SmallVector<Value, 1> qcoQubits{qcoOp.getQubitOut()};
    commitQubits(state, operation, qcQubits, qcoQubits, materialized, rewriter);

    // Erase the old (it has no results to replace)
    rewriter.eraseOp(op);

    return success();
  }
};

template <typename QCOpType, typename QCOOpType, std::size_t NumTargets,
          std::size_t NumParams>
struct ConvertQCGateToQCO final : StatefulOpConversionPattern<QCOpType> {
  using StatefulOpConversionPattern<QCOpType>::StatefulOpConversionPattern;

  template <std::size_t... TargetIndices, std::size_t... ParamIndices>
  auto createGate(ConversionPatternRewriter& rewriter, QCOpType op,
                  ValueRange qcoTargets,
                  std::index_sequence<TargetIndices...> /*targets*/,
                  std::index_sequence<ParamIndices...> /*params*/) const {
    auto params = op.getParameters();
    return QCOOpType::create(rewriter, op.getLoc(),
                             qcoTargets[TargetIndices]...,
                             params[ParamIndices]...);
  }

  LogicalResult
  matchAndRewrite(QCOpType op, QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto qcTargets = op.getTargets();
    auto materialized = materializeQubits(state, op, qcTargets, rewriter);
    auto qcoOp = createGate(rewriter, op, materialized.values,
                            std::make_index_sequence<NumTargets>{},
                            std::make_index_sequence<NumParams>{});

    commitQubits(state, op, qcTargets, qcoOp.getOutputTargets(), materialized,
                 rewriter);

    rewriter.eraseOp(op);

    return success();
  }
};

/** Converts a variadic dense qc.unitary to its value-semantics form. */
struct ConvertQCUnitaryOp final : StatefulOpConversionPattern<qc::UnitaryOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::UnitaryOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto qcQubits = op.getQubits();
    auto materialized = materializeQubits(state, operation, qcQubits, rewriter);
    auto qcoOp = qco::UnitaryOp::create(rewriter, op.getLoc(),
                                        materialized.values, op.getMatrix());

    commitQubits(state, operation, qcQubits, qcoOp.getQubitsOut(), materialized,
                 rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts qc.barrier to qco.barrier
 *
 * @par Example:
 * ```mlir
 * qc.barrier %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out:2 = qco.barrier %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit,
 * !qco.qubit
 * ```
 */
struct ConvertQCBarrierOp final : StatefulOpConversionPattern<qc::BarrierOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::BarrierOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto qcQubits = op.getQubits();
    auto materialized = materializeQubits(state, operation, qcQubits, rewriter);

    // Create qco.barrier
    auto qcoOp =
        qco::BarrierOp::create(rewriter, op.getLoc(), materialized.values);

    commitQubits(state, operation, qcQubits, qcoOp.getQubitsOut(), materialized,
                 rewriter);

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts qc.ctrl to qco.ctrl
 *
 * @par Example:
 * ```mlir
 * qc.ctrl(%q0) targets(%a0 = %q1) {
 *   qc.x %a0 : !qc.qubit
 * } : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %controls_out, %targets_out = qco.ctrl(%q0_in) targets(%a_in = %q1_in) {
 *   %a_res = qco.x %a_in : !qco.qubit -> !qco.qubit
 *   qco.yield %a_res : !qco.qubit
 * } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
 * ```
 */
struct ConvertQCCtrlOp final : StatefulOpConversionPattern<qc::CtrlOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::CtrlOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto qcControls = op.getControls();
    auto qcQubits = op.getQubits();
    auto materialized = materializeQubits(state, operation, qcQubits, rewriter);
    auto qcoControls =
        ValueRange(materialized.values).take_front(qcControls.size());
    auto qcoTargets =
        ValueRange(materialized.values).drop_front(qcControls.size());

    // Create qco.ctrl
    auto qcoOp =
        qco::CtrlOp::create(rewriter, op.getLoc(), qcoControls, qcoTargets);

    SmallVector<Value> qcoQubits;
    llvm::append_range(qcoQubits, qcoOp.getControlsOut());
    llvm::append_range(qcoQubits, qcoOp.getTargetsOut());
    commitQubits(state, operation, qcQubits, qcoQubits, materialized, rewriter);

    const SmallVector<Value> sourceArguments(
        op.getRegion().front().getArguments());

    // Inline region and convert the block signature to QCO types.
    if (failed(moveRegion(op.getRegion(), qcoOp.getRegion(), rewriter,
                          getTypeConverter()))) {
      return failure();
    }

    initializeModifierRegionState(qcoOp, sourceArguments, state);

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts qc.inv to qco.inv
 *
 * @par Example:
 * ```mlir
 * qc.inv {
 *   qc.s %q0 : !qc.qubit
 * } : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q0_out = qco.inv (%a0_in = %q0_in) {
 *   %a0_res = qco.s %a0_in : !qco.qubit -> !qco.qubit
 *   qco.yield %a0_res : !qco.qubit
 * } : {!qco.qubit} -> {!qco.qubit}
 * ```
 */
struct ConvertQCInvOp final : StatefulOpConversionPattern<qc::InvOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::InvOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto qcTargets = op.getTargets();
    auto materialized =
        materializeQubits(state, operation, qcTargets, rewriter);

    // Create qco.inv
    auto qcoOp = qco::InvOp::create(rewriter, op.getLoc(), materialized.values);

    commitQubits(state, operation, qcTargets, qcoOp.getOutputTargets(),
                 materialized, rewriter);

    const SmallVector<Value> sourceArguments(
        op.getRegion().front().getArguments());

    // Inline region and convert the block signature to QCO types.
    if (failed(moveRegion(op.getRegion(), qcoOp.getRegion(), rewriter,
                          getTypeConverter()))) {
      return failure();
    }

    initializeModifierRegionState(qcoOp, sourceArguments, state);

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts qc.pow to qco.pow
 *
 * @par Example:
 * ```mlir
 * qc.pow(%exponent) (%a0 = %q0) {
 *   qc.s %a0 : !qc.qubit
 * } : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q0_out = qco.pow(%exponent) (%a0_in = %q0_in) {
 *   %a0_res = qco.s %a0_in : !qco.qubit -> !qco.qubit
 *   qco.yield %a0_res
 * } : {!qco.qubit} -> {!qco.qubit}
 * ```
 */
struct ConvertQCPowOp final : StatefulOpConversionPattern<qc::PowOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::PowOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto qcTargets = op.getTargets();
    auto materialized =
        materializeQubits(state, operation, qcTargets, rewriter);

    // Create qco.pow with exponent.
    auto qcoOp = qco::PowOp::create(rewriter, op.getLoc(), materialized.values,
                                    op.getExponent());

    commitQubits(state, operation, qcTargets, qcoOp.getQubitsOut(),
                 materialized, rewriter);

    const SmallVector<Value> sourceArguments(
        op.getRegion().front().getArguments());

    // Inline region and convert the block signature to QCO types.
    if (failed(moveRegion(op.getRegion(), qcoOp.getRegion(), rewriter,
                          getTypeConverter()))) {
      return failure();
    }

    initializeModifierRegionState(qcoOp, sourceArguments, state);

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts qc.yield to qco.yield
 *
 * @par Example:
 * ```mlir
 * qc.yield
 * ```
 * is converted to
 * ```mlir
 * qco.yield %targets : !qco.qubit
 * ```
 */
struct ConvertQCYieldOp final : StatefulOpConversionPattern<qc::YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto* region = op->getParentRegion();
    const auto frame = state.modifierRegionQubits.find(region);
    if (frame == state.modifierRegionQubits.end()) {
      return rewriter.notifyMatchFailure(op, "missing modifier region state");
    }

    auto targets = resolveMappedQubits(state, operation, frame->second);
    rewriter.replaceOpWithNewOp<qco::YieldOp>(op, targets);
    state.qubitMap.erase(region);
    state.modifierRegionQubits.erase(frame);
    return success();
  }
};

/**
 * @brief Converts scf.for with memory semantics to scf.for with value
 * semantics for qubit values
 *
 * @par Example:
 * ```mlir
 * scf.for %iv = %lb to %ub step %step {
 *   %q0 = qc.load %memref[%iv] : !memref<3x!qc.qubit>
 *   qc.h %q0 : !qc.qubit
 * }
 * ```
 * is converted to
 * ```mlir
 * %targets_out = scf.for %iv = %lb to %ub step %step iter_args(%arg0 =
 * %qtensor) -> (tensor<3x!qco.qubit) {
 *   %t0, %q0 = qtensor.extract %arg0[%iv] : tensor<3x!qco.qubit>
 *   %q1 = qco.h %q0 : !qco.qubit -> !qco.qubit
 *   %t1 = qtensor.insert %q1 into %t0[%iv] : tensor<3x!qco.qubit>
 *   scf.yield %t1 : tensor<3x!qco.qubit>
 * }
 * ```
 */
struct ConvertSCFForOp final : StatefulOpConversionPattern<scf::ForOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto& registerMap = state.regionRegisterMap[op];
    auto& qubitMap = state.regionQubitMap[op];
    const auto numRegisters = registerMap.size();
    const auto numQubits = qubitMap.size();

    auto qcoTargets = resolveAllValues(state, operation);
    const auto numOriginalResults = op.getNumResults();
    SmallVector<Value> initArgs(op.getInitArgs());
    llvm::append_range(initArgs, qcoTargets);

    // Create the new ForOp
    auto newForOp =
        scf::ForOp::create(rewriter, op.getLoc(), op.getLowerBound(),
                           op.getUpperBound(), op.getStep(), initArgs);
    newForOp->setDiscardableAttrs(op->getDiscardableAttrDictionary());

    assignMappedTensors(state, op.getOperation(), registerMap,
                        newForOp.getResults()
                            .drop_front(numOriginalResults)
                            .take_front(numRegisters));
    assignMappedQubits(state, op.getOperation(), qubitMap,
                       newForOp->getResults().take_back(numQubits));

    // Move the contents from the old block into the new block
    auto& srcBlock = op.getRegion().front();
    auto& dstBlock = newForOp.getRegion().front();
    dstBlock.getOperations().splice(dstBlock.end(), srcBlock.getOperations());
    for (auto [oldArgument, newArgument] : llvm::zip_equal(
             srcBlock.getArguments(),
             dstBlock.getArguments().take_front(srcBlock.getNumArguments()))) {
      rewriter.replaceAllUsesWith(oldArgument, newArgument);
    }

    SmallVector<Value> qubits(qubitMap.begin(), qubitMap.end());
    SmallVector<RegisterId> registers(registerMap.begin(), registerMap.end());
    state.structuredValues[newForOp] = {.qubits = qubits,
                                        .registers = registers};
    seedRegionMappings(state, newForOp.getRegion(), qubits, registers,
                       dstBlock.getArguments().take_back(numQubits),
                       dstBlock.getArguments()
                           .drop_front(1 + numOriginalResults)
                           .take_front(numRegisters));

    rewriter.replaceOp(op,
                       newForOp.getResults().take_front(numOriginalResults));

    return success();
  }
};

/**
 * @brief Converts scf.while with memory semantics to scf.while with value
 * semantics for qubit values.
 *
 * @par Example:
 * ```mlir
 * scf.while : () -> () {
 *   %cond = qc.measure %q0 : !qc.qubit -> i1
 *   scf.condition(%cond)
 * } do {
 *   qc.h %q0 : !qc.qubit
 *   scf.yield
 * }
 * ```
 * is converted to
 * ```mlir
 * %targets_out = scf.while (%arg0 = %q0) : (!qco.qubit) -> !qco.qubit {
 *   %q1 = qco.measure %arg0 : !qco.qubit
 *   scf.condition(%cond) %q1 : !qco.qubit
 * } do {
 * ^bb0(%arg0: !qco.qubit):
 *   %q2 = qco.h %arg0 : !qco.qubit -> !qco.qubit
 *   scf.yield %q2 : !qco.qubit
 * }
 * ```
 */
struct ConvertSCFWhileOp final : StatefulOpConversionPattern<scf::WhileOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto& registerMap = state.regionRegisterMap[op];
    auto& qubitMap = state.regionQubitMap[op];
    const auto numRegisters = registerMap.size();
    const auto numQubits = qubitMap.size();

    auto qcoTargets = resolveAllValues(state, operation);
    const auto numOriginalInits = op.getInits().size();
    const auto numOriginalResults = op.getNumResults();
    SmallVector<Value> initArgs(op.getInits());
    llvm::append_range(initArgs, qcoTargets);
    SmallVector<Type> resultTypes(op.getResultTypes());
    llvm::append_range(resultTypes,
                       llvm::map_range(qcoTargets, [](Value value) {
                         return value.getType();
                       }));

    // Create the new WhileOp
    auto newWhileOp =
        scf::WhileOp::create(rewriter, op.getLoc(), resultTypes, initArgs);
    assignMappedTensors(state, op.getOperation(), registerMap,
                        newWhileOp.getResults()
                            .drop_front(numOriginalResults)
                            .take_front(numRegisters));
    assignMappedQubits(state, op.getOperation(), qubitMap,
                       newWhileOp->getResults().take_back(numQubits));

    auto& newBeforeRegion = newWhileOp.getBefore();
    auto& newAfterRegion = newWhileOp.getAfter();

    const SmallVector beforeLocs(initArgs.size(), op->getLoc());
    const SmallVector afterLocs(resultTypes.size(), op->getLoc());

    // Create the new blocks and move the contents from the old blocks into the
    // new ones
    auto* newBeforeBlock = rewriter.createBlock(
        &newBeforeRegion, {}, ValueRange(initArgs).getTypes(), beforeLocs);
    auto* newAfterBlock = rewriter.createBlock(
        &newAfterRegion, {}, TypeRange(resultTypes), afterLocs);
    auto* oldBeforeBlock = op.getBeforeBody();
    auto* oldAfterBlock = op.getAfterBody();
    newBeforeBlock->getOperations().splice(newBeforeBlock->end(),
                                           oldBeforeBlock->getOperations());
    newAfterBlock->getOperations().splice(newAfterBlock->end(),
                                          oldAfterBlock->getOperations());
    for (auto [oldArgument, newArgument] :
         llvm::zip_equal(oldBeforeBlock->getArguments(),
                         newBeforeBlock->getArguments().take_front(
                             oldBeforeBlock->getNumArguments()))) {
      rewriter.replaceAllUsesWith(oldArgument, newArgument);
    }
    for (auto [oldArgument, newArgument] :
         llvm::zip_equal(oldAfterBlock->getArguments(),
                         newAfterBlock->getArguments().take_front(
                             oldAfterBlock->getNumArguments()))) {
      rewriter.replaceAllUsesWith(oldArgument, newArgument);
    }

    SmallVector<Value> qubits(qubitMap.begin(), qubitMap.end());
    SmallVector<RegisterId> registers(registerMap.begin(), registerMap.end());
    state.structuredValues[newWhileOp] = {.qubits = qubits,
                                          .registers = registers};
    seedRegionMappings(state, newWhileOp.getBefore(), qubits, registers,
                       newBeforeBlock->getArguments().take_back(numQubits),
                       newBeforeBlock->getArguments()
                           .drop_front(numOriginalInits)
                           .take_front(numRegisters));
    seedRegionMappings(state, newWhileOp.getAfter(), qubits, registers,
                       newAfterBlock->getArguments().take_back(numQubits),
                       newAfterBlock->getArguments()
                           .drop_front(numOriginalResults)
                           .take_front(numRegisters));

    rewriter.replaceOp(op,
                       newWhileOp.getResults().take_front(numOriginalResults));
    return success();
  }
};

/**
 * @brief Converts scf.if to qco.if
 *
 * @par Example:
 * ```mlir
 * scf.if %cond {
 *   qc.h %q0 : !qc.qubit
 * }
 * ```
 * is converted to
 * ```mlir
 * %targets_out = qco.if %cond args(%arg0 = %q0) -> (!qco.qubit) {
 *   %q1 = qco.h %arg0 : !qco.qubit -> !qco.qubit
 *   qco.yield %q1 : !qco.qubit
 * } else args(%arg0 = %q0) {
 *   qco.yield %arg0 : !qco.qubit
 * }
 * ```
 */
struct ConvertSCFIfOp final : StatefulOpConversionPattern<scf::IfOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto& registerMap = state.regionRegisterMap[op];
    auto& qubitMap = state.regionQubitMap[op];
    const auto numRegisters = registerMap.size();
    const auto numQubits = qubitMap.size();

    auto qcoTargets = resolveAllValues(state, operation);

    // Create the new IfOp
    auto newIfOp = IfOp::create(rewriter, op.getLoc(), op.getResultTypes(),
                                ValueRange(qcoTargets).getTypes(),
                                op.getCondition(), qcoTargets);
    assignMappedTensors(state, op.getOperation(), registerMap,
                        newIfOp.getLinearResults().take_front(numRegisters));
    assignMappedQubits(state, op.getOperation(), qubitMap,
                       newIfOp.getLinearResults().take_back(numQubits));

    auto& thenRegion = newIfOp.getThenRegion();
    auto& elseRegion = newIfOp.getElseRegion();

    const SmallVector locs(qcoTargets.size(), op->getLoc());

    // Create the new blocks and move the contents from the old blocks into the
    // new ones
    const auto qcoTargetTypes = ValueRange(qcoTargets).getTypes();
    auto* thenBlock =
        rewriter.createBlock(&thenRegion, {}, qcoTargetTypes, locs);
    auto* elseBlock =
        rewriter.createBlock(&elseRegion, {}, qcoTargetTypes, locs);

    thenBlock->getOperations().splice(
        thenBlock->end(), op.getThenRegion().front().getOperations());

    SmallVector<Value> qubits(qubitMap.begin(), qubitMap.end());
    SmallVector<RegisterId> registers(registerMap.begin(), registerMap.end());
    state.structuredValues[newIfOp] = {.qubits = qubits,
                                       .registers = registers};

    if (!op.getElseRegion().empty()) {
      elseBlock->getOperations().splice(
          elseBlock->end(), op.getElseRegion().front().getOperations());
      seedRegionMappings(state, newIfOp.getElseRegion(), qubits, registers,
                         elseBlock->getArguments().take_back(numQubits),
                         elseBlock->getArguments().take_front(numRegisters));

    } else {
      // If the else block is empty, just create the new qco::YieldOp
      rewriter.setInsertionPointToEnd(elseBlock);
      qco::YieldOp::create(rewriter, op->getLoc(), elseBlock->getArguments());
    }

    seedRegionMappings(state, newIfOp.getThenRegion(), qubits, registers,
                       thenBlock->getArguments().take_back(numQubits),
                       thenBlock->getArguments().take_front(numRegisters));

    rewriter.replaceOp(op, newIfOp.getClassicalResults());
    return success();
  }
};

/**
 * @brief Converts scf.index_switch to qco.index_switch
 *
 * @par Example:
 * ```mlir
 * scf.index_switch %condition
 * case 0 {
 *   qc.x %q0 : !qc.qubit
 * }
 * default {
 *   qc.z %q0 : !qc.qubit
 * }
 * ```
 * is converted to
 * ```mlir
 * %result = qco.index_switch %condition -> !qco.qubit
 * case 0 args(%arg0 = %q0) {
 *   %q1 = qco.x %arg0 : !qco.qubit -> !qco.qubit
 *   qco.yield %q1 : !qco.qubit
 * }
 * default args(%arg0 = %q0) {
 *   %q2 = qco.z %arg0 : !qco.qubit -> !qco.qubit
 *   qco.yield %q2 : !qco.qubit
 * }
 * ```
 */
struct ConvertSCFIndexSwitchOp final
    : StatefulOpConversionPattern<scf::IndexSwitchOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IndexSwitchOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto& registerMap = state.regionRegisterMap[op];
    auto& qubitMap = state.regionQubitMap[op];
    const auto numRegisters = registerMap.size();
    const auto numQubits = qubitMap.size();

    const auto targets = resolveAllValues(state, operation);
    const auto linearResultTypes = ValueRange(targets).getTypes();
    const SmallVector locs(targets.size(), op.getLoc());

    auto newOp = IndexSwitchOp::create(
        rewriter, op.getLoc(), op.getResultTypes(), linearResultTypes,
        op.getArg(), op.getCases(), targets, op.getNumCases());

    assignMappedTensors(state, op.getOperation(), registerMap,
                        newOp.getLinearResults().take_front(numRegisters));
    assignMappedQubits(state, op.getOperation(), qubitMap,
                       newOp.getLinearResults().take_back(numQubits));

    SmallVector<Value> qubits(qubitMap.begin(), qubitMap.end());
    SmallVector<RegisterId> registers(registerMap.begin(), registerMap.end());
    state.structuredValues[newOp] = {.qubits = qubits, .registers = registers};

    const auto newCaseRegions = newOp.getCaseRegions();
    const auto oldCaseRegions = op.getCaseRegions();
    const auto buildRegion = [&](Region& oldRegion, Region& newRegion) {
      auto* oldBlock = &oldRegion.front();
      auto* newBlock = rewriter.createBlock(
          &newRegion, {}, newOp.getLinearResults().getTypes(), locs);
      newBlock->getOperations().splice(newBlock->end(),
                                       oldBlock->getOperations());

      seedRegionMappings(state, newRegion, qubits, registers,
                         newBlock->getArguments().take_back(numQubits),
                         newBlock->getArguments().take_front(numRegisters));
    };

    for (auto [oldRegion, newRegion] :
         llvm::zip_equal(oldCaseRegions, newCaseRegions)) {
      buildRegion(oldRegion, newRegion);
    }
    buildRegion(op.getDefaultRegion(), newOp.getDefaultRegion());

    rewriter.replaceOp(op, newOp.getClassicalResults());
    return success();
  }
};

/**
 * @brief Converts scf.yield with memory semantics to scf.yield with value
 * semantics for qubit values or to qco.yield if the parentOp is a qco::IfOp or
 * qco::IndexSwitchOp.
 *
 * @par Example:
 * ```mlir
 * scf.yield
 * ```
 * is converted to
 * ```mlir
 * scf.yield %targets
 * ```
 */
struct ConvertSCFYieldOp final : StatefulOpConversionPattern<scf::YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();

    SmallVector<Value> targets(op.getResults());
    llvm::append_range(targets, resolveAllValues(state, operation));

    if (isa<IfOp, IndexSwitchOp>(op->getParentOp())) {
      rewriter.replaceOpWithNewOp<qco::YieldOp>(op, targets);
    } else {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op, targets);
    }

    return success();
  }
};

/**
 * @brief Converts scf.condition with memory semantics to scf.condition with
 * value semantics for qubit values
 *
 * @par Example:
 * ```mlir
 * scf.condition(%cond)
 * ```
 * is converted to
 * ```mlir
 * scf.condition(%cond) %targets
 * ```
 */
struct ConvertSCFConditionOp final
    : StatefulOpConversionPattern<scf::ConditionOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();

    SmallVector<Value> targets(op.getArgs());
    llvm::append_range(targets, resolveAllValues(state, operation));

    rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, op.getCondition(),
                                                  targets);
    return success();
  }
};

/**
 * @brief Pass implementation for QC-to-QCO conversion
 *
 * @details
 * This pass converts QC dialect operations (reference semantics) to QCO dialect
 * operations (value semantics). The conversion is essential for enabling
 * optimization passes that rely on SSA form and explicit dataflow analysis.
 *
 * The pass operates in several phases:
 * 1. Type conversion: !qc.qubit -> !qco.qubit
 * 2. Operation conversion: Each QC op is converted to its QCO equivalent
 * 3. State tracking: A LoweringState maintains qubit value mappings
 * 4. Function/control-flow adaptation: Function signatures and control flow are
 * updated to use QCO types
 *
 * The conversion maintains semantic equivalence while transforming the
 * representation from imperative (mutation-based) to functional (SSA-based).
 */
struct QCToQCO final : impl::QCToQCOBase<QCToQCO> {
  using QCToQCOBase::QCToQCOBase;

protected:
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* moduleOp = getOperation();

    // Create state object to track qubit value flow
    LoweringState state;

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QCToQCOTypeConverter typeConverter(context);

    if (failed(validateModifierBodies(moduleOp)) ||
        failed(validateQuantumValueSources(moduleOp)) ||
        failed(collectRegisterAccesses(moduleOp, state))) {
      signalPassFailure();
      return;
    }

    // Get the quantum values captured by structured control-flow regions.
    collectStructuredCaptures(moduleOp, state);

    // Configure conversion target
    target.addIllegalDialect<QCDialect>();
    target.addLegalDialect<cbit::CBitDialect, QCODialect, arith::ArithDialect,
                           qtensor::QTensorDialect>();

    target.addDynamicallyLegalDialect<memref::MemRefDialect>([](Operation* op) {
      return llvm::none_of(op->getOperandTypes(), isQubitMemrefType) &&
             llvm::none_of(op->getResultTypes(), isQubitMemrefType);
    });

    target.addDynamicallyLegalDialect<scf::SCFDialect>([&](Operation* op) {
      auto& regionQubitMap = state.regionQubitMap[op];
      auto& regionTensorMap = state.regionRegisterMap[op];
      return regionQubitMap.empty() && regionTensorMap.empty();
    });
    // Structured terminators are lowered in a second conversion phase. The
    // first phase must finish converting every operation in a region so the
    // region-local maps contain the final QCO values that the terminator has
    // to yield. Converting terminators in the same worklist would make the
    // result depend on dialect-conversion traversal order.
    target.addLegalOp<scf::YieldOp, scf::ConditionOp>();

    // Register operation conversion patterns with state tracking.
    patterns
        .add<ConvertSCFForOp, ConvertSCFWhileOp, ConvertSCFIfOp,
             ConvertSCFIndexSwitchOp, ConvertMemRefAllocOp, ConvertMemRefLoadOp,
             ConvertMemRefDeallocOp, ConvertQCAllocOp, ConvertQCDeallocOp,
             ConvertQCStaticOp, ConvertQCMeasureOp, ConvertQCResetOp,
             ConvertQCUnitaryOp, ConvertQCBarrierOp, ConvertQCCtrlOp,
             ConvertQCInvOp, ConvertQCPowOp, ConvertQCYieldOp>(typeConverter,
                                                               context, &state);

    // Not part of the central gate table.
    patterns.add<ConvertQCGateToQCO<qc::GPhaseOp, qco::GPhaseOp, 0, 1>>(
        typeConverter, context, &state);

#define MQT_GATE(KEY, NAME, OP, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)   \
  patterns.add<                                                                \
      ConvertQCGateToQCO<qc::KEY##Op, qco::KEY##Op, (TARGETS), (PARAMS)>>(     \
      typeConverter, context, &state);
#include "mlir/Conversion/GateTable.def"

    // Conversion of qc types in func.func signatures
    // Note: This currently has limitations with signature changes
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    // Conversion of qc types in func.return
    //
    // Note: `func.return` may already be type-legal even though we still need
    // to insert sink operations (`qco.sink`) for dead qubit values. Therefore,
    // we mark it illegal as long as the qubit map of the region is not empty.
    patterns.add<ConvertFuncReturnOp>(typeConverter, context, &state);
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      if (!typeConverter.isLegal(op)) {
        return false;
      }
      const auto it = state.qubitMap.find(op->getParentRegion());
      return it == state.qubitMap.end() || it->second.empty();
    });

    // Conversion of qc types in func.call
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    // Conversion of qc types in control-flow ops (e.g., cf.br, cf.cond_br)
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    // Convert structured parents and their contents first.
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Source register values and loaded qubit references have been erased.
    // Structured conversion state uses stable register identifiers from here
    // on.
    state.registerIds.clear();
    state.registerAccesses.clear();
    state.convertedQubitAliases.clear();
    state.regionQubitMap.clear();
    state.regionRegisterMap.clear();

    ConversionTarget terminatorTarget(*context);
    terminatorTarget.markUnknownOpDynamicallyLegal(
        [](Operation*) { return true; });
    terminatorTarget.addDynamicallyLegalOp<scf::YieldOp, scf::ConditionOp>(
        [&](Operation* op) {
          auto* parentOp = op->getParentOp();
          if (!state.structuredValues.contains(parentOp)) {
            return true;
          }

          if (auto condition = dyn_cast<scf::ConditionOp>(op)) {
            return llvm::equal(condition.getArgs().getTypes(),
                               parentOp->getResultTypes());
          }
          if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
            return llvm::equal(op->getOperandTypes(),
                               whileOp.getInits().getTypes());
          }
          return llvm::equal(op->getOperandTypes(), parentOp->getResultTypes());
        });

    RewritePatternSet terminatorPatterns(context);
    terminatorPatterns.add<ConvertSCFYieldOp, ConvertSCFConditionOp>(
        typeConverter, context, &state);
    if (failed(applyPartialConversion(moduleOp, terminatorTarget,
                                      std::move(terminatorPatterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir

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
#include "mqt/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::qco {

class UnitaryOpInterface;

/// Immutable numerical costs prepared once before routing trials. Construction
/// requires linear QCO IR. No IR handles survive construction; target support
/// and operand direction remain the caller's responsibility.
class NativeCostTable {
public:
  static std::shared_ptr<const NativeCostTable>
  precompute(Operation* root, CompilerTarget::GateKind entangler,
             uint64_t seed);

private:
  friend class NativeCostAnalysis;
  struct Entry {
    Matrix4x4 matrix;
    CompilerTarget::GateKind entangler;
    std::optional<uint8_t> count;
  };
  const std::optional<uint8_t>* lookup(const Matrix4x4& matrix,
                                       CompilerTarget::GateKind entangler,
                                       uint64_t hash) const;

  uint64_t seed_ = 0;
  std::vector<Entry> entries_;
  std::unordered_multimap<uint64_t, size_t> index_;
};

/// Read-only native synthesis decisions with bounded numerical caches.
/// Own one instance per traversal; it retains no IR handles or target state.
/// Supplied sites follow operand order and match the operation's arity.
/// Unavailable results cover unsupported lowering and numerical failure.
class NativeCostAnalysis {
public:
  using Sites = std::optional<ArrayRef<CompilerTarget::SiteId>>;

  /// The optional shared table must outlive this analysis. Routing uses compact
  /// counts with a local fallback; emission caches full decompositions.
  explicit NativeCostAnalysis(uint64_t seed,
                              const NativeCostTable* shared = nullptr)
      : seed_(seed), shared_(shared) {}

  /// Whether an operation is native, and whether its operands must be reversed.
  /// Only operand-swap-invariant operations can use reversed native support.
  static std::optional<bool> nativeOrientation(UnitaryOpInterface operation,
                                               const CompilerTarget& target,
                                               Sites sites);

  /// Operand direction for a synthesis entangler, or unavailable placement.
  static std::optional<bool>
  entanglerOrientation(const CompilerTarget& target,
                       CompilerTarget::GateKind entangler, Sites sites);

  /// The returned reference is invalidated by the next decomposition query.
  const std::optional<decomposition::TwoQubitNativeDecomposition>&
  decompose(const Matrix4x4& matrix, CompilerTarget::GateKind entangler);

  /// Native two-qubit count. Unavailable lowering is never a zero-cost gate.
  std::optional<size_t> operationCost(UnitaryOpInterface operation,
                                      const CompilerTarget& target,
                                      Sites sites);
  std::optional<size_t> matrixCost(const Matrix4x4& matrix,
                                   const CompilerTarget& target, Sites sites);
  std::optional<size_t> swapCost(const CompilerTarget& target,
                                 ArrayRef<CompilerTarget::SiteId> sites);

  /// Preserve individual lowering unless resynthesis strictly reduces count.
  size_t runCost(const Matrix4x4& matrix, size_t separateCost,
                 const CompilerTarget& target, Sites sites);

private:
  std::optional<uint8_t> count(const Matrix4x4& matrix,
                               CompilerTarget::GateKind entangler);

  struct DecompositionEntry {
    Matrix4x4 matrix;
    CompilerTarget::GateKind entangler;
    std::optional<decomposition::TwoQubitNativeDecomposition> native;
  };
  static constexpr size_t CACHE_SIZE = 64;
  uint64_t seed_;
  const NativeCostTable* shared_;
  std::vector<NativeCostTable::Entry> counts_;
  std::vector<uint64_t> countHashes_;
  size_t nextCount_ = 0;
  std::optional<NativeCostTable::Entry> lastCount_;
  std::vector<DecompositionEntry> decompositions_;
  std::vector<uint64_t> decompositionHashes_;
  size_t nextDecomposition_ = 0;
  Matrix4x4 matrix_;
  std::optional<CompilerTarget::GateKind> entangler_;
  std::optional<decomposition::TwoQubitNativeDecomposition> native_;
};

/// Estimate a routed block without building IR. Vertices use the target's dense
/// numbering. Pending runs occupy disjoint physical pairs; state is O(sites).
/// Depth counts qubit dependencies only, without classical scheduling.
class NativeCostTracker {
public:
  NativeCostTracker(const CompilerTarget& target, uint64_t seed,
                    const NativeCostTable* shared = nullptr);

  /// Observe one original operation, with vertices in its operand order.
  void append(Operation* operation, ArrayRef<size_t> vertices);
  /// Observe a routing SWAP before updating the logical-to-physical layout.
  void appendSwap(size_t a, size_t b);
  /// End pending runs before entering another region; retain this block's
  /// depth.
  void flush();
  /// Include a finished nested block once, without execution-frequency weights.
  void merge(NativeCostTracker& child);
  /// Finish pending runs and return count/depth, or unavailable lowering.
  std::optional<std::pair<size_t, size_t>> score();
  /// First-SWAP discount against the current prefix; does not consume the run.
  int64_t swapDiscount(size_t a, size_t b, size_t standaloneCost);

private:
  struct Run {
    Matrix4x4 matrix;
    size_t separateCost = 0;
    bool canFuse = false;
  };

  void flush(size_t vertex);
  void appendPair(const Matrix4x4& matrix, size_t cost, size_t a, size_t b);
  void charge(size_t cost, size_t a, size_t b);

  const CompilerTarget& target_;
  NativeCostAnalysis analysis_;
  SmallVector<Run, 0> runs_;
  SmallVector<size_t> partners_;
  SmallVector<size_t> depths_;
  /// Immediate canceling successors are consumed before routing advances again.
  SmallPtrSet<Operation*, 4> cancellations_;
  size_t count_ = 0;
  size_t depth_ = 0;
  bool available_ = true;
};

} // namespace mlir::qco

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

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <utility>

namespace mlir::qco {
class Qubits {
  /**
   * @brief Specifies the qubit "location" (hardware or program).
   */
  enum class QubitLocation : std::uint8_t { Hardware, Program };

public:
  /**
   * @brief Add qubit with automatically assigned dynamic index.
   */
  [[maybe_unused]] void add(TypedValue<QubitType> q);

  /**
   * @brief Add qubit with static index.
   */
  void add(TypedValue<QubitType> q, std::size_t hw);

  /**
   * @brief Remap the qubit value from prev to next.
   */
  void remap(TypedValue<QubitType> prev, TypedValue<QubitType> next);

  /**
   * @brief Remove the qubit value.
   */
  void remove(TypedValue<QubitType> q);

  /**
   * @returns the qubit value assigned to a program index.
   */
  [[nodiscard]] TypedValue<QubitType> getProgramQubit(std::size_t index) const;

  /**
   * @returns the qubit value assigned to a hardware index.
   */
  [[nodiscard]] TypedValue<QubitType> getHardwareQubit(std::size_t index) const;

private:
  DenseMap<std::size_t, TypedValue<QubitType>> programToValue_;
  DenseMap<std::size_t, TypedValue<QubitType>> hardwareToValue_;
  DenseMap<TypedValue<QubitType>, std::pair<QubitLocation, std::size_t>>
      valueToIndex_;
};

using WalkProgramFn = function_ref<WalkResult(Operation*, Qubits&)>;

/**
 * @brief Perform top-down non-recursive walk of all operations within a
 * region of a quantum program and apply a callback function.
 * @details The signature of the callback function is:
 *
 *     (Operation*, Qubits& q) -> WalkResult
 *
 * where the Qubits object tracks the front of qubit SSA values.
 * @param region The targeted region.
 * @param fn The callback function.
 */
void walkProgram(Region& region, WalkProgramFn fn);

using ReleasedOps = SmallVector<UnitaryOpInterface, 8>;
using PendingWiresMap =
    DenseMap<UnitaryOpInterface, SmallVector<std::size_t, 2>>;

struct IsReady {
  bool operator()(PendingWiresMap::value_type& kv) const {
    return kv.second.size() == kv.first.getNumQubits();
  }
};

using ReadyRange =
    decltype(make_filter_range(std::declval<PendingWiresMap&>(), IsReady{}));

using WalkProgramGraphFn =
    function_ref<WalkResult(const ReadyRange&, ReleasedOps&)>;

/**
 * @brief Specifies the layering direction.
 */
enum class ProgramGraphWalkDirection : bool { Forward, Backward };

namespace impl {
template <ProgramGraphWalkDirection Dir> struct ProgramGraphWalkTraits;

template <> struct ProgramGraphWalkTraits<ProgramGraphWalkDirection::Forward> {
  static constexpr std::size_t step() { return 1; }

  static bool proceed(const WireIterator& it) {
    return it != std::default_sentinel;
  }
};

template <> struct ProgramGraphWalkTraits<ProgramGraphWalkDirection::Backward> {
  static constexpr std::size_t step() { return -1; }

  static bool proceed(const WireIterator& it) {
    if (it.operation() == nullptr) {
      return false;
    }
    return !isa<qco::AllocOp, StaticOp, qtensor::ExtractOp>(it.operation());
  }
};
}; // namespace impl

/**
 * @brief Walk the graph-like circuit IR of QCO dialect programs.
 * @details
 * Depending on the template parameter, the function collects the
 * layers in forward or backward direction, respectively. Towards that end,
 * the function traverses the def-use chain of each qubit until a multi-qubit
 * gate (including barriers) is found. If a multi-qubit gate is visited twice,
 * it is considered ready and inserted into the layer. This process is repeated
 * until no more multi-qubit gates are found anymore.
 *
 * The signature of the callback function is:
 *
 *     (const ReadyRange&, ReleasedOps&) -> WalkResult
 *
 * The operations inserted into the parameter "released" determine which
 * multi-qubit gates are released in next iteration.
 *
 * @param wires A mutable array-ref of circuit wires (wire iterators).
 * @param direction The traversal direction.
 * @param fn The callback function.
 *
 * @returns
 *     failure(), if the callback returns WalkResult::interrupt()
 *     failure(), if the callback returns WalkResult::skipped()
 *     success(), otherwise.
 */
template <ProgramGraphWalkDirection Dir>
LogicalResult walkProgramGraph(MutableArrayRef<WireIterator> wires,
                               WalkProgramGraphFn fn) {
  using Traits = impl::ProgramGraphWalkTraits<Dir>;
  enum class CircuitWalkResult : std::uint8_t { Advance, Hold, Fail };

  ReleasedOps released;

  PendingWiresMap pending;
  pending.reserve(wires.size());

  SmallVector<std::size_t> curr(wires.size());
  std::iota(curr.begin(), curr.end(), 0UL);

  SmallVector<std::size_t> next;
  next.reserve(wires.size());

  while (!curr.empty()) {
    for (std::size_t i : curr) {
      auto& it = wires[i];
      while (Traits::proceed(it)) {
        const auto res =
            TypeSwitch<Operation*, CircuitWalkResult>(it.operation())
                .template Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
                  // If there are fewer wires than the qubit requires inputs,
                  // it's impossible to release the operation. Hence, fail.
                  if (op.getNumQubits() > wires.size()) {
                    return CircuitWalkResult::Fail;
                  }

                  if (op.getNumQubits() == 1) {
                    std::ranges::advance(it, Traits::step());
                    return CircuitWalkResult::Advance;
                  }

                  // Insert the unitary to the pending map.
                  // The caller decides if this op should be released.
                  const auto [it, inserted] = pending.try_emplace(op);
                  auto& indices = it->second;

                  if (inserted) {
                    indices.reserve(op.getNumQubits());
                  }

                  indices.emplace_back(i);

                  return CircuitWalkResult::Hold; // Stop at multi-qubit gate.
                })
                .template Case<AllocOp, StaticOp, qtensor::ExtractOp, ResetOp,
                               MeasureOp, SinkOp, qtensor::InsertOp>([&](auto) {
                  std::ranges::advance(it, Traits::step());
                  return CircuitWalkResult::Advance;
                })
                .Default([&](Operation* op) {
                  const auto name = op->getName().getStringRef();
                  report_fatal_error("unknown op encountered: " + name);
                  return CircuitWalkResult::Fail;
                });

        if (res == CircuitWalkResult::Hold) {
          break;
        }

        if (res == CircuitWalkResult::Fail) {
          return failure();
        }
      }
    }

    released.clear();
    const auto ready = make_filter_range(pending, IsReady{});
    const auto res = std::invoke(fn, ready, released);
    if (res.wasInterrupted() || res.wasSkipped()) {
      return failure();
    }

    for (UnitaryOpInterface op : released) {
      const auto& mapIt = pending.find(op);
      assert(mapIt != pending.end());

      auto& indices = mapIt->second;
      for (std::size_t i : mapIt->second) {
        std::ranges::advance(wires[i], Traits::step());
        next.emplace_back(i);
      }

      pending.erase(mapIt);
    }

    curr.swap(next);
    next.clear();
  }

  return success();
}
} // namespace mlir::qco

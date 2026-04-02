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

#include <llvm/ADT/ADL.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <cstddef>
#include <iterator>
#include <utility>

namespace mlir::qco {
/**
 * @brief Specifies the layering direction.
 */
enum class WalkDirection : std::uint8_t { Forward, Backward };

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
  [[maybe_unused]] [[nodiscard]] TypedValue<QubitType>
  getProgramQubit(std::size_t index) const;

  /**
   * @returns the qubit value assigned to a hardware index.
   */
  [[nodiscard]] TypedValue<QubitType> getHardwareQubit(std::size_t index) const;

  /**
   * @returns the hardware index assigned to the given qubit value.
   */
  [[nodiscard]] std::size_t getHardwareIndex(TypedValue<QubitType> q) const;

private:
  DenseMap<std::size_t, TypedValue<QubitType>> programToValue_;
  DenseMap<std::size_t, TypedValue<QubitType>> hardwareToValue_;
  DenseMap<TypedValue<QubitType>, std::pair<QubitLocation, std::size_t>>
      valueToIndex_;
};

/**
 * @brief Perform top-down non-recursive walk of all operations within a
 * region and apply callback function.
 * @details The signature of the callback function is:
 *
 *     (Operation*, Qubits& q) -> WalkResult
 *
 * where the Qubits object tracks the front of qubit SSA values.
 * @param region The targeted region.
 * @param fn The callback function.
 */
template <typename Fn> void walkUnit(Region& region, Fn&& fn) {
  const auto ffn = std::forward<Fn>(fn);

  Qubits qubits;
  for (Operation& curr : region.getOps()) {
    if (ffn(&curr, qubits).wasInterrupted()) {
      break;
    };

    TypeSwitch<Operation*>(&curr)
        .template Case<StaticOp>(
            [&](StaticOp op) { qubits.add(op.getQubit(), op.getIndex()); })
        .template Case<AllocOp>([&](AllocOp op) { qubits.add(op.getResult()); })
        .template Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
          for (const auto& [prevV, nextV] :
               llvm::zip(op.getInputQubits(), op.getOutputQubits())) {
            const auto prevQ = cast<TypedValue<QubitType>>(prevV);
            const auto nextQ = cast<TypedValue<QubitType>>(nextV);
            qubits.remap(prevQ, nextQ);
          }
        })
        .template Case<ResetOp>([&](ResetOp op) {
          qubits.remap(op.getQubitIn(), op.getQubitOut());
        })
        .template Case<MeasureOp>([&](MeasureOp op) {
          qubits.remap(op.getQubitIn(), op.getQubitOut());
        })
        .template Case<SinkOp>(
            [&](SinkOp op) { qubits.remove(op.getQubit()); });
  }
}

namespace impl {
/**
 * @returns true if the wire iterator has not reached the end (Forward) or the
 * start (Backward) of the wire.
 */
template <WalkDirection d> static bool proceedOnWire(const WireIterator& it) {
  if constexpr (d == WalkDirection::Forward) {
    return it != std::default_sentinel;
  } else {
    return !isa<AllocOp>(it.operation()) && !isa<StaticOp>(it.operation());
  }
}

/**
 * @brief Skip the next two-qubit block of two wires.
 * @details Advances each of the two wire iterators until a two-qubit op is
 * found. If the ops match, repeat this process. Otherwise, stop.
 */
template <WalkDirection d>
static void skipTwoQubitBlock(WireIterator& first, WireIterator& second) {
  constexpr auto step = d == WalkDirection::Forward ? 1 : -1;

  const auto advanceUntilTwoQubitOp = [&](WireIterator& it) {
    while (proceedOnWire<d>(it)) {
      if (auto op = dyn_cast<UnitaryOpInterface>(it.operation())) {
        if (op.getNumQubits() > 1) {
          break;
        }
      }

      std::ranges::advance(it, step);
    }
  };

  while (true) {
    advanceUntilTwoQubitOp(first);
    advanceUntilTwoQubitOp(second);

    if (!proceedOnWire<d>(first) || !proceedOnWire<d>(second)) {
      break;
    }

    if (first.operation() != second.operation()) {
      break;
    }

    std::ranges::advance(first, step);
    std::ranges::advance(second, step);
  }
}

using PendingWiresMap =
    DenseMap<UnitaryOpInterface, SmallVector<WireIterator*, 2>>;

std::optional<ArrayRef<WireIterator*>>
tryReleaseReadyWires(PendingWiresMap& map, UnitaryOpInterface op,
                     WireIterator* wire);
}; // namespace impl

template <WalkDirection d, typename Fn>
void walkLayers(Region& region, Fn&& fn) {
  constexpr auto step = d == WalkDirection::Forward ? 1 : -1;
  const auto ffn = std::forward<Fn>(fn);

  Qubits qubits;
  impl::PendingWiresMap pending;
  SmallVector<WireIterator*> hold;
  SmallVector<UnitaryOpInterface> front;
  SmallVector<WireIterator> wires;

  // Collect the qubits.
  const auto dynamicOps = region.getOps<AllocOp>();
  const auto staticOps = region.getOps<StaticOp>();

  if (!staticOps.empty()) { // Static Addressing.
    assert(dynamicOps.empty() && "Mixing addressing modes is invalid.");
    wires.reserve(llvm::range_size(staticOps));
    for_each(staticOps,
             [&](StaticOp op) { wires.emplace_back(op.getQubit()); });
  } else { // Dynamic Addressing.
    assert(staticOps.empty() && "Mixing addressing modes is invalid.");
    wires.reserve(llvm::range_size(dynamicOps));
    for_each(dynamicOps,
             [&](AllocOp op) { wires.emplace_back(op.getResult()); });
  }

  pending.reserve(wires.size());
  front.reserve((wires.size() + 1) / 2);

  while (true) {
    for (WireIterator& it : wires) {
      while (impl::proceedOnWire<d>(it)) {
        const auto res =
            TypeSwitch<Operation*, WalkResult>(it.operation())
                .template Case<BarrierOp>([&](BarrierOp op) {
                  if (auto released =
                          impl::tryReleaseReadyWires(pending, op, &it)) {
                    for (const auto& [in, out] :
                         llvm::zip(op.getInputQubits(), op.getOutputQubits())) {
                      const auto inQ = cast<TypedValue<QubitType>>(in);
                      const auto outQ = cast<TypedValue<QubitType>>(out);
                      qubits.remap(inQ, outQ);
                    }
                    for (WireIterator* wire : *released) {
                      std::ranges::advance(*wire, step);
                    }
                    return WalkResult::advance();
                  }

                  return WalkResult::interrupt();
                })
                .template Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
                  assert(op.getNumQubits() > 0 && op.getNumQubits() <= 2);

                  if (op.getNumQubits() == 1) {
                    const auto in = op.getInputQubit(0);
                    const auto out = op.getOutputQubit(0);
                    const auto inQ = cast<TypedValue<QubitType>>(in);
                    const auto outQ = cast<TypedValue<QubitType>>(out);
                    qubits.remap(inQ, outQ);
                    std::ranges::advance(it, step);
                    return WalkResult::advance();
                  }

                  // Ready two-qubit gate?
                  if (auto released =
                          impl::tryReleaseReadyWires(pending, op, &it)) {
                    front.emplace_back(op);
                    for (WireIterator* x : *released) {
                      hold.emplace_back(x);
                    }
                  }

                  return WalkResult::interrupt();
                })
                .template Case<AllocOp>([&](AllocOp op) {
                  qubits.add(op.getResult());
                  std::ranges::advance(it, step);
                  return WalkResult::advance();
                })
                .template Case<StaticOp>([&](StaticOp op) {
                  qubits.add(op.getQubit(), op.getIndex());
                  std::ranges::advance(it, step);
                  return WalkResult::advance();
                })
                .template Case<ResetOp>([&](ResetOp op) {
                  qubits.remap(op.getQubitIn(), op.getQubitOut());
                  std::ranges::advance(it, step);
                  return WalkResult::advance();
                })
                .template Case<MeasureOp>([&](MeasureOp op) {
                  qubits.remap(op.getQubitIn(), op.getQubitOut());
                  std::ranges::advance(it, step);
                  return WalkResult::advance();
                })
                .template Case<SinkOp>([&](SinkOp op) {
                  qubits.remove(op.getQubit());
                  std::ranges::advance(it, step);
                  return WalkResult::advance();
                })
                .Default([&](Operation* op) {
                  const auto name = op->getName().getStringRef();
                  report_fatal_error("unknown op encountered: " + name);
                  return WalkResult::interrupt();
                });

        if (res.wasInterrupted()) {
          break;
        }
      }
    }

    if (front.empty()) {
      break;
    }

    if (ffn(front, qubits).wasInterrupted()) {
      break;
    };

    // Jump over ready two-qubit operations.
    for (UnitaryOpInterface op : front) {
      for (const auto& [in, out] :
           llvm::zip(op.getInputQubits(), op.getOutputQubits())) {
        const auto inQ = cast<TypedValue<QubitType>>(in);
        const auto outQ = cast<TypedValue<QubitType>>(out);
        qubits.remap(inQ, outQ);
      }
    }

    for (WireIterator* it : hold) {
      std::ranges::advance(*it, step);
    }

    front.clear();
    hold.clear();
    pending.clear();
  }
}
} // namespace mlir::qco

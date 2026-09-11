/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QC/Translation/MeasurementStores.h"

#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/QC/IR/QCInterfaces.h"
#include "mqt/Dialect/QC/IR/QCOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::qc {

static cbit::StoreOp measurementDestination(MeasureOp measure) {
  cbit::StoreOp destination;
  for (auto* user : measure.getResult().getUsers()) {
    if (auto store = dyn_cast<cbit::StoreOp>(user)) {
      if (destination) {
        return {};
      }
      destination = store;
    }
  }
  return destination;
}

DenseMap<Operation*, cbit::StoreOp> findMeasurementStores(func::FuncOp function,
                                                          bool allowOtherUses) {
  DenseMap<Operation*, cbit::StoreOp> destinations;
  function.walk([&](Block* block) {
    auto measurements = block->getOps<qc::MeasureOp>();
    if (llvm::all_of(measurements, [](MeasureOp measure) {
          auto store = measurementDestination(measure);
          return !store || store == measure->getNextNode();
        })) {
      for (auto measure : measurements) {
        if (auto store = measurementDestination(measure);
            store && (allowOtherUses || measure.getResult().hasOneUse())) {
          destinations.try_emplace(measure, store);
        }
      }
      return;
    }
    struct Accesses {
      SmallVector<size_t> positions;
      size_t next = 0U;
    };
    Accesses unknownAccesses;
    DenseMap<Value, Accesses> registerAccesses;
    DenseMap<std::pair<Value, int64_t>, Accesses> bitAccesses;
    DenseMap<Operation*, size_t> positions;
    for (auto [position, operation] : llvm::enumerate(*block)) {
      positions[&operation] = position;
      operation.walk<WalkOrder::PreOrder>([&](Operation* candidate) {
        return TypeSwitch<Operation*, WalkResult>(candidate)
            .Case([](qc::UnitaryOpInterface) {
              /// Verified unitary regions cannot access classical memory.
              return WalkResult::skip();
            })
            .Case([&](MemoryEffectOpInterface mem) {
              SmallVector<MemoryEffects::EffectInstance> effects;
              mem.getEffects(effects);
              const auto bit =
                  TypeSwitch<Operation*, std::optional<int64_t>>(candidate)
                      .Case<cbit::LoadOp, cbit::StoreOp>([](auto access) {
                        return getConstantIntValue(access.getIndex());
                      })
                      .Default(std::nullopt);
              for (const auto& effect : effects) {
                auto value = effect.getValue();
                if (!value) {
                  unknownAccesses.positions.push_back(position);
                } else if (bit) {
                  bitAccesses[{value, *bit}].positions.push_back(position);
                } else {
                  registerAccesses[value].positions.push_back(position);
                }
              }
              return WalkResult::advance();
            })
            .Default([&](Operation* op) {
              if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
                unknownAccesses.positions.push_back(position);
              }
              return WalkResult::advance();
            });
      });
    }
    llvm::SmallBitVector moved(positions.size());
    const auto conflicts = [&](Accesses& accesses, size_t measurement,
                               size_t destination) {
      /// Queries follow block order. Relocated stores precede every later
      /// query.
      while (accesses.next < accesses.positions.size() &&
             (accesses.positions[accesses.next] <= measurement ||
              moved[accesses.positions[accesses.next]])) {
        ++accesses.next;
      }
      return accesses.next < accesses.positions.size() &&
             accesses.positions[accesses.next] < destination;
    };
    for (auto measure : measurements) {
      auto destination = measurementDestination(measure);
      if (!destination ||
          (!allowOtherUses && !measure.getResult().hasOneUse())) {
        continue;
      }
      if (destination == measure->getNextNode()) {
        destinations.try_emplace(measure, destination);
        moved.set(positions.at(destination));
        continue;
      }
      const auto index = getConstantIntValue(destination.getIndex());
      if (!index) {
        continue;
      }
      if (destination->getBlock() != block ||
          positions.at(destination) <= positions.at(measure)) {
        continue;
      }
      const auto measurementPosition = positions.at(measure);
      const auto destinationPosition = positions.at(destination);
      const auto reg = registerAccesses.find(destination.getReg());
      const auto bit = bitAccesses.find({destination.getReg(), *index});
      if (conflicts(unknownAccesses, measurementPosition,
                    destinationPosition) ||
          (reg != registerAccesses.end() &&
           conflicts(reg->second, measurementPosition, destinationPosition)) ||
          (bit != bitAccesses.end() &&
           conflicts(bit->second, measurementPosition, destinationPosition))) {
        continue;
      }
      destinations.try_emplace(measure, destination);
      moved.set(destinationPosition);
    }
  });
  return destinations;
}

} // namespace mlir::qc

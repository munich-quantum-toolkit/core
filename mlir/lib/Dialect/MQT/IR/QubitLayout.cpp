/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/IR/QubitLayout.h"

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <vector>

using namespace mlir;
using namespace mlir::mqt;

DictionaryAttr QubitLayout::toAttr(MLIRContext* context) const {
  Builder builder(context);
  SmallVector<Attribute> groups;
  for (const auto& reg : registers) {
    groups.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("name", builder.getStringAttr(reg.name)),
        builder.getNamedAttr("slots", builder.getDenseI64ArrayAttr(reg.slots)),
        builder.getNamedAttr("ancillary", builder.getBoolAttr(reg.ancillary)),
    }));
  }
  SmallVector<NamedAttribute> fields{
      builder.getNamedAttr("physical_size",
                           builder.getI64IntegerAttr(physicalSize)),
      builder.getNamedAttr("initial", builder.getDenseI64ArrayAttr(initial)),
      builder.getNamedAttr("output_order",
                           builder.getDenseI64ArrayAttr(outputOrder)),
      builder.getNamedAttr("ancillas", builder.getDenseI64ArrayAttr(ancillas)),
      builder.getNamedAttr("registers", builder.getArrayAttr(groups)),
  };
  if (routing) {
    fields.push_back(builder.getNamedAttr(
        "routing", builder.getDenseI64ArrayAttr(*routing)));
  }
  if (inputCount) {
    fields.push_back(builder.getNamedAttr(
        "input_count", builder.getI64IntegerAttr(*inputCount)));
  }
  return builder.getDictionaryAttr(fields);
}

FailureOr<QubitLayout>
QubitLayout::fromAttr(Attribute attribute,
                      llvm::function_ref<InFlightDiagnostic()> emitError) {
  const auto dict = dyn_cast_or_null<DictionaryAttr>(attribute);
  if (!dict) {
    emitError() << "qubit layout must be a dictionary";
    return failure();
  }
  for (const auto field : dict) {
    if (!llvm::StringSwitch<bool>(field.getName().getValue())
             .Cases(
                 {
                     "physical_size",
                     "initial",
                     "routing",
                     "output_order",
                     "input_count",
                     "ancillas",
                     "registers",
                 },
                 true)
             .Default(false)) {
      emitError() << "unknown qubit layout field '" << field.getName() << "'";
      return failure();
    }
  }
  const auto size = dict.getAs<IntegerAttr>("physical_size");
  const auto initialAttr = dict.getAs<DenseI64ArrayAttr>("initial");
  const auto output = dict.getAs<DenseI64ArrayAttr>("output_order");
  const auto auxiliary = dict.getAs<DenseI64ArrayAttr>("ancillas");
  const auto groups = dict.getAs<ArrayAttr>("registers");
  if (!size || !size.getType().isSignlessInteger(64) || size.getInt() < 0 ||
      !initialAttr || !output || !auxiliary || !groups) {
    emitError() << "qubit layout requires physical_size, initial, "
                   "output_order, ancillas, and registers";
    return failure();
  }
  const auto validIndices = [](ArrayRef<int64_t> values, int64_t limit,
                               bool partial) {
    llvm::SmallDenseSet<int64_t> seen;
    return llvm::all_of(values, [&](int64_t value) {
      return (partial && value == -1) ||
             (value >= 0 && value < limit && seen.insert(value).second);
    });
  };
  if (!validIndices(initialAttr.asArrayRef(), size.getInt(), true) ||
      output.size() != size.getInt() ||
      !validIndices(output.asArrayRef(), size.getInt(), false) ||
      !validIndices(auxiliary.asArrayRef(), initialAttr.size(), false)) {
    emitError() << "qubit layout indices must be distinct and refer to "
                   "existing resources";
    return failure();
  }
  QubitLayout result{
      .physicalSize = size.getInt(),
      .initial = std::vector<int64_t>(initialAttr.asArrayRef().begin(),
                                      initialAttr.asArrayRef().end()),
      .outputOrder = std::vector<int64_t>(output.asArrayRef().begin(),
                                          output.asArrayRef().end()),
      .ancillas = std::vector<int64_t>(auxiliary.asArrayRef().begin(),
                                       auxiliary.asArrayRef().end()),
  };
  if (const auto raw = dict.get("routing")) {
    const auto routingAttr = dyn_cast<DenseI64ArrayAttr>(raw);
    if (!routingAttr || routingAttr.size() != size.getInt() ||
        !validIndices(routingAttr.asArrayRef(), size.getInt(), true)) {
      emitError() << "qubit layout routing must map physical resources to "
                     "distinct final positions";
      return failure();
    }
    result.routing.emplace(routingAttr.asArrayRef().begin(),
                           routingAttr.asArrayRef().end());
  }
  if (const auto raw = dict.get("input_count")) {
    const auto count = dyn_cast<IntegerAttr>(raw);
    if (!count || !count.getType().isSignlessInteger(64) ||
        count.getInt() < 0 || count.getInt() > initialAttr.size()) {
      emitError() << "qubit layout input_count exceeds its logical inputs";
      return failure();
    }
    result.inputCount = count.getInt();
  }
  llvm::StringSet<> names;
  llvm::SmallDenseSet<int64_t> registered;
  for (const auto raw : groups) {
    const auto group = dyn_cast<DictionaryAttr>(raw);
    const auto name = group ? group.getAs<StringAttr>("name") : StringAttr{};
    const auto slots =
        group ? group.getAs<DenseI64ArrayAttr>("slots") : DenseI64ArrayAttr{};
    const auto ancillary =
        group ? group.getAs<BoolAttr>("ancillary") : BoolAttr{};
    if (!group || group.size() != 3 || !name || name.getValue().empty() ||
        name.getValue().contains('\0') ||
        !names.insert(name.getValue()).second || !slots || slots.empty() ||
        !ancillary ||
        !validIndices(slots.asArrayRef(), initialAttr.size(), true)) {
      emitError() << "qubit layout register groups require unique names and "
                     "valid slots";
      return failure();
    }
    for (const auto slot : slots.asArrayRef()) {
      if (slot == -1) {
        continue;
      }
      if (!registered.insert(slot).second ||
          ancillary.getValue() != llvm::is_contained(result.ancillas, slot)) {
        emitError()
            << "qubit layout register membership or ancillary status conflicts";
        return failure();
      }
    }
    result.registers.push_back({
        .name = name.getValue().str(),
        .slots = std::vector<int64_t>(slots.asArrayRef().begin(),
                                      slots.asArrayRef().end()),
        .ancillary = ancillary.getValue(),
    });
  }
  return result;
}

void mlir::mqt::invalidateQubitLayout(ModuleOp moduleOp) {
  moduleOp.walk([](ModuleOp nested) {
    if (nested->removeAttr("mqt.layout")) {
      nested->setAttr("mqt.layout_invalidated",
                      UnitAttr::get(nested.getContext()));
    }
  });
}

void mlir::mqt::discardQubitLayout(ModuleOp moduleOp) {
  moduleOp->removeAttr("mqt.layout");
  moduleOp->removeAttr("mqt.layout_invalidated");
}

LogicalResult mlir::mqt::requireNoQubitLayout(ModuleOp moduleOp) {
  if (moduleOp->hasAttr("mqt.layout") ||
      moduleOp->hasAttr("mqt.layout_invalidated")) {
    return moduleOp.emitError("output cannot preserve qubit layout metadata; "
                              "explicitly discard the layout before export");
  }
  return success();
}

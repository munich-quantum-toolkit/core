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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::mqt;

DictionaryAttr QubitLayout::toAttr(MLIRContext* context) const {
  Builder builder(context);
  SmallVector<NamedAttribute> fields{
      builder.getNamedAttr("initial", builder.getDenseI64ArrayAttr(initial)),
      builder.getNamedAttr("input_count",
                           builder.getI64IntegerAttr(inputCount)),
  };
  if (routing) {
    fields.push_back(builder.getNamedAttr(
        "routing", builder.getDenseI64ArrayAttr(*routing)));
  }
  if (sites) {
    fields.push_back(
        builder.getNamedAttr("sites", builder.getDenseI64ArrayAttr(*sites)));
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
  if (dict.size() != 2 + static_cast<size_t>(dict.get("routing") != nullptr) +
                         static_cast<size_t>(dict.get("sites") != nullptr)) {
    emitError() << "qubit layout has unsupported fields";
    return failure();
  }
  const auto initialAttr = dict.getAs<DenseI64ArrayAttr>("initial");
  const auto count = dict.getAs<IntegerAttr>("input_count");
  if (!initialAttr || !count || !count.getType().isSignlessInteger(64) ||
      count.getInt() < 0 || count.getInt() > initialAttr.size()) {
    emitError() << "qubit layout requires initial and a valid input_count";
    return failure();
  }
  const auto size = initialAttr.size();
  const auto validPermutation = [size](ArrayRef<int64_t> values) {
    llvm::SmallDenseSet<int64_t> seen;
    return std::cmp_equal(values.size(), size) &&
           llvm::all_of(values, [&](int64_t value) {
             return value >= 0 && value < size && seen.insert(value).second;
           });
  };
  if (!validPermutation(initialAttr.asArrayRef())) {
    emitError() << "qubit layout initial must be a complete permutation";
    return failure();
  }
  QubitLayout result{
      .initial = std::vector<int64_t>(initialAttr.asArrayRef().begin(),
                                      initialAttr.asArrayRef().end()),
      .inputCount = count.getInt(),
  };
  if (const auto raw = dict.get("routing")) {
    const auto routingAttr = dyn_cast<DenseI64ArrayAttr>(raw);
    if (!routingAttr || !validPermutation(routingAttr.asArrayRef())) {
      emitError() << "qubit layout routing must be a complete permutation";
      return failure();
    }
    result.routing.emplace(routingAttr.asArrayRef().begin(),
                           routingAttr.asArrayRef().end());
  }
  if (const auto raw = dict.get("sites")) {
    const auto sitesAttr = dyn_cast<DenseI64ArrayAttr>(raw);
    llvm::SmallDenseSet<int64_t> seen;
    if (!sitesAttr || sitesAttr.size() != size ||
        !llvm::all_of(sitesAttr.asArrayRef(), [&](int64_t site) {
          return site >= 0 && seen.insert(site).second;
        })) {
      emitError() << "qubit layout sites must contain one distinct nonnegative "
                     "site ID per position";
      return failure();
    }
    result.sites.emplace(sitesAttr.asArrayRef().begin(),
                         sitesAttr.asArrayRef().end());
  }
  return result;
}

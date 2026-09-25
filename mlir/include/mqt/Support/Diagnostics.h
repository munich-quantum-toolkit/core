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

#include "support/Diagnostics.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <utility>

namespace mqt {
/// Consume an upstream error at the call boundary; native results carry no
/// error.
inline mlir::LogicalResult emitError(llvm::Error error) {
  return emitError(llvm::toString(std::move(error)));
}
} // namespace mqt

namespace mlir {
/// Preserve native metadata without adding it to the printed diagnostic text.
inline void emitNativeDiagnostic(Operation* op,
                                 const ::mqt::Diagnostic& native) {
  auto diagnostic = [&] {
    if (native.severity == ::mqt::DiagnosticSeverity::Warning) {
      return op->emitWarning(native.message);
    }
    if (native.severity == ::mqt::DiagnosticSeverity::Info) {
      return op->emitRemark(native.message);
    }
    return op->emitError(native.message);
  }();
  auto* context = op->getContext();
  auto integer = IntegerType::get(context, 64);
  SmallVector<NamedAttribute> metadata{
      NamedAttribute(
          StringAttr::get(context, "mqt.error_category"),
          IntegerAttr::get(integer, static_cast<int64_t>(native.category))),
  };
  if (native.status) {
    metadata.emplace_back(StringAttr::get(context, "mqt.qdmi_status"),
                          IntegerAttr::get(integer, *native.status));
  }
  diagnostic.getUnderlyingDiagnostic()->getMetadata().emplace_back(
      DictionaryAttr::get(context, metadata));
}

/// Convert MLIR diagnostics at a native boundary, retaining attached metadata.
inline ::mqt::Diagnostic toNativeDiagnostic(
    Diagnostic& diagnostic,
    ::mqt::ErrorCategory fallback = ::mqt::ErrorCategory::Runtime) {
  ::mqt::Diagnostic native{.message = {}, .category = fallback};
  llvm::raw_string_ostream stream(native.message);
  if (!llvm::isa<UnknownLoc>(diagnostic.getLocation())) {
    stream << diagnostic.getLocation() << ": ";
  }
  diagnostic.print(stream);
  if (diagnostic.getSeverity() == DiagnosticSeverity::Warning) {
    native.severity = ::mqt::DiagnosticSeverity::Warning;
  } else if (diagnostic.getSeverity() != DiagnosticSeverity::Error) {
    native.severity = ::mqt::DiagnosticSeverity::Info;
  }
  for (const auto& argument : diagnostic.getMetadata()) {
    if (argument.getKind() !=
        DiagnosticArgument::DiagnosticArgumentKind::Attribute) {
      continue;
    }
    auto metadata = llvm::dyn_cast<DictionaryAttr>(argument.getAsAttribute());
    if (!metadata) {
      continue;
    }
    if (auto category = metadata.getAs<IntegerAttr>("mqt.error_category");
        category && category.getInt() >= 0 &&
        category.getInt() <=
            static_cast<int64_t>(::mqt::ErrorCategory::NotSupported)) {
      native.category = static_cast<::mqt::ErrorCategory>(category.getInt());
    }
    if (auto status = metadata.getAs<IntegerAttr>("mqt.qdmi_status");
        status && std::in_range<int>(status.getInt())) {
      native.status = static_cast<int>(status.getInt());
    }
  }
  return native;
}

} // namespace mlir

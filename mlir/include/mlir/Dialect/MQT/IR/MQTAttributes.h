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

#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/IR/MQTEnums.h.inc" // IWYU pragma: export

#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include <cstdint>
#include <optional>

namespace mlir::mqt::detail {
template <typename Element>
[[nodiscard]] FailureOr<llvm::SmallVector<Element>>
parseArray(AsmParser& parser) {
  llvm::SmallVector<Element> elements;
  const auto parseElement = [&]() -> ParseResult {
    auto element = FieldParser<Element>::parse(parser);
    if (failed(element)) {
      return failure();
    }
    elements.emplace_back(std::move(*element));
    return success();
  };
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                            parseElement))) {
    return failure();
  }
  return elements;
}

template <typename Element>
void printArray(AsmPrinter& printer, const llvm::ArrayRef<Element> elements) {
  printer << '[';
  llvm::interleaveComma(elements, printer, [&](const Element element) {
    printer.printStrippedAttrOrType(element);
  });
  printer << ']';
}
} // namespace mlir::mqt::detail

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MQT/IR/MQTAttributes.h.inc" // IWYU pragma: export

/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

namespace mqt::ir::common {
/// @brief Parse operand qubit types.
inline mlir::ParseResult
parseQubitTypes(mlir::OpAsmParser& parser,
                llvm::SmallVectorImpl<mlir::Type>& qubits,
                llvm::SmallVectorImpl<mlir::Type>& posCtrlQubits,
                llvm::SmallVectorImpl<mlir::Type>& negCtrlQubits) {

  // No ":" delimiter -> no output types.
  if (parser.parseOptionalColon().failed()) {
    return mlir::success();
  }

  // Also allow `: ()`
  if (parser.parseOptionalLParen().succeeded()) {
    if (parser.parseRParen().failed()) {
      return mlir::failure();
    }
    return mlir::success();
  }

  // Parse the type of the target (there is no `parseOptionalTypeList` method so
  // we need to do this manually).
  mlir::Type target;
  if (parser.parseOptionalType(target).has_value()) {
    qubits.push_back(target);
    while (parser.parseOptionalComma().succeeded()) {
      if (parser.parseType(target).failed()) {
        return mlir::failure();
      }
      qubits.push_back(target);
    }
  }

  // Parse the control and negated control qubits if the corresponding keyword
  // exists.
  if (parser.parseOptionalKeyword("ctrl").succeeded()) {
    if (parser.parseTypeList(posCtrlQubits).failed()) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected at least one type after `ctrl` keyword");
      return mlir::failure();
    }
  }

  if (parser.parseOptionalKeyword("nctrl").succeeded()) {
    if (parser.parseTypeList(negCtrlQubits).failed()) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected at least one type after `nctrl` keyword");
      return mlir::failure();
    }
  }

  // If no types were parsed, this corresponds to e.g. like `mqtopt.i() %q :`
  if (qubits.empty() && posCtrlQubits.empty() && negCtrlQubits.empty()) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected at least one type after `:`");
    return mlir::failure();
  }

  return mlir::success();
}

/// @brief Print operand qubit types.
inline void printQubitTypes(mlir::OpAsmPrinter& printer,
                            mlir::Operation* /*op*/, mlir::TypeRange qubits,
                            mlir::TypeRange posCtrlQubits,
                            mlir::TypeRange negCtrlQubits) {
  const auto printTypes = [&printer](mlir::TypeRange rng) {
    llvm::interleaveComma(llvm::make_range(rng.begin(), rng.end()),
                          printer.getStream(),
                          [&printer](mlir::Type t) { printer.printType(t); });
  };

  if (qubits.empty() && posCtrlQubits.empty() && negCtrlQubits.empty()) {
    return;
  }

  printer << ": ";
  printTypes(qubits);

  if (!posCtrlQubits.empty()) {
    if (!qubits.empty()) {
      printer << " ";
    }
    printer << "ctrl ";
    printTypes(posCtrlQubits);
  }

  if (!negCtrlQubits.empty()) {
    if (!posCtrlQubits.empty() || !qubits.empty()) {
      printer << " ";
    }
    printer << "nctrl ";
    printTypes(negCtrlQubits);
  }
}
} // namespace mqt::ir::common

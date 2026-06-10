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

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

#include <cassert>
#include <cstddef>
#include <iterator>
#include <variant>

namespace mlir::utils {

constexpr auto TOLERANCE = 1e-15;

inline Value constantFromScalar(OpBuilder& builder, Location loc, double v) {
  return arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(v));
}

inline Value constantFromScalar(OpBuilder& builder, Location loc, int64_t v) {
  return arith::ConstantOp::create(builder, loc, builder.getIndexAttr(v));
}

inline Value constantFromScalar(OpBuilder& builder, Location loc, bool v) {
  return arith::ConstantOp::create(builder, loc, builder.getBoolAttr(v));
}

/**
 * @brief Convert a variant parameter (T or Value) to a Value.
 *
 * @param builder The operation builder.
 * @param loc The location of the operation.
 * @param parameter The parameter as a variant (T or Value).
 * @return Value The parameter as a Value.
 */
template <typename T>
[[nodiscard]] Value variantToValue(OpBuilder& builder, Location loc,
                                   const std::variant<T, Value>& parameter) {
  if (std::holds_alternative<Value>(parameter)) {
    return std::get<Value>(parameter);
  }
  return constantFromScalar(builder, loc, std::get<T>(parameter));
}

/**
 * @brief Try to convert a mlir::Value to a standard C++ double
 *
 * @details
 * Resolving the mlir::Value will only work if it is a static value, so a value
 * defined via a "arith.constant" operation. It must also be of type
 * float or integer.
 */
[[nodiscard]] inline std::optional<double> valueToDouble(Value value) {
  auto constantOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constantOp) {
    return std::nullopt;
  }
  if (auto floatAttr = dyn_cast<FloatAttr>(constantOp.getValue())) {
    return floatAttr.getValueAsDouble();
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
    if (intAttr.getType().isUnsignedInteger()) {
      return static_cast<double>(intAttr.getValue().getZExtValue());
    }
    // interpret both signed+signless as signed integers
    return static_cast<double>(intAttr.getValue().getSExtValue());
  }
  return std::nullopt;
}

template <typename QubitType>
[[nodiscard]]
ParseResult
parseTargetAliasing(OpAsmParser& parser, Region& region,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands) {
  // 1. Parse the opening parenthesis
  if (parser.parseLParen()) {
    return failure();
  }

  // Temporary storage for block arguments we are about to create
  SmallVector<OpAsmParser::Argument> blockArgs;

  // 2. Prepare to parse the list
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::Argument newArg;              // The "new" variable name
      OpAsmParser::UnresolvedOperand oldOperand; // The "old" input variable

      // Parse "%new"
      if (parser.parseArgument(newArg)) {
        return failure();
      }

      // Parse "="
      if (parser.parseEqual()) {
        return failure();
      }

      // Parse "%old"
      if (parser.parseOperand(oldOperand)) {
        return failure();
      }
      operands.push_back(oldOperand);

      // Hard-code QubitType since targets in CtrlOp are always qubits.
      // This avoids double-binding type($targets_in) in the assembly format
      // while keeping the parser simple and the assembly format clean.
      newArg.type = QubitType::get(parser.getBuilder().getContext());
      blockArgs.push_back(newArg);

    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen()) {
      return failure();
    }
  }

  // 4. Parse the Region
  // We explicitly pass the blockArgs we just parsed so they become the entry
  // block!
  if (parser.parseRegion(region, blockArgs)) {
    return failure();
  }

  return success();
}

inline void printTargetAliasing(OpAsmPrinter& printer, Region& region,
                                OperandRange targetsIn) {
  printer << "(";
  if (region.empty()) {
    printer << ") ";
    printer.printRegion(region, false);
    return;
  }
  Block& entryBlock = region.front();

  const auto numTargets = targetsIn.size();
  for (unsigned i = 0; i < numTargets; ++i) {
    if (i > 0) {
      printer << ", ";
    }
    printer.printOperand(entryBlock.getArgument(i));
    printer << " = ";
    printer.printOperand(targetsIn[i]);
  }
  printer << ") ";

  printer.printRegion(region, false);
}

/**
 * @brief Get the value corresponding to @p qubit from the block arguments @p
 * qubits if @p qubit is a block argument, otherwise return @p qubit itself.
 */
inline Value getValueFromBlockArgument(Value qubit, ValueRange qubits) {
  if (auto blockArg = dyn_cast<BlockArgument>(qubit)) {
    return qubits[blockArg.getArgNumber()];
  }
  return qubit;
}

/**
 * @brief Create a mapping between block arguments and qubit values.
 *
 * @details This helper function is used to resolve block arguments for nested
 * modifiers.
 */
inline void populateMapping(IRMapping& mapping, Block& block,
                            ValueRange innerQubits, ValueRange outerQubits,
                            ValueRange newQubits, ValueRange qubitArgs) {
  assert(innerQubits.size() == block.getNumArguments() &&
         "Size of innerQubits must match number of block arguments");
  for (auto arg : block.getArguments()) {
    auto innerQubit = innerQubits[arg.getArgNumber()];
    auto outerQubit = getValueFromBlockArgument(innerQubit, outerQubits);
    if (auto it = llvm::find(newQubits, outerQubit); it != newQubits.end()) {
      auto index = std::distance(newQubits.begin(), it);
      mapping.map(arg, qubitArgs[index]);
    } else {
      llvm::reportFatalInternalError("Outer qubit not found in new qubits");
    }
  }
}

/**
 * @brief Returns the number of operations implementing @p UnitaryInterface in
 * @p block.
 */
template <typename UnitaryInterface>
[[nodiscard]] size_t getNumBodyUnitaries(Block& block) {
  return static_cast<size_t>(llvm::count_if(
      block, [](Operation& op) { return isa<UnitaryInterface>(op); }));
}

/**
 * @brief Returns the @p i-th operation implementing @p UnitaryInterface in
 * @p block, reporting a fatal error if @p i is out of bounds.
 */
template <typename UnitaryInterface>
[[nodiscard]] UnitaryInterface getBodyUnitary(Block& block, size_t i) {
  auto unitaries = llvm::make_filter_range(
      block, [](Operation& op) { return isa<UnitaryInterface>(op); });
  auto it = std::next(unitaries.begin(), static_cast<std::ptrdiff_t>(i));
  if (it == unitaries.end()) {
    llvm::reportFatalUsageError("Unitary index out of bounds");
  }
  return cast<UnitaryInterface>(*it);
}

/**
 * @brief Returns the single operation implementing @p UnitaryInterface in
 * @p block, or a null interface if @p block does not contain exactly one.
 */
template <typename UnitaryInterface>
[[nodiscard]] UnitaryInterface getSoleBodyUnitary(Block& block) {
  auto unitaries = llvm::make_filter_range(
      block, [](Operation& op) { return isa<UnitaryInterface>(op); });
  auto it = unitaries.begin();
  if (it == unitaries.end()) {
    return {};
  }
  auto unitary = cast<UnitaryInterface>(*it);
  if (++it != unitaries.end()) {
    return {};
  }
  return unitary;
}

} // namespace mlir::utils

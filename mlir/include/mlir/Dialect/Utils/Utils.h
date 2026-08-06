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

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numbers>
#include <optional>
#include <variant>

namespace mlir::utils {

/// Attribute used to retain a source-level classical-register name.
inline constexpr llvm::StringLiteral CLASSICAL_REGISTER_NAME_ATTR =
    "mqt.classical_register_name";

/// Attribute used to retain a source-level quantum-register name.
inline constexpr llvm::StringLiteral QUANTUM_REGISTER_NAME_ATTR =
    "mqt.quantum_register_name";

/// Check if a floating-point value is an integer.
[[nodiscard]] inline bool isIntegerExponent(double r) {
  return r == std::floor(r) && std::isfinite(r);
}

/// Check if a floating-point value is an even integer.
/// Uses fmod to avoid UB from narrowing to int64_t for large values.
[[nodiscard]] inline bool isEvenExponent(double r) {
  return std::fmod(std::fabs(r), 2.0) == 0.0;
}

/// Normalize an angle to (-π, π].
[[nodiscard]] inline double normalizeAngle(double theta) {
  const double twoPi = 2.0 * std::numbers::pi;
  theta = std::fmod(theta, twoPi);
  if (theta > std::numbers::pi) {
    theta -= twoPi;
  }
  if (theta <= -std::numbers::pi) {
    theta += twoPi;
  }
  return theta;
}

/// Default absolute tolerance for MLIR dialect numerics (angle wrapping,
/// phase-zero checks).
constexpr auto TOLERANCE = 1e-15;

/// Largest supported magnitude of a global-phase angle in radians.
///
/// Keeping phase angles in this generous practical range makes binary64 angle
/// reduction accurate enough for exact-unitary compiler rewrites.
constexpr double MAX_GLOBAL_PHASE_ANGLE = 1.0e4;

/// Check the compiler-wide global-phase angle contract.
[[nodiscard]] inline bool isValidGlobalPhaseAngle(const double theta) {
  return std::isfinite(theta) && std::abs(theta) <= MAX_GLOBAL_PHASE_ANGLE;
}

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
[[nodiscard]] inline Value
variantToValue(OpBuilder& builder, Location loc,
               const std::variant<T, Value>& parameter) {
  if (const auto* value = std::get_if<Value>(&parameter)) {
    return *value;
  }
  return constantFromScalar(builder, loc, std::get<T>(parameter));
}

inline void validateMemRefIndex(Value memref,
                                const std::variant<int64_t, Value>& index) {
  const auto* constant = std::get_if<int64_t>(&index);
  if (constant == nullptr) {
    return;
  }
  if (*constant < 0) {
    llvm::reportFatalUsageError("Register index must be non-negative");
  }
  const auto type = dyn_cast<MemRefType>(memref.getType());
  if (type && type.getRank() == 1 && !type.isDynamicDim(0) &&
      *constant >= type.getDimSize(0)) {
    llvm::reportFatalUsageError("Register index is out of bounds");
  }
}

[[nodiscard]] inline std::optional<double> attributeToDouble(Attribute attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return floatAttr.getValueAsDouble();
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    const bool isSigned = !intAttr.getType().isUnsignedInteger();
    APFloat apf(APFloat::IEEEdouble(), APInt::getZero(64));
    apf.convertFromAPInt(intAttr.getValue(), isSigned,
                         APFloat::rmNearestTiesToEven);
    return apf.convertToDouble();
  }
  return std::nullopt;
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
  return attributeToDouble(constantOp.getValue());
}

/// Recursively constant-fold a pure SSA expression DAG to an attribute.
///
/// @p cache memoizes both successful and failed evaluations so shared SSA
/// operands are resolved once (linear in the expression DAG).
[[nodiscard]] inline std::optional<Attribute>
valueToConstantAttr(Value value,
                    llvm::DenseMap<Value, std::optional<Attribute>>& cache) {
  if (const auto it = cache.find(value); it != cache.end()) {
    return it->second;
  }

  Attribute attr;
  if (matchPattern(value, m_Constant(&attr))) {
    return cache[value] = attr;
  }

  Operation* op = value.getDefiningOp();
  if (op == nullptr || op->getNumRegions() != 0 || !isPure(op)) {
    return cache[value] = std::nullopt;
  }

  SmallVector<Attribute> operands;
  operands.reserve(op->getNumOperands());
  for (const Value operand : op->getOperands()) {
    const auto folded = valueToConstantAttr(operand, cache);
    if (!folded) {
      return cache[value] = std::nullopt;
    }
    operands.push_back(*folded);
  }

  SmallVector<OpFoldResult, 1> results;
  if (failed(op->fold(operands, results)) || results.size() != 1) {
    return cache[value] = std::nullopt;
  }
  std::optional<Attribute> folded;
  if (const auto resultAttr =
          llvm::dyn_cast_if_present<Attribute>(results.front())) {
    folded = resultAttr;
  } else if (const auto resultValue =
                 llvm::dyn_cast_if_present<Value>(results.front())) {
    // Identity-style folds may return an existing SSA value (e.g. `addf x,
    // -0`).
    folded = valueToConstantAttr(resultValue, cache);
  }
  return cache[value] = folded;
}

/// Recursively constant-fold a pure SSA expression DAG to an attribute.
[[nodiscard]] inline std::optional<Attribute> valueToConstantAttr(Value value) {
  llvm::DenseMap<Value, std::optional<Attribute>> cache;
  return valueToConstantAttr(value, cache);
}

/// Recursively constant-fold a pure SSA expression tree to a double.
///
/// Used by global-phase normalization so merged phases stay inside the
/// practical `gphase` angle contract instead of emitting long add-chains that
/// later constant-fold past `MAX_GLOBAL_PHASE_ANGLE`.
[[nodiscard]] inline std::optional<double> valueToConstantDouble(Value value) {
  if (const auto attr = valueToConstantAttr(value)) {
    return attributeToDouble(*attr);
  }
  return std::nullopt;
}

/**
 * @brief Parse a list of aliased qubits.
 *
 * @details
 * The modifier operations use aliased qubits inside of their region. This
 * function resolves the relationship between the block arguments and the qubit
 * operands. In the example below, the block argument `%a0` aliases the operand
 * `%q1`.
 *
 * ```mlir
 * qc.ctrl(%q0) targets(%a0 = %q1) {
 *   qc.x %a0 : !qc.qubit
 * } : !qc.qubit
 * ```
 */
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

      // Hard-code QubitType because the modifiers only alias qubits
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

/**
 * @brief Print a list of aliased qubits.
 *
 * @details
 * The modifier operations use aliased qubits inside of their region. This
 * function prints a representation of the relationship between the block
 * arguments and the qubit operands. In the example below, the block argument
 * `%a0` aliases the operand `%q1`.
 *
 * ```mlir
 * qc.ctrl(%q0) targets(%a0 = %q1) {
 *   qc.x %a0 : !qc.qubit
 * } : !qc.qubit
 * ```
 */
inline void printTargetAliasing(OpAsmPrinter& printer, Region& region,
                                OperandRange targetsIn) {
  printer << "(";
  if (region.empty()) {
    printer << ") ";
    printer.printRegion(region, false);
    return;
  }
  auto& entryBlock = region.front();

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
    assert(blockArg.getArgNumber() < qubits.size() &&
           "block argument index must be within qubits range");
    return qubits[blockArg.getArgNumber()];
  }
  return qubit;
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

/**
 * @brief Hoists a body's supporting ops out before the modifier is erased.
 *
 * @details Moves every operation in @p body except @p keep and the block
 * terminator to just before @p target. This keeps Values that feed @p keep
 * (e.g., an exponent produced by constants/arithmetic) available after the
 * modifier's region is erased, avoiding dangling operands.
 *
 * Unlike @c inlineBlockBefore, this is selective (@p keep and the terminator
 * stay in @p body) and does not remap block arguments; the moved ops are
 * classical and never reference the body's block arguments.
 */
inline void hoistSupportingOpsBefore(Block& body, Operation* keep,
                                     Operation* target,
                                     RewriterBase& rewriter) {
  for (auto& bodyOp : llvm::make_early_inc_range(body)) {
    if (&bodyOp != keep && !bodyOp.hasTrait<OpTrait::IsTerminator>()) {
      rewriter.moveOpBefore(&bodyOp, target);
    }
  }
}

/**
 * @brief Inlines a modifier body and replaces the modifier with its results.
 *
 * @details Inlines the operations of @p body in front of @p op, substituting
 * the block arguments of @p body with @p blockArgReplacements, and replaces
 * @p op with the values yielded by the body's terminator.
 */
inline void inlineModifierBody(Operation* op, Block& body,
                               ValueRange blockArgReplacements,
                               RewriterBase& rewriter) {
  auto* terminator = body.getTerminator();
  const SmallVector<Value> results(terminator->getOperands());
  rewriter.inlineBlockBefore(&body, op, blockArgReplacements);
  rewriter.eraseOp(terminator);
  rewriter.replaceOp(op, results);
}

/**
 * @brief Inline @p source into the block currently being built and return the
 * values its terminator yielded.
 *
 * @details Intended for use inside a modifier's body-builder callback, where
 * the current insertion block is the freshly created (still terminator-less)
 * body of the op under construction. Inlines @p source at the start of that
 * block, substituting @p source's block arguments with @p blockArgReplacements,
 * then drops @p source's terminator and returns the values it yielded so the
 * caller can re-yield them.
 *
 * Unlike @c inlineModifierBody, this does not replace an existing op: it
 * splices a body into the block being constructed and hands back the yielded
 * values. In dialects whose bodies yield nothing, the returned vector is empty.
 */
inline SmallVector<Value>
inlineBodyReturningYields(Block& source, ValueRange blockArgReplacements,
                          RewriterBase& rewriter) {
  auto* dest = rewriter.getInsertionBlock();
  rewriter.inlineBlockBefore(&source, dest, dest->begin(),
                             blockArgReplacements);
  auto yielded = llvm::to_vector(dest->back().getOperands());
  rewriter.eraseOp(&dest->back());
  return yielded;
}

/**
 * @brief Find the entry point function with the entry_point attribute
 *
 * @details
 * Searches for the function marked with the "entry_point" attribute in
 * the passthrough attributes. If multiple functions are marked, returns the
 * first one encountered.
 *
 * @param op The module operation to search in.
 * @returns the entry point function, or nullptr if not found.
 */
inline func::FuncOp getEntryPoint(ModuleOp op) {
  static constexpr StringRef PASSTHROUGH_LABEL = "passthrough";
  static constexpr StringRef ENTRY_POINT_LABEL = "entry_point";

  const auto isEntry = [](Attribute attr) {
    const auto strAttr = dyn_cast<StringAttr>(attr);
    return strAttr && strAttr.getValue() == ENTRY_POINT_LABEL;
  };

  for (auto func : op.getOps<func::FuncOp>()) {
    if (const auto passthrough =
            func->getAttrOfType<ArrayAttr>(PASSTHROUGH_LABEL);
        passthrough && llvm::any_of(passthrough, isEntry)) {
      return func;
    }
  }

  return nullptr;
}

} // namespace mlir::utils

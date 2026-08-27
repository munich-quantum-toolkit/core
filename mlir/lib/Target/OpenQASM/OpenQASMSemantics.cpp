/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Target/OpenQASM/Detail/OpenQASMSemantics.h"

#include "mlir/Target/OpenQASM/Detail/OpenQASMParser.h"
#include "mlir/Target/OpenQASM/Detail/OpenQASMSyntax.h"
#include "mlir/Target/OpenQASM/Frontend.h"
#include "mlir/Target/OpenQASM/GateCatalog.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Analysis/Presburger/IntegerRelation.h>
#include <mlir/Analysis/Presburger/PresburgerSpace.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mlir::oq3::frontend::detail {
namespace {

constexpr uint64_t REGISTER_WIDTH_LIMIT = 100'000;
constexpr uint64_t TOTAL_REGISTER_ELEMENT_LIMIT = 100'000;
constexpr size_t EXPRESSION_DEPTH_LIMIT = 256;
constexpr size_t GATE_DEPENDENCY_DEPTH_LIMIT = 64;
constexpr size_t TYPED_STATEMENT_LIMIT = 1'000'000;
constexpr size_t AFFINE_DISTINCTNESS_COMPARISON_LIMIT = 1'024;
constexpr uint32_t DEFAULT_ANGLE_WIDTH = 52;
constexpr uint32_t MAX_ANGLE_WIDTH = 52;
constexpr double TWO_PI = 2.0 * std::numbers::pi;
constexpr uint64_t TWO_PI_BITS = 0x401921FB54442D18ULL;
constexpr uint64_t TWO_PI_ODD_SIGNIFICAND = 0x3243F6A8885A3ULL;
constexpr uint64_t DOUBLE_FRACTION_MASK = (uint64_t{1} << 52U) - 1U;
constexpr uint64_t DOUBLE_EXPONENT_MASK = 0x7FFU;

static_assert(std::bit_cast<uint64_t>(TWO_PI) == TWO_PI_BITS);

struct FixedAngle {
  uint64_t bits = 0;
  uint32_t bitWidth = DEFAULT_ANGLE_WIDTH;
};

struct Constant {
  ScalarType type = ScalarType::Int;
  std::variant<bool, int64_t, uint64_t, double, FixedAngle> value = int64_t{0};
};

struct GateSignature {
  size_t parameterCount = 0;
  size_t qubitCount = 0;
  bool variadicControls = false;
};

enum class SymbolKind : uint8_t {
  Scalar,
  GateLocalScalar,
  Constant,
  Register,
  GateParameter,
  GateQubit,
};

struct Symbol {
  SymbolKind kind = SymbolKind::Scalar;
  ScalarType type = ScalarType::Int;
  uint32_t id = 0;
  std::optional<Constant> constant;
};

} // namespace

[[nodiscard]] static uint64_t angleMask(const uint32_t bitWidth) {
  assert(bitWidth >= 1 && bitWidth <= MAX_ANGLE_WIDTH);
  return (uint64_t{1} << bitWidth) - 1U;
}

[[nodiscard]] static llvm::APInt
roundUnsignedQuotient(const llvm::APInt& numerator,
                      const llvm::APInt& denominator) {
  const auto quotient = numerator.udiv(denominator);
  const auto remainder = numerator.urem(denominator);
  const auto twiceRemainder = remainder.shl(1U);
  const auto roundUp = twiceRemainder.ugt(denominator) ||
                       (twiceRemainder == denominator && quotient[0]);
  return roundUp ? quotient + 1U : quotient;
}

[[nodiscard]] static uint64_t
quantizeAngleMagnitude(const uint64_t significand, const int32_t binaryExponent,
                       const uint32_t bitWidth) {
  constexpr unsigned workingWidth = 128;
  const llvm::APInt modulus(workingWidth, TWO_PI_ODD_SIGNIFICAND);
  const llvm::APInt magnitude(workingWidth, significand);
  llvm::APInt rounded(workingWidth, 0);

  if (binaryExponent >= 0) {
    auto power = llvm::APInt(workingWidth, 2);
    auto factor = llvm::APInt(workingWidth, 1);
    auto exponent = static_cast<uint32_t>(binaryExponent);
    while (exponent != 0) {
      if ((exponent & 1U) != 0) {
        factor = (factor * power).urem(modulus);
      }
      power = (power * power).urem(modulus);
      exponent >>= 1U;
    }
    const auto remainder = (magnitude.urem(modulus) * factor).urem(modulus);
    rounded = roundUnsignedQuotient(remainder.shl(bitWidth), modulus);
  } else {
    const auto scaledExponent = binaryExponent + static_cast<int32_t>(bitWidth);
    if (scaledExponent >= 0) {
      rounded = roundUnsignedQuotient(
          magnitude.shl(static_cast<unsigned>(scaledExponent)), modulus);
    } else {
      const auto denominatorShift = static_cast<unsigned>(-scaledExponent);
      /// The numerator has at most 53 bits and the odd denominator has 50.
      /// Five more denominator bits put even the largest numerator below half.
      if (denominatorShift >= 5U) {
        return 0;
      }
      rounded = roundUnsignedQuotient(magnitude, modulus.shl(denominatorShift));
    }
  }

  return rounded.trunc(bitWidth).getZExtValue();
}

[[nodiscard]] static std::optional<uint64_t>
quantizeAngle(const double radians, const uint32_t bitWidth) {
  if (bitWidth < 1 || bitWidth > MAX_ANGLE_WIDTH || !std::isfinite(radians)) {
    return std::nullopt;
  }

  const auto representation = std::bit_cast<uint64_t>(radians);
  const auto exponent =
      static_cast<uint32_t>((representation >> 52U) & DOUBLE_EXPONENT_MASK);
  const auto fraction = representation & DOUBLE_FRACTION_MASK;
  if (exponent == 0 && fraction == 0) {
    return 0;
  }

  const auto significand =
      exponent == 0 ? fraction : fraction | (uint64_t{1} << 52U);
  /// The binary64 value of 2*pi is TWO_PI_ODD_SIGNIFICAND * 2^-47.
  const auto binaryExponent =
      exponent == 0 ? -1027 : static_cast<int32_t>(exponent) - 1028;
  auto result = quantizeAngleMagnitude(significand, binaryExponent, bitWidth);
  if ((representation >> 63U) != 0) {
    result = (uint64_t{0} - result) & angleMask(bitWidth);
  }
  return result;
}

[[nodiscard]] static double angleToRadians(const FixedAngle angle) {
  assert(angle.bitWidth >= 1 && angle.bitWidth <= MAX_ANGLE_WIDTH);
  return static_cast<double>(angle.bits) *
         std::ldexp(TWO_PI, -static_cast<int>(angle.bitWidth));
}

[[nodiscard]] static FixedAngle resizeAngle(const FixedAngle angle,
                                            const uint32_t targetWidth) {
  assert(targetWidth >= 1 && targetWidth <= MAX_ANGLE_WIDTH);
  if (angle.bitWidth == targetWidth) {
    return angle;
  }
  if (angle.bitWidth < targetWidth) {
    return {.bits = angle.bits << (targetWidth - angle.bitWidth),
            .bitWidth = targetWidth};
  }

  const auto discardedWidth = angle.bitWidth - targetWidth;
  auto retained = angle.bits >> discardedWidth;
  const auto discarded = angle.bits & ((uint64_t{1} << discardedWidth) - 1U);
  const auto halfway = uint64_t{1} << (discardedWidth - 1U);
  if (discarded > halfway || (discarded == halfway && (retained & 1U) != 0)) {
    ++retained;
  }
  return {.bits = retained & angleMask(targetWidth), .bitWidth = targetWidth};
}

[[nodiscard]] static ScalarType scalarType(const ScalarKind kind) {
  switch (kind) {
  case ScalarKind::Bool:
    return ScalarType::Bool;
  case ScalarKind::Int:
    return ScalarType::Int;
  case ScalarKind::Uint:
    return ScalarType::Uint;
  case ScalarKind::Float:
    return ScalarType::Float;
  case ScalarKind::Angle:
    return ScalarType::Angle;
  }
  llvm_unreachable("unknown syntax scalar kind");
}

[[nodiscard]] static bool isInteger(const ScalarType type) {
  return type == ScalarType::Int || type == ScalarType::Uint;
}

[[nodiscard]] static StringRef scalarTypeName(const ScalarType type) {
  switch (type) {
  case ScalarType::Bool:
    return "bool";
  case ScalarType::Int:
    return "int";
  case ScalarType::Uint:
    return "uint";
  case ScalarType::Float:
    return "float";
  case ScalarType::Angle:
    return "angle";
  }
  llvm_unreachable("unknown scalar type");
}

[[nodiscard]] static bool
belongsToStdGates(const GateAvailability availability) {
  return availability == GateAvailability::StandardLibrary ||
         availability == GateAvailability::StandardLibraryAndQELib1;
}

[[nodiscard]] static bool belongsToQELib1(const GateAvailability availability) {
  return availability == GateAvailability::QELib1 ||
         availability == GateAvailability::StandardLibraryAndQELib1;
}

[[nodiscard]] static double asDouble(const Constant& constant) {
  return std::visit(
      [](const auto value) {
        using T = std::remove_cvref_t<decltype(value)>;
        if constexpr (std::is_same_v<T, FixedAngle>) {
          return angleToRadians(value);
        } else {
          return static_cast<double>(value);
        }
      },
      constant.value);
}

[[nodiscard]] static std::optional<int64_t> asSigned(const Constant& constant) {
  if (constant.type == ScalarType::Uint) {
    const auto value = std::get<uint64_t>(constant.value);
    if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return std::nullopt;
    }
    return static_cast<int64_t>(value);
  }
  return std::get<int64_t>(constant.value);
}

[[nodiscard]] static bool canImplicitlyPromote(const Constant& initializer,
                                               const ScalarType destination) {
  if (initializer.type == destination) {
    return true;
  }
  switch (initializer.type) {
  case ScalarType::Bool:
    return destination == ScalarType::Int || destination == ScalarType::Uint ||
           destination == ScalarType::Float;
  case ScalarType::Int:
    return destination == ScalarType::Float ||
           (destination == ScalarType::Uint &&
            std::get<int64_t>(initializer.value) >= 0);
  case ScalarType::Uint:
    return destination == ScalarType::Float ||
           (destination == ScalarType::Int &&
            std::get<uint64_t>(initializer.value) <=
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  case ScalarType::Float:
    return destination == ScalarType::Angle;
  case ScalarType::Angle:
    return false;
  }
  llvm_unreachable("unknown scalar type");
}

[[nodiscard]] static int compareNumericConstants(const Constant& lhs,
                                                 const Constant& rhs) {
  if (lhs.type == ScalarType::Float || lhs.type == ScalarType::Angle ||
      rhs.type == ScalarType::Float || rhs.type == ScalarType::Angle) {
    const auto left = asDouble(lhs);
    const auto right = asDouble(rhs);
    if (left < right) {
      return -1;
    }
    return left > right ? 1 : 0;
  }
  if (lhs.type == ScalarType::Uint || rhs.type == ScalarType::Uint) {
    const auto asUnsigned = [](const Constant& constant) {
      if (constant.type == ScalarType::Uint) {
        return std::get<uint64_t>(constant.value);
      }
      return static_cast<uint64_t>(std::get<int64_t>(constant.value));
    };
    const auto left = asUnsigned(lhs);
    const auto right = asUnsigned(rhs);
    if (left < right) {
      return -1;
    }
    return left > right ? 1 : 0;
  }
  const auto left = std::get<int64_t>(lhs.value);
  const auto right = std::get<int64_t>(rhs.value);
  if (left < right) {
    return -1;
  }
  return left > right ? 1 : 0;
}

namespace {

#define MQT_OQ3_TRY_ASSIGN(NAME, EXPRESSION)                                   \
  auto NAME##Result = (EXPRESSION);                                            \
  if (::mlir::failed(NAME##Result)) {                                          \
    return ::mlir::failure();                                                  \
  }                                                                            \
  /* NOLINTNEXTLINE(bugprone-macro-parentheses) */                             \
  auto NAME = std::move(*NAME##Result)

class SemanticAnalyzer {
public:
  SemanticAnalyzer(const SyntaxProgram& syntaxProgram,
                   const llvm::SourceMgr& sourceManager,
                   const FrontendOptions& frontendOptions)
      : syntax(syntaxProgram), sources(sourceManager), options(frontendOptions),
        constantExpressionStatus(syntax.expressions.size(), 0),
        constantValues(syntax.expressions.size()),
        constantTypes(syntax.expressions.size()) {
    scopes.emplace_back();
  }

  [[nodiscard]] AnalysisResult run() {
    if (failed(analyzeVersion()) || failed(validateExpressionDepth()) ||
        failed(analyzeTopLevelBody()) || failed(validateGateCallGraph()) ||
        failed(finalizeOutputs())) {
      assert(failureDiagnostic.has_value());
      return {.diagnostics = {std::move(*failureDiagnostic)}};
    }
    return {.program = std::make_unique<TypedProgram>(std::move(program))};
  }

private:
  struct AffineForm {
    SmallVector<llvm::DynamicAPInt, 4> coefficients;
    llvm::DynamicAPInt constant{0};
  };

  struct DynamicBitFact {
    ExpressionId expression = 0;
    std::vector<std::pair<uint64_t, uint64_t>> dependencies;
  };
  using BitInitialization = std::vector<bool>;
  using DynamicBitFactSet = std::vector<DynamicBitFact>;

  // The analyzed syntax and source manager are mandatory and outlive this run.
  const SyntaxProgram& syntax;
  const llvm::SourceMgr& sources;
  FrontendOptions options;
  TypedProgram program;
  SmallVector<llvm::StringMap<Symbol>> scopes;
  llvm::StringMap<GateSignature> customGates;
  std::vector<std::shared_ptr<BitInitialization>> initializedBits;
  std::vector<std::shared_ptr<DynamicBitFactSet>> dynamicBitFacts;
  std::vector<bool> initializedScalars;
  std::vector<uint64_t> scalarGenerations;
  std::vector<std::optional<ExpressionId>> affineScalarValues;
  llvm::DenseMap<ScalarId, unsigned> activeInductions;
  llvm::DenseSet<ScalarId> loopVariantScalars;
  presburger::IntegerPolyhedron affineDomain{
      presburger::PresburgerSpace::getSetSpace()};
  std::vector<uint64_t> bitGenerations;
  std::vector<ProgramOutput> implicitOutputs;
  std::vector<ProgramOutput> explicitOutputs;
  mutable std::vector<int8_t> constantExpressionStatus;
  mutable std::vector<std::optional<Constant>> constantValues;
  mutable std::vector<std::optional<ScalarType>> constantTypes;
  bool insideGate = false;
  std::set<uint64_t> hardwareQubits;
  uint64_t totalRegisterElements = 0;
  std::optional<SyntaxIncludeContextId> currentIncludeContext;
  mutable std::optional<Diagnostic> failureDiagnostic;

  [[nodiscard]] SourceLocation getSourceLocation(const SMLoc location) const {
    auto result = sourceLocation(sources, location);
    if (!currentIncludeContext) {
      return result;
    }
    result.includeStack.clear();
    auto context = currentIncludeContext;
    while (context) {
      const auto& include = syntax.includeContexts.at(*context);
      const auto includeLocation = sourceLocation(sources, include.location);
      result.includeStack.push_back({.filename = includeLocation.filename,
                                     .line = includeLocation.line,
                                     .column = includeLocation.column});
      context = include.parent;
    }
    return result;
  }

  [[nodiscard]] LogicalResult fail(SourceLocation location,
                                   const Twine& message) const {
    assert(!failureDiagnostic.has_value());
    failureDiagnostic =
        Diagnostic{.location = std::move(location), .message = message.str()};
    return failure();
  }

  [[nodiscard]] LogicalResult fail(const SMLoc location,
                                   const Twine& message) const {
    return fail(getSourceLocation(location), message);
  }

  [[nodiscard]] static llvm::DynamicAPInt
  unsignedCoefficient(const uint64_t value) {
    return llvm::DynamicAPInt(llvm::APInt(65, value));
  }

  [[nodiscard]] static llvm::DynamicAPInt signedMinimum() {
    return llvm::DynamicAPInt(llvm::APInt::getSignedMinValue(64));
  }

  [[nodiscard]] static llvm::DynamicAPInt signedMaximum() {
    return llvm::DynamicAPInt(llvm::APInt::getSignedMaxValue(64));
  }

  [[nodiscard]] AffineForm
  constantAffineForm(const llvm::DynamicAPInt& value) const {
    return {.coefficients = SmallVector<llvm::DynamicAPInt, 4>(
                affineDomain.getNumDimVars(), llvm::DynamicAPInt(0)),
            .constant = value};
  }

  [[nodiscard]] static bool isAffineConstant(const AffineForm& form) {
    return llvm::all_of(form.coefficients,
                        [](const auto& value) { return value == 0; });
  }

  [[nodiscard]] static AffineForm addAffineForms(const AffineForm& lhs,
                                                 const AffineForm& rhs) {
    assert(lhs.coefficients.size() == rhs.coefficients.size());
    auto result = lhs;
    for (const auto [index, value] : llvm::enumerate(rhs.coefficients)) {
      result.coefficients[index] += value;
    }
    result.constant += rhs.constant;
    return result;
  }

  [[nodiscard]] static AffineForm negateAffineForm(const AffineForm& form) {
    auto result = form;
    for (auto& coefficient : result.coefficients) {
      coefficient = -coefficient;
    }
    result.constant = -result.constant;
    return result;
  }

  [[nodiscard]] static AffineForm
  scaleAffineForm(const AffineForm& form, const llvm::DynamicAPInt& scale) {
    auto result = form;
    for (auto& coefficient : result.coefficients) {
      coefficient *= scale;
    }
    result.constant *= scale;
    return result;
  }

  [[nodiscard]] static SmallVector<llvm::DynamicAPInt, 4>
  affineConstraint(const AffineForm& form) {
    SmallVector<llvm::DynamicAPInt, 4> result(form.coefficients);
    result.push_back(form.constant);
    return result;
  }

  [[nodiscard]] bool
  domainIsEmptyWithInequality(const AffineForm& inequality) const {
    presburger::IntegerPolyhedron candidate(affineDomain);
    candidate.addInequality(affineConstraint(inequality));
    return candidate.isIntegerEmpty();
  }

  [[nodiscard]] bool
  domainIsEmptyWithEquality(const AffineForm& equality) const {
    presburger::IntegerPolyhedron candidate(affineDomain);
    candidate.addEquality(affineConstraint(equality));
    return candidate.isIntegerEmpty();
  }

  [[nodiscard]] bool provesLowerBound(const AffineForm& form,
                                      const llvm::DynamicAPInt& lower) const {
    auto counterexample = negateAffineForm(form);
    counterexample.constant += lower - 1;
    return domainIsEmptyWithInequality(counterexample);
  }

  [[nodiscard]] bool provesUpperBound(const AffineForm& form,
                                      const llvm::DynamicAPInt& upper) const {
    auto counterexample = form;
    counterexample.constant -= upper + 1;
    return domainIsEmptyWithInequality(counterexample);
  }

  [[nodiscard]] bool affineFormFitsType(const AffineForm& form,
                                        const ScalarType type) const {
    if (type != ScalarType::Int && type != ScalarType::Uint) {
      return false;
    }
    return provesLowerBound(form, type == ScalarType::Int
                                      ? signedMinimum()
                                      : llvm::DynamicAPInt(0)) &&
           provesUpperBound(form, signedMaximum());
  }

  [[nodiscard]] std::optional<AffineForm>
  buildAffineForm(const ExpressionId expression) const {
    const auto& value = program.expressions.at(expression);
    std::optional<AffineForm> result;
    switch (value.kind) {
    case ExpressionKind::Constant:
      if (value.type == ScalarType::Int) {
        result = constantAffineForm(
            llvm::DynamicAPInt(std::get<int64_t>(value.constant)));
      } else if (value.type == ScalarType::Uint) {
        result = constantAffineForm(
            unsignedCoefficient(std::get<uint64_t>(value.constant)));
      }
      break;
    case ExpressionKind::Variable: {
      if (loopVariantScalars.contains(value.variable)) {
        break;
      }
      const auto induction = activeInductions.find(value.variable);
      if (induction != activeInductions.end()) {
        result = constantAffineForm(llvm::DynamicAPInt(0));
        result->coefficients[induction->second] = llvm::DynamicAPInt(1);
        break;
      }
      if (value.variable < affineScalarValues.size() &&
          affineScalarValues[value.variable]) {
        result = buildAffineForm(*affineScalarValues[value.variable]);
      }
      break;
    }
    case ExpressionKind::Cast:
      if (isInteger(value.type) &&
          isInteger(program.expressions.at(value.lhs).type)) {
        result = buildAffineForm(value.lhs);
      }
      break;
    case ExpressionKind::Negate:
      if (auto operand = buildAffineForm(value.lhs)) {
        result = negateAffineForm(*operand);
      }
      break;
    case ExpressionKind::Add:
    case ExpressionKind::Subtract: {
      auto lhs = buildAffineForm(value.lhs);
      auto rhs = buildAffineForm(value.rhs);
      if (lhs && rhs) {
        result = addAffineForms(*lhs, value.kind == ExpressionKind::Add
                                          ? *rhs
                                          : negateAffineForm(*rhs));
      }
      break;
    }
    case ExpressionKind::Multiply: {
      auto lhs = buildAffineForm(value.lhs);
      auto rhs = buildAffineForm(value.rhs);
      if (!lhs || !rhs) {
        break;
      }
      if (isAffineConstant(*lhs)) {
        result = scaleAffineForm(*rhs, lhs->constant);
      } else if (isAffineConstant(*rhs)) {
        result = scaleAffineForm(*lhs, rhs->constant);
      }
      break;
    }
    default:
      break;
    }
    if (!result || !affineFormFitsType(*result, value.type)) {
      return std::nullopt;
    }
    return result;
  }

  [[nodiscard]] std::optional<ExpressionId>
  expandAffineScalarValues(const ExpressionId expression) {
    const auto value = program.expressions.at(expression);
    switch (value.kind) {
    case ExpressionKind::Constant:
      return expression;
    case ExpressionKind::Variable:
      if (activeInductions.contains(value.variable)) {
        return expression;
      }
      if (value.variable < affineScalarValues.size()) {
        return affineScalarValues[value.variable];
      }
      return std::nullopt;
    case ExpressionKind::Cast:
    case ExpressionKind::Negate: {
      auto operand = expandAffineScalarValues(value.lhs);
      if (!operand) {
        return std::nullopt;
      }
      if (*operand == value.lhs) {
        return expression;
      }
      auto expanded = value;
      expanded.lhs = *operand;
      return addExpression(expanded);
    }
    case ExpressionKind::Add:
    case ExpressionKind::Subtract:
    case ExpressionKind::Multiply: {
      auto lhs = expandAffineScalarValues(value.lhs);
      auto rhs = expandAffineScalarValues(value.rhs);
      if (!lhs || !rhs) {
        return std::nullopt;
      }
      if (*lhs == value.lhs && *rhs == value.rhs) {
        return expression;
      }
      auto expanded = value;
      expanded.lhs = *lhs;
      expanded.rhs = *rhs;
      return addExpression(expanded);
    }
    default:
      return std::nullopt;
    }
  }

  [[nodiscard]] bool sameAffineValue(const ExpressionId lhs,
                                     const ExpressionId rhs) const {
    const auto left = buildAffineForm(lhs);
    const auto right = buildAffineForm(rhs);
    return left && right && left->coefficients == right->coefficients &&
           left->constant == right->constant;
  }

  void
  collectLoopMutations(const ArrayRef<SyntaxStatementId> statements,
                       llvm::DenseSet<ScalarId>& mutations,
                       llvm::DenseSet<StringRef>& blockLocalScalars) const {
    const auto collectNested = [&](const ArrayRef<SyntaxStatementId> body,
                                   llvm::DenseSet<StringRef> locals) {
      collectLoopMutations(body, mutations, locals);
    };
    for (const auto statement : statements) {
      const auto& data = syntax.statements[statement].data;
      if (const auto* assignment = std::get_if<SyntaxAssignment>(&data)) {
        if (!assignment->target.index &&
            !blockLocalScalars.contains(assignment->target.identifier)) {
          const auto* symbol = lookup(assignment->target.identifier);
          if (symbol != nullptr && symbol->kind == SymbolKind::Scalar) {
            mutations.insert(symbol->id);
          }
        }
        continue;
      }
      if (const auto* declaration =
              std::get_if<SyntaxScalarDeclaration>(&data)) {
        if (!declaration->isConst) {
          blockLocalScalars.insert(declaration->identifier);
        }
        continue;
      }
      if (const auto* conditional = std::get_if<SyntaxIf>(&data)) {
        collectNested(conditional->thenStatements, blockLocalScalars);
        collectNested(conditional->elseStatements, blockLocalScalars);
        continue;
      }
      if (const auto* loop = std::get_if<SyntaxFor>(&data)) {
        auto locals = blockLocalScalars;
        locals.insert(loop->inductionVariable);
        collectNested(loop->body, std::move(locals));
        continue;
      }
      if (const auto* loop = std::get_if<SyntaxWhile>(&data)) {
        collectNested(loop->body, blockLocalScalars);
        continue;
      }
      if (const auto* switchStatement = std::get_if<SyntaxSwitch>(&data)) {
        for (const auto& switchCase : switchStatement->cases) {
          collectNested(switchCase.body, blockLocalScalars);
        }
        collectNested(switchStatement->defaultStatements, blockLocalScalars);
      }
    }
  }

  [[nodiscard]] bool proveDistinct(const QubitReference& lhs,
                                   const QubitReference& rhs,
                                   size_t& affineComparisons) const {
    if (lhs.kind != rhs.kind) {
      return true;
    }
    if (lhs.kind == QubitReferenceKind::GateArgument) {
      return lhs.symbol != rhs.symbol;
    }
    if (lhs.kind == QubitReferenceKind::Hardware) {
      return lhs.index != rhs.index;
    }
    if (lhs.symbol != rhs.symbol) {
      return true;
    }
    if (!lhs.provenIndex && !rhs.provenIndex) {
      return lhs.index != rhs.index;
    }
    const auto indexForm =
        [&](const QubitReference& reference) -> std::optional<AffineForm> {
      if (reference.provenIndex) {
        return buildAffineForm(*reference.provenIndex);
      }
      return constantAffineForm(unsignedCoefficient(reference.index));
    };
    auto left = indexForm(lhs);
    auto right = indexForm(rhs);
    if (!left || !right) {
      return false;
    }
    const auto difference = addAffineForms(*left, negateAffineForm(*right));
    if (isAffineConstant(difference)) {
      return difference.constant != 0;
    }
    if (affineComparisons >= AFFINE_DISTINCTNESS_COMPARISON_LIMIT) {
      return false;
    }
    ++affineComparisons;
    return domainIsEmptyWithEquality(difference);
  }

  [[nodiscard]] LogicalResult validateExpressionDepth() const {
    std::vector<size_t> depths(syntax.expressions.size(), 1);
    for (const auto [id, expression] : llvm::enumerate(syntax.expressions)) {
      auto depth = size_t{1};
      if (expression.lhs) {
        depth = std::max(depth, depths[*expression.lhs] + 1);
      }
      if (expression.rhs) {
        depth = std::max(depth, depths[*expression.rhs] + 1);
      }
      if (depth > EXPRESSION_DEPTH_LIMIT) {
        return fail(expression.location,
                    Twine("expression depth exceeds the limit of ") +
                        Twine(static_cast<unsigned>(EXPRESSION_DEPTH_LIMIT)));
      }
      depths[id] = depth;
    }
    return success();
  }

  void restoreStatePrefix(
      const std::vector<std::shared_ptr<BitInitialization>>& bitsInitialized,
      const std::vector<bool>& scalarsInitialized,
      const std::vector<uint64_t>& generations,
      const std::vector<uint64_t>& registerGenerations) {
    for (size_t reg = 0; reg < bitsInitialized.size(); ++reg) {
      initializedBits[reg] = bitsInitialized[reg];
    }
    for (size_t scalar = 0; scalar < scalarsInitialized.size(); ++scalar) {
      initializedScalars[scalar] = scalarsInitialized[scalar];
      scalarGenerations[scalar] = generations[scalar];
    }
    for (size_t reg = 0; reg < registerGenerations.size(); ++reg) {
      bitGenerations[reg] = registerGenerations[reg];
    }
  }

  void restoreDynamicFactsPrefix(
      const std::vector<std::shared_ptr<DynamicBitFactSet>>& facts) {
    for (size_t reg = 0; reg < facts.size(); ++reg) {
      dynamicBitFacts[reg] = facts[reg];
    }
    for (size_t reg = facts.size(); reg < dynamicBitFacts.size(); ++reg) {
      dynamicBitFacts[reg] = std::make_shared<DynamicBitFactSet>();
    }
  }

  void restoreAffineScalarValuesPrefix(
      const std::vector<std::optional<ExpressionId>>& values) {
    const auto size = affineScalarValues.size();
    affineScalarValues = values;
    affineScalarValues.resize(size);
  }

  [[nodiscard]] BitInitialization&
  mutableBitInitialization(const RegisterId reg) {
    if (initializedBits[reg].use_count() != 1) {
      initializedBits[reg] =
          std::make_shared<BitInitialization>(*initializedBits[reg]);
    }
    return *initializedBits[reg];
  }

  [[nodiscard]] DynamicBitFactSet&
  mutableDynamicBitFacts(const RegisterId reg) {
    if (dynamicBitFacts[reg].use_count() != 1) {
      dynamicBitFacts[reg] =
          std::make_shared<DynamicBitFactSet>(*dynamicBitFacts[reg]);
    }
    return *dynamicBitFacts[reg];
  }

  [[nodiscard]] bool sameExpression(const ExpressionId lhs,
                                    const ExpressionId rhs) const {
    const auto& left = program.expressions[lhs];
    const auto& right = program.expressions[rhs];
    if (left.kind != right.kind || left.type != right.type ||
        left.constant != right.constant || left.parameter != right.parameter ||
        left.variable != right.variable) {
      return false;
    }
    switch (left.kind) {
    case ExpressionKind::Constant:
    case ExpressionKind::GateParameter:
    case ExpressionKind::Variable:
      return true;
    case ExpressionKind::PopCount:
      return sameBitVectorExpression(left.bitVector, right.bitVector);
    case ExpressionKind::Cast:
    case ExpressionKind::Negate:
    case ExpressionKind::ArcCos:
    case ExpressionKind::ArcSin:
    case ExpressionKind::ArcTan:
    case ExpressionKind::Ceiling:
    case ExpressionKind::Sin:
    case ExpressionKind::Cos:
    case ExpressionKind::Floor:
    case ExpressionKind::Tan:
    case ExpressionKind::Exp:
    case ExpressionKind::Log:
    case ExpressionKind::Sqrt:
      return sameExpression(left.lhs, right.lhs);
    default:
      return sameExpression(left.lhs, right.lhs) &&
             sameExpression(left.rhs, right.rhs);
    }
  }

  [[nodiscard]] bool
  sameBitVectorExpression(const BitVectorExpressionId lhs,
                          const BitVectorExpressionId rhs) const {
    const auto& left = program.bitVectorExpressions[lhs];
    const auto& right = program.bitVectorExpressions[rhs];
    if (left.kind != right.kind || left.width != right.width) {
      return false;
    }
    if (left.kind == BitVectorExpressionKind::Register) {
      return left.reg == right.reg;
    }
    return sameBitVectorExpression(left.operand, right.operand) &&
           sameExpression(left.distance, right.distance);
  }

  void collectBitVectorDependencies(
      const BitVectorExpressionId expression,
      std::vector<std::pair<uint64_t, uint64_t>>& dependencies) const {
    const auto& value = program.bitVectorExpressions[expression];
    if (value.kind == BitVectorExpressionKind::Register) {
      dependencies.emplace_back(value.reg, bitGenerations[value.reg]);
      return;
    }
    collectBitVectorDependencies(value.operand, dependencies);
    collectDependencies(value.distance, dependencies);
  }

  void collectDependencies(
      const ExpressionId expression,
      std::vector<std::pair<uint64_t, uint64_t>>& dependencies) const {
    const auto& value = program.expressions[expression];
    if (value.kind == ExpressionKind::Variable) {
      dependencies.emplace_back((uint64_t{1} << 63U) | value.variable,
                                scalarGenerations[value.variable]);
      return;
    }
    if (value.kind == ExpressionKind::Constant ||
        value.kind == ExpressionKind::GateParameter) {
      return;
    }
    if (value.kind == ExpressionKind::PopCount) {
      collectBitVectorDependencies(value.bitVector, dependencies);
      return;
    }
    collectDependencies(value.lhs, dependencies);
    if (value.kind != ExpressionKind::Cast &&
        value.kind != ExpressionKind::Negate &&
        value.kind != ExpressionKind::ArcCos &&
        value.kind != ExpressionKind::ArcSin &&
        value.kind != ExpressionKind::ArcTan &&
        value.kind != ExpressionKind::Ceiling &&
        value.kind != ExpressionKind::Sin &&
        value.kind != ExpressionKind::Cos &&
        value.kind != ExpressionKind::Floor &&
        value.kind != ExpressionKind::Tan &&
        value.kind != ExpressionKind::Exp &&
        value.kind != ExpressionKind::Log &&
        value.kind != ExpressionKind::Sqrt) {
      collectDependencies(value.rhs, dependencies);
    }
  }

  [[nodiscard]] FailureOr<std::optional<bool>>
  constantCondition(const SyntaxExpressionId expression) const {
    if (!isConstantExpression(expression)) {
      return std::optional<bool>{};
    }
    MQT_OQ3_TRY_ASSIGN(value, evaluateConstant(expression));
    if (value.type != ScalarType::Bool) {
      return std::optional<bool>{};
    }
    return std::optional<bool>(std::get<bool>(value.value));
  }

  [[nodiscard]] const Symbol* lookup(StringRef name) const {
    for (const auto& scope : llvm::reverse(scopes)) {
      if (const auto found = scope.find(name); found != scope.end()) {
        return &found->second;
      }
    }
    return nullptr;
  }

  [[nodiscard]] LogicalResult declare(SMLoc location, StringRef name,
                                      Symbol symbol) {
    const auto* catalog = lookupGate(name);
    const bool catalogNameReserved =
        catalog != nullptr &&
        (catalog->availability == GateAvailability::Language ||
         (belongsToStdGates(catalog->availability) &&
          program.stdGatesIncluded) ||
         (belongsToQELib1(catalog->availability) && program.qelib1Included) ||
         (program.openQASM2 && catalog->name == "CX"));
    if (builtinConstant(name) ||
        (scopes.size() == 1 &&
         (customGates.contains(name) || catalogNameReserved))) {
      return fail(location, "identifier '" + name + "' is already declared");
    }
    if (!scopes.back().insert({name, symbol}).second) {
      return fail(location, "identifier '" + name + "' is already declared");
    }
    return success();
  }

  [[nodiscard]] bool isGateAvailable(const GateCatalogEntry& gate) const {
    if (gate.availability == GateAvailability::Language) {
      return true;
    }
    if (options.gatePolicy == GatePolicy::MQTCompatibility) {
      return true;
    }
    return (belongsToStdGates(gate.availability) && program.stdGatesIncluded) ||
           (belongsToQELib1(gate.availability) && program.qelib1Included) ||
           (program.openQASM2 && gate.name == "CX");
  }

  [[nodiscard]] LogicalResult analyzeVersion() {
    if (!syntax.version) {
      return success();
    }
    const auto version = *syntax.version;
    if (version.major == 2 && version.minor == 0) {
      program.openQASM2 = true;
      return success();
    }
    if (version.major == 3 && (version.minor == 0 || version.minor == 1)) {
      return success();
    }
    return fail(syntax.versionLocation, "Unsupported OpenQASM version " +
                                            std::to_string(version.major) +
                                            "." +
                                            std::to_string(version.minor));
  }

  [[nodiscard]] LogicalResult validateGateCallGraph() const {
    llvm::StringMap<size_t> gateIndices;
    for (const auto [index, gate] : llvm::enumerate(program.gates)) {
      gateIndices[gate.name] = index;
    }
    enum class VisitState : uint8_t { Unvisited, Active, Complete };
    std::vector states(program.gates.size(), VisitState::Unvisited);
    std::vector<size_t> dependencyDepths(program.gates.size());
    const auto visitApplications = [&](auto&& self,
                                       ArrayRef<StatementId> statements,
                                       const auto& callback) -> LogicalResult {
      for (const auto statementId : statements) {
        const auto& statement = program.statements[statementId];
        if (failed(std::visit(
                [&](const auto& data) -> LogicalResult {
                  using T = std::decay_t<decltype(data)>;
                  if constexpr (std::is_same_v<T, GateApplication>) {
                    return callback(data, statement.location);
                  } else if constexpr (std::is_same_v<T, IfStatement>) {
                    if (failed(self(self, data.thenStatements, callback))) {
                      return failure();
                    }
                    return self(self, data.elseStatements, callback);
                  } else if constexpr (std::is_same_v<T, ForStatement> ||
                                       std::is_same_v<T, WhileStatement>) {
                    return self(self, data.body, callback);
                  } else if constexpr (std::is_same_v<T, SwitchStatement>) {
                    for (const auto& switchCase : data.cases) {
                      if (failed(self(self, switchCase.body, callback))) {
                        return failure();
                      }
                    }
                    return self(self, data.defaultStatements, callback);
                  }
                  return success();
                },
                statement.data))) {
          return failure();
        }
      }
      return success();
    };
    const auto visit = [&](auto&& self,
                           const size_t index) -> FailureOr<size_t> {
      if (states[index] == VisitState::Complete) {
        return dependencyDepths[index];
      }
      states[index] = VisitState::Active;
      size_t dependencyDepth = 1;
      if (failed(visitApplications(
              visitApplications, program.gates[index].body,
              [&](const GateApplication& application,
                  const SourceLocation& location) -> LogicalResult {
                const auto callee = gateIndices.find(application.callee);
                if (callee == gateIndices.end()) {
                  return success();
                }
                if (states[callee->second] == VisitState::Active) {
                  return fail(location,
                              "recursive custom gate definition involving '" +
                                  application.callee + "'");
                }
                const auto calleeDepth = self(self, callee->second);
                if (failed(calleeDepth)) {
                  return failure();
                }
                if (*calleeDepth >= GATE_DEPENDENCY_DEPTH_LIMIT) {
                  return fail(
                      location,
                      "custom gate dependency depth exceeds the limit of " +
                          std::to_string(GATE_DEPENDENCY_DEPTH_LIMIT));
                }
                dependencyDepth = std::max(dependencyDepth, *calleeDepth + 1);
                return success();
              }))) {
        return failure();
      }
      states[index] = VisitState::Complete;
      dependencyDepths[index] = dependencyDepth;
      return dependencyDepth;
    };
    for (size_t index = 0; index < program.gates.size(); ++index) {
      if (states[index] == VisitState::Unvisited) {
        if (failed(visit(visit, index))) {
          return failure();
        }
      }
    }
    return success();
  }

  [[nodiscard]] FailureOr<StatementId> addStatement(SMLoc location,
                                                    StatementData data) {
    if (program.statements.size() >= TYPED_STATEMENT_LIMIT) {
      return fail(location,
                  Twine("typed OpenQASM program exceeds the "
                        "statement limit of ") +
                      Twine(static_cast<unsigned>(TYPED_STATEMENT_LIMIT)));
    }
    const auto id = static_cast<StatementId>(program.statements.size());
    program.statements.push_back(
        {.data = std::move(data), .location = getSourceLocation(location)});
    return id;
  }

  [[nodiscard]] ExpressionId addExpression(ScalarExpression expression) {
    const auto id = static_cast<ExpressionId>(program.expressions.size());
    program.expressions.push_back(expression);
    return id;
  }

  [[nodiscard]] BitVectorExpressionId
  addBitVectorExpression(BitVectorExpression expression) {
    const auto id =
        static_cast<BitVectorExpressionId>(program.bitVectorExpressions.size());
    program.bitVectorExpressions.push_back(expression);
    return id;
  }

  [[nodiscard]] ConditionId addCondition(ConditionExpression condition) {
    const auto id = static_cast<ConditionId>(program.conditions.size());
    program.conditions.push_back(std::move(condition));
    return id;
  }

  [[nodiscard]] ExpressionId addConstant(const Constant& constant) {
    if (constant.type == ScalarType::Angle) {
      return addExpression(
          {.kind = ExpressionKind::Constant,
           .type = ScalarType::Angle,
           .constant = angleToRadians(std::get<FixedAngle>(constant.value))});
    }
    return std::visit(
        [&](const auto value) -> ExpressionId {
          using T = std::remove_cvref_t<decltype(value)>;
          if constexpr (std::is_same_v<T, FixedAngle>) {
            llvm_unreachable("fixed angles require angle type");
          } else {
            return addExpression({.kind = ExpressionKind::Constant,
                                  .type = constant.type,
                                  .constant = value});
          }
        },
        constant.value);
  }

  [[nodiscard]] static bool canImplicitlyConvert(const ScalarType source,
                                                 const ScalarType target) {
    if (source == target) {
      return true;
    }
    switch (source) {
    case ScalarType::Bool:
      return target == ScalarType::Int || target == ScalarType::Uint ||
             target == ScalarType::Float;
    case ScalarType::Int:
    case ScalarType::Uint:
      return target == ScalarType::Int || target == ScalarType::Uint ||
             target == ScalarType::Float || target == ScalarType::Angle;
    case ScalarType::Float:
      return target == ScalarType::Int || target == ScalarType::Uint ||
             target == ScalarType::Angle;
    case ScalarType::Angle:
      return target == ScalarType::Float;
    }
    llvm_unreachable("unknown scalar type");
  }

  [[nodiscard]] FailureOr<ExpressionId>
  castExpression(const ExpressionId expression, const ScalarType target,
                 const SMLoc location) {
    const auto source = program.expressions[expression].type;
    if (source == target) {
      return expression;
    }
    if (!canImplicitlyConvert(source, target)) {
      return fail(location, "expression of type '" + scalarTypeName(source) +
                                "' cannot be implicitly converted to '" +
                                scalarTypeName(target) + "'");
    }
    return addExpression(
        {.kind = ExpressionKind::Cast, .type = target, .lhs = expression});
  }

  [[nodiscard]] FailureOr<Constant>
  promoteConstInitializer(const Constant& initializer,
                          const ScalarType destination,
                          const SMLoc location) const {
    if (!canImplicitlyPromote(initializer, destination)) {
      return fail(location, "constant initializer of type '" +
                                scalarTypeName(initializer.type) +
                                "' cannot be implicitly promoted to '" +
                                scalarTypeName(destination) + "'");
    }
    if (initializer.type == destination) {
      return initializer;
    }
    switch (destination) {
    case ScalarType::Bool:
      llvm_unreachable("only bool constants can initialize bool constants");
    case ScalarType::Int:
      if (initializer.type == ScalarType::Bool) {
        return Constant{
            .type = ScalarType::Int,
            .value = static_cast<int64_t>(std::get<bool>(initializer.value))};
      }
      return Constant{
          .type = ScalarType::Int,
          .value = static_cast<int64_t>(std::get<uint64_t>(initializer.value))};
    case ScalarType::Uint:
      if (initializer.type == ScalarType::Bool) {
        return Constant{
            .type = ScalarType::Uint,
            .value = static_cast<uint64_t>(std::get<bool>(initializer.value))};
      }
      return Constant{
          .type = ScalarType::Uint,
          .value = static_cast<uint64_t>(std::get<int64_t>(initializer.value))};
    case ScalarType::Float:
      return Constant{.type = ScalarType::Float,
                      .value = asDouble(initializer)};
    case ScalarType::Angle:
      llvm_unreachable("fixed angle initializers require an explicit width");
    }
    llvm_unreachable("unknown scalar type");
  }

  [[nodiscard]] FailureOr<uint32_t>
  angleWidth(const std::optional<SyntaxExpressionId> size,
             const SMLoc location) const {
    if (!size) {
      return DEFAULT_ANGLE_WIDTH;
    }
    if (!isConstantExpression(*size)) {
      return fail(location,
                  "angle width must be a constant integer expression");
    }
    MQT_OQ3_TRY_ASSIGN(constant, evaluateConstant(*size));
    if (!isInteger(constant.type)) {
      return fail(location, "angle width must be an integer expression");
    }
    const auto width = asSigned(constant);
    if (!width || *width < 1 || std::cmp_greater(*width, MAX_ANGLE_WIDTH)) {
      return fail(location, Twine("angle width must be between 1 and ") +
                                Twine(MAX_ANGLE_WIDTH));
    }
    return static_cast<uint32_t>(*width);
  }

  [[nodiscard]] FailureOr<Constant>
  convertToFixedAngle(const Constant& source, const uint32_t bitWidth,
                      const SMLoc location) const {
    if (source.type == ScalarType::Angle) {
      return Constant{
          .type = ScalarType::Angle,
          .value = resizeAngle(std::get<FixedAngle>(source.value), bitWidth)};
    }
    if (source.type != ScalarType::Float) {
      return fail(location,
                  "angle conversion requires a float or angle operand");
    }
    const auto bits = quantizeAngle(std::get<double>(source.value), bitWidth);
    if (!bits) {
      return fail(location, "angle conversion requires a finite float value");
    }
    return Constant{.type = ScalarType::Angle,
                    .value = FixedAngle{.bits = *bits, .bitWidth = bitWidth}};
  }

  [[nodiscard]] FailureOr<uint64_t>
  angleIntegerLiteral(const SyntaxExpressionId id,
                      const uint32_t bitWidth) const {
    const auto& expression = syntax.expressions[id];
    if (expression.kind != Expr::Kind::Int || !expression.wideInteger.empty() ||
        expression.integer > angleMask(bitWidth)) {
      return fail(expression.location,
                  "angle multiplication and division require a nonnegative "
                  "integer literal that fits the angle width");
    }
    return expression.integer;
  }

  [[nodiscard]] bool expressionProducesBool(const SyntaxExpressionId id) const {
    const auto& expression = syntax.expressions[id];
    switch (expression.kind) {
    case Expr::Kind::Bool:
    case Expr::Kind::Not:
    case Expr::Kind::And:
    case Expr::Kind::Or:
    case Expr::Kind::Equal:
    case Expr::Kind::NotEqual:
    case Expr::Kind::Less:
    case Expr::Kind::LessEqual:
    case Expr::Kind::Greater:
    case Expr::Kind::GreaterEqual:
      return true;
    case Expr::Kind::Identifier: {
      const auto* symbol = lookup(expression.identifier);
      return symbol != nullptr &&
             ((symbol->kind == SymbolKind::Scalar &&
               symbol->type == ScalarType::Bool) ||
              (symbol->kind == SymbolKind::Constant && symbol->constant &&
               symbol->constant->type == ScalarType::Bool) ||
              (symbol->kind == SymbolKind::Register &&
               program.registers[symbol->id].kind == RegisterKind::Bit));
    }
    case Expr::Kind::Index:
      return true;
    default:
      return false;
    }
  }

  [[nodiscard]] FailureOr<ConditionId>
  analyzeBoolValue(const SyntaxExpressionId syntaxId) {
    if (expressionProducesBool(syntaxId)) {
      return analyzeCondition(syntaxId);
    }
    if (isConstantExpression(syntaxId)) {
      MQT_OQ3_TRY_ASSIGN(constant, evaluateConstant(syntaxId));
      return addCondition({.kind = ConditionKind::Literal,
                           .location = sourceLocation(
                               sources, syntax.expressions[syntaxId].location),
                           .literal = asDouble(constant) != 0.0});
    }
    MQT_OQ3_TRY_ASSIGN(value, analyzeExpression(syntaxId));
    const auto type = program.expressions[value].type;
    auto zeroValue = Constant{.type = ScalarType::Int, .value = int64_t{0}};
    if (type == ScalarType::Float || type == ScalarType::Angle) {
      zeroValue = Constant{.type = ScalarType::Float, .value = 0.0};
    } else if (type == ScalarType::Uint) {
      zeroValue = Constant{.type = ScalarType::Uint, .value = uint64_t{0}};
    }
    const auto zero = addConstant(zeroValue);
    return addCondition({.kind = ConditionKind::Comparison,
                         .location = sourceLocation(
                             sources, syntax.expressions[syntaxId].location),
                         .comparisonLhs = value,
                         .comparisonRhs = zero,
                         .comparison = ComparisonKind::NotEqual});
  }

  [[nodiscard]] static std::optional<Constant>
  builtinConstant(StringRef identifier) {
    if (identifier == "pi" || identifier == "π") {
      return Constant{.type = ScalarType::Float, .value = std::numbers::pi};
    }
    if (identifier == "tau" || identifier == "τ") {
      return Constant{.type = ScalarType::Float,
                      .value = 2.0 * std::numbers::pi};
    }
    if (identifier == "euler" || identifier == "ℇ") {
      return Constant{.type = ScalarType::Float, .value = std::numbers::e};
    }
    return std::nullopt;
  }

  [[nodiscard]] FailureOr<Constant>
  evaluateConstant(const SyntaxExpressionId id) const {
    if (constantValues[id]) {
      return *constantValues[id];
    }
    const auto result = [&]() -> FailureOr<Constant> {
      const auto& expression = syntax.expressions[id];
      switch (expression.kind) {
      case Expr::Kind::Int:
        if (!expression.wideInteger.empty()) {
          return fail(expression.location,
                      "integer literal exceeds 64-bit constant evaluation");
        }
        if (expression.integer <=
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          return Constant{.type = ScalarType::Int,
                          .value = static_cast<int64_t>(expression.integer)};
        }
        return Constant{.type = ScalarType::Uint, .value = expression.integer};
      case Expr::Kind::Float:
        return Constant{.type = ScalarType::Float,
                        .value = expression.floatingPoint};
      case Expr::Kind::Bool:
        return Constant{.type = ScalarType::Bool, .value = expression.boolean};
      case Expr::Kind::Identifier: {
        if (const auto builtin = builtinConstant(expression.identifier)) {
          return *builtin;
        }
        const auto* symbol = lookup(expression.identifier);
        if (symbol == nullptr || symbol->kind != SymbolKind::Constant ||
            !symbol->constant) {
          return fail(expression.location,
                      "expression is not a compile-time constant");
        }
        return *symbol->constant;
      }
      case Expr::Kind::AngleCast: {
        MQT_OQ3_TRY_ASSIGN(width,
                           angleWidth(expression.lhs, expression.location));
        MQT_OQ3_TRY_ASSIGN(operand, evaluateConstant(*expression.rhs));
        return convertToFixedAngle(operand, width, expression.location);
      }
      case Expr::Kind::Neg: {
        MQT_OQ3_TRY_ASSIGN(operand, evaluateConstant(*expression.lhs));
        if (operand.type == ScalarType::Bool) {
          return fail(expression.location,
                      "numeric negation requires a numeric operand");
        }
        if (operand.type == ScalarType::Angle) {
          const auto angle = std::get<FixedAngle>(operand.value);
          return Constant{.type = ScalarType::Angle,
                          .value =
                              FixedAngle{.bits = (uint64_t{0} - angle.bits) &
                                                 angleMask(angle.bitWidth),
                                         .bitWidth = angle.bitWidth}};
        }
        if (operand.type == ScalarType::Float) {
          return Constant{.type = ScalarType::Float,
                          .value = -std::get<double>(operand.value)};
        }
        if (operand.type == ScalarType::Uint) {
          const auto value = std::get<uint64_t>(operand.value);
          if (syntax.expressions[*expression.lhs].kind == Expr::Kind::Int) {
            if (value > (1ULL << 63)) {
              return fail(expression.location,
                          "integer negation overflows i64");
            }
            return Constant{.type = ScalarType::Int,
                            .value = std::numeric_limits<int64_t>::min()};
          }
          return Constant{.type = ScalarType::Uint, .value = 0ULL - value};
        }
        const auto value = std::get<int64_t>(operand.value);
        if (value == std::numeric_limits<int64_t>::min()) {
          return fail(expression.location, "integer negation overflows i64");
        }
        return Constant{.type = ScalarType::Int, .value = -value};
      }
      case Expr::Kind::Not: {
        MQT_OQ3_TRY_ASSIGN(operand, evaluateConstant(*expression.lhs));
        if (operand.type != ScalarType::Bool) {
          return fail(expression.location,
                      "logical negation requires a bool operand");
        }
        return Constant{.type = ScalarType::Bool,
                        .value = !std::get<bool>(operand.value)};
      }
      case Expr::Kind::BitNot: {
        return fail(
            expression.location,
            "bitwise operators require explicitly sized uint, bit, or angle "
            "operands, which are not supported yet");
      }
      case Expr::Kind::And:
      case Expr::Kind::Or: {
        MQT_OQ3_TRY_ASSIGN(lhs, evaluateConstant(*expression.lhs));
        if (lhs.type != ScalarType::Bool) {
          return fail(expression.location,
                      "logical operators require bool operands");
        }
        const auto left = std::get<bool>(lhs.value);
        const auto shortCircuits =
            expression.kind == Expr::Kind::And ? !left : left;
        if (shortCircuits) {
          MQT_OQ3_TRY_ASSIGN(rhsType, constantExpressionType(*expression.rhs));
          if (rhsType != ScalarType::Bool) {
            return fail(expression.location,
                        "logical operators require bool operands");
          }
          return Constant{.type = ScalarType::Bool, .value = left};
        }
        MQT_OQ3_TRY_ASSIGN(rhs, evaluateConstant(*expression.rhs));
        if (rhs.type != ScalarType::Bool) {
          return fail(expression.location,
                      "logical operators require bool operands");
        }
        const auto right = std::get<bool>(rhs.value);
        return Constant{.type = ScalarType::Bool,
                        .value = expression.kind == Expr::Kind::And
                                     ? left && right
                                     : left || right};
      }
      case Expr::Kind::Equal:
      case Expr::Kind::NotEqual:
      case Expr::Kind::Less:
      case Expr::Kind::LessEqual:
      case Expr::Kind::Greater:
      case Expr::Kind::GreaterEqual: {
        MQT_OQ3_TRY_ASSIGN(lhs, evaluateConstant(*expression.lhs));
        MQT_OQ3_TRY_ASSIGN(rhs, evaluateConstant(*expression.rhs));
        bool result = false;
        if (lhs.type == ScalarType::Bool || rhs.type == ScalarType::Bool) {
          if (lhs.type != ScalarType::Bool || rhs.type != ScalarType::Bool ||
              (expression.kind != Expr::Kind::Equal &&
               expression.kind != Expr::Kind::NotEqual)) {
            return fail(
                expression.location,
                "bool values only support equality comparisons with bool "
                "values");
          }
          const auto equal =
              std::get<bool>(lhs.value) == std::get<bool>(rhs.value);
          result = expression.kind == Expr::Kind::Equal ? equal : !equal;
        } else {
          auto ordering = 0;
          if (lhs.type == ScalarType::Angle || rhs.type == ScalarType::Angle) {
            if ((lhs.type != ScalarType::Angle &&
                 lhs.type != ScalarType::Float) ||
                (rhs.type != ScalarType::Angle &&
                 rhs.type != ScalarType::Float)) {
              return fail(expression.location,
                          "angles can only be compared with angles or finite "
                          "float constants");
            }
            uint32_t commonWidth = 0;
            if (lhs.type == ScalarType::Angle) {
              commonWidth = std::get<FixedAngle>(lhs.value).bitWidth;
            }
            if (rhs.type == ScalarType::Angle) {
              commonWidth = std::max(commonWidth,
                                     std::get<FixedAngle>(rhs.value).bitWidth);
            }
            MQT_OQ3_TRY_ASSIGN(
                leftAngle,
                convertToFixedAngle(lhs, commonWidth, expression.location));
            MQT_OQ3_TRY_ASSIGN(
                rightAngle,
                convertToFixedAngle(rhs, commonWidth, expression.location));
            const auto left =
                resizeAngle(std::get<FixedAngle>(leftAngle.value), commonWidth);
            const auto right = resizeAngle(
                std::get<FixedAngle>(rightAngle.value), commonWidth);
            if (left.bits < right.bits) {
              ordering = -1;
            } else if (left.bits > right.bits) {
              ordering = 1;
            }
          } else {
            ordering = compareNumericConstants(lhs, rhs);
          }
          switch (expression.kind) {
          case Expr::Kind::Equal:
            result = ordering == 0;
            break;
          case Expr::Kind::NotEqual:
            result = ordering != 0;
            break;
          case Expr::Kind::Less:
            result = ordering < 0;
            break;
          case Expr::Kind::LessEqual:
            result = ordering <= 0;
            break;
          case Expr::Kind::Greater:
            result = ordering > 0;
            break;
          case Expr::Kind::GreaterEqual:
            result = ordering >= 0;
            break;
          default:
            llvm_unreachable("not a comparison expression");
          }
        }
        return Constant{.type = ScalarType::Bool, .value = result};
      }
      case Expr::Kind::ArcCos:
      case Expr::Kind::ArcSin:
      case Expr::Kind::ArcTan:
      case Expr::Kind::Ceiling:
      case Expr::Kind::Cos:
      case Expr::Kind::Exp:
      case Expr::Kind::Floor:
      case Expr::Kind::Log:
      case Expr::Kind::Sin:
      case Expr::Kind::Sqrt:
      case Expr::Kind::Tan: {
        MQT_OQ3_TRY_ASSIGN(operand, evaluateConstant(*expression.lhs));
        if (operand.type == ScalarType::Bool) {
          return fail(expression.location,
                      "math functions require numeric operands");
        }
        const bool inverseTrig = expression.kind == Expr::Kind::ArcCos ||
                                 expression.kind == Expr::Kind::ArcSin ||
                                 expression.kind == Expr::Kind::ArcTan;
        const bool trig = expression.kind == Expr::Kind::Cos ||
                          expression.kind == Expr::Kind::Sin ||
                          expression.kind == Expr::Kind::Tan;
        if (operand.type == ScalarType::Angle && !trig) {
          return fail(
              expression.location,
              inverseTrig
                  ? "inverse trigonometric functions require a float operand"
                  : "this math function does not accept an angle operand");
        }
        const auto value = asDouble(operand);
        double result = 0.0;
        switch (expression.kind) {
        case Expr::Kind::ArcCos:
          result = std::acos(value);
          break;
        case Expr::Kind::ArcSin:
          result = std::asin(value);
          break;
        case Expr::Kind::ArcTan:
          result = std::atan(value);
          break;
        case Expr::Kind::Ceiling:
          result = std::ceil(value);
          break;
        case Expr::Kind::Cos:
          result = std::cos(value);
          break;
        case Expr::Kind::Exp:
          result = std::exp(value);
          break;
        case Expr::Kind::Floor:
          result = std::floor(value);
          break;
        case Expr::Kind::Log:
          result = std::log(value);
          break;
        case Expr::Kind::Sin:
          result = std::sin(value);
          break;
        case Expr::Kind::Sqrt:
          result = std::sqrt(value);
          break;
        case Expr::Kind::Tan:
          result = std::tan(value);
          break;
        default:
          llvm_unreachable("not a unary math expression");
        }
        if (!std::isfinite(result)) {
          return fail(expression.location,
                      "constant math expression has a non-finite result");
        }
        return Constant{.type = ScalarType::Float, .value = result};
      }
      case Expr::Kind::Add:
      case Expr::Kind::Sub:
      case Expr::Kind::Mul:
      case Expr::Kind::Div:
      case Expr::Kind::Mod:
      case Expr::Kind::BuiltinMod:
      case Expr::Kind::BuiltinPow:
      case Expr::Kind::Pow:
        return evaluateConstantBinary(expression);
      case Expr::Kind::BitAnd:
      case Expr::Kind::BitOr:
      case Expr::Kind::BitXor:
      case Expr::Kind::ShiftLeft:
      case Expr::Kind::ShiftRight:
        return fail(
            expression.location,
            "bitwise operators require explicitly sized uint, bit, or angle "
            "operands, which are not supported yet");
      case Expr::Kind::Index:
        return fail(expression.location,
                    "expression is not a compile-time constant");
      case Expr::Kind::PopCount:
      case Expr::Kind::RotateLeft:
      case Expr::Kind::RotateRight:
        return fail(expression.location,
                    "bit-register expressions are not compile-time constants");
      }
      llvm_unreachable("unknown syntax expression kind");
    }();
    if (failed(result)) {
      return failure();
    }
    constantValues[id] = *result;
    return *result;
  }

  [[nodiscard]] FailureOr<ScalarType>
  constantExpressionType(const SyntaxExpressionId id) const {
    if (constantTypes[id]) {
      return *constantTypes[id];
    }
    const auto result = [&]() -> FailureOr<ScalarType> {
      const auto& expression = syntax.expressions[id];
      switch (expression.kind) {
      case Expr::Kind::Int:
        if (!expression.wideInteger.empty()) {
          return fail(expression.location,
                      "integer literal exceeds 64-bit constant evaluation");
        }
        return expression.integer <= static_cast<uint64_t>(
                                         std::numeric_limits<int64_t>::max())
                   ? ScalarType::Int
                   : ScalarType::Uint;
      case Expr::Kind::Float:
        return ScalarType::Float;
      case Expr::Kind::Bool:
        return ScalarType::Bool;
      case Expr::Kind::Identifier: {
        if (const auto builtin = builtinConstant(expression.identifier)) {
          return builtin->type;
        }
        const auto* symbol = lookup(expression.identifier);
        if (symbol == nullptr || symbol->kind != SymbolKind::Constant ||
            !symbol->constant) {
          return fail(expression.location,
                      "expression is not a compile-time constant");
        }
        return symbol->constant->type;
      }
      case Expr::Kind::AngleCast: {
        MQT_OQ3_TRY_ASSIGN(constant, evaluateConstant(id));
        return constant.type;
      }
      case Expr::Kind::Neg: {
        MQT_OQ3_TRY_ASSIGN(type, constantExpressionType(*expression.lhs));
        if (type == ScalarType::Bool) {
          return fail(expression.location,
                      "numeric negation requires a numeric operand");
        }
        return type;
      }
      case Expr::Kind::Not: {
        MQT_OQ3_TRY_ASSIGN(type, constantExpressionType(*expression.lhs));
        if (type != ScalarType::Bool) {
          return fail(expression.location,
                      "logical negation requires a bool operand");
        }
        return ScalarType::Bool;
      }
      case Expr::Kind::And:
      case Expr::Kind::Or: {
        MQT_OQ3_TRY_ASSIGN(lhs, constantExpressionType(*expression.lhs));
        MQT_OQ3_TRY_ASSIGN(rhs, constantExpressionType(*expression.rhs));
        if (lhs != ScalarType::Bool || rhs != ScalarType::Bool) {
          return fail(expression.location,
                      "logical operators require bool operands");
        }
        return ScalarType::Bool;
      }
      case Expr::Kind::Equal:
      case Expr::Kind::NotEqual:
      case Expr::Kind::Less:
      case Expr::Kind::LessEqual:
      case Expr::Kind::Greater:
      case Expr::Kind::GreaterEqual: {
        MQT_OQ3_TRY_ASSIGN(lhs, constantExpressionType(*expression.lhs));
        MQT_OQ3_TRY_ASSIGN(rhs, constantExpressionType(*expression.rhs));
        if (lhs == ScalarType::Bool || rhs == ScalarType::Bool) {
          if (lhs != ScalarType::Bool || rhs != ScalarType::Bool ||
              (expression.kind != Expr::Kind::Equal &&
               expression.kind != Expr::Kind::NotEqual)) {
            return fail(
                expression.location,
                "bool values only support equality comparisons with bool "
                "values");
          }
        }
        return ScalarType::Bool;
      }
      case Expr::Kind::ArcCos:
      case Expr::Kind::ArcSin:
      case Expr::Kind::ArcTan: {
        MQT_OQ3_TRY_ASSIGN(type, constantExpressionType(*expression.lhs));
        if (type == ScalarType::Angle) {
          return fail(
              expression.location,
              "inverse trigonometric functions require a float operand");
        }
        if (type == ScalarType::Bool) {
          return fail(expression.location,
                      "math functions require numeric operands");
        }
        return ScalarType::Float;
      }
      case Expr::Kind::Ceiling:
      case Expr::Kind::Exp:
      case Expr::Kind::Floor:
      case Expr::Kind::Log:
      case Expr::Kind::Sqrt: {
        MQT_OQ3_TRY_ASSIGN(type, constantExpressionType(*expression.lhs));
        if (type == ScalarType::Bool) {
          return fail(expression.location,
                      "math functions require numeric operands");
        }
        if (type == ScalarType::Angle) {
          return fail(expression.location,
                      "this math function does not accept an angle operand");
        }
        return ScalarType::Float;
      }
      case Expr::Kind::Cos:
      case Expr::Kind::Sin:
      case Expr::Kind::Tan: {
        MQT_OQ3_TRY_ASSIGN(type, constantExpressionType(*expression.lhs));
        if (type == ScalarType::Bool) {
          return fail(expression.location,
                      "math functions require numeric operands");
        }
        return ScalarType::Float;
      }
      case Expr::Kind::Add:
      case Expr::Kind::Sub:
      case Expr::Kind::Mul:
      case Expr::Kind::Div:
      case Expr::Kind::Mod:
      case Expr::Kind::BuiltinMod:
      case Expr::Kind::BuiltinPow:
      case Expr::Kind::Pow: {
        MQT_OQ3_TRY_ASSIGN(lhs, constantExpressionType(*expression.lhs));
        MQT_OQ3_TRY_ASSIGN(rhs, constantExpressionType(*expression.rhs));
        if (lhs == ScalarType::Bool || rhs == ScalarType::Bool) {
          return fail(expression.location,
                      "arithmetic operators require numeric operands");
        }
        if (expression.kind == Expr::Kind::Mod &&
            (lhs == ScalarType::Float || rhs == ScalarType::Float)) {
          return fail(expression.location,
                      "the '%' operator requires integer operands; use mod() "
                      "for floating-point remainder");
        }
        if (lhs == ScalarType::Angle || rhs == ScalarType::Angle) {
          if (expression.kind == Expr::Kind::Add ||
              expression.kind == Expr::Kind::Sub ||
              (expression.kind == Expr::Kind::Mul &&
               lhs != ScalarType::Angle) ||
              (expression.kind == Expr::Kind::Mul &&
               rhs != ScalarType::Angle) ||
              (expression.kind == Expr::Kind::Div && lhs == ScalarType::Angle &&
               rhs != ScalarType::Angle)) {
            return ScalarType::Angle;
          }
          if (expression.kind == Expr::Kind::Div && lhs == ScalarType::Angle &&
              rhs == ScalarType::Angle) {
            return ScalarType::Float;
          }
          return fail(expression.location,
                      "unsupported arithmetic operation on angle operands");
        }
        if (lhs == ScalarType::Float || rhs == ScalarType::Float) {
          return ScalarType::Float;
        }
        if (expression.kind == Expr::Kind::BuiltinPow &&
            lhs == ScalarType::Int) {
          MQT_OQ3_TRY_ASSIGN(rhsConstant, evaluateConstant(*expression.rhs));
          if (rhs == ScalarType::Int &&
              std::get<int64_t>(rhsConstant.value) < 0) {
            return ScalarType::Float;
          }
          return ScalarType::Int;
        }
        return lhs == ScalarType::Uint || rhs == ScalarType::Uint
                   ? ScalarType::Uint
                   : ScalarType::Int;
      }
      case Expr::Kind::BitNot:
      case Expr::Kind::BitAnd:
      case Expr::Kind::BitOr:
      case Expr::Kind::BitXor:
      case Expr::Kind::ShiftLeft:
      case Expr::Kind::ShiftRight:
        return fail(
            expression.location,
            "bitwise operators require explicitly sized uint, bit, or angle "
            "operands, which are not supported yet");
      case Expr::Kind::Index:
        return fail(expression.location,
                    "expression is not a compile-time constant");
      case Expr::Kind::PopCount:
        return ScalarType::Uint;
      case Expr::Kind::RotateLeft:
      case Expr::Kind::RotateRight:
        return fail(expression.location,
                    "bit-register rotations are not scalar expressions");
      }
      llvm_unreachable("unknown syntax expression kind");
    }();
    if (failed(result)) {
      return failure();
    }
    constantTypes[id].emplace(*result);
    return *result;
  }

  [[nodiscard]] FailureOr<Constant>
  evaluateConstantBinary(const SyntaxExpression& expression) const {
    MQT_OQ3_TRY_ASSIGN(lhs, evaluateConstant(*expression.lhs));
    MQT_OQ3_TRY_ASSIGN(rhs, evaluateConstant(*expression.rhs));
    if (lhs.type == ScalarType::Bool || rhs.type == ScalarType::Bool) {
      return fail(expression.location,
                  "arithmetic operators require numeric operands");
    }
    const bool builtinFloatPower = expression.kind == Expr::Kind::BuiltinPow &&
                                   rhs.type == ScalarType::Int &&
                                   std::get<int64_t>(rhs.value) < 0;
    if (lhs.type == ScalarType::Angle || rhs.type == ScalarType::Angle) {
      if (expression.kind == Expr::Kind::Add ||
          expression.kind == Expr::Kind::Sub) {
        if (lhs.type != ScalarType::Angle || rhs.type != ScalarType::Angle) {
          return fail(expression.location,
                      "angle addition and subtraction require angle operands");
        }
        const auto lhsAngle = std::get<FixedAngle>(lhs.value);
        const auto rhsAngle = std::get<FixedAngle>(rhs.value);
        const auto bitWidth = std::max(lhsAngle.bitWidth, rhsAngle.bitWidth);
        const auto left = resizeAngle(lhsAngle, bitWidth).bits;
        const auto right = resizeAngle(rhsAngle, bitWidth).bits;
        const auto bits =
            expression.kind == Expr::Kind::Add ? left + right : left - right;
        return Constant{.type = ScalarType::Angle,
                        .value = FixedAngle{.bits = bits & angleMask(bitWidth),
                                            .bitWidth = bitWidth}};
      }
      if (expression.kind == Expr::Kind::Mul &&
          (lhs.type == ScalarType::Angle) != (rhs.type == ScalarType::Angle)) {
        const auto angle = std::get<FixedAngle>(
            lhs.type == ScalarType::Angle ? lhs.value : rhs.value);
        const auto literalId =
            lhs.type == ScalarType::Angle ? *expression.rhs : *expression.lhs;
        MQT_OQ3_TRY_ASSIGN(factor,
                           angleIntegerLiteral(literalId, angle.bitWidth));
        return Constant{.type = ScalarType::Angle,
                        .value = FixedAngle{.bits = (angle.bits * factor) &
                                                    angleMask(angle.bitWidth),
                                            .bitWidth = angle.bitWidth}};
      }
      if (expression.kind == Expr::Kind::Div && lhs.type == ScalarType::Angle &&
          rhs.type != ScalarType::Angle) {
        const auto angle = std::get<FixedAngle>(lhs.value);
        MQT_OQ3_TRY_ASSIGN(
            divisor, angleIntegerLiteral(*expression.rhs, angle.bitWidth));
        if (divisor == 0) {
          return fail(expression.location, "division by zero");
        }
        return Constant{.type = ScalarType::Angle,
                        .value = FixedAngle{.bits = angle.bits / divisor,
                                            .bitWidth = angle.bitWidth}};
      }
      return fail(expression.location,
                  "unsupported compile-time arithmetic on angle operands");
    }
    if (lhs.type == ScalarType::Float || rhs.type == ScalarType::Float ||
        builtinFloatPower) {
      if (expression.kind == Expr::Kind::Mod) {
        return fail(expression.location,
                    "the '%' operator requires integer operands; use mod() "
                    "for floating-point remainder");
      }
      const auto left = asDouble(lhs);
      const auto right = asDouble(rhs);
      double result = 0.0;
      switch (expression.kind) {
      case Expr::Kind::Add:
        result = left + right;
        break;
      case Expr::Kind::Sub:
        result = left - right;
        break;
      case Expr::Kind::Mul:
        result = left * right;
        break;
      case Expr::Kind::Div:
        if (right == 0.0) {
          return fail(expression.location, "division by zero");
        }
        result = left / right;
        break;
      case Expr::Kind::BuiltinMod:
        if (right == 0.0) {
          return fail(expression.location, "modulo by zero");
        }
        result = std::fmod(left, right);
        break;
      case Expr::Kind::Pow:
      case Expr::Kind::BuiltinPow:
        result = std::pow(left, right);
        break;
      default:
        llvm_unreachable("not a binary expression");
      }
      if (!std::isfinite(result)) {
        return fail(expression.location,
                    "constant arithmetic has a non-finite result");
      }
      return Constant{.type = ScalarType::Float, .value = result};
    }

    if (expression.kind == Expr::Kind::BuiltinPow &&
        lhs.type == ScalarType::Int && rhs.type == ScalarType::Uint) {
      auto result = int64_t{1};
      auto base = std::get<int64_t>(lhs.value);
      auto exponent = std::get<uint64_t>(rhs.value);
      bool overflow = false;
      while (exponent != 0 && !overflow) {
        if ((exponent & 1U) != 0) {
          overflow = llvm::MulOverflow(result, base, result) != 0;
        }
        exponent >>= 1U;
        if (exponent != 0 && !overflow) {
          overflow = llvm::MulOverflow(base, base, base) != 0;
        }
      }
      if (overflow) {
        return fail(expression.location,
                    "constant integer arithmetic overflows i64");
      }
      return Constant{.type = ScalarType::Int, .value = result};
    }

    if (lhs.type == ScalarType::Uint || rhs.type == ScalarType::Uint) {
      const auto asUnsigned = [](const Constant& constant) {
        return constant.type == ScalarType::Uint
                   ? std::get<uint64_t>(constant.value)
                   : static_cast<uint64_t>(std::get<int64_t>(constant.value));
      };
      const auto left = asUnsigned(lhs);
      const auto right = asUnsigned(rhs);
      uint64_t result = 0;
      switch (expression.kind) {
      case Expr::Kind::Add:
        result = left + right;
        break;
      case Expr::Kind::Sub:
        result = left - right;
        break;
      case Expr::Kind::Mul:
        result = left * right;
        break;
      case Expr::Kind::Div:
        if (right == 0) {
          return fail(expression.location, "division by zero");
        }
        result = left / right;
        break;
      case Expr::Kind::Mod:
      case Expr::Kind::BuiltinMod:
        if (right == 0) {
          return fail(expression.location, "modulo by zero");
        }
        result = left % right;
        break;
      case Expr::Kind::Pow:
      case Expr::Kind::BuiltinPow:
        result = 1;
        for (auto base = left, exponent = right; exponent != 0;
             exponent >>= 1U, base *= base) {
          if ((exponent & 1U) != 0) {
            result *= base;
          }
        }
        break;
      default:
        llvm_unreachable("not a binary expression");
      }
      return Constant{.type = ScalarType::Uint, .value = result};
    }

    const auto left = std::get<int64_t>(lhs.value);
    const auto right = std::get<int64_t>(rhs.value);
    int64_t result = 0;
    bool overflow = false;
    switch (expression.kind) {
    case Expr::Kind::Add:
      overflow = llvm::AddOverflow(left, right, result) != 0;
      break;
    case Expr::Kind::Sub:
      overflow = llvm::SubOverflow(left, right, result) != 0;
      break;
    case Expr::Kind::Mul:
      overflow = llvm::MulOverflow(left, right, result) != 0;
      break;
    case Expr::Kind::Div:
      if (right == 0) {
        return fail(expression.location, "division by zero");
      }
      if (left == std::numeric_limits<int64_t>::min() && right == -1) {
        overflow = true;
      } else {
        result = left / right;
      }
      break;
    case Expr::Kind::Mod:
    case Expr::Kind::BuiltinMod:
      if (right == 0) {
        return fail(expression.location, "modulo by zero");
      }
      if (left == std::numeric_limits<int64_t>::min() && right == -1) {
        overflow = true;
      } else {
        result = left % right;
      }
      break;
    case Expr::Kind::Pow:
    case Expr::Kind::BuiltinPow: {
      if (right < 0) {
        assert(expression.kind == Expr::Kind::Pow &&
               "negative built-in powers use the floating overload");
        return fail(expression.location,
                    "integer power requires a nonnegative exponent");
      }
      result = 1;
      auto base = left;
      auto exponent = static_cast<uint64_t>(right);
      while (exponent != 0 && !overflow) {
        if ((exponent & 1U) != 0) {
          overflow = llvm::MulOverflow(result, base, result) != 0;
        }
        exponent >>= 1U;
        if (exponent != 0 && !overflow) {
          overflow = llvm::MulOverflow(base, base, base) != 0;
        }
      }
      break;
    }
    default:
      llvm_unreachable("not a binary expression");
    }
    if (overflow) {
      return fail(expression.location,
                  "constant integer arithmetic overflows i64");
    }
    return Constant{.type = ScalarType::Int, .value = result};
  }

  [[nodiscard]] bool isConstantExpression(const SyntaxExpressionId id) const {
    if (constantExpressionStatus[id] != 0) {
      return constantExpressionStatus[id] > 0;
    }
    const auto& expression = syntax.expressions[id];
    const auto result = [&] {
      switch (expression.kind) {
      case Expr::Kind::Identifier: {
        if (builtinConstant(expression.identifier)) {
          return true;
        }
        const auto* symbol = lookup(expression.identifier);
        return symbol != nullptr && symbol->kind == SymbolKind::Constant;
      }
      case Expr::Kind::Int:
      case Expr::Kind::Float:
      case Expr::Kind::Bool:
        return true;
      case Expr::Kind::Index:
      case Expr::Kind::PopCount:
      case Expr::Kind::RotateLeft:
      case Expr::Kind::RotateRight:
        return false;
      default:
        return (!expression.lhs || isConstantExpression(*expression.lhs)) &&
               (!expression.rhs || isConstantExpression(*expression.rhs));
      }
    }();
    constantExpressionStatus[id] = result ? 1 : -1;
    return result;
  }

  [[nodiscard]] LogicalResult
  validateGateExpression(const SyntaxExpressionId id) const {
    const auto& expression = syntax.expressions[id];
    if (expression.kind == Expr::Kind::Identifier &&
        !builtinConstant(expression.identifier)) {
      const auto* symbol = lookup(expression.identifier);
      if (symbol == nullptr || (symbol->kind != SymbolKind::GateParameter &&
                                symbol->kind != SymbolKind::GateLocalScalar &&
                                symbol->kind != SymbolKind::Constant)) {
        return fail(expression.location,
                    "gate definitions cannot capture outer scalar '" +
                        expression.identifier + "'");
      }
    }
    return success();
  }

  [[nodiscard]] FailureOr<BitVectorExpressionId>
  analyzeBitVectorExpression(const SyntaxExpressionId syntaxId) {
    const auto& expression = syntax.expressions[syntaxId];
    if (expression.kind == Expr::Kind::Identifier) {
      const auto* symbol = lookup(expression.identifier);
      if (symbol == nullptr || symbol->kind != SymbolKind::Register ||
          program.registers[symbol->id].kind != RegisterKind::Bit) {
        return fail(expression.location,
                    "bit-vector expression requires a bit register");
      }
      const auto reg = static_cast<RegisterId>(symbol->id);
      if (program.registers[reg].isScalar) {
        return fail(
            expression.location,
            "bit-vector expression requires a bit register, not scalar bit");
      }
      const auto width = program.registers[reg].width;
      for (uint64_t bit = 0; bit < width; ++bit) {
        if (failed(ensureBitInitialized({.reg = reg, .index = bit},
                                        expression.location))) {
          return failure();
        }
      }
      return addBitVectorExpression({.kind = BitVectorExpressionKind::Register,
                                     .width = width,
                                     .reg = reg});
    }
    if (expression.kind != Expr::Kind::RotateLeft &&
        expression.kind != Expr::Kind::RotateRight) {
      return fail(expression.location,
                  "bit-vector expression requires a bit register or rotation");
    }
    MQT_OQ3_TRY_ASSIGN(operand, analyzeBitVectorExpression(*expression.lhs));
    MQT_OQ3_TRY_ASSIGN(distance, analyzeExpression(*expression.rhs));
    if (program.expressions[distance].type != ScalarType::Int) {
      return fail(syntax.expressions[*expression.rhs].location,
                  "bit-register rotation distance must have signed int type");
    }
    return addBitVectorExpression(
        {.kind = expression.kind == Expr::Kind::RotateLeft
                     ? BitVectorExpressionKind::RotateLeft
                     : BitVectorExpressionKind::RotateRight,
         .width = program.bitVectorExpressions[operand].width,
         .operand = operand,
         .distance = distance});
  }

  [[nodiscard]] FailureOr<ExpressionId>
  analyzeExpression(const SyntaxExpressionId syntaxId) {
    const auto& expression = syntax.expressions[syntaxId];
    if (insideGate && failed(validateGateExpression(syntaxId))) {
      return failure();
    }
    if (isConstantExpression(syntaxId)) {
      MQT_OQ3_TRY_ASSIGN(constant, evaluateConstant(syntaxId));
      return addConstant(constant);
    }
    if (expression.kind == Expr::Kind::PopCount) {
      MQT_OQ3_TRY_ASSIGN(bitVector,
                         analyzeBitVectorExpression(*expression.lhs));
      return addExpression({.kind = ExpressionKind::PopCount,
                            .type = ScalarType::Uint,
                            .bitVector = bitVector});
    }
    if (expression.kind == Expr::Kind::Identifier) {
      const auto* symbol = lookup(expression.identifier);
      if (symbol == nullptr) {
        return fail(expression.location, "unknown scalar identifier '" +
                                             expression.identifier + "'");
      }
      if (symbol->kind == SymbolKind::GateParameter) {
        return addExpression({.kind = ExpressionKind::GateParameter,
                              .type = ScalarType::Angle,
                              .parameter = symbol->id});
      }
      if (symbol->kind != SymbolKind::Scalar &&
          symbol->kind != SymbolKind::GateLocalScalar) {
        return fail(expression.location, "identifier '" +
                                             expression.identifier +
                                             "' is not a scalar value");
      }
      if (!initializedScalars.at(symbol->id)) {
        return fail(expression.location,
                    "scalar '" + expression.identifier + "' is uninitialized");
      }
      return addExpression({.kind = ExpressionKind::Variable,
                            .type = symbol->type,
                            .variable = symbol->id});
    }

    auto kind = ExpressionKind::Constant;
    switch (expression.kind) {
    case Expr::Kind::AngleCast:
      return fail(expression.location,
                  "runtime angle conversions are not supported");
    case Expr::Kind::Neg:
      kind = ExpressionKind::Negate;
      break;
    case Expr::Kind::BitNot:
      return fail(
          expression.location,
          "bitwise operators require explicitly sized uint, bit, or angle "
          "operands, which are not supported yet");
    case Expr::Kind::Add:
      kind = ExpressionKind::Add;
      break;
    case Expr::Kind::Sub:
      kind = ExpressionKind::Subtract;
      break;
    case Expr::Kind::Mul:
      kind = ExpressionKind::Multiply;
      break;
    case Expr::Kind::Div:
      kind = ExpressionKind::Divide;
      break;
    case Expr::Kind::Mod:
    case Expr::Kind::BuiltinMod:
      kind = ExpressionKind::Modulo;
      break;
    case Expr::Kind::Pow:
    case Expr::Kind::BuiltinPow:
      kind = ExpressionKind::Power;
      break;
    case Expr::Kind::BitAnd:
    case Expr::Kind::BitOr:
    case Expr::Kind::BitXor:
    case Expr::Kind::ShiftLeft:
    case Expr::Kind::ShiftRight:
      return fail(
          expression.location,
          "bitwise operators require explicitly sized uint, bit, or angle "
          "operands, which are not supported yet");
    case Expr::Kind::RotateLeft:
    case Expr::Kind::RotateRight:
      return fail(expression.location,
                  "bit-register rotations require a whole-register assignment");
    case Expr::Kind::PopCount:
      llvm_unreachable("handled bit-register population count");
    case Expr::Kind::ArcCos:
      kind = ExpressionKind::ArcCos;
      break;
    case Expr::Kind::ArcSin:
      kind = ExpressionKind::ArcSin;
      break;
    case Expr::Kind::ArcTan:
      kind = ExpressionKind::ArcTan;
      break;
    case Expr::Kind::Ceiling:
      kind = ExpressionKind::Ceiling;
      break;
    case Expr::Kind::Cos:
      kind = ExpressionKind::Cos;
      break;
    case Expr::Kind::Exp:
      kind = ExpressionKind::Exp;
      break;
    case Expr::Kind::Floor:
      kind = ExpressionKind::Floor;
      break;
    case Expr::Kind::Log:
      kind = ExpressionKind::Log;
      break;
    case Expr::Kind::Sin:
      kind = ExpressionKind::Sin;
      break;
    case Expr::Kind::Sqrt:
      kind = ExpressionKind::Sqrt;
      break;
    case Expr::Kind::Tan:
      kind = ExpressionKind::Tan;
      break;
    case Expr::Kind::Not:
    case Expr::Kind::Equal:
    case Expr::Kind::NotEqual:
    case Expr::Kind::Less:
    case Expr::Kind::LessEqual:
    case Expr::Kind::Greater:
    case Expr::Kind::GreaterEqual:
    case Expr::Kind::And:
    case Expr::Kind::Or:
    case Expr::Kind::Index:
      return fail(expression.location,
                  "expected a scalar arithmetic expression");
    case Expr::Kind::Int:
    case Expr::Kind::Float:
    case Expr::Kind::Bool:
    case Expr::Kind::Identifier:
      llvm_unreachable("handled expression kind");
    }
    MQT_OQ3_TRY_ASSIGN(lhs, analyzeExpression(*expression.lhs));
    std::optional<ExpressionId> rhs;
    if (expression.rhs) {
      MQT_OQ3_TRY_ASSIGN(evaluatedRhs, analyzeExpression(*expression.rhs));
      rhs = evaluatedRhs;
    }
    auto lhsType = program.expressions[lhs].type;
    auto rhsType =
        rhs ? std::optional<ScalarType>(program.expressions[*rhs].type)
            : std::nullopt;
    if (lhsType == ScalarType::Bool ||
        (rhsType && *rhsType == ScalarType::Bool)) {
      return fail(expression.location,
                  "arithmetic operators require numeric operands");
    }

    const bool inverseTrig = kind == ExpressionKind::ArcCos ||
                             kind == ExpressionKind::ArcSin ||
                             kind == ExpressionKind::ArcTan;
    const bool trig = kind == ExpressionKind::Cos ||
                      kind == ExpressionKind::Sin ||
                      kind == ExpressionKind::Tan;
    const bool otherMath =
        kind == ExpressionKind::Ceiling || kind == ExpressionKind::Exp ||
        kind == ExpressionKind::Floor || kind == ExpressionKind::Log ||
        kind == ExpressionKind::Sqrt;
    if (inverseTrig || trig || otherMath) {
      if (lhsType == ScalarType::Angle && !trig) {
        return fail(
            expression.location,
            inverseTrig
                ? "inverse trigonometric functions require a float operand"
                : "this math function does not accept an angle operand");
      }
      MQT_OQ3_TRY_ASSIGN(
          convertedLhs,
          castExpression(lhs, trig ? ScalarType::Angle : ScalarType::Float,
                         expression.location));
      lhs = convertedLhs;
      return addExpression(
          {.kind = kind, .type = ScalarType::Float, .lhs = lhs});
    }
    if (!rhs) {
      return addExpression({.kind = kind, .type = lhsType, .lhs = lhs});
    }

    if (expression.kind == Expr::Kind::Mod &&
        (lhsType == ScalarType::Float || lhsType == ScalarType::Angle ||
         *rhsType == ScalarType::Float || *rhsType == ScalarType::Angle)) {
      return fail(expression.location,
                  "the '%' operator requires integer operands; use mod() for "
                  "floating-point remainder");
    }

    if (lhsType == ScalarType::Angle || *rhsType == ScalarType::Angle) {
      ScalarType type = ScalarType::Angle;
      if (kind == ExpressionKind::Add || kind == ExpressionKind::Subtract) {
        MQT_OQ3_TRY_ASSIGN(convertedLhs, castExpression(lhs, ScalarType::Angle,
                                                        expression.location));
        MQT_OQ3_TRY_ASSIGN(convertedRhs, castExpression(*rhs, ScalarType::Angle,
                                                        expression.location));
        lhs = convertedLhs;
        *rhs = convertedRhs;
      } else if (kind == ExpressionKind::Multiply &&
                 (lhsType == ScalarType::Angle) !=
                     (*rhsType == ScalarType::Angle)) {
        if (lhsType != ScalarType::Angle) {
          MQT_OQ3_TRY_ASSIGN(
              convertedLhs,
              castExpression(lhs, ScalarType::Float, expression.location));
          lhs = convertedLhs;
        } else {
          MQT_OQ3_TRY_ASSIGN(
              convertedRhs,
              castExpression(*rhs, ScalarType::Float, expression.location));
          *rhs = convertedRhs;
        }
      } else if (kind == ExpressionKind::Divide &&
                 lhsType == ScalarType::Angle) {
        if (*rhsType == ScalarType::Angle) {
          type = ScalarType::Float;
        } else {
          MQT_OQ3_TRY_ASSIGN(
              convertedRhs,
              castExpression(*rhs, ScalarType::Float, expression.location));
          *rhs = convertedRhs;
        }
      } else {
        return fail(expression.location,
                    "unsupported arithmetic operation on angle operands");
      }
      return addExpression(
          {.kind = kind, .type = type, .lhs = lhs, .rhs = *rhs});
    }

    if (expression.kind == Expr::Kind::BuiltinPow) {
      auto type = ScalarType::Float;
      if (lhsType == ScalarType::Int && *rhsType == ScalarType::Uint) {
        type = ScalarType::Int;
      } else if (lhsType == ScalarType::Int && *rhsType == ScalarType::Int &&
                 program.expressions[*rhs].kind == ExpressionKind::Constant &&
                 std::get<int64_t>(program.expressions[*rhs].constant) >= 0) {
        type = ScalarType::Int;
        MQT_OQ3_TRY_ASSIGN(convertedRhs, castExpression(*rhs, ScalarType::Uint,
                                                        expression.location));
        *rhs = convertedRhs;
      } else if (lhsType == ScalarType::Uint &&
                 (*rhsType == ScalarType::Uint ||
                  (*rhsType == ScalarType::Int &&
                   program.expressions[*rhs].kind == ExpressionKind::Constant &&
                   std::get<int64_t>(program.expressions[*rhs].constant) >=
                       0))) {
        type = ScalarType::Uint;
        MQT_OQ3_TRY_ASSIGN(convertedRhs, castExpression(*rhs, ScalarType::Uint,
                                                        expression.location));
        *rhs = convertedRhs;
      } else {
        MQT_OQ3_TRY_ASSIGN(convertedLhs, castExpression(lhs, ScalarType::Float,
                                                        expression.location));
        MQT_OQ3_TRY_ASSIGN(convertedRhs, castExpression(*rhs, ScalarType::Float,
                                                        expression.location));
        lhs = convertedLhs;
        *rhs = convertedRhs;
      }
      return addExpression(
          {.kind = kind, .type = type, .lhs = lhs, .rhs = *rhs});
    }

    auto type = ScalarType::Int;
    if (lhsType == ScalarType::Float || *rhsType == ScalarType::Float) {
      type = ScalarType::Float;
    } else if (lhsType == ScalarType::Uint || *rhsType == ScalarType::Uint) {
      type = ScalarType::Uint;
    }
    MQT_OQ3_TRY_ASSIGN(convertedLhs,
                       castExpression(lhs, type, expression.location));
    MQT_OQ3_TRY_ASSIGN(convertedRhs,
                       castExpression(*rhs, type, expression.location));
    lhs = convertedLhs;
    *rhs = convertedRhs;
    return addExpression({.kind = kind, .type = type, .lhs = lhs, .rhs = *rhs});
  }

  [[nodiscard]] FailureOr<uint64_t>
  constantWidth(const std::optional<SyntaxExpressionId> size,
                SMLoc location) const {
    if (!size) {
      return 1;
    }
    if (!isConstantExpression(*size)) {
      return fail(location,
                  "register width must be a constant integer expression");
    }
    MQT_OQ3_TRY_ASSIGN(constant, evaluateConstant(*size));
    if (!isInteger(constant.type)) {
      return fail(location, "register width must be an integer expression");
    }
    const auto value =
        constant.type == ScalarType::Uint
            ? std::get<uint64_t>(constant.value)
            : static_cast<uint64_t>(std::get<int64_t>(constant.value));
    if (value == 0 || (constant.type == ScalarType::Int &&
                       std::get<int64_t>(constant.value) < 0)) {
      return fail(location, "register width must be greater than zero");
    }
    if (value > REGISTER_WIDTH_LIMIT) {
      return fail(location, Twine("register width exceeds the limit of ") +
                                Twine(REGISTER_WIDTH_LIMIT));
    }
    return value;
  }

  [[nodiscard]] FailureOr<std::optional<uint64_t>>
  constantIndex(const SyntaxExpressionId id, const uint64_t width,
                SMLoc location) const {
    if (!isConstantExpression(id)) {
      return std::optional<uint64_t>{};
    }
    MQT_OQ3_TRY_ASSIGN(constant, evaluateConstant(id));
    if (!isInteger(constant.type)) {
      return fail(location, "index must be an integer expression");
    }
    auto value = asSigned(constant);
    if (!value) {
      return fail(location, "unsigned value does not fit in signed i64");
    }
    if (*value < 0) {
      *value += static_cast<int64_t>(width);
    }
    if (*value < 0) {
      return fail(location, "index is out of bounds");
    }
    return std::optional<uint64_t>(static_cast<uint64_t>(*value));
  }

  [[nodiscard]] LogicalResult analyzeTopLevelBody() {
    assert(syntax.body.size() == syntax.bodyIncludeContexts.size());
    for (const auto [id, includeContext] :
         llvm::zip_equal(syntax.body, syntax.bodyIncludeContexts)) {
      currentIncludeContext = includeContext;
      if (failed(analyzeStatement(syntax.statements[id], program.body,
                                  /*global=*/true))) {
        currentIncludeContext.reset();
        return failure();
      }
    }
    currentIncludeContext.reset();
    return success();
  }

  [[nodiscard]] LogicalResult analyzeBody(ArrayRef<SyntaxStatementId> source,
                                          std::vector<StatementId>& destination,
                                          const bool global) {
    for (const auto id : source) {
      if (failed(
              analyzeStatement(syntax.statements[id], destination, global))) {
        return failure();
      }
    }
    return success();
  }

  [[nodiscard]] LogicalResult
  analyzeStatement(const SyntaxStatement& statement,
                   std::vector<StatementId>& destination, const bool global) {
    return std::visit(
        [&](const auto& data) -> LogicalResult {
          using T = std::decay_t<decltype(data)>;
          if constexpr (!std::is_same_v<T, SyntaxGateCall> &&
                        !std::is_same_v<T, SyntaxFor> &&
                        !std::is_same_v<T, SyntaxWhile>) {
            if (insideGate) {
              return fail(
                  statement.location,
                  "gate bodies may contain only gate calls and loops over "
                  "gate calls");
            }
          }
          if constexpr (std::is_same_v<T, SyntaxStandardLibraryInclude>) {
            return activateStandardLibrary(statement.location, data.kind);
          } else if constexpr (std::is_same_v<T, SyntaxScalarDeclaration>) {
            return analyzeScalarDeclaration(statement.location, data,
                                            destination, global);
          } else if constexpr (std::is_same_v<T, SyntaxAssignment>) {
            return analyzeAssignment(statement.location, data, destination);
          } else if constexpr (std::is_same_v<T, SyntaxQubitDeclaration> ||
                               std::is_same_v<T, SyntaxBitDeclaration>) {
            return analyzeRegisterDeclaration(statement.location, data,
                                              destination, global);
          } else if constexpr (std::is_same_v<T, SyntaxMeasurement>) {
            MQT_OQ3_TRY_ASSIGN(analyzed,
                               analyzeMeasurement(statement.location, data));
            destination.push_back(analyzed);
            return success();
          } else if constexpr (std::is_same_v<T, SyntaxReset>) {
            MQT_OQ3_TRY_ASSIGN(analyzed,
                               analyzeReset(statement.location, data));
            destination.push_back(analyzed);
            return success();
          } else if constexpr (std::is_same_v<T, SyntaxBarrier>) {
            MQT_OQ3_TRY_ASSIGN(analyzed,
                               analyzeBarrier(statement.location, data));
            destination.push_back(analyzed);
            return success();
          } else if constexpr (std::is_same_v<T, SyntaxGateCall>) {
            MQT_OQ3_TRY_ASSIGN(applications, analyzeGateApplication(data));
            for (auto& application : applications) {
              MQT_OQ3_TRY_ASSIGN(
                  analyzed,
                  addStatement(statement.location, std::move(application)));
              destination.push_back(analyzed);
            }
            return success();
          } else if constexpr (std::is_same_v<T, SyntaxGateDefinition>) {
            if (!global) {
              return fail(statement.location,
                          "gate definitions are only allowed at global scope");
            }
            return analyzeGateDefinition(statement.location, data);
          } else if constexpr (std::is_same_v<T, SyntaxIf>) {
            MQT_OQ3_TRY_ASSIGN(analyzed, analyzeIf(statement.location, data));
            destination.push_back(analyzed);
            return success();
          } else if constexpr (std::is_same_v<T, SyntaxFor>) {
            MQT_OQ3_TRY_ASSIGN(analyzed, analyzeFor(statement.location, data));
            destination.push_back(analyzed);
            return success();
          } else if constexpr (std::is_same_v<T, SyntaxWhile>) {
            MQT_OQ3_TRY_ASSIGN(analyzed,
                               analyzeWhile(statement.location, data));
            destination.push_back(analyzed);
            return success();
          } else if constexpr (std::is_same_v<T, SyntaxSwitch>) {
            MQT_OQ3_TRY_ASSIGN(analyzed,
                               analyzeSwitch(statement.location, data));
            destination.push_back(analyzed);
            return success();
          }
          llvm_unreachable("unknown syntax statement kind");
        },
        statement.data);
  }

  [[nodiscard]] LogicalResult
  activateStandardLibrary(SMLoc location, const StandardLibraryKind kind) {
    auto& alreadyIncluded = kind == StandardLibraryKind::StdGates
                                ? program.stdGatesIncluded
                                : program.qelib1Included;
    if (alreadyIncluded) {
      return fail(location, kind == StandardLibraryKind::StdGates
                                ? "stdgates.inc is included more than once"
                                : "qelib1.inc is included more than once");
    }
    for (const auto& gate : getGateCatalog()) {
      const bool belongsToLibrary = kind == StandardLibraryKind::StdGates
                                        ? belongsToStdGates(gate.availability)
                                        : belongsToQELib1(gate.availability);
      if (!belongsToLibrary) {
        continue;
      }
      if (customGates.contains(gate.name) || lookup(gate.name) != nullptr) {
        return fail(location, "standard-library gate '" + gate.name +
                                  "' is already declared");
      }
    }
    alreadyIncluded = true;
    return success();
  }

  [[nodiscard]] LogicalResult analyzeScalarDeclaration(
      SMLoc location, const SyntaxScalarDeclaration& declaration,
      std::vector<StatementId>& destination, const bool global) {
    if (declaration.output && !global) {
      return fail(location, "outputs must be declared at global scope");
    }
    const auto type = scalarType(declaration.kind);
    if (type == ScalarType::Angle) {
      if (declaration.output) {
        return fail(location, "angle outputs are not supported");
      }
      if (!declaration.initializer) {
        return fail(location,
                    "angle declarations require a compile-time initializer");
      }
      if (!isConstantExpression(*declaration.initializer)) {
        return fail(location,
                    "angle declarations require a compile-time initializer");
      }
      MQT_OQ3_TRY_ASSIGN(width, angleWidth(declaration.size, location));
      MQT_OQ3_TRY_ASSIGN(initializer,
                         evaluateConstant(*declaration.initializer));
      MQT_OQ3_TRY_ASSIGN(constant,
                         convertToFixedAngle(initializer, width, location));
      return declare(location, declaration.identifier,
                     {.kind = SymbolKind::Constant,
                      .type = ScalarType::Angle,
                      .constant = constant});
    }
    if (declaration.isConst) {
      if (!declaration.initializer ||
          !isConstantExpression(*declaration.initializer)) {
        return fail(location,
                    "const declaration requires a constant initializer");
      }
      MQT_OQ3_TRY_ASSIGN(initializer,
                         evaluateConstant(*declaration.initializer));
      MQT_OQ3_TRY_ASSIGN(constant,
                         promoteConstInitializer(initializer, type, location));
      return declare(
          location, declaration.identifier,
          {.kind = SymbolKind::Constant, .type = type, .constant = constant});
    }

    const auto id = static_cast<ScalarId>(program.scalars.size());
    program.scalars.push_back({.type = type,
                               .name = declaration.identifier.str(),
                               .location = getSourceLocation(location)});
    initializedScalars.push_back(false);
    scalarGenerations.push_back(0);
    affineScalarValues.emplace_back();
    if (failed(declare(location, declaration.identifier,
                       {.kind = SymbolKind::Scalar, .type = type, .id = id}))) {
      return failure();
    }
    if (global) {
      const ProgramOutput output{.kind = OutputKind::Scalar, .symbol = id};
      implicitOutputs.push_back(output);
      if (declaration.output) {
        explicitOutputs.push_back(output);
      }
    }
    ScalarDeclarationStatement typed{.scalar = id};
    if (declaration.initializer) {
      if (type == ScalarType::Bool) {
        MQT_OQ3_TRY_ASSIGN(conditionInitializer,
                           analyzeBoolValue(*declaration.initializer));
        typed.conditionInitializer = conditionInitializer;
      } else {
        MQT_OQ3_TRY_ASSIGN(initializer,
                           analyzeExpression(*declaration.initializer));
        MQT_OQ3_TRY_ASSIGN(
            convertedInitializer,
            castExpression(
                initializer, type,
                syntax.expressions[*declaration.initializer].location));
        typed.initializer = convertedInitializer;
        if (buildAffineForm(convertedInitializer)) {
          affineScalarValues[id] =
              expandAffineScalarValues(convertedInitializer);
        }
      }
      initializedScalars[id] = true;
    }
    MQT_OQ3_TRY_ASSIGN(statement, addStatement(location, typed));
    destination.push_back(statement);
    return success();
  }

  void markBitInitialized(const frontend::BitReference& target) {
    ++bitGenerations[target.reg];
    if (!target.dynamicIndex) {
      mutableBitInitialization(target.reg)[target.index] = true;
      return;
    }
    DynamicBitFact fact{.expression = *target.dynamicIndex};
    collectDependencies(*target.dynamicIndex, fact.dependencies);
    auto& facts = mutableDynamicBitFacts(target.reg);
    if (llvm::none_of(facts, [&](const auto& existing) {
          return existing.dependencies == fact.dependencies &&
                 sameExpression(existing.expression, fact.expression);
        })) {
      facts.push_back(std::move(fact));
    }
  }

  [[nodiscard]] LogicalResult
  analyzeAssignment(SMLoc location, const SyntaxAssignment& assignment,
                    std::vector<StatementId>& destination) {
    const auto* symbol = lookup(assignment.target.identifier);
    if (symbol != nullptr && symbol->kind == SymbolKind::Scalar) {
      if (assignment.target.index) {
        return fail(location, "scalar assignments cannot have an index");
      }
      ScalarAssignmentStatement typed{.scalar = symbol->id};
      if (symbol->type == ScalarType::Bool) {
        MQT_OQ3_TRY_ASSIGN(condition, analyzeBoolValue(assignment.value));
        typed.condition = condition;
        affineScalarValues[symbol->id].reset();
      } else {
        MQT_OQ3_TRY_ASSIGN(value, analyzeExpression(assignment.value));
        MQT_OQ3_TRY_ASSIGN(
            convertedValue,
            castExpression(value, symbol->type,
                           syntax.expressions[assignment.value].location));
        typed.value = convertedValue;
        affineScalarValues[symbol->id] =
            buildAffineForm(convertedValue)
                ? expandAffineScalarValues(convertedValue)
                : std::nullopt;
      }
      initializedScalars[symbol->id] = true;
      ++scalarGenerations[symbol->id];
      MQT_OQ3_TRY_ASSIGN(statement, addStatement(location, typed));
      destination.push_back(statement);
      return success();
    }
    if (symbol == nullptr || symbol->kind != SymbolKind::Register ||
        program.registers[symbol->id].kind != RegisterKind::Bit) {
      return fail(location,
                  "cannot assign to '" + assignment.target.identifier + "'");
    }
    const auto targetReg = static_cast<RegisterId>(symbol->id);
    const auto& value = syntax.expressions[assignment.value];
    const auto* valueSymbol = value.kind == Expr::Kind::Identifier
                                  ? lookup(value.identifier)
                                  : nullptr;
    const bool bitVectorValue =
        value.kind == Expr::Kind::RotateLeft ||
        value.kind == Expr::Kind::RotateRight ||
        (valueSymbol != nullptr && valueSymbol->kind == SymbolKind::Register &&
         program.registers[valueSymbol->id].kind == RegisterKind::Bit &&
         !program.registers[valueSymbol->id].isScalar);
    if (!assignment.target.index && bitVectorValue) {
      MQT_OQ3_TRY_ASSIGN(bitVector,
                         analyzeBitVectorExpression(assignment.value));
      if (program.bitVectorExpressions[bitVector].width !=
          program.registers[targetReg].width) {
        return fail(location, "bit-register assignment widths must match");
      }
      for (uint64_t bit = 0; bit < program.registers[targetReg].width; ++bit) {
        markBitInitialized({.reg = targetReg, .index = bit});
      }
      MQT_OQ3_TRY_ASSIGN(
          statement,
          addStatement(location, BitVectorAssignmentStatement{
                                     .target = targetReg, .value = bitVector}));
      destination.push_back(statement);
      return success();
    }
    MQT_OQ3_TRY_ASSIGN(targets, resolveBits(assignment.target));
    if (targets.size() > 1) {
      return fail(
          location,
          "whole-register bit assignment requires a bit-register value");
    }
    MQT_OQ3_TRY_ASSIGN(condition, analyzeBoolValue(assignment.value));
    markBitInitialized(targets.front());
    MQT_OQ3_TRY_ASSIGN(
        statement,
        addStatement(location, BitAssignmentStatement{.target = targets.front(),
                                                      .value = condition}));
    destination.push_back(statement);
    return success();
  }

  [[nodiscard]] LogicalResult analyzeRegisterDeclaration(
      SMLoc location, StringRef identifier,
      const std::optional<SyntaxExpressionId> size,
      const std::optional<SyntaxExpressionId> initializer, const bool output,
      const bool isQubit, std::vector<StatementId>& destination,
      const bool global) {
    if (isQubit && !global) {
      return fail(location, "qubits must be declared at global scope");
    }
    if (!isQubit && output && !global) {
      return fail(location, "outputs must be declared at global scope");
    }
    MQT_OQ3_TRY_ASSIGN(width, constantWidth(size, location));
    const auto id = static_cast<RegisterId>(program.registers.size());
    if (width > TOTAL_REGISTER_ELEMENT_LIMIT - totalRegisterElements) {
      return fail(location,
                  Twine("total register elements exceed the limit of ") +
                      Twine(TOTAL_REGISTER_ELEMENT_LIMIT));
    }
    totalRegisterElements += width;
    program.registers.push_back(
        {.kind = isQubit ? RegisterKind::Qubit : RegisterKind::Bit,
         .name = identifier.str(),
         .width = width,
         .isScalar = !size.has_value(),
         .location = getSourceLocation(location)});
    // OpenQASM 2 classical bits are zero-initialized; OpenQASM 3 bits are not.
    const bool initiallyInitialized = !isQubit && program.openQASM2;
    initializedBits.push_back(
        std::make_shared<BitInitialization>(width, initiallyInitialized));
    dynamicBitFacts.push_back(std::make_shared<DynamicBitFactSet>());
    bitGenerations.push_back(0);
    if (failed(declare(location, identifier,
                       {.kind = SymbolKind::Register, .id = id}))) {
      return failure();
    }
    if (!isQubit && global) {
      const ProgramOutput programOutput{.kind = OutputKind::BitRegister,
                                        .symbol = id};
      implicitOutputs.push_back(programOutput);
      if (output || program.openQASM2) {
        explicitOutputs.push_back(programOutput);
      }
    }
    MQT_OQ3_TRY_ASSIGN(statement,
                       addStatement(location, DeclarationStatement{.reg = id}));
    destination.push_back(statement);
    if (!isQubit && initializer) {
      if (width != 1) {
        return fail(
            location,
            "bit expression initializers require a scalar bit declaration");
      }
      return analyzeAssignment(
          location,
          SyntaxAssignment{.target =
                               SyntaxBitReference{.location = location,
                                                  .identifier = identifier},
                           .value = *initializer},
          destination);
    }
    return success();
  }

  [[nodiscard]] LogicalResult analyzeRegisterDeclaration(
      SMLoc location, const SyntaxQubitDeclaration& declaration,
      std::vector<StatementId>& destination, const bool global) {
    return analyzeRegisterDeclaration(location, declaration.identifier,
                                      declaration.size, std::nullopt, false,
                                      true, destination, global);
  }

  [[nodiscard]] LogicalResult analyzeRegisterDeclaration(
      SMLoc location, const SyntaxBitDeclaration& declaration,
      std::vector<StatementId>& destination, const bool global) {
    return analyzeRegisterDeclaration(location, declaration.identifier,
                                      declaration.size, declaration.initializer,
                                      declaration.output, false, destination,
                                      global);
  }

  [[nodiscard]] LogicalResult
  analyzeGateDefinition(SMLoc location,
                        const SyntaxGateDefinition& declaration) {
    if (customGates.contains(declaration.identifier) ||
        lookup(declaration.identifier) != nullptr) {
      return fail(location,
                  "gate '" + declaration.identifier + "' is already declared");
    }
    if (const auto* catalog = lookupGate(declaration.identifier);
        catalog != nullptr && isGateAvailable(*catalog)) {
      // OpenQASM 2 programs often repeat standard library gate definitions
      // (e.g. `gate sx`). Prefer the standard-library entry and skip the
      // duplicate body to avoid later inlining overhead.
      if (program.openQASM2) {
        return success();
      }
      // MQT-compatible OpenQASM 3 may carry portable definitions for gates
      // that the compatibility catalog lowers directly. Prefer the native
      // lowering when the declaration has the catalog signature. Strict mode
      // does not make compatibility entries available and analyzes the body as
      // an ordinary custom gate.
      if (catalog->availability == GateAvailability::Compatibility) {
        if (declaration.parameters.size() != catalog->parameterCount ||
            declaration.qubits.size() != catalog->qubitCount()) {
          return fail(location,
                      "gate '" + declaration.identifier +
                          "' does not match its compatibility signature");
        }
        return success();
      }
      // OpenQASM 3 rejects shadowing language and standard-library gates.
      return fail(location,
                  "gate '" + declaration.identifier + "' is already declared");
    }
    customGates[declaration.identifier] = {
        .parameterCount = declaration.parameters.size(),
        .qubitCount = declaration.qubits.size()};
    GateDefinition definition{.name = declaration.identifier.str(),
                              .parameterCount = declaration.parameters.size(),
                              .qubitCount = declaration.qubits.size(),
                              .location = getSourceLocation(location)};
    scopes.emplace_back();
    for (const auto [index, parameter] :
         llvm::enumerate(declaration.parameters)) {
      if (failed(declare(location, parameter,
                         {.kind = SymbolKind::GateParameter,
                          .type = ScalarType::Angle,
                          .id = static_cast<uint32_t>(index)}))) {
        scopes.pop_back();
        return failure();
      }
    }
    for (const auto [index, qubit] : llvm::enumerate(declaration.qubits)) {
      if (failed(declare(location, qubit,
                         {.kind = SymbolKind::GateQubit,
                          .id = static_cast<uint32_t>(index)}))) {
        scopes.pop_back();
        return failure();
      }
    }
    insideGate = true;
    const auto bodyResult =
        analyzeBody(declaration.body, definition.body, /*global=*/false);
    insideGate = false;
    scopes.pop_back();
    if (failed(bodyResult)) {
      return failure();
    }
    program.gates.push_back(std::move(definition));
    return success();
  }

  [[nodiscard]] FailureOr<StatementId>
  analyzeMeasurement(SMLoc location, const SyntaxMeasurement& measurement) {
    MQT_OQ3_TRY_ASSIGN(qubits, resolveQubitOperand(measurement.source));
    if (!measurement.target) {
      return addStatement(location,
                          MeasurementStatement{.qubits = std::move(qubits)});
    }
    const auto* destination = lookup(measurement.target->identifier);
    if (destination != nullptr && destination->kind == SymbolKind::Scalar) {
      if (destination->type == ScalarType::Bool) {
        return fail(
            location,
            "measurement results have type 'bit' and cannot be assigned to "
            "'bool' without an explicit cast");
      }
      return fail(location,
                  "measurement assignment requires a bit-register destination");
    }
    MQT_OQ3_TRY_ASSIGN(targets, resolveBits(*measurement.target));
    if (targets.size() != qubits.size()) {
      return fail(
          location,
          "measurement target and qubit operand must have the same width");
    }
    for (const auto& target : targets) {
      markBitInitialized(target);
    }
    return addStatement(location,
                        MeasurementStatement{.targets = std::move(targets),
                                             .qubits = std::move(qubits)});
  }

  [[nodiscard]] FailureOr<StatementId> analyzeReset(SMLoc location,
                                                    const SyntaxReset& reset) {
    MQT_OQ3_TRY_ASSIGN(qubits, resolveQubitOperand(reset.operand));
    return addStatement(location, ResetStatement{.qubits = std::move(qubits)});
  }

  [[nodiscard]] FailureOr<StatementId>
  analyzeBarrier(SMLoc location, const SyntaxBarrier& barrier) {
    std::vector<QubitReference> qubits;
    if (barrier.operands.empty()) {
      for (const auto [registerId, declaration] :
           llvm::enumerate(program.registers)) {
        if (declaration.kind != RegisterKind::Qubit) {
          continue;
        }
        for (uint64_t index = 0; index < declaration.width; ++index) {
          qubits.push_back({.kind = QubitReferenceKind::Register,
                            .symbol = static_cast<RegisterId>(registerId),
                            .index = index});
        }
      }
      for (const auto index : hardwareQubits) {
        qubits.push_back(
            {.kind = QubitReferenceKind::Hardware, .index = index});
      }
    }
    for (const auto& operand : barrier.operands) {
      MQT_OQ3_TRY_ASSIGN(selection, resolveQubitOperand(operand));
      qubits.insert(qubits.end(), selection.begin(), selection.end());
    }

    if (barrier.operands.size() > 1) {
      llvm::DenseSet<std::pair<RegisterId, uint64_t>> staticRegisterQubits;
      llvm::DenseSet<std::pair<RegisterId, ExpressionId>> provenRegisterQubits;
      llvm::DenseSet<uint32_t> gateArguments;
      llvm::DenseSet<uint64_t> hardwareQubitOperands;
      llvm::DenseMap<RegisterId, size_t> staticRegisterQubitCounts;

      for (const auto& qubit : qubits) {
        bool inserted = false;
        switch (qubit.kind) {
        case QubitReferenceKind::Register:
          if (qubit.provenIndex) {
            inserted =
                provenRegisterQubits.insert({qubit.symbol, *qubit.provenIndex})
                    .second;
          } else {
            inserted =
                staticRegisterQubits.insert({qubit.symbol, qubit.index}).second;
            if (inserted) {
              ++staticRegisterQubitCounts[qubit.symbol];
            }
          }
          break;
        case QubitReferenceKind::GateArgument:
          inserted = gateArguments.insert(qubit.symbol).second;
          break;
        case QubitReferenceKind::Hardware:
          inserted = hardwareQubitOperands.insert(qubit.index).second;
          break;
        }
        if (!inserted) {
          return fail(
              location,
              "barrier operands must not reference the same qubit more than "
              "once");
        }
      }

      for (const auto& provenRegisterQubit : provenRegisterQubits) {
        const auto reg = provenRegisterQubit.first;
        if (staticRegisterQubitCounts.lookup(reg) ==
            program.registers.at(reg).width) {
          return fail(
              location,
              "barrier operands must not reference the same qubit more than "
              "once");
        }
      }

      if (!provenRegisterQubits.empty()) {
        size_t affineComparisons = 0;
        for (const auto [position, qubit] : llvm::enumerate(qubits)) {
          for (const auto& previous : ArrayRef(qubits).take_front(position)) {
            if (qubit.kind != QubitReferenceKind::Register ||
                previous.kind != QubitReferenceKind::Register ||
                qubit.symbol != previous.symbol ||
                (!qubit.provenIndex && !previous.provenIndex)) {
              continue;
            }
            if (!proveDistinct(previous, qubit, affineComparisons)) {
              return fail(
                  location,
                  "cannot prove that barrier operands reference distinct "
                  "qubits");
            }
          }
        }
      }
    }

    return addStatement(location,
                        BarrierStatement{.qubits = std::move(qubits)});
  }

  [[nodiscard]] FailureOr<StatementId> analyzeIf(SMLoc location,
                                                 const SyntaxIf& conditional) {
    MQT_OQ3_TRY_ASSIGN(condition, analyzeCondition(conditional.condition));
    IfStatement result{.condition = condition};
    const auto beforeBitsInitialized = initializedBits;
    const auto beforeInitialized = initializedScalars;
    const auto beforeGenerations = scalarGenerations;
    const auto beforeAffineScalarValues = affineScalarValues;
    const auto beforeBitGenerations = bitGenerations;
    const auto beforeDynamicBitFacts = dynamicBitFacts;
    scopes.emplace_back();
    const auto thenResult =
        analyzeBody(conditional.thenStatements, result.thenStatements,
                    /*global=*/false);
    if (failed(thenResult)) {
      scopes.pop_back();
      return failure();
    }
    const auto afterThenBitsInitialized = initializedBits;
    const auto afterThenInitialized = initializedScalars;
    const auto afterThenGenerations = scalarGenerations;
    const auto afterThenAffineScalarValues = affineScalarValues;
    const auto afterThenBitGenerations = bitGenerations;
    const auto afterThenDynamicBitFacts = dynamicBitFacts;
    scopes.pop_back();

    restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                       beforeGenerations, beforeBitGenerations);
    restoreAffineScalarValuesPrefix(beforeAffineScalarValues);
    restoreDynamicFactsPrefix(beforeDynamicBitFacts);
    scopes.emplace_back();
    const auto elseResult =
        analyzeBody(conditional.elseStatements, result.elseStatements,
                    /*global=*/false);
    if (failed(elseResult)) {
      scopes.pop_back();
      return failure();
    }
    const auto afterElseBitsInitialized = initializedBits;
    const auto afterElseInitialized = initializedScalars;
    const auto afterElseGenerations = scalarGenerations;
    const auto afterElseAffineScalarValues = affineScalarValues;
    const auto afterElseBitGenerations = bitGenerations;
    const auto afterElseDynamicBitFacts = dynamicBitFacts;
    scopes.pop_back();

    MQT_OQ3_TRY_ASSIGN(knownCondition,
                       constantCondition(conditional.condition));
    if (knownCondition) {
      const auto& knownBitsInitialized =
          *knownCondition ? afterThenBitsInitialized : afterElseBitsInitialized;
      const auto& knownInitialized =
          *knownCondition ? afterThenInitialized : afterElseInitialized;
      const auto& knownGenerations =
          *knownCondition ? afterThenGenerations : afterElseGenerations;
      const auto& knownAffineScalarValues = *knownCondition
                                                ? afterThenAffineScalarValues
                                                : afterElseAffineScalarValues;
      const auto& knownBitGenerations =
          *knownCondition ? afterThenBitGenerations : afterElseBitGenerations;
      const auto& knownDynamicBitFacts =
          *knownCondition ? afterThenDynamicBitFacts : afterElseDynamicBitFacts;
      restoreStatePrefix(knownBitsInitialized, knownInitialized,
                         knownGenerations, knownBitGenerations);
      restoreAffineScalarValuesPrefix(knownAffineScalarValues);
      restoreDynamicFactsPrefix(knownDynamicBitFacts);
      return addStatement(location, std::move(result));
    }

    restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                       beforeGenerations, beforeBitGenerations);
    restoreAffineScalarValuesPrefix(beforeAffineScalarValues);
    restoreDynamicFactsPrefix(beforeDynamicBitFacts);
    for (size_t reg = 0; reg < beforeBitsInitialized.size(); ++reg) {
      auto& merged = mutableBitInitialization(static_cast<RegisterId>(reg));
      for (size_t bit = 0; bit < beforeBitsInitialized[reg]->size(); ++bit) {
        merged[bit] = (*afterThenBitsInitialized[reg])[bit] &&
                      (*afterElseBitsInitialized[reg])[bit];
      }
    }
    for (size_t reg = 0; reg < beforeDynamicBitFacts.size(); ++reg) {
      auto& merged = mutableDynamicBitFacts(static_cast<RegisterId>(reg));
      merged.clear();
      for (const auto& thenFact : *afterThenDynamicBitFacts[reg]) {
        if (llvm::any_of(
                *afterElseDynamicBitFacts[reg], [&](const auto& elseFact) {
                  return thenFact.dependencies == elseFact.dependencies &&
                         sameExpression(thenFact.expression,
                                        elseFact.expression);
                })) {
          merged.push_back(thenFact);
        }
      }
    }
    for (size_t scalar = 0; scalar < beforeInitialized.size(); ++scalar) {
      initializedScalars[scalar] =
          afterThenInitialized[scalar] && afterElseInitialized[scalar];
      scalarGenerations[scalar] =
          std::max(afterThenGenerations[scalar], afterElseGenerations[scalar]);
      if (afterThenAffineScalarValues[scalar] &&
          afterElseAffineScalarValues[scalar] &&
          sameAffineValue(*afterThenAffineScalarValues[scalar],
                          *afterElseAffineScalarValues[scalar])) {
        affineScalarValues[scalar] = afterThenAffineScalarValues[scalar];
      } else {
        affineScalarValues[scalar].reset();
      }
    }
    for (size_t reg = 0; reg < beforeBitGenerations.size(); ++reg) {
      bitGenerations[reg] =
          std::max(afterThenBitGenerations[reg], afterElseBitGenerations[reg]);
    }
    return addStatement(location, std::move(result));
  }

  [[nodiscard]] FailureOr<StatementId> analyzeFor(SMLoc location,
                                                  const SyntaxFor& loop) {
    MQT_OQ3_TRY_ASSIGN(start, analyzeExpression(loop.start));
    MQT_OQ3_TRY_ASSIGN(step, analyzeExpression(loop.step));
    MQT_OQ3_TRY_ASSIGN(stop, analyzeExpression(loop.stop));
    ForStatement result{.start = start, .step = step, .stop = stop};
    for (const auto expression : {result.start, result.step, result.stop}) {
      if (!isInteger(program.expressions[expression].type)) {
        return fail(location, "for-loop ranges require integer expressions");
      }
    }
    const auto startForm = buildAffineForm(result.start);
    const auto stepForm = buildAffineForm(result.step);
    const auto stopForm = buildAffineForm(result.stop);
    if (stepForm && isAffineConstant(*stepForm) && stepForm->constant == 0) {
      return fail(location, "for-loop range step must not be zero");
    }
    const bool unsignedEndpoints =
        program.expressions[result.start].type == ScalarType::Uint ||
        program.expressions[result.stop].type == ScalarType::Uint;
    const bool endpointValuesPreserved =
        !unsignedEndpoints ||
        (startForm && stopForm &&
         provesLowerBound(*startForm, llvm::DynamicAPInt(0)) &&
         provesLowerBound(*stopForm, llvm::DynamicAPInt(0)));
    std::optional<llvm::DynamicAPInt> positiveStep;
    if (stepForm && isAffineConstant(*stepForm) && stepForm->constant > 0) {
      positiveStep = stepForm->constant;
    }
    result.provenPositiveRange =
        startForm && stopForm && endpointValuesPreserved && positiveStep &&
        *positiveStep <= signedMaximum() &&
        provesUpperBound(*stopForm, signedMaximum() - 1);
    if (result.provenPositiveRange) {
      const auto expandedStart = expandAffineScalarValues(result.start);
      const auto expandedStep = expandAffineScalarValues(result.step);
      const auto expandedStop = expandAffineScalarValues(result.stop);
      assert(expandedStart && expandedStep && expandedStop &&
             "proven affine ranges must have expandable expressions");
      result.start = *expandedStart;
      result.step = *expandedStep;
      result.stop = *expandedStop;
    }

    const auto beforeBitsInitialized = initializedBits;
    const auto beforeInitialized = initializedScalars;
    const auto beforeGenerations = scalarGenerations;
    const auto beforeAffineScalarValues = affineScalarValues;
    const auto beforeBitGenerations = bitGenerations;
    const auto beforeDynamicBitFacts = dynamicBitFacts;
    scopes.emplace_back();
    const auto scalar = static_cast<ScalarId>(program.scalars.size());
    const auto type = loop.isUnsigned ? ScalarType::Uint : ScalarType::Int;
    program.scalars.push_back(
        {.type = type, .name = loop.inductionVariable.str()});
    initializedScalars.push_back(true);
    scalarGenerations.push_back(0);
    affineScalarValues.emplace_back();
    if (failed(declare(location, loop.inductionVariable,
                       {.kind = insideGate ? SymbolKind::GateLocalScalar
                                           : SymbolKind::Scalar,
                        .type = type,
                        .id = scalar}))) {
      scopes.pop_back();
      return failure();
    }
    result.inductionVariable = scalar;
    const auto outerDomain = affineDomain;
    const auto outerLoopVariantScalars = loopVariantScalars;
    llvm::DenseSet<StringRef> blockLocalScalars;
    collectLoopMutations(loop.body, loopVariantScalars, blockLocalScalars);
    if (result.provenPositiveRange) {
      affineDomain.appendVar(presburger::VarKind::SetDim);
      auto lower = *startForm;
      auto upper = *stopForm;
      lower.coefficients.push_back(llvm::DynamicAPInt(0));
      upper.coefficients.push_back(llvm::DynamicAPInt(0));
      auto induction = constantAffineForm(llvm::DynamicAPInt(0));
      induction.coefficients.back() = llvm::DynamicAPInt(1);
      affineDomain.addInequality(
          affineConstraint(addAffineForms(induction, negateAffineForm(lower))));
      affineDomain.addInequality(
          affineConstraint(addAffineForms(upper, negateAffineForm(induction))));
      activeInductions.insert({scalar, affineDomain.getNumDimVars() - 1});
    }
    const auto bodyResult =
        analyzeBody(loop.body, result.body, /*global=*/false);
    if (failed(bodyResult)) {
      activeInductions.erase(scalar);
      affineDomain = outerDomain;
      loopVariantScalars = outerLoopVariantScalars;
      scopes.pop_back();
      return failure();
    }
    const auto afterBodyBitsInitialized = initializedBits;
    const auto afterBodyInitialized = initializedScalars;
    const auto afterBodyGenerations = scalarGenerations;
    const auto afterBodyBitGenerations = bitGenerations;
    const auto afterBodyDynamicBitFacts = dynamicBitFacts;
    activeInductions.erase(scalar);
    affineDomain = outerDomain;
    loopVariantScalars = outerLoopVariantScalars;
    scopes.pop_back();
    restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                       beforeGenerations, beforeBitGenerations);
    restoreAffineScalarValuesPrefix(beforeAffineScalarValues);
    restoreDynamicFactsPrefix(beforeDynamicBitFacts);
    bool rangeMayExecute = true;
    if (isConstantExpression(loop.start) && isConstantExpression(loop.step) &&
        isConstantExpression(loop.stop)) {
      MQT_OQ3_TRY_ASSIGN(startConstant, evaluateConstant(loop.start));
      MQT_OQ3_TRY_ASSIGN(stepConstant, evaluateConstant(loop.step));
      MQT_OQ3_TRY_ASSIGN(stopConstant, evaluateConstant(loop.stop));
      const bool unsignedEndpoints = startConstant.type == ScalarType::Uint ||
                                     stopConstant.type == ScalarType::Uint;
      const auto compareRangeValues = [&](const Constant& lhs,
                                          const Constant& rhs) {
        if (!unsignedEndpoints) {
          const auto left = std::get<int64_t>(lhs.value);
          const auto right = std::get<int64_t>(rhs.value);
          if (left < right) {
            return -1;
          }
          return left > right ? 1 : 0;
        }
        const auto asUnsigned = [](const Constant& value) {
          if (value.type == ScalarType::Uint) {
            return std::get<uint64_t>(value.value);
          }
          return static_cast<uint64_t>(std::get<int64_t>(value.value));
        };
        const auto left = asUnsigned(lhs);
        const auto right = asUnsigned(rhs);
        if (left < right) {
          return -1;
        }
        return left > right ? 1 : 0;
      };
      const bool positiveStep = stepConstant.type == ScalarType::Uint ||
                                std::get<int64_t>(stepConstant.value) > 0;
      const auto endpointOrder =
          compareRangeValues(startConstant, stopConstant);
      const bool nonempty =
          positiveStep ? endpointOrder <= 0 : endpointOrder >= 0;
      rangeMayExecute = nonempty;
      if (nonempty) {
        for (size_t reg = 0; reg < beforeBitsInitialized.size(); ++reg) {
          initializedBits[reg] = afterBodyBitsInitialized[reg];
        }
        for (size_t scalar = 0; scalar < beforeInitialized.size(); ++scalar) {
          initializedScalars[scalar] = afterBodyInitialized[scalar];
          scalarGenerations[scalar] = afterBodyGenerations[scalar];
        }
        for (size_t reg = 0; reg < beforeBitGenerations.size(); ++reg) {
          bitGenerations[reg] = afterBodyBitGenerations[reg];
        }
        for (size_t reg = 0; reg < beforeDynamicBitFacts.size(); ++reg) {
          dynamicBitFacts[reg] = afterBodyDynamicBitFacts[reg];
        }
      }
    }
    if (rangeMayExecute) {
      for (size_t scalar = 0; scalar < beforeGenerations.size(); ++scalar) {
        if (afterBodyGenerations[scalar] != beforeGenerations[scalar]) {
          affineScalarValues[scalar].reset();
        }
      }
    }
    return addStatement(location, std::move(result));
  }

  [[nodiscard]] FailureOr<StatementId> analyzeWhile(SMLoc location,
                                                    const SyntaxWhile& loop) {
    MQT_OQ3_TRY_ASSIGN(condition, analyzeCondition(loop.condition));
    WhileStatement result{.condition = condition};
    const auto beforeBitsInitialized = initializedBits;
    const auto beforeInitialized = initializedScalars;
    const auto beforeGenerations = scalarGenerations;
    const auto beforeAffineScalarValues = affineScalarValues;
    const auto beforeBitGenerations = bitGenerations;
    const auto beforeDynamicBitFacts = dynamicBitFacts;
    scopes.emplace_back();
    const auto outerLoopVariantScalars = loopVariantScalars;
    llvm::DenseSet<StringRef> blockLocalScalars;
    collectLoopMutations(loop.body, loopVariantScalars, blockLocalScalars);
    const auto bodyResult =
        analyzeBody(loop.body, result.body, /*global=*/false);
    loopVariantScalars = outerLoopVariantScalars;
    scopes.pop_back();
    if (failed(bodyResult)) {
      return failure();
    }
    const auto afterBodyGenerations = scalarGenerations;
    const auto afterBodyBitGenerations = bitGenerations;
    restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                       beforeGenerations, beforeBitGenerations);
    restoreAffineScalarValuesPrefix(beforeAffineScalarValues);
    restoreDynamicFactsPrefix(beforeDynamicBitFacts);
    for (size_t scalar = 0; scalar < beforeGenerations.size(); ++scalar) {
      scalarGenerations[scalar] =
          std::max(beforeGenerations[scalar], afterBodyGenerations[scalar]);
      if (afterBodyGenerations[scalar] != beforeGenerations[scalar]) {
        affineScalarValues[scalar].reset();
      }
    }
    for (size_t reg = 0; reg < beforeBitGenerations.size(); ++reg) {
      bitGenerations[reg] =
          std::max(beforeBitGenerations[reg], afterBodyBitGenerations[reg]);
    }
    return addStatement(location, std::move(result));
  }

  [[nodiscard]] FailureOr<StatementId>
  analyzeSwitch(SMLoc location, const SyntaxSwitch& switchSyntax) {
    MQT_OQ3_TRY_ASSIGN(control, analyzeExpression(switchSyntax.control));
    SwitchStatement result{.control = control};
    if (!isInteger(program.expressions[result.control].type)) {
      return fail(location, "switch control expression must have integer type");
    }

    std::set<int64_t> labels;
    const auto beforeBitsInitialized = initializedBits;
    const auto beforeInitialized = initializedScalars;
    const auto beforeGenerations = scalarGenerations;
    const auto beforeAffineScalarValues = affineScalarValues;
    const auto beforeBitGenerations = bitGenerations;
    const auto beforeDynamicBitFacts = dynamicBitFacts;
    auto mergedScalarGenerations = beforeGenerations;
    auto mergedBitGenerations = beforeBitGenerations;
    std::vector<std::vector<std::shared_ptr<BitInitialization>>>
        branchBitsInitialized;
    std::vector<std::vector<bool>> branchScalarsInitialized;
    std::vector<std::vector<std::optional<ExpressionId>>>
        branchAffineScalarValues;
    const auto analyzeBranch =
        [&](const ArrayRef<SyntaxStatementId> syntaxStatements,
            std::vector<StatementId>& statements) -> LogicalResult {
      restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                         beforeGenerations, beforeBitGenerations);
      restoreAffineScalarValuesPrefix(beforeAffineScalarValues);
      restoreDynamicFactsPrefix(beforeDynamicBitFacts);
      scopes.emplace_back();
      const auto branchResult =
          analyzeBody(syntaxStatements, statements, /*global=*/false);
      scopes.pop_back();
      if (failed(branchResult)) {
        return failure();
      }
      branchBitsInitialized.push_back(initializedBits);
      branchScalarsInitialized.push_back(initializedScalars);
      branchAffineScalarValues.push_back(affineScalarValues);
      for (size_t index = 0; index < beforeGenerations.size(); ++index) {
        mergedScalarGenerations[index] =
            std::max(mergedScalarGenerations[index], scalarGenerations[index]);
      }
      for (size_t index = 0; index < beforeBitGenerations.size(); ++index) {
        mergedBitGenerations[index] =
            std::max(mergedBitGenerations[index], bitGenerations[index]);
      }
      return success();
    };

    result.cases.reserve(switchSyntax.cases.size());
    for (const auto& syntaxCase : switchSyntax.cases) {
      SwitchCase switchCase;
      switchCase.labels.reserve(syntaxCase.labels.size());
      for (const auto labelExpression : syntaxCase.labels) {
        if (!isConstantExpression(labelExpression)) {
          return fail(
              syntax.expressions[labelExpression].location,
              "switch case labels must be constant integer expressions");
        }
        MQT_OQ3_TRY_ASSIGN(label, evaluateConstant(labelExpression));
        if (!isInteger(label.type)) {
          return fail(syntax.expressions[labelExpression].location,
                      "switch case labels must have integer type");
        }
        if (label.type == ScalarType::Uint &&
            std::get<uint64_t>(label.value) >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          return fail(syntax.expressions[labelExpression].location,
                      "switch case label does not fit in signed i64");
        }
        const auto value = asSigned(label);
        assert(value.has_value() && "label range checked above");
        if (!labels.insert(*value).second) {
          return fail(syntax.expressions[labelExpression].location,
                      "duplicate switch case label");
        }
        switchCase.labels.push_back(*value);
      }
      if (failed(analyzeBranch(syntaxCase.body, switchCase.body))) {
        return failure();
      }
      result.cases.push_back(std::move(switchCase));
    }
    if (failed(analyzeBranch(switchSyntax.defaultStatements,
                             result.defaultStatements))) {
      return failure();
    }

    restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                       mergedScalarGenerations, mergedBitGenerations);
    restoreAffineScalarValuesPrefix(beforeAffineScalarValues);
    restoreDynamicFactsPrefix(beforeDynamicBitFacts);
    for (size_t reg = 0; reg < beforeBitsInitialized.size(); ++reg) {
      auto& initialized =
          mutableBitInitialization(static_cast<RegisterId>(reg));
      for (size_t bit = 0; bit < initialized.size(); ++bit) {
        initialized[bit] =
            llvm::all_of(branchBitsInitialized, [&](const auto& branch) {
              return (*branch[reg])[bit];
            });
      }
    }
    for (size_t scalar = 0; scalar < beforeInitialized.size(); ++scalar) {
      initializedScalars[scalar] =
          llvm::all_of(branchScalarsInitialized,
                       [&](const auto& branch) { return branch[scalar]; });
      const auto& first = branchAffineScalarValues.front()[scalar];
      if (first &&
          llvm::all_of(branchAffineScalarValues, [&](const auto& branch) {
            return branch[scalar] && sameAffineValue(*first, *branch[scalar]);
          })) {
        affineScalarValues[scalar] = first;
      } else {
        affineScalarValues[scalar].reset();
      }
    }
    return addStatement(location, std::move(result));
  }

  [[nodiscard]] FailureOr<ConditionId>
  analyzeCondition(const SyntaxExpressionId syntaxId) {
    const auto& condition = syntax.expressions[syntaxId];
    ConditionExpression typed{.location =
                                  getSourceLocation(condition.location)};
    if (isConstantExpression(syntaxId)) {
      MQT_OQ3_TRY_ASSIGN(constant, evaluateConstant(syntaxId));
      if (constant.type != ScalarType::Bool) {
        return fail(condition.location, "condition must have bool type");
      }
      typed.kind = ConditionKind::Literal;
      typed.literal = std::get<bool>(constant.value);
      return addCondition(std::move(typed));
    }
    switch (condition.kind) {
    case Expr::Kind::Identifier: {
      const auto* symbol = lookup(condition.identifier);
      if (symbol == nullptr) {
        return fail(condition.location, "unknown condition identifier '" +
                                            condition.identifier + "'");
      }
      if (symbol->kind == SymbolKind::Scalar &&
          symbol->type == ScalarType::Bool) {
        if (!initializedScalars.at(symbol->id)) {
          return fail(condition.location,
                      "scalar '" + condition.identifier + "' is uninitialized");
        }
        typed.kind = ConditionKind::Scalar;
        typed.scalar = symbol->id;
        break;
      }
      if (symbol->kind != SymbolKind::Register ||
          program.registers[symbol->id].kind != RegisterKind::Bit) {
        return fail(condition.location, "identifier '" + condition.identifier +
                                            "' is not bool or a classical bit");
      }
      MQT_OQ3_TRY_ASSIGN(bits,
                         resolveBits({.location = condition.location,
                                      .identifier = condition.identifier}));
      if (bits.size() != 1) {
        return fail(condition.location,
                    "condition must select exactly one classical bit");
      }
      if (failed(ensureBitInitialized(bits.front(), condition.location))) {
        return failure();
      }
      typed.kind = ConditionKind::Bit;
      typed.bit = bits.front();
      break;
    }
    case Expr::Kind::Index: {
      MQT_OQ3_TRY_ASSIGN(bits, resolveBits({.location = condition.location,
                                            .identifier = condition.identifier,
                                            .index = condition.lhs}));
      if (bits.size() != 1) {
        return fail(condition.location,
                    "condition must select exactly one classical bit");
      }
      if (failed(ensureBitInitialized(bits.front(), condition.location))) {
        return failure();
      }
      typed.kind = ConditionKind::Bit;
      typed.bit = bits.front();
      break;
    }
    case Expr::Kind::Not: {
      typed.kind = ConditionKind::Not;
      MQT_OQ3_TRY_ASSIGN(lhs, analyzeCondition(*condition.lhs));
      typed.lhs = lhs;
      break;
    }
    case Expr::Kind::And:
    case Expr::Kind::Or: {
      typed.kind = condition.kind == Expr::Kind::And ? ConditionKind::And
                                                     : ConditionKind::Or;
      MQT_OQ3_TRY_ASSIGN(lhs, analyzeCondition(*condition.lhs));
      MQT_OQ3_TRY_ASSIGN(rhs, analyzeCondition(*condition.rhs));
      typed.lhs = lhs;
      typed.rhs = rhs;
      break;
    }
    case Expr::Kind::Equal:
    case Expr::Kind::NotEqual:
    case Expr::Kind::Less:
    case Expr::Kind::LessEqual:
    case Expr::Kind::Greater:
    case Expr::Kind::GreaterEqual: {
      const auto& lhsSyntax = syntax.expressions[*condition.lhs];
      const auto* lhsSymbol = lhsSyntax.kind == Expr::Kind::Identifier
                                  ? lookup(lhsSyntax.identifier)
                                  : nullptr;
      if (program.openQASM2 && condition.kind == Expr::Kind::Equal &&
          lhsSymbol != nullptr && lhsSymbol->kind == SymbolKind::Register &&
          program.registers[lhsSymbol->id].kind == RegisterKind::Bit &&
          isConstantExpression(*condition.rhs)) {
        const auto& rhsSyntax = syntax.expressions[*condition.rhs];
        MQT_OQ3_TRY_ASSIGN(bits,
                           resolveBits({.location = lhsSyntax.location,
                                        .identifier = lhsSyntax.identifier}));
        // OpenQASM 2 classical bits default to 0, so partially written
        // registers are valid in `if (c == k)` (e.g. mid-circuit feedback).
        llvm::APInt expectedBits;
        if (rhsSyntax.kind == Expr::Kind::Int &&
            !rhsSyntax.wideInteger.empty()) {
          llvm::SmallString<64> digits;
          for (const char value : rhsSyntax.wideInteger) {
            if (value != '_') {
              digits.push_back(value);
            }
          }
          const auto width = static_cast<unsigned>(
              std::max<size_t>(bits.size(), digits.size() * 4));
          expectedBits = llvm::APInt(width, digits, /*radix=*/10);
        } else {
          MQT_OQ3_TRY_ASSIGN(expected, evaluateConstant(*condition.rhs));
          if (!isInteger(expected.type) ||
              (expected.type == ScalarType::Int &&
               std::get<int64_t>(expected.value) < 0)) {
            return fail(
                condition.location,
                "OpenQASM 2 register conditions require an unsigned integer");
          }
          const auto expectedValue =
              expected.type == ScalarType::Uint
                  ? std::get<uint64_t>(expected.value)
                  : static_cast<uint64_t>(std::get<int64_t>(expected.value));
          expectedBits = llvm::APInt(/*numBits=*/64, expectedValue);
        }
        if (expectedBits.getActiveBits() > bits.size()) {
          // Value cannot equal the register contents.
          return addCondition(
              {.kind = ConditionKind::Literal,
               .location = getSourceLocation(condition.location),
               .literal = false});
        }
        if (expectedBits.getBitWidth() < bits.size()) {
          expectedBits = expectedBits.zext(static_cast<unsigned>(bits.size()));
        } else if (expectedBits.getBitWidth() > bits.size()) {
          expectedBits = expectedBits.trunc(static_cast<unsigned>(bits.size()));
        }
        auto result =
            addCondition({.kind = ConditionKind::Literal,
                          .location = getSourceLocation(condition.location),
                          .literal = true});
        for (const auto [index, bit] : llvm::enumerate(bits)) {
          auto bitCondition =
              addCondition({.kind = ConditionKind::Bit,
                            .location = getSourceLocation(condition.location),
                            .bit = bit});
          if (!expectedBits[index]) {
            bitCondition =
                addCondition({.kind = ConditionKind::Not,
                              .location = getSourceLocation(condition.location),
                              .lhs = bitCondition});
          }
          result =
              addCondition({.kind = ConditionKind::And,
                            .location = getSourceLocation(condition.location),
                            .lhs = result,
                            .rhs = bitCondition});
        }
        return result;
      }
      typed.kind = ConditionKind::Comparison;
      MQT_OQ3_TRY_ASSIGN(comparisonLhs, analyzeExpression(*condition.lhs));
      MQT_OQ3_TRY_ASSIGN(comparisonRhs, analyzeExpression(*condition.rhs));
      typed.comparisonLhs = comparisonLhs;
      typed.comparisonRhs = comparisonRhs;
      const auto lhsType = program.expressions[typed.comparisonLhs].type;
      const auto rhsType = program.expressions[typed.comparisonRhs].type;
      const bool boolComparison =
          lhsType == ScalarType::Bool || rhsType == ScalarType::Bool;
      if (boolComparison &&
          (lhsType != ScalarType::Bool || rhsType != ScalarType::Bool ||
           (condition.kind != Expr::Kind::Equal &&
            condition.kind != Expr::Kind::NotEqual))) {
        return fail(
            condition.location,
            "bool values only support equality comparisons with bool values");
      }
      if (!boolComparison) {
        auto comparisonType = ScalarType::Int;
        if (lhsType == ScalarType::Angle || rhsType == ScalarType::Angle) {
          comparisonType = ScalarType::Angle;
        } else if (lhsType == ScalarType::Float ||
                   rhsType == ScalarType::Float) {
          comparisonType = ScalarType::Float;
        } else if (lhsType == ScalarType::Uint || rhsType == ScalarType::Uint) {
          comparisonType = ScalarType::Uint;
        }
        MQT_OQ3_TRY_ASSIGN(
            convertedLhs,
            castExpression(typed.comparisonLhs, comparisonType,
                           syntax.expressions[*condition.lhs].location));
        MQT_OQ3_TRY_ASSIGN(
            convertedRhs,
            castExpression(typed.comparisonRhs, comparisonType,
                           syntax.expressions[*condition.rhs].location));
        typed.comparisonLhs = convertedLhs;
        typed.comparisonRhs = convertedRhs;
      }
      switch (condition.kind) {
      case Expr::Kind::Equal:
        typed.comparison = ComparisonKind::Equal;
        break;
      case Expr::Kind::NotEqual:
        typed.comparison = ComparisonKind::NotEqual;
        break;
      case Expr::Kind::Less:
        typed.comparison = ComparisonKind::Less;
        break;
      case Expr::Kind::LessEqual:
        typed.comparison = ComparisonKind::LessEqual;
        break;
      case Expr::Kind::Greater:
        typed.comparison = ComparisonKind::Greater;
        break;
      case Expr::Kind::GreaterEqual:
        typed.comparison = ComparisonKind::GreaterEqual;
        break;
      default:
        llvm_unreachable("not a comparison expression");
      }
      break;
    }
    case Expr::Kind::Int:
    case Expr::Kind::Float:
    case Expr::Kind::Bool:
    case Expr::Kind::AngleCast:
    case Expr::Kind::Neg:
    case Expr::Kind::BitNot:
    case Expr::Kind::Add:
    case Expr::Kind::Sub:
    case Expr::Kind::Mul:
    case Expr::Kind::Div:
    case Expr::Kind::ArcCos:
    case Expr::Kind::ArcSin:
    case Expr::Kind::ArcTan:
    case Expr::Kind::Ceiling:
    case Expr::Kind::Cos:
    case Expr::Kind::Exp:
    case Expr::Kind::Floor:
    case Expr::Kind::Log:
    case Expr::Kind::Mod:
    case Expr::Kind::BuiltinMod:
    case Expr::Kind::Pow:
    case Expr::Kind::BuiltinPow:
    case Expr::Kind::BitAnd:
    case Expr::Kind::BitOr:
    case Expr::Kind::BitXor:
    case Expr::Kind::ShiftLeft:
    case Expr::Kind::ShiftRight:
    case Expr::Kind::Sin:
    case Expr::Kind::Sqrt:
    case Expr::Kind::Tan:
      return fail(condition.location, "condition must have bool type");
    }
    return addCondition(std::move(typed));
  }

  [[nodiscard]] FailureOr<std::vector<GateApplication>>
  analyzeGateApplication(const SyntaxGateCall& call) {
    std::string callee = call.identifier.str();
    const GateCatalogEntry* standard = lookupGate(callee);
    auto custom = customGates.find(callee);
    uint64_t compatibilityControls = 0;
    if (standard == nullptr && custom == customGates.end() &&
        program.openQASM2) {
      auto stripped = callee;
      while (!stripped.empty() && stripped.front() == 'c') {
        stripped.erase(stripped.begin());
        ++compatibilityControls;
      }
      standard = lookupGate(stripped);
      custom = customGates.find(stripped);
      if (standard != nullptr || custom != customGates.end()) {
        callee = std::move(stripped);
      }
    }
    if (standard != nullptr && !isGateAvailable(*standard)) {
      standard = nullptr;
    }
    // Prefer a user-defined gate when present. Available standard-library
    // names are not registered as custom gates (OpenQASM 2 redefinitions are
    // skipped), so this only applies to non-stdlib custom definitions.
    if (custom != customGates.end()) {
      standard = nullptr;
    }
    if (standard == nullptr && custom == customGates.end()) {
      return fail(call.location, "No OpenQASM definition found for gate '" +
                                     call.identifier + "'.");
    }

    const auto signature =
        standard != nullptr
            ? GateSignature{.parameterCount = standard->parameterCount,
                            .qubitCount = standard->qubitCount(),
                            .variadicControls = standard->variadicControls}
            : custom->second;
    if (signature.parameterCount != call.parameters.size()) {
      return fail(call.location, "Invalid number of parameters for gate '" +
                                     call.identifier + "'.");
    }
    std::vector<ExpressionId> parameters;
    parameters.reserve(call.parameters.size());
    for (const auto expression : call.parameters) {
      MQT_OQ3_TRY_ASSIGN(parameter, analyzeExpression(expression));
      if (program.expressions[parameter].type == ScalarType::Bool) {
        return fail(call.location,
                    "gate parameters require numeric expressions");
      }
      MQT_OQ3_TRY_ASSIGN(
          convertedParameter,
          castExpression(parameter, ScalarType::Angle,
                         syntax.expressions[expression].location));
      parameters.push_back(convertedParameter);
    }

    std::vector<GateModifier> modifiers;
    size_t addedControls = compatibilityControls;
    if (addedControls > call.operands.size()) {
      return fail(call.location, "Invalid number of qubit operands for gate '" +
                                     call.identifier + "'.");
    }
    for (const auto& modifier : call.modifiers) {
      switch (modifier.kind) {
      case Modifier::Kind::Inv:
        modifiers.push_back({.kind = ModifierKind::Inv});
        break;
      case Modifier::Kind::Pow:
        if (!modifier.argument) {
          return fail(call.location, "pow modifier requires an argument");
        }
        {
          MQT_OQ3_TRY_ASSIGN(operand, analyzeExpression(*modifier.argument));
          if (program.expressions[operand].type == ScalarType::Bool ||
              program.expressions[operand].type == ScalarType::Angle) {
            return fail(call.location,
                        "pow modifier requires a numeric argument");
          }
          modifiers.push_back({.kind = ModifierKind::Pow, .operand = operand});
        }
        break;
      case Modifier::Kind::Ctrl:
      case Modifier::Kind::NegCtrl: {
        uint64_t count = 1;
        std::optional<ExpressionId> operand;
        if (modifier.argument) {
          if (!isConstantExpression(*modifier.argument)) {
            return fail(call.location,
                        "gate control count must be a constant integer");
          }
          MQT_OQ3_TRY_ASSIGN(constant, evaluateConstant(*modifier.argument));
          const auto signedCount =
              isInteger(constant.type) ? asSigned(constant) : std::nullopt;
          if (!signedCount || *signedCount <= 0) {
            return fail(call.location, "gate control count must be positive");
          }
          count = static_cast<uint64_t>(*signedCount);
          operand = addConstant(
              {.type = ScalarType::Int, .value = static_cast<int64_t>(count)});
        }
        if (count > call.operands.size() - addedControls) {
          return fail(call.location,
                      "Invalid number of qubit operands for gate '" +
                          call.identifier + "'.");
        }
        addedControls += static_cast<size_t>(count);
        modifiers.push_back({.kind = modifier.kind == Modifier::Kind::Ctrl
                                         ? ModifierKind::Ctrl
                                         : ModifierKind::NegCtrl,
                             .operand = operand});
        break;
      }
      }
    }
    if (compatibilityControls != 0) {
      modifiers.insert(modifiers.begin(),
                       {.kind = ModifierKind::Ctrl,
                        .operand = addConstant({.type = ScalarType::Int,
                                                .value = static_cast<int64_t>(
                                                    compatibilityControls)})});
    }

    const auto baseOperandCount = call.operands.size() - addedControls;
    if (signature.variadicControls ? baseOperandCount < signature.qubitCount
                                   : baseOperandCount != signature.qubitCount) {
      return fail(call.location, "Invalid number of qubit operands for gate '" +
                                     call.identifier + "'.");
    }

    size_t emittedOperandCount = call.operands.size();
    if (standard != nullptr && standard->variadicControls) {
      size_t activeBaseOperands = baseOperandCount;
      if (standard->name == "mcx_vchain") {
        if (baseOperandCount < 5) {
          return fail(call.location,
                      "mcx_vchain requires controls, a target, and ancillas");
        }
        const auto ancillas = ((baseOperandCount + 1) / 2) - 2;
        activeBaseOperands -= ancillas;
      } else if (standard->name == "mcx_recursive" && baseOperandCount > 5) {
        --activeBaseOperands;
      }
      if (activeBaseOperands <= standard->targetCount) {
        return fail(call.location, "Invalid number of controls for gate '" +
                                       call.identifier + "'.");
      }
      const auto intrinsicControls = activeBaseOperands - standard->targetCount;
      modifiers.push_back(
          {.kind = ModifierKind::Ctrl,
           .operand = addConstant(
               {.type = ScalarType::Int,
                .value = static_cast<int64_t>(intrinsicControls)})});
      callee = canonicalGateName(standard->lowering).str();
      emittedOperandCount = addedControls + activeBaseOperands;
    }

    std::vector<std::vector<QubitReference>> selections;
    size_t broadcastWidth = 1;
    for (const auto& operand : call.operands) {
      MQT_OQ3_TRY_ASSIGN(selection, resolveQubitOperand(operand));
      if (selection.size() > 1) {
        if (broadcastWidth != 1 && broadcastWidth != selection.size()) {
          return fail(call.location,
                      "all broadcasting operands must have the same width");
        }
        broadcastWidth = selection.size();
      }
      selections.push_back(std::move(selection));
    }

    std::vector<GateApplication> applications;
    applications.reserve(broadcastWidth);
    size_t affineComparisons = 0;
    for (size_t index = 0; index < broadcastWidth; ++index) {
      GateApplication application{
          .callee = callee, .parameters = parameters, .modifiers = modifiers};
      for (const auto& selection :
           ArrayRef(selections).take_front(emittedOperandCount)) {
        application.qubits.push_back(
            selection[selection.size() == 1 ? 0 : index]);
      }
      for (const auto [position, qubit] : llvm::enumerate(application.qubits)) {
        for (const auto& previous :
             ArrayRef(application.qubits).take_front(position)) {
          if (!proveDistinct(previous, qubit, affineComparisons)) {
            return fail(call.location,
                        "cannot prove that gate operands reference distinct "
                        "qubits");
          }
        }
      }
      applications.push_back(std::move(application));
    }
    return applications;
  }

  [[nodiscard]] FailureOr<std::vector<QubitReference>>
  resolveQubitOperand(const SyntaxOperand& operand) {
    if (operand.hardwareQubit) {
      if (insideGate) {
        return fail(operand.location,
                    "hardware qubits are not allowed in gate definitions");
      }
      hardwareQubits.insert(*operand.hardwareQubit);
      return std::vector<QubitReference>{{.kind = QubitReferenceKind::Hardware,
                                          .index = *operand.hardwareQubit}};
    }
    const auto* symbol = lookup(operand.identifier);
    if (insideGate) {
      if (symbol == nullptr || symbol->kind != SymbolKind::GateQubit) {
        return fail(operand.location,
                    "unknown gate-local qubit '" + operand.identifier + "'");
      }
      if (operand.index) {
        return fail(operand.location, "gate-local qubits cannot be indexed");
      }
      return std::vector<QubitReference>{
          {.kind = QubitReferenceKind::GateArgument, .symbol = symbol->id}};
    }
    if (symbol == nullptr || symbol->kind != SymbolKind::Register ||
        program.registers[symbol->id].kind != RegisterKind::Qubit) {
      return fail(operand.location,
                  "unknown qubit register '" + operand.identifier + "'");
    }
    const auto reg = static_cast<RegisterId>(symbol->id);
    const auto width = program.registers[reg].width;
    if (!operand.index) {
      std::vector<QubitReference> selection;
      selection.reserve(width);
      for (uint64_t index = 0; index < width; ++index) {
        selection.push_back({.kind = QubitReferenceKind::Register,
                             .symbol = reg,
                             .index = index});
      }
      return selection;
    }
    MQT_OQ3_TRY_ASSIGN(constant,
                       constantIndex(*operand.index, width, operand.location));
    if (constant) {
      if (*constant >= width) {
        return fail(operand.location,
                    "cannot prove that qubit index is in bounds");
      }
      return std::vector<QubitReference>{{.kind = QubitReferenceKind::Register,
                                          .symbol = reg,
                                          .index = *constant}};
    }
    MQT_OQ3_TRY_ASSIGN(index, analyzeExpression(*operand.index));
    if (!isInteger(program.expressions[index].type)) {
      return fail(operand.location,
                  "qubit index must be an integer expression");
    }
    const auto form = buildAffineForm(index);
    if (!form) {
      return fail(operand.location,
                  "cannot prove that qubit index is in bounds");
    }
    if (isAffineConstant(*form)) {
      auto value = static_cast<int64_t>(form->constant);
      if (value < 0) {
        value += static_cast<int64_t>(width);
      }
      if (value < 0 || std::cmp_greater_equal(value, width)) {
        return fail(operand.location,
                    "cannot prove that qubit index is in bounds");
      }
      return std::vector<QubitReference>{
          {.kind = QubitReferenceKind::Register,
           .symbol = reg,
           .index = static_cast<uint64_t>(value)}};
    }
    if (!provesLowerBound(*form, llvm::DynamicAPInt(0)) ||
        !provesUpperBound(
            *form, unsignedCoefficient(static_cast<uint64_t>(width - 1)))) {
      return fail(operand.location,
                  "cannot prove that qubit index is in bounds");
    }
    const auto expandedIndex = expandAffineScalarValues(index);
    assert(expandedIndex &&
           "proven affine indices must have expandable expressions");
    return std::vector<QubitReference>{{.kind = QubitReferenceKind::Register,
                                        .symbol = reg,
                                        .provenIndex = expandedIndex}};
  }

  [[nodiscard]] FailureOr<std::vector<frontend::BitReference>>
  resolveBits(const SyntaxBitReference& reference) {
    const auto* symbol = lookup(reference.identifier);
    if (symbol == nullptr || symbol->kind != SymbolKind::Register ||
        program.registers[symbol->id].kind == RegisterKind::Qubit) {
      return fail(reference.location,
                  "unknown classical register '" + reference.identifier + "'");
    }
    const auto reg = static_cast<RegisterId>(symbol->id);
    const auto width = program.registers[reg].width;
    if (!reference.index) {
      std::vector<frontend::BitReference> result;
      result.reserve(width);
      for (uint64_t index = 0; index < width; ++index) {
        result.push_back({.reg = reg, .index = index});
      }
      return result;
    }
    MQT_OQ3_TRY_ASSIGN(
        constant, constantIndex(*reference.index, width, reference.location));
    if (constant) {
      if (*constant >= width) {
        return fail(reference.location, "classical bit index is out of bounds");
      }
      return std::vector<frontend::BitReference>{
          {.reg = reg, .index = *constant}};
    }
    MQT_OQ3_TRY_ASSIGN(dynamic, analyzeExpression(*reference.index));
    if (!isInteger(program.expressions[dynamic].type)) {
      return fail(reference.location,
                  "classical bit index must be an integer expression");
    }
    return std::vector<frontend::BitReference>{
        {.reg = reg, .dynamicIndex = dynamic}};
  }

  [[nodiscard]] LogicalResult
  ensureBitInitialized(const frontend::BitReference& bit,
                       SMLoc location) const {
    if (bit.dynamicIndex) {
      if (llvm::all_of(*initializedBits[bit.reg],
                       [](const bool initialized) { return initialized; })) {
        return success();
      }
      std::vector<std::pair<uint64_t, uint64_t>> dependencies;
      collectDependencies(*bit.dynamicIndex, dependencies);
      if (llvm::any_of(*dynamicBitFacts[bit.reg], [&](const auto& fact) {
            return fact.dependencies == dependencies &&
                   sameExpression(fact.expression, *bit.dynamicIndex);
          })) {
        return success();
      }
      return fail(location,
                  "dynamic classical index may read an uninitialized bit");
    }
    if (!(*initializedBits[bit.reg])[bit.index]) {
      return fail(location, "classical condition bit has not been initialized");
    }
    return success();
  }

  [[nodiscard]] LogicalResult finalizeOutputs() {
    program.outputs =
        explicitOutputs.empty() ? implicitOutputs : explicitOutputs;
    for (const auto output : program.outputs) {
      if (output.kind == OutputKind::Scalar) {
        if (!initializedScalars[output.symbol]) {
          return fail(program.scalars[output.symbol].location,
                      "Output scalar '" + program.scalars[output.symbol].name +
                          "' is not initialized.");
        }
        continue;
      }
      const auto reg = static_cast<RegisterId>(output.symbol);
      if (llvm::any_of(*initializedBits[reg],
                       [](const bool initialized) { return !initialized; })) {
        return fail(program.registers[reg].location,
                    "Output register '" + program.registers[reg].name +
                        "' is not fully initialized.");
      }
    }
    return success();
  }
};

#undef MQT_OQ3_TRY_ASSIGN

} // namespace

SourceLocation sourceLocation(const llvm::SourceMgr& sources,
                              const llvm::SMLoc location) {
  if (!location.isValid()) {
    return {};
  }
  const auto bufferId = sources.FindBufferContainingLoc(location);
  if (bufferId == 0) {
    return {};
  }
  const auto [line, column] = sources.getLineAndColumn(location, bufferId);
  const auto* buffer = sources.getMemoryBuffer(bufferId);
  SourceLocation result{.filename = buffer->getBufferIdentifier().str(),
                        .line = line,
                        .column = column};
  auto parent = sources.getParentIncludeLoc(bufferId);
  while (parent.isValid()) {
    const auto parentBufferId = sources.FindBufferContainingLoc(parent);
    if (parentBufferId == 0) {
      break;
    }
    const auto [parentLine, parentColumn] =
        sources.getLineAndColumn(parent, parentBufferId);
    result.includeStack.push_back(
        {.filename = sources.getMemoryBuffer(parentBufferId)
                         ->getBufferIdentifier()
                         .str(),
         .line = parentLine,
         .column = parentColumn});
    parent = sources.getParentIncludeLoc(parentBufferId);
  }
  return result;
}

AnalysisResult analyzeSyntaxProgram(const SyntaxProgram& syntax,
                                    const llvm::SourceMgr& sources,
                                    const FrontendOptions& options) {
  return SemanticAnalyzer(syntax, sources, options).run();
}

} // namespace mlir::oq3::frontend::detail

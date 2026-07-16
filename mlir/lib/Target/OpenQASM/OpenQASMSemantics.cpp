/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "OpenQASMSemantics.h"

#include "mlir/Dialect/OQ3/IR/GateCatalog.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mlir::oq3::frontend::detail {
namespace {

class SemanticError final : public std::runtime_error {
public:
  Diagnostic diagnostic;

  explicit SemanticError(Diagnostic value)
      : std::runtime_error(value.message), diagnostic(std::move(value)) {}
};

struct Constant {
  ScalarType type = ScalarType::Int;
  std::variant<bool, std::int64_t, std::uint64_t, double> value =
      std::int64_t{0};
};

struct GateSignature {
  std::size_t parameterCount = 0;
  std::size_t qubitCount = 0;
  bool variadicControls = false;
};

enum class SymbolKind : std::uint8_t {
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
  std::uint32_t id = 0;
  std::optional<Constant> constant;
};

[[nodiscard]] ScalarType scalarType(const ScalarKind kind) {
  switch (kind) {
  case ScalarKind::Bool:
    return ScalarType::Bool;
  case ScalarKind::Int:
    return ScalarType::Int;
  case ScalarKind::Uint:
    return ScalarType::Uint;
  case ScalarKind::Float:
    return ScalarType::Float;
  }
  llvm_unreachable("unknown syntax scalar kind");
}

[[nodiscard]] bool isInteger(const ScalarType type) {
  return type == ScalarType::Int || type == ScalarType::Uint;
}

[[nodiscard]] double asDouble(const Constant& constant) {
  return std::visit([](const auto value) { return static_cast<double>(value); },
                    constant.value);
}

[[nodiscard]] std::int64_t asSigned(const Constant& constant) {
  if (constant.type == ScalarType::Uint) {
    const auto value = std::get<std::uint64_t>(constant.value);
    if (value >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::overflow_error("unsigned value does not fit in signed i64");
    }
    return static_cast<std::int64_t>(value);
  }
  return std::get<std::int64_t>(constant.value);
}

[[nodiscard]] int compareNumericConstants(const Constant& lhs,
                                          const Constant& rhs) {
  if (lhs.type == ScalarType::Float || rhs.type == ScalarType::Float) {
    const auto left = asDouble(lhs);
    const auto right = asDouble(rhs);
    return left < right ? -1 : left > right ? 1 : 0;
  }
  if (lhs.type == ScalarType::Uint || rhs.type == ScalarType::Uint) {
    const auto asUnsigned = [](const Constant& constant) {
      return constant.type == ScalarType::Uint
                 ? std::get<std::uint64_t>(constant.value)
                 : static_cast<std::uint64_t>(
                       std::get<std::int64_t>(constant.value));
    };
    const auto left = asUnsigned(lhs);
    const auto right = asUnsigned(rhs);
    return left < right ? -1 : left > right ? 1 : 0;
  }
  const auto left = std::get<std::int64_t>(lhs.value);
  const auto right = std::get<std::int64_t>(rhs.value);
  return left < right ? -1 : left > right ? 1 : 0;
}

class SemanticAnalyzer {
public:
  SemanticAnalyzer(const SyntaxProgram& syntaxProgram,
                   const llvm::SourceMgr& sourceManager,
                   const FrontendOptions& frontendOptions)
      : syntax(syntaxProgram), sources(sourceManager),
        options(frontendOptions) {
    program.gatePolicy = options.gatePolicy;
    scopes.emplace_back();
  }

  [[nodiscard]] AnalysisResult run() {
    try {
      analyzeVersion();
      analyzeBody(syntax.body, program.body, /*global=*/true);
      validateGateCallGraph();
      finalizeOutputs();
      return {.program = std::make_unique<TypedProgram>(std::move(program))};
    } catch (const SemanticError& error) {
      return {.diagnostics = {error.diagnostic}};
    } catch (const std::exception& error) {
      return {.diagnostics = {{.message = error.what()}}};
    }
  }

private:
  struct DynamicBitFact {
    ExpressionId expression = 0;
    std::vector<std::pair<ScalarId, std::uint64_t>> dependencies;
  };

  const SyntaxProgram& syntax;
  const llvm::SourceMgr& sources;
  FrontendOptions options;
  TypedProgram program;
  SmallVector<llvm::StringMap<Symbol>> scopes;
  llvm::StringMap<GateSignature> customGates;
  std::vector<std::vector<bool>> initializedBits;
  std::vector<std::vector<DynamicBitFact>> dynamicBitFacts;
  std::vector<bool> initializedScalars;
  std::vector<std::uint64_t> scalarGenerations;
  std::vector<RegisterId> bitRegisters;
  std::vector<RegisterId> explicitOutputs;
  bool insideGate = false;
  bool hasVirtualQubits = false;
  bool hasHardwareQubits = false;
  std::set<std::uint64_t> hardwareQubits;

  [[noreturn]] void fail(SMLoc location, const Twine& message) const {
    throw SemanticError({.location = sourceLocation(sources, location),
                         .message = message.str()});
  }

  void validateDynamicDispatchCost(SMLoc location,
                                   ArrayRef<QubitReference> references) const {
    std::size_t leaves = 1;
    for (const auto& reference : references) {
      if (!reference.dynamicIndex) {
        continue;
      }
      const auto width = program.registers.at(reference.symbol).width;
      if (width > kDynamicQubitDispatchLeafLimit / leaves) {
        fail(location,
             "dynamic qubit selection exceeds the structured-dispatch "
             "expansion budget");
      }
      leaves *= static_cast<std::size_t>(width);
    }
  }

  void restoreStatePrefix(const std::vector<std::vector<bool>>& bitsInitialized,
                          const std::vector<bool>& scalarsInitialized,
                          const std::vector<std::uint64_t>& generations) {
    for (std::size_t reg = 0; reg < bitsInitialized.size(); ++reg) {
      initializedBits[reg] = bitsInitialized[reg];
    }
    for (std::size_t scalar = 0; scalar < scalarsInitialized.size(); ++scalar) {
      initializedScalars[scalar] = scalarsInitialized[scalar];
      scalarGenerations[scalar] = generations[scalar];
    }
  }

  void restoreDynamicFactsPrefix(
      const std::vector<std::vector<DynamicBitFact>>& facts) {
    for (std::size_t reg = 0; reg < facts.size(); ++reg) {
      dynamicBitFacts[reg] = facts[reg];
    }
    for (std::size_t reg = facts.size(); reg < dynamicBitFacts.size(); ++reg) {
      dynamicBitFacts[reg].clear();
    }
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
    case ExpressionKind::Negate:
    case ExpressionKind::ArcCos:
    case ExpressionKind::ArcSin:
    case ExpressionKind::ArcTan:
    case ExpressionKind::Sin:
    case ExpressionKind::Cos:
    case ExpressionKind::Tan:
    case ExpressionKind::Exp:
    case ExpressionKind::Ln:
    case ExpressionKind::Sqrt:
      return sameExpression(left.lhs, right.lhs);
    default:
      return sameExpression(left.lhs, right.lhs) &&
             sameExpression(left.rhs, right.rhs);
    }
  }

  void collectDependencies(
      const ExpressionId expression,
      std::vector<std::pair<ScalarId, std::uint64_t>>& dependencies) const {
    const auto& value = program.expressions[expression];
    if (value.kind == ExpressionKind::Variable) {
      dependencies.emplace_back(value.variable,
                                scalarGenerations[value.variable]);
      return;
    }
    if (value.kind == ExpressionKind::Constant ||
        value.kind == ExpressionKind::GateParameter) {
      return;
    }
    collectDependencies(value.lhs, dependencies);
    if (value.kind != ExpressionKind::Negate &&
        value.kind != ExpressionKind::ArcCos &&
        value.kind != ExpressionKind::ArcSin &&
        value.kind != ExpressionKind::ArcTan &&
        value.kind != ExpressionKind::Sin &&
        value.kind != ExpressionKind::Cos &&
        value.kind != ExpressionKind::Tan &&
        value.kind != ExpressionKind::Exp && value.kind != ExpressionKind::Ln &&
        value.kind != ExpressionKind::Sqrt) {
      collectDependencies(value.rhs, dependencies);
    }
  }

  [[nodiscard]] std::optional<bool>
  constantCondition(const SyntaxExpressionId expression) const {
    if (!isConstantExpression(expression)) {
      return std::nullopt;
    }
    const auto value = evaluateConstant(expression);
    if (value.type != ScalarType::Bool) {
      return std::nullopt;
    }
    return std::get<bool>(value.value);
  }

  [[nodiscard]] const Symbol* lookup(StringRef name) const {
    for (const auto& scope : llvm::reverse(scopes)) {
      if (const auto found = scope.find(name); found != scope.end()) {
        return &found->second;
      }
    }
    return nullptr;
  }

  void declare(SMLoc location, StringRef name, Symbol symbol) {
    const auto* catalog = lookupGate(name);
    const bool catalogNameReserved =
        catalog != nullptr &&
        (catalog->availability == GateAvailability::Language ||
         (catalog->availability == GateAvailability::StandardLibrary &&
          program.standardLibraryIncluded));
    if (builtinConstant(name) ||
        (scopes.size() == 1 &&
         (customGates.contains(name) || catalogNameReserved))) {
      fail(location, "identifier '" + name + "' is already declared");
    }
    if (!scopes.back().insert({name, std::move(symbol)}).second) {
      fail(location, "identifier '" + name + "' is already declared");
    }
  }

  [[nodiscard]] bool isGateAvailable(const GateCatalogEntry& gate) const {
    if (gate.availability == GateAvailability::Language) {
      return true;
    }
    if (options.gatePolicy == GatePolicy::MQTCompatibility) {
      return true;
    }
    return gate.availability == GateAvailability::StandardLibrary &&
           program.standardLibraryIncluded;
  }

  void analyzeVersion() {
    if (!syntax.version) {
      return;
    }
    const auto version = *syntax.version;
    if (version.major == 2 && version.minor == 0) {
      program.openQASM2 = true;
      return;
    }
    if (version.major == 3 && (version.minor == 0 || version.minor == 1)) {
      return;
    }
    fail(syntax.versionLocation, "Unsupported OpenQASM version " +
                                     std::to_string(version.major) + "." +
                                     std::to_string(version.minor));
  }

  void validateGateCallGraph() const {
    llvm::StringMap<std::size_t> gateIndices;
    for (const auto [index, gate] : llvm::enumerate(program.gates)) {
      gateIndices[gate.name] = index;
    }
    enum class VisitState : std::uint8_t { Unvisited, Active, Complete };
    std::vector states(program.gates.size(), VisitState::Unvisited);
    const auto visitApplications = [&](auto&& self,
                                       ArrayRef<StatementId> statements,
                                       const auto& callback) -> void {
      for (const auto statementId : statements) {
        const auto& statement = program.statements[statementId];
        std::visit(
            [&](const auto& data) {
              using T = std::decay_t<decltype(data)>;
              if constexpr (std::is_same_v<T, GateApplication>) {
                callback(data, statement.location);
              } else if constexpr (std::is_same_v<T, IfStatement>) {
                self(self, data.thenStatements, callback);
                self(self, data.elseStatements, callback);
              } else if constexpr (std::is_same_v<T, ForStatement> ||
                                   std::is_same_v<T, WhileStatement>) {
                self(self, data.body, callback);
              }
            },
            statement.data);
      }
    };
    const auto visit = [&](auto&& self, const std::size_t index) -> void {
      if (states[index] == VisitState::Complete) {
        return;
      }
      states[index] = VisitState::Active;
      visitApplications(
          visitApplications, program.gates[index].body,
          [&](const GateApplication& application,
              const SourceLocation location) {
            const auto callee = gateIndices.find(application.callee);
            if (callee == gateIndices.end()) {
              return;
            }
            if (states[callee->second] == VisitState::Active) {
              throw SemanticError(
                  {.location = location,
                   .message = "recursive custom gate definition involving '" +
                              application.callee + "'"});
            }
            self(self, callee->second);
          });
      states[index] = VisitState::Complete;
    };
    for (std::size_t index = 0; index < program.gates.size(); ++index) {
      if (states[index] == VisitState::Unvisited) {
        visit(visit, index);
      }
    }
  }

  [[nodiscard]] StatementId addStatement(SMLoc location, StatementData data) {
    const auto id = static_cast<StatementId>(program.statements.size());
    program.statements.push_back(
        {.data = std::move(data),
         .location = sourceLocation(sources, location)});
    return id;
  }

  [[nodiscard]] ExpressionId addExpression(ScalarExpression expression) {
    const auto id = static_cast<ExpressionId>(program.expressions.size());
    program.expressions.push_back(std::move(expression));
    return id;
  }

  [[nodiscard]] ConditionId addCondition(ConditionExpression condition) {
    const auto id = static_cast<ConditionId>(program.conditions.size());
    program.conditions.push_back(std::move(condition));
    return id;
  }

  [[nodiscard]] ExpressionId addConstant(const Constant& constant) {
    return addExpression({.kind = ExpressionKind::Constant,
                          .type = constant.type,
                          .constant = constant.value});
  }

  [[nodiscard]] Constant coerceConstant(Constant constant,
                                        const ScalarType type,
                                        const SMLoc location) const {
    if (type == ScalarType::Bool) {
      if (constant.type == ScalarType::Bool) {
        return constant;
      }
      return {.type = ScalarType::Bool, .value = asDouble(constant) != 0.0};
    }
    if (type == ScalarType::Float) {
      return {.type = ScalarType::Float, .value = asDouble(constant)};
    }
    if (constant.type == ScalarType::Bool) {
      constant = {.type = ScalarType::Int,
                  .value = std::int64_t{std::get<bool>(constant.value)}};
    } else if (constant.type == ScalarType::Float) {
      const auto value = std::get<double>(constant.value);
      const auto withinTargetRange =
          type == ScalarType::Int
              ? value >= static_cast<double>(
                             std::numeric_limits<std::int64_t>::min()) &&
                    value < std::ldexp(1.0, 63)
              : value > -1.0 && value < std::ldexp(1.0, 64);
      if (!std::isfinite(value) || !withinTargetRange) {
        fail(location, "floating-point value is outside the integer range");
      }
      if (type == ScalarType::Int) {
        constant = {.type = ScalarType::Int,
                    .value = static_cast<std::int64_t>(value)};
      } else {
        constant = {.type = ScalarType::Uint,
                    .value = static_cast<std::uint64_t>(value)};
      }
    }
    if (type == ScalarType::Int) {
      if (constant.type == ScalarType::Uint) {
        return {.type = ScalarType::Int,
                .value = std::bit_cast<std::int64_t>(
                    std::get<std::uint64_t>(constant.value))};
      }
      return constant;
    }
    if (constant.type == ScalarType::Int) {
      return {.type = ScalarType::Uint,
              .value = std::bit_cast<std::uint64_t>(
                  std::get<std::int64_t>(constant.value))};
    }
    return constant;
  }

  [[nodiscard]] bool expressionProducesBool(const SyntaxExpressionId id) const {
    const auto& expression = syntax.expressions[id];
    switch (expression.kind) {
    case Expr::Kind::Bool:
    case Expr::Kind::Measurement:
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

  [[nodiscard]] ConditionId
  analyzeBoolValue(const SyntaxExpressionId syntaxId) {
    if (expressionProducesBool(syntaxId)) {
      return analyzeCondition(syntaxId);
    }
    if (isConstantExpression(syntaxId)) {
      const auto constant = evaluateConstant(syntaxId);
      return addCondition({.kind = ConditionKind::Literal,
                           .location = sourceLocation(
                               sources, syntax.expressions[syntaxId].location),
                           .literal = asDouble(constant) != 0.0});
    }
    const auto value = analyzeExpression(syntaxId);
    const auto type = program.expressions[value].type;
    const auto zero = addConstant(
        type == ScalarType::Float
            ? Constant{.type = ScalarType::Float, .value = 0.0}
        : type == ScalarType::Uint
            ? Constant{.type = ScalarType::Uint, .value = std::uint64_t{0}}
            : Constant{.type = ScalarType::Int, .value = std::int64_t{0}});
    return addCondition({.kind = ConditionKind::Comparison,
                         .location = sourceLocation(
                             sources, syntax.expressions[syntaxId].location),
                         .comparisonLhs = value,
                         .comparisonRhs = zero,
                         .comparison = ComparisonKind::NotEqual});
  }

  [[nodiscard]] std::optional<Constant>
  builtinConstant(StringRef identifier) const {
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

  [[nodiscard]] Constant evaluateConstant(const SyntaxExpressionId id) const {
    const auto& expression = syntax.expressions[id];
    switch (expression.kind) {
    case Expr::Kind::Int:
      if (expression.integer <= static_cast<std::uint64_t>(
                                    std::numeric_limits<std::int64_t>::max())) {
        return {.type = ScalarType::Int,
                .value = static_cast<std::int64_t>(expression.integer)};
      }
      return {.type = ScalarType::Uint, .value = expression.integer};
    case Expr::Kind::Float:
      return {.type = ScalarType::Float, .value = expression.floatingPoint};
    case Expr::Kind::Bool:
      return {.type = ScalarType::Bool, .value = expression.boolean};
    case Expr::Kind::Identifier: {
      if (const auto builtin = builtinConstant(expression.identifier)) {
        return *builtin;
      }
      const auto* symbol = lookup(expression.identifier);
      if (symbol == nullptr || symbol->kind != SymbolKind::Constant ||
          !symbol->constant) {
        fail(expression.location, "expression is not a compile-time constant");
      }
      return *symbol->constant;
    }
    case Expr::Kind::Neg: {
      auto operand = evaluateConstant(*expression.lhs);
      if (operand.type == ScalarType::Bool) {
        fail(expression.location,
             "numeric negation requires a numeric operand");
      }
      if (operand.type == ScalarType::Float) {
        return {.type = ScalarType::Float, .value = -asDouble(operand)};
      }
      if (operand.type == ScalarType::Uint) {
        const auto value = std::get<std::uint64_t>(operand.value);
        if (syntax.expressions[*expression.lhs].kind == Expr::Kind::Int) {
          if (value > (1ULL << 63)) {
            fail(expression.location, "integer negation overflows i64");
          }
          return {.type = ScalarType::Int,
                  .value = std::numeric_limits<std::int64_t>::min()};
        }
        return {.type = ScalarType::Uint, .value = 0ULL - value};
      }
      const auto value = std::get<std::int64_t>(operand.value);
      if (value == std::numeric_limits<std::int64_t>::min()) {
        fail(expression.location, "integer negation overflows i64");
      }
      return {.type = ScalarType::Int, .value = -value};
    }
    case Expr::Kind::Not: {
      const auto operand = evaluateConstant(*expression.lhs);
      if (operand.type != ScalarType::Bool) {
        fail(expression.location, "logical negation requires a bool operand");
      }
      return {.type = ScalarType::Bool,
              .value = !std::get<bool>(operand.value)};
    }
    case Expr::Kind::BitNot: {
      fail(expression.location,
           "bitwise operators require explicitly sized uint, bit, or angle "
           "operands, which are not supported yet");
    }
    case Expr::Kind::And:
    case Expr::Kind::Or: {
      const auto lhs = evaluateConstant(*expression.lhs);
      const auto rhs = evaluateConstant(*expression.rhs);
      if (lhs.type != ScalarType::Bool || rhs.type != ScalarType::Bool) {
        fail(expression.location, "logical operators require bool operands");
      }
      const auto left = std::get<bool>(lhs.value);
      const auto right = std::get<bool>(rhs.value);
      return {.type = ScalarType::Bool,
              .value = expression.kind == Expr::Kind::And ? left && right
                                                          : left || right};
    }
    case Expr::Kind::Equal:
    case Expr::Kind::NotEqual:
    case Expr::Kind::Less:
    case Expr::Kind::LessEqual:
    case Expr::Kind::Greater:
    case Expr::Kind::GreaterEqual: {
      const auto lhs = evaluateConstant(*expression.lhs);
      const auto rhs = evaluateConstant(*expression.rhs);
      bool result = false;
      if (lhs.type == ScalarType::Bool || rhs.type == ScalarType::Bool) {
        if (lhs.type != ScalarType::Bool || rhs.type != ScalarType::Bool ||
            (expression.kind != Expr::Kind::Equal &&
             expression.kind != Expr::Kind::NotEqual)) {
          fail(expression.location,
               "bool values only support equality comparisons with bool "
               "values");
        }
        const auto equal =
            std::get<bool>(lhs.value) == std::get<bool>(rhs.value);
        result = expression.kind == Expr::Kind::Equal ? equal : !equal;
      } else {
        const auto ordering = compareNumericConstants(lhs, rhs);
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
      return {.type = ScalarType::Bool, .value = result};
    }
    case Expr::Kind::ArcCos:
    case Expr::Kind::ArcSin:
    case Expr::Kind::ArcTan:
    case Expr::Kind::Cos:
    case Expr::Kind::Exp:
    case Expr::Kind::Log:
    case Expr::Kind::Sin:
    case Expr::Kind::Sqrt:
    case Expr::Kind::Tan: {
      const auto value = asDouble(evaluateConstant(*expression.lhs));
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
      case Expr::Kind::Cos:
        result = std::cos(value);
        break;
      case Expr::Kind::Exp:
        result = std::exp(value);
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
        fail(expression.location,
             "constant math expression has a non-finite result");
      }
      return {.type = ScalarType::Float, .value = result};
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
      fail(expression.location,
           "bitwise operators require explicitly sized uint, bit, or angle "
           "operands, which are not supported yet");
    case Expr::Kind::Index:
    case Expr::Kind::Measurement:
      fail(expression.location, "expression is not a compile-time constant");
    }
    llvm_unreachable("unknown syntax expression kind");
  }

  [[nodiscard]] Constant
  evaluateConstantBinary(const SyntaxExpression& expression) const {
    const auto lhs = evaluateConstant(*expression.lhs);
    const auto rhs = evaluateConstant(*expression.rhs);
    if (lhs.type == ScalarType::Bool || rhs.type == ScalarType::Bool) {
      fail(expression.location,
           "arithmetic operators require numeric operands");
    }
    const bool builtinFloatPower = expression.kind == Expr::Kind::BuiltinPow &&
                                   rhs.type == ScalarType::Int &&
                                   std::get<std::int64_t>(rhs.value) < 0;
    if (lhs.type == ScalarType::Float || rhs.type == ScalarType::Float ||
        builtinFloatPower) {
      if (expression.kind == Expr::Kind::Mod) {
        fail(expression.location,
             "the '%' operator requires integer operands; use mod() for "
             "floating-point remainder");
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
          fail(expression.location, "division by zero");
        }
        result = left / right;
        break;
      case Expr::Kind::BuiltinMod:
        if (right == 0.0) {
          fail(expression.location, "modulo by zero");
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
        fail(expression.location,
             "constant arithmetic has a non-finite result");
      }
      return {.type = ScalarType::Float, .value = result};
    }

    if (lhs.type == ScalarType::Uint || rhs.type == ScalarType::Uint) {
      const auto asUnsigned = [](const Constant& constant) {
        return constant.type == ScalarType::Uint
                   ? std::get<std::uint64_t>(constant.value)
                   : static_cast<std::uint64_t>(
                         std::get<std::int64_t>(constant.value));
      };
      const auto left = asUnsigned(lhs);
      const auto right = asUnsigned(rhs);
      std::uint64_t result = 0;
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
          fail(expression.location, "division by zero");
        }
        result = left / right;
        break;
      case Expr::Kind::Mod:
      case Expr::Kind::BuiltinMod:
        if (right == 0) {
          fail(expression.location, "modulo by zero");
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
      return {.type = ScalarType::Uint, .value = result};
    }

    const auto left = std::get<std::int64_t>(lhs.value);
    const auto right = std::get<std::int64_t>(rhs.value);
    std::int64_t result = 0;
    bool overflow = false;
    switch (expression.kind) {
    case Expr::Kind::Add:
      overflow = __builtin_add_overflow(left, right, &result);
      break;
    case Expr::Kind::Sub:
      overflow = __builtin_sub_overflow(left, right, &result);
      break;
    case Expr::Kind::Mul:
      overflow = __builtin_mul_overflow(left, right, &result);
      break;
    case Expr::Kind::Div:
      if (right == 0) {
        fail(expression.location, "division by zero");
      }
      if (left == std::numeric_limits<std::int64_t>::min() && right == -1) {
        overflow = true;
      } else {
        result = left / right;
      }
      break;
    case Expr::Kind::Mod:
    case Expr::Kind::BuiltinMod:
      if (right == 0) {
        fail(expression.location, "modulo by zero");
      }
      if (left == std::numeric_limits<std::int64_t>::min() && right == -1) {
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
        fail(expression.location,
             "integer power requires a nonnegative exponent");
      }
      result = 1;
      auto base = left;
      auto exponent = static_cast<std::uint64_t>(right);
      while (exponent != 0 && !overflow) {
        if ((exponent & 1U) != 0) {
          overflow = __builtin_mul_overflow(result, base, &result);
        }
        exponent >>= 1U;
        if (exponent != 0 && !overflow) {
          overflow = __builtin_mul_overflow(base, base, &base);
        }
      }
      break;
    }
    default:
      llvm_unreachable("not a binary expression");
    }
    if (overflow) {
      fail(expression.location, "constant integer arithmetic overflows i64");
    }
    return {.type = ScalarType::Int, .value = result};
  }

  [[nodiscard]] bool isConstantExpression(const SyntaxExpressionId id) const {
    const auto& expression = syntax.expressions[id];
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
    case Expr::Kind::Measurement:
      return false;
    default:
      return (!expression.lhs || isConstantExpression(*expression.lhs)) &&
             (!expression.rhs || isConstantExpression(*expression.rhs));
    }
  }

  void validateGateExpression(const SyntaxExpressionId id) const {
    const auto& expression = syntax.expressions[id];
    if (expression.kind == Expr::Kind::Identifier &&
        !builtinConstant(expression.identifier)) {
      const auto* symbol = lookup(expression.identifier);
      if (symbol == nullptr || (symbol->kind != SymbolKind::GateParameter &&
                                symbol->kind != SymbolKind::GateLocalScalar &&
                                symbol->kind != SymbolKind::Constant)) {
        fail(expression.location,
             "gate definitions cannot capture outer scalar '" +
                 expression.identifier + "'");
      }
    }
    if (expression.lhs) {
      validateGateExpression(*expression.lhs);
    }
    if (expression.rhs) {
      validateGateExpression(*expression.rhs);
    }
  }

  [[nodiscard]] ExpressionId
  analyzeExpression(const SyntaxExpressionId syntaxId) {
    const auto& expression = syntax.expressions[syntaxId];
    if (insideGate) {
      validateGateExpression(syntaxId);
    }
    if (isConstantExpression(syntaxId)) {
      return addConstant(evaluateConstant(syntaxId));
    }
    if (expression.kind == Expr::Kind::Identifier) {
      const auto* symbol = lookup(expression.identifier);
      if (symbol == nullptr) {
        fail(expression.location,
             "unknown scalar identifier '" + expression.identifier + "'");
      }
      if (symbol->kind == SymbolKind::GateParameter) {
        return addExpression({.kind = ExpressionKind::GateParameter,
                              .type = ScalarType::Float,
                              .parameter = symbol->id});
      }
      if (symbol->kind != SymbolKind::Scalar &&
          symbol->kind != SymbolKind::GateLocalScalar) {
        fail(expression.location, "identifier '" + expression.identifier +
                                      "' is not a scalar value");
      }
      if (!initializedScalars.at(symbol->id)) {
        fail(expression.location,
             "scalar '" + expression.identifier + "' is uninitialized");
      }
      return addExpression({.kind = ExpressionKind::Variable,
                            .type = symbol->type,
                            .variable = symbol->id});
    }

    ExpressionKind kind;
    switch (expression.kind) {
    case Expr::Kind::Neg:
      kind = ExpressionKind::Negate;
      break;
    case Expr::Kind::BitNot:
      fail(expression.location,
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
      fail(expression.location,
           "bitwise operators require explicitly sized uint, bit, or angle "
           "operands, which are not supported yet");
    case Expr::Kind::ArcCos:
      kind = ExpressionKind::ArcCos;
      break;
    case Expr::Kind::ArcSin:
      kind = ExpressionKind::ArcSin;
      break;
    case Expr::Kind::ArcTan:
      kind = ExpressionKind::ArcTan;
      break;
    case Expr::Kind::Cos:
      kind = ExpressionKind::Cos;
      break;
    case Expr::Kind::Exp:
      kind = ExpressionKind::Exp;
      break;
    case Expr::Kind::Log:
      kind = ExpressionKind::Ln;
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
    case Expr::Kind::Measurement:
      fail(expression.location, "expected a scalar arithmetic expression");
    case Expr::Kind::Int:
    case Expr::Kind::Float:
    case Expr::Kind::Bool:
    case Expr::Kind::Identifier:
      llvm_unreachable("handled expression kind");
    }
    const auto lhs = analyzeExpression(*expression.lhs);
    const auto rhs =
        expression.rhs
            ? std::optional<ExpressionId>(analyzeExpression(*expression.rhs))
            : std::nullopt;
    if (program.expressions[lhs].type == ScalarType::Bool ||
        (rhs && program.expressions[*rhs].type == ScalarType::Bool)) {
      fail(expression.location,
           "arithmetic operators require numeric operands");
    }
    if (expression.kind == Expr::Kind::Mod &&
        (program.expressions[lhs].type == ScalarType::Float ||
         (rhs && program.expressions[*rhs].type == ScalarType::Float))) {
      fail(expression.location,
           "the '%' operator requires integer operands; use mod() for "
           "floating-point remainder");
    }
    auto type = program.expressions[lhs].type;
    if (kind == ExpressionKind::ArcCos || kind == ExpressionKind::ArcSin ||
        kind == ExpressionKind::ArcTan || kind == ExpressionKind::Cos ||
        kind == ExpressionKind::Exp || kind == ExpressionKind::Ln ||
        kind == ExpressionKind::Sin || kind == ExpressionKind::Sqrt ||
        kind == ExpressionKind::Tan || type == ScalarType::Float ||
        (expression.kind == Expr::Kind::BuiltinPow && rhs &&
         program.expressions[*rhs].type == ScalarType::Int) ||
        (rhs && program.expressions[*rhs].type == ScalarType::Float)) {
      type = ScalarType::Float;
    } else if (rhs && (type == ScalarType::Uint ||
                       program.expressions[*rhs].type == ScalarType::Uint)) {
      type = ScalarType::Uint;
    }
    return addExpression(
        {.kind = kind, .type = type, .lhs = lhs, .rhs = rhs.value_or(0)});
  }

  [[nodiscard]] std::uint64_t
  constantWidth(const std::optional<SyntaxExpressionId> size,
                SMLoc location) const {
    if (!size) {
      return 1;
    }
    if (!isConstantExpression(*size)) {
      fail(location, "register width must be a constant integer expression");
    }
    const auto constant = evaluateConstant(*size);
    if (!isInteger(constant.type)) {
      fail(location, "register width must be an integer expression");
    }
    const auto value = asSigned(constant);
    if (value <= 0) {
      fail(location, "register width must be greater than zero");
    }
    return static_cast<std::uint64_t>(value);
  }

  [[nodiscard]] std::optional<std::uint64_t>
  constantIndex(const SyntaxExpressionId id, const std::uint64_t width,
                SMLoc location) const {
    if (!isConstantExpression(id)) {
      return std::nullopt;
    }
    const auto constant = evaluateConstant(id);
    if (!isInteger(constant.type)) {
      fail(location, "index must be an integer expression");
    }
    auto value = asSigned(constant);
    if (value < 0) {
      value += static_cast<std::int64_t>(width);
    }
    if (value < 0) {
      fail(location, "index is out of bounds");
    }
    return static_cast<std::uint64_t>(value);
  }

  void analyzeBody(ArrayRef<SyntaxStatementId> source,
                   std::vector<StatementId>& destination, const bool global) {
    for (const auto id : source) {
      analyzeStatement(syntax.statements[id], destination, global);
    }
  }

  void analyzeStatement(const SyntaxStatement& statement,
                        std::vector<StatementId>& destination,
                        const bool global) {
    std::visit(
        [&](const auto& data) {
          using T = std::decay_t<decltype(data)>;
          if constexpr (!std::is_same_v<T, SyntaxGateCall> &&
                        !std::is_same_v<T, SyntaxFor> &&
                        !std::is_same_v<T, SyntaxWhile>) {
            if (insideGate) {
              fail(statement.location,
                   "gate bodies may contain only gate calls and loops over "
                   "gate calls");
            }
          }
          if constexpr (std::is_same_v<T, SyntaxStandardLibraryInclude>) {
            activateStandardLibrary(statement.location);
          } else if constexpr (std::is_same_v<T, SyntaxScalarDeclaration>) {
            analyzeScalarDeclaration(statement.location, data, destination);
          } else if constexpr (std::is_same_v<T, SyntaxAssignment>) {
            analyzeAssignment(statement.location, data, destination);
          } else if constexpr (std::is_same_v<T, SyntaxQubitDeclaration>) {
            analyzeRegisterDeclaration(statement.location, data, destination,
                                       global);
          } else if constexpr (std::is_same_v<T, SyntaxBitDeclaration>) {
            analyzeRegisterDeclaration(statement.location, data, destination,
                                       global);
          } else if constexpr (std::is_same_v<T, SyntaxMeasurement>) {
            destination.push_back(analyzeMeasurement(statement.location, data));
          } else if constexpr (std::is_same_v<T, SyntaxReset>) {
            destination.push_back(analyzeReset(statement.location, data));
          } else if constexpr (std::is_same_v<T, SyntaxBarrier>) {
            destination.push_back(analyzeBarrier(statement.location, data));
          } else if constexpr (std::is_same_v<T, SyntaxGateCall>) {
            auto applications = analyzeGateApplication(data);
            for (auto& application : applications) {
              destination.push_back(
                  addStatement(statement.location, std::move(application)));
            }
          } else if constexpr (std::is_same_v<T, SyntaxGateDefinition>) {
            if (!global) {
              fail(statement.location,
                   "gate definitions are only allowed at global scope");
            }
            analyzeGateDefinition(statement.location, data);
          } else if constexpr (std::is_same_v<T, SyntaxIf>) {
            destination.push_back(analyzeIf(statement.location, data));
          } else if constexpr (std::is_same_v<T, SyntaxFor>) {
            destination.push_back(analyzeFor(statement.location, data));
          } else if constexpr (std::is_same_v<T, SyntaxWhile>) {
            destination.push_back(analyzeWhile(statement.location, data));
          }
        },
        statement.data);
  }

  void activateStandardLibrary(SMLoc location) {
    if (program.standardLibraryIncluded) {
      fail(location, "standard library is included more than once");
    }
    for (const auto& gate : getGateCatalog()) {
      if (gate.availability != GateAvailability::StandardLibrary) {
        continue;
      }
      if (customGates.contains(gate.name) || lookup(gate.name) != nullptr) {
        fail(location,
             "standard-library gate '" + gate.name + "' is already declared");
      }
    }
    program.standardLibraryIncluded = true;
  }

  void analyzeScalarDeclaration(SMLoc location,
                                const SyntaxScalarDeclaration& declaration,
                                std::vector<StatementId>& destination) {
    const auto type = scalarType(declaration.kind);
    if (declaration.isConst) {
      if (!declaration.initializer ||
          !isConstantExpression(*declaration.initializer)) {
        fail(location, "const declaration requires a constant initializer");
      }
      auto constant = coerceConstant(evaluateConstant(*declaration.initializer),
                                     type, location);
      declare(location, declaration.identifier,
              {.kind = SymbolKind::Constant,
               .type = type,
               .constant = std::move(constant)});
      return;
    }

    const auto id = static_cast<ScalarId>(program.scalars.size());
    program.scalars.push_back(
        {.type = type, .name = declaration.identifier.str()});
    initializedScalars.push_back(false);
    scalarGenerations.push_back(0);
    declare(location, declaration.identifier,
            {.kind = SymbolKind::Scalar, .type = type, .id = id});
    ScalarDeclarationStatement typed{.scalar = id};
    if (declaration.initializer) {
      if (type == ScalarType::Bool) {
        typed.conditionInitializer = analyzeBoolValue(*declaration.initializer);
      } else {
        typed.initializer = analyzeExpression(*declaration.initializer);
      }
      initializedScalars[id] = true;
    }
    destination.push_back(addStatement(location, typed));
  }

  void markBitInitialized(const frontend::BitReference& target) {
    if (!target.dynamicIndex) {
      initializedBits[target.reg][target.index] = true;
      return;
    }
    DynamicBitFact fact{.expression = *target.dynamicIndex};
    collectDependencies(*target.dynamicIndex, fact.dependencies);
    auto& facts = dynamicBitFacts[target.reg];
    if (llvm::none_of(facts, [&](const auto& existing) {
          return existing.dependencies == fact.dependencies &&
                 sameExpression(existing.expression, fact.expression);
        })) {
      facts.push_back(std::move(fact));
    }
  }

  void analyzeAssignment(SMLoc location, const SyntaxAssignment& assignment,
                         std::vector<StatementId>& destination) {
    const auto* symbol = lookup(assignment.target.identifier);
    if (symbol != nullptr && symbol->kind == SymbolKind::Scalar) {
      if (assignment.target.index) {
        fail(location, "scalar assignments cannot have an index");
      }
      ScalarAssignmentStatement typed{.scalar = symbol->id};
      if (symbol->type == ScalarType::Bool) {
        typed.condition = analyzeBoolValue(assignment.value);
      } else {
        typed.value = analyzeExpression(assignment.value);
      }
      initializedScalars[symbol->id] = true;
      ++scalarGenerations[symbol->id];
      destination.push_back(addStatement(location, typed));
      return;
    }
    if (symbol == nullptr || symbol->kind != SymbolKind::Register ||
        program.registers[symbol->id].kind != RegisterKind::Bit) {
      fail(location, "cannot assign to '" + assignment.target.identifier + "'");
    }
    auto targets = resolveBits(assignment.target);
    if (targets.size() > 1) {
      const auto& value = syntax.expressions[assignment.value];
      if (value.kind != Expr::Kind::Identifier) {
        fail(location,
             "whole-register bit assignment requires a bit-register value");
      }
      auto sourceBits = resolveBits(
          {.location = value.location, .identifier = value.identifier});
      if (sourceBits.size() != targets.size()) {
        fail(location, "bit-register assignment widths must match");
      }
      for (const auto& source : sourceBits) {
        ensureBitInitialized(source, value.location);
      }
      for (const auto [target, source] : llvm::zip_equal(targets, sourceBits)) {
        const auto condition =
            addCondition({.kind = ConditionKind::Bit,
                          .location = sourceLocation(sources, value.location),
                          .bit = source});
        markBitInitialized(target);
        destination.push_back(
            addStatement(location, BitAssignmentStatement{.target = target,
                                                          .value = condition}));
      }
      return;
    }
    const auto value = analyzeBoolValue(assignment.value);
    markBitInitialized(targets.front());
    destination.push_back(addStatement(
        location, BitAssignmentStatement{.target = std::move(targets.front()),
                                         .value = value}));
  }

  template <class Declaration>
  void analyzeRegisterDeclaration(SMLoc location,
                                  const Declaration& declaration,
                                  std::vector<StatementId>& destination,
                                  const bool global) {
    constexpr bool isQubit =
        std::is_same_v<Declaration, SyntaxQubitDeclaration>;
    if constexpr (isQubit) {
      if (!global) {
        fail(location, "qubits must be declared at global scope");
      }
      if (hasHardwareQubits) {
        fail(location,
             "mixing physical and declared qubits is not supported by the QC "
             "target");
      }
      hasVirtualQubits = true;
    } else if (declaration.output && !global) {
      fail(location, "outputs must be declared at global scope");
    }
    const auto width = constantWidth(declaration.size, location);
    const auto id = static_cast<RegisterId>(program.registers.size());
    const bool output = [&] {
      if constexpr (isQubit) {
        return false;
      } else {
        return declaration.output;
      }
    }();
    program.registers.push_back(
        {.kind = isQubit ? RegisterKind::Qubit : RegisterKind::Bit,
         .name = declaration.identifier.str(),
         .width = width,
         .location = sourceLocation(sources, location)});
    initializedBits.emplace_back(width, false);
    dynamicBitFacts.emplace_back();
    declare(location, declaration.identifier,
            {.kind = SymbolKind::Register, .id = id});
    if (!isQubit && global) {
      bitRegisters.push_back(id);
      if (output || program.openQASM2) {
        explicitOutputs.push_back(id);
      }
    }
    destination.push_back(
        addStatement(location, DeclarationStatement{.reg = id}));
    if constexpr (!isQubit) {
      if (declaration.initializer) {
        if (width != 1) {
          fail(location,
               "bit expression initializers require a scalar bit declaration");
        }
        analyzeAssignment(
            location,
            SyntaxAssignment{
                .target =
                    SyntaxBitReference{.location = location,
                                       .identifier = declaration.identifier},
                .value = *declaration.initializer},
            destination);
      }
    }
  }

  void analyzeGateDefinition(SMLoc location,
                             const SyntaxGateDefinition& declaration) {
    if (customGates.contains(declaration.identifier) ||
        lookup(declaration.identifier) != nullptr) {
      fail(location,
           "gate '" + declaration.identifier + "' is already declared");
    }
    if (const auto* catalog = lookupGate(declaration.identifier);
        catalog != nullptr && isGateAvailable(*catalog)) {
      fail(location,
           "gate '" + declaration.identifier + "' is already declared");
    }
    customGates[declaration.identifier] = {
        .parameterCount = declaration.parameters.size(),
        .qubitCount = declaration.qubits.size()};
    GateDefinition definition{.name = declaration.identifier.str(),
                              .parameterCount = declaration.parameters.size(),
                              .qubitCount = declaration.qubits.size(),
                              .location = sourceLocation(sources, location)};
    scopes.emplace_back();
    for (const auto [index, parameter] :
         llvm::enumerate(declaration.parameters)) {
      declare(location, parameter,
              {.kind = SymbolKind::GateParameter,
               .type = ScalarType::Float,
               .id = static_cast<std::uint32_t>(index)});
    }
    for (const auto [index, qubit] : llvm::enumerate(declaration.qubits)) {
      declare(location, qubit,
              {.kind = SymbolKind::GateQubit,
               .id = static_cast<std::uint32_t>(index)});
    }
    insideGate = true;
    analyzeBody(declaration.body, definition.body, /*global=*/false);
    insideGate = false;
    scopes.pop_back();
    program.gates.push_back(std::move(definition));
  }

  [[nodiscard]] StatementId
  analyzeMeasurement(SMLoc location, const SyntaxMeasurement& measurement) {
    auto qubits = resolveQubitOperand(measurement.source);
    validateDynamicDispatchCost(location, qubits);
    if (!measurement.target) {
      return addStatement(location,
                          MeasurementStatement{.qubits = std::move(qubits)});
    }
    const auto* destination = lookup(measurement.target->identifier);
    if (destination != nullptr && destination->kind == SymbolKind::Scalar) {
      if (measurement.target->index || destination->type != ScalarType::Bool) {
        fail(location, "measurement assignment requires a bool scalar or bit "
                       "register destination");
      }
      if (qubits.size() != 1) {
        fail(location, "bool measurement assignment requires one qubit");
      }
      const auto condition =
          addCondition({.kind = ConditionKind::Measurement,
                        .location = sourceLocation(sources, location),
                        .measurement = qubits.front()});
      initializedScalars[destination->id] = true;
      ++scalarGenerations[destination->id];
      return addStatement(location,
                          ScalarAssignmentStatement{.scalar = destination->id,
                                                    .condition = condition});
    }
    auto targets = resolveBits(*measurement.target);
    if (targets.size() != qubits.size()) {
      fail(location,
           "measurement target and qubit operand must have the same width");
    }
    for (const auto& target : targets) {
      markBitInitialized(target);
    }
    return addStatement(location,
                        MeasurementStatement{.targets = std::move(targets),
                                             .qubits = std::move(qubits)});
  }

  [[nodiscard]] StatementId analyzeReset(SMLoc location,
                                         const SyntaxReset& reset) {
    auto qubits = resolveQubitOperand(reset.operand);
    validateDynamicDispatchCost(location, qubits);
    return addStatement(location, ResetStatement{.qubits = std::move(qubits)});
  }

  [[nodiscard]] StatementId analyzeBarrier(SMLoc location,
                                           const SyntaxBarrier& barrier) {
    std::vector<QubitReference> qubits;
    if (barrier.operands.empty()) {
      for (const auto [registerId, declaration] :
           llvm::enumerate(program.registers)) {
        if (declaration.kind != RegisterKind::Qubit) {
          continue;
        }
        for (std::uint64_t index = 0; index < declaration.width; ++index) {
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
      auto selection = resolveQubitOperand(operand);
      qubits.insert(qubits.end(), selection.begin(), selection.end());
    }
    validateDynamicDispatchCost(location, qubits);
    return addStatement(location,
                        BarrierStatement{.qubits = std::move(qubits)});
  }

  [[nodiscard]] StatementId analyzeIf(SMLoc location,
                                      const SyntaxIf& conditional) {
    IfStatement result{.condition = analyzeCondition(conditional.condition)};
    const auto beforeBitsInitialized = initializedBits;
    const auto beforeInitialized = initializedScalars;
    const auto beforeGenerations = scalarGenerations;
    const auto beforeDynamicBitFacts = dynamicBitFacts;
    scopes.emplace_back();
    analyzeBody(conditional.thenStatements, result.thenStatements,
                /*global=*/false);
    const auto afterThenBitsInitialized = initializedBits;
    const auto afterThenInitialized = initializedScalars;
    const auto afterThenGenerations = scalarGenerations;
    const auto afterThenDynamicBitFacts = dynamicBitFacts;
    scopes.pop_back();

    restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                       beforeGenerations);
    restoreDynamicFactsPrefix(beforeDynamicBitFacts);
    scopes.emplace_back();
    analyzeBody(conditional.elseStatements, result.elseStatements,
                /*global=*/false);
    const auto afterElseBitsInitialized = initializedBits;
    const auto afterElseInitialized = initializedScalars;
    const auto afterElseGenerations = scalarGenerations;
    const auto afterElseDynamicBitFacts = dynamicBitFacts;
    scopes.pop_back();

    restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                       beforeGenerations);
    restoreDynamicFactsPrefix(beforeDynamicBitFacts);
    const auto knownCondition = constantCondition(conditional.condition);
    for (std::size_t reg = 0; reg < beforeBitsInitialized.size(); ++reg) {
      for (std::size_t bit = 0; bit < beforeBitsInitialized[reg].size();
           ++bit) {
        initializedBits[reg][bit] =
            knownCondition
                ? (*knownCondition ? afterThenBitsInitialized[reg][bit]
                                   : afterElseBitsInitialized[reg][bit])
                : afterThenBitsInitialized[reg][bit] &&
                      afterElseBitsInitialized[reg][bit];
      }
    }
    for (std::size_t reg = 0; reg < beforeDynamicBitFacts.size(); ++reg) {
      if (knownCondition) {
        dynamicBitFacts[reg] = *knownCondition ? afterThenDynamicBitFacts[reg]
                                               : afterElseDynamicBitFacts[reg];
        continue;
      }
      dynamicBitFacts[reg].clear();
      for (const auto& thenFact : afterThenDynamicBitFacts[reg]) {
        if (llvm::any_of(
                afterElseDynamicBitFacts[reg], [&](const auto& elseFact) {
                  return thenFact.dependencies == elseFact.dependencies &&
                         sameExpression(thenFact.expression,
                                        elseFact.expression);
                })) {
          dynamicBitFacts[reg].push_back(thenFact);
        }
      }
    }
    for (std::size_t scalar = 0; scalar < beforeInitialized.size(); ++scalar) {
      initializedScalars[scalar] =
          knownCondition
              ? (*knownCondition ? afterThenInitialized[scalar]
                                 : afterElseInitialized[scalar])
              : afterThenInitialized[scalar] && afterElseInitialized[scalar];
      scalarGenerations[scalar] =
          knownCondition ? (*knownCondition ? afterThenGenerations[scalar]
                                            : afterElseGenerations[scalar])
                         : std::max(afterThenGenerations[scalar],
                                    afterElseGenerations[scalar]);
    }
    return addStatement(location, std::move(result));
  }

  [[nodiscard]] StatementId analyzeFor(SMLoc location, const SyntaxFor& loop) {
    ForStatement result{.start = analyzeExpression(loop.start),
                        .step = analyzeExpression(loop.step),
                        .stop = analyzeExpression(loop.stop)};
    for (const auto expression : {result.start, result.step, result.stop}) {
      if (!isInteger(program.expressions[expression].type)) {
        fail(location, "for-loop ranges require integer expressions");
      }
    }
    const auto constantIsZero = [](const Constant& value) {
      return value.type == ScalarType::Uint
                 ? std::get<std::uint64_t>(value.value) == 0
                 : std::get<std::int64_t>(value.value) == 0;
    };
    if (isConstantExpression(loop.step) &&
        constantIsZero(evaluateConstant(loop.step))) {
      fail(location, "for-loop range step must not be zero");
    }

    const auto beforeBitsInitialized = initializedBits;
    const auto beforeInitialized = initializedScalars;
    const auto beforeGenerations = scalarGenerations;
    const auto beforeDynamicBitFacts = dynamicBitFacts;
    scopes.emplace_back();
    const auto scalar = static_cast<ScalarId>(program.scalars.size());
    const auto type = loop.isUnsigned ? ScalarType::Uint : ScalarType::Int;
    program.scalars.push_back(
        {.type = type, .name = loop.inductionVariable.str()});
    initializedScalars.push_back(true);
    scalarGenerations.push_back(0);
    declare(
        location, loop.inductionVariable,
        {.kind = insideGate ? SymbolKind::GateLocalScalar : SymbolKind::Scalar,
         .type = type,
         .id = scalar});
    result.inductionVariable = scalar;
    analyzeBody(loop.body, result.body, /*global=*/false);
    const auto afterBodyBitsInitialized = initializedBits;
    const auto afterBodyInitialized = initializedScalars;
    const auto afterBodyGenerations = scalarGenerations;
    const auto afterBodyDynamicBitFacts = dynamicBitFacts;
    scopes.pop_back();
    restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                       beforeGenerations);
    restoreDynamicFactsPrefix(beforeDynamicBitFacts);
    if (isConstantExpression(loop.start) && isConstantExpression(loop.step) &&
        isConstantExpression(loop.stop)) {
      const auto startConstant = evaluateConstant(loop.start);
      const auto stepConstant = evaluateConstant(loop.step);
      const auto stopConstant = evaluateConstant(loop.stop);
      const bool unsignedEndpoints = startConstant.type == ScalarType::Uint ||
                                     stopConstant.type == ScalarType::Uint;
      const auto compareRangeValues = [&](const Constant& lhs,
                                          const Constant& rhs) {
        if (!unsignedEndpoints) {
          const auto left = std::get<std::int64_t>(lhs.value);
          const auto right = std::get<std::int64_t>(rhs.value);
          return left < right ? -1 : left > right ? 1 : 0;
        }
        const auto asUnsigned = [](const Constant& value) {
          return value.type == ScalarType::Uint
                     ? std::get<std::uint64_t>(value.value)
                     : static_cast<std::uint64_t>(
                           std::get<std::int64_t>(value.value));
        };
        const auto left = asUnsigned(lhs);
        const auto right = asUnsigned(rhs);
        return left < right ? -1 : left > right ? 1 : 0;
      };
      const bool positiveStep = stepConstant.type == ScalarType::Uint ||
                                std::get<std::int64_t>(stepConstant.value) > 0;
      const auto endpointOrder =
          compareRangeValues(startConstant, stopConstant);
      const bool nonempty =
          positiveStep ? endpointOrder <= 0 : endpointOrder >= 0;
      if (nonempty) {
        for (std::size_t reg = 0; reg < beforeBitsInitialized.size(); ++reg) {
          initializedBits[reg] = afterBodyBitsInitialized[reg];
        }
        for (std::size_t scalar = 0; scalar < beforeInitialized.size();
             ++scalar) {
          initializedScalars[scalar] = afterBodyInitialized[scalar];
          scalarGenerations[scalar] = afterBodyGenerations[scalar];
        }
        for (std::size_t reg = 0; reg < beforeDynamicBitFacts.size(); ++reg) {
          dynamicBitFacts[reg] = afterBodyDynamicBitFacts[reg];
        }
      }
    }
    return addStatement(location, std::move(result));
  }

  [[nodiscard]] StatementId analyzeWhile(SMLoc location,
                                         const SyntaxWhile& loop) {
    WhileStatement result{.condition = analyzeCondition(loop.condition)};
    const auto beforeBitsInitialized = initializedBits;
    const auto beforeInitialized = initializedScalars;
    const auto beforeGenerations = scalarGenerations;
    const auto beforeDynamicBitFacts = dynamicBitFacts;
    scopes.emplace_back();
    analyzeBody(loop.body, result.body, /*global=*/false);
    scopes.pop_back();
    const auto afterBodyGenerations = scalarGenerations;
    restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                       beforeGenerations);
    restoreDynamicFactsPrefix(beforeDynamicBitFacts);
    for (std::size_t scalar = 0; scalar < beforeGenerations.size(); ++scalar) {
      scalarGenerations[scalar] =
          std::max(beforeGenerations[scalar], afterBodyGenerations[scalar]);
    }
    return addStatement(location, std::move(result));
  }

  [[nodiscard]] ConditionId
  analyzeCondition(const SyntaxExpressionId syntaxId) {
    const auto& condition = syntax.expressions[syntaxId];
    ConditionExpression typed{.location =
                                  sourceLocation(sources, condition.location)};
    if (isConstantExpression(syntaxId)) {
      const auto constant = evaluateConstant(syntaxId);
      if (constant.type != ScalarType::Bool) {
        fail(condition.location, "condition must have bool type");
      }
      typed.kind = ConditionKind::Literal;
      typed.literal = std::get<bool>(constant.value);
      return addCondition(std::move(typed));
    }
    switch (condition.kind) {
    case Expr::Kind::Identifier: {
      const auto* symbol = lookup(condition.identifier);
      if (symbol == nullptr) {
        fail(condition.location,
             "unknown condition identifier '" + condition.identifier + "'");
      }
      if (symbol->kind == SymbolKind::Scalar &&
          symbol->type == ScalarType::Bool) {
        if (!initializedScalars.at(symbol->id)) {
          fail(condition.location,
               "scalar '" + condition.identifier + "' is uninitialized");
        }
        typed.kind = ConditionKind::Scalar;
        typed.scalar = symbol->id;
        break;
      }
      if (symbol->kind != SymbolKind::Register ||
          program.registers[symbol->id].kind != RegisterKind::Bit) {
        fail(condition.location, "identifier '" + condition.identifier +
                                     "' is not bool or a classical bit");
      }
      auto bits = resolveBits(
          {.location = condition.location, .identifier = condition.identifier});
      if (bits.size() != 1) {
        fail(condition.location,
             "condition must select exactly one classical bit");
      }
      ensureBitInitialized(bits.front(), condition.location);
      typed.kind = ConditionKind::Bit;
      typed.bit = bits.front();
      break;
    }
    case Expr::Kind::Index: {
      auto bits = resolveBits({.location = condition.location,
                               .identifier = condition.identifier,
                               .index = condition.lhs});
      if (bits.size() != 1) {
        fail(condition.location,
             "condition must select exactly one classical bit");
      }
      ensureBitInitialized(bits.front(), condition.location);
      typed.kind = ConditionKind::Bit;
      typed.bit = bits.front();
      break;
    }
    case Expr::Kind::Measurement: {
      auto qubits =
          resolveQubitOperand({.location = condition.location,
                               .identifier = condition.identifier,
                               .index = condition.lhs,
                               .hardwareQubit = condition.hardwareQubit});
      if (qubits.size() != 1) {
        fail(condition.location,
             "measurement condition must select exactly one qubit");
      }
      typed.kind = ConditionKind::Measurement;
      typed.measurement = qubits.front();
      break;
    }
    case Expr::Kind::Not:
      typed.kind = ConditionKind::Not;
      typed.lhs = analyzeCondition(*condition.lhs);
      break;
    case Expr::Kind::And:
    case Expr::Kind::Or:
      typed.kind = condition.kind == Expr::Kind::And ? ConditionKind::And
                                                     : ConditionKind::Or;
      typed.lhs = analyzeCondition(*condition.lhs);
      typed.rhs = analyzeCondition(*condition.rhs);
      break;
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
        const auto expected = evaluateConstant(*condition.rhs);
        if (!isInteger(expected.type) ||
            (expected.type == ScalarType::Int &&
             std::get<std::int64_t>(expected.value) < 0)) {
          fail(condition.location,
               "OpenQASM 2 register conditions require an unsigned integer");
        }
        const auto expectedValue =
            expected.type == ScalarType::Uint
                ? std::get<std::uint64_t>(expected.value)
                : static_cast<std::uint64_t>(
                      std::get<std::int64_t>(expected.value));
        auto bits = resolveBits({.location = lhsSyntax.location,
                                 .identifier = lhsSyntax.identifier});
        for (const auto& bit : bits) {
          ensureBitInitialized(bit, condition.location);
        }
        const bool fits =
            bits.size() >= 64 || (expectedValue >> bits.size()) == 0;
        auto result = addCondition(
            {.kind = ConditionKind::Literal,
             .location = sourceLocation(sources, condition.location),
             .literal = fits});
        for (const auto [index, bit] : llvm::enumerate(bits)) {
          auto bitCondition = addCondition(
              {.kind = ConditionKind::Bit,
               .location = sourceLocation(sources, condition.location),
               .bit = bit});
          const bool expectedBit =
              index < 64 && ((expectedValue >> index) & 1U) != 0;
          if (!expectedBit) {
            bitCondition = addCondition(
                {.kind = ConditionKind::Not,
                 .location = sourceLocation(sources, condition.location),
                 .lhs = bitCondition});
          }
          result = addCondition(
              {.kind = ConditionKind::And,
               .location = sourceLocation(sources, condition.location),
               .lhs = result,
               .rhs = bitCondition});
        }
        return result;
      }
      typed.kind = ConditionKind::Comparison;
      typed.comparisonLhs = analyzeExpression(*condition.lhs);
      typed.comparisonRhs = analyzeExpression(*condition.rhs);
      const auto lhsType = program.expressions[typed.comparisonLhs].type;
      const auto rhsType = program.expressions[typed.comparisonRhs].type;
      const bool boolComparison =
          lhsType == ScalarType::Bool || rhsType == ScalarType::Bool;
      if (boolComparison &&
          (lhsType != ScalarType::Bool || rhsType != ScalarType::Bool ||
           (condition.kind != Expr::Kind::Equal &&
            condition.kind != Expr::Kind::NotEqual))) {
        fail(condition.location,
             "bool values only support equality comparisons with bool values");
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
    case Expr::Kind::Neg:
    case Expr::Kind::BitNot:
    case Expr::Kind::Add:
    case Expr::Kind::Sub:
    case Expr::Kind::Mul:
    case Expr::Kind::Div:
    case Expr::Kind::ArcCos:
    case Expr::Kind::ArcSin:
    case Expr::Kind::ArcTan:
    case Expr::Kind::Cos:
    case Expr::Kind::Exp:
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
      fail(condition.location, "condition must have bool type");
    }
    return addCondition(std::move(typed));
  }

  [[nodiscard]] std::vector<GateApplication>
  analyzeGateApplication(const SyntaxGateCall& call) {
    std::string callee = call.identifier.str();
    const GateCatalogEntry* standard = lookupGate(callee);
    auto custom = customGates.find(callee);
    std::uint64_t compatibilityControls = 0;
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
    if (standard == nullptr && custom == customGates.end()) {
      fail(call.location,
           "No OpenQASM definition found for gate '" + call.identifier + "'.");
    }

    const auto signature =
        standard != nullptr
            ? GateSignature{.parameterCount = standard->parameterCount,
                            .qubitCount = standard->qubitCount(),
                            .variadicControls = standard->variadicControls}
            : custom->second;
    if (signature.parameterCount != call.parameters.size()) {
      fail(call.location,
           "Invalid number of parameters for gate '" + call.identifier + "'.");
    }
    std::vector<ExpressionId> parameters;
    parameters.reserve(call.parameters.size());
    for (const auto expression : call.parameters) {
      const auto parameter = analyzeExpression(expression);
      if (program.expressions[parameter].type == ScalarType::Bool) {
        fail(call.location, "gate parameters require numeric expressions");
      }
      parameters.push_back(parameter);
    }

    std::vector<GateModifier> modifiers;
    std::size_t addedControls = compatibilityControls;
    if (addedControls > call.operands.size()) {
      fail(call.location, "Invalid number of qubit operands for gate '" +
                              call.identifier + "'.");
    }
    for (const auto& modifier : call.modifiers) {
      switch (modifier.kind) {
      case Modifier::Kind::Inv:
        modifiers.push_back({.kind = ModifierKind::Inv});
        break;
      case Modifier::Kind::Pow:
        if (!modifier.argument) {
          fail(call.location, "pow modifier requires an argument");
        }
        {
          const auto operand = analyzeExpression(*modifier.argument);
          if (program.expressions[operand].type == ScalarType::Bool) {
            fail(call.location, "pow modifier requires a numeric argument");
          }
          modifiers.push_back({.kind = ModifierKind::Pow, .operand = operand});
        }
        break;
      case Modifier::Kind::Ctrl:
      case Modifier::Kind::NegCtrl: {
        std::uint64_t count = 1;
        std::optional<ExpressionId> operand;
        if (modifier.argument) {
          if (!isConstantExpression(*modifier.argument)) {
            fail(call.location,
                 "gate control count must be a constant integer");
          }
          const auto constant = evaluateConstant(*modifier.argument);
          if (!isInteger(constant.type) || asSigned(constant) <= 0) {
            fail(call.location, "gate control count must be positive");
          }
          count = static_cast<std::uint64_t>(asSigned(constant));
          operand = addConstant({.type = ScalarType::Int,
                                 .value = static_cast<std::int64_t>(count)});
        }
        if (count > call.operands.size() - addedControls) {
          fail(call.location, "Invalid number of qubit operands for gate '" +
                                  call.identifier + "'.");
        }
        addedControls += static_cast<std::size_t>(count);
        modifiers.push_back({.kind = modifier.kind == Modifier::Kind::Ctrl
                                         ? ModifierKind::Ctrl
                                         : ModifierKind::NegCtrl,
                             .operand = operand});
        break;
      }
      }
    }
    if (compatibilityControls != 0) {
      modifiers.insert(
          modifiers.begin(),
          {.kind = ModifierKind::Ctrl,
           .operand = addConstant(
               {.type = ScalarType::Int,
                .value = static_cast<std::int64_t>(compatibilityControls)})});
    }

    const auto baseOperandCount = call.operands.size() - addedControls;
    if (signature.variadicControls ? baseOperandCount < signature.qubitCount
                                   : baseOperandCount != signature.qubitCount) {
      fail(call.location, "Invalid number of qubit operands for gate '" +
                              call.identifier + "'.");
    }

    std::size_t emittedOperandCount = call.operands.size();
    if (standard != nullptr && standard->variadicControls) {
      std::size_t activeBaseOperands = baseOperandCount;
      if (standard->name == "mcx_vchain") {
        if (baseOperandCount < 5) {
          fail(call.location,
               "mcx_vchain requires controls, a target, and ancillas");
        }
        const auto ancillas = ((baseOperandCount + 1) / 2) - 2;
        activeBaseOperands -= ancillas;
      } else if (standard->name == "mcx_recursive" && baseOperandCount > 5) {
        --activeBaseOperands;
      }
      if (activeBaseOperands <= standard->targetCount) {
        fail(call.location,
             "Invalid number of controls for gate '" + call.identifier + "'.");
      }
      const auto intrinsicControls = activeBaseOperands - standard->targetCount;
      modifiers.push_back(
          {.kind = ModifierKind::Ctrl,
           .operand = addConstant(
               {.type = ScalarType::Int,
                .value = static_cast<std::int64_t>(intrinsicControls)})});
      callee = standard->primitive.str();
      emittedOperandCount = addedControls + activeBaseOperands;
    }

    std::vector<std::vector<QubitReference>> selections;
    std::size_t broadcastWidth = 1;
    for (const auto& operand : call.operands) {
      auto selection = resolveQubitOperand(operand);
      if (selection.size() > 1) {
        if (broadcastWidth != 1 && broadcastWidth != selection.size()) {
          fail(call.location,
               "all broadcasting operands must have the same width");
        }
        broadcastWidth = selection.size();
      }
      selections.push_back(std::move(selection));
    }

    std::vector<GateApplication> applications;
    applications.reserve(broadcastWidth);
    for (std::size_t index = 0; index < broadcastWidth; ++index) {
      GateApplication application{
          .callee = callee, .parameters = parameters, .modifiers = modifiers};
      for (const auto& selection :
           ArrayRef(selections).take_front(emittedOperandCount)) {
        application.qubits.push_back(
            selection[selection.size() == 1 ? 0 : index]);
      }
      for (const auto [position, qubit] : llvm::enumerate(application.qubits)) {
        if (llvm::is_contained(
                ArrayRef(application.qubits).take_front(position), qubit)) {
          fail(call.location,
               "gate operands must not reference the same qubit more than "
               "once");
        }
      }
      validateDynamicDispatchCost(call.location, application.qubits);
      applications.push_back(std::move(application));
    }
    return applications;
  }

  [[nodiscard]] std::vector<QubitReference>
  resolveQubitOperand(const SyntaxOperand& operand) {
    if (operand.hardwareQubit) {
      if (insideGate) {
        fail(operand.location,
             "hardware qubits are not allowed in gate definitions");
      }
      if (hasVirtualQubits) {
        fail(operand.location,
             "mixing physical and declared qubits is not supported by the QC "
             "target");
      }
      hasHardwareQubits = true;
      hardwareQubits.insert(*operand.hardwareQubit);
      return {{.kind = QubitReferenceKind::Hardware,
               .index = *operand.hardwareQubit}};
    }
    const auto* symbol = lookup(operand.identifier);
    if (insideGate) {
      if (symbol == nullptr || symbol->kind != SymbolKind::GateQubit) {
        fail(operand.location,
             "unknown gate-local qubit '" + operand.identifier + "'");
      }
      if (operand.index) {
        fail(operand.location, "gate-local qubits cannot be indexed");
      }
      return {{.kind = QubitReferenceKind::GateArgument, .symbol = symbol->id}};
    }
    if (symbol == nullptr || symbol->kind != SymbolKind::Register ||
        program.registers[symbol->id].kind != RegisterKind::Qubit) {
      fail(operand.location,
           "unknown qubit register '" + operand.identifier + "'");
    }
    const auto reg = static_cast<RegisterId>(symbol->id);
    const auto width = program.registers[reg].width;
    if (!operand.index) {
      std::vector<QubitReference> selection;
      selection.reserve(width);
      for (std::uint64_t index = 0; index < width; ++index) {
        selection.push_back({.kind = QubitReferenceKind::Register,
                             .symbol = reg,
                             .index = index});
      }
      return selection;
    }
    if (const auto index =
            constantIndex(*operand.index, width, operand.location)) {
      if (*index >= width) {
        fail(operand.location, "qubit index is out of bounds");
      }
      return {{.kind = QubitReferenceKind::Register,
               .symbol = reg,
               .index = *index}};
    }
    const auto dynamic = analyzeExpression(*operand.index);
    if (!isInteger(program.expressions[dynamic].type)) {
      fail(operand.location, "qubit index must be an integer expression");
    }
    return {{.kind = QubitReferenceKind::Register,
             .symbol = reg,
             .dynamicIndex = dynamic}};
  }

  [[nodiscard]] std::vector<frontend::BitReference>
  resolveBits(const SyntaxBitReference& reference) {
    const auto* symbol = lookup(reference.identifier);
    if (symbol == nullptr || symbol->kind != SymbolKind::Register ||
        program.registers[symbol->id].kind == RegisterKind::Qubit) {
      fail(reference.location,
           "unknown classical register '" + reference.identifier + "'");
    }
    const auto reg = static_cast<RegisterId>(symbol->id);
    const auto width = program.registers[reg].width;
    if (!reference.index) {
      std::vector<frontend::BitReference> result;
      result.reserve(width);
      for (std::uint64_t index = 0; index < width; ++index) {
        result.push_back({.reg = reg, .index = index});
      }
      return result;
    }
    if (const auto index =
            constantIndex(*reference.index, width, reference.location)) {
      if (*index >= width) {
        fail(reference.location, "classical bit index is out of bounds");
      }
      return {{.reg = reg, .index = *index}};
    }
    const auto dynamic = analyzeExpression(*reference.index);
    if (!isInteger(program.expressions[dynamic].type)) {
      fail(reference.location,
           "classical bit index must be an integer expression");
    }
    return {{.reg = reg, .dynamicIndex = dynamic}};
  }

  void ensureBitInitialized(const frontend::BitReference& bit,
                            SMLoc location) const {
    if (bit.dynamicIndex) {
      if (llvm::all_of(initializedBits[bit.reg],
                       [](const bool initialized) { return initialized; })) {
        return;
      }
      std::vector<std::pair<ScalarId, std::uint64_t>> dependencies;
      collectDependencies(*bit.dynamicIndex, dependencies);
      if (llvm::any_of(dynamicBitFacts[bit.reg], [&](const auto& fact) {
            return fact.dependencies == dependencies &&
                   sameExpression(fact.expression, *bit.dynamicIndex);
          })) {
        return;
      }
      fail(location, "dynamic classical index may read an uninitialized bit");
    }
    if (!initializedBits[bit.reg][bit.index]) {
      fail(location, "classical condition bit has not been initialized");
    }
  }

  void finalizeOutputs() {
    program.outputs = explicitOutputs.empty() ? bitRegisters : explicitOutputs;
    for (const auto reg : program.outputs) {
      if (llvm::any_of(initializedBits[reg],
                       [](const bool initialized) { return !initialized; })) {
        throw SemanticError({.location = program.registers[reg].location,
                             .message = "Output register '" +
                                        program.registers[reg].name +
                                        "' is not fully initialized."});
      }
    }
  }
};

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
  return {.filename = buffer->getBufferIdentifier().str(),
          .line = line,
          .column = column};
}

AnalysisResult analyzeSyntaxProgram(const SyntaxProgram& syntax,
                                    const llvm::SourceMgr& sources,
                                    const FrontendOptions& options) {
  return SemanticAnalyzer(syntax, sources, options).run();
}

} // namespace mlir::oq3::frontend::detail

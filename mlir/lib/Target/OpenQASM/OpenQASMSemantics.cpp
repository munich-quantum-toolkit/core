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
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
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

class SemanticError final : public std::runtime_error {
public:
  Diagnostic diagnostic;

  explicit SemanticError(Diagnostic value)
      : std::runtime_error(value.message), diagnostic(std::move(value)) {}
};

struct Constant {
  ScalarType type = ScalarType::Int;
  std::variant<bool, int64_t, uint64_t, double> value = int64_t{0};
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
  return std::visit([](const auto value) { return static_cast<double>(value); },
                    constant.value);
}

[[nodiscard]] static int64_t asSigned(const Constant& constant) {
  if (constant.type == ScalarType::Uint) {
    const auto value = std::get<uint64_t>(constant.value);
    if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      throw std::overflow_error("unsigned value does not fit in signed i64");
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
           destination == ScalarType::Angle ||
           (destination == ScalarType::Uint &&
            std::get<int64_t>(initializer.value) >= 0);
  case ScalarType::Uint:
    return destination == ScalarType::Float ||
           destination == ScalarType::Angle ||
           (destination == ScalarType::Int &&
            std::get<uint64_t>(initializer.value) <=
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  case ScalarType::Float:
    return destination == ScalarType::Angle;
  case ScalarType::Angle:
    return destination == ScalarType::Float;
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
    try {
      analyzeVersion();
      validateExpressionDepth();
      analyzeTopLevelBody();
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
    std::vector<std::pair<uint64_t, uint64_t>> dependencies;
  };
  using BitInitialization = std::vector<bool>;
  using DynamicBitFactSet = std::vector<DynamicBitFact>;

  // The analyzed syntax and source manager are mandatory and outlive this run.
  const SyntaxProgram&
      syntax; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
  const llvm::SourceMgr&
      sources; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
  FrontendOptions options;
  TypedProgram program;
  SmallVector<llvm::StringMap<Symbol>> scopes;
  llvm::StringMap<GateSignature> customGates;
  std::vector<std::shared_ptr<BitInitialization>> initializedBits;
  std::vector<std::shared_ptr<DynamicBitFactSet>> dynamicBitFacts;
  std::vector<bool> initializedScalars;
  std::vector<uint64_t> scalarGenerations;
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

  [[noreturn]] void fail(SMLoc location, const Twine& message) const {
    throw SemanticError(
        {.location = getSourceLocation(location), .message = message.str()});
  }

  void validateExpressionDepth() const {
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
        fail(expression.location,
             Twine("expression depth exceeds the limit of ") +
                 Twine(static_cast<unsigned>(EXPRESSION_DEPTH_LIMIT)));
      }
      depths[id] = depth;
    }
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
         (belongsToStdGates(catalog->availability) &&
          program.stdGatesIncluded) ||
         (belongsToQELib1(catalog->availability) && program.qelib1Included) ||
         (program.openQASM2 && catalog->name == "CX"));
    if (builtinConstant(name) ||
        (scopes.size() == 1 &&
         (customGates.contains(name) || catalogNameReserved))) {
      fail(location, "identifier '" + name + "' is already declared");
    }
    if (!scopes.back().insert({name, symbol}).second) {
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
    return (belongsToStdGates(gate.availability) && program.stdGatesIncluded) ||
           (belongsToQELib1(gate.availability) && program.qelib1Included) ||
           (program.openQASM2 && gate.name == "CX");
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
    llvm::StringMap<size_t> gateIndices;
    for (const auto [index, gate] : llvm::enumerate(program.gates)) {
      gateIndices[gate.name] = index;
    }
    enum class VisitState : uint8_t { Unvisited, Active, Complete };
    std::vector states(program.gates.size(), VisitState::Unvisited);
    std::vector<size_t> dependencyDepths(program.gates.size());
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
    const auto visit = [&](auto&& self, const size_t index) -> size_t {
      if (states[index] == VisitState::Complete) {
        return dependencyDepths[index];
      }
      states[index] = VisitState::Active;
      size_t dependencyDepth = 1;
      visitApplications(
          visitApplications, program.gates[index].body,
          [&](const GateApplication& application,
              const SourceLocation& location) {
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
            const auto calleeDepth = self(self, callee->second);
            if (calleeDepth >= GATE_DEPENDENCY_DEPTH_LIMIT) {
              throw SemanticError(
                  {.location = location,
                   .message =
                       "custom gate dependency depth exceeds the limit of " +
                       std::to_string(GATE_DEPENDENCY_DEPTH_LIMIT)});
            }
            dependencyDepth = std::max(dependencyDepth, calleeDepth + 1);
          });
      states[index] = VisitState::Complete;
      dependencyDepths[index] = dependencyDepth;
      return dependencyDepth;
    };
    for (size_t index = 0; index < program.gates.size(); ++index) {
      if (states[index] == VisitState::Unvisited) {
        std::ignore = visit(visit, index);
      }
    }
  }

  [[nodiscard]] StatementId addStatement(SMLoc location, StatementData data) {
    if (program.statements.size() >= TYPED_STATEMENT_LIMIT) {
      fail(location, Twine("typed OpenQASM program exceeds the statement "
                           "limit of ") +
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
    return addExpression({.kind = ExpressionKind::Constant,
                          .type = constant.type,
                          .constant = constant.value});
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

  [[nodiscard]] ExpressionId castExpression(const ExpressionId expression,
                                            const ScalarType target,
                                            const SMLoc location) {
    const auto source = program.expressions[expression].type;
    if (source == target) {
      return expression;
    }
    if (!canImplicitlyConvert(source, target)) {
      fail(location, "expression of type '" + scalarTypeName(source) +
                         "' cannot be implicitly converted to '" +
                         scalarTypeName(target) + "'");
    }
    return addExpression(
        {.kind = ExpressionKind::Cast, .type = target, .lhs = expression});
  }

  [[nodiscard]] Constant promoteConstInitializer(const Constant& initializer,
                                                 const ScalarType destination,
                                                 const SMLoc location) const {
    if (!canImplicitlyPromote(initializer, destination)) {
      fail(location, "constant initializer of type '" +
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
        return {.type = ScalarType::Int,
                .value =
                    static_cast<int64_t>(std::get<bool>(initializer.value))};
      }
      return {.type = ScalarType::Int,
              .value =
                  static_cast<int64_t>(std::get<uint64_t>(initializer.value))};
    case ScalarType::Uint:
      if (initializer.type == ScalarType::Bool) {
        return {.type = ScalarType::Uint,
                .value =
                    static_cast<uint64_t>(std::get<bool>(initializer.value))};
      }
      return {.type = ScalarType::Uint,
              .value =
                  static_cast<uint64_t>(std::get<int64_t>(initializer.value))};
    case ScalarType::Float:
      return {.type = ScalarType::Float, .value = asDouble(initializer)};
    case ScalarType::Angle:
      return {.type = ScalarType::Angle, .value = asDouble(initializer)};
    }
    llvm_unreachable("unknown scalar type");
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
      return Constant{.type = ScalarType::Angle, .value = std::numbers::pi};
    }
    if (identifier == "tau" || identifier == "τ") {
      return Constant{.type = ScalarType::Angle,
                      .value = 2.0 * std::numbers::pi};
    }
    if (identifier == "euler" || identifier == "ℇ") {
      return Constant{.type = ScalarType::Float, .value = std::numbers::e};
    }
    return std::nullopt;
  }

  [[nodiscard]] Constant evaluateConstant(const SyntaxExpressionId id) const {
    if (constantValues[id]) {
      return *constantValues[id];
    }
    const auto result = [&]() -> Constant {
      const auto& expression = syntax.expressions[id];
      switch (expression.kind) {
      case Expr::Kind::Int:
        if (!expression.wideInteger.empty()) {
          fail(expression.location,
               "integer literal exceeds 64-bit constant evaluation");
        }
        if (expression.integer <=
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          return {.type = ScalarType::Int,
                  .value = static_cast<int64_t>(expression.integer)};
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
          fail(expression.location,
               "expression is not a compile-time constant");
        }
        return *symbol->constant;
      }
      case Expr::Kind::Neg: {
        auto operand = evaluateConstant(*expression.lhs);
        if (operand.type == ScalarType::Bool) {
          fail(expression.location,
               "numeric negation requires a numeric operand");
        }
        if (operand.type == ScalarType::Float ||
            operand.type == ScalarType::Angle) {
          return {.type = operand.type, .value = -asDouble(operand)};
        }
        if (operand.type == ScalarType::Uint) {
          const auto value = std::get<uint64_t>(operand.value);
          if (syntax.expressions[*expression.lhs].kind == Expr::Kind::Int) {
            if (value > (1ULL << 63)) {
              fail(expression.location, "integer negation overflows i64");
            }
            return {.type = ScalarType::Int,
                    .value = std::numeric_limits<int64_t>::min()};
          }
          return {.type = ScalarType::Uint, .value = 0ULL - value};
        }
        const auto value = std::get<int64_t>(operand.value);
        if (value == std::numeric_limits<int64_t>::min()) {
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
        if (lhs.type != ScalarType::Bool) {
          fail(expression.location, "logical operators require bool operands");
        }
        const auto left = std::get<bool>(lhs.value);
        const auto shortCircuits =
            expression.kind == Expr::Kind::And ? !left : left;
        if (shortCircuits) {
          if (constantExpressionType(*expression.rhs) != ScalarType::Bool) {
            fail(expression.location,
                 "logical operators require bool operands");
          }
          return {.type = ScalarType::Bool, .value = left};
        }
        const auto rhs = evaluateConstant(*expression.rhs);
        if (rhs.type != ScalarType::Bool) {
          fail(expression.location, "logical operators require bool operands");
        }
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
      case Expr::Kind::Ceiling:
      case Expr::Kind::Cos:
      case Expr::Kind::Exp:
      case Expr::Kind::Floor:
      case Expr::Kind::Log:
      case Expr::Kind::Sin:
      case Expr::Kind::Sqrt:
      case Expr::Kind::Tan: {
        const auto operand = evaluateConstant(*expression.lhs);
        if (operand.type == ScalarType::Bool) {
          fail(expression.location, "math functions require numeric operands");
        }
        const bool inverseTrig = expression.kind == Expr::Kind::ArcCos ||
                                 expression.kind == Expr::Kind::ArcSin ||
                                 expression.kind == Expr::Kind::ArcTan;
        const bool trig = expression.kind == Expr::Kind::Cos ||
                          expression.kind == Expr::Kind::Sin ||
                          expression.kind == Expr::Kind::Tan;
        if (operand.type == ScalarType::Angle && !trig) {
          fail(expression.location,
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
          fail(expression.location,
               "constant math expression has a non-finite result");
        }
        return {.type = inverseTrig ? ScalarType::Angle : ScalarType::Float,
                .value = result};
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
        fail(expression.location, "expression is not a compile-time constant");
      case Expr::Kind::PopCount:
      case Expr::Kind::RotateLeft:
      case Expr::Kind::RotateRight:
        fail(expression.location,
             "bit-register expressions are not compile-time constants");
      }
      llvm_unreachable("unknown syntax expression kind");
    }();
    constantValues[id] = result;
    return result;
  }

  [[nodiscard]] ScalarType
  constantExpressionType(const SyntaxExpressionId id) const {
    if (constantTypes[id]) {
      return *constantTypes[id];
    }
    const auto result = [&]() -> ScalarType {
      const auto& expression = syntax.expressions[id];
      switch (expression.kind) {
      case Expr::Kind::Int:
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
          fail(expression.location,
               "expression is not a compile-time constant");
        }
        return symbol->constant->type;
      }
      case Expr::Kind::Neg: {
        const auto type = constantExpressionType(*expression.lhs);
        if (type == ScalarType::Bool) {
          fail(expression.location,
               "numeric negation requires a numeric operand");
        }
        return type;
      }
      case Expr::Kind::Not:
        if (constantExpressionType(*expression.lhs) != ScalarType::Bool) {
          fail(expression.location, "logical negation requires a bool operand");
        }
        return ScalarType::Bool;
      case Expr::Kind::And:
      case Expr::Kind::Or:
        if (constantExpressionType(*expression.lhs) != ScalarType::Bool ||
            constantExpressionType(*expression.rhs) != ScalarType::Bool) {
          fail(expression.location, "logical operators require bool operands");
        }
        return ScalarType::Bool;
      case Expr::Kind::Equal:
      case Expr::Kind::NotEqual:
      case Expr::Kind::Less:
      case Expr::Kind::LessEqual:
      case Expr::Kind::Greater:
      case Expr::Kind::GreaterEqual: {
        const auto lhs = constantExpressionType(*expression.lhs);
        const auto rhs = constantExpressionType(*expression.rhs);
        if (lhs == ScalarType::Bool || rhs == ScalarType::Bool) {
          if (lhs != ScalarType::Bool || rhs != ScalarType::Bool ||
              (expression.kind != Expr::Kind::Equal &&
               expression.kind != Expr::Kind::NotEqual)) {
            fail(expression.location,
                 "bool values only support equality comparisons with bool "
                 "values");
          }
        }
        return ScalarType::Bool;
      }
      case Expr::Kind::ArcCos:
      case Expr::Kind::ArcSin:
      case Expr::Kind::ArcTan:
        if (constantExpressionType(*expression.lhs) == ScalarType::Angle) {
          fail(expression.location,
               "inverse trigonometric functions require a float operand");
        }
        if (constantExpressionType(*expression.lhs) == ScalarType::Bool) {
          fail(expression.location, "math functions require numeric operands");
        }
        return ScalarType::Angle;
      case Expr::Kind::Ceiling:
      case Expr::Kind::Exp:
      case Expr::Kind::Floor:
      case Expr::Kind::Log:
      case Expr::Kind::Sqrt:
        if (constantExpressionType(*expression.lhs) == ScalarType::Bool) {
          fail(expression.location, "math functions require numeric operands");
        }
        if (constantExpressionType(*expression.lhs) == ScalarType::Angle) {
          fail(expression.location,
               "this math function does not accept an angle operand");
        }
        return ScalarType::Float;
      case Expr::Kind::Cos:
      case Expr::Kind::Sin:
      case Expr::Kind::Tan:
        if (constantExpressionType(*expression.lhs) == ScalarType::Bool) {
          fail(expression.location, "math functions require numeric operands");
        }
        return ScalarType::Float;
      case Expr::Kind::Add:
      case Expr::Kind::Sub:
      case Expr::Kind::Mul:
      case Expr::Kind::Div:
      case Expr::Kind::Mod:
      case Expr::Kind::BuiltinMod:
      case Expr::Kind::BuiltinPow:
      case Expr::Kind::Pow: {
        const auto lhs = constantExpressionType(*expression.lhs);
        const auto rhs = constantExpressionType(*expression.rhs);
        if (lhs == ScalarType::Bool || rhs == ScalarType::Bool) {
          fail(expression.location,
               "arithmetic operators require numeric operands");
        }
        if (expression.kind == Expr::Kind::Mod &&
            (lhs == ScalarType::Float || rhs == ScalarType::Float)) {
          fail(expression.location,
               "the '%' operator requires integer operands; use mod() for "
               "floating-point remainder");
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
          fail(expression.location,
               "unsupported arithmetic operation on angle operands");
        }
        if (lhs == ScalarType::Float || rhs == ScalarType::Float) {
          return ScalarType::Float;
        }
        if (expression.kind == Expr::Kind::BuiltinPow &&
            lhs == ScalarType::Int) {
          if (rhs == ScalarType::Int &&
              std::get<int64_t>(evaluateConstant(*expression.rhs).value) < 0) {
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
        fail(expression.location,
             "bitwise operators require explicitly sized uint, bit, or angle "
             "operands, which are not supported yet");
      case Expr::Kind::Index:
        fail(expression.location, "expression is not a compile-time constant");
      case Expr::Kind::PopCount:
        return ScalarType::Uint;
      case Expr::Kind::RotateLeft:
      case Expr::Kind::RotateRight:
        fail(expression.location,
             "bit-register rotations are not scalar expressions");
      }
      llvm_unreachable("unknown syntax expression kind");
    }();
    constantTypes[id] = result;
    return result;
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
                                   std::get<int64_t>(rhs.value) < 0;
    if (lhs.type == ScalarType::Angle || rhs.type == ScalarType::Angle) {
      const auto left = asDouble(lhs);
      const auto right = asDouble(rhs);
      if (expression.kind == Expr::Kind::Add ||
          expression.kind == Expr::Kind::Sub) {
        return {.type = ScalarType::Angle,
                .value = expression.kind == Expr::Kind::Add ? left + right
                                                            : left - right};
      }
      if (expression.kind == Expr::Kind::Mul &&
          (lhs.type == ScalarType::Angle) != (rhs.type == ScalarType::Angle)) {
        return {.type = ScalarType::Angle, .value = left * right};
      }
      if (expression.kind == Expr::Kind::Div && lhs.type == ScalarType::Angle) {
        if (right == 0.0) {
          fail(expression.location, "division by zero");
        }
        return {.type = rhs.type == ScalarType::Angle ? ScalarType::Float
                                                      : ScalarType::Angle,
                .value = left / right};
      }
      fail(expression.location,
           "unsupported arithmetic operation on angle operands");
    }
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
        fail(expression.location, "constant integer arithmetic overflows i64");
      }
      return {.type = ScalarType::Int, .value = result};
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
        fail(expression.location, "division by zero");
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
        fail(expression.location, "modulo by zero");
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
        fail(expression.location,
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
      fail(expression.location, "constant integer arithmetic overflows i64");
    }
    return {.type = ScalarType::Int, .value = result};
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
  }

  [[nodiscard]] BitVectorExpressionId
  analyzeBitVectorExpression(const SyntaxExpressionId syntaxId) {
    const auto& expression = syntax.expressions[syntaxId];
    if (expression.kind == Expr::Kind::Identifier) {
      const auto* symbol = lookup(expression.identifier);
      if (symbol == nullptr || symbol->kind != SymbolKind::Register ||
          program.registers[symbol->id].kind != RegisterKind::Bit) {
        fail(expression.location,
             "bit-vector expression requires a bit register");
      }
      const auto reg = static_cast<RegisterId>(symbol->id);
      if (program.registers[reg].isScalar) {
        fail(expression.location,
             "bit-vector expression requires a bit register, not scalar bit");
      }
      const auto width = program.registers[reg].width;
      for (uint64_t bit = 0; bit < width; ++bit) {
        ensureBitInitialized({.reg = reg, .index = bit}, expression.location);
      }
      return addBitVectorExpression({.kind = BitVectorExpressionKind::Register,
                                     .width = width,
                                     .reg = reg});
    }
    if (expression.kind != Expr::Kind::RotateLeft &&
        expression.kind != Expr::Kind::RotateRight) {
      fail(expression.location,
           "bit-vector expression requires a bit register or rotation");
    }
    const auto operand = analyzeBitVectorExpression(*expression.lhs);
    const auto distance = analyzeExpression(*expression.rhs);
    if (program.expressions[distance].type != ScalarType::Int) {
      fail(syntax.expressions[*expression.rhs].location,
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

  [[nodiscard]] ExpressionId
  analyzeExpression(const SyntaxExpressionId syntaxId) {
    const auto& expression = syntax.expressions[syntaxId];
    if (insideGate) {
      validateGateExpression(syntaxId);
    }
    if (isConstantExpression(syntaxId)) {
      return addConstant(evaluateConstant(syntaxId));
    }
    if (expression.kind == Expr::Kind::PopCount) {
      return addExpression(
          {.kind = ExpressionKind::PopCount,
           .type = ScalarType::Uint,
           .bitVector = analyzeBitVectorExpression(*expression.lhs)});
    }
    if (expression.kind == Expr::Kind::Identifier) {
      const auto* symbol = lookup(expression.identifier);
      if (symbol == nullptr) {
        fail(expression.location,
             "unknown scalar identifier '" + expression.identifier + "'");
      }
      if (symbol->kind == SymbolKind::GateParameter) {
        return addExpression({.kind = ExpressionKind::GateParameter,
                              .type = ScalarType::Angle,
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

    auto kind = ExpressionKind::Constant;
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
    case Expr::Kind::RotateLeft:
    case Expr::Kind::RotateRight:
      fail(expression.location,
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
      fail(expression.location, "expected a scalar arithmetic expression");
    case Expr::Kind::Int:
    case Expr::Kind::Float:
    case Expr::Kind::Bool:
    case Expr::Kind::Identifier:
      llvm_unreachable("handled expression kind");
    }
    auto lhs = analyzeExpression(*expression.lhs);
    auto rhs =
        expression.rhs
            ? std::optional<ExpressionId>(analyzeExpression(*expression.rhs))
            : std::nullopt;
    auto lhsType = program.expressions[lhs].type;
    auto rhsType =
        rhs ? std::optional<ScalarType>(program.expressions[*rhs].type)
            : std::nullopt;
    if (lhsType == ScalarType::Bool ||
        (rhsType && *rhsType == ScalarType::Bool)) {
      fail(expression.location,
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
        fail(expression.location,
             inverseTrig
                 ? "inverse trigonometric functions require a float operand"
                 : "this math function does not accept an angle operand");
      }
      lhs = castExpression(lhs, trig ? ScalarType::Angle : ScalarType::Float,
                           expression.location);
      return addExpression(
          {.kind = kind,
           .type = inverseTrig ? ScalarType::Angle : ScalarType::Float,
           .lhs = lhs});
    }
    if (!rhs) {
      return addExpression({.kind = kind, .type = lhsType, .lhs = lhs});
    }

    if (expression.kind == Expr::Kind::Mod &&
        (lhsType == ScalarType::Float || lhsType == ScalarType::Angle ||
         *rhsType == ScalarType::Float || *rhsType == ScalarType::Angle)) {
      fail(expression.location,
           "the '%' operator requires integer operands; use mod() for "
           "floating-point remainder");
    }

    if (lhsType == ScalarType::Angle || *rhsType == ScalarType::Angle) {
      ScalarType type = ScalarType::Angle;
      if (kind == ExpressionKind::Add || kind == ExpressionKind::Subtract) {
        lhs = castExpression(lhs, ScalarType::Angle, expression.location);
        *rhs = castExpression(*rhs, ScalarType::Angle, expression.location);
      } else if (kind == ExpressionKind::Multiply &&
                 (lhsType == ScalarType::Angle) !=
                     (*rhsType == ScalarType::Angle)) {
        if (lhsType != ScalarType::Angle) {
          lhs = castExpression(lhs, ScalarType::Float, expression.location);
        } else {
          *rhs = castExpression(*rhs, ScalarType::Float, expression.location);
        }
      } else if (kind == ExpressionKind::Divide &&
                 lhsType == ScalarType::Angle) {
        if (*rhsType == ScalarType::Angle) {
          type = ScalarType::Float;
        } else {
          *rhs = castExpression(*rhs, ScalarType::Float, expression.location);
        }
      } else {
        fail(expression.location,
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
        *rhs = castExpression(*rhs, ScalarType::Uint, expression.location);
      } else if (lhsType == ScalarType::Uint &&
                 (*rhsType == ScalarType::Uint ||
                  (*rhsType == ScalarType::Int &&
                   program.expressions[*rhs].kind == ExpressionKind::Constant &&
                   std::get<int64_t>(program.expressions[*rhs].constant) >=
                       0))) {
        type = ScalarType::Uint;
        *rhs = castExpression(*rhs, ScalarType::Uint, expression.location);
      } else {
        lhs = castExpression(lhs, ScalarType::Float, expression.location);
        *rhs = castExpression(*rhs, ScalarType::Float, expression.location);
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
    lhs = castExpression(lhs, type, expression.location);
    *rhs = castExpression(*rhs, type, expression.location);
    return addExpression({.kind = kind, .type = type, .lhs = lhs, .rhs = *rhs});
  }

  [[nodiscard]] uint64_t
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
    const auto value =
        constant.type == ScalarType::Uint
            ? std::get<uint64_t>(constant.value)
            : static_cast<uint64_t>(std::get<int64_t>(constant.value));
    if (value == 0 || (constant.type == ScalarType::Int &&
                       std::get<int64_t>(constant.value) < 0)) {
      fail(location, "register width must be greater than zero");
    }
    if (value > REGISTER_WIDTH_LIMIT) {
      fail(location, Twine("register width exceeds the limit of ") +
                         Twine(REGISTER_WIDTH_LIMIT));
    }
    return value;
  }

  [[nodiscard]] std::optional<uint64_t>
  constantIndex(const SyntaxExpressionId id, const uint64_t width,
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
      value += static_cast<int64_t>(width);
    }
    if (value < 0) {
      fail(location, "index is out of bounds");
    }
    return static_cast<uint64_t>(value);
  }

  void analyzeTopLevelBody() {
    assert(syntax.body.size() == syntax.bodyIncludeContexts.size());
    for (const auto [id, includeContext] :
         llvm::zip_equal(syntax.body, syntax.bodyIncludeContexts)) {
      currentIncludeContext = includeContext;
      analyzeStatement(syntax.statements[id], program.body, /*global=*/true);
    }
    currentIncludeContext.reset();
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
            activateStandardLibrary(statement.location, data.kind);
          } else if constexpr (std::is_same_v<T, SyntaxScalarDeclaration>) {
            analyzeScalarDeclaration(statement.location, data, destination,
                                     global);
          } else if constexpr (std::is_same_v<T, SyntaxAssignment>) {
            analyzeAssignment(statement.location, data, destination);
          } else if constexpr (std::is_same_v<T, SyntaxQubitDeclaration> ||
                               std::is_same_v<T, SyntaxBitDeclaration>) {
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

  void activateStandardLibrary(SMLoc location, const StandardLibraryKind kind) {
    auto& alreadyIncluded = kind == StandardLibraryKind::StdGates
                                ? program.stdGatesIncluded
                                : program.qelib1Included;
    if (alreadyIncluded) {
      fail(location, kind == StandardLibraryKind::StdGates
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
        fail(location,
             "standard-library gate '" + gate.name + "' is already declared");
      }
    }
    alreadyIncluded = true;
  }

  void analyzeScalarDeclaration(SMLoc location,
                                const SyntaxScalarDeclaration& declaration,
                                std::vector<StatementId>& destination,
                                const bool global) {
    if (declaration.output && !global) {
      fail(location, "outputs must be declared at global scope");
    }
    const auto type = scalarType(declaration.kind);
    if (declaration.isConst) {
      if (!declaration.initializer ||
          !isConstantExpression(*declaration.initializer)) {
        fail(location, "const declaration requires a constant initializer");
      }
      auto constant = promoteConstInitializer(
          evaluateConstant(*declaration.initializer), type, location);
      declare(
          location, declaration.identifier,
          {.kind = SymbolKind::Constant, .type = type, .constant = constant});
      return;
    }

    const auto id = static_cast<ScalarId>(program.scalars.size());
    program.scalars.push_back({.type = type,
                               .name = declaration.identifier.str(),
                               .location = getSourceLocation(location)});
    initializedScalars.push_back(false);
    scalarGenerations.push_back(0);
    declare(location, declaration.identifier,
            {.kind = SymbolKind::Scalar, .type = type, .id = id});
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
        typed.conditionInitializer = analyzeBoolValue(*declaration.initializer);
      } else {
        typed.initializer = castExpression(
            analyzeExpression(*declaration.initializer), type,
            syntax.expressions[*declaration.initializer].location);
      }
      initializedScalars[id] = true;
    }
    destination.push_back(addStatement(location, typed));
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
        typed.value =
            castExpression(analyzeExpression(assignment.value), symbol->type,
                           syntax.expressions[assignment.value].location);
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
      const auto bitVector = analyzeBitVectorExpression(assignment.value);
      if (program.bitVectorExpressions[bitVector].width !=
          program.registers[targetReg].width) {
        fail(location, "bit-register assignment widths must match");
      }
      for (uint64_t bit = 0; bit < program.registers[targetReg].width; ++bit) {
        markBitInitialized({.reg = targetReg, .index = bit});
      }
      destination.push_back(
          addStatement(location, BitVectorAssignmentStatement{
                                     .target = targetReg, .value = bitVector}));
      return;
    }
    auto targets = resolveBits(assignment.target);
    if (targets.size() > 1) {
      fail(location,
           "whole-register bit assignment requires a bit-register value");
    }
    const auto condition = analyzeBoolValue(assignment.value);
    markBitInitialized(targets.front());
    destination.push_back(
        addStatement(location, BitAssignmentStatement{.target = targets.front(),
                                                      .value = condition}));
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
    if (width > TOTAL_REGISTER_ELEMENT_LIMIT - totalRegisterElements) {
      fail(location, Twine("total register elements exceed the limit of ") +
                         Twine(TOTAL_REGISTER_ELEMENT_LIMIT));
    }
    totalRegisterElements += width;
    program.registers.push_back(
        {.kind = isQubit ? RegisterKind::Qubit : RegisterKind::Bit,
         .name = declaration.identifier.str(),
         .width = width,
         .isScalar = !declaration.size.has_value(),
         .location = getSourceLocation(location)});
    // OpenQASM 2 classical bits are zero-initialized; OpenQASM 3 bits are not.
    const bool initiallyInitialized = !isQubit && program.openQASM2;
    initializedBits.push_back(
        std::make_shared<BitInitialization>(width, initiallyInitialized));
    dynamicBitFacts.push_back(std::make_shared<DynamicBitFactSet>());
    bitGenerations.push_back(0);
    declare(location, declaration.identifier,
            {.kind = SymbolKind::Register, .id = id});
    if (!isQubit && global) {
      const ProgramOutput programOutput{.kind = OutputKind::BitRegister,
                                        .symbol = id};
      implicitOutputs.push_back(programOutput);
      if (output || program.openQASM2) {
        explicitOutputs.push_back(programOutput);
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
    // OpenQASM 2 corpora commonly redefine qelib1 gates (e.g. `gate sx`).
    // Allow those definitions to shadow the standard-library entry; OpenQASM 3
    // keeps the stricter "no shadowing stdgates" rule.
    if (const auto* catalog = lookupGate(declaration.identifier);
        catalog != nullptr && isGateAvailable(*catalog) && !program.openQASM2) {
      fail(location,
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
      declare(location, parameter,
              {.kind = SymbolKind::GateParameter,
               .type = ScalarType::Angle,
               .id = static_cast<uint32_t>(index)});
    }
    for (const auto [index, qubit] : llvm::enumerate(declaration.qubits)) {
      declare(
          location, qubit,
          {.kind = SymbolKind::GateQubit, .id = static_cast<uint32_t>(index)});
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
    if (!measurement.target) {
      return addStatement(location,
                          MeasurementStatement{.qubits = std::move(qubits)});
    }
    const auto* destination = lookup(measurement.target->identifier);
    if (destination != nullptr && destination->kind == SymbolKind::Scalar) {
      if (destination->type == ScalarType::Bool) {
        fail(location,
             "measurement results have type 'bit' and cannot be assigned to "
             "'bool' without an explicit cast");
      }
      fail(location,
           "measurement assignment requires a bit-register destination");
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
      auto selection = resolveQubitOperand(operand);
      qubits.insert(qubits.end(), selection.begin(), selection.end());
    }
    return addStatement(location,
                        BarrierStatement{.qubits = std::move(qubits)});
  }

  [[nodiscard]] StatementId analyzeIf(SMLoc location,
                                      const SyntaxIf& conditional) {
    IfStatement result{.condition = analyzeCondition(conditional.condition)};
    const auto beforeBitsInitialized = initializedBits;
    const auto beforeInitialized = initializedScalars;
    const auto beforeGenerations = scalarGenerations;
    const auto beforeBitGenerations = bitGenerations;
    const auto beforeDynamicBitFacts = dynamicBitFacts;
    scopes.emplace_back();
    analyzeBody(conditional.thenStatements, result.thenStatements,
                /*global=*/false);
    const auto afterThenBitsInitialized = initializedBits;
    const auto afterThenInitialized = initializedScalars;
    const auto afterThenGenerations = scalarGenerations;
    const auto afterThenBitGenerations = bitGenerations;
    const auto afterThenDynamicBitFacts = dynamicBitFacts;
    scopes.pop_back();

    restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                       beforeGenerations, beforeBitGenerations);
    restoreDynamicFactsPrefix(beforeDynamicBitFacts);
    scopes.emplace_back();
    analyzeBody(conditional.elseStatements, result.elseStatements,
                /*global=*/false);
    const auto afterElseBitsInitialized = initializedBits;
    const auto afterElseInitialized = initializedScalars;
    const auto afterElseGenerations = scalarGenerations;
    const auto afterElseBitGenerations = bitGenerations;
    const auto afterElseDynamicBitFacts = dynamicBitFacts;
    scopes.pop_back();

    const auto knownCondition = constantCondition(conditional.condition);
    if (knownCondition) {
      const auto& knownBitsInitialized =
          *knownCondition ? afterThenBitsInitialized : afterElseBitsInitialized;
      const auto& knownInitialized =
          *knownCondition ? afterThenInitialized : afterElseInitialized;
      const auto& knownGenerations =
          *knownCondition ? afterThenGenerations : afterElseGenerations;
      const auto& knownBitGenerations =
          *knownCondition ? afterThenBitGenerations : afterElseBitGenerations;
      const auto& knownDynamicBitFacts =
          *knownCondition ? afterThenDynamicBitFacts : afterElseDynamicBitFacts;
      restoreStatePrefix(knownBitsInitialized, knownInitialized,
                         knownGenerations, knownBitGenerations);
      restoreDynamicFactsPrefix(knownDynamicBitFacts);
      return addStatement(location, std::move(result));
    }

    restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                       beforeGenerations, beforeBitGenerations);
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
    }
    for (size_t reg = 0; reg < beforeBitGenerations.size(); ++reg) {
      bitGenerations[reg] =
          std::max(afterThenBitGenerations[reg], afterElseBitGenerations[reg]);
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
                 ? std::get<uint64_t>(value.value) == 0
                 : std::get<int64_t>(value.value) == 0;
    };
    if (isConstantExpression(loop.step) &&
        constantIsZero(evaluateConstant(loop.step))) {
      fail(location, "for-loop range step must not be zero");
    }

    const auto beforeBitsInitialized = initializedBits;
    const auto beforeInitialized = initializedScalars;
    const auto beforeGenerations = scalarGenerations;
    const auto beforeBitGenerations = bitGenerations;
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
    const auto afterBodyBitGenerations = bitGenerations;
    const auto afterBodyDynamicBitFacts = dynamicBitFacts;
    scopes.pop_back();
    restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                       beforeGenerations, beforeBitGenerations);
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
    return addStatement(location, std::move(result));
  }

  [[nodiscard]] StatementId analyzeWhile(SMLoc location,
                                         const SyntaxWhile& loop) {
    WhileStatement result{.condition = analyzeCondition(loop.condition)};
    const auto beforeBitsInitialized = initializedBits;
    const auto beforeInitialized = initializedScalars;
    const auto beforeGenerations = scalarGenerations;
    const auto beforeBitGenerations = bitGenerations;
    const auto beforeDynamicBitFacts = dynamicBitFacts;
    scopes.emplace_back();
    analyzeBody(loop.body, result.body, /*global=*/false);
    scopes.pop_back();
    const auto afterBodyGenerations = scalarGenerations;
    const auto afterBodyBitGenerations = bitGenerations;
    restoreStatePrefix(beforeBitsInitialized, beforeInitialized,
                       beforeGenerations, beforeBitGenerations);
    restoreDynamicFactsPrefix(beforeDynamicBitFacts);
    for (size_t scalar = 0; scalar < beforeGenerations.size(); ++scalar) {
      scalarGenerations[scalar] =
          std::max(beforeGenerations[scalar], afterBodyGenerations[scalar]);
    }
    for (size_t reg = 0; reg < beforeBitGenerations.size(); ++reg) {
      bitGenerations[reg] =
          std::max(beforeBitGenerations[reg], afterBodyBitGenerations[reg]);
    }
    return addStatement(location, std::move(result));
  }

  [[nodiscard]] ConditionId
  analyzeCondition(const SyntaxExpressionId syntaxId) {
    const auto& condition = syntax.expressions[syntaxId];
    ConditionExpression typed{.location =
                                  getSourceLocation(condition.location)};
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
        const auto& rhsSyntax = syntax.expressions[*condition.rhs];
        auto bits = resolveBits({.location = lhsSyntax.location,
                                 .identifier = lhsSyntax.identifier});
        // OpenQASM 2 classical bits default to 0, so partially written
        // registers are valid in `if (c == k)` (e.g. mid-circuit feedback).
        llvm::APInt expectedBits;
        if (rhsSyntax.kind == Expr::Kind::Int &&
            !rhsSyntax.wideInteger.empty()) {
          const auto width = static_cast<unsigned>(
              std::max<size_t>(bits.size(), rhsSyntax.wideInteger.size() * 4));
          expectedBits =
              llvm::APInt(width, rhsSyntax.wideInteger, /*radix=*/10);
        } else {
          const auto expected = evaluateConstant(*condition.rhs);
          if (!isInteger(expected.type) ||
              (expected.type == ScalarType::Int &&
               std::get<int64_t>(expected.value) < 0)) {
            fail(condition.location,
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
        typed.comparisonLhs =
            castExpression(typed.comparisonLhs, comparisonType,
                           syntax.expressions[*condition.lhs].location);
        typed.comparisonRhs =
            castExpression(typed.comparisonRhs, comparisonType,
                           syntax.expressions[*condition.rhs].location);
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
      fail(condition.location, "condition must have bool type");
    }
    return addCondition(std::move(typed));
  }

  [[nodiscard]] std::vector<GateApplication>
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
    // A user-defined gate shadows the standard-library entry with the same
    // name (needed for OpenQASM 2 qelib1 redefinitions).
    if (custom != customGates.end()) {
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
      auto parameter = analyzeExpression(expression);
      if (program.expressions[parameter].type == ScalarType::Bool) {
        fail(call.location, "gate parameters require numeric expressions");
      }
      parameter = castExpression(parameter, ScalarType::Angle,
                                 syntax.expressions[expression].location);
      parameters.push_back(parameter);
    }

    std::vector<GateModifier> modifiers;
    size_t addedControls = compatibilityControls;
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
          if (program.expressions[operand].type == ScalarType::Bool ||
              program.expressions[operand].type == ScalarType::Angle) {
            fail(call.location, "pow modifier requires a numeric argument");
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
            fail(call.location,
                 "gate control count must be a constant integer");
          }
          const auto constant = evaluateConstant(*modifier.argument);
          if (!isInteger(constant.type) || asSigned(constant) <= 0) {
            fail(call.location, "gate control count must be positive");
          }
          count = static_cast<uint64_t>(asSigned(constant));
          operand = addConstant(
              {.type = ScalarType::Int, .value = static_cast<int64_t>(count)});
        }
        if (count > call.operands.size() - addedControls) {
          fail(call.location, "Invalid number of qubit operands for gate '" +
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
      fail(call.location, "Invalid number of qubit operands for gate '" +
                              call.identifier + "'.");
    }

    size_t emittedOperandCount = call.operands.size();
    if (standard != nullptr && standard->variadicControls) {
      size_t activeBaseOperands = baseOperandCount;
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
                .value = static_cast<int64_t>(intrinsicControls)})});
      callee = canonicalGateName(standard->lowering).str();
      emittedOperandCount = addedControls + activeBaseOperands;
    }

    std::vector<std::vector<QubitReference>> selections;
    size_t broadcastWidth = 1;
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
    for (size_t index = 0; index < broadcastWidth; ++index) {
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
      for (uint64_t index = 0; index < width; ++index) {
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
      for (uint64_t index = 0; index < width; ++index) {
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
      if (llvm::all_of(*initializedBits[bit.reg],
                       [](const bool initialized) { return initialized; })) {
        return;
      }
      std::vector<std::pair<uint64_t, uint64_t>> dependencies;
      collectDependencies(*bit.dynamicIndex, dependencies);
      if (llvm::any_of(*dynamicBitFacts[bit.reg], [&](const auto& fact) {
            return fact.dependencies == dependencies &&
                   sameExpression(fact.expression, *bit.dynamicIndex);
          })) {
        return;
      }
      fail(location, "dynamic classical index may read an uninitialized bit");
    }
    if (!(*initializedBits[bit.reg])[bit.index]) {
      fail(location, "classical condition bit has not been initialized");
    }
  }

  void finalizeOutputs() {
    program.outputs =
        explicitOutputs.empty() ? implicitOutputs : explicitOutputs;
    for (const auto output : program.outputs) {
      if (output.kind == OutputKind::Scalar) {
        if (!initializedScalars[output.symbol]) {
          throw SemanticError(
              {.location = program.scalars[output.symbol].location,
               .message = "Output scalar '" +
                          program.scalars[output.symbol].name +
                          "' is not initialized."});
        }
        continue;
      }
      const auto reg = static_cast<RegisterId>(output.symbol);
      if (llvm::any_of(*initializedBits[reg],
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

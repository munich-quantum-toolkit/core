/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Target/OpenQASM/OpenQASM.h"

#include "mlir/Dialect/OQ3/IR/OQ3Ops.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "qasm3Lexer.h"
#include "qasm3Parser.h"

#include <antlr4-runtime.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Verifier.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace mlir::oq3 {
namespace {

enum class SourceVersion { OpenQASM2, OpenQASM31 };

struct GateSignature {
  size_t numParameters;
  size_t numQubits;
};

const llvm::StringMap<GateSignature>& standardGates31() {
  static const llvm::StringMap<GateSignature> gates = {
      {"p", {1, 1}},   {"x", {0, 1}},    {"y", {0, 1}},   {"z", {0, 1}},
      {"h", {0, 1}},   {"s", {0, 1}},    {"sdg", {0, 1}}, {"t", {0, 1}},
      {"tdg", {0, 1}}, {"sx", {0, 1}},   {"rx", {1, 1}},  {"ry", {1, 1}},
      {"rz", {1, 1}},  {"cx", {0, 2}},   {"cy", {0, 2}},  {"cz", {0, 2}},
      {"cp", {1, 2}},  {"crx", {1, 2}},  {"cry", {1, 2}}, {"crz", {1, 2}},
      {"ch", {0, 2}},  {"swap", {0, 2}}, {"ccx", {0, 3}}, {"cswap", {0, 3}},
      {"cu", {4, 2}},
  };
  return gates;
}

const llvm::StringMap<GateSignature>& qelib1Gates() {
  static const llvm::StringMap<GateSignature> gates = {
      {"u3", {3, 1}},    {"u2", {2, 1}},  {"u1", {1, 1}},  {"cx", {0, 2}},
      {"id", {0, 1}},    {"x", {0, 1}},   {"y", {0, 1}},   {"z", {0, 1}},
      {"h", {0, 1}},     {"s", {0, 1}},   {"sdg", {0, 1}}, {"t", {0, 1}},
      {"tdg", {0, 1}},   {"rx", {1, 1}},  {"ry", {1, 1}},  {"rz", {1, 1}},
      {"cz", {0, 2}},    {"cy", {0, 2}},  {"ch", {0, 2}},  {"ccx", {0, 3}},
      {"crz", {1, 2}},   {"cu1", {1, 2}}, {"cu3", {3, 2}}, {"swap", {0, 2}},
      {"cswap", {0, 3}},
  };
  return gates;
}

Location locationFor(MLIRContext& context, llvm::StringRef filename,
                     const antlr4::Token* token) {
  if (token == nullptr) {
    return UnknownLoc::get(&context);
  }
  return FileLineColLoc::get(&context, filename, token->getLine(),
                             token->getCharPositionInLine() + 1);
}

class DiagnosticErrorListener final : public antlr4::BaseErrorListener {
public:
  DiagnosticErrorListener(MLIRContext& context, llvm::StringRef filename)
      : context(context), filename(filename) {}

  void syntaxError(antlr4::Recognizer* /*recognizer*/,
                   antlr4::Token* offendingSymbol, size_t line,
                   size_t charPositionInLine, const std::string& message,
                   std::exception_ptr /*exception*/) override {
    hadError = true;
    auto loc =
        FileLineColLoc::get(&context, filename, line, charPositionInLine + 1);
    emitError(loc) << "OpenQASM syntax error: " << message;
    if (offendingSymbol != nullptr) {
      emitRemark(loc) << "while parsing '" << offendingSymbol->getText() << "'";
    }
  }

  bool failed() const { return hadError; }

private:
  MLIRContext& context;
  std::string filename;
  bool hadError = false;
};

std::string removeSeparators(std::string text) {
  std::erase(text, '_');
  return text;
}

std::optional<int64_t> parseIntegerText(std::string text) {
  text = removeSeparators(std::move(text));
  bool negative = false;
  if (!text.empty() && text.front() == '-') {
    negative = true;
    text.erase(text.begin());
  }

  unsigned base = 10;
  if (text.starts_with("0x") || text.starts_with("0X")) {
    base = 16;
    text.erase(0, 2);
  } else if (text.starts_with("0b") || text.starts_with("0B")) {
    base = 2;
    text.erase(0, 2);
  } else if (text.starts_with("0o") || text.starts_with("0O")) {
    base = 8;
    text.erase(0, 2);
  }

  uint64_t value = 0;
  if (text.empty() || llvm::StringRef(text).getAsInteger(base, value) ||
      value > static_cast<uint64_t>(INT64_MAX) + negative) {
    return std::nullopt;
  }
  if (negative && value == static_cast<uint64_t>(INT64_MAX) + 1) {
    return INT64_MIN;
  }
  return negative ? -static_cast<int64_t>(value) : static_cast<int64_t>(value);
}

class SemanticBuilder {
public:
  SemanticBuilder(MLIRContext& context, llvm::SourceMgr& sourceMgr,
                  llvm::StringRef filename, const SourceVersion version,
                  const OpenQASMTranslationOptions& options)
      : context(context), sourceMgr(sourceMgr), filename(filename),
        version(version), includeDirectories(options.includeDirectories),
        builder(&context), module(ModuleOp::create(UnknownLoc::get(&context))) {
    context.loadDialect<OQ3Dialect, qc::QCDialect, arith::ArithDialect,
                        func::FuncDialect, memref::MemRefDialect,
                        math::MathDialect, scf::SCFDialect>();
    module->getOperation()->setAttr(
        "oq3.version",
        builder.getStringAttr(version == SourceVersion::OpenQASM2 ? "2.0-compat"
                                                                  : "3.1"));
    auto functionType = builder.getFunctionType({}, {});
    main = func::FuncOp::create(module->getLoc(), "main", functionType);
    module->getBody()->push_back(main);
    entry = main.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    declareExternalGate("U", {3, 1});
    if (version == SourceVersion::OpenQASM31) {
      declareExternalGate("gphase", {1, 0});
    }
  }

  OwningOpRef<ModuleOp> build(qasm3Parser::ProgramContext* program) {
    if (failed(configureProgramSignature(program))) {
      return {};
    }
    for (auto* item : program->statementOrScope()) {
      if (item->scope() != nullptr) {
        return fail<OwningOpRef<ModuleOp>>(item, "anonymous top-level scope");
      }
      if (failed(processStatement(item->statement()))) {
        return {};
      }
    }
    builder.setInsertionPointToEnd(entry);
    llvm::SmallVector<Value> outputs;
    for (const std::string& name : outputNames) {
      const auto found = bits.find(name);
      if (found == bits.end()) {
        emitError(module->getLoc())
            << "output bit register was not declared: " << name;
        return {};
      }
      llvm::SmallVector<Value> values;
      values.reserve(found->second.width);
      for (int64_t index = 0; index < found->second.width; ++index) {
        const Value position =
            arith::ConstantIndexOp::create(builder, module->getLoc(), index);
        values.push_back(memref::LoadOp::create(
            builder, module->getLoc(), found->second.storage, position));
      }
      OperationState state(module->getLoc(), PackBitsOp::getOperationName());
      state.addOperands(values);
      state.addTypes(BitType::get(&context, found->second.width));
      outputs.push_back(builder.create(state)->getResult(0));
    }
    func::ReturnOp::create(builder, module->getLoc(), outputs);
    if (mlir::failed(verify(module.get()))) {
      emitError(module->getLoc()) << "constructed invalid typed OpenQASM IR";
      return {};
    }
    return std::move(module);
  }

private:
  LogicalResult
  configureProgramSignature(qasm3Parser::ProgramContext* program) {
    llvm::SmallVector<Type> inputTypes;
    llvm::SmallVector<Type> resultTypes;
    llvm::SmallVector<std::pair<std::string, Type>> inputs;
    llvm::StringSet<> names;
    for (auto* item : program->statementOrScope()) {
      if (item->statement() == nullptr ||
          item->statement()->ioDeclarationStatement() == nullptr) {
        continue;
      }
      auto* declaration = item->statement()->ioDeclarationStatement();
      if (declaration->arrayType() != nullptr ||
          declaration->scalarType()->BIT() == nullptr) {
        return unsupported(declaration,
                           "non-bit input/output declarations and arrays");
      }
      const auto width =
          evaluateDesignator(declaration->scalarType()->designator());
      if (!width || *width <= 0) {
        return fail(
            declaration,
            "input/output bit width must be a positive constant integer");
      }
      const std::string name = declaration->Identifier()->getText();
      if (!names.insert(name).second) {
        return fail(declaration,
                    llvm::Twine("input/output identifier already declared: ")
                        .concat(name)
                        .str());
      }
      const Type type = BitType::get(&context, *width);
      if (declaration->INPUT() != nullptr) {
        inputTypes.push_back(type);
        inputs.emplace_back(name, type);
      } else {
        resultTypes.push_back(type);
        outputNames.push_back(name);
      }
    }
    main.setType(builder.getFunctionType(inputTypes, resultTypes));
    for (const auto& input : inputs) {
      inputValues.insert(
          {input.first, entry->addArgument(input.second, main.getLoc())});
    }
    return success();
  }

  template <typename Result = LogicalResult>
  Result fail(antlr4::ParserRuleContext* context, llvm::StringRef message) {
    emitError(locationFor(this->context, filename, context->getStart()))
        << message;
    if constexpr (std::is_same_v<Result, LogicalResult>) {
      return failure();
    } else {
      return {};
    }
  }

  LogicalResult unsupported(antlr4::ParserRuleContext* context,
                            llvm::StringRef feature) {
    return fail(
        context,
        llvm::Twine(
            "OpenQASM feature is not yet supported by the typed frontend: ")
            .concat(feature)
            .str());
  }

  LogicalResult processStatement(qasm3Parser::StatementContext* statement) {
    if (statement->includeStatement() != nullptr) {
      return processInclude(statement->includeStatement());
    }
    if (statement->quantumDeclarationStatement() != nullptr) {
      return processQuantumDeclaration(
          statement->quantumDeclarationStatement());
    }
    if (statement->oldStyleDeclarationStatement() != nullptr) {
      return processOldStyleDeclaration(
          statement->oldStyleDeclarationStatement());
    }
    if (statement->gateCallStatement() != nullptr) {
      return processGateCall(statement->gateCallStatement());
    }
    if (statement->resetStatement() != nullptr) {
      return processReset(statement->resetStatement());
    }
    if (statement->barrierStatement() != nullptr) {
      return processBarrier(statement->barrierStatement());
    }
    if (statement->forStatement() != nullptr) {
      return processFor(statement->forStatement());
    }
    if (statement->gateStatement() != nullptr) {
      return processGateDefinition(statement->gateStatement());
    }
    if (statement->classicalDeclarationStatement() != nullptr) {
      return processClassicalDeclaration(
          statement->classicalDeclarationStatement());
    }
    if (statement->constDeclarationStatement() != nullptr) {
      return processConstDeclaration(statement->constDeclarationStatement());
    }
    if (statement->ioDeclarationStatement() != nullptr) {
      return processIoDeclaration(statement->ioDeclarationStatement());
    }
    if (statement->assignmentStatement() != nullptr) {
      return processAssignment(statement->assignmentStatement());
    }
    if (statement->measureArrowAssignmentStatement() != nullptr) {
      return processMeasurement(
          statement->measureArrowAssignmentStatement()->measureExpression(),
          statement->measureArrowAssignmentStatement()->indexedIdentifier());
    }
    if (statement->ifStatement() != nullptr) {
      return processIf(statement->ifStatement());
    }
    if (statement->whileStatement() != nullptr) {
      return processWhile(statement->whileStatement());
    }
    if (statement->aliasDeclarationStatement() != nullptr) {
      return unsupported(statement->aliasDeclarationStatement(), "aliases");
    }
    if (statement->defStatement() != nullptr ||
        statement->externStatement() != nullptr) {
      return unsupported(statement, "subroutines and externs");
    }
    if (statement->switchStatement() != nullptr ||
        statement->breakStatement() != nullptr ||
        statement->continueStatement() != nullptr) {
      return unsupported(statement, "switch/break/continue control flow");
    }
    if (statement->boxStatement() != nullptr ||
        statement->delayStatement() != nullptr) {
      return unsupported(statement, "timing and box statements");
    }
    if (statement->calStatement() != nullptr ||
        statement->defcalStatement() != nullptr ||
        statement->calibrationGrammarStatement() != nullptr) {
      return unsupported(statement, "calibration statements");
    }
    if (statement->pragma() != nullptr || !statement->annotation().empty()) {
      return unsupported(statement, "pragmas and annotations");
    }
    return unsupported(statement, "statement");
  }

  LogicalResult processInclude(qasm3Parser::IncludeStatementContext* include) {
    std::string name = include->StringLiteral()->getText();
    if (name.size() >= 2) {
      name = name.substr(1, name.size() - 2);
    }
    if (name == "stdgates.inc") {
      if (version == SourceVersion::OpenQASM2) {
        return fail(include,
                    "stdgates.inc is only available in OpenQASM 3.1 mode");
      }
      for (const auto& [gate, signature] : standardGates31()) {
        declareExternalGate(gate, signature);
      }
      return success();
    }
    if (name == "qelib1.inc") {
      if (version != SourceVersion::OpenQASM2) {
        return fail(
            include,
            "qelib1.inc is only available in OpenQASM 2.0 compatibility mode");
      }
      for (const auto& [gate, signature] : qelib1Gates()) {
        declareExternalGate(gate, signature);
      }
      return success();
    }
    llvm::SmallVector<std::string> candidates;
    if (!filename.starts_with("<")) {
      llvm::SmallString<256> parent(filename);
      llvm::sys::path::remove_filename(parent);
      llvm::sys::path::append(parent, name);
      candidates.push_back(parent.str().str());
    }
    for (const auto& directory : includeDirectories) {
      llvm::SmallString<256> candidate(directory);
      llvm::sys::path::append(candidate, name);
      candidates.push_back(candidate.str().str());
    }
    for (const auto& candidate : candidates) {
      if (llvm::sys::fs::exists(candidate)) {
        return processIncludedFile(include, candidate);
      }
    }
    return fail(include,
                llvm::Twine("include file not found: ").concat(name).str());
  }

  LogicalResult
  processIncludedFile(qasm3Parser::IncludeStatementContext* include,
                      const llvm::StringRef path) {
    if (llvm::is_contained(includeStack, path)) {
      return fail(
          include,
          llvm::Twine("recursive include detected: ").concat(path).str());
    }
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) {
      return fail(
          include,
          llvm::Twine("unable to read include file: ").concat(path).str());
    }
    const unsigned bufferId =
        sourceMgr.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());
    const auto* source = sourceMgr.getMemoryBuffer(bufferId);
    antlr4::ANTLRInputStream input(source->getBuffer().str());
    qasm3Lexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    qasm3Parser parser(&tokens);
    DiagnosticErrorListener errors(context, path);
    lexer.removeErrorListeners();
    parser.removeErrorListeners();
    lexer.addErrorListener(&errors);
    parser.addErrorListener(&errors);
    auto* program = parser.program();
    if (errors.failed() || parser.getNumberOfSyntaxErrors() != 0) {
      return failure();
    }
    const std::string savedFilename = std::exchange(filename, path.str());
    if (program->version() != nullptr) {
      const auto result = fail(
          program->version(),
          "included OpenQASM files must not contain a version declaration");
      filename = savedFilename;
      return result;
    }
    includeStack.push_back(path.str());
    LogicalResult result = success();
    for (auto* item : program->statementOrScope()) {
      if (item->scope() != nullptr ||
          failed(processStatement(item->statement()))) {
        result = failure();
        break;
      }
    }
    includeStack.pop_back();
    filename = savedFilename;
    if (failed(result)) {
      emitRemark(locationFor(context, filename, include->getStart()))
          << "while processing include '" << path << "'";
    }
    return result;
  }

  FunctionType gateFunctionType(const GateSignature signature) {
    llvm::SmallVector<Type> inputs(signature.numParameters,
                                   builder.getF64Type());
    inputs.append(signature.numQubits, qc::QubitType::get(&context));
    return builder.getFunctionType(inputs, {});
  }

  void declareExternalGate(llvm::StringRef name,
                           const GateSignature signature) {
    if (!gates.try_emplace(name, signature).second) {
      return;
    }
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(main);
    GateDeclOp::create(builder, module->getLoc(), name,
                       gateFunctionType(signature));
  }

  LogicalResult processGateDefinition(qasm3Parser::GateStatementContext* gate) {
    const std::string name = gate->Identifier()->getText();
    if (gates.contains(name)) {
      return fail(gate,
                  llvm::Twine("gate already declared: ").concat(name).str());
    }
    const size_t numParameters =
        gate->params == nullptr ? 0 : gate->params->Identifier().size();
    const size_t numQubits = gate->qubits->Identifier().size();
    const GateSignature signature{numParameters, numQubits};

    OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPoint(main);
    OperationState state(locationFor(context, filename, gate->getStart()),
                         GateOp::getOperationName());
    state.addAttribute(SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr(name));
    state.addAttribute("function_type",
                       TypeAttr::get(gateFunctionType(signature)));
    Region* body = state.addRegion();
    body->push_back(new Block());
    for (Type type : gateFunctionType(signature).getInputs()) {
      body->front().addArgument(type, state.location);
    }
    auto* operation = builder.create(state);
    body = &cast<GateOp>(operation).getBody();
    gates.insert({name, signature});

    auto savedQubits = std::move(qubits);
    auto savedParameters = std::move(parameters);
    const std::string savedGate = std::move(currentGate);
    qubits.clear();
    parameters.clear();
    currentGate = name;
    const auto parameterIdentifiers =
        gate->params == nullptr ? std::vector<antlr4::tree::TerminalNode*>{}
                                : gate->params->Identifier();
    for (const auto [index, identifier] :
         llvm::enumerate(parameterIdentifiers)) {
      parameters.insert(
          {identifier->getText(), body->front().getArgument(index)});
    }
    for (const auto [index, identifier] :
         llvm::enumerate(gate->qubits->Identifier())) {
      qubits.insert({identifier->getText(),
                     {body->front().getArgument(numParameters + index)}});
    }

    builder.setInsertionPointToStart(&body->front());
    LogicalResult result = success();
    for (auto* item : gate->scope()->statementOrScope()) {
      if (item->scope() != nullptr ||
          failed(processStatement(item->statement()))) {
        result = failure();
        break;
      }
    }
    if (succeeded(result)) {
      YieldOp::create(builder, state.location);
    }
    qubits = std::move(savedQubits);
    parameters = std::move(savedParameters);
    currentGate = savedGate;
    if (failed(result)) {
      gates.erase(name);
      builder.clearInsertionPoint();
      operation->erase();
    }
    return result;
  }

  std::optional<int64_t> evaluateInteger(qasm3Parser::ExpressionContext* expr) {
    if (auto* literal =
            dynamic_cast<qasm3Parser::LiteralExpressionContext*>(expr)) {
      if (literal->DecimalIntegerLiteral() != nullptr ||
          literal->BinaryIntegerLiteral() != nullptr ||
          literal->OctalIntegerLiteral() != nullptr ||
          literal->HexIntegerLiteral() != nullptr) {
        return parseIntegerText(literal->getText());
      }
      return std::nullopt;
    }
    if (auto* unary = dynamic_cast<qasm3Parser::UnaryExpressionContext*>(expr);
        unary != nullptr && unary->MINUS() != nullptr) {
      auto value = evaluateInteger(unary->expression());
      if (!value || *value == INT64_MIN) {
        return std::nullopt;
      }
      return -*value;
    }
    return std::nullopt;
  }

  std::optional<int64_t>
  evaluateDesignator(qasm3Parser::DesignatorContext* designator,
                     const int64_t defaultValue = 1) {
    if (designator == nullptr) {
      return defaultValue;
    }
    return evaluateInteger(designator->expression());
  }

  LogicalResult declareQubits(antlr4::ParserRuleContext* context,
                              llvm::StringRef name, const int64_t width) {
    if (width <= 0) {
      return fail(context, "qubit-register width must be greater than zero");
    }
    if (qubits.contains(name)) {
      return fail(
          context,
          llvm::Twine("identifier already declared: ").concat(name).str());
    }
    llvm::SmallVector<Value> values;
    values.reserve(width);
    const auto loc = locationFor(this->context, filename, context->getStart());
    for (int64_t i = 0; i < width; ++i) {
      values.push_back(qc::AllocOp::create(builder, loc));
    }
    qubits.insert({name, std::move(values)});
    return success();
  }

  LogicalResult declareBits(antlr4::ParserRuleContext* declaration,
                            const llvm::StringRef name, const int64_t width) {
    if (width <= 0) {
      return fail(declaration, "bit-register width must be greater than zero");
    }
    if (bits.contains(name) || qubits.contains(name)) {
      return fail(
          declaration,
          llvm::Twine("identifier already declared: ").concat(name).str());
    }
    const auto loc = locationFor(context, filename, declaration->getStart());
    const auto type = MemRefType::get({width}, builder.getI1Type());
    const Value storage = memref::AllocaOp::create(builder, loc, type);
    const Value zero = arith::ConstantIntOp::create(builder, loc, 0, 1);
    for (int64_t index = 0; index < width; ++index) {
      const Value position =
          arith::ConstantIndexOp::create(builder, loc, index);
      memref::StoreOp::create(builder, loc, zero, storage, position);
    }
    bits.insert({name, BitRegister{storage, width}});
    return success();
  }

  LogicalResult processClassicalDeclaration(
      qasm3Parser::ClassicalDeclarationStatementContext* declaration) {
    if (declaration->arrayType() != nullptr) {
      return unsupported(declaration, "classical arrays");
    }
    if (declaration->scalarType()->BIT() == nullptr) {
      auto type = convertScalarType(declaration->scalarType());
      if (failed(type)) {
        return failure();
      }
      Value initializer;
      if (declaration->declarationExpression() != nullptr) {
        if (declaration->declarationExpression()->expression() == nullptr) {
          return unsupported(declaration->declarationExpression(),
                             "non-expression scalar initializers");
        }
        auto value =
            buildScalar(declaration->declarationExpression()->expression());
        if (failed(value)) {
          return failure();
        }
        auto cast =
            castScalar(*value, *type,
                       locationFor(context, filename, declaration->getStart()));
        if (failed(cast)) {
          return failure();
        }
        initializer = *cast;
      }
      return declareScalar(declaration, declaration->Identifier()->getText(),
                           *type, initializer, false);
    }
    const auto width =
        evaluateDesignator(declaration->scalarType()->designator());
    if (!width) {
      return fail(declaration,
                  "bit-register designator must be a constant integer");
    }
    const std::string name = declaration->Identifier()->getText();
    if (failed(declareBits(declaration, name, *width))) {
      return failure();
    }
    if (declaration->declarationExpression() == nullptr) {
      return success();
    }
    if (declaration->declarationExpression()->measureExpression() == nullptr) {
      return unsupported(declaration->declarationExpression(),
                         "classical bit initializers");
    }
    return processMeasurement(
        declaration->declarationExpression()->measureExpression(), nullptr,
        name);
  }

  FailureOr<Type>
  convertScalarType(qasm3Parser::ScalarTypeContext* scalarType) {
    const auto width = evaluateDesignator(
        scalarType->designator(), scalarType->FLOAT() != nullptr ? 64 : 32);
    if (!width || *width <= 0 || *width > 64) {
      emitError(locationFor(context, filename, scalarType->getStart()))
          << "scalar width must be a positive integer no greater than 64";
      return failure();
    }
    if (scalarType->BOOL() != nullptr) {
      return builder.getI1Type();
    }
    if (scalarType->INT() != nullptr) {
      return IntegerType::get(&context, *width);
    }
    if (scalarType->UINT() != nullptr) {
      return IntegerType::get(&context, *width);
    }
    if (scalarType->FLOAT() != nullptr) {
      if (*width == 16) {
        return builder.getF16Type();
      }
      if (*width == 32) {
        return builder.getF32Type();
      }
      if (*width == 64) {
        return builder.getF64Type();
      }
      emitError(locationFor(context, filename, scalarType->getStart()))
          << "floating-point width must be 16, 32, or 64";
      return failure();
    }
    emitError(locationFor(context, filename, scalarType->getStart()))
        << "OpenQASM scalar type is parsed but not yet supported in "
           "expressions";
    return failure();
  }

  FailureOr<Value> castScalar(const Value value, const Type target,
                              const Location loc) {
    const Type source = value.getType();
    if (source == target) {
      return value;
    }
    if (const auto sourceInteger = dyn_cast<IntegerType>(source)) {
      if (const auto targetInteger = dyn_cast<IntegerType>(target)) {
        if (sourceInteger.getWidth() < targetInteger.getWidth()) {
          if (sourceInteger.isUnsigned()) {
            return arith::ExtUIOp::create(builder, loc, target, value)
                .getResult();
          }
          return arith::ExtSIOp::create(builder, loc, target, value)
              .getResult();
        }
        return arith::TruncIOp::create(builder, loc, target, value).getResult();
      }
      if (isa<FloatType>(target)) {
        if (sourceInteger.isUnsigned()) {
          return arith::UIToFPOp::create(builder, loc, target, value)
              .getResult();
        }
        return arith::SIToFPOp::create(builder, loc, target, value).getResult();
      }
    }
    if (auto sourceFloat = dyn_cast<FloatType>(source)) {
      if (auto targetFloat = dyn_cast<FloatType>(target)) {
        if (sourceFloat.getWidth() < targetFloat.getWidth()) {
          return arith::ExtFOp::create(builder, loc, target, value).getResult();
        }
        return arith::TruncFOp::create(builder, loc, target, value).getResult();
      }
      if (const auto targetInteger = dyn_cast<IntegerType>(target)) {
        if (targetInteger.isUnsigned()) {
          return arith::FPToUIOp::create(builder, loc, target, value)
              .getResult();
        }
        return arith::FPToSIOp::create(builder, loc, target, value).getResult();
      }
    }
    emitError(loc) << "unsupported scalar cast from " << source << " to "
                   << target;
    return failure();
  }

  FailureOr<Value> zeroForType(const Type type, const Location loc) {
    if (const auto integer = dyn_cast<IntegerType>(type)) {
      return arith::ConstantIntOp::create(builder, loc, 0, integer.getWidth())
          .getResult();
    }
    if (const auto floating = dyn_cast<FloatType>(type)) {
      return arith::ConstantFloatOp::create(builder, loc, floating,
                                            llvm::APFloat(0.0))
          .getResult();
    }
    return failure();
  }

  LogicalResult declareScalar(antlr4::ParserRuleContext* declaration,
                              const llvm::StringRef name, const Type type,
                              Value initializer, const bool immutable) {
    if (scalars.contains(name) || bits.contains(name) ||
        qubits.contains(name)) {
      return fail(
          declaration,
          llvm::Twine("identifier already declared: ").concat(name).str());
    }
    const auto loc = locationFor(context, filename, declaration->getStart());
    if (!initializer) {
      auto zero = zeroForType(type, loc);
      if (failed(zero)) {
        return fail(declaration, "scalar type has no default value");
      }
      initializer = *zero;
    }
    if (immutable) {
      scalars.insert({name, ScalarVariable{{}, type, initializer, true}});
      return success();
    }
    const Value storage =
        memref::AllocaOp::create(builder, loc, MemRefType::get({}, type));
    memref::StoreOp::create(builder, loc, initializer, storage, ValueRange{});
    scalars.insert({name, ScalarVariable{storage, type, {}, false}});
    return success();
  }

  LogicalResult processConstDeclaration(
      qasm3Parser::ConstDeclarationStatementContext* declaration) {
    auto type = convertScalarType(declaration->scalarType());
    if (failed(type) ||
        declaration->declarationExpression()->expression() == nullptr) {
      return failure();
    }
    auto value =
        buildScalar(declaration->declarationExpression()->expression());
    if (failed(value)) {
      return failure();
    }
    auto cast = castScalar(
        *value, *type, locationFor(context, filename, declaration->getStart()));
    if (failed(cast)) {
      return failure();
    }
    return declareScalar(declaration, declaration->Identifier()->getText(),
                         *type, *cast, true);
  }

  LogicalResult processIoDeclaration(
      qasm3Parser::IoDeclarationStatementContext* declaration) {
    const auto width =
        evaluateDesignator(declaration->scalarType()->designator());
    const std::string name = declaration->Identifier()->getText();
    if (!width || failed(declareBits(declaration, name, *width))) {
      return failure();
    }
    if (declaration->OUTPUT() != nullptr) {
      return success();
    }
    const auto input = inputValues.find(name);
    if (input == inputValues.end()) {
      return fail(declaration, "input is missing from the program signature");
    }
    const auto& destination = bits.find(name)->second;
    const auto loc = locationFor(context, filename, declaration->getStart());
    for (int64_t index = 0; index < *width; ++index) {
      OperationState state(loc, UnpackBitOp::getOperationName());
      state.addOperands(input->second);
      state.addAttribute("index", builder.getI64IntegerAttr(index));
      state.addTypes(builder.getI1Type());
      const Value bit = builder.create(state)->getResult(0);
      const Value position =
          arith::ConstantIndexOp::create(builder, loc, index);
      memref::StoreOp::create(builder, loc, bit, destination.storage, position);
    }
    return success();
  }

  LogicalResult processQuantumDeclaration(
      qasm3Parser::QuantumDeclarationStatementContext* declaration) {
    const auto width =
        evaluateDesignator(declaration->qubitType()->designator());
    if (!width) {
      return fail(declaration, "qubit-register designator must be a constant "
                               "integer in the foundation frontend");
    }
    return declareQubits(declaration, declaration->Identifier()->getText(),
                         *width);
  }

  LogicalResult processOldStyleDeclaration(
      qasm3Parser::OldStyleDeclarationStatementContext* declaration) {
    if (version != SourceVersion::OpenQASM2) {
      return fail(
          declaration,
          "qreg/creg declarations require OpenQASM 2.0 compatibility mode");
    }
    if (declaration->CREG() != nullptr) {
      const auto width = evaluateDesignator(declaration->designator());
      if (!width) {
        return fail(declaration, "creg designator must be a constant integer");
      }
      return declareBits(declaration, declaration->Identifier()->getText(),
                         *width);
    }
    const auto width = evaluateDesignator(declaration->designator());
    if (!width) {
      return fail(declaration, "qreg designator must be a constant integer");
    }
    return declareQubits(declaration, declaration->Identifier()->getText(),
                         *width);
  }

  FailureOr<llvm::SmallVector<int64_t>>
  resolveBitIndices(qasm3Parser::IndexedIdentifierContext* identifier,
                    const llvm::StringRef fallbackName = {}) {
    const std::string name = identifier == nullptr
                                 ? fallbackName.str()
                                 : identifier->Identifier()->getText();
    const auto found = bits.find(name);
    if (found == bits.end()) {
      emitError(identifier == nullptr
                    ? UnknownLoc::get(&context)
                    : locationFor(context, filename, identifier->getStart()))
          << "unknown bit-register identifier '" << name << "'";
      return failure();
    }
    llvm::SmallVector<int64_t> indices;
    if (identifier == nullptr || identifier->indexOperator().empty()) {
      indices.reserve(found->second.width);
      for (int64_t index = 0; index < found->second.width; ++index) {
        indices.push_back(index);
      }
      return indices;
    }
    if (identifier->indexOperator().size() != 1 ||
        identifier->indexOperator(0)->expression().size() != 1 ||
        !identifier->indexOperator(0)->rangeExpression().empty()) {
      emitError(locationFor(context, filename, identifier->getStart()))
          << "only a single constant bit-register index is supported";
      return failure();
    }
    const auto index =
        evaluateInteger(identifier->indexOperator(0)->expression(0));
    if (!index || *index < 0 || *index >= found->second.width) {
      emitError(locationFor(context, filename, identifier->getStart()))
          << "bit-register index is not a valid constant index";
      return failure();
    }
    indices.push_back(*index);
    return indices;
  }

  LogicalResult
  processMeasurement(qasm3Parser::MeasureExpressionContext* measurement,
                     qasm3Parser::IndexedIdentifierContext* target,
                     const llvm::StringRef fallbackTarget = {}) {
    auto measured = resolveOperand(measurement->gateOperand());
    if (failed(measured)) {
      return failure();
    }
    llvm::SmallVector<int64_t> targetIndices;
    BitRegister* targetRegister = nullptr;
    std::string targetName;
    if (target != nullptr || !fallbackTarget.empty()) {
      targetName = target == nullptr ? fallbackTarget.str()
                                     : target->Identifier()->getText();
      auto indices = resolveBitIndices(target, fallbackTarget);
      if (failed(indices)) {
        return failure();
      }
      targetIndices = std::move(*indices);
      if (targetIndices.size() != measured->size()) {
        return fail(measurement,
                    "measurement source and target widths must match");
      }
      targetRegister = &bits.find(targetName)->second;
    }

    const auto loc = locationFor(context, filename, measurement->getStart());
    for (const auto [index, qubit] : llvm::enumerate(*measured)) {
      const Value result =
          qc::MeasureOp::create(builder, loc, qubit).getResult();
      if (targetRegister != nullptr) {
        const Value position =
            arith::ConstantIndexOp::create(builder, loc, targetIndices[index]);
        memref::StoreOp::create(builder, loc, result, targetRegister->storage,
                                position);
      }
    }
    return success();
  }

  LogicalResult
  processAssignment(qasm3Parser::AssignmentStatementContext* assignment) {
    if (assignment->measureExpression() != nullptr &&
        assignment->EQUALS() != nullptr) {
      return processMeasurement(assignment->measureExpression(),
                                assignment->indexedIdentifier());
    }
    if (assignment->expression() == nullptr ||
        !assignment->indexedIdentifier()->indexOperator().empty()) {
      return unsupported(assignment, "indexed or non-expression assignments");
    }
    const std::string name =
        assignment->indexedIdentifier()->Identifier()->getText();
    const auto variable = scalars.find(name);
    if (variable == scalars.end()) {
      return fail(assignment, llvm::Twine("unknown mutable scalar '")
                                  .concat(name)
                                  .concat("'")
                                  .str());
    }
    if (variable->second.immutable) {
      return fail(assignment, llvm::Twine("cannot assign to constant '")
                                  .concat(name)
                                  .concat("'")
                                  .str());
    }
    auto value = buildScalar(assignment->expression());
    if (failed(value)) {
      return failure();
    }
    const auto loc = locationFor(context, filename, assignment->getStart());
    auto cast = castScalar(*value, variable->second.type, loc);
    if (failed(cast)) {
      return failure();
    }
    memref::StoreOp::create(builder, loc, *cast, variable->second.storage,
                            ValueRange{});
    return success();
  }

  FailureOr<llvm::SmallVector<Value>>
  resolveOperand(qasm3Parser::GateOperandContext* operand) {
    if (operand->HardwareQubit() != nullptr) {
      emitError(locationFor(context, filename, operand->getStart()))
          << "hardware qubits are not yet supported by the typed frontend";
      return failure();
    }
    auto* identifier = operand->indexedIdentifier();
    const std::string name = identifier->Identifier()->getText();
    const auto found = qubits.find(name);
    if (found == qubits.end()) {
      emitError(locationFor(context, filename, operand->getStart()))
          << "unknown qubit identifier '" << name << "'";
      return failure();
    }
    if (identifier->indexOperator().empty()) {
      return llvm::SmallVector<Value>(found->second.begin(),
                                      found->second.end());
    }
    if (identifier->indexOperator().size() != 1 ||
        identifier->indexOperator(0)->expression().size() != 1 ||
        !identifier->indexOperator(0)->rangeExpression().empty() ||
        identifier->indexOperator(0)->setExpression() != nullptr) {
      emitError(locationFor(context, filename, operand->getStart()))
          << "only a single constant qubit index is supported in the "
             "foundation frontend";
      return failure();
    }
    auto index = evaluateInteger(identifier->indexOperator(0)->expression(0));
    if (!index || *index < 0 ||
        static_cast<size_t>(*index) >= found->second.size()) {
      emitError(locationFor(context, filename, operand->getStart()))
          << "qubit index is not a valid constant index";
      return failure();
    }
    return llvm::SmallVector<Value>{found->second[*index]};
  }

  FailureOr<Value> buildScalar(qasm3Parser::ExpressionContext* expr) {
    const auto loc = locationFor(context, filename, expr->getStart());
    if (auto* parenthesis =
            dynamic_cast<qasm3Parser::ParenthesisExpressionContext*>(expr)) {
      return buildScalar(parenthesis->expression());
    }
    if (auto integer = evaluateInteger(expr)) {
      return arith::ConstantIntOp::create(builder, loc, *integer, 64)
          .getResult();
    }
    if (auto* literal =
            dynamic_cast<qasm3Parser::LiteralExpressionContext*>(expr);
        literal != nullptr && literal->Identifier() != nullptr) {
      const auto found = parameters.find(literal->Identifier()->getText());
      if (found != parameters.end()) {
        return found->second;
      }
      const auto scalar = scalars.find(literal->Identifier()->getText());
      if (scalar != scalars.end()) {
        if (scalar->second.immutable) {
          return scalar->second.constant;
        }
        return memref::LoadOp::create(builder, loc, scalar->second.storage,
                                      ValueRange{})
            .getResult();
      }
      const auto bitRegister = bits.find(literal->Identifier()->getText());
      if (bitRegister != bits.end()) {
        if (bitRegister->second.width > 64) {
          emitError(loc)
              << "bit registers wider than 64 bits cannot yet be used as "
                 "scalar expressions";
          return failure();
        }
        Value packed = arith::ConstantIntOp::create(builder, loc, 0, 64);
        for (int64_t index = 0; index < bitRegister->second.width; ++index) {
          const Value position =
              arith::ConstantIndexOp::create(builder, loc, index);
          const Value bit = memref::LoadOp::create(
              builder, loc, bitRegister->second.storage, position);
          Value extended =
              arith::ExtUIOp::create(builder, loc, builder.getI64Type(), bit);
          if (index != 0) {
            const Value shift =
                arith::ConstantIntOp::create(builder, loc, index, 64);
            extended = arith::ShLIOp::create(builder, loc, extended, shift);
          }
          packed = arith::OrIOp::create(builder, loc, packed, extended);
        }
        return packed;
      }
      const StringRef name = literal->Identifier()->getText();
      if (name == "pi" || name == "tau" || name == "euler") {
        const double value = name == "pi"    ? std::numbers::pi
                             : name == "tau" ? 2.0 * std::numbers::pi
                                             : std::numbers::e;
        return arith::ConstantFloatOp::create(
                   builder, loc, builder.getF64Type(), llvm::APFloat(value))
            .getResult();
      }
      emitError(loc) << "unknown scalar identifier '"
                     << literal->Identifier()->getText() << "'";
      return failure();
    }
    if (auto* literal =
            dynamic_cast<qasm3Parser::LiteralExpressionContext*>(expr);
        literal != nullptr && literal->FloatLiteral() != nullptr) {
      const double value = std::stod(removeSeparators(literal->getText()));
      return arith::ConstantFloatOp::create(builder, loc, builder.getF64Type(),
                                            llvm::APFloat(value))
          .getResult();
    }
    if (auto* literal =
            dynamic_cast<qasm3Parser::LiteralExpressionContext*>(expr);
        literal != nullptr && literal->BooleanLiteral() != nullptr) {
      return arith::ConstantIntOp::create(builder, loc,
                                          literal->getText() == "true", 1)
          .getResult();
    }
    if (auto* unary =
            dynamic_cast<qasm3Parser::UnaryExpressionContext*>(expr)) {
      auto operand = buildScalar(unary->expression());
      if (failed(operand)) {
        return failure();
      }
      if (isa<FloatType>((*operand).getType())) {
        if (unary->MINUS() != nullptr) {
          return arith::NegFOp::create(builder, loc, *operand).getResult();
        }
        emitError(loc) << "only unary minus is valid for a floating operand";
        return failure();
      }
      if (!isa<IntegerType>((*operand).getType())) {
        emitError(loc) << "unary operator requires a numeric operand";
        return failure();
      }
      const auto type = cast<IntegerType>((*operand).getType());
      if (unary->MINUS() != nullptr) {
        const Value zero =
            arith::ConstantIntOp::create(builder, loc, 0, type.getWidth());
        return arith::SubIOp::create(builder, loc, zero, *operand).getResult();
      }
      if (unary->EXCLAMATION_POINT() != nullptr) {
        const Value zero =
            arith::ConstantIntOp::create(builder, loc, 0, type.getWidth());
        return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                     *operand, zero)
            .getResult();
      }
      const Value allOnes =
          arith::ConstantIntOp::create(builder, loc, -1, type.getWidth());
      return arith::XOrIOp::create(builder, loc, *operand, allOnes).getResult();
    }

    auto buildBinary = [&](auto* binary) -> FailureOr<Value> {
      auto lhs = buildScalar(binary->expression(0));
      auto rhs = buildScalar(binary->expression(1));
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      if ((*lhs).getType() != (*rhs).getType()) {
        if (isa<FloatType>((*lhs).getType()) &&
            isa<IntegerType>((*rhs).getType())) {
          *rhs = arith::SIToFPOp::create(builder, loc, (*lhs).getType(), *rhs);
        } else if (isa<IntegerType>((*lhs).getType()) &&
                   isa<FloatType>((*rhs).getType())) {
          *lhs = arith::SIToFPOp::create(builder, loc, (*rhs).getType(), *lhs);
        } else {
          emitError(loc) << "binary operator requires compatible operands";
          return failure();
        }
      }
      const StringRef op = binary->op->getText();
      if (isa<FloatType>((*lhs).getType())) {
        if (op == "+") {
          return arith::AddFOp::create(builder, loc, *lhs, *rhs).getResult();
        }
        if (op == "-") {
          return arith::SubFOp::create(builder, loc, *lhs, *rhs).getResult();
        }
        if (op == "*") {
          return arith::MulFOp::create(builder, loc, *lhs, *rhs).getResult();
        }
        if (op == "/") {
          return arith::DivFOp::create(builder, loc, *lhs, *rhs).getResult();
        }
        const auto predicate =
            llvm::StringSwitch<arith::CmpFPredicate>(op)
                .Case("==", arith::CmpFPredicate::OEQ)
                .Case("!=", arith::CmpFPredicate::UNE)
                .Case("<",
                      arith::CmpFPredicate::OLT) // spellchecker:disable-line
                .Case("<=", arith::CmpFPredicate::OLE)
                .Case(">", arith::CmpFPredicate::OGT)
                .Case(">=", arith::CmpFPredicate::OGE)
                .Default(arith::CmpFPredicate::AlwaysFalse);
        if (predicate == arith::CmpFPredicate::AlwaysFalse) {
          emitError(loc) << "operator '" << op
                         << "' is not valid for floating operands";
          return failure();
        }
        return arith::CmpFOp::create(builder, loc, predicate, *lhs, *rhs)
            .getResult();
      }
      if (!isa<IntegerType>((*lhs).getType())) {
        emitError(loc) << "binary operator requires numeric operands";
        return failure();
      }
      if (op == "+") {
        return arith::AddIOp::create(builder, loc, *lhs, *rhs).getResult();
      }
      if (op == "-") {
        return arith::SubIOp::create(builder, loc, *lhs, *rhs).getResult();
      }
      if (op == "*") {
        return arith::MulIOp::create(builder, loc, *lhs, *rhs).getResult();
      }
      if (op == "/") {
        return arith::DivSIOp::create(builder, loc, *lhs, *rhs).getResult();
      }
      if (op == "%") {
        return arith::RemSIOp::create(builder, loc, *lhs, *rhs).getResult();
      }
      if (op == "&" || op == "&&") {
        return arith::AndIOp::create(builder, loc, *lhs, *rhs).getResult();
      }
      if (op == "|" || op == "||") {
        return arith::OrIOp::create(builder, loc, *lhs, *rhs).getResult();
      }
      if (op == "^") {
        return arith::XOrIOp::create(builder, loc, *lhs, *rhs).getResult();
      }
      if (op == "<<") {
        return arith::ShLIOp::create(builder, loc, *lhs, *rhs).getResult();
      }
      if (op == ">>") {
        return arith::ShRSIOp::create(builder, loc, *lhs, *rhs).getResult();
      }
      const auto predicate = llvm::StringSwitch<arith::CmpIPredicate>(op)
                                 .Case("==", arith::CmpIPredicate::eq)
                                 .Case("!=", arith::CmpIPredicate::ne)
                                 .Case("<", arith::CmpIPredicate::slt)
                                 .Case("<=", arith::CmpIPredicate::sle)
                                 .Case(">", arith::CmpIPredicate::sgt)
                                 .Case(">=", arith::CmpIPredicate::sge)
                                 .Default(arith::CmpIPredicate::eq);
      return arith::CmpIOp::create(builder, loc, predicate, *lhs, *rhs)
          .getResult();
    };
    if (auto* binary =
            dynamic_cast<qasm3Parser::AdditiveExpressionContext*>(expr)) {
      return buildBinary(binary);
    }
    if (auto* binary =
            dynamic_cast<qasm3Parser::MultiplicativeExpressionContext*>(expr)) {
      return buildBinary(binary);
    }
    if (auto* binary =
            dynamic_cast<qasm3Parser::EqualityExpressionContext*>(expr)) {
      return buildBinary(binary);
    }
    if (auto* binary =
            dynamic_cast<qasm3Parser::ComparisonExpressionContext*>(expr)) {
      return buildBinary(binary);
    }
    if (auto* binary =
            dynamic_cast<qasm3Parser::BitwiseAndExpressionContext*>(expr)) {
      return buildBinary(binary);
    }
    if (auto* binary =
            dynamic_cast<qasm3Parser::BitwiseOrExpressionContext*>(expr)) {
      return buildBinary(binary);
    }
    if (auto* binary =
            dynamic_cast<qasm3Parser::BitwiseXorExpressionContext*>(expr)) {
      return buildBinary(binary);
    }
    if (auto* binary =
            dynamic_cast<qasm3Parser::LogicalAndExpressionContext*>(expr)) {
      return buildBinary(binary);
    }
    if (auto* binary =
            dynamic_cast<qasm3Parser::LogicalOrExpressionContext*>(expr)) {
      return buildBinary(binary);
    }
    if (auto* binary =
            dynamic_cast<qasm3Parser::BitshiftExpressionContext*>(expr)) {
      return buildBinary(binary);
    }
    if (auto* power =
            dynamic_cast<qasm3Parser::PowerExpressionContext*>(expr)) {
      auto lhs = buildScalar(power->expression(0));
      auto rhs = buildScalar(power->expression(1));
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      if (isa<FloatType>((*lhs).getType()) &&
          isa<IntegerType>((*rhs).getType())) {
        *rhs = arith::SIToFPOp::create(builder, loc, (*lhs).getType(), *rhs);
      }
      if ((*lhs).getType() != (*rhs).getType()) {
        emitError(loc) << "power operands must have compatible numeric types";
        return failure();
      }
      if (isa<FloatType>((*lhs).getType())) {
        return math::PowFOp::create(builder, loc, *lhs, *rhs).getResult();
      }
      if (isa<IntegerType>((*lhs).getType())) {
        return math::IPowIOp::create(builder, loc, *lhs, *rhs).getResult();
      }
    }
    emitError(loc) << "expression is not yet supported by the typed frontend";
    return failure();
  }

  LogicalResult processScopedBody(qasm3Parser::StatementOrScopeContext* body) {
    if (body->scope() == nullptr) {
      return processStatement(body->statement());
    }
    for (auto* item : body->scope()->statementOrScope()) {
      if (item->scope() != nullptr ||
          failed(processStatement(item->statement()))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult processIf(qasm3Parser::IfStatementContext* statement) {
    auto condition = buildScalar(statement->expression());
    if (failed(condition)) {
      return failure();
    }
    if (!(*condition).getType().isInteger(1)) {
      if (!isa<IntegerType>((*condition).getType())) {
        return fail(statement, "if condition must be boolean or integer");
      }
      const auto type = cast<IntegerType>((*condition).getType());
      const Value zero = arith::ConstantIntOp::create(
          builder, (*condition).getLoc(), 0, type.getWidth());
      *condition =
          arith::CmpIOp::create(builder, (*condition).getLoc(),
                                arith::CmpIPredicate::ne, *condition, zero);
    }
    const bool hasElse = statement->else_body != nullptr;
    auto ifOp = scf::IfOp::create(
        builder, locationFor(context, filename, statement->getStart()),
        *condition, hasElse);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    if (failed(processScopedBody(statement->if_body))) {
      ifOp.erase();
      return failure();
    }
    if (hasElse) {
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      if (failed(processScopedBody(statement->else_body))) {
        ifOp.erase();
        return failure();
      }
    }
    return success();
  }

  LogicalResult processWhile(qasm3Parser::WhileStatementContext* statement) {
    LogicalResult result = success();
    auto whileOp = scf::WhileOp::create(
        builder, locationFor(context, filename, statement->getStart()),
        TypeRange{}, ValueRange{},
        [&](OpBuilder& nestedBuilder, const Location loc, ValueRange) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(nestedBuilder.getInsertionBlock());
          auto condition = buildScalar(statement->expression());
          if (failed(condition)) {
            result = failure();
            const Value falseValue =
                arith::ConstantIntOp::create(builder, loc, 0, 1);
            scf::ConditionOp::create(builder, loc, falseValue, ValueRange{});
            return;
          }
          Value conditionValue = *condition;
          if (!conditionValue.getType().isInteger(1)) {
            if (!isa<IntegerType>(conditionValue.getType())) {
              emitError(conditionValue.getLoc())
                  << "while condition must be boolean or integer";
              result = failure();
              conditionValue = arith::ConstantIntOp::create(builder, loc, 0, 1);
            } else {
              const auto type = cast<IntegerType>(conditionValue.getType());
              const Value zero = arith::ConstantIntOp::create(builder, loc, 0,
                                                              type.getWidth());
              conditionValue = arith::CmpIOp::create(
                  builder, loc, arith::CmpIPredicate::ne, conditionValue, zero);
            }
          }
          scf::ConditionOp::create(builder, loc, conditionValue, ValueRange{});
        },
        [&](OpBuilder& nestedBuilder, const Location loc, ValueRange) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(nestedBuilder.getInsertionBlock());
          if (failed(processScopedBody(statement->body))) {
            result = failure();
          }
          scf::YieldOp::create(builder, loc);
        });
    if (failed(result)) {
      whileOp.erase();
    }
    return result;
  }

  LogicalResult processGateCall(qasm3Parser::GateCallStatementContext* call) {
    const std::string name = call->Identifier() != nullptr
                                 ? call->Identifier()->getText()
                                 : "gphase";
    const auto signature = gates.find(name);
    if (signature == gates.end()) {
      return fail(call, llvm::Twine("unknown gate '")
                            .concat(name)
                            .concat("'; include the appropriate standard "
                                    "library or declare it before use")
                            .str());
    }
    if (name == currentGate) {
      return fail(call,
                  llvm::Twine("recursive gate definition is not allowed: ")
                      .concat(name)
                      .str());
    }

    llvm::SmallVector<Value> parameters;
    if (call->expressionList() != nullptr) {
      for (auto* expression : call->expressionList()->expression()) {
        auto value = buildScalar(expression);
        if (failed(value)) {
          return failure();
        }
        if (isa<IntegerType>((*value).getType())) {
          parameters.push_back(arith::SIToFPOp::create(
              builder, (*value).getLoc(), builder.getF64Type(), *value));
        } else {
          parameters.push_back(*value);
        }
      }
    }
    if (parameters.size() != signature->second.numParameters) {
      return fail(call, "gate parameter count does not match its declaration");
    }

    llvm::SmallVector<llvm::SmallVector<Value>> operands;
    if (call->gateOperandList() != nullptr) {
      for (auto* operand : call->gateOperandList()->gateOperand()) {
        auto resolved = resolveOperand(operand);
        if (failed(resolved)) {
          return failure();
        }
        operands.push_back(std::move(*resolved));
      }
    }
    size_t minimumQubits = signature->second.numQubits;
    bool dynamicControlCount = false;
    for (auto* modifier : call->gateModifier()) {
      if (modifier->CTRL() == nullptr && modifier->NEGCTRL() == nullptr) {
        continue;
      }
      if (modifier->expression() == nullptr) {
        ++minimumQubits;
        continue;
      }
      const auto count = evaluateInteger(modifier->expression());
      if (!count) {
        dynamicControlCount = true;
      } else if (*count <= 0) {
        return fail(modifier, "control-modifier count must be positive");
      } else {
        minimumQubits += static_cast<size_t>(*count);
      }
    }
    if ((!dynamicControlCount && operands.size() != minimumQubits) ||
        (dynamicControlCount && operands.size() < minimumQubits)) {
      return fail(call, "gate qubit-operand count does not match its "
                        "declaration and modifiers");
    }

    size_t broadcastWidth = 1;
    for (const auto& operand : operands) {
      if (operand.size() > 1) {
        if (broadcastWidth != 1 && broadcastWidth != operand.size()) {
          return fail(call,
                      "broadcasted qubit registers must have equal widths");
        }
        broadcastWidth = operand.size();
      }
    }

    llvm::SmallVector<int32_t> modifierKinds;
    llvm::SmallVector<int32_t> modifierIndices;
    llvm::SmallVector<Value> modifierOperands;
    for (auto* modifier : call->gateModifier()) {
      GateModifierKind kind = GateModifierKind::inv;
      if (modifier->CTRL() != nullptr) {
        kind = GateModifierKind::ctrl;
      } else if (modifier->NEGCTRL() != nullptr) {
        kind = GateModifierKind::negctrl;
      } else if (modifier->POW() != nullptr) {
        kind = GateModifierKind::pow;
      }
      modifierKinds.push_back(static_cast<int32_t>(kind));
      if (modifier->expression() != nullptr) {
        auto value = buildScalar(modifier->expression());
        if (failed(value)) {
          return failure();
        }
        modifierIndices.push_back(
            static_cast<int32_t>(modifierOperands.size()));
        modifierOperands.push_back(*value);
      } else {
        modifierIndices.push_back(-1);
      }
    }

    const auto loc = locationFor(context, filename, call->getStart());
    for (size_t i = 0; i < broadcastWidth; ++i) {
      llvm::SmallVector<Value> callQubits;
      for (const auto& operand : operands) {
        callQubits.push_back(operand.size() == 1 ? operand.front()
                                                 : operand[i]);
      }
      OperationState state(loc, ApplyGateOp::getOperationName());
      state.addOperands(parameters);
      state.addOperands(callQubits);
      state.addOperands(modifierOperands);
      state.addAttribute("callee", FlatSymbolRefAttr::get(&context, name));
      state.addAttribute("modifier_kinds",
                         DenseI32ArrayAttr::get(&context, modifierKinds));
      state.addAttribute("modifier_operand_indices",
                         DenseI32ArrayAttr::get(&context, modifierIndices));
      state.addAttribute(
          "operandSegmentSizes",
          DenseI32ArrayAttr::get(
              &context, {static_cast<int32_t>(parameters.size()),
                         static_cast<int32_t>(callQubits.size()),
                         static_cast<int32_t>(modifierOperands.size())}));
      builder.create(state);
    }
    return success();
  }

  LogicalResult processReset(qasm3Parser::ResetStatementContext* reset) {
    auto operands = resolveOperand(reset->gateOperand());
    if (failed(operands)) {
      return failure();
    }
    const auto loc = locationFor(context, filename, reset->getStart());
    for (const Value qubit : *operands) {
      qc::ResetOp::create(builder, loc, qubit);
    }
    return success();
  }

  LogicalResult processBarrier(qasm3Parser::BarrierStatementContext* barrier) {
    llvm::SmallVector<Value> qubitValues;
    if (barrier->gateOperandList() == nullptr) {
      for (const auto& entry : qubits) {
        qubitValues.append(entry.second.begin(), entry.second.end());
      }
    } else {
      for (auto* operand : barrier->gateOperandList()->gateOperand()) {
        auto values = resolveOperand(operand);
        if (failed(values)) {
          return failure();
        }
        qubitValues.append(values->begin(), values->end());
      }
    }
    qc::BarrierOp::create(builder,
                          locationFor(context, filename, barrier->getStart()),
                          qubitValues);
    return success();
  }

  LogicalResult processFor(qasm3Parser::ForStatementContext* loop) {
    if (loop->rangeExpression() == nullptr) {
      return unsupported(loop, "non-range for loops");
    }
    const auto expressions = loop->rangeExpression()->expression();
    if (expressions.size() < 2 || expressions.size() > 3) {
      return fail(loop,
                  "for-loop ranges require explicit start and stop values");
    }
    const auto loc = locationFor(context, filename, loop->getStart());
    auto start = buildScalar(expressions[0]);
    auto stop = buildScalar(expressions.back());
    FailureOr<Value> step =
        expressions.size() == 3
            ? buildScalar(expressions[1])
            : FailureOr<Value>(arith::ConstantIntOp::create(builder, loc, 1, 64)
                                   .getResult());
    if (failed(start) || failed(stop) || failed(step) ||
        !isa<IntegerType>((*start).getType()) ||
        (*start).getType() != (*stop).getType() ||
        (*start).getType() != (*step).getType()) {
      return fail(loop,
                  "for-loop range values must have matching integer types");
    }
    if (auto constant = (*step).getDefiningOp<arith::ConstantIntOp>();
        constant && constant.value() == 0) {
      return fail(loop, "OpenQASM range step cannot be zero");
    }

    OperationState state(loc, ForOp::getOperationName());
    state.addOperands({*start, *stop, *step});
    Region* body = state.addRegion();
    body->push_back(new Block());
    body->front().addArgument((*start).getType(), loc);
    auto* operation = builder.create(state);
    body = &cast<ForOp>(operation).getBody();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&body->front());
    const std::string inductionName = loop->Identifier()->getText();
    const auto previous = parameters.find(inductionName);
    const bool wasDeclared = previous != parameters.end();
    Value previousValue;
    if (wasDeclared) {
      previousValue = previous->second;
    }
    parameters[inductionName] = body->front().getArgument(0);
    const auto restoreInduction = [&]() {
      if (wasDeclared) {
        parameters[inductionName] = previousValue;
      } else {
        parameters.erase(inductionName);
      }
    };
    auto* bodyItem = loop->body;
    if (bodyItem->scope() != nullptr) {
      for (auto* item : bodyItem->scope()->statementOrScope()) {
        if (item->scope() != nullptr ||
            failed(processStatement(item->statement()))) {
          restoreInduction();
          builder.clearInsertionPoint();
          operation->erase();
          return failure();
        }
      }
    } else if (failed(processStatement(bodyItem->statement()))) {
      restoreInduction();
      builder.clearInsertionPoint();
      operation->erase();
      return failure();
    }
    YieldOp::create(builder, loc);
    restoreInduction();
    return success();
  }

  MLIRContext& context;
  llvm::SourceMgr& sourceMgr;
  std::string filename;
  SourceVersion version;
  llvm::SmallVector<std::string> includeDirectories;
  llvm::SmallVector<std::string> includeStack;
  OpBuilder builder;
  OwningOpRef<ModuleOp> module;
  func::FuncOp main;
  Block* entry = nullptr;
  llvm::StringMap<llvm::SmallVector<Value>> qubits;
  struct BitRegister {
    Value storage;
    int64_t width;
  };
  llvm::StringMap<BitRegister> bits;
  struct ScalarVariable {
    Value storage;
    Type type;
    Value constant;
    bool immutable;
  };
  llvm::StringMap<ScalarVariable> scalars;
  llvm::StringMap<Value> inputValues;
  llvm::SmallVector<std::string> outputNames;
  llvm::StringMap<Value> parameters;
  llvm::StringMap<GateSignature> gates;
  std::string currentGate;
};

std::optional<SourceVersion> parseVersion(qasm3Parser::ProgramContext* program,
                                          MLIRContext& context,
                                          llvm::StringRef filename) {
  if (program->version() == nullptr) {
    return SourceVersion::OpenQASM31;
  }
  const std::string version = program->version()->VersionSpecifier()->getText();
  if (version == "3.1") {
    return SourceVersion::OpenQASM31;
  }
  if (version == "2.0") {
    return SourceVersion::OpenQASM2;
  }
  emitError(locationFor(context, filename, program->version()->getStart()))
      << "unsupported OpenQASM version '" << version
      << "'; supported versions are 3.1 and 2.0 compatibility mode";
  return std::nullopt;
}

LogicalResult validateVersionPlacement(antlr4::CommonTokenStream& tokens,
                                       MLIRContext& context,
                                       const llvm::StringRef filename) {
  tokens.fill();
  antlr4::Token* firstSourceToken = nullptr;
  llvm::SmallVector<antlr4::Token*> declarations;
  for (antlr4::Token* token : tokens.getTokens()) {
    if (token->getType() == antlr4::Token::EOF ||
        token->getChannel() != antlr4::Token::DEFAULT_CHANNEL) {
      continue;
    }
    if (firstSourceToken == nullptr) {
      firstSourceToken = token;
    }
    if (token->getType() == qasm3Lexer::OPENQASM) {
      declarations.push_back(token);
    }
  }
  if (declarations.size() > 1) {
    emitError(locationFor(context, filename, declarations[1]))
        << "OpenQASM source may contain only one version declaration";
    return failure();
  }
  if (!declarations.empty() && declarations.front() != firstSourceToken) {
    emitError(locationFor(context, filename, declarations.front()))
        << "OpenQASM version declaration must be the first non-comment source "
           "item";
    return failure();
  }
  tokens.seek(0);
  return success();
}

} // namespace

OwningOpRef<ModuleOp>
translateOpenQASMToOQ3(llvm::SourceMgr& sourceMgr, MLIRContext& context,
                       const OpenQASMTranslationOptions& options) {
  const auto* buffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  const llvm::StringRef filename = buffer->getBufferIdentifier();
  antlr4::ANTLRInputStream input(buffer->getBuffer().str());
  qasm3Lexer lexer(&input);
  antlr4::CommonTokenStream tokens(&lexer);
  qasm3Parser parser(&tokens);

  DiagnosticErrorListener errors(context, filename);
  lexer.removeErrorListeners();
  parser.removeErrorListeners();
  lexer.addErrorListener(&errors);
  parser.addErrorListener(&errors);
  if (failed(validateVersionPlacement(tokens, context, filename))) {
    return {};
  }
  auto* program = parser.program();
  if (errors.failed() || parser.getNumberOfSyntaxErrors() != 0) {
    return {};
  }

  auto version = parseVersion(program, context, filename);
  if (!version) {
    return {};
  }
  return SemanticBuilder(context, sourceMgr, filename, *version, options)
      .build(program);
}

OwningOpRef<ModuleOp>
translateOpenQASMToOQ3(const llvm::StringRef source, MLIRContext& context,
                       const OpenQASMTranslationOptions& options) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(source, "<openqasm>"),
      llvm::SMLoc());
  return translateOpenQASMToOQ3(sourceMgr, context, options);
}

} // namespace mlir::oq3

/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for
 * more information.
 */

#pragma once

#include "Scanner.hpp"
#include "Statement_fwd.hpp"
#include "Token.hpp"
#include "Types_fwd.hpp"

#include <istream>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <utility>
#include <vector>

namespace qc {
// forward declarations
class Permutation;
} // namespace qc

namespace qasm3 {
class Parser final {
  struct ScannerState {
  private:
    std::unique_ptr<std::istream> is;

  public:
    Token last{0, 0};
    Token t{0, 0};
    Token next{0, 0};
    std::unique_ptr<Scanner> scanner;
    std::optional<std::string> filename;
    bool isImplicitInclude;

    bool scan() {
      last = t;
      t = next;
      next = scanner->next();

      return t.kind != Token::Kind::Eof;
    }

    explicit ScannerState(
        std::istream* in,
        std::optional<std::string> debugFilename = std::nullopt,
        const bool implicitInclude = false)
        : scanner(std::make_unique<Scanner>(in)),
          filename(std::move(debugFilename)),
          isImplicitInclude(implicitInclude) {
      scan();
    }

    explicit ScannerState(
        std::unique_ptr<std::istream> in,
        std::optional<std::string> debugFilename = std::nullopt,
        const bool implicitInclude = false)
        : is(std::move(in)), scanner(std::make_unique<Scanner>(is.get())),
          filename(std::move(debugFilename)),
          isImplicitInclude(implicitInclude) {
      scan();
    }
  };

  std::stack<ScannerState> scanner;
  std::shared_ptr<DebugInfo> includeDebugInfo{nullptr};

  [[noreturn]] void error(const Token& token, const std::string& msg);

  [[nodiscard]] Token last() const;

  [[nodiscard]] Token current() const;

  [[nodiscard]] Token peek() const;

  Token expect(const Token::Kind& expected,
               const std::optional<std::string>& context = std::nullopt);

public:
  explicit Parser(std::istream& is, bool implicitlyIncludeStdgates = true);

  ~Parser() = default;

  std::shared_ptr<VersionDeclaration> parseVersionDeclaration();

  std::vector<std::shared_ptr<Statement>> parseProgram();

  std::shared_ptr<Statement> parseStatement();

  std::shared_ptr<QuantumStatement> parseQuantumStatement();

  void parseInclude();

  std::shared_ptr<AssignmentStatement> parseAssignmentStatement();

  std::shared_ptr<AssignmentStatement> parseMeasureStatement();

  std::shared_ptr<ResetStatement> parseResetStatement();

  std::shared_ptr<BarrierStatement> parseBarrierStatement();

  std::shared_ptr<Statement> parseDeclaration(bool isConst);

  std::shared_ptr<GateDeclaration> parseGateDefinition();

  std::shared_ptr<GateDeclaration> parseOpaqueGateDefinition();

  std::shared_ptr<GateCallStatement> parseGateCallStatement();

  std::shared_ptr<GateModifier> parseGateModifier();

  std::shared_ptr<IndexOperator> parseIndexOperator();

  std::shared_ptr<IndexedIdentifier> parseIndexedIdentifier();

  std::shared_ptr<GateOperand> parseGateOperand();

  std::shared_ptr<DeclarationExpression> parseDeclarationExpression();

  std::shared_ptr<MeasureExpression> parseMeasureExpression();

  std::shared_ptr<Expression> exponentiation();

  std::shared_ptr<Expression> factor();

  std::shared_ptr<Expression> term();

  std::shared_ptr<Expression> comparison();

  std::shared_ptr<Expression> parseExpression();

  std::shared_ptr<IdentifierList> parseIdentifierList();

  std::pair<std::shared_ptr<Type<std::shared_ptr<Expression>>>, bool>
  parseType();

  std::shared_ptr<Expression> parseTypeDesignator();

  static qc::Permutation parsePermutation(std::string s);

  void scan();

  std::shared_ptr<DebugInfo> makeDebugInfo(Token const& begin,
                                           Token const& /*end*/);

  std::shared_ptr<DebugInfo> makeDebugInfo(Token const& token);

  [[nodiscard]] bool isAtEnd() const {
    return current().kind == Token::Kind::Eof;
  }
  std::shared_ptr<IfStatement> parseIfStatement();
  std::vector<std::shared_ptr<Statement>> parseBlockOrStatement();
};

} // namespace qasm3

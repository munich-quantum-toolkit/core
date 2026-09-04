/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Target/OpenQASM/Frontend.h"

#include <gtest/gtest.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/VirtualFileSystem.h>

#include <array>
#include <cstddef>
#include <string>
#include <variant>

using namespace mlir;

namespace {

TEST(OpenQASMFrontendTest, PreservesExactAndOptionalVersionSemantics) {
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM("qubit q; x q;"));
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM("OPENQASM 3; qubit q; x q;"));

  auto unsupported =
      oq3::frontend::analyzeOpenQASM("OPENQASM 3.10; qubit q; x q;");
  ASSERT_FALSE(unsupported);
  ASSERT_FALSE(unsupported.diagnostics.empty());
  EXPECT_NE(unsupported.diagnostics.front().message.find("3.10"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsUnsupportedOpenQASM3MinorVersions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.2;
qubit q;
x q;
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("Unsupported OpenQASM"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, StrictPolicyRequiresTheStandardLibraryInclude) {
  constexpr llvm::StringLiteral withoutInclude = R"qasm(
OPENQASM 3.0;
qubit q;
x q;
)qasm";
  constexpr llvm::StringLiteral withInclude = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
x q;
)qasm";
  oq3::frontend::FrontendOptions strict;
  strict.gatePolicy = oq3::frontend::GatePolicy::Strict;

  EXPECT_FALSE(oq3::frontend::analyzeOpenQASM(withoutInclude, strict));
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(withInclude, strict));
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(withoutInclude));
}

TEST(OpenQASMFrontendTest, PreservesSourceNamesInSemanticDiagnostics) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.0;\nqubit q;\nunknown q;\n", "fixture.qasm"),
      llvm::SMLoc());
  auto analyzed = oq3::frontend::analyzeOpenQASM(sourceMgr);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_EQ(analyzed.diagnostics.front().location.filename, "fixture.qasm");
  EXPECT_EQ(analyzed.diagnostics.front().location.line, 3);
}

TEST(OpenQASMFrontendTest, LocatesVersionAndOutputDiagnosticsPrecisely) {
  llvm::SourceMgr versionSources;
  versionSources.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("OPENQASM 3.2;\nqubit q;\n",
                                           "unsupported-version.qasm"),
      llvm::SMLoc());
  auto version = oq3::frontend::analyzeOpenQASM(versionSources);
  ASSERT_FALSE(version);
  ASSERT_FALSE(version.diagnostics.empty());
  EXPECT_EQ(version.diagnostics.front().location.filename,
            "unsupported-version.qasm");
  EXPECT_EQ(version.diagnostics.front().location.line, 1);

  llvm::SourceMgr outputSources;
  outputSources.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.1;\nqubit q;\noutput bit result;\n",
          "incomplete-output.qasm"),
      llvm::SMLoc());
  auto output = oq3::frontend::analyzeOpenQASM(outputSources);
  ASSERT_FALSE(output);
  ASSERT_FALSE(output.diagnostics.empty());
  EXPECT_EQ(output.diagnostics.front().location.filename,
            "incomplete-output.qasm");
  EXPECT_EQ(output.diagnostics.front().location.line, 3);
}

TEST(OpenQASMFrontendTest, TracksLexicalScopeAndEnclosingAssignments) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
int value = 1;
if (true) {
  int value = 2;
  value += 3;
} else {
  value = 4;
}
value += 5;
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->scalars.size(), 2);

  size_t outerAssignments = 0;
  size_t innerAssignments = 0;
  for (const auto& statement : analyzed.program->statements) {
    if (const auto* assignment =
            std::get_if<oq3::frontend::ScalarAssignmentStatement>(
                &statement.data)) {
      outerAssignments += static_cast<size_t>(assignment->scalar == 0);
      innerAssignments += static_cast<size_t>(assignment->scalar == 1);
    }
  }
  EXPECT_EQ(outerAssignments, 2);
  EXPECT_EQ(innerAssignments, 1);
}

TEST(OpenQASMFrontendTest, OwnsAndAnalyzesProvidedIncludeBuffers) {
  oq3::frontend::ParseResult parsed;
  {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBufferCopy(R"qasm(
OPENQASM 3.1;
include "custom.inc";
qubit q;
custom q;
bit result = measure q;
)qasm",
                                             "main.qasm"),
        llvm::SMLoc());
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBufferCopy(R"qasm(
gate custom q { x q; }
)qasm",
                                             "custom.inc"),
        llvm::SMLoc());
    parsed = oq3::frontend::parseOpenQASM(sourceMgr);
  }

  ASSERT_TRUE(parsed) << parsed.diagnostics.front().message;
  auto analyzed = oq3::frontend::analyzeOpenQASM(*parsed.program);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->gates.size(), 1);
  EXPECT_EQ(analyzed.program->gates.front().name, "custom");
  EXPECT_EQ(analyzed.program->gates.front().location.filename, "custom.inc");
}

TEST(OpenQASMFrontendTest, ExpandsNestedIncludesAtTheirSourceLocations) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(R"qasm(
OPENQASM 3.1;
include "outer.inc";
int result = outer + nested;
)qasm",
                                           "main.qasm"),
      llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(R"qasm(
int outer = 1;
include "nested.inc";
int after = nested;
)qasm",
                                           "outer.inc"),
      llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("int nested = 2;\n", "nested.inc"),
      llvm::SMLoc());

  auto analyzed = oq3::frontend::analyzeOpenQASM(sourceMgr);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->scalars.size(), 4);
  EXPECT_EQ(analyzed.program->scalars[0].name, "outer");
  EXPECT_EQ(analyzed.program->scalars[1].name, "nested");
  EXPECT_EQ(analyzed.program->scalars[2].name, "after");
  EXPECT_EQ(analyzed.program->scalars[3].name, "result");
}

TEST(OpenQASMFrontendTest, PreservesNestedIncludeStacksInDiagnostics) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.1;\ninclude \"outer.inc\";\n", "main.qasm"),
      llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "include \"nested.inc\";\n", "outer.inc"),
                               llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "int value = missing;\n", "nested.inc"),
                               llvm::SMLoc());

  auto analyzed = oq3::frontend::analyzeOpenQASM(sourceMgr);
  ASSERT_FALSE(analyzed);
  ASSERT_EQ(analyzed.diagnostics.size(), 1);
  const auto& location = analyzed.diagnostics.front().location;
  EXPECT_EQ(location.filename, "nested.inc");
  EXPECT_EQ(location.line, 1);
  ASSERT_EQ(location.includeStack.size(), 2);
  EXPECT_EQ(location.includeStack[0].filename, "outer.inc");
  EXPECT_EQ(location.includeStack[0].line, 1);
  EXPECT_EQ(location.includeStack[1].filename, "main.qasm");
  EXPECT_EQ(location.includeStack[1].line, 2);
}

TEST(OpenQASMFrontendTest, PreservesDistinctProvenanceForRepeatedIncludes) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.1;\ninclude \"a.inc\";\ninclude \"b.inc\";\n",
          "main.qasm"),
      llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "include \"shared.inc\";\n", "a.inc"),
                               llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "include \"shared.inc\";\n", "b.inc"),
                               llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "int duplicate = 1;\n", "shared.inc"),
                               llvm::SMLoc());

  auto analyzed = oq3::frontend::analyzeOpenQASM(sourceMgr);
  ASSERT_FALSE(analyzed);
  ASSERT_EQ(analyzed.diagnostics.size(), 1);
  const auto& location = analyzed.diagnostics.front().location;
  EXPECT_EQ(location.filename, "shared.inc");
  ASSERT_EQ(location.includeStack.size(), 2);
  EXPECT_EQ(location.includeStack[0].filename, "b.inc");
  EXPECT_EQ(location.includeStack[0].line, 1);
  EXPECT_EQ(location.includeStack[1].filename, "main.qasm");
  EXPECT_EQ(location.includeStack[1].line, 3);
}

TEST(OpenQASMFrontendTest, RejectsRecursiveIncludesResolvedThroughSearchPaths) {
  auto fileSystem = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  ASSERT_TRUE(fileSystem->addFile(
      "/includes/recursive.inc", 0,
      llvm::MemoryBuffer::getMemBuffer("include \"recursive.inc\";")));

  llvm::SourceMgr sourceMgr;
  sourceMgr.setVirtualFileSystem(fileSystem);
  sourceMgr.setIncludeDirs({"/includes"});
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.1; include \"recursive.inc\";", "main.qasm"),
      llvm::SMLoc());

  auto parsed = oq3::frontend::parseOpenQASM(sourceMgr);
  ASSERT_FALSE(parsed);
  ASSERT_FALSE(parsed.diagnostics.empty());
  EXPECT_NE(parsed.diagnostics.front().message.find("recursive include"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, LimitsIncludeNesting) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.1; include \"depth-0.inc\";", "main.qasm"),
      llvm::SMLoc());
  for (size_t index = 0; index <= 64; ++index) {
    std::string source;
    if (index == 64) {
      source = "int leaf = 1;";
    } else {
      source = "include \"depth-" + std::to_string(index + 1) + ".inc\";";
    }
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBufferCopy(
            source, "depth-" + std::to_string(index) + ".inc"),
        llvm::SMLoc());
  }

  auto parsed = oq3::frontend::parseOpenQASM(sourceMgr);
  ASSERT_FALSE(parsed);
  ASSERT_FALSE(parsed.diagnostics.empty());
  EXPECT_NE(parsed.diagnostics.front().message.find("include nesting"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, LimitsTextualIncludeExpansion) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.1; include \"level-0.inc\";", "main.qasm"),
      llvm::SMLoc());
  for (size_t index = 0; index < 21; ++index) {
    std::string source;
    if (index == 20) {
      source = "int leaf = 1;";
    } else {
      const auto next = "level-" + std::to_string(index + 1) + ".inc";
      source.append("include \"")
          .append(next)
          .append("\"; include \"")
          .append(next)
          .append("\";");
    }
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBufferCopy(
            source, "level-" + std::to_string(index) + ".inc"),
        llvm::SMLoc());
  }

  auto parsed = oq3::frontend::parseOpenQASM(sourceMgr);
  ASSERT_FALSE(parsed);
  ASSERT_FALSE(parsed.diagnostics.empty());
  EXPECT_NE(parsed.diagnostics.front().message.find("statement limit"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, EnforcesUnicodeIdentifierCategoriesAndUtf8) {
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      "OPENQASM 3.1; int θ = 1; int Ångström = θ;"));

  auto symbol = oq3::frontend::analyzeOpenQASM("OPENQASM 3.1; int 💥 = 1;");
  ASSERT_FALSE(symbol);
  ASSERT_FALSE(symbol.diagnostics.empty());

  std::string invalid = "OPENQASM 3.1; int ";
  invalid.push_back(static_cast<char>(0xC3));
  invalid += " = 1;";
  auto malformed = oq3::frontend::analyzeOpenQASM(invalid);
  ASSERT_FALSE(malformed);
  ASSERT_FALSE(malformed.diagnostics.empty());
}

TEST(OpenQASMFrontendTest, ResolvesIncludedNamesWithoutBasenameAliasing) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(R"qasm(
OPENQASM 3.1;
include "a/defs.inc";
include "b/defs.inc";
counter += 1;
qubit q;
if (enabled) { x q; }
bit result = measure q;
)qasm",
                                           "main.qasm"),
      llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("int counter = 0;\n", "a/defs.inc"),
      llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "bool enabled = true;\n", "b/defs.inc"),
                               llvm::SMLoc());

  auto analyzed = oq3::frontend::analyzeOpenQASM(sourceMgr);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->scalars.size(), 2);
  EXPECT_EQ(analyzed.program->scalars[0].name, "counter");
  EXPECT_EQ(analyzed.program->scalars[1].name, "enabled");
}

TEST(OpenQASMFrontendTest, ExpandsEveryTextualIncludeOccurrence) {
  llvm::SourceMgr sources;
  sources.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(R"qasm(
OPENQASM 3.1;
qubit q;
include "operations.inc";
include "operations.inc";
bit result = measure q;
)qasm",
                                                                  "main.qasm"),
                             llvm::SMLoc());
  sources.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("x q;\n", "operations.inc"),
      llvm::SMLoc());

  auto analyzed = oq3::frontend::analyzeOpenQASM(sources);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  size_t applications = 0;
  for (const auto& statement : analyzed.program->statements) {
    applications += static_cast<size_t>(
        std::holds_alternative<oq3::frontend::GateApplication>(statement.data));
  }
  EXPECT_EQ(applications, 2);
}

TEST(OpenQASMFrontendTest, RejectsRecursiveAndRepeatedStandardIncludes) {
  llvm::SourceMgr recursiveSources;
  recursiveSources.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("OPENQASM 3.1; include \"a.inc\";",
                                           "main.qasm"),
      llvm::SMLoc());
  recursiveSources.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("include \"b.inc\";", "a.inc"),
      llvm::SMLoc());
  recursiveSources.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("include \"a.inc\";", "b.inc"),
      llvm::SMLoc());
  auto recursive = oq3::frontend::parseOpenQASM(recursiveSources);
  ASSERT_FALSE(recursive);
  ASSERT_FALSE(recursive.diagnostics.empty());
  EXPECT_NE(recursive.diagnostics.front().message.find("recursive"),
            std::string::npos);

  auto repeated = oq3::frontend::analyzeOpenQASM(
      "OPENQASM 3.1; include \"stdgates.inc\"; include "
      "\"stdgates.inc\";");
  ASSERT_FALSE(repeated);
  ASSERT_FALSE(repeated.diagnostics.empty());
  EXPECT_NE(repeated.diagnostics.front().message.find("more than once"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsIncludesInsideBlocks) {
  auto parsed = oq3::frontend::parseOpenQASM(
      "OPENQASM 3.1; if (true) { include \"nested.inc\"; }");
  ASSERT_FALSE(parsed);
  ASSERT_FALSE(parsed.diagnostics.empty());
  EXPECT_NE(parsed.diagnostics.front().message.find("only allowed globally"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, AcceptsBothIncludeStringQuoteStyles) {
  EXPECT_TRUE(
      oq3::frontend::parseOpenQASM("OPENQASM 3.1; include \"stdgates.inc\";"));
  EXPECT_TRUE(
      oq3::frontend::parseOpenQASM("OPENQASM 3.1; include 'stdgates.inc';"));
}

TEST(OpenQASMFrontendTest, RejectsInvalidIncludeStringsAtTheOffendingByte) {
  struct InvalidInclude {
    llvm::StringRef source;
    size_t line{};
    size_t column{};
  };
  constexpr auto includes = std::to_array<InvalidInclude>({
      {.source = "include \"\";", .line = 1, .column = 10},
      {.source = "include \"bad\tname.inc\";", .line = 1, .column = 13},
      {.source = "include \"bad\nname.inc\";", .line = 1, .column = 13},
      {.source = "include \"bad\rname.inc\";", .line = 1, .column = 13},
  });

  for (const auto& include : includes) {
    SCOPED_TRACE(include.source.str());
    llvm::SourceMgr sources;
    sources.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   include.source, "invalid-include.qasm"),
                               llvm::SMLoc());
    auto parsed = oq3::frontend::parseOpenQASM(sources);
    ASSERT_FALSE(parsed);
    ASSERT_FALSE(parsed.diagnostics.empty());
    EXPECT_EQ(parsed.diagnostics.front().location.filename,
              "invalid-include.qasm");
    EXPECT_EQ(parsed.diagnostics.front().location.line, include.line);
    EXPECT_EQ(parsed.diagnostics.front().location.column, include.column);
  }
}

TEST(OpenQASMFrontendTest, RejectsMisplacedVersionsAndRecursiveGates) {
  constexpr llvm::StringLiteral misplacedVersion = R"qasm(
qubit q;
OPENQASM 3.1;
)qasm";
  constexpr llvm::StringLiteral recursiveGates = R"qasm(
OPENQASM 3.1;
gate first q { first q; }
qubit q;
first q;
bit result = measure q;
)qasm";

  auto misplaced = oq3::frontend::analyzeOpenQASM(misplacedVersion);
  ASSERT_FALSE(misplaced);
  ASSERT_FALSE(misplaced.diagnostics.empty());
  EXPECT_NE(misplaced.diagnostics.front().message.find("must be the first"),
            std::string::npos);

  auto recursive = oq3::frontend::analyzeOpenQASM(recursiveGates);
  ASSERT_FALSE(recursive);
  ASSERT_FALSE(recursive.diagnostics.empty());
  EXPECT_NE(recursive.diagnostics.front().message.find("recursive"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, DiagnosesMalformedLexicalAndGrammarFamilies) {
  struct InvalidSource {
    llvm::StringRef name;
    llvm::StringRef source;
  };
  const auto fixtures = std::to_array<InvalidSource>({
      {.name = "unterminated-comment", .source = "OPENQASM 3.1; /*"},
      {.name = "unterminated-string",
       .source = "OPENQASM 3.1; include \"missing.inc;"},
      {.name = "missing-include",
       .source = "OPENQASM 3.1; include \"missing.inc\";"},
      {.name = "invalid-hardware-qubit",
       .source = "OPENQASM 3.1; qubit q; x $;"},
      {.name = "float-overflow",
       .source = "OPENQASM 3.1; float value = 1e99999;"},
      {.name = "unsupported-duration",
       .source = "OPENQASM 3.1; duration delay;"},
      {.name = "unsupported-opaque",
       .source = "OPENQASM 3.1; opaque custom q;"},
      {.name = "output-qubit", .source = "OPENQASM 3.1; output qubit q;"},
      {.name = "const-qubit", .source = "OPENQASM 3.1; const qubit q;"},
      {.name = "duplicate-version", .source = "OPENQASM 3.1; OPENQASM 3.1;"},
      {.name = "non-string-include",
       .source = "OPENQASM 3.1; include stdgates.inc;"},
      {.name = "gate-designator",
       .source = "OPENQASM 3.1; gate custom[2] q {}"},
      {.name = "missing-range-members",
       .source = "OPENQASM 3.1; for int i in [:] {}"},
      {.name = "missing-while-condition",
       .source = "OPENQASM 3.1; while () {}"},
      {.name = "switch-without-cases",
       .source = "OPENQASM 3.1; int value = 0; switch (value) {}"},
      {.name = "switch-case-after-default",
       .source = "OPENQASM 3.1; int value = 0; switch (value) { "
                 "default {} case 0 {} }"},
      {.name = "switch-with-repeated-default",
       .source = "OPENQASM 3.1; int value = 0; switch (value) { "
                 "case 0 {} default {} default {} }"},
      {.name = "const-without-initializer",
       .source = "OPENQASM 3.1; const int value;"},
  });

  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(fixture.name.str());
    auto parsed = oq3::frontend::parseOpenQASM(fixture.source);
    ASSERT_FALSE(parsed);
    ASSERT_FALSE(parsed.diagnostics.empty());
    EXPECT_FALSE(parsed.diagnostics.front().message.empty());
  }
}

TEST(OpenQASMFrontendTest, ParsesFixedAngleDeclarationsAndCasts) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
const uint WIDTH = 8;
const angle[WIDTH] fixed = angle[WIDTH](pi / 2);
angle machine = angle(tau / 4);
bit[2] value;
if (uint[2](value) == 3) {}
if (int[2](value) == -1) {}
)qasm";

  auto parsed = oq3::frontend::parseOpenQASM(source);
  ASSERT_TRUE(parsed) << parsed.diagnostics.front().message;
}

TEST(OpenQASMFrontendTest, RejectsUnsupportedReservedWordsAsIdentifiers) {
  constexpr auto reservedWords = std::to_array<llvm::StringLiteral>({
      "defcalgrammar", "def",      "cal",     "defcal", "extern",  "box",
      "let",           "continue", "end",     "return", "pragma",  "input",
      "readonly",      "mutable",  "complex", "array",  "void",    "stretch",
      "durationof",    "delay",    "im",      "#dim",   "#pragma",
  });
  for (const auto keyword : reservedWords) {
    SCOPED_TRACE(keyword.str());
    const std::string source = "OPENQASM 3.1; int " + keyword.str() + " = 0;";
    auto parsed = oq3::frontend::parseOpenQASM(source);
    ASSERT_FALSE(parsed);
    ASSERT_FALSE(parsed.diagnostics.empty());
    EXPECT_NE(parsed.diagnostics.front().message.find("reserved keyword"),
              std::string::npos);
  }
}

TEST(OpenQASMFrontendTest, DiagnosesUnsupportedReservedFeatureSyntax) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1; input int value;",
      "OPENQASM 3.1; const complex value = 0;",
      "OPENQASM 3.1; output array[int, 2] values;",
      "OPENQASM 3.1; for complex value in [0:1] {}",
      "OPENQASM 3.1; int value = durationof({});",
  });
  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    auto parsed = oq3::frontend::parseOpenQASM(source);
    ASSERT_FALSE(parsed);
    ASSERT_FALSE(parsed.diagnostics.empty());
    EXPECT_NE(parsed.diagnostics.front().message.find("reserved keyword"),
              std::string::npos);
  }
}

TEST(OpenQASMFrontendTest, EnforcesNumericSeparatorPlacement) {
  constexpr auto invalidLiterals = std::to_array<llvm::StringLiteral>({
      "1e+_2",
      "1e-_2",
      "1_e2",
      "1._2",
      "1e_2",
      "0xA__B",
      "0b_1",
      "0o7_",
  });
  for (const auto literal : invalidLiterals) {
    SCOPED_TRACE(literal.str());
    const std::string source =
        "OPENQASM 3.1; float value = " + literal.str() + ";";
    auto parsed = oq3::frontend::parseOpenQASM(source);
    ASSERT_FALSE(parsed);
    ASSERT_FALSE(parsed.diagnostics.empty());
  }

  auto valid = oq3::frontend::parseOpenQASM(
      "OPENQASM 3.1; int hex = 0xA_B; float value = 1_2.3_4e+5_6;");
  ASSERT_TRUE(valid) << valid.diagnostics.front().message;
}

TEST(OpenQASMFrontendTest, AcceptsWideIntegerLiteralsWithDigitSeparators) {
  // DecimalIntegerLiteral allows '_' separators; values beyond uint64_t are
  // still valid tokens (wide integers). Constant evaluation may reject them.
  auto parsed = oq3::frontend::parseOpenQASM(
      "OPENQASM 3.1; int value = 999_999_999_999_999_999_999;");
  ASSERT_TRUE(parsed) << parsed.diagnostics.front().message;
}

TEST(OpenQASMFrontendTest, SourceManagerOverloadsPreserveParseFailures) {
  llvm::SourceMgr sources;
  sources.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                 "OPENQASM 3.1; qubit ;", "broken.qasm"),
                             llvm::SMLoc());

  auto parsed = oq3::frontend::parseOpenQASM(sources);
  ASSERT_FALSE(parsed);
  ASSERT_FALSE(parsed.diagnostics.empty());
  EXPECT_EQ(parsed.diagnostics.front().location.filename, "broken.qasm");

  auto analyzed = oq3::frontend::analyzeOpenQASM(sources);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_EQ(analyzed.diagnostics.front().location.filename, "broken.qasm");
}

TEST(OpenQASMFrontendTest, SupportsRequiredLiteralFormsAndOperatorPrecedence) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
const int binary = 0b1010;
const int octal = 0o12;
const int hexadecimal = 0xA;
const int separated = 1_0;
const float fraction = .5;
const float trailing = 1.;
const float separated_float = 1_0.5_0;
const bool precedence = 1 < 2 == true;
int powered = 2;
powered **= 3;
qubit q;
if (precedence && powered == binary && binary == octal && octal == hexadecimal &&
    hexadecimal == separated && fraction + trailing + separated_float > 0.0) {
  x q;
}
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
}

} // namespace

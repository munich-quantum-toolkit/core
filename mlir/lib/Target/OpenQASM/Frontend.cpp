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

#include "mlir/Target/OpenQASM/Detail/OpenQASMLexer.h"
#include "mlir/Target/OpenQASM/Detail/OpenQASMParser.h"
#include "mlir/Target/OpenQASM/Detail/OpenQASMSemantics.h"
#include "mlir/Target/OpenQASM/Detail/OpenQASMSyntax.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlir::oq3::frontend {

struct ParsedProgram::Impl {
  std::unique_ptr<llvm::SourceMgr> sources;
  detail::SyntaxProgram syntax;
};

ParsedProgram::ParsedProgram(std::unique_ptr<Impl> implementation)
    : impl(std::move(implementation)) {}
ParsedProgram::ParsedProgram(ParsedProgram&&) noexcept = default;
ParsedProgram& ParsedProgram::operator=(ParsedProgram&&) noexcept = default;
ParsedProgram::~ParsedProgram() = default;

namespace {

struct ParseArtifacts {
  std::unique_ptr<llvm::SourceMgr> sources;
  detail::SyntaxProgram syntax;
  std::vector<Diagnostic> diagnostics;
};

constexpr size_t INCLUDE_NESTING_LIMIT = 64;
constexpr size_t EXPANDED_STATEMENT_LIMIT = 1'000'000;

} // namespace

[[nodiscard]] static std::optional<detail::StandardLibraryKind>
standardLibraryKind(const llvm::StringRef filename) {
  if (filename == "stdgates.inc") {
    return detail::StandardLibraryKind::StdGates;
  }
  if (filename == "qelib1.inc") {
    return detail::StandardLibraryKind::QELib1;
  }
  return std::nullopt;
}

static ParseArtifacts
parseBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer,
            const llvm::SourceMgr* providedSources = nullptr) {
  ParseArtifacts result;
  auto sources = std::make_unique<llvm::SourceMgr>();
  if (providedSources != nullptr) {
    sources->setVirtualFileSystem(providedSources->getVirtualFileSystem());
    sources->setIncludeDirs(
        std::vector<std::string>(providedSources->getIncludeDirs()));
  }
  const auto mainBufferId =
      sources->AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  llvm::StringMap<const llvm::MemoryBuffer*> providedIncludeBuffers;
  if (providedSources != nullptr) {
    for (unsigned id = 2; id <= providedSources->getNumBuffers(); ++id) {
      const auto* included = providedSources->getMemoryBuffer(id);
      providedIncludeBuffers.try_emplace(included->getBufferIdentifier(),
                                         included);
    }
  }

  llvm::BumpPtrAllocator allocator;
  detail::SyntaxBuilder builder;
  bool failedParsing = false;
  const auto reportIncludeNestingLimit = [&](const llvm::SMLoc location) {
    std::ignore = builder.error(
        location,
        llvm::Twine("include nesting exceeds the limit of ") +
            llvm::Twine(static_cast<unsigned>(INCLUDE_NESTING_LIMIT)));
    failedParsing = true;
  };
  const auto reportStatementLimit = [&](const llvm::SMLoc location) {
    std::ignore = builder.error(
        location,
        llvm::Twine(
            "expanded OpenQASM program exceeds the statement limit of ") +
            llvm::Twine(static_cast<unsigned>(EXPANDED_STATEMENT_LIMIT)));
    failedParsing = true;
  };
  struct ParsedSource {
    size_t bodyBegin = 0;
    size_t bodyEnd = 0;
    size_t includeBegin = 0;
    size_t includeEnd = 0;
  };
  llvm::DenseMap<unsigned, ParsedSource> parsedSources;
  llvm::StringMap<unsigned> includeBuffers;
  const auto parseSource = [&](const unsigned bufferId) {
    const auto bodyBegin = builder.getBody().size();
    const auto includeBegin = builder.getIncludes().size();
    detail::Lexer lexer(sources->getMemoryBuffer(bufferId)->getBuffer());
    detail::Parser parser(lexer, builder, allocator);
    if (failed(parser.parseProgram())) {
      failedParsing = true;
    }
    parsedSources.try_emplace(
        bufferId, ParsedSource{.bodyBegin = bodyBegin,
                               .bodyEnd = builder.getBody().size(),
                               .includeBegin = includeBegin,
                               .includeEnd = builder.getIncludes().size()});
  };
  parseSource(mainBufferId);

  llvm::SmallSet<unsigned, 8> parsedBuffers;
  llvm::SmallVector<unsigned> includeTargets;
  llvm::SmallVector<size_t> includeDepths(builder.getIncludes().size(), 1);
  parsedBuffers.insert(mainBufferId);
  for (size_t includeIndex = 0; includeIndex < builder.getIncludes().size();
       ++includeIndex) {
    includeTargets.resize(builder.getIncludes().size());
    const auto include = builder.getIncludes()[includeIndex];
    if (includeDepths[includeIndex] > INCLUDE_NESTING_LIMIT) {
      reportIncludeNestingLimit(include.location);
      continue;
    }
    if (standardLibraryKind(include.filename)) {
      continue;
    }
    auto bufferId = includeBuffers.lookup(include.filename);
    if (bufferId == 0) {
      if (const auto* provided =
              providedIncludeBuffers.lookup(include.filename)) {
        bufferId = sources->AddNewSourceBuffer(
            llvm::MemoryBuffer::getMemBufferCopy(
                provided->getBuffer(), provided->getBufferIdentifier()),
            include.location);
        includeBuffers[include.filename] = bufferId;
      }
    }
    if (bufferId == 0) {
      std::string includedPath;
      auto included =
          sources->OpenIncludeFile(include.filename.str(), includedPath);
      if (included) {
        bufferId = includeBuffers.lookup(includedPath);
        if (bufferId == 0) {
          bufferId = sources->AddNewSourceBuffer(std::move(*included),
                                                 include.location);
          includeBuffers[includedPath] = bufferId;
        }
      }
    }
    if (bufferId == 0) {
      std::ignore = builder.error(
          include.location, llvm::Twine("could not open included file '") +
                                include.filename + "'");
      failedParsing = true;
      continue;
    }
    if (parsedBuffers.insert(bufferId).second) {
      parseSource(bufferId);
      includeDepths.resize(builder.getIncludes().size(),
                           includeDepths[includeIndex] + 1);
    }
    includeTargets[includeIndex] = bufferId;
  }

  std::vector<std::optional<detail::SyntaxStatementId>> includeMarkers(
      builder.getIncludes().size());
  for (const auto [index, include] : llvm::enumerate(builder.getIncludes())) {
    if (const auto kind = standardLibraryKind(include.filename)) {
      includeMarkers[index] =
          builder.standardLibraryInclude(include.location, *kind);
    }
  }

  std::vector<detail::SyntaxStatementId> expandedBody;
  std::vector<std::optional<detail::SyntaxIncludeContextId>>
      expandedIncludeContexts;
  std::vector<detail::SyntaxIncludeContext> includeContexts;
  llvm::SmallSet<unsigned, 8> activeBuffers;
  const auto appendBodyRange =
      [&](const size_t begin, const size_t end, const llvm::SMLoc location,
          const std::optional<detail::SyntaxIncludeContextId> includeContext) {
        const auto count = end - begin;
        if (count > EXPANDED_STATEMENT_LIMIT - expandedBody.size()) {
          reportStatementLimit(location);
          return false;
        }
        const auto* const bodyBegin = std::next(
            builder.getBody().begin(), static_cast<std::ptrdiff_t>(begin));
        const auto* const bodyEnd = std::next(builder.getBody().begin(),
                                              static_cast<std::ptrdiff_t>(end));
        expandedBody.insert(expandedBody.end(), bodyBegin, bodyEnd);
        expandedIncludeContexts.insert(expandedIncludeContexts.end(), count,
                                       includeContext);
        return true;
      };
  const auto appendSource =
      [&](auto&& self, const unsigned bufferId, const size_t depth,
          const std::optional<detail::SyntaxIncludeContextId> includeContext)
      -> bool {
    activeBuffers.insert(bufferId);
    const auto parsed = parsedSources.lookup(bufferId);
    auto cursor = parsed.bodyBegin;
    for (auto includeIndex = parsed.includeBegin;
         includeIndex < parsed.includeEnd; ++includeIndex) {
      const auto offset = builder.getIncludes()[includeIndex].bodyOffset;
      const auto includeLocation = builder.getIncludes()[includeIndex].location;
      if (!appendBodyRange(cursor, offset, includeLocation, includeContext)) {
        activeBuffers.erase(bufferId);
        return false;
      }
      if (includeMarkers[includeIndex]) {
        if (expandedBody.size() >= EXPANDED_STATEMENT_LIMIT) {
          reportStatementLimit(includeLocation);
          activeBuffers.erase(bufferId);
          return false;
        }
        expandedBody.push_back(*includeMarkers[includeIndex]);
        expandedIncludeContexts.push_back(includeContext);
      } else if (includeTargets[includeIndex] != 0) {
        const auto target = includeTargets[includeIndex];
        if (activeBuffers.contains(target)) {
          std::ignore = builder.error(includeLocation,
                                      "recursive include is not allowed");
          failedParsing = true;
        } else if (depth >= INCLUDE_NESTING_LIMIT) {
          reportIncludeNestingLimit(includeLocation);
          activeBuffers.erase(bufferId);
          return false;
        } else {
          const auto childContext = static_cast<detail::SyntaxIncludeContextId>(
              includeContexts.size());
          includeContexts.push_back(
              {.location = includeLocation, .parent = includeContext});
          if (!self(self, target, depth + 1, childContext)) {
            activeBuffers.erase(bufferId);
            return false;
          }
        }
      }
      cursor = offset;
    }
    const auto* source = sources->getMemoryBuffer(bufferId);
    if (!appendBodyRange(cursor, parsed.bodyEnd,
                         llvm::SMLoc::getFromPointer(source->getBufferStart()),
                         includeContext)) {
      activeBuffers.erase(bufferId);
      return false;
    }
    activeBuffers.erase(bufferId);
    return true;
  };
  std::ignore = appendSource(appendSource, mainBufferId, 0, std::nullopt);
  builder.replaceBody(std::move(expandedBody),
                      std::move(expandedIncludeContexts),
                      std::move(includeContexts));

  if (failedParsing) {
    for (const auto& diagnostic : builder.getDiagnostics()) {
      result.diagnostics.push_back(
          {.location = detail::sourceLocation(*sources, diagnostic.location),
           .message = diagnostic.message});
    }
    if (result.diagnostics.empty()) {
      result.diagnostics.push_back({.message = "OpenQASM parsing failed"});
    }
    return result;
  }

  result.sources = std::move(sources);
  result.syntax = builder.takeProgram();
  return result;
}

ParseResult parseOpenQASM(llvm::SourceMgr& sourceMgr) {
  const auto* source = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  auto parsed =
      parseBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                      source->getBuffer(), source->getBufferIdentifier()),
                  &sourceMgr);
  if (!parsed.sources) {
    return {.diagnostics = std::move(parsed.diagnostics)};
  }
  auto implementation = std::make_unique<ParsedProgram::Impl>();
  implementation->sources = std::move(parsed.sources);
  implementation->syntax = std::move(parsed.syntax);
  return {.program = std::unique_ptr<ParsedProgram>(
              new ParsedProgram(std::move(implementation)))};
}

ParseResult parseOpenQASM(const llvm::StringRef source) {
  auto parsed =
      parseBuffer(llvm::MemoryBuffer::getMemBufferCopy(source, "<input>"));
  if (!parsed.sources) {
    return {.diagnostics = std::move(parsed.diagnostics)};
  }
  auto implementation = std::make_unique<ParsedProgram::Impl>();
  implementation->sources = std::move(parsed.sources);
  implementation->syntax = std::move(parsed.syntax);
  return {.program = std::unique_ptr<ParsedProgram>(
              new ParsedProgram(std::move(implementation)))};
}

AnalysisResult analyzeOpenQASM(const ParsedProgram& parsedProgram,
                               const FrontendOptions& options) {
  return detail::analyzeSyntaxProgram(parsedProgram.impl->syntax,
                                      *parsedProgram.impl->sources, options);
}

AnalysisResult analyzeOpenQASM(llvm::SourceMgr& sourceMgr,
                               const FrontendOptions& options) {
  auto parsed = parseOpenQASM(sourceMgr);
  if (!parsed) {
    return {.diagnostics = std::move(parsed.diagnostics)};
  }
  return analyzeOpenQASM(*parsed.program, options);
}

AnalysisResult analyzeOpenQASM(const llvm::StringRef source,
                               const FrontendOptions& options) {
  auto parsed = parseOpenQASM(source);
  if (!parsed) {
    return {.diagnostics = std::move(parsed.diagnostics)};
  }
  return analyzeOpenQASM(*parsed.program, options);
}

} // namespace mlir::oq3::frontend

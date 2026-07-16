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

#include "OpenQASMLexer.h"
#include "OpenQASMParser.h"
#include "OpenQASMSemantics.h"
#include "OpenQASMSyntax.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/VirtualFileSystem.h>

#include <memory>
#include <optional>
#include <string>
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

constexpr std::size_t includeNestingLimit = 64;
constexpr std::size_t expandedStatementLimit = 100'000;

[[nodiscard]] bool isStandardLibrary(const llvm::StringRef filename) {
  return filename == "stdgates.inc" || filename == "qelib1.inc";
}

ParseArtifacts parseBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer,
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
  if (providedSources != nullptr) {
    for (unsigned id = 2; id <= providedSources->getNumBuffers(); ++id) {
      const auto* included = providedSources->getMemoryBuffer(id);
      sources->AddNewSourceBuffer(
          llvm::MemoryBuffer::getMemBufferCopy(included->getBuffer(),
                                               included->getBufferIdentifier()),
          llvm::SMLoc());
    }
  }

  llvm::BumpPtrAllocator allocator;
  detail::SyntaxBuilder builder;
  bool failedParsing = false;
  const auto reportIncludeNestingLimit = [&](const llvm::SMLoc location) {
    (void)builder.error(
        location, llvm::Twine("include nesting exceeds the limit of ") +
                      llvm::Twine(static_cast<unsigned>(includeNestingLimit)));
    failedParsing = true;
  };
  const auto reportStatementLimit = [&](const llvm::SMLoc location) {
    (void)builder.error(
        location,
        llvm::Twine(
            "expanded OpenQASM program exceeds the statement limit of ") +
            llvm::Twine(static_cast<unsigned>(expandedStatementLimit)));
    failedParsing = true;
  };
  struct ParsedSource {
    std::size_t bodyBegin = 0;
    std::size_t bodyEnd = 0;
    std::size_t includeBegin = 0;
    std::size_t includeEnd = 0;
  };
  llvm::DenseMap<unsigned, ParsedSource> parsedSources;
  llvm::StringMap<unsigned> includeBuffers;
  for (unsigned id = 2; id <= sources->getNumBuffers(); ++id) {
    includeBuffers.try_emplace(
        sources->getMemoryBuffer(id)->getBufferIdentifier(), id);
  }
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
  llvm::SmallVector<std::size_t> includeDepths(builder.getIncludes().size(), 1);
  parsedBuffers.insert(mainBufferId);
  for (std::size_t includeIndex = 0;
       includeIndex < builder.getIncludes().size(); ++includeIndex) {
    includeTargets.resize(builder.getIncludes().size());
    const auto include = builder.getIncludes()[includeIndex];
    if (includeDepths[includeIndex] > includeNestingLimit) {
      reportIncludeNestingLimit(include.location);
      continue;
    }
    if (isStandardLibrary(include.filename)) {
      continue;
    }
    auto bufferId = includeBuffers.lookup(include.filename);
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
      (void)builder.error(include.location,
                          llvm::Twine("could not open included file '") +
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
    if (isStandardLibrary(include.filename)) {
      includeMarkers[index] = builder.standardLibraryInclude(include.location);
    }
  }

  std::vector<detail::SyntaxStatementId> expandedBody;
  llvm::SmallSet<unsigned, 8> activeBuffers;
  const auto appendBodyRange = [&](const std::size_t begin,
                                   const std::size_t end,
                                   const llvm::SMLoc location) {
    const auto count = end - begin;
    if (count > expandedStatementLimit - expandedBody.size()) {
      reportStatementLimit(location);
      return false;
    }
    expandedBody.insert(expandedBody.end(), builder.getBody().begin() + begin,
                        builder.getBody().begin() + end);
    return true;
  };
  const auto appendSource = [&](auto&& self, const unsigned bufferId,
                                const std::size_t depth) -> bool {
    activeBuffers.insert(bufferId);
    const auto parsed = parsedSources.lookup(bufferId);
    auto cursor = parsed.bodyBegin;
    for (auto includeIndex = parsed.includeBegin;
         includeIndex < parsed.includeEnd; ++includeIndex) {
      const auto offset = builder.getIncludes()[includeIndex].bodyOffset;
      const auto includeLocation = builder.getIncludes()[includeIndex].location;
      if (!appendBodyRange(cursor, offset, includeLocation)) {
        activeBuffers.erase(bufferId);
        return false;
      }
      if (includeMarkers[includeIndex]) {
        if (expandedBody.size() >= expandedStatementLimit) {
          reportStatementLimit(includeLocation);
          activeBuffers.erase(bufferId);
          return false;
        }
        expandedBody.push_back(*includeMarkers[includeIndex]);
      } else if (includeTargets[includeIndex] != 0) {
        const auto target = includeTargets[includeIndex];
        if (activeBuffers.contains(target)) {
          (void)builder.error(includeLocation,
                              "recursive include is not allowed");
          failedParsing = true;
        } else if (depth >= includeNestingLimit) {
          reportIncludeNestingLimit(includeLocation);
          activeBuffers.erase(bufferId);
          return false;
        } else {
          if (!self(self, target, depth + 1)) {
            activeBuffers.erase(bufferId);
            return false;
          }
        }
      }
      cursor = offset;
    }
    const auto* source = sources->getMemoryBuffer(bufferId);
    if (!appendBodyRange(
            cursor, parsed.bodyEnd,
            llvm::SMLoc::getFromPointer(source->getBufferStart()))) {
      activeBuffers.erase(bufferId);
      return false;
    }
    activeBuffers.erase(bufferId);
    return true;
  };
  (void)appendSource(appendSource, mainBufferId, 0);
  builder.replaceBody(std::move(expandedBody));

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

} // namespace

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

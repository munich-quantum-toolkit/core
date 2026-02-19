/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Support/PrettyPrinting.h"

#include <cstddef>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <string>
#include <utility>

namespace mlir {

constexpr auto TOTAL_WIDTH = 120;
constexpr auto BORDER_WIDTH = 2; // "║ " on each side
constexpr int CONTENT_WIDTH = TOTAL_WIDTH - (2 * BORDER_WIDTH);

// Pre-built strings, initialised once on first call. Each UTF-8 "═" is 3
// bytes. BORDER_SEP is the "═" run between box corners; SPACES is used for
// padding.
static llvm::StringRef getBorderSep() {
  static const std::string BORDER_SEP = [] {
    std::string s;
    s.reserve(static_cast<size_t>(TOTAL_WIDTH - 2) * 3U);
    for (auto i = 0; i < TOTAL_WIDTH - 2; ++i) {
      s += "═";
    }
    return s;
  }();
  return BORDER_SEP;
}

static llvm::StringRef getSpaces() {
  static const std::string SPACES(static_cast<size_t>(CONTENT_WIDTH), ' ');
  return SPACES;
}

int calculateDisplayWidth(llvm::StringRef str) {
  auto displayWidth = 0;
  for (size_t i = 0; i < str.size();) {
    if (const unsigned char c = str[i]; (c & 0x80) == 0) {
      // ASCII character (1 byte)
      ++displayWidth;
      ++i;
    } else if ((c & 0xE0) == 0xC0) {
      // 2-byte UTF-8 character
      ++displayWidth;
      i += 2;
    } else if ((c & 0xF0) == 0xE0) {
      // 3-byte UTF-8 character (like → and ✓)
      ++displayWidth;
      i += 3;
    } else if ((c & 0xF8) == 0xF0) {
      // 4-byte UTF-8 character (most emojis take 2 display columns)
      displayWidth += 2;
      i += 4;
    } else {
      // Invalid UTF-8, skip
      ++i;
    }
  }
  return displayWidth;
}

void wrapLine(llvm::StringRef line, const int maxWidth,
              llvm::SmallVectorImpl<llvm::SmallString<128>>& result,
              const int indent) {
  if (line.empty()) {
    result.emplace_back("");
    return;
  }

  // Detect leading whitespace (indentation) in the original line
  size_t leadingSpaces = 0;
  for (const char c : line) {
    if (c == ' ') {
      ++leadingSpaces;
    } else if (c == '\t') {
      leadingSpaces += 4; // Count tabs as 4 spaces
    } else {
      break;
    }
  }

  // Extract the content without leading whitespace
  const llvm::StringRef content = line.substr(line.find_first_not_of(" \t"));
  if (content.empty()) {
    result.emplace_back(line);
    return;
  }

  // Calculate available width accounting for indentation and wrap indicators
  // First line: original indent + content
  // Continuation lines: "↳ " (2 chars) + same indent + content
  const int firstLineWidth =
      maxWidth - indent - static_cast<int>(leadingSpaces);
  const int contLineWidth =
      maxWidth - indent - static_cast<int>(leadingSpaces) - 2; // "↳ "

  if (firstLineWidth <= 10 || contLineWidth <= 10) {
    // Not enough space to wrap intelligently, just return original
    result.emplace_back(line);
    return;
  }

  llvm::SmallString<128> currentLine;
  llvm::SmallString<64> currentWord;
  auto currentWidth = 0;
  auto isFirstLine = true;

  // Helper: build and emit a completed line with proper indent prefix.
  // `addArrow` appends " →" to signal the line continues.
  auto flushLine = [&](const bool addArrow, const bool lastLine) {
    llvm::SmallString<128> lineWithIndent;
    lineWithIndent.append(leadingSpaces, ' ');
    lineWithIndent += currentLine;
    if (addArrow && (!isFirstLine || !lastLine)) {
      lineWithIndent += " →";
    }
    result.emplace_back(std::move(lineWithIndent));
  };

  auto addWord = [&](llvm::StringRef word) -> bool {
    const int wordWidth = calculateDisplayWidth(word);
    const int spaceWidth = currentLine.empty() ? 0 : 1;
    const int effectiveWidth = isFirstLine ? firstLineWidth : contLineWidth;

    if (currentWidth + spaceWidth + wordWidth <= effectiveWidth) {
      // Word fits on current line
      if (!currentLine.empty()) {
        currentLine += ' ';
        ++currentWidth;
      }
      currentLine += word;
      currentWidth += wordWidth;
      return true;
    }
    return false;
  };

  // Process the content word by word
  for (const auto& c : content) {
    if (c == ' ' || c == '\t') {
      // End of word - try to add it to current line
      if (!currentWord.empty()) {
        if (!addWord(currentWord)) {
          // Word doesn't fit - finalize current line and start new one
          if (!currentLine.empty()) {
            flushLine(/*addArrow=*/true, /*lastLine=*/false);
          }
          // Start new continuation line
          currentLine = currentWord;
          currentWidth = calculateDisplayWidth(llvm::StringRef(currentWord));
          isFirstLine = false;
        }
        currentWord.clear();
      }
    } else {
      currentWord += c;
    }
  }

  // Add remaining word
  if (!currentWord.empty()) {
    if (!addWord(currentWord)) {
      // Finalize current line
      if (!currentLine.empty()) {
        flushLine(/*addArrow=*/true, /*lastLine=*/false);
      }
      // Add word on new continuation line (no arrow — this is the last line)
      llvm::SmallString<128> contLine("↳ ");
      contLine.append(leadingSpaces, ' ');
      contLine += currentWord;
      result.emplace_back(std::move(contLine));
      isFirstLine = false;
    } else {
      // Word fit: emit final line (no arrow)
      if (!currentLine.empty()) {
        flushLine(/*addArrow=*/false, /*lastLine=*/true);
      }
    }
  } else if (!currentLine.empty()) {
    // No remaining word: emit the last line (no arrow)
    flushLine(/*addArrow=*/false, /*lastLine=*/true);
  }

  // Safety net: if we somehow produced nothing, return the original line
  if (result.empty()) {
    result.emplace_back(line);
    return;
  }

  // Prepend "↳ " to all continuation lines (index >= 1) that don't have it yet
  for (size_t i = 1; i < result.size(); ++i) {
    if (!llvm::StringRef(result[i]).contains("↳")) {
      llvm::SmallString<128> newLine("↳ ");
      const llvm::StringRef lineRef = result[i];
      newLine += lineRef.substr(leadingSpaces);
      result[i] = std::move(newLine);
    }
  }
}

void printBoxTop(llvm::raw_ostream& os) {
  os << "╔" << getBorderSep() << "╗\n";
}

void printBoxMiddle(llvm::raw_ostream& os) {
  os << "╠" << getBorderSep() << "╣\n";
}

void printBoxBottom(llvm::raw_ostream& os) {
  os << "╚" << getBorderSep() << "╝\n";
}

// Internal helper: emit one already-wrapped line inside the box with padding.
static void emitBoxedLine(llvm::StringRef line, const int indent,
                          llvm::raw_ostream& os) {
  const int displayWidth = calculateDisplayWidth(line);
  const int padding = CONTENT_WIDTH - indent - displayWidth;

  os << "║ ";
  os.indent(static_cast<unsigned>(indent));
  os << line;
  // Write padding as a single slice of the pre-built spaces string
  if (padding > 0) {
    os << getSpaces().substr(0, static_cast<size_t>(padding));
  }
  os << " ║\n";
}

void printBoxLine(llvm::StringRef text, const int indent,
                  llvm::raw_ostream& os) {
  // Trim trailing whitespace before processing
  const auto trimmedText = text.rtrim();

  // Fast path: if the line fits without wrapping, skip wrapLine entirely
  const int displayWidth = calculateDisplayWidth(trimmedText);
  if (displayWidth <= CONTENT_WIDTH - indent) {
    emitBoxedLine(trimmedText, indent, os);
    return;
  }

  // Wrap the line
  llvm::SmallVector<llvm::SmallString<128>, 4> wrappedLines;
  wrapLine(trimmedText, CONTENT_WIDTH, wrappedLines, indent);

  for (const auto& line : wrappedLines) {
    emitBoxedLine(line, indent, os);
  }
}

void printBoxText(llvm::StringRef text, const int indent,
                  llvm::raw_ostream& os) {
  // Trim trailing newlines from the entire text, then iterate line-by-line
  llvm::StringRef remaining = text.rtrim();

  while (!remaining.empty()) {
    auto [lineStr, rest] = remaining.split('\n');
    remaining = rest;
    printBoxLine(lineStr, indent, os);
  }
}

void printProgram(ModuleOp module, const llvm::StringRef header,
                  llvm::raw_ostream& os) {
  printBoxTop(os);
  printBoxLine(header, 0, os);
  printBoxMiddle(os);

  // Capture the IR to a string so we can wrap it in box lines.
  llvm::SmallString<4096> irString;
  llvm::raw_svector_ostream irStream(irString);
  module.print(irStream);

  // Print the IR with box lines and wrapping
  printBoxText(irString, 0, os);

  printBoxBottom(os);
  os.flush();
}

} // namespace mlir

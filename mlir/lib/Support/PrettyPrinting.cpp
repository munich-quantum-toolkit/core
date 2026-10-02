/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Support/PrettyPrinting.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <string>
#include <utility>

namespace mlir {

constexpr auto TOTAL_WIDTH = 120;
constexpr auto BORDER_WIDTH = 2; // "║ " on each side
constexpr int CONTENT_WIDTH = TOTAL_WIDTH - (2 * BORDER_WIDTH);

/// Cache the border between box corners; each UTF-8 "═" occupies three bytes.
static StringRef getBorderSep() {
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

static StringRef getSpaces() {
  static const std::string SPACES(static_cast<size_t>(CONTENT_WIDTH), ' ');
  return SPACES;
}

int calculateDisplayWidth(StringRef str) {
  auto displayWidth = 0;
  for (size_t i = 0; i < str.size();) {
    if (const unsigned char c = str[i]; (c & 0x80U) == 0U) {
      // ASCII character (1 byte)
      ++displayWidth;
      ++i;
    } else if ((c & 0xE0U) == 0xC0U) {
      // 2-byte UTF-8 character
      ++displayWidth;
      i += 2;
    } else if ((c & 0xF0U) == 0xE0U) {
      // 3-byte UTF-8 character (like → and ✓)
      ++displayWidth;
      i += 3;
    } else if ((c & 0xF8U) == 0xF0U) {
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

void wrapLine(StringRef line, const int maxWidth,
              SmallVectorImpl<SmallString<128>>& result, const int indent) {
  if (line.empty()) {
    result.emplace_back("");
    return;
  }

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

  const StringRef content = line.substr(line.find_first_not_of(" \t"));
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
    result.emplace_back(line);
    return;
  }

  SmallString<128> currentLine;
  SmallString<64> currentWord;
  auto currentWidth = 0;
  auto isFirstLine = true;

  // Helper: build and emit a completed line with proper indent prefix.
  // `addArrow` appends " →" to signal the line continues.
  auto flushLine = [&](const bool addArrow, const bool lastLine) {
    SmallString<128> lineWithIndent;
    lineWithIndent.append(leadingSpaces, ' ');
    lineWithIndent += currentLine;
    if (addArrow && (!isFirstLine || !lastLine)) {
      lineWithIndent += " →";
    }
    result.emplace_back(std::move(lineWithIndent));
  };

  auto addWord = [&](StringRef word) -> bool {
    const int wordWidth = calculateDisplayWidth(word);
    const int spaceWidth = currentLine.empty() ? 0 : 1;
    const int effectiveWidth = isFirstLine ? firstLineWidth : contLineWidth;

    if (currentWidth + spaceWidth + wordWidth <= effectiveWidth) {
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

  for (const auto& c : content) {
    if (c == ' ' || c == '\t') {
      if (!currentWord.empty()) {
        if (!addWord(currentWord)) {
          if (!currentLine.empty()) {
            flushLine(/*addArrow=*/true, /*lastLine=*/false);
          }
          currentLine = currentWord;
          currentWidth = calculateDisplayWidth(StringRef(currentWord));
          isFirstLine = false;
        }
        currentWord.clear();
      }
    } else {
      currentWord += c;
    }
  }

  if (!currentWord.empty()) {
    if (!addWord(currentWord)) {
      if (!currentLine.empty()) {
        flushLine(/*addArrow=*/true, /*lastLine=*/false);
      }
      SmallString<128> contLine("↳ ");
      contLine.append(leadingSpaces, ' ');
      contLine += currentWord;
      result.emplace_back(std::move(contLine));
      isFirstLine = false;
    } else {
      if (!currentLine.empty()) {
        flushLine(/*addArrow=*/false, /*lastLine=*/true);
      }
    }
  } else if (!currentLine.empty()) {
    flushLine(/*addArrow=*/false, /*lastLine=*/true);
  }

  if (result.empty()) {
    result.emplace_back(line);
    return;
  }

  // Prepend "↳ " to all continuation lines (index >= 1) that don't have it yet
  for (size_t i = 1; i < result.size(); ++i) {
    if (!StringRef(result[i]).contains("↳")) {
      SmallString<128> newLine("↳ ");
      const StringRef lineRef = result[i];
      newLine += lineRef.substr(leadingSpaces);
      result[i] = std::move(newLine);
    }
  }
}

void printBoxTop(raw_ostream& os) { os << "╔" << getBorderSep() << "╗\n"; }

void printBoxMiddle(raw_ostream& os) { os << "╠" << getBorderSep() << "╣\n"; }

void printBoxBottom(raw_ostream& os) { os << "╚" << getBorderSep() << "╝\n"; }

// Internal helper: emit one already-wrapped line inside the box with padding.
static void emitBoxedLine(StringRef line, const int indent, raw_ostream& os) {
  const int displayWidth = calculateDisplayWidth(line);
  const int padding = CONTENT_WIDTH - indent - displayWidth;

  os << "║ ";
  os.indent(static_cast<unsigned>(indent));
  os << line;
  if (padding > 0) {
    os << getSpaces().substr(0, static_cast<size_t>(padding));
  }
  os << " ║\n";
}

void printBoxLine(StringRef text, const int indent, raw_ostream& os) {
  const auto trimmedText = text.rtrim();

  /// Avoid allocating wrapped lines when the text already fits.
  const int displayWidth = calculateDisplayWidth(trimmedText);
  if (displayWidth <= CONTENT_WIDTH - indent) {
    emitBoxedLine(trimmedText, indent, os);
    return;
  }

  SmallVector<SmallString<128>, 4> wrappedLines;
  wrapLine(trimmedText, CONTENT_WIDTH, wrappedLines, indent);

  for (const auto& line : wrappedLines) {
    emitBoxedLine(line, indent, os);
  }
}

void printBoxText(StringRef text, const int indent, raw_ostream& os) {
  // Trim trailing newlines from the entire text, then iterate line-by-line
  StringRef remaining = text.rtrim();

  while (!remaining.empty()) {
    auto [lineStr, rest] = remaining.split('\n');
    remaining = rest;
    printBoxLine(lineStr, indent, os);
  }
}

void printProgram(ModuleOp module, const StringRef header, raw_ostream& os) {
  printBoxTop(os);
  printBoxLine(header, 0, os);
  printBoxMiddle(os);

  // Capture the IR to a string so we can wrap it in box lines.
  SmallString<4096> irString;
  llvm::raw_svector_ostream irStream(irString);
  module.print(irStream);

  printBoxText(irString, 0, os);

  printBoxBottom(os);
  os.flush();
}

} // namespace mlir

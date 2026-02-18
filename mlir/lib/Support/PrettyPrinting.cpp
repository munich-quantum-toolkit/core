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

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>

namespace mlir {

constexpr auto TOTAL_WIDTH = 120;
constexpr auto BORDER_WIDTH = 2; // "║ " on each side

int calculateDisplayWidth(llvm::StringRef str) {
  auto displayWidth = 0;
  for (size_t i = 0; i < str.size();) {
    if (const unsigned char c = str[i]; (c & 0x80) == 0) {
      // ASCII character (1 byte)
      displayWidth++;
      i++;
    } else if ((c & 0xE0) == 0xC0) {
      // 2-byte UTF-8 character
      displayWidth++;
      i += 2;
    } else if ((c & 0xF0) == 0xE0) {
      // 3-byte UTF-8 character (like → and ✓)
      displayWidth++;
      i += 3;
    } else if ((c & 0xF8) == 0xF0) {
      // 4-byte UTF-8 character
      displayWidth += 2; // Most emojis take 2 display columns
      i += 4;
    } else {
      // Invalid UTF-8, skip
      i++;
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
      leadingSpaces++;
    } else if (c == '\t') {
      leadingSpaces += 4; // Count tabs as 4 spaces
    } else {
      break;
    }
  }

  // Extract the content without leading whitespace
  llvm::StringRef content = line.substr(line.find_first_not_of(" \t"));
  if (content.empty()) {
    result.emplace_back(line.str());
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
    result.emplace_back(line.str());
    return;
  }

  llvm::SmallString<128> currentLine;
  llvm::SmallString<64> currentWord;
  auto currentWidth = 0;
  auto isFirstLine = true;

  auto addWord = [&](llvm::StringRef word) {
    const int wordWidth = calculateDisplayWidth(word);
    const int spaceWidth = currentLine.empty() ? 0 : 1;
    const int effectiveWidth = isFirstLine ? firstLineWidth : contLineWidth;

    if (currentWidth + spaceWidth + wordWidth <= effectiveWidth) {
      // Word fits on current line
      if (!currentLine.empty()) {
        currentLine += ' ';
        currentWidth++;
      }
      currentLine += word;
      currentWidth += wordWidth;
      return true;
    }
    return false;
  };

  // Process the content word by word
  for (size_t i = 0; i < content.size(); ++i) {
    const char c = content[i];

    if (c == ' ' || c == '\t') {
      // End of word - try to add it to current line
      if (!currentWord.empty()) {
        if (!addWord(currentWord)) {
          // Word doesn't fit - finalize current line and start new one
          if (!currentLine.empty()) {
            // Add wrap indicator to the end
            llvm::SmallString<128> lineWithIndent;
            lineWithIndent.append(leadingSpaces, ' ');
            lineWithIndent += currentLine;
            if (!isFirstLine || (i < content.size() - 1)) {
              lineWithIndent += " →";
            }
            result.emplace_back(lineWithIndent.str().str());
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
        llvm::SmallString<128> lineWithIndent;
        lineWithIndent.append(leadingSpaces, ' ');
        lineWithIndent += currentLine;
        lineWithIndent += " →";
        result.emplace_back(lineWithIndent.str().str());
      }

      // Add word on new line
      llvm::SmallString<128> lineWithIndent;
      lineWithIndent.append(leadingSpaces, ' ');
      llvm::SmallString<128> contLine("↳ ");
      contLine.append(leadingSpaces, ' ');
      contLine += currentWord;
      result.emplace_back(contLine.str().str());
      isFirstLine = false;
    } else {
      // Word fit, add the final line
      if (!currentLine.empty()) {
        llvm::SmallString<128> lineWithIndent;
        lineWithIndent.append(leadingSpaces, ' ');
        lineWithIndent += currentLine;
        result.emplace_back(lineWithIndent.str().str());
      }
    }
  } else if (!currentLine.empty()) {
    // Add the final line
    llvm::SmallString<128> lineWithIndent;
    lineWithIndent.append(leadingSpaces, ' ');
    lineWithIndent += currentLine;
    result.emplace_back(lineWithIndent.str().str());
  }

  // If we didn't wrap anything, return the original line
  if (result.empty()) {
    result.emplace_back(line.str());
  } else if (result.size() > 1) {
    // Add continuation indicator to all but the first and last lines
    for (size_t i = 1; i < result.size(); ++i) {
      llvm::StringRef lineRef = result[i];
      if (lineRef.find("↳") == llvm::StringRef::npos) {
        llvm::SmallString<128> newLine("↳ ");
        newLine.append(leadingSpaces, ' ');
        newLine += lineRef.substr(leadingSpaces);
        result[i] = newLine.str().str();
      }
    }
  }
}

void printBoxTop(llvm::raw_ostream& os) {
  os << "╔";
  for (auto i = 0; i < TOTAL_WIDTH - 2; ++i) {
    os << "═";
  }
  os << "╗\n";
}

void printBoxMiddle(llvm::raw_ostream& os) {
  os << "╠";
  for (auto i = 0; i < TOTAL_WIDTH - 2; ++i) {
    os << "═";
  }
  os << "╣\n";
}

void printBoxBottom(llvm::raw_ostream& os) {
  os << "╚";
  for (auto i = 0; i < TOTAL_WIDTH - 2; ++i) {
    os << "═";
  }
  os << "╝\n";
}

void printBoxLine(llvm::StringRef text, const int indent,
                  llvm::raw_ostream& os) {
  // Content width = Total width - left border (2 chars) - right border (2
  // chars)
  constexpr int contentWidth =
      TOTAL_WIDTH - (2 * BORDER_WIDTH); // "║ " and " ║"

  // Trim trailing whitespace before processing
  auto trimmedText = text.rtrim();

  // Wrap the line if needed
  llvm::SmallVector<llvm::SmallString<128>, 4> wrappedLines;
  wrapLine(trimmedText, contentWidth, wrappedLines, indent);

  for (const auto& line : wrappedLines) {
    const int displayWidth = calculateDisplayWidth(line);
    const int padding = contentWidth - indent - displayWidth;

    os << "║ ";
    for (auto i = 0; i < indent; ++i) {
      os << " ";
    }
    os << line;
    for (auto i = 0; i < padding; ++i) {
      os << " ";
    }
    os << " ║\n";
  }
}

void printBoxText(llvm::StringRef text, const int indent,
                  llvm::raw_ostream& os) {
  // Trim trailing newlines from the entire text
  llvm::StringRef trimmedText = text.rtrim();

  // Split the text by newlines and process each line
  llvm::SmallVector<llvm::StringRef, 16> lines;
  trimmedText.split(lines, '\n', -1, false);

  for (const auto& line : lines) {
    printBoxLine(line, indent, os);
  }
}

void printProgram(ModuleOp module, const llvm::StringRef header,
                  llvm::raw_ostream& os) {
  printBoxTop(os);
  printBoxLine(header, 0, os);
  printBoxMiddle(os);
  // Capture the IR to a string so we can wrap it in box lines

  llvm::SmallString<1024> irString;
  llvm::raw_svector_ostream irStream(irString);
  module.print(irStream);

  // Print the IR with box lines and wrapping
  printBoxText(irString, 0, os);

  printBoxBottom(os);
  os.flush();
}

} // namespace mlir

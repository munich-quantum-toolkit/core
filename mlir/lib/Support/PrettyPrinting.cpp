/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Support/PrettyPrinting.h"

#include <llvm/Support/raw_ostream.h>
#include <sstream>
#include <string>
#include <vector>

namespace mlir {

constexpr auto TOTAL_WIDTH = 120;
constexpr auto BORDER_WIDTH = 2; // "║ " on each side

/**
 * @brief Trim trailing whitespace from a string
 */
static std::string trimTrailingWhitespace(const std::string& str) {
  const size_t end = str.find_last_not_of(" \t\r\n");
  return (end == std::string::npos) ? "" : str.substr(0, end + 1);
}

int calculateDisplayWidth(const std::string& str) {
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

std::vector<std::string> wrapLine(const std::string& line, const int maxWidth,
                                  const int indent) {
  std::vector<std::string> wrapped;

  if (line.empty()) {
    wrapped.emplace_back("");
    return wrapped;
  }

  // Detect leading whitespace (indentation) in the original line
  size_t leadingSpaces = 0;
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == ' ') {
      leadingSpaces++;
    } else if (line[i] == '\t') {
      leadingSpaces += 4; // Count tabs as 4 spaces
    } else {
      break;
    }
  }

  // Extract the content without leading whitespace
  std::string content = line.substr(line.find_first_not_of(" \t"));
  if (content.empty()) {
    wrapped.emplace_back(line);
    return wrapped;
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
    wrapped.emplace_back(line);
    return wrapped;
  }

  std::string currentLine;
  std::string currentWord;
  auto currentWidth = 0;
  auto isFirstLine = true;
  auto addWord = [&](const std::string& word) {
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
            std::string lineWithIndent(leadingSpaces, ' ');
            lineWithIndent += currentLine;
            if (!isFirstLine || (i < content.size() - 1)) {
              lineWithIndent += " →";
            }
            wrapped.push_back(lineWithIndent);
          }

          // Start new continuation line
          currentLine = currentWord;
          currentWidth = calculateDisplayWidth(currentWord);
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
        std::string lineWithIndent(leadingSpaces, ' ');
        lineWithIndent += currentLine + " →";
        wrapped.push_back(lineWithIndent);
      }

      // Add word on new line
      std::string lineWithIndent(leadingSpaces, ' ');
      lineWithIndent = "↳ " + lineWithIndent + currentWord;
      wrapped.push_back(lineWithIndent);
      isFirstLine = false;
    } else {
      // Word fit, add the final line
      if (!currentLine.empty()) {
        std::string lineWithIndent(leadingSpaces, ' ');
        lineWithIndent += currentLine;
        wrapped.push_back(lineWithIndent);
      }
    }
  } else if (!currentLine.empty()) {
    // Add the final line
    std::string lineWithIndent(leadingSpaces, ' ');
    lineWithIndent += currentLine;
    wrapped.push_back(lineWithIndent);
  }

  // If we didn't wrap anything, return the original line
  if (wrapped.empty()) {
    wrapped.push_back(line);
  } else if (wrapped.size() > 1) {
    // Add continuation indicator to all but the first and last lines
    for (size_t i = 1; i < wrapped.size(); ++i) {
      if (wrapped[i].find("↳") == std::string::npos) {
        std::string indentStr(leadingSpaces, ' ');
        wrapped[i] = "↳ " + indentStr + wrapped[i].substr(leadingSpaces);
      }
    }
  }

  return wrapped;
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

void printBoxLine(const std::string& text, const int indent,
                  llvm::raw_ostream& os) {
  // Content width = Total width - left border (2 chars) - right border (2
  // chars)
  constexpr int contentWidth = TOTAL_WIDTH - 2 * BORDER_WIDTH; // "║ " and " ║"

  // Trim trailing whitespace before processing
  const std::string trimmedText = trimTrailingWhitespace(text);

  // Wrap the line if needed
  for (const auto wrappedLines = wrapLine(trimmedText, contentWidth, indent);
       const auto& line : wrappedLines) {
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

void printBoxText(const std::string& text, const int indent,
                  llvm::raw_ostream& os) {
  // Trim trailing newlines from the entire text
  std::string trimmedText = text;
  while (!trimmedText.empty() &&
         (trimmedText.back() == '\n' || trimmedText.back() == '\r')) {
    trimmedText.pop_back();
  }

  std::istringstream stream(trimmedText);
  std::string line;

  while (std::getline(stream, line)) {
    printBoxLine(line, indent, os);
  }
}

} // namespace mlir

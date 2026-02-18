/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

namespace mlir {

class ModuleOp;

/**
 * @brief Calculate UTF-8 display width of a string
 *
 * @details
 * Counts the visual display width, not byte count. UTF-8 multi-byte
 * characters like → and ✓ are counted as 1 display column.
 *
 * @param str The string to measure
 * @return The display width in columns
 */
int calculateDisplayWidth(llvm::StringRef str);

/**
 * @brief Wrap a long line into multiple lines that fit within the box
 *
 * @details
 * Splits a line that's too long into multiple lines, preferring to break
 * at whitespace when possible. Each wrapped line will fit within the
 * available width inside the box.
 *
 * @param line The line to wrap
 * @param maxWidth Maximum width for each line (excluding box borders and
 * indent)
 * @param indent Number of spaces to indent wrapped lines
 * @param result Output vector to store wrapped lines
 */
void wrapLine(llvm::StringRef line, int maxWidth,
              llvm::SmallVectorImpl<llvm::SmallString<128>>& result,
              int indent = 0);

/**
 * @brief Print top border of a box
 *
 * @param os Output stream to write to
 */
void printBoxTop(llvm::raw_ostream& os = llvm::errs());

/**
 * @brief Print middle separator of a box
 *
 * @param os Output stream to write to
 */
void printBoxMiddle(llvm::raw_ostream& os = llvm::errs());

/**
 * @brief Print bottom border of a box
 *
 * @param os Output stream to write to
 */
void printBoxBottom(llvm::raw_ostream& os = llvm::errs());

/**
 * @brief Print a box line with text and proper padding
 *
 * @details
 * If the text is too long, it will be wrapped across multiple lines.
 *
 * @param text The text to display in the box
 * @param indent Number of spaces to indent the text (0 for left-aligned)
 * @param os Output stream to write to
 */
void printBoxLine(llvm::StringRef text, int indent = 0,
                  llvm::raw_ostream& os = llvm::errs());

/**
 * @brief Print multiple lines of text within the box, with line wrapping
 *
 * @details
 * Takes a multi-line string and prints each line within the box borders,
 * wrapping long lines as needed.
 *
 * @param text The text to display (may contain newlines)
 * @param indent Number of spaces to indent the text
 * @param os Output stream to write to
 */
void printBoxText(llvm::StringRef text, int indent = 0,
                  llvm::raw_ostream& os = llvm::errs());

/**
 * @brief Pretty print an MLIR module with a header and box formatting
 *
 * @param module The MLIR module to print
 * @param header Optional header text to display above the module
 * @param os Output stream to write to
 */
void printProgram(ModuleOp module, llvm::StringRef header = "",
                  llvm::raw_ostream& os = llvm::errs());

} // namespace mlir

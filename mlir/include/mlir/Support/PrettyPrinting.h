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

#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LLVM.h>

namespace mlir {

class ModuleOp;

/// Calculate UTF-8 display width of a string
///
/// Counts the visual display width, not byte count. UTF-8 multi-byte
/// characters like → and ✓ are counted as 1 display column.
///
/// @param str The string to measure
/// @return The display width in columns
int calculateDisplayWidth(StringRef str);

/// Wrap a long line into multiple lines that fit within the box
///
/// Splits a line that's too long into multiple lines, preferring to break
/// at whitespace when possible. Each wrapped line will fit within the
/// available width inside the box.
///
/// @param line The line to wrap
/// @param maxWidth Maximum width for each line (excluding box borders and
/// indent)
/// @param indent Number of spaces to indent wrapped lines
/// @param result Output vector to store wrapped lines
void wrapLine(StringRef line, int maxWidth,
              SmallVectorImpl<SmallString<128>>& result, int indent = 0);

/// Print top border of a box
///
/// @param os Output stream to write to
void printBoxTop(raw_ostream& os = llvm::errs());

/// Print middle separator of a box
///
/// @param os Output stream to write to
void printBoxMiddle(raw_ostream& os = llvm::errs());

/// Print bottom border of a box
///
/// @param os Output stream to write to
void printBoxBottom(raw_ostream& os = llvm::errs());

/// Print a box line with text and proper padding
///
/// If the text is too long, it will be wrapped across multiple lines.
///
/// @param text The text to display in the box
/// @param indent Number of spaces to indent the text (0 for left-aligned)
/// @param os Output stream to write to
void printBoxLine(StringRef text, int indent = 0,
                  raw_ostream& os = llvm::errs());

/// Print multiple lines of text within the box, with line wrapping
///
/// Takes a multi-line string and prints each line within the box borders,
/// wrapping long lines as needed.
///
/// @param text The text to display (may contain newlines)
/// @param indent Number of spaces to indent the text
/// @param os Output stream to write to
void printBoxText(StringRef text, int indent = 0,
                  raw_ostream& os = llvm::errs());

/// Pretty print an MLIR module with a header and box formatting
///
/// @param module The MLIR module to print
/// @param header Optional header text to display above the module
/// @param os Output stream to write to
void printProgram(ModuleOp module, StringRef header = "",
                  raw_ostream& os = llvm::errs());

} // namespace mlir

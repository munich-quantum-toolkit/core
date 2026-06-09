// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QCO_IR_QCOINTERFACES_TD
#define MLIR_DIALECT_QCO_IR_QCOINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// UnitaryOpInterface
//===----------------------------------------------------------------------===//

def UnitaryOpInterface : OpInterface<"UnitaryOpInterface"> {
  let description = [{
        This interface provides a unified API for all operations that apply or
        produce a unitary transformation in the QCO dialect. This includes base
        gates, user-defined gates, modifier operations (control, inverse, power),
        and sequences.

        The interface enables uniform introspection and composition capabilities
        across all unitary operations with value semantics.
    }];

  let cppNamespace = "::mlir::qco";

  // Size-specific bodies
  // Each helper only accepts the matching matrix type.
  defvar unitaryMatrix1x1MethodBody = [{
      if constexpr (requires { $_op.getUnitaryMatrix().has_value(); }) {
        using ValueT = std::remove_cvref_t<decltype(*std::declval<decltype($_op.getUnitaryMatrix())>())>;
        if constexpr (std::is_same_v<ValueT, Matrix1x1>) {
          if (auto matrix = $_op.getUnitaryMatrix()) {
            out = *matrix;
            return true;
          }
          return false;
        } else {
          return false;
        }
      } else if constexpr (requires { $_op.getUnitaryMatrix(); }) {
        using ValueT = std::remove_cvref_t<decltype($_op.getUnitaryMatrix())>;
        if constexpr (std::is_same_v<ValueT, Matrix1x1>) {
          out = $_op.getUnitaryMatrix();
          return true;
        }
        return false;
      } else {
        llvm::reportFatalUsageError("Operation '" + $_op.getBaseSymbol() + "' has no unitary matrix definition!");
      }
    }];

  defvar unitaryMatrix2x2MethodBody = [{
      if constexpr (requires { $_op.getUnitaryMatrix().has_value(); }) {
        using ValueT = std::remove_cvref_t<decltype(*std::declval<decltype($_op.getUnitaryMatrix())>())>;
        if constexpr (std::is_same_v<ValueT, Matrix2x2>) {
          if (auto matrix = $_op.getUnitaryMatrix()) {
            out = *matrix;
            return true;
          }
          return false;
        } else {
          return false;
        }
      } else if constexpr (requires { $_op.getUnitaryMatrix(); }) {
        using ValueT = std::remove_cvref_t<decltype($_op.getUnitaryMatrix())>;
        if constexpr (std::is_same_v<ValueT, Matrix2x2>) {
          out = $_op.getUnitaryMatrix();
          return true;
        }
        return false;
      } else {
        llvm::reportFatalUsageError("Operation '" + $_op.getBaseSymbol() + "' has no unitary matrix definition!");
      }
    }];

  defvar unitaryMatrix4x4MethodBody = [{
      if constexpr (requires { $_op.getUnitaryMatrix().has_value(); }) {
        using ValueT = std::remove_cvref_t<decltype(*std::declval<decltype($_op.getUnitaryMatrix())>())>;
        if constexpr (std::is_same_v<ValueT, Matrix4x4>) {
          if (auto matrix = $_op.getUnitaryMatrix()) {
            out = *matrix;
            return true;
          }
          return false;
        } else {
          return false;
        }
      } else if constexpr (requires { $_op.getUnitaryMatrix(); }) {
        using ValueT = std::remove_cvref_t<decltype($_op.getUnitaryMatrix())>;
        if constexpr (std::is_same_v<ValueT, Matrix4x4>) {
          out = $_op.getUnitaryMatrix();
          return true;
        }
        return false;
      } else {
        llvm::reportFatalUsageError("Operation '" + $_op.getBaseSymbol() + "' has no unitary matrix definition!");
      }
    }];

  defvar unitaryMatrixDynamicMethodBody = [{
    if constexpr (requires { $_op.getUnitaryMatrix().has_value(); }) {
        if (auto matrix = $_op.getUnitaryMatrix()) {
          out.assignFrom(*matrix);
          return true;
        }
        return false;
      } else if constexpr (requires { $_op.getUnitaryMatrix(); }) {
        out.assignFrom($_op.getUnitaryMatrix());
        return true;
      } else {
        llvm::reportFatalUsageError("Operation '" + $_op.getBaseSymbol() + "' has no unitary matrix definition!");
      }
    }];

  let methods = [
      // Qubit accessors
      InterfaceMethod<
          "Returns the number of qubits acted on by the unitary operation.",
          "size_t", "getNumQubits", (ins)>,
      InterfaceMethod<
          "Returns the number of target qubits (excluding control qubits).",
          "size_t", "getNumTargets", (ins)>,
      InterfaceMethod<
          "Returns the number of control qubits (both positive and negative).",
          "size_t", "getNumControls", (ins)>,
      InterfaceMethod<
          "Returns the i-th input qubit (targets + controls combined).",
          "Value", "getInputQubit", (ins "size_t":$i)>,
      InterfaceMethod<
          "Returns a range of all input qubits (targets + controls combined).",
          "OperandRange", "getInputQubits", (ins)>,
      InterfaceMethod<
          "Returns the i-th output qubit (targets + controls combined).",
          "Value", "getOutputQubit", (ins "size_t":$i)>,
      InterfaceMethod<
          "Returns a range of all output qubits (targets + controls combined).",
          "ResultRange", "getOutputQubits", (ins)>,
      InterfaceMethod<"Returns the i-th target input qubit.", "Value",
                      "getInputTarget", (ins "size_t":$i)>,
      InterfaceMethod<"Returns a range of all target input qubits.",
                      "OperandRange", "getInputTargets", (ins)>,
      InterfaceMethod<"Returns the i-th target output qubit.", "Value",
                      "getOutputTarget", (ins "size_t":$i)>,
      InterfaceMethod<"Returns a range of all target output qubits.",
                      "ResultRange", "getOutputTargets", (ins)>,
      InterfaceMethod<"Returns the i-th control input qubit.", "Value",
                      "getInputControl", (ins "size_t":$i)>,
      InterfaceMethod<"Returns a range of all control input qubits.",
                      "OperandRange", "getInputControls", (ins)>,
      InterfaceMethod<"Returns the i-th control output qubit.", "Value",
                      "getOutputControl", (ins "size_t":$i)>,
      InterfaceMethod<"Returns a range of all control output qubits.",
                      "ResultRange", "getOutputControls", (ins)>,
      InterfaceMethod<
          "Returns the input qubit corresponding to the given output qubit.",
          "Value", "getInputForOutput", (ins "Value":$output)>,
      InterfaceMethod<
          "Returns the output qubit corresponding to the given input qubit.",
          "Value", "getOutputForInput", (ins "Value":$input)>,

      // Parameter handling
      InterfaceMethod<"Returns the number of parameters.", "size_t",
                      "getNumParams", (ins)>,
      InterfaceMethod<"Returns the i-th parameter.", "Value", "getParameter",
                      (ins "size_t":$i)>,
      InterfaceMethod<"Returns a range of all parameters.", "ValueRange",
                      "getParameters", (ins)>,

      // Convenience methods
      InterfaceMethod<"Returns true if the operation has any control qubits, "
                      "otherwise false.",
                      "bool", "isControlled", (ins),
                      [{ return $_op.getNumControls() > 0; }]>,
      InterfaceMethod<
          "Returns true if the operation only acts on a single qubit.", "bool",
          "isSingleQubit", (ins), [{ return $_op.getNumQubits() == 1; }]>,
      InterfaceMethod<"Returns true if the operation acts on two qubits.",
                      "bool", "isTwoQubit", (ins),
                      [{ return $_op.getNumQubits() == 2; }]>,

      // Identification
      InterfaceMethod<"Returns the base symbol/mnemonic of the operation.",
                      "StringRef", "getBaseSymbol", (ins)>,

      // Unitary matrix helpers
      InterfaceMethod<"Populates the given 1x1 unitary matrix if possible.",
                      "bool", "getUnitaryMatrix1x1", (ins "Matrix1x1&":$out),
                      unitaryMatrix1x1MethodBody>,
      InterfaceMethod<"Populates the given 2x2 unitary matrix if possible.",
                      "bool", "getUnitaryMatrix2x2", (ins "Matrix2x2&":$out),
                      unitaryMatrix2x2MethodBody>,
      InterfaceMethod<"Populates the given 4x4 unitary matrix if possible.",
                      "bool", "getUnitaryMatrix4x4", (ins "Matrix4x4&":$out),
                      unitaryMatrix4x4MethodBody>,
      InterfaceMethod<"Populates the given dynamic unitary matrix.", "bool",
                      "getUnitaryMatrixDynamic", (ins "DynamicMatrix&":$out),
                      unitaryMatrixDynamicMethodBody>];

  let extraClassDeclaration = [{
        template<typename MatrixType>
        std::optional<MatrixType> getUnitaryMatrix() {
          static_assert(is_supported_matrix_v<MatrixType>,
                        "MatrixType must be Matrix1x1, Matrix2x2, Matrix4x4, or DynamicMatrix");
          MatrixType out;
          bool result = false;

          if constexpr (std::is_same_v<MatrixType, Matrix1x1>) {
            result = this->getUnitaryMatrix1x1(out);
          } else if constexpr (std::is_same_v<MatrixType, Matrix2x2>) {
            result = this->getUnitaryMatrix2x2(out);
          } else if constexpr (std::is_same_v<MatrixType, Matrix4x4>) {
            result = this->getUnitaryMatrix4x4(out);
          } else if constexpr (std::is_same_v<MatrixType, DynamicMatrix>) {
            result = this->getUnitaryMatrixDynamic(out);
          }

          if (result) {
            return out;
          }
          return std::nullopt;
        }
    }];
}

#endif // MLIR_DIALECT_QCO_IR_QCOINTERFACES_TD

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

        Every statically known floating-point value in a parameter expression
        must be finite. Dynamic parameter values have the same runtime
        precondition.
    }];

  let cppNamespace = "::mlir::qco";

  defvar unitaryMatrixMethodBody = [{
      using OutputT = std::remove_cvref_t<decltype(out)>;
      auto assign = [&out](const auto& matrix) {
        using InputT = std::remove_cvref_t<decltype(matrix)>;
        if constexpr (std::is_same_v<OutputT, DynamicMatrix>) {
          out.assignFrom(matrix);
          return true;
        } else if constexpr (std::is_same_v<OutputT, InputT>) {
          out = matrix;
          return true;
        } else if constexpr (std::is_same_v<InputT, DynamicMatrix>) {
          return out.assignFrom(matrix);
        } else {
          return false;
        }
      };
      if constexpr (requires { $_op.getUnitaryMatrix().has_value(); }) {
        if (auto matrix = $_op.getUnitaryMatrix()) {
          return assign(*matrix);
        }
        return false;
      } else if constexpr (requires { $_op.getUnitaryMatrix(); }) {
        return assign($_op.getUnitaryMatrix());
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
      InterfaceMethod<"Returns a range of all parameters.", "OperandRange",
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
      InterfaceMethod<"Returns true if the operation has a compile-time known "
                      "unitary matrix representation, false otherwise.",
                      "bool", "hasCompileTimeKnownUnitaryMatrix", (ins)>,
      InterfaceMethod<"Populates the given 1x1 unitary matrix if possible.",
                      "bool", "getUnitaryMatrix1x1", (ins "Matrix1x1&":$out),
                      unitaryMatrixMethodBody>,
      InterfaceMethod<"Populates the given 2x2 unitary matrix if possible.",
                      "bool", "getUnitaryMatrix2x2", (ins "Matrix2x2&":$out),
                      unitaryMatrixMethodBody>,
      InterfaceMethod<"Populates the given 4x4 unitary matrix if possible.",
                      "bool", "getUnitaryMatrix4x4", (ins "Matrix4x4&":$out),
                      unitaryMatrixMethodBody>,
      InterfaceMethod<"Populates the given 8x8 unitary matrix if possible.",
                      "bool", "getUnitaryMatrix8x8", (ins "Matrix8x8&":$out),
                      unitaryMatrixMethodBody>,
      InterfaceMethod<"Populates the given dynamic unitary matrix.", "bool",
                      "getUnitaryMatrixDynamic", (ins "DynamicMatrix&":$out),
                      unitaryMatrixMethodBody>];

  let extraClassDeclaration = [{
        template<SupportedMatrix MatrixType>
        std::optional<MatrixType> getUnitaryMatrix() {
          MatrixType out;
          bool result = false;

          if constexpr (std::is_same_v<MatrixType, Matrix1x1>) {
            result = this->getUnitaryMatrix1x1(out);
          } else if constexpr (std::is_same_v<MatrixType, Matrix2x2>) {
            result = this->getUnitaryMatrix2x2(out);
          } else if constexpr (std::is_same_v<MatrixType, Matrix4x4>) {
            result = this->getUnitaryMatrix4x4(out);
          } else if constexpr (std::is_same_v<MatrixType, Matrix8x8>) {
            result = this->getUnitaryMatrix8x8(out);
          } else if constexpr (std::is_same_v<MatrixType, DynamicMatrix>) {
            result = this->getUnitaryMatrixDynamic(out);
          }

          if (result) {
            return out;
          }
          return std::nullopt;
        }
    }];

  let verify = [{
    return ::mlir::qco::verifyUnitaryOpInterface($_op);
  }];
}

#endif // MLIR_DIALECT_QCO_IR_QCOINTERFACES_TD

/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ProgramBuilder.h"

#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/Utils/DenseUnitary.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QC/Translation/StandardGate.h"

#include "nanobind/ndarray.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/variant.h"
#include "nanobind/stl/vector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mqt {
namespace nb = nanobind;
using namespace nb::literals;
namespace {
using Parameter = std::variant<double, std::string>;
using Matrix = nb::ndarray<const std::complex<double>, nb::ndim<2>,
                           nb::c_contig, nb::device::cpu>;

class PythonQCBuilder {
public:
  explicit PythonQCBuilder(const uint32_t numQubits, const uint32_t numClbits)
      : context_(mlir::createCompilerContext()), builder_(context_.get()),
        numClbits_(numClbits) {
    builder_.initialize(mlir::TypeRange{});
    module_ = builder_.getInsertionBlock()
                  ->getParentOp()
                  ->getParentOfType<mlir::ModuleOp>();
    qubits_.reserve(numQubits);
    for (uint32_t index = 0; index < numQubits; ++index) {
      qubits_.push_back(builder_.allocQubit());
    }
    if (numClbits != 0) {
      bits_ = builder_.allocClassicalBitRegister(numClbits);
      builder_.retype(bits_.getType());
    }
  }

  PythonQCBuilder& gate(const std::string& name,
                        const std::vector<uint32_t>& targets,
                        const std::vector<Parameter>& parameters,
                        const std::vector<uint32_t>& controls) {
    requireActive();
    const auto* descriptor =
        mlir::qc::lookupStandardGateByOperationSymbol(name);
    if (descriptor == nullptr ||
        descriptor->gate == mlir::qc::StandardGate::BuiltinU ||
        descriptor->gate == mlir::qc::StandardGate::CU) {
      throw nb::value_error(
          "unknown primitive gate; use controls for controlled gates");
    }
    if (targets.size() != descriptor->targetCount ||
        parameters.size() != descriptor->parameterCount) {
      throw nb::value_error(
          "gate operand or parameter count does not match its definition");
    }
    auto operands = checkedQubits(targets, controls);
    for (const auto& parameter : parameters) {
      if (const auto* number = std::get_if<double>(&parameter)) {
        if (!std::isfinite(*number)) {
          throw nb::value_error("gate parameters must be finite");
        }
      } else {
        const auto& symbol = std::get<std::string>(parameter);
        if (symbol.empty() || symbol.find('\0') != std::string::npos) {
          throw nb::value_error(
              "parameter names must be nonempty and contain no NUL");
        }
      }
    }
    llvm::SmallVector<mlir::Value> values;
    for (const auto& parameter : parameters) {
      values.push_back(parameterValue(parameter));
    }
    const auto emit = [&](mlir::ValueRange qubits) {
      const auto result = mlir::qc::emitStandardGate(
          builder_, builder_.getLoc(), descriptor->gate, values, qubits);
      assert(mlir::succeeded(result) &&
             "validated primitive gate emission failed");
    };
    if (controls.empty()) {
      emit(operands);
    } else {
      builder_.ctrl(mlir::ValueRange(operands).take_back(controls.size()),
                    mlir::ValueRange(operands).take_front(targets.size()),
                    emit);
    }
    return *this;
  }

  PythonQCBuilder& unitary(const Matrix& matrix,
                           const std::vector<uint32_t>& targets) {
    requireActive();
    auto operands = checkedQubits(targets);
    if (targets.empty() ||
        targets.size() > mlir::mqt::MAX_DENSE_UNITARY_QUBITS) {
      throw nb::value_error(
          "dense unitaries require between one and eight target qubits");
    }
    const auto dimension = uint64_t{1} << targets.size();
    if (matrix.shape(0) != dimension || matrix.shape(1) != dimension) {
      throw nb::value_error("matrix shape must be (2**n, 2**n) for n targets");
    }
    const auto type = mlir::RankedTensorType::get(
        {static_cast<int64_t>(dimension), static_cast<int64_t>(dimension)},
        mlir::ComplexType::get(builder_.getF64Type()));
    const auto attribute = mlir::DenseElementsAttr::get(
        type, llvm::ArrayRef(matrix.data(), matrix.size()));
    auto op = mlir::qc::UnitaryOp::create(builder_, builder_.getLoc(),
                                          attribute, operands);
    if (mlir::failed(mlir::verify(op))) {
      op.erase();
      throw nb::value_error(
          "matrix must be finite and unitary within Core's tolerance");
    }
    return *this;
  }

  PythonQCBuilder& measure(const uint32_t qubit, const uint32_t bit) {
    requireActive();
    if (bit >= numClbits_) {
      throw nb::index_error("classical bit index out of range");
    }
    builder_.measure(checkedQubits({qubit}).front(), bits_,
                     static_cast<int64_t>(bit));
    return *this;
  }

  PythonQCBuilder& reset(const uint32_t qubit) {
    requireActive();
    builder_.reset(checkedQubits({qubit}).front());
    return *this;
  }

  mlir::QCProgram finish() {
    requireActive();
    auto finished =
        builder_.finalize(bits_ ? mlir::ValueRange{bits_} : mlir::ValueRange{});
    module_.release();
    auto program = mlir::QCProgram::fromModule(context_, std::move(finished));
    if (!program) {
      throw nb::value_error("constructed QC program failed verification");
    }
    return std::move(*program);
  }

private:
  void requireActive() const {
    if (!module_) {
      throw nb::value_error("builder has already been finished");
    }
  }

  [[nodiscard]] llvm::SmallVector<mlir::Value>
  checkedQubits(const std::vector<uint32_t>& targets,
                const std::vector<uint32_t>& controls = {}) const {
    llvm::SmallVector<uint32_t> indices(targets.begin(), targets.end());
    indices.append(controls.begin(), controls.end());
    llvm::SmallVector<mlir::Value> values;
    for (const auto index : indices) {
      if (index >= qubits_.size()) {
        throw nb::index_error("qubit index out of range");
      }
      values.push_back(qubits_[index]);
    }
    std::ranges::sort(indices);
    if (std::ranges::adjacent_find(indices) != indices.end()) {
      throw nb::value_error("qubit operands must be distinct");
    }
    return values;
  }

  mlir::Value parameterValue(const Parameter& parameter) {
    if (const auto* number = std::get_if<double>(&parameter)) {
      return builder_.floatConstant(*number);
    }
    const auto& name = std::get<std::string>(parameter);
    if (const auto found = parameters_.find(name); found != parameters_.end()) {
      return found->second;
    }
    auto entry = mlir::mqt::getEntryPoint(*module_);
    const auto index = entry.getNumArguments();
    const auto attrs = builder_.getDictionaryAttr({
        builder_.getNamedAttr(
            mlir::mqt::MQTDialect::InputNameAttrHelper::getNameStr(),
            builder_.getStringAttr(name)),
    });
    if (mlir::failed(entry.insertArgument(index, builder_.getF64Type(), attrs,
                                          builder_.getLoc()))) {
      throw nb::value_error("cannot add parameter input");
    }
    auto value = entry.getArgument(index);
    parameters_.try_emplace(name, value);
    return value;
  }

  std::shared_ptr<mlir::MLIRContext> context_;
  mlir::qc::QCProgramBuilder builder_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  llvm::SmallVector<mlir::Value> qubits_;
  mlir::Value bits_;
  uint32_t numClbits_;
  llvm::StringMap<mlir::Value> parameters_;
};
} // namespace

void registerProgramBuilder(nb::module_& module) {
  nb::class_<PythonQCBuilder>(
      module, "QCProgramBuilder",
      R"pb(Build a straight-line QC program without a frontend dependency.

Qubits are zero-based logical positions. Parameters are finite floats or names
of f64 inputs, reused when the same name occurs. ``finish()`` transfers the
program to a QCProgram and makes the builder unusable.)pb")
      .def(nb::init<uint32_t, uint32_t>(), "num_qubits"_a, "num_clbits"_a = 0)
      .def("gate", &PythonQCBuilder::gate, "name"_a, "qubits"_a,
           "parameters"_a = std::vector<Parameter>{}, nb::kw_only(),
           "controls"_a = std::vector<uint32_t>{},
           nb::rv_policy::reference_internal,
           R"pb(Append a primitive QC gate with optional positive controls.

Use QC operation names, for example ``h``, ``x``, ``ry``, ``rzz``, or ``u``.
For CX use ``gate("x", [target], controls=[control])``. The first qubit is
the most-significant tensor factor. Arity, distinct operands, and parameters
are checked before changing the builder.)pb")
      .def(
          "unitary", &PythonQCBuilder::unitary, "matrix"_a, "qubits"_a,
          nb::rv_policy::reference_internal,
          nb::sig("def unitary(self, matrix: "
                  "typing.Annotated[numpy.typing.NDArray[numpy.complex128], "
                  "{'shape': (None, None)}], "
                  "qubits: collections.abc.Sequence[int]) -> QCProgramBuilder"),
          "Append a dense unitary; first target is the most-significant basis "
          "bit.")
      .def("measure", &PythonQCBuilder::measure, "qubit"_a, "bit"_a,
           nb::rv_policy::reference_internal, "Measure into a classical bit.")
      .def("reset", &PythonQCBuilder::reset, "qubit"_a,
           nb::rv_policy::reference_internal, "Reset a qubit to zero.")
      .def("finish", &PythonQCBuilder::finish,
           "Return the verified QC program and invalidate this builder.");
}
} // namespace mqt

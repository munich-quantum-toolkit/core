/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/BV.hpp"
#include "bench/FanOut.hpp"
#include "bench/GHZ.hpp"
#include "bench/Grover.hpp"
#include "bench/GroverWeakMeasurement.hpp"
#include "bench/MagicStateDistillation.hpp"
#include "bench/ModularMultiplier.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QAOA.hpp"
#include "bench/QFT.hpp"
#include "bench/QFTAdder.hpp"
#include "bench/QPE.hpp"
#include "bench/RepeatUntilSuccess.hpp"
#include "bench/Shor.hpp"
#include "bench/Teleportation.hpp"
#include "bench/WState.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "mqt/Compiler/CompilationOptions.h"
#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/QDMIAdapter.h"
#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/bench/Generate.h"
#include "qdmi/QDMI.hpp"

#include "Result.hpp"
#include "Target/Qiskit/Qiskit.h"

#include "capnp/common.h"
#include "capnp/message.h"
#include "capnp/serialize.h"
#include "kj/array.h"
#include "kj/exception.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/filesystem.h"
#include "nanobind/stl/map.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/variant.h"
#include "nanobind/stl/vector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cassert>
#include <cctype>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

using DenseVector = nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::ndim<1>,
                                nb::c_contig, nb::device::cpu>;
using DenseMatrix = nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::ndim<2>,
                                nb::c_contig, nb::device::cpu>;
using JeffSegment = nb::ndarray<nb::memview, const uint8_t, nb::ndim<1>,
                                nb::c_contig, nb::device::cpu>;

using PythonCustomJobParameter =
    std::variant<std::string, bool, int, double, nb::bytes>;

[[nodiscard]] static std::optional<qdmi::CustomJobParameter>
toCustomJobParameter(const std::optional<PythonCustomJobParameter>& parameter) {
  if (!parameter) {
    return std::nullopt;
  }
  return std::visit(
      [](const auto& value) -> qdmi::CustomJobParameter {
        if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                     nb::bytes>) {
          return std::as_bytes(std::span(value.c_str(), value.size()));
        } else {
          return value;
        }
      },
      *parameter);
}

template <class T>
[[nodiscard]] static T takeResult(mlir::FailureOr<T>&& result) {
  if (mlir::failed(result)) {
    throw std::runtime_error(
        "Compiler action failed; see diagnostics for details.");
  }
  return *std::move(result);
}

template <class T> [[nodiscard]] static T takeResult(llvm::Expected<T> result) {
  if (!result) {
    const auto message = llvm::toString(result.takeError());
    throw nb::value_error(message.c_str());
  }
  return std::move(*result);
}

template <class T>
static void constructFromExpected(T& self, llvm::Expected<T>&& result) {
  std::construct_at(&self, takeResult(std::move(result)));
}

static void requireSuccess(const mlir::LogicalResult result) {
  if (mlir::failed(result)) {
    throw std::runtime_error(
        "Compiler action failed; see diagnostics for details.");
  }
}

static void requireValid(const mlir::Program& program) {
  if (!program.isValid()) {
    throw std::runtime_error(
        "This compiler program has already been consumed.");
  }
}

template <class ProgramType>
[[nodiscard]] static ProgramType copyProgram(const ProgramType& program) {
  requireValid(program);
  return program.copy();
}

template <class T>
[[nodiscard]] static nb::typed<nb::list, T>
toPythonList(llvm::ArrayRef<T> values) {
  nb::list_builder result(values.size());
  for (const auto& value : values) {
    result.put(value);
  }
  return result.commit();
}

namespace {
template <auto Function> struct FallibleFunctionAdapter;

template <class T, class... Args, mlir::FailureOr<T> (*Function)(Args...)>
struct FallibleFunctionAdapter<Function> {
  static T call(Args... args) {
    return takeResult(Function(std::forward<Args>(args)...));
  }
};

template <auto Method> struct FallibleMemberAdapter;

template <class Class, class... Args,
          mlir::LogicalResult (Class::*Method)(Args...)>
struct FallibleMemberAdapter<Method> {
  static void call(Class& self, Args... args) {
    if constexpr (std::is_base_of_v<mlir::Program, Class>) {
      requireValid(self);
    }
    requireSuccess((self.*Method)(std::forward<Args>(args)...));
  }
};

template <class Class, class... Args,
          mlir::LogicalResult (Class::*Method)(Args...) const>
struct FallibleMemberAdapter<Method> {
  static void call(const Class& self, Args... args) {
    if constexpr (std::is_base_of_v<mlir::Program, Class>) {
      requireValid(self);
    }
    requireSuccess((self.*Method)(std::forward<Args>(args)...));
  }
};
} // namespace

[[nodiscard]] static mlir::func::FuncOp
entryFunc(const mlir::QCOProgram& program) {
  requireValid(program);
  auto func = mlir::mqt::getEntryPoint(program.module());
  if (!func) {
    throw nb::value_error("QCO program has no func.func entry point");
  }
  return func;
}

[[nodiscard]] static std::mt19937_64 makeRng(const uint64_t seed) {
  if (seed == 0) {
    return std::mt19937_64(std::random_device{}());
  }
  return std::mt19937_64(seed);
}

/// Run @p fn under a diagnostic handler and raise the chosen Python exception,
/// appending any emitted MLIR diagnostics to @p message.
template <nb::exception_type Exception = nb::exception_type::value_error,
          typename Fn>
static auto withDiagnostics(mlir::MLIRContext* context, const char* message,
                            Fn&& fn) {
  std::string diagnostics;
  const mlir::ScopedDiagnosticHandler handler(
      context, [&](mlir::Diagnostic& diag) {
        if (!diagnostics.empty()) {
          diagnostics.push_back('\n');
        }
        llvm::raw_string_ostream os(diagnostics);
        if (!llvm::isa<mlir::UnknownLoc>(diag.getLocation())) {
          os << diag.getLocation() << ": ";
        }
        os << diag;
        return mlir::success();
      });
  auto result = std::forward<Fn>(fn)();
  if (mlir::failed(result)) {
    std::string full = message;
    if (!diagnostics.empty()) {
      full.append(": ").append(diagnostics);
    }
    throw nb::builtin_exception(Exception, full.c_str());
  }
  if constexpr (!std::is_same_v<decltype(result), mlir::LogicalResult>) {
    return *std::move(result);
  }
}

template <class T>
static void registerParameterBinding(nb::class_<T, mlir::Program>& binding) {
  binding
      .def_prop_ro(
          "parameters",
          [](const T& program) {
            requireValid(program);
            return program.parameters();
          },
          "Named f64 scalar and fixed-length array inputs in function argument "
          "order.")
      .def(
          "bind_parameters",
          [](T& program, const nb::typed<nb::mapping, nb::str,
                                         mlir::ParameterValue>& values) {
            requireValid(program);
            nb::dict snapshot;
            try {
              snapshot.update(values);
            } catch (const nb::python_error&) {
              throw nb::type_error("values must be a mapping with string keys");
            }
            std::map<std::string, mlir::ParameterValue> bindings;
            for (const auto [key, value] : snapshot) {
              std::string name;
              if (!nb::try_cast(key, name)) {
                throw nb::type_error(
                    "values must be a mapping with string keys");
              }
              // NumPy < 2.4 can coerce one-element rows to floats.
              if (nb::cast<int>(nb::getattr(value, "ndim", nb::int_(0))) > 1) {
                throw nb::value_error(
                    "array parameter values must be one-dimensional");
              }
              mlir::ParameterValue parameter;
              if (!nb::try_cast(value, parameter)) {
                throw nb::type_error(
                    "parameter values must be floats or one-dimensional "
                    "sequences of floats");
              }
              bindings.emplace(std::move(name), std::move(parameter));
            }
            withDiagnostics(program.module().getContext(),
                            "cannot bind parameters",
                            [&] { return program.bindParameters(bindings); });
          },
          "values"_a,
          R"pb(Bind named scalar and array parameters in place.

Inputs must be f64 scalars or fixed-length, one-dimensional f64 arrays.
Supply arrays as sequences or NumPy arrays with the expected length.
Partial binding preserves unbound inputs and their source identities.
Unknown names, mismatched scalar or array values, incorrect lengths,
non-finite values, and references to the entry point raise ValueError
without changing the program.

Binding folds expressions using the bound values. Folding failures may
leave the program modified. Call ``copy()`` first to preserve the input.)pb");
}

template <class T>
static void registerInspection(nb::class_<T, mlir::Program>& binding) {
  binding
      .def(
          "inspect",
          [](const T& program) {
            requireValid(program);
            return program.inspect();
          },
          R"pb(Inspect quantum resources and static IR statistics.

Returns a :class:`QuantumProgramInfo` snapshot with gate, control-flow, and
full operation counts.)pb")
      .def(
          "num_gates",
          [](const T& program) {
            requireValid(program);
            return program.numGates();
          },
          R"pb(Return the static gate count of the entry-point IR.

Unitary operations, measurements, and resets each count once. Barriers are
excluded. Modifiers and calls count atomically. Gates in every control-flow
region count once, regardless of runtime paths or loop iterations.)pb")
      .def(
          "num_single_qubit_gates",
          [](const T& program) {
            requireValid(program);
            return program.numSingleQubitGates();
          },
          R"pb(Count gates acting on exactly one qubit.

Uses the counting rules of :meth:`num_gates`, including measurements and resets.)pb")
      .def(
          "num_two_qubit_gates",
          [](const T& program) {
            requireValid(program);
            return program.numTwoQubitGates();
          },
          R"pb(Count gates acting on exactly two qubits.

Uses the counting rules of :meth:`num_gates`.)pb")
      .def(
          "gate_counts",
          [](const T& program) {
            requireValid(program);
            return program.gateCounts();
          },
          R"pb(Count entry-point gates by name.

Uses the counting rules of :meth:`num_gates`. Controls on a single primitive
gate add a ``c`` per control: ``cx``, ``ccx``.
Other single-gate modifiers use ``inv(h)``, ``pow(rx)``, or ``ctrl(inv(x))``;
multiple controls use ``ctrl(2,inv(x))``. Parameters do not split buckets.
Composite bodies or unused modifier targets retain ``ctrl``, ``inv``, or
``pow``. Calls use the callee name; explicit phases use ``gphase``.)pb")
      .def(
          "control_flow_counts",
          [](const T& program) {
            requireValid(program);
            return program.controlFlowCounts();
          },
          R"pb(Count entry-point control-flow operations.

Keys are full MLIR names such as ``scf.for`` and ``qco.if``. Every region is
visited once, without expanding calls. Region terminators
such as ``scf.yield`` are excluded.)pb");
}

[[nodiscard]] static qdmi::Device openQDMIDevice(
    const std::string& deviceId,
    std::optional<std::filesystem::path> driverPath,
    std::optional<std::string> token,
    std::optional<std::filesystem::path> authFile,
    std::optional<std::string> authUrl, std::optional<std::string> username,
    std::optional<std::string> password, std::optional<std::string> projectId,
    std::optional<std::string> custom1, std::optional<std::string> custom2,
    std::optional<std::string> custom3, std::optional<std::string> custom4,
    std::optional<std::string> custom5) {
  return qdmi::Session::openDevice(deviceId,
                                   {
                                       .driverPath = std::move(driverPath),
                                       .token = std::move(token),
                                       .authFile = std::move(authFile),
                                       .authUrl = std::move(authUrl),
                                       .username = std::move(username),
                                       .password = std::move(password),
                                       .projectId = std::move(projectId),
                                       .custom1 = std::move(custom1),
                                       .custom2 = std::move(custom2),
                                       .custom3 = std::move(custom3),
                                       .custom4 = std::move(custom4),
                                       .custom5 = std::move(custom5),
                                   });
}

template <class ProgramType>
[[nodiscard]] static ProgramType copiedOrConsumed(ProgramType& program,
                                                  const bool copy) {
  requireValid(program);
  if (copy) {
    return program.copy();
  }
  return std::move(program);
}

/// Check whether @p input unambiguously looks like source text.
[[nodiscard]] static bool isSourceString(const std::string_view input) {
  auto source = input;
  while (!source.empty() &&
         std::isspace(static_cast<unsigned char>(source.front())) != 0) {
    source.remove_prefix(1);
  }
  return input.find('\n') != std::string_view::npos ||
         input.find("OPENQASM") != std::string_view::npos ||
         (source.starts_with("module") && source.size() > 6U &&
          std::isspace(static_cast<unsigned char>(source[6])) != 0);
}

namespace {
/// Python output choices include objects constructed outside the native
/// compiler.
enum class OutputFormat : uint8_t {
  QCImport = static_cast<uint8_t>(mlir::ProgramFormat::QCImport),
  QCO = static_cast<uint8_t>(mlir::ProgramFormat::QCO),
  QCOOptimized = static_cast<uint8_t>(mlir::ProgramFormat::QCOOptimized),
  QC = static_cast<uint8_t>(mlir::ProgramFormat::QC),
  OpenQASM3 = static_cast<uint8_t>(mlir::ProgramFormat::OpenQASM3),
  Jeff = static_cast<uint8_t>(mlir::ProgramFormat::Jeff),
  QIRBase = static_cast<uint8_t>(mlir::ProgramFormat::QIRBase),
  QIRAdaptive = static_cast<uint8_t>(mlir::ProgramFormat::QIRAdaptive),
  IQMJSON = static_cast<uint8_t>(mlir::ProgramFormat::IQMJSON),
  Qiskit,
};
struct ImportedProgram {
  mlir::CompilerInput program;
  OutputFormat format;
};

struct PythonInputTypes {
  nb::object sequence =
      nb::module_::import_("collections.abc").attr("Sequence");
  nb::object pathLike = nb::module_::import_("os").attr("PathLike");
  nb::object string = nb::borrow<nb::object>(nb::str("").type());
  nb::object list = nb::borrow<nb::object>(nb::list().type());
  nb::object tuple = nb::borrow<nb::object>(nb::tuple().type());
  nb::object qiskitCircuit;

  [[nodiscard]] nb::handle quantumCircuit() {
    // Absence is retryable because Qiskit can be imported after this module.
    if (!qiskitCircuit) {
      const auto modules =
          nb::cast<nb::dict>(nb::module_::import_("sys").attr("modules"));
      const auto circuit = modules.get("qiskit.circuit", nb::none());
      if (!circuit.is_none()) {
        auto type = nb::getattr(circuit, "QuantumCircuit", nb::none());
        if (!type.is_none()) {
          qiskitCircuit = std::move(type);
        }
      }
    }
    return qiskitCircuit;
  }

  [[nodiscard]] nb::handle inputType(nb::handle item) {
    if (nb::isinstance<nb::str>(item)) {
      return string;
    }
    if (nb::hasattr(item, "__fspath__") && nb::isinstance(item, pathLike)) {
      return pathLike;
    }
    for (const auto type : {
             nb::type<mlir::QCProgram>(),
             nb::type<mlir::QCOProgram>(),
             nb::type<mlir::JeffProgram>(),
             nb::type<mlir::OpenQASMProgram>(),
         }) {
      if (nb::isinstance(item, type)) {
        return type;
      }
    }
    if (const auto type = quantumCircuit();
        type && nb::isinstance(item, type)) {
      return type;
    }
    return {};
  }
};

template <auto Function> struct PythonInputAdapter;

template <class Result, class... Args,
          Result (*Function)(PythonInputTypes&, Args...)>
struct PythonInputAdapter<Function> {
  std::shared_ptr<PythonInputTypes> types;

  Result operator()(Args... args) const {
    return Function(*types, std::forward<Args>(args)...);
  }
};
} // namespace

/// Construct a frontend program from a file path.
[[nodiscard]] static ImportedProgram
programFromPath(const std::filesystem::path& path) {
  if (path.empty()) {
    throw std::runtime_error("Input path must not be empty.");
  }

  std::error_code error;
  const auto exists = std::filesystem::exists(path, error);
  if (error) {
    throw std::runtime_error("Failed to inspect path '" + path.string() +
                             "': " + error.message());
  }
  if (!exists) {
    throw std::runtime_error("Input file '" + path.string() +
                             "' does not exist.");
  }
  if (!std::filesystem::is_regular_file(path, error) || error) {
    throw std::runtime_error("Input path '" + path.string() +
                             "' is not a file.");
  }

  const auto extension = path.extension().string();
  if (extension == ".jeff") {
    return {
        .program = takeResult(mlir::JeffProgram::fromFile(path)),
        .format = OutputFormat::Jeff,
    };
  }
  if (extension == ".mlir") {
    return {
        .program = takeResult(mlir::QCProgram::fromMLIRFile(path)),
        .format = OutputFormat::QC,
    };
  }
  if (extension == ".qasm") {
    return {
        .program = takeResult(mlir::QCProgram::fromOpenQASMFile(path)),
        .format = OutputFormat::OpenQASM3,
    };
  }
  throw std::runtime_error("Input file '" + path.string() +
                           "' has unsupported extension '" + extension + "'.");
}

/// Construct a frontend program from a string containing source or path.
[[nodiscard]] static ImportedProgram
programFromString(const std::string_view input) {
  if (isSourceString(input)) {
    if (input.find("OPENQASM") != std::string_view::npos) {
      return {
          .program = takeResult(mlir::QCProgram::fromOpenQASMString(input)),
          .format = OutputFormat::OpenQASM3,
      };
    }
    return {
        .program = takeResult(mlir::QCProgram::fromMLIRString(input)),
        .format = OutputFormat::QC,
    };
  }
  return programFromPath(std::filesystem::path(input));
}

/// Convert a Python object to a compiler program.
///
/// Consume typed MLIR programs unless @p copy requests an independent copy.
/// Text, paths and external circuit objects remain borrowed.
[[nodiscard]] static ImportedProgram programFromInput(PythonInputTypes& types,
                                                      const nb::handle program,
                                                      const bool copy) {
  if (nb::isinstance<nb::str>(program)) {
    const auto input = nb::cast<std::string_view>(program);
    const nb::gil_scoped_release release;
    return programFromString(input);
  }
  if (nb::hasattr(program, "__fspath__")) {
    const auto path = nb::cast<std::filesystem::path>(program);
    const nb::gil_scoped_release release;
    return programFromPath(path);
  }
  if (nb::isinstance<mlir::QCProgram>(program)) {
    auto& value = nb::cast<mlir::QCProgram&>(program);
    return {
        .program = copiedOrConsumed(value, copy),
        .format = OutputFormat::QC,
    };
  }
  if (nb::isinstance<mlir::QCOProgram>(program)) {
    auto& value = nb::cast<mlir::QCOProgram&>(program);
    return {
        .program = copiedOrConsumed(value, copy),
        .format = OutputFormat::QCOOptimized,
    };
  }
  if (nb::isinstance<mlir::JeffProgram>(program)) {
    auto& value = nb::cast<mlir::JeffProgram&>(program);
    return {
        .program = copiedOrConsumed(value, copy),
        .format = OutputFormat::Jeff,
    };
  }
  if (nb::isinstance<mlir::OpenQASMProgram>(program)) {
    const auto& value = nb::cast<const mlir::OpenQASMProgram&>(program);
    const nb::gil_scoped_release release;
    return {
        .program =
            takeResult(mlir::QCProgram::fromOpenQASMString(value.source())),
        .format = OutputFormat::OpenQASM3,
    };
  }

  if (const auto type = types.quantumCircuit();
      type && nb::isinstance(program, type)) {
    return {
        .program = bindings::qiskit::importCircuit(program),
        .format = OutputFormat::Qiskit,
    };
  }
  throw std::runtime_error(
      "Program type " + nb::cast<std::string>(program.type().attr("__name__")) +
      " is not supported.");
}

/// Resolve an open device or registered ID.
[[nodiscard]] static qdmi::Device resolveDevice(const nb::handle target) {
  if (nb::isinstance<qdmi::Device>(target)) {
    return nb::cast<qdmi::Device>(target);
  }
  if (nb::isinstance<nb::str>(target)) {
    const auto id = nb::cast<std::string>(target);
    const nb::gil_scoped_release release;
    return qdmi::Session::openDevice(id);
  }
  throw nb::type_error("target must be a QDMI Device or registered device ID");
}

/// Lowering constraints for the requested Python representation.
[[nodiscard]] static mlir::PayloadSpecification
payloadForOutput(OutputFormat output) {
  switch (output) {
  case OutputFormat::OpenQASM3:
  case OutputFormat::IQMJSON:
  case OutputFormat::QIRBase:
  case OutputFormat::QIRAdaptive:
    return takeResult(mlir::PayloadSpecification::forProgramFormat(
        static_cast<mlir::ProgramFormat>(output)));
  case OutputFormat::Qiskit:
    return bindings::qiskit::payloadSpecification();
  default:
    // Intermediate representations have no execution transport constraints.
    return takeResult(mlir::PayloadSpecification::create(
        {.id = "mqt", .version = "1.0", .profile = ""},
        {
            {.id = mlir::ProgramCapability::FORWARD_BRANCHING.str()},
            {.id = mlir::ProgramCapability::COUNTED_ITERATION.str()},
            {.id = mlir::ProgramCapability::CONDITIONAL_LOOP.str()},
            {.id = mlir::ProgramCapability::MULTIWAY_BRANCHING.str()},
        }));
  }
}

/// Compile one program with an already resolved target and output default.
[[nodiscard]] static nb::object compileSingleProgram(
    PythonInputTypes& types, const nb::handle program, const nb::handle target,
    const std::optional<mlir::TargetEnvironment>& targetEnvironment,
    std::optional<OutputFormat> output, bool copy,
    const std::string& qcoPipeline, const mlir::CompilationOptions& options) {
  auto input = programFromInput(types, program, copy);
  if (targetEnvironment && !output) {
    auto compiled = [&] {
      const nb::gil_scoped_release release;
      return takeResult(mlir::CompiledProgram::compile(
          std::move(input.program), *targetEnvironment, options));
    }();
    return nb::cast(std::move(compiled));
  }

  const auto selected = output.value_or(input.format);
  std::optional<mlir::TargetEnvironment> environment;
  if (!target.is_none()) {
    auto payload = payloadForOutput(selected);
    if (targetEnvironment) {
      payload = payload.restrictCapabilities(
          targetEnvironment->payloadSpecification());
      environment.emplace(targetEnvironment->target(), std::move(payload));
    } else {
      environment.emplace(nb::cast<const mlir::CompilerTarget&>(target),
                          std::move(payload));
    }
  }
  if (selected == OutputFormat::Qiskit) {
    std::visit(
        [](const auto& value) {
          if constexpr (std::is_base_of_v<
                            mlir::Program,
                            std::remove_cvref_t<decltype(value)>>) {
            if (value.module()->hasAttr("mqt.layout")) {
              throw std::runtime_error(
                  "Qiskit output requires input without layout metadata; use "
                  "to_qiskit() for a layout-preserving conversion");
            }
          }
        },
        input.program);
  }
  const auto nativeOutput = selected == OutputFormat::Qiskit
                                ? mlir::ProgramFormat::QC
                                : static_cast<mlir::ProgramFormat>(selected);
  const auto context = std::visit(
      [](const auto& value) -> std::shared_ptr<mlir::MLIRContext> {
        if constexpr (std::is_base_of_v<mlir::Program,
                                        std::remove_cvref_t<decltype(value)>>) {
          return value.context();
        } else {
          return nullptr;
        }
      },
      input.program);
  assert(context && "Python compiler inputs must be imported to MLIR");
  auto compiled = [&] {
    const nb::gil_scoped_release release;
    return withDiagnostics<nb::exception_type::runtime_error>(
        context.get(), "Compilation failed", [&] {
          return environment
                     ? mlir::runDefaultPipeline(std::move(input.program),
                                                nativeOutput, *environment,
                                                options)
                     : mlir::runDefaultPipeline(std::move(input.program),
                                                nativeOutput, qcoPipeline,
                                                options);
        });
  }();
  if (selected == OutputFormat::Qiskit) {
    auto qc = std::move(std::get<mlir::QCProgram>(compiled));
    return environment
               ? bindings::qiskit::exportCircuit(std::move(qc), *environment)
               : bindings::qiskit::exportCircuit(std::move(qc));
  }
  return nb::cast(std::move(compiled));
}

/// Compile a scalar input or a homogeneous sequence using one target snapshot.
[[nodiscard]] static nb::object compilePythonProgram(
    PythonInputTypes& types, const nb::handle program, const nb::handle target,
    std::optional<OutputFormat> output, bool copy,
    const std::string& qcoPipeline, const mlir::CompilationOptions& options) {
  if (!target.is_none() && qcoPipeline != "mqt-qco-default") {
    throw nb::value_error(
        "A custom QCO pipeline cannot be combined with a target");
  }
  if (!target.is_none() &&
      (output == OutputFormat::QCImport || output == OutputFormat::QCO)) {
    throw nb::value_error(
        "QC_IMPORT and raw QCO output stop before target compilation");
  }

  std::optional<nb::tuple> programs;
  const bool builtinSequence =
      program.type().is(types.list) || program.type().is(types.tuple);
  if (!nb::isinstance<nb::str>(program) &&
      !nb::hasattr(program, "__fspath__") &&
      !nb::isinstance<mlir::Program>(program) &&
      !nb::isinstance<mlir::OpenQASMProgram>(program) &&
      !nb::isinstance<nb::bytes>(program) &&
      !nb::isinstance<nb::bytearray>(program) &&
      !nb::isinstance<nb::memoryview>(program) &&
      (builtinSequence || nb::isinstance(program, types.sequence))) {
    if (builtinSequence || !types.inputType(program)) {
      programs.emplace(program);
      if (programs->empty()) {
        return nb::list();
      }
      const nb::handle first = (*programs)[0];
      const auto type = types.inputType(first);
      for (size_t i = 0; i < programs->size(); ++i) {
        const nb::handle item = (*programs)[i];
        if (!type || (!item.type().is(first.type()) &&
                      !types.inputType(item).is(type))) {
          throw nb::type_error(("Program sequences must contain one supported "
                                "input kind; invalid "
                                "element at index " +
                                std::to_string(i))
                                   .c_str());
        }
      }
    }
  }

  std::optional<mlir::TargetEnvironment> environment;
  const bool compilerTarget = nb::isinstance<mlir::CompilerTarget>(target);
  const bool deviceTarget =
      nb::isinstance<qdmi::Device>(target) || nb::isinstance<nb::str>(target);
  if (deviceTarget) {
    const auto device = resolveDevice(target);
    const nb::gil_scoped_release release;
    environment = takeResult(mlir::targetEnvironmentFromDevice(device));
  } else if (!target.is_none() && !compilerTarget) {
    environment.emplace(
        bindings::qiskit::importTarget(target, nb::none(), std::nullopt),
        bindings::qiskit::payloadSpecification(target));
    output = output.value_or(OutputFormat::Qiskit);
  }

  if (!programs) {
    return compileSingleProgram(types, program, target, environment, output,
                                copy, qcoPipeline, options);
  }
  nb::list_builder results(programs->size());
  for (const auto item : *programs) {
    results.put(compileSingleProgram(types, item, target, environment, output,
                                     copy, qcoPipeline, options));
  }
  return results.commit();
}

/// Compile or submit through the shared C++ QDMI adapter.
[[nodiscard]] static qdmi::Job
submitProgram(PythonInputTypes& types, const nb::handle program,
              const nb::handle target, int64_t numShots,
              std::optional<QDMI_Program_Format> programFormat,
              const std::optional<PythonCustomJobParameter>& custom1,
              const std::optional<PythonCustomJobParameter>& custom2,
              const std::optional<PythonCustomJobParameter>& custom3,
              const std::optional<PythonCustomJobParameter>& custom4,
              const std::optional<PythonCustomJobParameter>& custom5,
              std::optional<mlir::CompilationOptions> options) {
  if (numShots < 0) {
    throw nb::value_error("num_shots must be nonnegative");
  }
  const auto device = resolveDevice(target);
  const auto params = std::array{
      toCustomJobParameter(custom1), toCustomJobParameter(custom2),
      toCustomJobParameter(custom3), toCustomJobParameter(custom4),
      toCustomJobParameter(custom5),
  };
  if (nb::isinstance<mlir::CompiledProgram>(program)) {
    if (options) {
      throw nb::value_error(
          "Compilation options do not apply to an already compiled program");
    }
    const auto& compiled = nb::cast<const mlir::CompiledProgram&>(program);
    if (programFormat && *programFormat != compiled.programFormat()) {
      throw nb::value_error(
          "program_format conflicts with the compiled payload");
    }
    const nb::gil_scoped_release release;
    return takeResult(mlir::submitProgram(device, compiled, numShots, params[0],
                                          params[1], params[2], params[3],
                                          params[4]));
  }
  auto input = programFromInput(types, program, true).program;
  const nb::gil_scoped_release release;
  return takeResult(
      mlir::submitProgram(device, std::move(input), numShots, programFormat,
                          params[0], params[1], params[2], params[3], params[4],
                          options.value_or(mlir::CompilationOptions{})));
}

template <class Function>
[[nodiscard]] static auto withQCOProgram(PythonInputTypes& types,
                                         const nb::handle program,
                                         Function&& function) {
  if (nb::isinstance<mlir::QCOProgram>(program)) {
    return std::forward<Function>(function)(
        nb::cast<const mlir::QCOProgram&>(program));
  }
  auto compiled = [&] {
    auto input = programFromInput(types, program, true).program;
    const nb::gil_scoped_release release;
    return takeResult(
        mlir::runDefaultPipeline(std::move(input), mlir::ProgramFormat::QCO));
  }();
  return std::forward<Function>(function)(std::get<mlir::QCOProgram>(compiled));
}

[[nodiscard]] static dd::MatrixDD
buildQCOFunctionality(const mlir::QCOProgram& program, dd::Package& ddPackage) {
  auto func = entryFunc(program);
  return withDiagnostics(
      func.getContext(), "cannot build DD functionality for this QCO program",
      [&] { return mlir::qco::buildFunctionality(func, ddPackage); });
}

[[nodiscard]] static dd::VectorDD simulateQCO(const mlir::QCOProgram& program,
                                              const dd::VectorDD& initialState,
                                              dd::Package& ddPackage,
                                              uint64_t seed) {
  if (dd::VectorDD::trackingRequired(initialState) &&
      !ddPackage.getRootSet<dd::vNode>().contains(initialState)) {
    throw nb::value_error(
        "initial_state must have a live reference in dd_package");
  }
  auto func = entryFunc(program);
  auto rng = makeRng(seed);
  return withDiagnostics(
      func.getContext(), "cannot simulate this QCO program",
      [&] { return mlir::qco::simulate(func, initialState, ddPackage, rng); });
}

[[nodiscard]] static std::map<std::string, size_t>
sampleQCO(const mlir::QCOProgram& program, size_t shots, uint64_t seed) {
  auto func = entryFunc(program);
  return withDiagnostics(func.getContext(), "cannot sample this QCO program",
                         [&] { return mlir::qco::sample(func, shots, seed); });
}

[[nodiscard]] static DenseVector toDenseVector(const dd::VectorDD& state) {
  if (!state.isTerminal()) {
    const auto numQubits = static_cast<size_t>(state.p->v) + 1U;
    if (numQubits >= std::numeric_limits<size_t>::digits ||
        (size_t{1} << numQubits) >
            std::numeric_limits<size_t>::max() / sizeof(std::complex<dd::fp>)) {
      throw nb::value_error(
          "dense statevector dimensions exceed addressable memory");
    }
  }
  auto dataPtr =
      std::make_unique<dd::CVec>(bindings::unwrap(dd::getVector(state)));
  auto* const data = dataPtr->data();
  const auto size = dataPtr->size();
  const nb::capsule owner(dataPtr.get(), [](void* ptr) noexcept {
    delete static_cast<dd::CVec*>(ptr);
  });
  [[maybe_unused]] const auto* const releasedDataPtr = dataPtr.release();
  return DenseVector(data, {size}, owner);
}

[[nodiscard]] static DenseMatrix toDenseMatrix(const dd::MatrixDD& matrix,
                                               size_t numQubits) {
  if (numQubits >= std::numeric_limits<size_t>::digits) {
    throw nb::value_error("dense unitary dimensions exceed addressable memory");
  }

  const size_t dim = size_t{1} << numQubits;
  if (dim >
      std::numeric_limits<size_t>::max() / sizeof(std::complex<dd::fp>) / dim) {
    throw nb::value_error("dense unitary dimensions exceed addressable memory");
  }
  auto dataPtr = std::make_unique<dd::CVec>(dim * dim);
  auto* const data = dataPtr->data();
  ::mqt::bindings::unwrap(dd::traverseMatrix(
      matrix, std::complex<dd::fp>{1., 0.}, 0ULL, 0ULL,
      [data, dim](size_t i, size_t j, const std::complex<dd::fp>& value) {
        data[i * dim + j] = value;
      },
      numQubits));
  const nb::capsule owner(dataPtr.get(), [](void* ptr) noexcept {
    delete static_cast<dd::CVec*>(ptr);
  });
  [[maybe_unused]] const auto* const releasedDataPtr = dataPtr.release();
  return DenseMatrix(data, {dim, dim}, owner);
}

[[nodiscard]] static DenseMatrix
buildDenseFunctionality(PythonInputTypes& types, const nb::handle program) {
  return withQCOProgram(types, program, [](const mlir::QCOProgram& qco) {
    auto ddPackage = ::mqt::bindings::unwrap(dd::Package::create(0));
    const auto matrix = buildQCOFunctionality(qco, *ddPackage);
    return toDenseMatrix(matrix, ddPackage->qubits());
  });
}

[[nodiscard]] static DenseVector simulateDense(PythonInputTypes& types,
                                               const nb::handle program) {
  return withQCOProgram(types, program, [](const mlir::QCOProgram& qco) {
    auto ddPackage = ::mqt::bindings::unwrap(dd::Package::create(0));
    auto func = entryFunc(qco);
    const auto state = withDiagnostics(
        func.getContext(), "cannot simulate this QCO program",
        [&] { return mlir::qco::simulateStatevector(func, *ddPackage); });
    return toDenseVector(state);
  });
}

[[nodiscard]] static std::map<std::string, size_t>
sample(PythonInputTypes& types, const nb::handle program, size_t shots,
       uint64_t seed) {
  return withQCOProgram(types, program, [&](const mlir::QCOProgram& qco) {
    return sampleQCO(qco, shots, seed);
  });
}

template <class Benchmark>
[[nodiscard]] static mlir::QCProgram
generateBenchmark(const Benchmark& benchmark) {
  auto generated = bench::generate(benchmark);
  if (mlir::failed(generated)) {
    throw std::runtime_error("failed to generate benchmark");
  }
  return std::move(*generated);
}

// NOLINTNEXTLINE(misc-use-internal-linkage): module entry point.
void registerMLIR(nb::module_& m) {

  m.doc() = "MQT Core MLIR compiler bindings.";

  nb::module_::import_("typing");
  const auto inputTypes = std::make_shared<PythonInputTypes>();
  [[maybe_unused]] const auto initialQiskitType = inputTypes->quantumCircuit();

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  m.def("_generate_benchmark", &generateBenchmark<bench::TYPE>, "benchmark"_a, \
        "Generate the QC program for a benchmark instance.");
#include "bench/BenchmarkFamilies.inc"

  nb::enum_<mlir::QIRProfile>(m, "QIRProfile", "QIR target profiles.")
      .value("BASE", mlir::QIRProfile::Base, "The QIR Base Profile.")
      .value("ADAPTIVE", mlir::QIRProfile::Adaptive,
             "The QIR Adaptive Profile.");
  nb::enum_<OutputFormat>(m, "OutputFormat", "Default compiler output formats.")
      .value("QC_IMPORT", OutputFormat::QCImport,
             "QC directly after frontend import.")
      .value("QCO", OutputFormat::QCO,
             "QCO immediately after conversion, before optimization.")
      .value("QCO_OPTIMIZED", OutputFormat::QCOOptimized,
             "QCO after the configured optimization pipeline.")
      .value("QC", OutputFormat::QC, "QC after the optimized QCO round trip.")
      .value("OPENQASM3", OutputFormat::OpenQASM3,
             "OpenQASM 3 after the optimized QCO round trip.")
      .value("IQM_JSON", OutputFormat::IQMJSON,
             "An IQM circuit for an explicit physical target.")
      .value("JEFF", OutputFormat::Jeff, "Serializable ``jeff`` MLIR.")
      .value("QIR_BASE", OutputFormat::QIRBase, "QIR for the Base Profile.")
      .value("QIR_ADAPTIVE", OutputFormat::QIRAdaptive,
             "QIR for the Adaptive Profile.")
      .value("QISKIT", OutputFormat::Qiskit, "A Qiskit QuantumCircuit.");

  nb::enum_<mlir::PayloadEncoding>(m, "PayloadEncoding",
                                   "Payload representation encoding.")
      .value("TEXT", mlir::PayloadEncoding::Text)
      .value("BINARY", mlir::PayloadEncoding::Binary);

  nb::class_<mlir::PayloadFormat>(m, "PayloadFormat", "Exact payload identity.")
      .def(nb::init<std::string, std::string, std::string,
                    mlir::PayloadEncoding>(),
           "format_id"_a, "version"_a, "profile"_a = "",
           "encoding"_a = mlir::PayloadEncoding::Text)
      .def_rw("format_id", &mlir::PayloadFormat::id)
      .def_rw("version", &mlir::PayloadFormat::version)
      .def_rw("profile", &mlir::PayloadFormat::profile)
      .def_rw("encoding", &mlir::PayloadFormat::encoding);

  auto programConstraint =
      nb::class_<mlir::ProgramConstraint>(m, "ProgramConstraint",
                                          "One payload capability constraint.")
          .def(nb::init<std::string, uint64_t>(), "constraint_id"_a, "value"_a)
          .def_rw("constraint_id", &mlir::ProgramConstraint::id)
          .def_rw("value", &mlir::ProgramConstraint::value);
  programConstraint.attr("MAX_NESTING_DEPTH") =
      mlir::ProgramConstraint::MAX_NESTING_DEPTH.str();
  programConstraint.attr("MAX_ITERATION_COUNT") =
      mlir::ProgramConstraint::MAX_ITERATION_COUNT.str();
  programConstraint.attr("MAX_CASE_COUNT") =
      mlir::ProgramConstraint::MAX_CASE_COUNT.str();

  auto programCapability =
      nb::class_<mlir::ProgramCapability>(m, "ProgramCapability",
                                          "One payload execution capability.")
          .def(nb::init<std::string, uint64_t,
                        std::vector<mlir::ProgramConstraint>>(),
               "capability_id"_a, "value"_a = 0,
               "constraints"_a = std::vector<mlir::ProgramConstraint>{})
          .def_rw("capability_id", &mlir::ProgramCapability::id)
          .def_rw("value", &mlir::ProgramCapability::value)
          .def_rw("constraints", &mlir::ProgramCapability::constraints);
  programCapability.attr("FORWARD_BRANCHING") =
      mlir::ProgramCapability::FORWARD_BRANCHING.str();
  programCapability.attr("COUNTED_ITERATION") =
      mlir::ProgramCapability::COUNTED_ITERATION.str();
  programCapability.attr("CONDITIONAL_LOOP") =
      mlir::ProgramCapability::CONDITIONAL_LOOP.str();
  programCapability.attr("MULTIWAY_BRANCHING") =
      mlir::ProgramCapability::MULTIWAY_BRANCHING.str();

  nb::class_<mlir::PayloadSpecification>(m, "PayloadSpecification",
                                         "Selected payload execution contract.")
      .def(
          "__init__",
          [](mlir::PayloadSpecification& self, mlir::PayloadFormat format,
             std::vector<mlir::ProgramCapability> capabilities,
             const bool optionalCapabilitiesKnown) {
            constructFromExpected(self, mlir::PayloadSpecification::create(
                                            std::move(format),
                                            std::move(capabilities),
                                            optionalCapabilitiesKnown));
          },
          "payload_format"_a,
          "capabilities"_a = std::vector<mlir::ProgramCapability>{},
          "optional_capabilities_known"_a = false)
      .def_prop_ro(
          "format",
          [](const mlir::PayloadSpecification& environment) {
            return environment.format();
          },
          "The exact selected payload format.")
      .def_prop_ro(
          "capabilities",
          [](const mlir::PayloadSpecification& environment) {
            return toPythonList(environment.capabilities());
          },
          "The effective payload capabilities.")
      .def_prop_ro("optional_capabilities_known",
                   &mlir::PayloadSpecification::optionalCapabilitiesKnown,
                   "Whether optional capability metadata is complete.");

  auto compilerTarget = nb::class_<mlir::CompilerTarget>(
      m, "CompilerTarget", R"pb(Immutable MQT compiler target.

Every target has either all-to-all or explicitly enumerated connectivity and
either unrestricted or explicitly enumerated native-operation support.)pb");

  auto durationUnit = nb::class_<mlir::CompilerTarget::DurationUnit>(
      compilerTarget, "DurationUnit", "Unit for raw target timing metadata.");
  durationUnit
      .def(
          "__init__",
          [](mlir::CompilerTarget::DurationUnit& self, std::string unit,
             const double scaleFactor) {
            constructFromExpected(self,
                                  mlir::CompilerTarget::DurationUnit::create(
                                      std::move(unit), scaleFactor));
          },
          "unit"_a, "scale_factor"_a)
      .def_prop_ro(
          "unit",
          [](const mlir::CompilerTarget::DurationUnit& value) {
            return std::string_view(value.unit());
          },
          "The reported duration unit.")
      .def_prop_ro("scale_factor",
                   &mlir::CompilerTarget::DurationUnit::scaleFactor,
                   "The multiplier applied to raw timing values.");

  auto targetSite = nb::class_<mlir::CompilerTarget::Site>(
      compilerTarget, "Site", nb::pooled(),
      "A hardware site and its optional metadata.");
  targetSite
      .def(
          "__init__",
          [](mlir::CompilerTarget::Site& self,
             const mlir::CompilerTarget::SiteId siteId,
             std::optional<std::string> name, const std::optional<uint64_t> t1,
             const std::optional<uint64_t> t2) {
            constructFromExpected(self, mlir::CompilerTarget::Site::create(
                                            siteId, std::move(name), t1, t2));
          },
          "site_id"_a, "name"_a = nb::none(), "t1"_a = nb::none(),
          "t2"_a = nb::none())
      .def_prop_ro("id", &mlir::CompilerTarget::Site::id,
                   "The target-defined nonnegative site identifier.")
      .def_prop_ro(
          "name",
          [](const mlir::CompilerTarget::Site& site) {
            const auto name = site.name();
            return name ? std::optional<std::string_view>(*name) : std::nullopt;
          },
          "The reported site name, if available.")
      .def_prop_ro("t1", &mlir::CompilerTarget::Site::t1,
                   "The raw T1 coherence time, if available.")
      .def_prop_ro("t2", &mlir::CompilerTarget::Site::t2,
                   "The raw T2 coherence time, if available.");

  auto siteTuple = nb::class_<mlir::CompilerTarget::SiteTuple>(
      compilerTarget, "SiteTuple",
      "A supported ordered placement with optional calibration.");
  siteTuple
      .def(
          "__init__",
          [](mlir::CompilerTarget::SiteTuple& self,
             std::vector<mlir::CompilerTarget::SiteId> sites,
             const std::optional<uint64_t> duration,
             const std::optional<double> fidelity) {
            constructFromExpected(self,
                                  mlir::CompilerTarget::SiteTuple::create(
                                      std::move(sites), duration, fidelity));
          },
          "sites"_a, "duration"_a = nb::none(), "fidelity"_a = nb::none())
      .def_prop_ro(
          "sites",
          [](const mlir::CompilerTarget::SiteTuple& tuple) {
            return toPythonList(tuple.sites());
          },
          "The ordered target site identifiers.")
      .def_prop_ro("duration", &mlir::CompilerTarget::SiteTuple::duration,
                   "The raw operation duration, if available.")
      .def_prop_ro("fidelity", &mlir::CompilerTarget::SiteTuple::fidelity,
                   "The operation fidelity, if available.");

  nb::implicitly_convertible<std::vector<mlir::CompilerTarget::SiteId>,
                             mlir::CompilerTarget::SiteTuple>();

  nb::enum_<mlir::CompilerTarget::OperationCapability::Arity::Kind>(
      compilerTarget, "OperationArityKind",
      "How an operation capability accepts qubit widths.")
      .value("FIXED",
             mlir::CompilerTarget::OperationCapability::Arity::Kind::Fixed)
      .value("VARIADIC",
             mlir::CompilerTarget::OperationCapability::Arity::Kind::Variadic);

  auto operationArity =
      nb::class_<mlir::CompilerTarget::OperationCapability::Arity>(
          compilerTarget, "OperationArity", "Accepted operation qubit widths.");
  operationArity
      .def_static("fixed",
                  &mlir::CompilerTarget::OperationCapability::Arity::fixed,
                  "value"_a, "Create an exact operation arity.")
      .def_static("variadic",
                  &mlir::CompilerTarget::OperationCapability::Arity::variadic,
                  "minimum"_a,
                  R"pb(Create an operation arity with an inclusive minimum.

Capability construction requires a positive minimum.)pb")
      .def_prop_ro("kind",
                   &mlir::CompilerTarget::OperationCapability::Arity::kind,
                   "The arity kind.")
      .def_prop_ro("value",
                   &mlir::CompilerTarget::OperationCapability::Arity::value,
                   "The exact arity or inclusive variadic minimum.")
      .def("accepts",
           &mlir::CompilerTarget::OperationCapability::Arity::accepts,
           "width"_a, "Whether this arity accepts a concrete width.");

  auto targetOperation = nb::class_<mlir::CompilerTarget::OperationCapability>(
      compilerTarget, "OperationCapability", nb::pooled(),
      "A target operation capability, calibration, and ordered "
      "applicability.");
  targetOperation
      .def(
          "__init__",
          [](mlir::CompilerTarget::OperationCapability& self, std::string name,
             const mlir::CompilerTarget::OperationCapability::Arity arity,
             const size_t numParameters,
             std::optional<std::vector<mlir::CompilerTarget::SiteTuple>>
                 siteTuples,
             const std::optional<uint64_t> duration,
             const std::optional<double> fidelity,
             std::vector<std::optional<double>> fixedParameters,
             std::optional<std::string> canonicalName,
             std::vector<std::optional<
                 mlir::CompilerTarget::OperationCapability::ParameterBounds>>
                 parameterBounds) {
            constructFromExpected(
                self,
                mlir::CompilerTarget::OperationCapability::create(
                    std::move(name), arity, numParameters,
                    std::move(siteTuples)
                        .value_or(
                            std::vector<mlir::CompilerTarget::SiteTuple>{}),
                    duration, fidelity, std::move(fixedParameters),
                    std::move(canonicalName), std::move(parameterBounds)));
          },
          "name"_a, "arity"_a, "num_parameters"_a, "site_tuples"_a = nb::none(),
          "duration"_a = nb::none(), "fidelity"_a = nb::none(), nb::kw_only(),
          "fixed_parameters"_a = std::vector<std::optional<double>>{},
          "canonical_name"_a = nb::none(),
          "parameter_bounds"_a = std::vector<std::optional<
              mlir::CompilerTarget::OperationCapability::ParameterBounds>>{})
      .def(
          "__init__",
          [](mlir::CompilerTarget::OperationCapability& self, std::string name,
             const size_t arity, const size_t numParameters,
             std::optional<std::vector<mlir::CompilerTarget::SiteTuple>>
                 siteTuples,
             const std::optional<uint64_t> duration,
             const std::optional<double> fidelity,
             std::vector<std::optional<double>> fixedParameters,
             std::optional<std::string> canonicalName,
             std::vector<std::optional<
                 mlir::CompilerTarget::OperationCapability::ParameterBounds>>
                 parameterBounds) {
            constructFromExpected(
                self,
                mlir::CompilerTarget::OperationCapability::create(
                    std::move(name), arity, numParameters,
                    std::move(siteTuples)
                        .value_or(
                            std::vector<mlir::CompilerTarget::SiteTuple>{}),
                    duration, fidelity, std::move(fixedParameters),
                    std::move(canonicalName), std::move(parameterBounds)));
          },
          "name"_a, "arity"_a, "num_parameters"_a, "site_tuples"_a = nb::none(),
          "duration"_a = nb::none(), "fidelity"_a = nb::none(), nb::kw_only(),
          "fixed_parameters"_a = std::vector<std::optional<double>>{},
          "canonical_name"_a = nb::none(),
          "parameter_bounds"_a = std::vector<std::optional<
              mlir::CompilerTarget::OperationCapability::ParameterBounds>>{})
      .def_prop_ro(
          "name",
          [](const mlir::CompilerTarget::OperationCapability& operation) {
            return std::string_view(operation.name());
          },
          "The exact reported operation name.")
      .def_prop_ro(
          "canonical_name",
          [](const mlir::CompilerTarget::OperationCapability& operation) {
            return std::string_view(operation.canonicalName());
          },
          "The normalized compiler operation name.")
      .def_prop_ro("arity", &mlir::CompilerTarget::OperationCapability::arity,
                   "The accepted operation arity.")
      .def_prop_ro("num_parameters",
                   &mlir::CompilerTarget::OperationCapability::numParameters,
                   "The number of real-valued parameters.")
      .def_prop_ro(
          "site_tuples",
          [](const mlir::CompilerTarget::OperationCapability& operation) {
            return toPythonList(operation.siteTuples());
          },
          "Supported ordered placements with optional calibration; empty means "
          "general applicability.")
      .def_prop_ro(
          "parameter_bounds",
          [](const mlir::CompilerTarget::OperationCapability& operation) {
            return toPythonList(operation.parameterBounds());
          },
          "Inclusive parameter intervals; None leaves a parameter unbounded.")
      .def_prop_ro(
          "fixed_parameters",
          [](const mlir::CompilerTarget::OperationCapability& operation) {
            return toPythonList(operation.fixedParameters());
          },
          R"pb(Fixed values or None per parameter; empty means unrestricted.

Constants use absolute tolerance 1e-15 without angle wrapping.)pb")
      .def_prop_ro("duration",
                   &mlir::CompilerTarget::OperationCapability::duration,
                   "The raw default duration, if available.")
      .def_prop_ro("fidelity",
                   &mlir::CompilerTarget::OperationCapability::fidelity,
                   "The default fidelity, if available.");

  nb::enum_<mlir::CompilerTarget::GateKind>(
      compilerTarget, "GateKind", "Recognized native gate capability.")
      .value("U", mlir::CompilerTarget::GateKind::U)
      .value("X", mlir::CompilerTarget::GateKind::X)
      .value("SX", mlir::CompilerTarget::GateKind::SX)
      .value("RZ", mlir::CompilerTarget::GateKind::RZ)
      .value("RX", mlir::CompilerTarget::GateKind::RX)
      .value("RY", mlir::CompilerTarget::GateKind::RY)
      .value("R", mlir::CompilerTarget::GateKind::R)
      .value("RXX", mlir::CompilerTarget::GateKind::RXX)
      .value("RYY", mlir::CompilerTarget::GateKind::RYY)
      .value("RZX", mlir::CompilerTarget::GateKind::RZX)
      .value("RZZ", mlir::CompilerTarget::GateKind::RZZ)
      .value("ISWAP", mlir::CompilerTarget::GateKind::ISWAP)
      .value("CZ", mlir::CompilerTarget::GateKind::CZ)
      .value("CX", mlir::CompilerTarget::GateKind::CX)
      .value("ECR", mlir::CompilerTarget::GateKind::ECR)
      .value("SQRTISWAP", mlir::CompilerTarget::GateKind::SQRTISWAP);

  nb::enum_<mlir::CompilerTarget::SingleQubitBasis>(
      compilerTarget, "SingleQubitBasis",
      "Recognized target-wide single-qubit synthesis basis.")
      .value("U", mlir::CompilerTarget::SingleQubitBasis::U)
      .value("ZSXX", mlir::CompilerTarget::SingleQubitBasis::ZSXX)
      .value("R", mlir::CompilerTarget::SingleQubitBasis::R)
      .value("GPI", mlir::CompilerTarget::SingleQubitBasis::GPI)
      .value("XZX", mlir::CompilerTarget::SingleQubitBasis::XZX)
      .value("XYX", mlir::CompilerTarget::SingleQubitBasis::XYX)
      .value("ZYZ", mlir::CompilerTarget::SingleQubitBasis::ZYZ)
      .value("ZXZ", mlir::CompilerTarget::SingleQubitBasis::ZXZ);

  nb::enum_<mlir::CompilerTarget::AngleSupport>(
      compilerTarget, "AngleSupport",
      "Angle domain used by native entangler synthesis.")
      .value("FIXED", mlir::CompilerTarget::AngleSupport::Fixed)
      .value("UNRESTRICTED", mlir::CompilerTarget::AngleSupport::Unrestricted)
      .value("ZERO_TO_HALF_PI",
             mlir::CompilerTarget::AngleSupport::ZeroToHalfPi);

  nb::class_<mlir::CompilerTarget::Entangler>(
      compilerTarget, "Entangler",
      "A native synthesis entangler and its angle support.")
      .def_ro("gate", &mlir::CompilerTarget::Entangler::gate,
              "The native gate kind.")
      .def_prop_ro("parameterized",
                   &mlir::CompilerTarget::Entangler::parameterized,
                   "Whether synthesis can vary the entangler angle.")
      .def_ro("angles", &mlir::CompilerTarget::Entangler::angles,
              "The angle domain used by synthesis.");

  auto synthesisBasis = nb::class_<mlir::CompilerTarget::SynthesisBasis>(
      compilerTarget, "SynthesisBasis",
      "One synthesis basis usable across the complete target.");
  synthesisBasis
      .def_ro("single_qubit",
              &mlir::CompilerTarget::SynthesisBasis::singleQubit,
              "The single-qubit synthesis basis.")
      .def_ro("entangler", &mlir::CompilerTarget::SynthesisBasis::entangler,
              "The two-qubit entangler, or None when none is usable.");

  nb::enum_<mlir::CompilerTarget::Connectivity::Kind>(
      compilerTarget, "ConnectivityKind", "The target connectivity model.")
      .value("ALL_TO_ALL", mlir::CompilerTarget::Connectivity::Kind::AllToAll)
      .value("EXPLICIT", mlir::CompilerTarget::Connectivity::Kind::Explicit);

  auto connectivity = nb::class_<mlir::CompilerTarget::Connectivity>(
      compilerTarget, "Connectivity", "A target connectivity model.");
  connectivity
      .def(
          "__init__",
          [](mlir::CompilerTarget::Connectivity& self,
             const std::vector<mlir::CompilerTarget::Coupling>& couplings) {
            new (&self) mlir::CompilerTarget::Connectivity(
                mlir::CompilerTarget::Connectivity::fromCouplings(couplings));
          },
          "couplings"_a, "Create an explicit connectivity model.")
      .def_static("all_to_all", &mlir::CompilerTarget::Connectivity::allToAll,
                  "Create an all-to-all connectivity model.")
      .def_prop_ro("kind", &mlir::CompilerTarget::Connectivity::kind,
                   "The connectivity model.")
      .def_prop_ro(
          "couplings",
          [](const mlir::CompilerTarget::Connectivity& value) {
            return toPythonList(value.couplings());
          },
          "The explicit couplings, if present.");

  nb::enum_<mlir::CompilerTarget::NativeOperations::Kind>(
      compilerTarget, "NativeOperationsKind",
      "The native-operation support model.")
      .value("UNRESTRICTED",
             mlir::CompilerTarget::NativeOperations::Kind::Unrestricted)
      .value("EXPLICIT",
             mlir::CompilerTarget::NativeOperations::Kind::Explicit);

  auto nativeOperations = nb::class_<mlir::CompilerTarget::NativeOperations>(
      compilerTarget, "NativeOperations", "Native-operation support.");
  nativeOperations
      .def(
          "__init__",
          [](mlir::CompilerTarget::NativeOperations& self,
             const std::vector<mlir::CompilerTarget::OperationCapability>&
                 operations) {
            new (&self) mlir::CompilerTarget::NativeOperations(
                mlir::CompilerTarget::NativeOperations::fromOperations(
                    operations));
          },
          "operations"_a, "Create explicit native-operation support.")
      .def_static("from_qiskit", &bindings::qiskit::importNativeOperations,
                  "source"_a, nb::kw_only(), "operation_names"_a = nb::none(),
                  nb::sig("def from_qiskit(source: qiskit.transpiler.Target | "
                          "qiskit.providers.BackendV2, *, operation_names: "
                          "collections.abc.Iterable[str] | None = None) -> "
                          "mqt.core.mlir.CompilerTarget.NativeOperations"),
                  "Import gate capabilities and parameter constraints, "
                  "ignoring physical placement.\n\n"
                  "Unsupported explicit selections raise ValueError; otherwise "
                  "they warn and are omitted.")
      .def_static("unrestricted",
                  &mlir::CompilerTarget::NativeOperations::unrestricted,
                  "Create unrestricted native-operation support.")
      .def_prop_ro("kind", &mlir::CompilerTarget::NativeOperations::kind,
                   "The native-operation support model.")
      .def_prop_ro(
          "operations",
          [](const mlir::CompilerTarget::NativeOperations& value) {
            return toPythonList(value.operations());
          },
          "The explicit operations, if present.");

  compilerTarget
      .def(
          "__init__",
          [](mlir::CompilerTarget& self, const size_t numSites,
             mlir::CompilerTarget::Connectivity connectivity,
             mlir::CompilerTarget::NativeOperations nativeOperations,
             std::optional<mlir::CompilerTarget::DurationUnit> durationUnit) {
            constructFromExpected(self, mlir::CompilerTarget::create(
                                            numSites, std::move(connectivity),
                                            std::move(nativeOperations),
                                            std::move(durationUnit)));
          },
          "num_sites"_a, nb::kw_only(), "connectivity"_a, "native_operations"_a,
          "duration_unit"_a = nb::none())
      .def(
          "__init__",
          [](mlir::CompilerTarget& self, std::string name,
             const size_t numSites,
             mlir::CompilerTarget::Connectivity connectivity,
             mlir::CompilerTarget::NativeOperations nativeOperations,
             std::optional<mlir::CompilerTarget::DurationUnit> durationUnit) {
            constructFromExpected(
                self, mlir::CompilerTarget::create(std::move(name), numSites,
                                                   std::move(connectivity),
                                                   std::move(nativeOperations),
                                                   std::move(durationUnit)));
          },
          "name"_a, "num_sites"_a, nb::kw_only(), "connectivity"_a,
          "native_operations"_a, "duration_unit"_a = nb::none())
      .def(
          "__init__",
          [](mlir::CompilerTarget& self,
             std::vector<mlir::CompilerTarget::Site> sites,
             mlir::CompilerTarget::Connectivity connectivity,
             mlir::CompilerTarget::NativeOperations nativeOperations,
             std::optional<mlir::CompilerTarget::DurationUnit> durationUnit) {
            constructFromExpected(
                self, mlir::CompilerTarget::create(std::move(sites),
                                                   std::move(connectivity),
                                                   std::move(nativeOperations),
                                                   std::move(durationUnit)));
          },
          "sites"_a, nb::kw_only(), "connectivity"_a, "native_operations"_a,
          "duration_unit"_a = nb::none())
      .def(
          "__init__",
          [](mlir::CompilerTarget& self, std::string name,
             std::vector<mlir::CompilerTarget::Site> sites,
             mlir::CompilerTarget::Connectivity connectivity,
             mlir::CompilerTarget::NativeOperations nativeOperations,
             std::optional<mlir::CompilerTarget::DurationUnit> durationUnit) {
            constructFromExpected(self, mlir::CompilerTarget::create(
                                            std::move(name), std::move(sites),
                                            std::move(connectivity),
                                            std::move(nativeOperations),
                                            std::move(durationUnit)));
          },
          "name"_a, "sites"_a, nb::kw_only(), "connectivity"_a,
          "native_operations"_a, "duration_unit"_a = nb::none())
      .def(
          "__str__",
          [](const mlir::CompilerTarget& target) {
            auto context = mlir::createCompilerContext();
            std::string result;
            llvm::raw_string_ostream stream(result);
            stream << target.materialize(*context);
            return result;
          },
          "Serialize the target as a typed MLIR attribute.")
      .def_static(
          "from_device",
          [](const qdmi::Device& device) {
            auto target = [&device] {
              const nb::gil_scoped_release release;
              return mlir::compilerTargetFromDevice(device);
            }();
            return takeResult(std::move(target));
          },
          "device"_a, "Snapshot a circuit-model QDMI device.")
      .def_static("from_qiskit", &bindings::qiskit::importTarget, "source"_a,
                  nb::kw_only(), "operation_names"_a = nb::none(),
                  "name"_a = nb::none(),
                  nb::sig("def from_qiskit(source: qiskit.transpiler.Target | "
                          "qiskit.providers.BackendV2, *, operation_names: "
                          "collections.abc.Iterable[str] | None = None, "
                          "name: str | None = None) -> CompilerTarget"),
                  R"pb(Snapshot native operations and connectivity from Qiskit.

Args:
    source: Qiskit Target or BackendV2. Physical import requires a known positive qubit count.
    operation_names: Qiskit Target operation names to retain. By default,
        include every representable operation. Explicit selections must all be
        representable.
    name: Override the target name. By default, use the backend name when
        source is a BackendV2; a Target produces an unnamed snapshot.

Returns:
    An independent compiler target. Unrepresentable gates are omitted with
    warnings when operation_names is not set. Calibration and scheduling data
    are not included.

Raises:
    TypeError: If source is neither a Target nor a BackendV2.
    ValueError: If the selected operations or connectivity cannot be represented.)pb")
      .def_static(
          "from_device_id",
          [](const std::string& deviceId,
             std::optional<std::filesystem::path> driverPath,
             std::optional<std::string> token,
             std::optional<std::filesystem::path> authFile,
             std::optional<std::string> authUrl,
             std::optional<std::string> username,
             std::optional<std::string> password,
             std::optional<std::string> projectId,
             std::optional<std::string> custom1,
             std::optional<std::string> custom2,
             std::optional<std::string> custom3,
             std::optional<std::string> custom4,
             std::optional<std::string> custom5) {
            auto target = [&] {
              const nb::gil_scoped_release release;
              auto device = openQDMIDevice(
                  deviceId, std::move(driverPath), std::move(token),
                  std::move(authFile), std::move(authUrl), std::move(username),
                  std::move(password), std::move(projectId), std::move(custom1),
                  std::move(custom2), std::move(custom3), std::move(custom4),
                  std::move(custom5));
              return mlir::compilerTargetFromDevice(device);
            }();
            return takeResult(std::move(target));
          },
          "device_id"_a, nb::kw_only(), "driver_path"_a = std::nullopt,
          "token"_a = std::nullopt, "auth_file"_a = std::nullopt,
          "auth_url"_a = std::nullopt, "username"_a = std::nullopt,
          "password"_a = std::nullopt, "project_id"_a = std::nullopt,
          "custom1"_a = std::nullopt, "custom2"_a = std::nullopt,
          "custom3"_a = std::nullopt, "custom4"_a = std::nullopt,
          "custom5"_a = std::nullopt,
          "Open a client-visible device and snapshot its compiler target.")
      .def_prop_ro(
          "name",
          [](const mlir::CompilerTarget& target) {
            const auto name = target.name();
            return name ? std::optional<std::string_view>(*name) : std::nullopt;
          },
          "The target name, if available.")
      .def_prop_ro("duration_unit", &mlir::CompilerTarget::durationUnit,
                   "The target timing unit, if available.")
      .def_prop_ro("num_sites", &mlir::CompilerTarget::numSites,
                   "The number of target sites.")
      .def_prop_ro(
          "sites",
          [](const mlir::CompilerTarget& target) {
            return toPythonList(target.sites());
          },
          "Detailed sites in compiler-vertex order.")
      .def_prop_ro("connectivity_kind", &mlir::CompilerTarget::connectivityKind,
                   "The target connectivity model.")
      .def_prop_ro(
          "couplings",
          [](const mlir::CompilerTarget& target) {
            return toPythonList(target.couplings());
          },
          "Canonical undirected couplings in target site IDs.")
      .def_prop_ro("native_operations_kind",
                   &mlir::CompilerTarget::nativeOperationsKind,
                   "The target native-operation support model.")
      .def_prop_ro(
          "operations",
          [](const mlir::CompilerTarget& target) {
            return toPythonList(target.operations());
          },
          "Operation capabilities in reported order.")
      .def_prop_ro(
          "supported_gates",
          [](const mlir::CompilerTarget& target) {
            return toPythonList(target.supportedGates());
          },
          "Recognized native gates supported by the target.")
      .def_prop_ro("synthesis_basis", &mlir::CompilerTarget::synthesisBasis,
                   nb::rv_policy::copy,
                   "A target-wide single-qubit basis with an optional "
                   "entangler, or None when no single-qubit basis is usable.")
      .def(
          "supports_operation",
          [](const mlir::CompilerTarget& target, const std::string_view name,
             const size_t arity, const std::optional<size_t> numParameters,
             const std::optional<std::vector<mlir::CompilerTarget::SiteId>>&
                 sites,
             const std::vector<std::optional<double>>& parameters) {
            if (sites) {
              return target.supportsOperation(name, arity, numParameters,
                                              *sites, parameters);
            }
            return target.supportsOperation(name, arity, numParameters,
                                            std::nullopt, parameters);
          },
          "name"_a, "arity"_a, "num_parameters"_a = nb::none(),
          "sites"_a = nb::none(), nb::kw_only(),
          "parameters"_a.sig("()") = std::vector<std::optional<double>>{},
          R"pb(Check whether the target supports an operation.

Args:
    name: Operation name. Recognized aliases are normalized.
    arity: Number of qubits used by the operation.
    num_parameters: Number of real-valued parameters. None accepts any count.
    sites: Ordered target site IDs. None checks support on any placement.
    parameters: Known parameter values. Omitted or None values require
        unrestricted support.)pb");

  nb::class_<mlir::TargetEnvironment>(
      m, "TargetEnvironment",
      "A compiler target and its selected payload specification.")
      .def(nb::init<mlir::CompilerTarget, mlir::PayloadSpecification>(),
           "target"_a, "payload_specification"_a)
      .def_prop_ro("target", &mlir::TargetEnvironment::target,
                   "The compiler target.")
      .def_prop_ro("payload_specification",
                   &mlir::TargetEnvironment::payloadSpecification,
                   "The selected payload specification.");

  auto program = nb::class_<mlir::Program>(
      m, "Program", R"pb(Base class for a typed MLIR compiler program.

Programs own their MLIR module. Conversions can consume a program; use
``is_valid`` to check whether it can still be used.)pb");
  program
      .def_prop_ro("is_valid", &mlir::Program::isValid,
                   "Whether this program still owns its module.")
      .def(
          "operation_counts",
          [](const mlir::Program& value) {
            requireValid(value);
            return value.operationCounts();
          },
          R"pb(Count every operation by its full MLIR name.

Includes the root module, helper functions, modifier bodies, terminators,
and nested modules.)pb")
      .def_prop_ro(
          "ir",
          [](const mlir::Program& value) {
            requireValid(value);
            return value.str();
          },
          "The textual MLIR representation of this program.")
      .def(
          "__str__",
          [](const mlir::Program& value) {
            requireValid(value);
            return value.str();
          },
          "Return the textual MLIR representation of this program.");

  nb::class_<mlir::MappingOptions>(m, "MappingOptions",
                                   "Native mapping controls.")
      .def(nb::init<std::optional<size_t>, size_t, size_t, size_t>(),
           nb::kw_only(), "trials"_a = nb::none(),
           "iterations"_a = mlir::MappingOptions{}.iterations,
           "lookahead"_a = mlir::MappingOptions{}.lookahead,
           "search_memory_limit"_a = mlir::MappingOptions{}.searchMemoryLimit)
      .def_rw(
          "trials", &mlir::MappingOptions::trials,
          "Positive trial count; None uses the available logical CPU count.")
      .def_rw("iterations", &mlir::MappingOptions::iterations,
              "Forward/backward refinement rounds; zero scores each start "
              "directly.")
      .def_rw("lookahead", &mlir::MappingOptions::lookahead,
              "Additional two-qubit gates considered during routing; zero "
              "disables lookahead.")
      .def_rw(
          "search_memory_limit", &mlir::MappingOptions::searchMemoryLimit,
          R"pb(Estimated node and layout bytes per routing search, per concurrent trial.

Zero disables node expansion. Container overhead, caches, and IR are extra.)pb");

  nb::class_<mlir::CompilationOptions>(m, "CompilationOptions",
                                       R"pb(Shared compiler controls.

An explicit seed overrides all compiler randomness; None preserves pass defaults and custom pipeline seeds.)pb")
      .def(
          nb::init<std::optional<uint64_t>, bool, bool, mlir::MappingOptions>(),
          nb::kw_only(), "seed"_a = nb::none(), "enable_timing"_a = false,
          "enable_statistics"_a = false, "mapping"_a = mlir::MappingOptions{})
      .def_rw("seed", &mlir::CompilationOptions::seed)
      .def_rw("enable_timing", &mlir::CompilationOptions::enableTiming)
      .def_rw("enable_statistics", &mlir::CompilationOptions::enableStatistics)
      .def_rw("mapping", &mlir::CompilationOptions::mapping);

  nb::class_<mlir::QuantumProgramInfo>(
      m, "QuantumProgramInfo", R"pb(Quantum resources and static IR statistics.

Use :meth:`QCProgram.inspect` or :meth:`QCOProgram.inspect` to collect a snapshot.)pb")
      .def_ro("num_qubits", &mlir::QuantumProgramInfo::numQubits,
              R"pb(Declared quantum capacity.

Counts allocated qubits or distinct static site IDs. ``None`` means the width
is unknown. Resource inspection includes helper functions and excludes nested
modules; it describes declared capacity rather than peak live width.)pb")
      .def_ro("static_qubits", &mlir::QuantumProgramInfo::staticQubits,
              R"pb(Sorted distinct physical site IDs.

Includes declarations in helper functions and excludes nested modules.)pb")
      .def_ro("has_control_flow", &mlir::QuantumProgramInfo::hasControlFlow,
              R"pb(Whether the module contains control flow.

Includes branches and region-control operations in helper functions, but
excludes nested modules.)pb")
      .def_ro(
          "num_gates", &mlir::QuantumProgramInfo::numGates,
          "Static entry-point gate count.\n\nSee :meth:`QCProgram.num_gates`.")
      .def_ro("num_single_qubit_gates",
              &mlir::QuantumProgramInfo::numSingleQubitGates,
              "Static single-qubit gate count.\n\n"
              "See :meth:`QCProgram.num_single_qubit_gates`.")
      .def_ro("num_two_qubit_gates",
              &mlir::QuantumProgramInfo::numTwoQubitGates,
              "Static two-qubit gate count.\n\n"
              "See :meth:`QCProgram.num_two_qubit_gates`.")
      .def_ro(
          "gate_counts", &mlir::QuantumProgramInfo::gateCounts,
          "Entry-point gate histogram.\n\nSee :meth:`QCProgram.gate_counts`.")
      .def_ro("control_flow_counts",
              &mlir::QuantumProgramInfo::controlFlowCounts,
              "Entry-point control-flow histogram.\n\n"
              "See :meth:`QCProgram.control_flow_counts`.")
      .def_ro("operation_counts", &mlir::QuantumProgramInfo::operationCounts,
              "Full module operation histogram.\n\n"
              "See :meth:`Program.operation_counts`.");

  auto qcProgram = nb::class_<mlir::QCProgram, mlir::Program>(
      m, "QCProgram", nb::pooled(), R"pb(A compiler program in the QC dialect.

QC programs use reference semantics and represent frontend quantum programs
before conversion to QCO.)pb");
  qcProgram
      .def_static(
          "from_mlir_str",
          &FallibleFunctionAdapter<&mlir::QCProgram::fromMLIRString>::call,
          "source"_a, "Parse a QC MLIR source string.")
      .def_static(
          "from_mlir_file",
          &FallibleFunctionAdapter<&mlir::QCProgram::fromMLIRFile>::call,
          "path"_a, "Parse QC MLIR from a file.")
      .def_static(
          "from_openqasm_str",
          &FallibleFunctionAdapter<&mlir::QCProgram::fromOpenQASMString>::call,
          "source"_a,
          R"pb(Translate supported OpenQASM to QC MLIR.

Accepts versionless input and versions 2.0, 3.0, and 3.1.)pb")
      .def_static(
          "from_openqasm_file",
          &FallibleFunctionAdapter<&mlir::QCProgram::fromOpenQASMFile>::call,
          "path"_a,
          R"pb(Translate a supported OpenQASM file to QC MLIR.

Accepts versionless input and versions 2.0, 3.0, and 3.1.)pb")
      .def_static(
          "from_qiskit", &bindings::qiskit::importCircuit, "circuit"_a,
          nb::sig("def from_qiskit(circuit: qiskit.circuit.QuantumCircuit) "
                  "-> QCProgram"),
          R"pb(Translate a Qiskit {py:class}`~qiskit.circuit.QuantumCircuit` to QC MLIR.

Args:
    circuit: Circuit to import. A complete transpiler layout is retained as
        metadata.)pb")
      .def("copy", &copyProgram<mlir::QCProgram>,
           "Return an independent copy of this program.")
      .def("cleanup", &FallibleMemberAdapter<&mlir::QCProgram::cleanup>::call,
           "Run the standard QC cleanup pipeline in place.")
      .def(
          "normalize_global_phases",
          &FallibleMemberAdapter<&mlir::QCProgram::normalizeGlobalPhases>::call,
          "Normalize scoped global phases in place.")
      .def(
          "to_openqasm3",
          [](mlir::QCProgram& program, const bool copy) {
            auto source = copiedOrConsumed(program, copy);
            const auto context = source.context();
            return withDiagnostics<nb::exception_type::runtime_error>(
                context.get(), "cannot export QC program to OpenQASM 3",
                [&] { return std::move(source).intoOpenQASM3(); });
          },
          nb::kw_only(), "copy"_a = false,
          R"pb(Clean up and emit this QC program as OpenQASM 3 without QCO optimization.

The source is consumed, including when export fails. Set ``copy=True`` to preserve it.

Args:
    copy: Preserve the source program by exporting a copy.)pb")
      .def(
          "to_qiskit",
          [](mlir::QCProgram& program, const mlir::CompilerTarget* target,
             const bool copy) {
            return bindings::qiskit::exportCircuit(
                copiedOrConsumed(program, copy), target);
          },
          nb::kw_only(), "target"_a = nb::none(), "copy"_a = false,
          nb::sig("def to_qiskit(self, *, target: CompilerTarget | None = "
                  "None, copy: bool = False) -> qiskit.circuit.QuantumCircuit"),
          R"pb(Convert this program to a Qiskit circuit.

The source is consumed, including when export fails. Set ``copy=True`` to preserve it.
The exporter restores attached layout metadata when it is compatible with the
selected target.

Args:
    target: Map static site IDs to qubit indices in target site order. All
        qubits must be static sites of the target. Select applicable standard
        gate names without checking device execution support. None applies no
        target site mapping.
    copy: Preserve the source program by exporting a copy.)pb")
      .def(
          "to_qco",
          [](mlir::QCProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoQCO());
          },
          nb::kw_only(), "copy"_a = false,
          R"pb(Convert this program to QCO.

The source is consumed even on failure. Set ``copy=True`` to preserve it.)pb")
      .def(
          "to_qir",
          [](mlir::QCProgram& value, const mlir::QIRProfile profile,
             const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoQIR(profile));
          },
          "profile"_a, nb::kw_only(), "copy"_a = false,
          R"pb(Lower this program to QIR for the requested profile.

Entry results may be direct measurements or CBit registers. Adaptive Profile
also supports computed Boolean (``i1``) results. Other scalar outputs are
unsupported. The generated QIR returns a zero execution status.
Base conversion unrolls static counted loops; Adaptive conversion retains them.
The source is consumed even on failure. Set ``copy=True`` to preserve it.)pb");

  auto qcoProgram = nb::class_<mlir::QCOProgram, mlir::Program>(
      m, "QCOProgram", R"pb(A compiler program in the QCO dialect.

QCO programs use value semantics and expose optimization and transformation
operations.)pb");
  qcoProgram
      .def_static(
          "from_mlir_str",
          &FallibleFunctionAdapter<&mlir::QCOProgram::fromMLIRString>::call,
          "source"_a, "Parse a QCO MLIR source string.")
      .def_static(
          "from_mlir_file",
          &FallibleFunctionAdapter<&mlir::QCOProgram::fromMLIRFile>::call,
          "path"_a, "Parse QCO MLIR from a file.")
      .def("copy", &copyProgram<mlir::QCOProgram>,
           "Return an independent copy of this program.")
      .def("cleanup", &FallibleMemberAdapter<&mlir::QCOProgram::cleanup>::call,
           "Run the standard QCO cleanup pipeline in place.")
      .def("normalize_global_phases",
           &FallibleMemberAdapter<
               &mlir::QCOProgram::normalizeGlobalPhases>::call,
           "Normalize scoped global phases in place.")
      .def(
          "run_pass_pipeline",
          [](mlir::QCOProgram& program, const std::string& pipeline,
             mlir::CompilationOptions options) {
            requireValid(program);
            requireSuccess(program.runPassPipeline(pipeline, options));
          },
          "pipeline"_a, nb::kw_only(), "options"_a = mlir::CompilationOptions{},
          "Run a textual MLIR pass pipeline in place.")
      .def("merge_single_qubit_rotation_gates",
           &FallibleMemberAdapter<
               &mlir::QCOProgram::mergeSingleQubitRotationGates>::call,
           "Merge compatible consecutive single-qubit rotation gates.")
      .def(
          "fuse_single_qubit_unitary_runs",
          &FallibleMemberAdapter<
              &mlir::QCOProgram::fuseSingleQubitUnitaryRuns>::call,
          nb::kw_only(), "basis"_a = "zyz",
          "Fuse single-qubit unitary runs into the chosen decomposition basis.")
      .def("unroll_quantum_loops",
           &FallibleMemberAdapter<&mlir::QCOProgram::unrollQuantumLoops>::call,
           nb::kw_only(), "unroll_factor"_a = -1,
           "Unroll quantum loops, optionally using a maximum unroll factor.")
      .def("lift_hadamards",
           &FallibleMemberAdapter<&mlir::QCOProgram::liftHadamards>::call,
           "Move Hadamard gates through compatible operations.")
      .def("reuse_qubits",
           &FallibleMemberAdapter<&mlir::QCOProgram::reuseQubits>::call,
           "Reuse independent single-qubit allocations.")
      .def("run_qubit_reuse_pipeline",
           &FallibleMemberAdapter<
               &mlir::QCOProgram::runQubitReusePipeline>::call,
           "Prepare the program for qubit reuse and reuse eligible qubits.")
      .def("decompose_multi_controlled",
           &FallibleMemberAdapter<
               &mlir::QCOProgram::decomposeMultiControlled>::call,
           nb::kw_only(), "min_qubits"_a = 3,
           R"pb(Decompose gates that act on at least min_qubits qubits.

Supports controlled X/Y/Z/SWAP and RX/RY/RZ gates, qco.rccx, and constant-angle phase gates. min_qubits must be
at least 3; default 3 means wider than two-qubit.)pb")
      .def(
          "compile_for_target",
          [](mlir::QCOProgram& program,
             const mlir::TargetEnvironment& environment,
             mlir::CompilationOptions options) {
            requireValid(program);
            withDiagnostics<nb::exception_type::runtime_error>(
                program.module().getContext(), "Target compilation failed",
                [&] {
                  const nb::gil_scoped_release release;
                  return program.compileForTarget(environment, options);
                });
          },
          "target_environment"_a, nb::kw_only(),
          "options"_a = mlir::CompilationOptions{},
          R"pb(Compile for the target and attach layout metadata when possible.

Reject existing layout metadata. Do not rely on program contents if compilation fails. Failures raise
RuntimeError with MLIR diagnostics.)pb")
      .def(
          "synthesize_for_target",
          [](mlir::QCOProgram& program,
             const mlir::TargetEnvironment& environment,
             mlir::CompilationOptions options) {
            requireValid(program);
            withDiagnostics<nb::exception_type::runtime_error>(
                program.module().getContext(), "Target synthesis failed", [&] {
                  return program.synthesizeForTarget(environment, options);
                });
          },
          "target_environment"_a, nb::kw_only(),
          "options"_a = mlir::CompilationOptions{},
          R"pb(Synthesize native operations without routing.

Dynamic qubits require all-to-all connectivity and receive layout metadata when possible. Static qubits keep
their device site IDs and must fit the target topology. Do not rely on the program contents if synthesis fails.
Failures raise RuntimeError with the emitted MLIR diagnostics.)pb")
      .def(
          "to_qiskit",
          [](mlir::QCOProgram& program, const mlir::CompilerTarget* target,
             const bool copy) {
            auto source = copiedOrConsumed(program, copy);
            auto qc = takeResult(std::move(source).intoQC());
            return bindings::qiskit::exportCircuit(std::move(qc), target);
          },
          nb::kw_only(), "target"_a = nb::none(), "copy"_a = false,
          nb::sig("def to_qiskit(self, *, target: CompilerTarget | None = "
                  "None, copy: bool = False) -> qiskit.circuit.QuantumCircuit"),
          R"pb(Convert this program through QC to a Qiskit circuit.

The source is consumed, including when conversion or export fails. Set ``copy=True`` to preserve it.
The exporter restores attached layout metadata when it is compatible with the
selected target.

Args:
    target: Map static site IDs to qubit indices in target site order. All
        qubits must be static sites of the target. Select applicable standard
        gate names without checking device execution support. None applies no
        target site mapping.
    copy: Preserve the source program by exporting a copy.)pb")
      .def(
          "to_qc",
          [](mlir::QCOProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoQC());
          },
          nb::kw_only(), "copy"_a = false,
          R"pb(Convert this program to QC.

The source is consumed even on failure. Set ``copy=True`` to preserve it.)pb")
      .def(
          "to_jeff",
          [](mlir::QCOProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoJeff());
          },
          nb::kw_only(), "copy"_a = false,
          R"pb(Convert this program to ``jeff`` MLIR.

The source is consumed even on failure. Set ``copy=True`` to preserve it.)pb");

  registerInspection(qcProgram);
  registerInspection(qcoProgram);
  registerParameterBinding(qcProgram);
  registerParameterBinding(qcoProgram);

  auto jeffProgram = nb::class_<mlir::JeffProgram, mlir::Program>(
      m, "JeffProgram",
      R"pb(A serializable compiler program in the ``jeff`` dialect.

``jeff`` programs can be stored as bytes or files and converted back to QCO for
further compilation.)pb");
  jeffProgram
      .def_static(
          "from_segments",
          [](const std::vector<JeffSegment>& segments) {
            if (segments.empty()) {
              throw nb::value_error("at least one jeff segment is required");
            }
            std::vector<kj::ArrayPtr<const capnp::word>> views;
            std::vector<kj::Array<capnp::word>> aligned;
            views.reserve(segments.size());
            for (const auto& segment : segments) {
              if (segment.size() % sizeof(capnp::word) != 0U) {
                throw nb::value_error("jeff segment size must be a multiple of "
                                      "the Cap'n Proto word size");
              }
              const auto size = segment.size() / sizeof(capnp::word);
              const auto* data =
                  reinterpret_cast<const capnp::word*>(segment.data());
              if (reinterpret_cast<uintptr_t>(data) % alignof(capnp::word) !=
                  0U) {
                auto words = kj::heapArray<capnp::word>(size);
                std::memcpy(words.begin(), segment.data(), segment.size());
                data = words.begin();
                aligned.push_back(std::move(words));
              }
              views.emplace_back(data, size);
            }
            mlir::FailureOr<mlir::JeffProgram> program;
            auto exception = kj::runCatchingExceptions([&] {
              capnp::SegmentArrayMessageReader reader(
                  kj::arrayPtr(views.data(), views.size()));
              program = mlir::JeffProgram::fromMessage(
                  reader.getRoot<::jeff::Module>());
            });
            KJ_IF_MAYBE (error, exception) {
              throw std::runtime_error(error->getDescription().cStr());
            }
            return takeResult(std::move(program));
          },
          "segments"_a.noconvert(),
          R"pb(Deserialize a ``jeff`` program from Cap'n Proto segments.

Each segment must be a contiguous one-dimensional byte buffer whose size is a
multiple of eight. Keep the buffers unchanged until this call returns. Aligned
buffers are borrowed; unaligned buffers are copied into aligned storage. The
returned program does not retain the buffers.)pb")
      .def_static(
          "from_bytes",
          [](const nb::bytes& bytes) {
            const auto view =
                std::span(reinterpret_cast<const std::byte*>(bytes.c_str()),
                          bytes.size());
            return takeResult(mlir::JeffProgram::fromBytes(view));
          },
          "data"_a, "Deserialize a ``jeff`` program from bytes.")
      .def_static("from_file",
                  &FallibleFunctionAdapter<&mlir::JeffProgram::fromFile>::call,
                  "path"_a, "Read a ``jeff`` program from a file.")
      .def("copy", &copyProgram<mlir::JeffProgram>,
           "Return an independent copy of this program.")
      .def("cleanup", &FallibleMemberAdapter<&mlir::JeffProgram::cleanup>::call,
           "Run the standard ``jeff`` cleanup pipeline in place.")
      .def(
          "to_segment_views",
          [](const mlir::JeffProgram& value) {
            requireValid(value);
            auto message = std::make_unique<capnp::MallocMessageBuilder>();
            value.toMessage(*message);
            const nb::capsule owner(message.get(), [](void* pointer) noexcept {
              delete static_cast<capnp::MallocMessageBuilder*>(pointer);
            });
            auto* builder = message.release();
            std::vector<JeffSegment> result;
            for (auto segment : builder->getSegmentsForOutput()) {
              const auto bytes = segment.asBytes();
              result.push_back(
                  JeffSegment(reinterpret_cast<const uint8_t*>(bytes.begin()),
                              {bytes.size()}, owner));
            }
            return result;
          },
          R"pb(Serialize this program into read-only Cap'n Proto segment views.

The views keep their message storage alive independently of this program. No
segment data is copied or flattened.)pb")
      .def(
          "to_bytes",
          [](const mlir::JeffProgram& value) {
            requireValid(value);
            capnp::MallocMessageBuilder message;
            value.toMessage(message);
            const auto serialized = capnp::messageToFlatArray(message);
            const auto bytes = serialized.asBytes();
            return nb::bytes(bytes.begin(), bytes.size());
          },
          "Serialize this program to its ``jeff`` byte representation.")
      .def("write", &FallibleMemberAdapter<&mlir::JeffProgram::write>::call,
           "path"_a, "Write this program to a ``jeff`` file.")
      .def(
          "to_qco",
          [](mlir::JeffProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoQCO());
          },
          nb::kw_only(), "copy"_a = false,
          R"pb(Convert this program to QCO.

The source is consumed even on failure. Set ``copy=True`` to preserve it.)pb");

  nb::class_<mlir::IQMJSONProgram>(
      m, "IQMJSONProgram", "An immutable output-only IQM circuit object.")
      .def_prop_ro("source", &mlir::IQMJSONProgram::source,
                   "The emitted IQM JSON source.")
      .def("write", &FallibleMemberAdapter<&mlir::IQMJSONProgram::write>::call,
           "path"_a, "Write the IQM JSON source to a file.")
      .def("__str__", &mlir::IQMJSONProgram::str,
           "Return the emitted IQM JSON source.");

  nb::class_<mlir::OpenQASMProgram>(
      m, "OpenQASMProgram",
      "An immutable compiler program containing OpenQASM 3 source.")
      .def_prop_ro("source", &mlir::OpenQASMProgram::source,
                   "The emitted OpenQASM 3 source.")
      .def("write", &FallibleMemberAdapter<&mlir::OpenQASMProgram::write>::call,
           "path"_a, "Write the emitted source to a file.")
      .def("__str__", &mlir::OpenQASMProgram::str,
           "Return the emitted OpenQASM 3 source.");

  auto qirProgram = nb::class_<mlir::QIRProgram, mlir::Program>(
      m, "QIRProgram", R"pb(A compiler program lowered to QIR.

QIR programs retain their target profile and can be emitted as LLVM IR or
LLVM bitcode.)pb");
  qirProgram
      .def("copy", &copyProgram<mlir::QIRProgram>,
           "Return an independent copy of this program.")
      .def("cleanup", &FallibleMemberAdapter<&mlir::QIRProgram::cleanup>::call,
           "Run the standard QIR cleanup pipeline in place.")
      .def_prop_ro("profile", &mlir::QIRProgram::profile,
                   "The QIR target profile used to produce this program.")
      .def(
          "to_llvm_ir",
          [](mlir::QIRProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            const auto context = source.context();
            return withDiagnostics<nb::exception_type::runtime_error>(
                context.get(), "cannot export QIR as LLVM IR",
                [&] { return std::move(source).intoLLVMIR(); });
          },
          nb::kw_only(), "copy"_a = false,
          R"pb(Consume this program and emit LLVM IR text.

The source is consumed even on failure. Set ``copy=True`` to preserve it.

Args:
    copy: Preserve the source program by exporting a copy.)pb")
      .def(
          "to_bitcode",
          [](mlir::QIRProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            const auto context = source.context();
            std::string bytes;
            llvm::raw_string_ostream stream(bytes);
            withDiagnostics<nb::exception_type::runtime_error>(
                context.get(), "cannot export QIR as bitcode",
                [&] { return std::move(source).intoBitcode(stream); });
            return nb::bytes(bytes.data(), bytes.size());
          },
          nb::kw_only(), "copy"_a = false,
          R"pb(Consume this program and emit LLVM bitcode.

The source is consumed even on failure. Set ``copy=True`` to preserve it.

Args:
    copy: Preserve the source program by exporting a copy.)pb")
      .def(
          "write_bitcode",
          [](mlir::QIRProgram& value, const std::filesystem::path& path,
             const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            const auto context = source.context();
            withDiagnostics<nb::exception_type::runtime_error>(
                context.get(), "cannot write QIR bitcode",
                [&] { return std::move(source).writeBitcode(path); });
          },
          "path"_a, nb::kw_only(), "copy"_a = false,
          R"pb(Consume this program and write LLVM bitcode to a file.

The source is consumed even on failure. Set ``copy=True`` to preserve it.

Args:
    path: Destination file.
    copy: Preserve the source program by exporting a copy.)pb");

  qcoProgram.def(
      "build_functionality", &buildQCOFunctionality, "dd_package"_a,
      // Keep the DD package alive while the returned matrix DD is alive.
      nb::keep_alive<0, 2>(),
      R"pb(Build a matrix DD for a static unitary QCO program.

Args:
    dd_package: DD package with enough qubits for the program.

Returns:
    Matrix DD of the program functionality.

Raises:
    ValueError: When the program is unsupported for functionality construction.)pb");

  qcoProgram.def(
      "simulate", &simulateQCO, "initial_state"_a, "dd_package"_a,
      "seed"_a = 0U,
      // Keep the DD package alive while the returned vector DD is alive.
      nb::keep_alive<0, 3>(),
      R"pb(Simulate a QCO program on a DD state.

Args:
    initial_state: Input state DD that spans at least the program's qubits and
        has a live reference in ``dd_package``. Higher wires are preserved. A
        valid input reference is consumed.
    dd_package: DD package with enough qubits for the program.
    seed: RNG seed. ``0`` (default) selects nondeterministic seeding. Any other
        value produces reproducible measurement and reset results.

Returns:
    Output state DD.

Raises:
    ValueError: When ``initial_state`` has no live reference in ``dd_package``,
        has too few qubits, or the program is unsupported for simulation.)pb");

  qcoProgram.def("sample", &sampleQCO, "shots"_a = 1024U, "seed"_a = 0U,
                 R"pb(Sample the declared outputs of a QCO program.

Args:
    shots: Number of shots (default 1024).
    seed: RNG seed. ``0`` (default) selects nondeterministic seeding. Any other
        value produces reproducible results.

Returns:
    Histogram keys use conventional count-string order. The last returned
    CBit register or Boolean value comes first, and each register is MSB-first.
    Other classical output types are rejected. Without classical results,
    final ``measureAll`` bitstrings are used instead.

Raises:
    ValueError: When the program is unsupported for sampling.)pb");

  m.def("build_functionality",
        PythonInputAdapter<&buildDenseFunctionality>{inputTypes}, "program"_a,
        nb::sig("def build_functionality(program: str | os.PathLike[str] | "
                "qiskit.circuit.QuantumCircuit | QCProgram | QCOProgram | "
                "JeffProgram | OpenQASMProgram) -> "
                "typing.Annotated[numpy.typing.NDArray[numpy.complex128], "
                "{'shape': (None, None)}]"),
        R"pb(Build the full unitary matrix of a supported compiler input.

The DD package is managed internally. The matrix is materialized directly into
the returned NumPy array without an additional copy. The full matrix grows
exponentially, and the caller is responsible for requesting a result that fits
in memory.

Raises:
    MemoryError: When the dense matrix does not fit in memory.
    ValueError: When the program is unsupported or the matrix dimensions exceed
        addressable memory.)pb");

  m.def("simulate", PythonInputAdapter<&simulateDense>{inputTypes}, "program"_a,
        nb::sig("def simulate(program: str | os.PathLike[str] | "
                "qiskit.circuit.QuantumCircuit | QCProgram | QCOProgram | "
                "JeffProgram | OpenQASMProgram) -> "
                "typing.Annotated[numpy.typing.NDArray[numpy.complex128], "
                "{'shape': (None,)}]"),
        R"pb(Simulate a closed compiler input from the all-zero state.

The DD package is managed internally. Terminal measurements returned directly
or stored in returned classical registers do not collapse the state.
Mid-circuit measurement feedback and resets are unsupported; use
:py:meth:`QCOProgram.simulate` with an explicit DD package for those workflows
or for a custom initial state.

Args:
    program: Compiler input to lower directly to QCO.

Returns:
    Full statevector, materialized directly into the returned NumPy array.

Raises:
    MemoryError: When the dense statevector does not fit in memory.
    ValueError: When the program is not closed, is unsupported for statevector
        simulation, or the statevector dimensions exceed addressable memory.)pb");

  m.def("sample", PythonInputAdapter<&sample>{inputTypes}, "program"_a,
        "shots"_a = 1024U, "seed"_a = 0U,
        nb::sig("def sample(program: str | os.PathLike[str] | "
                "qiskit.circuit.QuantumCircuit | QCProgram | QCOProgram | "
                "JeffProgram | OpenQASMProgram, shots: int = 1024, seed: int = "
                "0) -> dict[str, int]"),
        R"pb(Sample a supported input after translating or converting it to QCO.

An existing QCO program is used without copying. See
{py:meth}`QCOProgram.sample` for the shot, seed, histogram, and error
contracts.)pb");

  nb::class_<mlir::CompiledProgram>(
      m, "CompiledProgram", "A compiled program ready for QDMI submission.")
      .def_prop_ro("program_format", &mlir::CompiledProgram::programFormat,
                   "The exact QDMI program format.")
      .def_prop_ro(
          "payload",
          [](const mlir::CompiledProgram& self) -> nb::object {
            const auto& payload = self.payload();
            if (qdmi::isBinaryProgramFormat(self.programFormat())) {
              return nb::bytes(payload.data(), payload.size());
            }
            return nb::str(payload.data(), payload.size());
          },
          nb::sig("def payload(self) -> str | bytes"),
          "The serialized program.")
      .def_prop_ro(
          "target",
          [](const mlir::CompiledProgram& self) {
            return self.environment().target();
          },
          "The hardware snapshot used for compilation.")
      .def_prop_ro(
          "payload_specification",
          [](const mlir::CompiledProgram& self) {
            return self.environment().payloadSpecification();
          },
          "The payload format and capabilities used for compilation.");

  m.def(
      "compile_program", PythonInputAdapter<&compilePythonProgram>{inputTypes},
      "program"_a, nb::kw_only(), "target"_a = nb::none(),
      "output"_a = nb::none(), "copy"_a = false,
      "qco_pipeline"_a = "mqt-qco-default",
      "options"_a = mlir::CompilationOptions{},
      R"pb(Compile a program or a homogeneous sequence for an optional hardware target.

A sequence returns a list in input order, including for empty and singleton
inputs. Elements must share one supported input kind; subclasses are accepted.
Strings and paths are single inputs. Generators and binary buffers are unsupported.

An explicit ``output`` selects each result's representation. Without it, a
Qiskit target selects QuantumCircuit and a QDMI device selects
:class:`CompiledProgram` for :func:`submit_program`. Otherwise, each input
retains its format family: OpenQASM source returns :class:`OpenQASMProgram`,
QC MLIR returns :class:`QCProgram`, jeff files return :class:`JeffProgram`, and
QCO inputs return optimized QCO.

Typed MLIR inputs are consumed; set ``copy=True`` to preserve them. Source text,
paths, Qiskit circuits and OpenQASM programs remain reusable. Sequence input
kinds are checked before compilation. Compilation stops at the first failure
and returns no results; processed inputs, including the failing input, may be
consumed, while later inputs remain usable.

Args:
    program: Source text, a file path, a Qiskit circuit, a typed compiler program,
        or a homogeneous sequence of these inputs.
    target: A CompilerTarget, Qiskit BackendV2 or Target, QDMI Device, or device ID.
    output: The returned representation. QC_IMPORT and raw QCO are checkpoints
        before target compilation and cannot be combined with a target.
    copy: Preserve typed inputs by compiling copies. Defaults to False.
    qco_pipeline: The QCO pipeline, defaulting to "mqt-qco-default".
        Custom pipelines require no target.
    options: Shared compilation controls.

Device defaults prefer Adaptive QIR (binary, text), OpenQASM 3, IQM JSON, then Base QIR
(binary, text). Select an exact execution format with
:func:`submit_program` and its ``program_format`` argument. An explicit output
need not be a device's execution format. Jeff output does not retain hardware
placement.)pb");

  m.def("submit_program", PythonInputAdapter<&submitProgram>{inputTypes},
        "program"_a, nb::kw_only(), "target"_a, "num_shots"_a = 1024,
        "program_format"_a = nb::none(), "custom1"_a = nb::none(),
        "custom2"_a = nb::none(), "custom3"_a = nb::none(),
        "custom4"_a = nb::none(), "custom5"_a = nb::none(),
        "options"_a = nb::none(),
        R"pb(Compile source or submit a compiled program to a device.

``target`` accepts a registered device ID or an open device. For source input,
``program_format`` selects the exact QDMI execution format, including text or
binary QIR; omitting it selects a supported native format. A compiled program
retains its payload and must match the destination contract.)pb");
}

} // namespace mqt

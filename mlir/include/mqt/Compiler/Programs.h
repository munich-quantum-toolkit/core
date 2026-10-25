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

#include "mqt/Compiler/CompilationOptions.h"

#include "capnp/message.h"
#include "jeff.capnp.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"

#include "llvm/Support/LogicalResult.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {

class QCProgram;
class QCOProgram;
class JeffProgram;
class OpenQASMProgram;
class IQMJSONProgram;
class QIRProgram;
class TargetEnvironment;

/// A value for a scalar or statically sized one-dimensional f64 input.
using ParameterValue = std::variant<std::vector<double>, double>;

/// Quantum resources and static IR statistics.
struct QuantumProgramInfo {
  /// Declared quantum capacity.
  ///
  /// Counts allocated qubits or distinct static site IDs.
  /// Unknown for runtime-sized allocations, quantum inputs, size overflow,
  /// or no entry point. Includes helpers and excludes nested modules.
  std::optional<uint64_t> numQubits;
  /// Sorted, distinct physical site IDs declared in the module.
  std::vector<uint64_t> staticQubits;
  /// Whether the module contains branching or region-based control flow.
  bool hasControlFlow = false;
  /// Entry-point gate counts with the same semantics as QCProgram::numGates.
  size_t numGates = 0;
  size_t numSingleQubitGates = 0;
  size_t numTwoQubitGates = 0;
  /// Entry-point gate histogram, as in QCProgram::gateCounts().
  std::map<std::string, size_t> gateCounts;
  /// Entry-point branches and region-based control flow by MLIR operation name.
  std::map<std::string, size_t> controlFlowCounts;
  /// All operations in the module, as in Program::operationCounts().
  std::map<std::string, size_t> operationCounts;
};

/// Cached resource and gate inspection for QC and QCO module passes.
///
/// Uses the same static counting rules as QCProgram::inspect() and
/// QCOProgram::inspect(). Request it with getAnalysis<QuantumProgramAnalysis>()
/// from a module pass. MLIR invalidates the snapshot after passes that do not
/// preserve this analysis. Direct IR changes require explicit invalidation.
class QuantumProgramAnalysis {
public:
  explicit QuantumProgramAnalysis(ModuleOp moduleOp);

  /// Return the resource and operation counts without changing the IR.
  [[nodiscard]] const QuantumProgramInfo& info() const { return info_; }

private:
  QuantumProgramInfo info_;
};

/// The QIR profile represented by a QIR program.
enum class QIRProfile : uint8_t {
  /// The QIR Base Profile.
  Base,
  /// The QIR Adaptive Profile.
  Adaptive,
};

/// Formats accepted and produced by the default compiler pipeline.
enum class ProgramFormat : uint8_t {
  /// QC directly after frontend import, without any compiler pass.
  QCImport,
  /// QCO immediately after conversion, before cleanup and optimization.
  QCO,
  /// QCO after the default or user-supplied optimization pipeline.
  QCOOptimized,
  /// QC after the optimized QCO round trip.
  QC,
  /// Portable OpenQASM after the optimized QCO round trip.
  OpenQASM3,
  /// Serializable `jeff` MLIR.
  Jeff,
  /// QIR for the Base Profile.
  QIRBase,
  /// QIR for the Adaptive Profile.
  QIRAdaptive,
  /// An IQM circuit with physical PRX/CZ gates and terminal readout.
  IQMJSON,
};

/// A move-aware MLIR program with a shared dialect context.
///
/// Programs own their module and keep the context alive for its full
/// lifetime. Dialect-changing operations consume an rvalue program, making
/// ownership transfer explicit and avoiding expensive implicit cloning.
/// Consuming operations invalidate the source even when they fail.
class Program {
public:
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;
  Program(Program&&) noexcept = default;
  Program& operator=(Program&& other) noexcept;
  virtual ~Program() = default;

  /// Check whether this program still owns a module.
  [[nodiscard]] bool isValid() const noexcept;

  /// Return the program as textual MLIR.
  [[nodiscard]] std::string str() const;

  /// Count every operation by its full MLIR name.
  ///
  /// Includes the root module, helper functions, modifier bodies, terminators,
  /// and nested modules.
  [[nodiscard]] std::map<std::string, size_t> operationCounts() const;

  /// Borrow the owned MLIR module.
  ///
  /// The returned operation remains valid while this program owns its
  /// module. Consuming or destroying the program invalidates the operation.
  [[nodiscard]] ModuleOp module() const;

  /// Keep the dialect context alive across consuming compiler operations.
  [[nodiscard]] std::shared_ptr<MLIRContext> context() const noexcept;

protected:
  struct Storage {
    std::shared_ptr<MLIRContext> context;
    OwningOpRef<ModuleOp> mod;
  };

  explicit Program(Storage storage);

  /// Return the owned module.
  ///
  /// Requires a valid program.
  [[nodiscard]] ModuleOp mod() const;

  /// Clone the owned module while sharing its immutable dialect context.
  [[nodiscard]] Storage cloneStorage() const;

  /// Transfer module ownership to a new program.
  [[nodiscard]] Storage releaseStorage() &&;

private:
  Storage storage_;
};

/// An owned OpenQASM source program.
class OpenQASMProgram final {
public:
  explicit OpenQASMProgram(std::string source) : source_(std::move(source)) {}

  /// Return the OpenQASM source.
  [[nodiscard]] const std::string& source() const noexcept;

  /// Transfer ownership of the OpenQASM source.
  [[nodiscard]] std::string intoSource() &&;

  /// Return the OpenQASM source.
  [[nodiscard]] const std::string& str() const noexcept;

  /// Write the OpenQASM source to a file.
  [[nodiscard]] LogicalResult write(const std::filesystem::path& path) const;

private:
  std::string source_;
};

/// An emitted IQM circuit object, without submission options.
class IQMJSONProgram final {
public:
  explicit IQMJSONProgram(std::string source) : source_(std::move(source)) {}

  /// Return the IQM JSON source.
  [[nodiscard]] const std::string& source() const noexcept { return source_; }

  /// Transfer ownership of the IQM JSON source.
  [[nodiscard]] std::string intoSource() && { return std::move(source_); }

  /// Return the IQM JSON source.
  [[nodiscard]] const std::string& str() const noexcept { return source_; }

  /// Write the IQM JSON source to a file.
  [[nodiscard]] LogicalResult write(const std::filesystem::path& path) const;

private:
  std::string source_;
};

/// A QC program with reference semantics.
class QCProgram final : public Program {
public:
  explicit QCProgram(Storage storage) : Program(std::move(storage)) {}

  /// Parse QC MLIR assembly.
  [[nodiscard]] static FailureOr<QCProgram>
  fromMLIRString(std::string_view source);

  /// Parse QC MLIR assembly from a file.
  [[nodiscard]] static FailureOr<QCProgram>
  fromMLIRFile(const std::filesystem::path& path);

  /// Translate supported OpenQASM source to QC.
  ///
  /// Accepts versionless input and versions 2.0, 3.0, and 3.1.
  [[nodiscard]] static FailureOr<QCProgram>
  fromOpenQASMString(std::string_view source);

  /// Translate a supported OpenQASM file to QC.
  ///
  /// Accepts versionless input and versions 2.0, 3.0, and 3.1.
  [[nodiscard]] static FailureOr<QCProgram>
  fromOpenQASMFile(const std::filesystem::path& path);

  /// Take ownership of an MLIR module that contains a QC program.
  ///
  /// The context must own every dialect referenced by the module and must
  /// remain the module's context. The factory verifies the module and rejects
  /// QCO and QTensor operations. QC operations are not required. Dynamic
  /// quantum allocations require an `mqt.entry_point` function and must appear
  /// directly in its entry block.
  [[nodiscard]] static FailureOr<QCProgram>
  fromModule(std::shared_ptr<MLIRContext> context,
             OwningOpRef<ModuleOp> moduleOp);
  // Reject implicit adoption of a borrowed module.
  static FailureOr<QCProgram> fromModule(std::shared_ptr<MLIRContext>,
                                         ModuleOp) = delete;
  /// Create an independent QC program copy.
  [[nodiscard]] QCProgram copy() const;

  /// Inspect quantum resources and static IR statistics.
  ///
  /// Returns resource information and gate, control-flow, and operation counts.
  [[nodiscard]] QuantumProgramInfo inspect() const;

  /// Return names of scalar and static rank-one f64 inputs in argument order.
  [[nodiscard]] std::vector<std::string> parameters() const;

  /// Bind named f64 scalars and arrays in place, preserving unbound inputs.
  ///
  /// Arrays must match the length of a static rank-one f64 tensor input.
  /// Reject unknown names, mismatched types or lengths, non-finite values, or
  /// references to the entry point without changing the program. Preserve
  /// unbound input metadata and fold expressions using the bound values.
  /// Folding failures may leave the program modified.
  [[nodiscard]] LogicalResult
  bindParameters(const std::map<std::string, ParameterValue>& values);

  /// Bind a scalar-only map under the same contract.
  template <std::same_as<double> T = double>
  [[nodiscard]] LogicalResult
  bindParameters(const std::map<std::string, T>& values) {
    return bindParameters(
        std::map<std::string, ParameterValue>(values.begin(), values.end()));
  }

  /// Run the standard QC cleanup passes in place.
  [[nodiscard]] LogicalResult cleanup();

  /// Normalize scoped global phases in place.
  [[nodiscard]] LogicalResult normalizeGlobalPhases();

  /// Consume this program and translate it to portable OpenQASM 3.1.
  ///
  /// Run QC cleanup before export. The program is consumed even when export
  /// fails; call copy() first to preserve it.
  [[nodiscard]] FailureOr<OpenQASMProgram> intoOpenQASM3() &&;

  /// Consume this program and convert it to QCO.
  [[nodiscard]] FailureOr<QCOProgram> intoQCO() &&;

  /// Consume this program and lower it to QIR.
  ///
  /// Entry results must be direct measurements or CBit registers. Other scalar
  /// outputs are unsupported. The generated QIR returns a zero execution
  /// status. Base conversion unrolls static counted loops, allowing at most
  /// one billion cloned body operations, and rejects runtime-dependent loops.
  /// Adaptive conversion retains loops.
  [[nodiscard]] FailureOr<QIRProgram> intoQIR(QIRProfile profile) &&;

  /// Return the static gate count of the entry-point IR.
  ///
  /// Unitary operations, measurements, and resets each count once. Barriers
  /// are excluded. Modifiers and calls count atomically. Gates in every
  /// control-flow region count once, regardless of runtime paths or loop
  /// iterations.
  [[nodiscard]] size_t numGates() const;

  /// Count gates acting on exactly one qubit.
  ///
  /// Uses the rules of numGates(), including measurements and resets.
  [[nodiscard]] size_t numSingleQubitGates() const;

  /// Count gates acting on exactly two qubits.
  ///
  /// Uses the rules of numGates().
  [[nodiscard]] size_t numTwoQubitGates() const;

  /// Count entry-point gates by name.
  ///
  /// Uses the rules of numGates(). Controls on a single primitive gate add a
  /// `c` per control: `cx`, `ccx`.
  /// Other single-gate modifiers use `inv(h)`, `pow(rx)`, or `ctrl(inv(x))`;
  /// multiple controls use `ctrl(2,inv(x))`. Parameters do not split buckets.
  /// Composite bodies or unused modifier targets retain `ctrl`, `inv`, or
  /// `pow`. Unitary calls use the callee name; explicit phases use `gphase`.
  [[nodiscard]] std::map<std::string, size_t> gateCounts() const;

  /// Count entry-point control-flow operations.
  ///
  /// Keys are full MLIR names such as `scf.for` and `qco.if`.
  /// Visit every region once, without expanding calls.
  /// Region terminators such as `scf.yield` are excluded.
  [[nodiscard]] std::map<std::string, size_t> controlFlowCounts() const;
};

/// A QCO program with value semantics.
class QCOProgram final : public Program {
public:
  /// Parse QCO MLIR assembly.
  [[nodiscard]] static FailureOr<QCOProgram>
  fromMLIRString(std::string_view source);

  /// Parse QCO MLIR assembly from a file.
  [[nodiscard]] static FailureOr<QCOProgram>
  fromMLIRFile(const std::filesystem::path& path);

  /// Take ownership of an MLIR module that contains a QCO program.
  ///
  /// The context must own every dialect referenced by the module and must
  /// remain the module's context. The factory verifies the module and QCO
  /// linearity and rejects QC operations. QCO operations are not required.
  /// Dynamic quantum allocations require an `mqt.entry_point` function and
  /// must appear directly in its entry block.
  [[nodiscard]] static FailureOr<QCOProgram>
  fromModule(std::shared_ptr<MLIRContext> context,
             OwningOpRef<ModuleOp> moduleOp);
  // Reject implicit adoption of a borrowed module.
  static FailureOr<QCOProgram> fromModule(std::shared_ptr<MLIRContext>,
                                          ModuleOp) = delete;

  /// Create an independent QCO program copy.
  [[nodiscard]] QCOProgram copy() const;

  /// Inspect quantum resources and static IR statistics.
  ///
  /// Uses the rules of QCProgram::inspect().
  [[nodiscard]] QuantumProgramInfo inspect() const;

  /// Return names of scalar and static rank-one f64 inputs in argument order.
  [[nodiscard]] std::vector<std::string> parameters() const;

  /// Bind named f64 scalars and arrays under the same contract as QCProgram.
  [[nodiscard]] LogicalResult
  bindParameters(const std::map<std::string, ParameterValue>& values);

  /// Bind a scalar-only map under the same contract as QCProgram.
  template <std::same_as<double> T = double>
  [[nodiscard]] LogicalResult
  bindParameters(const std::map<std::string, T>& values) {
    return bindParameters(
        std::map<std::string, ParameterValue>(values.begin(), values.end()));
  }

  /// Run the standard QCO cleanup passes in place.
  [[nodiscard]] LogicalResult cleanup();

  /// Normalize scoped global phases in place.
  [[nodiscard]] LogicalResult normalizeGlobalPhases();

  /// Run an MLIR textual QCO pass pipeline in place.
  [[nodiscard]] LogicalResult
  runPassPipeline(std::string_view pipeline,
                  const CompilationOptions& options = {});

  /// Merge consecutive single-qubit rotation gates.
  [[nodiscard]] LogicalResult mergeSingleQubitRotationGates();

  /// Fuse single-qubit unitary runs into the selected Euler basis.
  [[nodiscard]] LogicalResult
  fuseSingleQubitUnitaryRuns(std::string_view basis = "zyz");

  /// Unroll loops containing quantum operations.
  [[nodiscard]] LogicalResult unrollQuantumLoops(int64_t factor = -1);

  /// Lift Hadamard gates away from measurements.
  [[nodiscard]] LogicalResult liftHadamards();

  /// Reuse independent single-qubit allocations.
  [[nodiscard]] LogicalResult reuseQubits();

  /// Prepare the program for qubit reuse and reuse eligible qubits.
  [[nodiscard]] LogicalResult runQubitReusePipeline();

  /// Decompose gates that act on at least @p minQubits qubits.
  ///
  /// Supports controlled X/Y/Z/SWAP and RX/RY/RZ gates, `qco.rccx`, and
  /// constant-angle phase gates. @p minQubits must be at least 3; default 3
  /// means wider than two-qubit.
  [[nodiscard]] LogicalResult decomposeMultiControlled(uint64_t minQubits = 3);

  /// Compile for a target and attach layout metadata when possible.
  ///
  /// Reject a program with attached layout metadata. Do not rely on the program
  /// contents if compilation fails.
  [[nodiscard]] LogicalResult
  compileForTarget(const TargetEnvironment& environment,
                   const CompilationOptions& options = {});

  /// Synthesize native operations without routing.
  ///
  /// Dynamic qubits require all-to-all connectivity and receive layout metadata
  /// when possible. Static qubits keep their device site IDs and must fit the
  /// target topology. Do not rely on the program contents if synthesis fails.
  [[nodiscard]] LogicalResult
  synthesizeForTarget(const TargetEnvironment& environment,
                      const CompilationOptions& options = {});

  /// Consume this program and convert it to QC.
  [[nodiscard]] FailureOr<QCProgram> intoQC() &&;

  /// Consume this program and convert it to `jeff` MLIR.
  [[nodiscard]] FailureOr<JeffProgram> intoJeff() &&;

  /// Return the static entry-point gate count.
  ///
  /// Uses the rules of QCProgram::numGates().
  [[nodiscard]] size_t numGates() const;

  /// Count static single-qubit gates.
  ///
  /// Uses the rules of QCProgram::numSingleQubitGates().
  [[nodiscard]] size_t numSingleQubitGates() const;

  /// Count static two-qubit gates.
  ///
  /// Uses the rules of QCProgram::numTwoQubitGates().
  [[nodiscard]] size_t numTwoQubitGates() const;

  /// Count entry-point gates by name.
  ///
  /// Uses the rules of QCProgram::gateCounts().
  [[nodiscard]] std::map<std::string, size_t> gateCounts() const;

  /// Count entry-point control-flow operations.
  ///
  /// Uses the rules of QCProgram::controlFlowCounts().
  [[nodiscard]] std::map<std::string, size_t> controlFlowCounts() const;

private:
  friend class QCProgram;
  friend class JeffProgram;

  explicit QCOProgram(Storage storage) : Program(std::move(storage)) {}
};

/// A serializable compiler program in the `jeff` dialect.
class JeffProgram final : public Program {
public:
  explicit JeffProgram(Storage storage) : Program(std::move(storage)) {}

  /// Deserialize an existing jeff message.
  ///
  /// Keep the reader's backing storage alive and unchanged until this function
  /// returns. Return failure and emit a diagnostic if deserialization fails.
  [[nodiscard]] static FailureOr<JeffProgram>
  fromMessage(::jeff::Module::Reader module);

  /// Deserialize a `jeff` binary buffer.
  [[nodiscard]] static FailureOr<JeffProgram>
  fromBytes(std::span<const std::byte> bytes);

  /// Deserialize a `jeff` binary file.
  [[nodiscard]] static FailureOr<JeffProgram>
  fromFile(const std::filesystem::path& path);

  /// Create an independent `jeff` program copy.
  [[nodiscard]] JeffProgram copy() const;

  /// Run the standard `jeff` cleanup passes in place.
  [[nodiscard]] LogicalResult cleanup();

  /// Serialize this program into a fresh caller-owned Cap'n Proto message.
  ///
  /// Keep the message alive and unchanged while using its root or the views
  /// returned by `getSegmentsForOutput()`. Discard the message if encoding
  /// fails.
  void toMessage(capnp::MessageBuilder& message) const;

  /// Serialize this program to a binary `jeff` buffer.
  ///
  /// Return an empty vector and emit a diagnostic if serialization fails.
  [[nodiscard]] std::vector<std::byte> toBytes() const;

  /// Serialize this program to a binary `jeff` file.
  [[nodiscard]] LogicalResult write(const std::filesystem::path& path) const;

  /// Consume this program and convert it to QCO.
  [[nodiscard]] FailureOr<QCOProgram> intoQCO() &&;
};

/// A QIR program.
class QIRProgram final : public Program {
public:
  QIRProgram(Storage storage, QIRProfile profile);

  /// Create an independent QIR program copy.
  [[nodiscard]] QIRProgram copy() const;

  /// Run QIR cleanup passes in place.
  [[nodiscard]] LogicalResult cleanup();

  /// Return the selected QIR profile.
  [[nodiscard]] QIRProfile profile() const noexcept;

  /// Consume this QIR program and translate it to LLVM IR text.
  [[nodiscard]] FailureOr<std::string> intoLLVMIR() &&;

  /// Consume this QIR program and translate it to LLVM bitcode in memory.
  [[nodiscard]] FailureOr<std::vector<std::byte>> intoBitcode() &&;

  /// Consume this QIR program and write verified bitcode to a stream.
  ///
  /// Leave the stream unchanged if translation or verification fails.
  [[nodiscard]] LogicalResult intoBitcode(llvm::raw_ostream& output) &&;

  /// Consume this QIR program and write it as LLVM bitcode.
  [[nodiscard]] LogicalResult
  writeBitcode(const std::filesystem::path& path) &&;

private:
  QIRProfile profile_;
};

/// Creates a context that holds every dialect the pipelines need.
///
/// Callers that build a module themselves need a context with the same dialects
/// that the conversions and the backends create operations from.
[[nodiscard]] std::shared_ptr<MLIRContext> createCompilerContext();

/// Valid input variants for the default compiler pipeline.
using CompilerInput =
    std::variant<QCProgram, QCOProgram, JeffProgram, OpenQASMProgram>;

/// The program variants returned by the default compiler pipeline.
using CompilerProgram =
    std::variant<QCProgram, QCOProgram, JeffProgram, OpenQASMProgram,
                 QIRProgram, IQMJSONProgram>;

/// Run the coordinated default compiler pipeline.
///
/// The supplied program is consumed. Call `copy()` before this function
/// when the source program must remain available for another pipeline branch.
[[nodiscard]] FailureOr<CompilerProgram>
runDefaultPipeline(CompilerInput program, ProgramFormat output,
                   std::string_view qcoPipeline = "mqt-qco-default",
                   const CompilationOptions& options = {});

/// Compile for a target and return the requested representation.
///
/// QCImport and raw QCO stop before target compilation and are not accepted.
/// The environment must include the requested output's lowering constraints.
[[nodiscard]] FailureOr<CompilerProgram>
runDefaultPipeline(CompilerInput program, ProgramFormat output,
                   const TargetEnvironment& environment,
                   const CompilationOptions& options = {});

/// Run the coordinated default compiler pipeline for a target.
///
/// The supplied program is consumed. Call `copy()` before this function
/// when the source program must remain available for another pipeline branch.
[[nodiscard]] FailureOr<CompilerProgram>
runDefaultPipeline(CompilerInput program, const TargetEnvironment& environment,
                   const CompilationOptions& options = {});

} // namespace mlir

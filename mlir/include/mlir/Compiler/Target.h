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

#include "mlir/Dialect/MQT/IR/MQTAttributes.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir {

class MLIRContext;
class Operation;

/// Immutable description of an MLIR compiler target.
///
/// Hardware sites retain their target-defined nonnegative i64
/// identifiers. Routing algorithms use dense zero-based vertices in site order.
/// Connectivity is either all-to-all or explicitly enumerated. Native-operation
/// support is either unrestricted or explicitly enumerated.
///
/// Compiler targets have shared immutable storage, making copies cheap while
/// preserving validated topology and capability caches.
class CompilerTarget {
public:
  using SiteId = int64_t;
  using Coupling = std::pair<SiteId, SiteId>;

  /// Target connectivity.
  class Connectivity {
  public:
    enum class Kind : uint8_t { AllToAll, Explicit };

    /// Create unrestricted all-to-all connectivity.
    [[nodiscard]] static Connectivity allToAll();

    /// Create explicitly enumerated connectivity.
    [[nodiscard]] static Connectivity
    fromCouplings(llvm::ArrayRef<Coupling> couplings);

    /// Return the connectivity kind.
    [[nodiscard]] Kind kind() const noexcept;

    /// Return explicitly enumerated couplings, if any.
    [[nodiscard]] llvm::ArrayRef<Coupling> couplings() const noexcept;

  private:
    friend class CompilerTarget;

    Connectivity(Kind kind, llvm::ArrayRef<Coupling> couplings);

    Kind kind_;
    llvm::SmallVector<Coupling> couplings_;
  };

  /// Unit shared by all raw timing metadata on a target.
  ///
  /// A raw duration denotes `value * scaleFactor()` units.
  class DurationUnit {
  public:
    /// Create a validated duration unit.
    [[nodiscard]] static llvm::Expected<DurationUnit>
    create(std::string unit, double scaleFactor);

    /// Return the target's duration unit.
    [[nodiscard]] llvm::StringRef unit() const noexcept;

    /// Return the positive finite multiplier for raw timing values.
    [[nodiscard]] double scaleFactor() const noexcept;

  private:
    DurationUnit(std::string unit, double scaleFactor);

    std::string unit_;
    double scaleFactor_;
  };

  /// A hardware site and its optional target metadata.
  class Site {
  public:
    /// Create validated hardware-site metadata.
    [[nodiscard]] static llvm::Expected<Site>
    create(SiteId id, std::optional<std::string> name = std::nullopt,
           std::optional<uint64_t> t1 = std::nullopt,
           std::optional<uint64_t> t2 = std::nullopt);

    /// Return the target-defined nonnegative site identifier.
    [[nodiscard]] SiteId id() const noexcept;

    /// Return the reported site name, if available.
    [[nodiscard]] std::optional<llvm::StringRef> name() const noexcept;

    /// Return the raw T1 coherence time, if available.
    [[nodiscard]] std::optional<uint64_t> t1() const noexcept;

    /// Return the raw T2 coherence time, if available.
    [[nodiscard]] std::optional<uint64_t> t2() const noexcept;

  private:
    Site(SiteId id, std::optional<std::string> name, std::optional<uint64_t> t1,
         std::optional<uint64_t> t2);

    SiteId id_;
    std::optional<std::string> name_;
    std::optional<uint64_t> t1_;
    std::optional<uint64_t> t2_;
  };

  /// Calibration data for an ordered tuple of hardware sites.
  class SiteTuple {
  public:
    /// Create validated calibration data for a site tuple.
    [[nodiscard]] static llvm::Expected<SiteTuple>
    create(std::vector<SiteId> sites,
           std::optional<uint64_t> duration = std::nullopt,
           std::optional<double> fidelity = std::nullopt);

    /// Return the ordered target site identifiers.
    [[nodiscard]] llvm::ArrayRef<SiteId> sites() const noexcept;

    /// Return the raw operation duration, if available.
    [[nodiscard]] std::optional<uint64_t> duration() const noexcept;

    /// Return the operation fidelity, if available.
    [[nodiscard]] std::optional<double> fidelity() const noexcept;

  private:
    SiteTuple(std::vector<SiteId> sites, std::optional<uint64_t> duration,
              std::optional<double> fidelity);

    std::vector<SiteId> sites_;
    std::optional<uint64_t> duration_;
    std::optional<double> fidelity_;
  };

  /// An operation capability described by a target.
  ///
  /// The reported name is retained verbatim while
  /// @ref canonicalName contains its normalized compiler spelling. Operations
  /// are available throughout the target; site tuples carry optional
  /// site-specific calibration data only.
  class Operation {
  public:
    /// The accepted number of qubits for an operation capability.
    class Arity {
    public:
      enum class Kind : uint8_t { Fixed, Variadic };

      /// Create an exact operation arity.
      [[nodiscard]] static Arity fixed(size_t value) noexcept;

      /// Create an operation arity with the given inclusive minimum.
      /// Operation construction requires a positive minimum.
      [[nodiscard]] static Arity variadic(size_t minimum) noexcept;

      /// Return the arity kind.
      [[nodiscard]] Kind kind() const noexcept;

      /// Return the exact arity or inclusive variadic minimum.
      [[nodiscard]] size_t value() const noexcept;

      /// Return whether this arity accepts a concrete operation width.
      [[nodiscard]] bool accepts(size_t width) const noexcept;

      friend bool operator==(const Arity&, const Arity&) = default;

    private:
      Arity(Kind kind, size_t value) noexcept;

      Kind kind_;
      size_t value_;
    };

    /// Create a validated operation capability.
    [[nodiscard]] static llvm::Expected<Operation>
    create(std::string name, size_t arity, size_t numParameters,
           std::vector<SiteTuple> siteTuples = {},
           std::optional<uint64_t> duration = std::nullopt,
           std::optional<double> fidelity = std::nullopt);

    /// Create a validated operation capability.
    [[nodiscard]] static llvm::Expected<Operation>
    create(std::string name, Arity arity, size_t numParameters,
           std::vector<SiteTuple> siteTuples = {},
           std::optional<uint64_t> duration = std::nullopt,
           std::optional<double> fidelity = std::nullopt);

    /// Return the exact reported operation name.
    [[nodiscard]] llvm::StringRef name() const noexcept;

    /// Return the canonical lower-case compiler operation name.
    [[nodiscard]] llvm::StringRef canonicalName() const noexcept;

    /// Return the accepted operation arity.
    [[nodiscard]] Arity arity() const noexcept;

    /// Return the number of real-valued operation parameters.
    [[nodiscard]] size_t numParameters() const noexcept;

    /// Return ordered site-specific calibration data.
    [[nodiscard]] llvm::ArrayRef<SiteTuple> siteTuples() const noexcept;

    /// Return the raw default operation duration, if available.
    [[nodiscard]] std::optional<uint64_t> duration() const noexcept;

    /// Return the default operation fidelity, if available.
    [[nodiscard]] std::optional<double> fidelity() const noexcept;

  private:
    Operation(std::string name, std::string canonicalName, Arity arity,
              size_t numParameters, std::vector<SiteTuple> siteTuples,
              std::optional<uint64_t> duration, std::optional<double> fidelity);

    std::string name_;
    std::string canonicalName_;
    Arity arity_;
    size_t numParameters_;
    std::vector<SiteTuple> siteTuples_;
    std::optional<uint64_t> duration_;
    std::optional<double> fidelity_;
  };

  /// Native-operation support.
  class NativeOperations {
  public:
    enum class Kind : uint8_t { Unrestricted, Explicit };

    /// Create unrestricted native-operation support.
    [[nodiscard]] static NativeOperations unrestricted();

    /// Create explicitly enumerated native-operation support.
    [[nodiscard]] static NativeOperations
    fromOperations(llvm::ArrayRef<Operation> operations);

    /// Return the native-operation support kind.
    [[nodiscard]] Kind kind() const noexcept;

    /// Return explicitly enumerated operations, if any.
    [[nodiscard]] llvm::ArrayRef<Operation> operations() const noexcept;

  private:
    friend class CompilerTarget;

    NativeOperations(Kind kind, llvm::ArrayRef<Operation> operations);

    Kind kind_;
    llvm::SmallVector<Operation> operations_;
  };

  /// Recognized native gate capability independent of synthesis code.
  enum class GateKind : uint8_t {
    U,
    X,
    SX,
    RZ,
    RX,
    RY,
    R,
    RXX,
    RYY,
    RZX,
    RZZ,
    ISWAP,
    CZ,
    CX,
    ECR,
  };

  /// Recognized globally usable single-qubit synthesis basis.
  enum class SingleQubitBasis : uint8_t {
    U,    ///< `U(theta, phi, lambda)`.
    ZSXX, ///< `RZ` / `SX` / `X` synthesis via a ZYZ decomposition.
    R,    ///< XYX synthesis expressed with `R(theta, phi)`.
    XZX,  ///< `RX(phi) * RZ(theta) * RX(lambda)`.
    XYX,  ///< `RX(phi) * RY(theta) * RX(lambda)`.
    ZYZ,  ///< `RZ(phi) * RY(theta) * RZ(lambda)`.
    ZXZ,  ///< `RZ(phi) * RX(theta) * RZ(lambda)`.
  };

  /// One single-qubit basis and entangler usable across the target.
  struct SynthesisBasis {
    SingleQubitBasis singleQubit;
    GateKind entangler;

    friend bool operator==(const SynthesisBasis&,
                           const SynthesisBasis&) = default;
  };

  /// Create an unnamed target with dense site IDs `0..numSites-1`.
  [[nodiscard]] static llvm::Expected<CompilerTarget>
  create(size_t numSites, Connectivity connectivity,
         NativeOperations nativeOperations,
         std::optional<DurationUnit> durationUnit = std::nullopt);

  /// Create a named target with dense site IDs `0..numSites-1`.
  [[nodiscard]] static llvm::Expected<CompilerTarget>
  create(std::string name, size_t numSites, Connectivity connectivity,
         NativeOperations nativeOperations,
         std::optional<DurationUnit> durationUnit = std::nullopt);

  /// Create an unnamed target from detailed sites.
  [[nodiscard]] static llvm::Expected<CompilerTarget>
  create(std::vector<Site> sites, Connectivity connectivity,
         NativeOperations nativeOperations,
         std::optional<DurationUnit> durationUnit = std::nullopt);

  /// Create a named target from detailed sites.
  [[nodiscard]] static llvm::Expected<CompilerTarget>
  create(std::string name, std::vector<Site> sites, Connectivity connectivity,
         NativeOperations nativeOperations,
         std::optional<DurationUnit> durationUnit = std::nullopt);

  /// Reconstruct a validated compiler target from its MLIR attribute.
  [[nodiscard]] static llvm::Expected<CompilerTarget>
  create(mqt::CompilationTargetAttr attribute);

  /// Copying shares immutable storage; rvalues copy and keep the source valid.
  CompilerTarget(const CompilerTarget&) noexcept = default;
  CompilerTarget& operator=(const CompilerTarget&) noexcept = default;
  ~CompilerTarget() = default;

  /// Return the target name, if provided.
  [[nodiscard]] std::optional<llvm::StringRef> name() const noexcept;

  /// Return the unit shared by all raw timing metadata, if provided.
  [[nodiscard]] const std::optional<DurationUnit>&
  durationUnit() const noexcept;

  /// Return the number of compiler vertices and hardware sites.
  [[nodiscard]] size_t numSites() const noexcept;

  /// Return detailed sites in dense compiler-vertex order.
  [[nodiscard]] llvm::ArrayRef<Site> sites() const noexcept;

  /// Return target site identifiers in dense compiler-vertex order.
  [[nodiscard]] llvm::ArrayRef<SiteId> siteIds() const noexcept;

  /// Return the dense compiler vertex for a target site identifier.
  [[nodiscard]] std::optional<size_t> vertexForSite(SiteId site) const noexcept;

  /// Return the target site identifier for a valid dense compiler vertex.
  [[nodiscard]] SiteId siteForVertex(size_t vertex) const;

  /// Return the connectivity kind.
  [[nodiscard]] Connectivity::Kind connectivityKind() const noexcept;

  /// Return sorted canonical undirected couplings in target site IDs.
  [[nodiscard]] llvm::ArrayRef<Coupling> couplings() const noexcept;

  /// Return whether two valid dense compiler vertices are adjacent.
  [[nodiscard]] bool areAdjacent(size_t source, size_t target) const;

  /// Return the cached shortest-path distance between valid vertices.
  [[nodiscard]] size_t distanceBetween(size_t source, size_t target) const;

  /// Invoke @p callback for every neighbour of a valid dense vertex.
  void forEachNeighbour(size_t vertex,
                        llvm::function_ref<void(size_t)> callback) const;

  /// Return the maximum degree of the target's routing topology.
  [[nodiscard]] size_t maxDegree() const noexcept;

  /// Return the native-operation support kind.
  [[nodiscard]] NativeOperations::Kind nativeOperationsKind() const noexcept;

  /// Return operation capabilities in reported order.
  [[nodiscard]] llvm::ArrayRef<Operation> operations() const noexcept;

  /// Return whether an operation capability is supported by the target.
  [[nodiscard]] bool
  supportsOperation(llvm::StringRef name, size_t arity,
                    std::optional<size_t> numParameters = std::nullopt) const;

  /// Return whether a QCO operation is supported.
  [[nodiscard]] bool supports(::mlir::Operation* operation) const;

  /// Return whether a recognized gate is supported.
  [[nodiscard]] bool supports(GateKind gate) const;

  /// Return the recognized gates supported by the target.
  [[nodiscard]] llvm::ArrayRef<GateKind> supportedGates() const noexcept;

  /// Return one complete globally usable synthesis basis, if available.
  [[nodiscard]] std::optional<SynthesisBasis> synthesisBasis() const noexcept;

  /// Materialize the source target facts as a typed MLIR attribute.
  [[nodiscard]] mqt::CompilationTargetAttr
  materialize(MLIRContext& context) const;

private:
  struct Storage;

  explicit CompilerTarget(std::shared_ptr<const Storage> storage);

  [[nodiscard]] static llvm::Expected<CompilerTarget>
  createImpl(std::optional<std::string> name, std::vector<Site> sites,
             Connectivity connectivity, NativeOperations nativeOperations,
             std::optional<DurationUnit> durationUnit);

  [[nodiscard]] llvm::ArrayRef<size_t> explicitNeighbours(size_t vertex) const;

  std::shared_ptr<const Storage> storage_;
};

} // namespace mlir

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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLFunctionalExtras.h>
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

class Operation;

/**
 * @brief Immutable description of an MLIR compiler target.
 *
 * @details Hardware sites retain their target-defined nonnegative i64
 * identifiers. Routing algorithms use dense zero-based vertices in site order.
 * An absent topology means all-to-all connectivity. An absent operation set
 * means that every operation is native; a present empty set means that no
 * hardware operation is native.
 *
 * Compiler targets have shared immutable storage, making copies cheap while
 * preserving validated topology and capability caches.
 */
class CompilerTarget {
public:
  using SiteId = int64_t;
  using Coupling = std::pair<SiteId, SiteId>;

  /**
   * @brief Unit shared by all raw timing metadata on a target.
   *
   * @details A raw duration denotes `value * scaleFactor()` units.
   */
  class DurationUnit {
  public:
    /**
     * @brief Create a validated duration unit.
     */
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

  /**
   * @brief A hardware site and its optional target metadata.
   */
  class Site {
  public:
    /**
     * @brief Create validated hardware-site metadata.
     */
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

  /**
   * @brief Calibration data for an ordered tuple of hardware sites.
   */
  class SiteTuple {
  public:
    /**
     * @brief Create validated calibration data for a site tuple.
     */
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

  /**
   * @brief An operation capability described by a target.
   *
   * @details The reported name is retained verbatim while
   * @ref canonicalName contains its normalized compiler spelling. Operations
   * are available throughout the target; site tuples carry optional
   * site-specific calibration data only.
   */
  class Operation {
  public:
    /**
     * @brief Create a validated operation capability.
     */
    [[nodiscard]] static llvm::Expected<Operation>
    create(std::string name, size_t numQubits, size_t numParameters,
           std::vector<SiteTuple> siteTuples = {},
           std::optional<uint64_t> duration = std::nullopt,
           std::optional<double> fidelity = std::nullopt);

    /// Return the exact reported operation name.
    [[nodiscard]] llvm::StringRef name() const noexcept;

    /// Return the canonical lower-case compiler operation name.
    [[nodiscard]] llvm::StringRef canonicalName() const noexcept;

    /// Return the positive fixed operation arity.
    [[nodiscard]] size_t numQubits() const noexcept;

    /// Return the number of real-valued operation parameters.
    [[nodiscard]] size_t numParameters() const noexcept;

    /// Return ordered site-specific calibration data.
    [[nodiscard]] llvm::ArrayRef<SiteTuple> siteTuples() const noexcept;

    /// Return the raw default operation duration, if available.
    [[nodiscard]] std::optional<uint64_t> duration() const noexcept;

    /// Return the default operation fidelity, if available.
    [[nodiscard]] std::optional<double> fidelity() const noexcept;

  private:
    Operation(std::string name, std::string canonicalName, size_t numQubits,
              size_t numParameters, std::vector<SiteTuple> siteTuples,
              std::optional<uint64_t> duration, std::optional<double> fidelity);

    std::string name_;
    std::string canonicalName_;
    size_t numQubits_;
    size_t numParameters_;
    std::vector<SiteTuple> siteTuples_;
    std::optional<uint64_t> duration_;
    std::optional<double> fidelity_;
  };

  /**
   * @brief Runtime classical-control capabilities supported by a target.
   *
   * @details Capabilities are opt-in. A target that declares none supports
   * only straight-line quantum programs.
   */
  enum class ClassicalControl : uint8_t {
    /// Runtime forward branching such as `qco.if` or `scf.if`.
    Conditional,
    /// Structured counted iteration such as `scf.for`.
    Iteration,
    /// Runtime condition-terminated looping such as `scf.while`.
    ConditionalLoop,
    /// Runtime multiway branching such as `qco.index_switch` or
    /// `scf.index_switch`.
    MultiwayBranch,
  };

  /**
   * @brief Recognized native gate capability independent of synthesis code.
   */
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

  /**
   * @brief Recognized globally usable single-qubit synthesis basis.
   */
  enum class SingleQubitBasis : uint8_t {
    U,    ///< `U(theta, phi, lambda)`.
    ZSXX, ///< `RZ` / `SX` / `X` synthesis via a ZYZ decomposition.
    R,    ///< XYX synthesis expressed with `R(theta, phi)`.
    XZX,  ///< `RX(phi) * RZ(theta) * RX(lambda)`.
    XYX,  ///< `RX(phi) * RY(theta) * RX(lambda)`.
    ZYZ,  ///< `RZ(phi) * RY(theta) * RZ(lambda)`.
    ZXZ,  ///< `RZ(phi) * RX(theta) * RZ(lambda)`.
  };

  /**
   * @brief One single-qubit basis and entangler usable across the target.
   */
  struct SynthesisBasis {
    SingleQubitBasis singleQubit;
    GateKind entangler;

    friend bool operator==(const SynthesisBasis&,
                           const SynthesisBasis&) = default;
  };

  /**
   * @brief Create an unnamed target with dense site IDs `0..numQubits-1`.
   */
  [[nodiscard]] static llvm::Expected<CompilerTarget>
  create(size_t numQubits,
         std::optional<std::vector<Coupling>> couplings = std::nullopt,
         std::optional<std::vector<Operation>> operations = std::nullopt,
         std::optional<DurationUnit> durationUnit = std::nullopt);

  /// Create an unnamed dense target with explicit classical-control support.
  [[nodiscard]] static llvm::Expected<CompilerTarget>
  create(size_t numQubits, std::optional<std::vector<Coupling>> couplings,
         std::optional<std::vector<Operation>> operations,
         std::optional<DurationUnit> durationUnit,
         std::vector<ClassicalControl> classicalControl);

  /**
   * @brief Create a named target with dense site IDs `0..numQubits-1`.
   */
  [[nodiscard]] static llvm::Expected<CompilerTarget>
  create(std::string name, size_t numQubits,
         std::optional<std::vector<Coupling>> couplings = std::nullopt,
         std::optional<std::vector<Operation>> operations = std::nullopt,
         std::optional<DurationUnit> durationUnit = std::nullopt);

  /// Create a named dense target with explicit classical-control support.
  [[nodiscard]] static llvm::Expected<CompilerTarget>
  create(std::string name, size_t numQubits,
         std::optional<std::vector<Coupling>> couplings,
         std::optional<std::vector<Operation>> operations,
         std::optional<DurationUnit> durationUnit,
         std::vector<ClassicalControl> classicalControl);

  /**
   * @brief Create an unnamed target from detailed sites.
   */
  [[nodiscard]] static llvm::Expected<CompilerTarget>
  create(std::vector<Site> sites,
         std::optional<std::vector<Coupling>> couplings = std::nullopt,
         std::optional<std::vector<Operation>> operations = std::nullopt,
         std::optional<DurationUnit> durationUnit = std::nullopt);

  /// Create an unnamed sparse target with explicit classical-control support.
  [[nodiscard]] static llvm::Expected<CompilerTarget>
  create(std::vector<Site> sites,
         std::optional<std::vector<Coupling>> couplings,
         std::optional<std::vector<Operation>> operations,
         std::optional<DurationUnit> durationUnit,
         std::vector<ClassicalControl> classicalControl);

  /**
   * @brief Create a named target from detailed sites.
   */
  [[nodiscard]] static llvm::Expected<CompilerTarget>
  create(std::string name, std::vector<Site> sites,
         std::optional<std::vector<Coupling>> couplings = std::nullopt,
         std::optional<std::vector<Operation>> operations = std::nullopt,
         std::optional<DurationUnit> durationUnit = std::nullopt);

  /// Create a named sparse target with explicit classical-control support.
  [[nodiscard]] static llvm::Expected<CompilerTarget>
  create(std::string name, std::vector<Site> sites,
         std::optional<std::vector<Coupling>> couplings,
         std::optional<std::vector<Operation>> operations,
         std::optional<DurationUnit> durationUnit,
         std::vector<ClassicalControl> classicalControl);

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
  [[nodiscard]] size_t numQubits() const noexcept;

  /// Return detailed sites in dense compiler-vertex order.
  [[nodiscard]] llvm::ArrayRef<Site> sites() const noexcept;

  /// Return target site identifiers in dense compiler-vertex order.
  [[nodiscard]] llvm::ArrayRef<SiteId> siteIds() const noexcept;

  /// Return the dense compiler vertex for a target site identifier.
  [[nodiscard]] std::optional<size_t> vertexForSite(SiteId site) const noexcept;

  /// Return the target site identifier for a valid dense compiler vertex.
  [[nodiscard]] SiteId siteForVertex(size_t vertex) const;

  /// Return whether the target contains an explicit coupling topology.
  [[nodiscard]] bool hasExplicitTopology() const noexcept;

  /**
   * @brief Return sorted canonical undirected couplings in target site IDs.
   */
  [[nodiscard]] llvm::ArrayRef<Coupling> couplings() const noexcept;

  /// Return whether two valid dense compiler vertices are adjacent.
  [[nodiscard]] bool areAdjacent(size_t source, size_t target) const;

  /**
   * @brief Return the cached shortest-path distance between valid vertices.
   */
  [[nodiscard]] size_t distanceBetween(size_t source, size_t target) const;

  /**
   * @brief Invoke @p callback for every neighbour of a valid dense vertex.
   */
  void forEachNeighbour(size_t vertex,
                        llvm::function_ref<void(size_t)> callback) const;

  /// Return the maximum degree of the target's routing topology.
  [[nodiscard]] size_t maxDegree() const noexcept;

  /// Return whether the target contains an explicit operation set.
  [[nodiscard]] bool hasExplicitOperations() const noexcept;

  /// Return operation capabilities in reported order.
  [[nodiscard]] llvm::ArrayRef<Operation> operations() const noexcept;

  /// Return the sorted classical-control capabilities supported by the target.
  [[nodiscard]] llvm::ArrayRef<ClassicalControl>
  classicalControl() const noexcept;

  /// Return whether the target supports a classical-control capability.
  [[nodiscard]] bool
  supportsClassicalControl(ClassicalControl capability) const noexcept;

  /**
   * @brief Return whether an operation capability is supported by the target.
   */
  [[nodiscard]] bool
  supportsOperation(llvm::StringRef name, size_t numQubits,
                    std::optional<size_t> numParameters = std::nullopt) const;

  /// Return whether a QCO operation is supported by the target.
  [[nodiscard]] bool supports(::mlir::Operation* operation) const;

  /// Return whether a recognized gate is supported by the target.
  [[nodiscard]] bool supports(GateKind gate) const;

  /// Return the recognized gates supported by the target.
  [[nodiscard]] llvm::ArrayRef<GateKind> supportedGates() const noexcept;

  /// Return one complete globally usable synthesis basis, if available.
  [[nodiscard]] std::optional<SynthesisBasis> synthesisBasis() const noexcept;

private:
  struct Storage;

  explicit CompilerTarget(std::shared_ptr<const Storage> storage);

  [[nodiscard]] static llvm::Expected<CompilerTarget>
  createImpl(std::optional<std::string> name, std::vector<Site> sites,
             std::optional<std::vector<Coupling>> couplings,
             std::optional<std::vector<Operation>> operations,
             std::optional<DurationUnit> durationUnit,
             std::vector<ClassicalControl> classicalControl);

  [[nodiscard]] llvm::ArrayRef<size_t> explicitNeighbours(size_t vertex) const;

  std::shared_ptr<const Storage> storage_;
};

} // namespace mlir

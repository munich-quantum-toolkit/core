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
 * @brief Immutable, provider-independent description of an MLIR compiler
 * target.
 *
 * @details Hardware sites retain their provider-defined nonnegative i64
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
    DurationUnit(std::string unit, double scaleFactor);

    /// Return the provider-defined duration unit.
    [[nodiscard]] llvm::StringRef unit() const noexcept;

    /// Return the positive finite multiplier for raw timing values.
    [[nodiscard]] double scaleFactor() const noexcept;

  private:
    std::string unit_;
    double scaleFactor_;
  };

  /**
   * @brief A hardware site and its optional provider metadata.
   */
  class Site {
  public:
    explicit Site(SiteId id, std::optional<std::string> name = std::nullopt,
                  std::optional<uint64_t> t1 = std::nullopt,
                  std::optional<uint64_t> t2 = std::nullopt);

    /// Return the provider-defined nonnegative site identifier.
    [[nodiscard]] SiteId id() const noexcept;

    /// Return the provider-defined site name, if available.
    [[nodiscard]] std::optional<llvm::StringRef> name() const noexcept;

    /// Return the raw T1 coherence time, if available.
    [[nodiscard]] std::optional<uint64_t> t1() const noexcept;

    /// Return the raw T2 coherence time, if available.
    [[nodiscard]] std::optional<uint64_t> t2() const noexcept;

  private:
    SiteId id_;
    std::optional<std::string> name_;
    std::optional<uint64_t> t1_;
    std::optional<uint64_t> t2_;
  };

  /**
   * @brief An ordered hardware locus and its optional calibration data.
   */
  class OperationLocus {
  public:
    explicit OperationLocus(std::vector<SiteId> sites,
                            std::optional<uint64_t> duration = std::nullopt,
                            std::optional<double> fidelity = std::nullopt);

    /// Return the ordered provider site identifiers.
    [[nodiscard]] llvm::ArrayRef<SiteId> sites() const noexcept;

    /// Return the raw operation duration, if available.
    [[nodiscard]] std::optional<uint64_t> duration() const noexcept;

    /// Return the operation fidelity, if available.
    [[nodiscard]] std::optional<double> fidelity() const noexcept;

  private:
    std::vector<SiteId> sites_;
    std::optional<uint64_t> duration_;
    std::optional<double> fidelity_;
  };

  /**
   * @brief An operation capability reported by a target provider.
   *
   * @details The provider name is retained verbatim while @ref canonicalName
   * contains its normalized compiler spelling. An absent locus set means the
   * operation applies to every valid ordered tuple of its arity; a present
   * empty set supports no locus.
   */
  class Operation {
  public:
    Operation(std::string providerName, size_t numQubits, size_t numParameters,
              std::optional<std::vector<OperationLocus>> loci = std::nullopt,
              std::optional<uint64_t> duration = std::nullopt,
              std::optional<double> fidelity = std::nullopt);

    /// Return the exact operation name reported by the provider.
    [[nodiscard]] llvm::StringRef providerName() const noexcept;

    /// Return the canonical lower-case compiler operation name.
    [[nodiscard]] llvm::StringRef canonicalName() const noexcept;

    /// Return the positive fixed operation arity.
    [[nodiscard]] size_t numQubits() const noexcept;

    /// Return the number of real-valued operation parameters.
    [[nodiscard]] size_t numParameters() const noexcept;

    /// Return whether this operation is available at every valid locus.
    [[nodiscard]] bool hasGlobalLoci() const noexcept;

    /// Return the explicitly supported ordered loci.
    [[nodiscard]] llvm::ArrayRef<OperationLocus> loci() const noexcept;

    /// Return the raw default operation duration, if available.
    [[nodiscard]] std::optional<uint64_t> duration() const noexcept;

    /// Return the default operation fidelity, if available.
    [[nodiscard]] std::optional<double> fidelity() const noexcept;

    /// Return whether this capability supports an ordered hardware locus.
    [[nodiscard]] bool supports(llvm::ArrayRef<SiteId> locus) const;

  private:
    std::string providerName_;
    std::string canonicalName_;
    size_t numQubits_;
    size_t numParameters_;
    std::optional<std::vector<OperationLocus>> loci_;
    std::optional<uint64_t> duration_;
    std::optional<double> fidelity_;
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
  enum class SingleQubitBasis : uint8_t { U, ZSXX, R, XZX, XYX, ZYZ };

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
   * @brief Construct an unnamed target with dense site IDs `0..numQubits-1`.
   */
  explicit CompilerTarget(
      size_t numQubits,
      std::optional<std::vector<Coupling>> couplings = std::nullopt,
      std::optional<std::vector<Operation>> operations = std::nullopt,
      std::optional<DurationUnit> durationUnit = std::nullopt);

  /**
   * @brief Construct a named target with dense site IDs `0..numQubits-1`.
   */
  CompilerTarget(
      std::string name, size_t numQubits,
      std::optional<std::vector<Coupling>> couplings = std::nullopt,
      std::optional<std::vector<Operation>> operations = std::nullopt,
      std::optional<DurationUnit> durationUnit = std::nullopt);

  /**
   * @brief Construct an unnamed target from detailed provider sites.
   */
  explicit CompilerTarget(
      std::vector<Site> sites,
      std::optional<std::vector<Coupling>> couplings = std::nullopt,
      std::optional<std::vector<Operation>> operations = std::nullopt,
      std::optional<DurationUnit> durationUnit = std::nullopt);

  /**
   * @brief Construct a named target from detailed provider sites.
   */
  CompilerTarget(
      std::string name, std::vector<Site> sites,
      std::optional<std::vector<Coupling>> couplings = std::nullopt,
      std::optional<std::vector<Operation>> operations = std::nullopt,
      std::optional<DurationUnit> durationUnit = std::nullopt);

  /// Copying shares immutable storage; rvalues copy and keep the source valid.
  CompilerTarget(const CompilerTarget&) noexcept = default;
  CompilerTarget& operator=(const CompilerTarget&) noexcept = default;
  ~CompilerTarget() = default;

  /// Return the target/device name, if provided.
  [[nodiscard]] std::optional<llvm::StringRef> name() const noexcept;

  /// Return the unit shared by all raw timing metadata, if provided.
  [[nodiscard]] const std::optional<DurationUnit>&
  durationUnit() const noexcept;

  /// Return the number of compiler vertices and hardware sites.
  [[nodiscard]] size_t numQubits() const noexcept;

  /// Return detailed sites in dense compiler-vertex order.
  [[nodiscard]] llvm::ArrayRef<Site> sites() const noexcept;

  /// Return provider site identifiers in dense compiler-vertex order.
  [[nodiscard]] llvm::ArrayRef<SiteId> siteIds() const noexcept;

  /// Return the dense compiler vertex for a provider site identifier.
  [[nodiscard]] std::optional<size_t> vertexForSite(SiteId site) const noexcept;

  /// Return the provider site identifier for a dense compiler vertex.
  [[nodiscard]] SiteId siteForVertex(size_t vertex) const;

  /// Return whether the target contains an explicit coupling topology.
  [[nodiscard]] bool hasExplicitTopology() const noexcept;

  /**
   * @brief Return sorted canonical undirected couplings in provider site IDs.
   */
  [[nodiscard]] llvm::ArrayRef<Coupling> couplings() const noexcept;

  /// Return whether two dense compiler vertices are adjacent.
  [[nodiscard]] bool areAdjacent(size_t source, size_t target) const;

  /**
   * @brief Return the cached shortest-path distance between dense vertices.
   */
  [[nodiscard]] size_t distanceBetween(size_t source, size_t target) const;

  /**
   * @brief Invoke @p callback for every neighbour of a dense compiler vertex.
   */
  void forEachNeighbour(size_t vertex,
                        llvm::function_ref<void(size_t)> callback) const;

  /// Return the maximum degree of the target's routing topology.
  [[nodiscard]] size_t maxDegree() const noexcept;

  /// Return whether the target contains an explicit operation set.
  [[nodiscard]] bool hasExplicitOperations() const noexcept;

  /// Return provider operation capabilities in provider order.
  [[nodiscard]] llvm::ArrayRef<Operation> operations() const noexcept;

  /**
   * @brief Return whether a canonical/provider operation is supported at an
   * ordered hardware locus.
   */
  [[nodiscard]] bool
  supportsOperation(llvm::StringRef name, llvm::ArrayRef<SiteId> locus,
                    std::optional<size_t> numParameters = std::nullopt) const;

  /**
   * @brief Return whether a QCO operation is supported at an ordered locus.
   */
  [[nodiscard]] bool supports(::mlir::Operation* operation,
                              llvm::ArrayRef<SiteId> locus) const;

  /**
   * @brief Return whether a recognized gate is supported at an ordered locus.
   */
  [[nodiscard]] bool supports(GateKind gate,
                              llvm::ArrayRef<SiteId> locus) const;

  /// Return recognized gates that are usable across the complete target.
  [[nodiscard]] llvm::ArrayRef<GateKind>
  globallySupportedGates() const noexcept;

  /// Return one complete globally usable synthesis basis, if available.
  [[nodiscard]] std::optional<SynthesisBasis> synthesisBasis() const noexcept;

private:
  struct Storage;
  struct StorageConstructorTag {};

  CompilerTarget(std::optional<std::string> name, std::vector<Site> sites,
                 std::optional<std::vector<Coupling>> couplings,
                 std::optional<std::vector<Operation>> operations,
                 std::optional<DurationUnit> durationUnit,
                 StorageConstructorTag storageConstructorTag);

  [[noreturn]] static void throwVertexOutOfRange();
  [[nodiscard]] llvm::ArrayRef<size_t> explicitNeighbours(size_t vertex) const;

  std::shared_ptr<const Storage> storage_;
};

} // namespace mlir

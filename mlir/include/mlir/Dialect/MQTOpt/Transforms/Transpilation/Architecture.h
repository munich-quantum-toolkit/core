#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <cstddef>
#include <string>
#include <string_view>

namespace transpilation {
/// @brief A quantum accelerator's architecture.
class Architecture {
public:
  using CouplingMap = llvm::DenseSet<std::pair<std::size_t, std::size_t>>;

  explicit Architecture(std::string name, std::size_t nqubits,
                        CouplingMap couplingMap)
      : name_(std::move(name)), nqubits_(nqubits),
        couplingMap_(std::move(couplingMap)),
        dist_(nqubits, llvm::SmallVector<std::size_t>(nqubits, UINT64_MAX)),
        prev_(nqubits, llvm::SmallVector<std::size_t>(nqubits, UINT64_MAX)) {
    floydWarshallWithPathReconstruction();
  }

  /// @brief Return the architecture's name.
  [[nodiscard]] constexpr std::string_view name() const { return name_; }

  /// @brief Return the architecture's number of qubits.
  [[nodiscard]] constexpr std::size_t nqubits() const { return nqubits_; }

  /// @brief Return true if @p u and @p v are adjacent.
  [[nodiscard]] bool areAdjacent(std::size_t u, std::size_t v) const {
    return couplingMap_.contains({u, v});
  }

  /// @brief Collect the shortest path between @p u and @p v.
  [[nodiscard]] llvm::SmallVector<std::size_t>
  shortestPathBetween(std::size_t u, std::size_t v) const;

private:
  using Matrix = llvm::SmallVector<llvm::SmallVector<std::size_t>>;

  /**
   * @brief Find all shortest paths in the coupling map between two qubits.
   * @details Vertices are the qubits. Edges connected two qubits.
   * @link Adapted from https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm
   */
  void floydWarshallWithPathReconstruction();

  std::string name_;
  std::size_t nqubits_;
  CouplingMap couplingMap_;

  Matrix dist_;
  Matrix prev_;
};

/// @brief Get architecture by its name.
std::unique_ptr<Architecture> getArchitecture(const std::string& name);

}; // namespace transpilation
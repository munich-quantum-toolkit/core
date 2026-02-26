/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Passes/Passes.h"

#include <cstdint>
#include <iterator>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#define DEBUG_TYPE "heuristic-mapping-pass"

namespace mlir::qco {

#define GEN_PASS_DEF_HEURISTICMAPPINGPASS
#include "mlir/Passes/Passes.h.inc"

struct HeuristicMappingPass
    : impl::HeuristicMappingPassBase<HeuristicMappingPass> {
private:
  /**
   * @brief A quantum accelerator's architecture.
   * @details Computes all-shortest paths at construction.
   */
  class Architecture {
  public:
    using CouplingSet = mlir::DenseSet<std::pair<uint32_t, uint32_t>>;
    using NeighbourVector = mlir::SmallVector<mlir::SmallVector<uint32_t, 4>>;

    explicit Architecture(std::string name, std::size_t nqubits,
                          CouplingSet couplingSet)
        : name_(std::move(name)), nqubits_(nqubits),
          couplingSet_(std::move(couplingSet)), neighbours_(nqubits),
          dist_(nqubits, llvm::SmallVector<std::size_t>(nqubits, UINT64_MAX)),
          prev_(nqubits, llvm::SmallVector<std::size_t>(nqubits, UINT64_MAX)) {
      floydWarshallWithPathReconstruction();
      collectNeighbours();
    }

    /**
     * @brief Return the architecture's name.
     */
    [[nodiscard]] constexpr std::string_view name() const { return name_; }

    /**
     * @brief Return the architecture's number of qubits.
     */
    [[nodiscard]] constexpr std::size_t nqubits() const { return nqubits_; }

    /**
     * @brief Return true if @p u and @p v are adjacent.
     */
    [[nodiscard]] bool areAdjacent(uint32_t u, uint32_t v) const {
      return couplingSet_.contains({u, v});
    }

    /**
     * @brief Collect the shortest SWAP sequence to make @p u and @p v adjacent.
     * @returns The SWAP sequence from the destination (v) to source (u) qubit.
     */
    [[nodiscard]] llvm::SmallVector<std::pair<uint32_t, uint32_t>>
    shortestSWAPsBetween(uint32_t u, uint32_t v) const {
      if (u == v) {
        return {};
      }

      if (prev_[u][v] == UINT64_MAX) {
        throw std::domain_error("No path between qubits " + std::to_string(u) +
                                " and " + std::to_string(v));
      }

      llvm::SmallVector<std::pair<uint32_t, uint32_t>> swaps;
      uint32_t last = v;
      uint32_t curr = prev_[u][v];

      while (curr != u) {
        swaps.emplace_back(last, curr); // Insert SWAP(last, curr).
        last = curr;
        curr = prev_[u][curr];
      }

      return swaps;
    }

    /**
     * @brief Return the length of the shortest path between @p u and @p v.
     */
    [[nodiscard]] std::size_t distanceBetween(uint32_t u, uint32_t v) const {
      if (dist_[u][v] == UINT64_MAX) {
        throw std::domain_error("No path between qubits " + std::to_string(u) +
                                " and " + std::to_string(v));
      }
      return dist_[u][v];
    }

    /**
     * @brief Collect all neighbours of @p u.
     */
    [[nodiscard]] llvm::SmallVector<uint32_t, 4>
    neighboursOf(uint32_t u) const {
      return neighbours_[u];
    }

  private:
    using Matrix = llvm::SmallVector<llvm::SmallVector<std::size_t>>;

    /**
     * @brief Find all shortest paths in the coupling map between two qubits.
     * @details Vertices are the qubits. Edges connected two qubits. Has a time
     * and memory complexity of O(nqubits^3) and O(nqubits^2), respectively.
     * @link Adapted from https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm
     */
    void floydWarshallWithPathReconstruction() {
      for (const auto& [u, v] : couplingSet_) {
        dist_[u][v] = 1;
        prev_[u][v] = u;
      }
      for (std::size_t v = 0; v < nqubits(); ++v) {
        dist_[v][v] = 0;
        prev_[v][v] = v;
      }

      for (std::size_t k = 0; k < nqubits(); ++k) {
        for (std::size_t i = 0; i < nqubits(); ++i) {
          for (std::size_t j = 0; j < nqubits(); ++j) {
            if (dist_[i][k] == UINT64_MAX || dist_[k][j] == UINT64_MAX) {
              continue; // avoid overflow with "infinite" distances
            }
            const std::size_t sum = dist_[i][k] + dist_[k][j];
            if (dist_[i][j] > sum) {
              dist_[i][j] = sum;
              prev_[i][j] = prev_[k][j];
            }
          }
        }
      }
    }

    /**
     * @brief Collect the neighbours of all qubits.
     * @details Has a time complexity of O(nqubits)
     */
    void collectNeighbours() {
      for (const auto& [u, v] : couplingSet_) {
        neighbours_[u].push_back(v);
      }
    }

    std::string name_;
    std::size_t nqubits_;
    CouplingSet couplingSet_;
    NeighbourVector neighbours_;

    Matrix dist_;
    Matrix prev_;
  };

  /**
   * @brief A qubit layout that maps program and hardware indices without
   * storing Values. Used for efficient memory usage when Value tracking isn't
   * needed.
   *
   * Note that we use the terminology "hardware" and "program" qubits here,
   * because "virtual" (opposed to physical) and "static" (opposed to dynamic)
   * are C++ keywords.
   */
  class [[nodiscard]] ThinLayout {
  public:
    explicit ThinLayout(const std::size_t nqubits)
        : programToHardware_(nqubits), hardwareToProgram_(nqubits) {}

    /**
     * @brief Insert program:hardware index mapping.
     * @param prog The program index.
     * @param hw The hardware index.
     */
    void add(uint32_t prog, uint32_t hw) {
      assert(prog < programToHardware_.size() &&
             "add: program index out of bounds");
      assert(hw < hardwareToProgram_.size() &&
             "add: hardware index out of bounds");
      programToHardware_[prog] = hw;
      hardwareToProgram_[hw] = prog;
    }

    /**
     * @brief Look up program index for a hardware index.
     * @param hw The hardware index.
     * @return The program index of the respective hardware index.
     */
    [[nodiscard]] uint32_t getProgramIndex(const uint32_t hw) const {
      assert(hw < hardwareToProgram_.size() &&
             "getProgramIndex: hardware index out of bounds");
      return hardwareToProgram_[hw];
    }

    /**
     * @brief Look up hardware index for a program index.
     * @param prog The program index.
     * @return The hardware index of the respective program index.
     */
    [[nodiscard]] uint32_t getHardwareIndex(const uint32_t prog) const {
      assert(prog < programToHardware_.size() &&
             "getHardwareIndex: program index out of bounds");
      return programToHardware_[prog];
    }

    /**
     * @brief Convenience function to lookup multiple hardware indices at once.
     * @param progs The program indices.
     * @return A tuple of hardware indices.
     */
    template <typename... ProgIndices>
      requires(sizeof...(ProgIndices) > 0) &&
              ((std::is_convertible_v<ProgIndices, uint32_t>) && ...)
    [[nodiscard]] auto getHardwareIndices(ProgIndices... progs) const {
      return std::tuple{getHardwareIndex(static_cast<uint32_t>(progs))...};
    }

    /**
     * @brief Convenience function to lookup multiple program indices at once.
     * @param hws The hardware indices.
     * @return A tuple of program indices.
     */
    template <typename... HwIndices>
      requires(sizeof...(HwIndices) > 0) &&
              ((std::is_convertible_v<HwIndices, uint32_t>) && ...)
    [[nodiscard]] auto getProgramIndices(HwIndices... hws) const {
      return std::tuple{getProgramIndex(static_cast<uint32_t>(hws))...};
    }

    /**
     * @brief Swap the mapping to hardware indices of two program indices.
     */
    void swap(const uint32_t prog0, const uint32_t prog1) {
      const uint32_t hw0 = programToHardware_[prog0];
      const uint32_t hw1 = programToHardware_[prog1];

      std::swap(programToHardware_[prog0], programToHardware_[prog1]);
      std::swap(hardwareToProgram_[hw0], hardwareToProgram_[hw1]);
    }

    /**
     * @returns the number of qubits handled by the layout.
     */
    [[nodiscard]] std::size_t getNumQubits() const {
      return programToHardware_.size();
    }

  protected:
    /**
     * @brief Maps a program qubit index to its hardware index.
     */
    mlir::SmallVector<uint32_t> programToHardware_;

    /**
     * @brief Maps a hardware qubit index to its program index.
     */
    mlir::SmallVector<uint32_t> hardwareToProgram_;
  };

  struct Circuit {
  public:
    Circuit() = default;

    void extend(TypedValue<QubitType> q) { qubits_.emplace_back(q); }

    template <typename Range> void extend(Range&& range) {
      llvm::append_range(qubits_, std::forward<Range>(range));
    }

    /// @returns the i-th qubit of the circuit.
    [[nodiscard]] TypedValue<QubitType> qubit(std::size_t i) const {
      return qubits_[i];
    }

    /// @returns the number of qubits the circuit contains.
    [[nodiscard]] std::size_t size() const { return qubits_.size(); }

    /// @brief Assign hardware index to qubit.
    void setHardwareIndex(const TypedValue<QubitType> q,
                          const std::size_t hwIndex) {
      mapping_.try_emplace(q, hwIndex);
    }

    /// @returns the hardware index associated with the qubit value.
    [[nodiscard]] std::size_t
    getHardwareIndex(const TypedValue<QubitType> q) const {
      return mapping_.at(q);
    }

    /// @returns a view of the qubits.
    [[nodiscard]] ArrayRef<TypedValue<QubitType>> qubits() const {
      return qubits_;
    }

    /// @brief Replace dynamic qubits with static qubits.
    // void staticize(const SmallVector<QubitMapping>& mapping,
    //                IRRewriter& rewriter) {
    //   for (const auto [i, m] : llvm::enumerate(mapping)) {
    //     Operation* alloc = m.qubit().getDefiningOp();
    //     assert(llvm::isa<AllocOp>(alloc));

    //     rewriter.setInsertionPoint(alloc);
    //     qubits_[i] = rewriter.replaceOpWithNewOp<StaticOp>(alloc, m.index());
    //   }
    // }

  private:
    /// @brief The qubits of the circuit.
    SmallVector<TypedValue<QubitType>, 32> qubits_;
    /// @brief Maps qubit values to hardware indices.
    DenseMap<TypedValue<QubitType>, std::size_t> mapping_;
  };

public:
  using HeuristicMappingPassBase::HeuristicMappingPassBase;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());

    static const Architecture::CouplingSet COUPLING{
        {0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 3}, {3, 1}, {2, 3},
        {3, 2}, {2, 4}, {4, 2}, {3, 5}, {5, 3}, {4, 5}, {5, 4}};

    Architecture arch("RigettiNovera", 9, COUPLING);

    for (auto func : getOperation().getOps<func::FuncOp>()) {
      Region& region = func.getFunctionBody();

      // Stage 1: Apply initial program-to-hardware mapping strategy.

      // Find circuit qubits (via their allocations).
      Circuit circ;
      circ.extend(map_range(region.getOps<AllocOp>(),
                            [](AllocOp op) { return op.getResult(); }));

      // Compute layers.
      SmallVector<WireIterator> wires;
      wires.reserve(circ.size());
      for (const auto q : circ.qubits()) {
        wires.emplace_back(q);
      }
      SmallVector<WireIterator> wireStarts(wires);

      const auto forwardLayers =
          getLayers<Forward>(wires, [&](std::size_t, const WireIterator& it) {
            return it != std::default_sentinel;
          });
      const auto backwardLayers = getLayers<Backward>(
          wires, [&](std::size_t idx, const WireIterator& it) {
            return it != wireStarts[idx];
          });

      llvm::dbgs() << "forward:\n";
      for (const auto& layer : forwardLayers) {
        layer.dump();
        llvm::dbgs() << " --- layer end ---\n";
      }

      llvm::dbgs() << "backward:\n";
      for (const auto& layer : backwardLayers) {
        layer.dump();
        llvm::dbgs() << " --- layer end ---\n";
      }

      ThinLayout layout(8);

      // Apply identity layout.
      for (std::size_t i = 0; i < circ.size(); ++i) {
        layout.add(i, i);
      }

      // Stage 2: Recomputing starting program-to-hardware mapping by
      // repeating forwards and backwards traversals.
      const std::size_t repeats = 1;
      for (std::size_t i = 0; i < repeats; ++i) {
        // forward(circ, arch, layout);
        // mapping = backward(mapping)
      }

      // Stage 3: Apply mapping and final traversal.
      // circ.staticize(mapping, rewriter);
      // forward(circ, arch, true);
    }
  }

private:
  struct Layer {
    /// @brief Set of two-qubit gates.
    DenseSet<std::pair<std::size_t, std::size_t>> gates;

#ifndef NDEBUG
    LLVM_DUMP_METHOD void dump(llvm::raw_ostream& os = llvm::dbgs()) const {
      os << "gates= ";
      for (const auto [i1, i2] : gates) {
        os << "(" << i1 << ", " << i2 << ") ";
      }
      os << "\n";
    }
#endif
  };

  enum Direction : std::uint8_t { Forward, Backward };

  template <Direction d, typename EndCheckF>
  [[nodiscard]] static SmallVector<Layer>
  getLayers(MutableArrayRef<WireIterator> wires, EndCheckF endCheck) {
    constexpr std::size_t step = d == Forward ? 1 : -1;

    SmallVector<Layer> layers;
    DenseMap<UnitaryOpInterface, std::size_t> occ;

    while (true) {
      Layer l;
      // SetVector<TypedValue<QubitType>> ssaFront;

      for (const auto [index, it] : llvm::enumerate(wires)) {
        while (endCheck(index, it)) {
          const auto res =
              TypeSwitch<Operation*, WalkResult>(it.operation())
                  .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
                    assert(op.getNumQubits() > 0 && op.getNumQubits() <= 2);

                    if (op.getNumQubits() == 1) {
                      std::ranges::advance(it, step);
                      return WalkResult::advance();
                    }

                    // ssaFront.insert(
                    //     cast<TypedValue<QubitType>>(std::prev(it).qubit()));

                    if (occ.contains(op)) {
                      const auto otherIndex = occ[op];
                      l.gates.insert(std::make_pair(index, otherIndex));
                      occ.erase(op);
                    } else {
                      occ.try_emplace(op, index);
                    }

                    return WalkResult::interrupt();
                  })
                  .template Case<AllocOp, StaticOp, ResetOp, MeasureOp,
                                 DeallocOp>([&](auto) {
                    std::ranges::advance(it, step);
                    return WalkResult::advance();
                  })
                  .Default([&](Operation* op) {
                    report_fatal_error("unknown op in wireiterator use: " +
                                       op->getName().getStringRef());
                    return WalkResult::interrupt();
                  });

          if (res.wasInterrupted()) {
            break;
          }
        }
      }

      // If there are no more two-qubit gates, early exit.
      if (l.gates.empty()) {
        break;
      }

      for (auto [i1, i2] : l.gates) {
        std::ranges::advance(wires[i1], step);
        std::ranges::advance(wires[i2], step);
      }

      layers.emplace_back(l);
    }

    return layers;
  }

  static ThinLayout getIdentityLayout(const Architecture& arch) {
    ThinLayout layout(8);
    for (std::size_t i = 0; i < arch.nqubits(); ++i) {
      layout.add(i, i);
    }
    return layout;
  }

  static void forward(const Circuit& circ, const Architecture& arch,
                      ThinLayout layout) {}
};
} // namespace mlir::qco

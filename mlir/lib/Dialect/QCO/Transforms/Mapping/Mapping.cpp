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
#include "mlir/Dialect/QCO/Transforms/Mapping/Architecture.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Threading.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#define DEBUG_TYPE "mapping-pass"

namespace mlir::qco {

#define GEN_PASS_DEF_MAPPINGPASS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

LogicalResult isExecutable(Region& region, const Architecture& arch) {
  bool executable = true;
  walkUnit(region, [&](Operation* op, Qubits& qubits) {
    if (auto u = dyn_cast<UnitaryOpInterface>(op)) {
      if (isa<BarrierOp>(u)) {
        return WalkResult::advance();
      }
      if (u.getNumQubits() > 1) {
        const auto q0 = cast<TypedValue<QubitType>>(u.getInputQubit(0));
        const auto q1 = cast<TypedValue<QubitType>>(u.getInputQubit(1));
        const auto i0 = qubits.getHardwareIndex(q0);
        const auto i1 = qubits.getHardwareIndex(q1);
        if (!arch.areAdjacent(i0, i1)) {
          llvm::dbgs() << "not adjacent: " << i0 << " and " << i1 << '\n';
          llvm::dbgs() << op->getLoc() << '\n';
          executable = false;
          return WalkResult::interrupt();
        }
      }
    }

    return WalkResult::advance();
  });

  if (executable) {
    return success();
  }

  return failure();
}

namespace {

struct MappingPass : impl::MappingPassBase<MappingPass> {
private:
  using QubitValue = TypedValue<QubitType>;
  using IndexType = std::size_t;
  using IndexGate = std::pair<IndexType, IndexType>;
  using IndexGateSet = DenseSet<IndexGate>;

  class LayerRef {
  public:
    LayerRef(ArrayRef<UnitaryOpInterface> ops, ArrayRef<IndexGate> indices)
        : ops_(ops), indices_(indices) {}

    /**
     * @returns the operations of the two-qubit gates.
     */
    ArrayRef<UnitaryOpInterface> operations() const { return ops_; }

    /**
     * @returns the program indices of the two-qubit gates.
     */
    ArrayRef<IndexGate> indices() const { return indices_; }

  private:
    ArrayRef<UnitaryOpInterface> ops_;
    ArrayRef<IndexGate> indices_;
  };

  class Layer {
  public:
    /**
     * @returns the operations of the two-qubit gates.
     */
    ArrayRef<UnitaryOpInterface> operations() const { return ops_; }

    /**
     * @returns the program indices of the two-qubit gates.
     */
    ArrayRef<IndexGate> indices() const { return indices_; }

    /**
     * @brief Add a two-qubit gate to the layer.
     */
    void addGate(UnitaryOpInterface op, IndexGate indicesOfGate) {
      ops_.emplace_back(op);
      indices_.emplace_back(indicesOfGate);
    }

    /**
     * @returns the amount of two-qubit gates.
     */
    [[nodiscard]] std::size_t size() const {
      assert(ops_.size() == indices_.size());
      return ops_.size();
    }

    /**
     * @returns true if the amount of two-qubit gates is zero.
     */
    [[nodiscard]] bool empty() const {
      assert(ops_.size() == indices_.size());
      return ops_.empty();
    }

    /**
     * @brief Chop off the first N layers, and keep M layers.
     */
    [[nodiscard]] LayerRef slice(std::size_t n, std::size_t m) const {
      return {operations().slice(n, m), indices().slice(n, m)};
    }

  private:
    SmallVector<UnitaryOpInterface> ops_;
    SmallVector<IndexGate> indices_;
  };

  /**
   * @brief Specifies the layering direction.
   */
  enum class Direction : std::uint8_t { Forward, Backward };

  class LayoutInfo;

  /**
   * @brief A qubit layout that maps program and hardware indices without
   * storing Values. Used for efficient memory usage when Value tracking isn't
   * needed.
   *
   * Note that we use the terminology "hardware" and "program" qubits here,
   * because "virtual" (opposed to physical) and "static" (opposed to dynamic)
   * are C++ keywords.
   */
  class [[nodiscard]] Layout {
  public:
    /**
     * @brief Constructs the identity (i->i) layout.
     * @param nqubits The number of qubits.
     * @return The identity layout.
     */
    static Layout identity(const std::size_t nqubits) {
      Layout layout(nqubits);
      for (std::size_t i = 0; i < nqubits; ++i) {
        layout.add(i, i);
      }
      return layout;
    }

    /**
     * @brief Constructs a random layout.
     * @param nqubits The number of qubits.
     * @param seed A seed for randomization.
     * @return The random layout.
     */
    static Layout random(const std::size_t nqubits, const std::size_t seed) {
      SmallVector<IndexType> mapping(nqubits);
      std::iota(mapping.begin(), mapping.end(), IndexType{0});
      std::ranges::shuffle(mapping, std::mt19937_64{seed});

      Layout layout(nqubits);
      for (const auto [prog, hw] : enumerate(mapping)) {
        layout.add(prog, hw);
      }

      return layout;
    }

    /**
     * @brief Constructs a layer-fitted layout.
     * @param layers The layers of the circuit.
     * @param arch The targeted architecture.
     * @param seed A seed for randomization.
     * @return The fitted layout.
     */
    static Layout fit(ArrayRef<Layer> layers, const Architecture& arch,
                      const std::size_t seed) {
      std::mt19937_64 gen{seed};

      Layout layout(arch.nqubits());
      SetVector<IndexType> freeProgram;
      SetVector<IndexType> freeHardware;

      for (IndexType i = 0; i < arch.nqubits(); ++i) {
        freeProgram.insert(i);
        freeHardware.insert(i);
      }

      const auto drawHardwareIndex = [&]() {
        const std::size_t n = freeHardware.size() - 1;
        std::uniform_int_distribution<IndexType> distr(0, n);
        const auto index = distr(gen);
        const auto hw = freeHardware[index];
        freeHardware.remove(hw);
        return hw;
      };

      const auto drawClosestHardwareIndex = [&](const IndexType hwRef) {
        std::size_t bestDistance = std::numeric_limits<std::size_t>::max();

        SmallVector<IndexType> nearest;
        nearest.reserve(freeHardware.size());

        for (const auto hw : freeHardware) {
          const auto dist = arch.distanceBetween(hwRef, hw);
          if (dist < bestDistance) {
            bestDistance = dist;
            nearest.clear();
            nearest.emplace_back(hw);
          } else if (dist == bestDistance) {
            nearest.emplace_back(hw);
          }
        }

        std::uniform_int_distribution<IndexType> distr(0, nearest.size() - 1);
        const auto hw = nearest[distr(gen)];
        freeHardware.remove(hw);
        return hw;
      };

      const auto getRandomNeighbour =
          [&](const IndexType hw) -> std::optional<IndexType> {
        SmallVector<IndexType> neighbours;
        for (const auto hwN : arch.neighboursOf(hw)) {
          if (freeHardware.contains(hwN)) {
            neighbours.emplace_back(hwN);
          }
        }

        if (neighbours.empty()) {
          return std::nullopt;
        }

        std::uniform_int_distribution<IndexType> distr(0,
                                                       neighbours.size() - 1);
        const auto hwN = neighbours[distr(gen)];
        freeHardware.remove(hwN);
        return hwN;
      };

      const auto findOther = [&](const std::size_t progA,
                                 const std::size_t progB) {
        const auto hwA = layout.getHardwareIndex(progA);

        // Get random neighbour.
        if (const auto opt = getRandomNeighbour(hwA)) {
          layout.add(progB, *opt);
        } else {
          // If no random neighbour is available, draw nearest.
          layout.add(progB, drawClosestHardwareIndex(hwA));
        }
        freeProgram.remove(progB);
      };

      for (const auto& layer : layers) {
        for (const auto& [prog0, prog1] : layer.indices()) {
          if (freeProgram.contains(prog0) && freeProgram.contains(prog1)) {
            const auto hw0 = drawHardwareIndex();
            layout.add(prog0, hw0);
            freeProgram.remove(prog0);

            // Get random neighbour.
            if (const auto opt = getRandomNeighbour(hw0)) {
              layout.add(prog1, *opt);
              freeProgram.remove(prog1);
              continue;
            }

            // If no random neighbour is available, draw nearest.
            layout.add(prog1, drawClosestHardwareIndex(hw0));
            freeProgram.remove(prog1);
          } else if (freeProgram.contains(prog0) &&
                     !freeProgram.contains(prog1)) {
            findOther(prog1, prog0);
          } else if (!freeProgram.contains(prog0) &&
                     freeProgram.contains(prog1)) {
            findOther(prog0, prog1);
          }

          // Both already mapped. Nothing to do.
        }
      }

      for (const auto& [prog, hw] : zip_equal(freeProgram, freeHardware)) {
        layout.add(prog, hw);
      }

      return layout;
    }

    /**
     * @brief Insert program:hardware index mapping.
     * @param prog The program index.
     * @param hw The hardware index.
     */
    void add(IndexType prog, IndexType hw) {
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
    [[nodiscard]] IndexType getProgramIndex(const IndexType hw) const {
      assert(hw < hardwareToProgram_.size() &&
             "getProgramIndex: hardware index out of bounds");
      return hardwareToProgram_[hw];
    }

    /**
     * @brief Look up hardware index for a program index.
     * @param prog The program index.
     * @return The hardware index of the respective program index.
     */
    [[nodiscard]] IndexType getHardwareIndex(const IndexType prog) const {
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
              ((std::is_convertible_v<ProgIndices, IndexType>) && ...)
    [[nodiscard]] auto getHardwareIndices(ProgIndices... progs) const {
      return std::tuple{getHardwareIndex(static_cast<IndexType>(progs))...};
    }

    /**
     * @brief Convenience function to lookup multiple program indices at once.
     * @param hws The hardware indices.
     * @return A tuple of program indices.
     */
    template <typename... HwIndices>
      requires(sizeof...(HwIndices) > 0) &&
              ((std::is_convertible_v<HwIndices, std::size_t>) && ...)
    [[nodiscard]] auto getProgramIndices(HwIndices... hws) const {
      return std::tuple{getProgramIndex(static_cast<IndexType>(hws))...};
    }

    /**
     * @brief Swap the mapping to program indices of two hardware indices.
     */
    void swap(const IndexType hw0, const IndexType hw1) {
      const auto prog0 = hardwareToProgram_[hw0];
      const auto prog1 = hardwareToProgram_[hw1];

      std::swap(hardwareToProgram_[hw0], hardwareToProgram_[hw1]);
      std::swap(programToHardware_[prog0], programToHardware_[prog1]);
    }

    /**
     * @returns the number of qubits managed by the layout.
     */
    [[nodiscard]] std::size_t nqubits() const {
      return programToHardware_.size();
    }

  protected:
    /**
     * @brief Maps a program qubit index to its hardware index.
     */
    SmallVector<IndexType> programToHardware_;

    /**
     * @brief Maps a hardware qubit index to its program index.
     */
    SmallVector<IndexType> hardwareToProgram_;

  private:
    friend class MappingPass::LayoutInfo;

    Layout() = default;
    explicit Layout(const std::size_t nqubits)
        : programToHardware_(nqubits), hardwareToProgram_(nqubits) {}
  };

  /**
   * @brief Required to use Layout as a key for LLVM maps and sets.
   */
  class [[nodiscard]] LayoutInfo {
    using Info = DenseMapInfo<SmallVector<IndexType>>;

  public:
    static Layout getEmptyKey() {
      Layout l;
      l.programToHardware_ = Info::getEmptyKey();
      l.hardwareToProgram_ = Info::getEmptyKey();
      return l;
    }

    static Layout getTombstoneKey() {
      Layout l;
      l.programToHardware_ = Info::getTombstoneKey();
      l.hardwareToProgram_ = Info::getTombstoneKey();
      return l;
    }

    static unsigned getHashValue(const Layout& l) {
      return Info::getHashValue(l.programToHardware_);
    }

    static bool isEqual(const Layout& a, const Layout& b) {
      return Info::isEqual(a.programToHardware_, b.programToHardware_);
    }
  };

  /**
   * @brief Parameters influencing the behavior of the A* search algorithm.
   */
  struct [[nodiscard]] Parameters {
    Parameters(const float alpha, const float lambda,
               const std::size_t nlookahead)
        : alpha(alpha), decay(1 + nlookahead) {
      decay[0] = 1.;
      for (std::size_t i = 1; i < decay.size(); ++i) {
        decay[i] = decay[i - 1] * lambda;
      }
    }

    float alpha;
    SmallVector<float> decay;
  };

  /**
   * @brief Describes a node in the A* search graph.
   */
  struct [[nodiscard]] Node {
    struct ComparePointer {
      bool operator()(const Node* lhs, const Node* rhs) const {
        return lhs->f > rhs->f;
      }
    };

    Layout layout;
    IndexGate swap;
    Node* parent;
    std::size_t depth;
    float f;

    /**
     * @brief Construct a root node with the given layout. Initialize the
     * sequence with an empty vector and set the cost to zero.
     */
    explicit Node(Layout layout)
        : layout(std::move(layout)), parent(nullptr), depth(0), f(0) {}

    /**
     * @brief Construct a non-root node from its parent node. Apply the given
     * swap to the layout of the parent node.
     */
    Node(Node* parent, IndexGate swap, ArrayRef<LayerRef> layers,
         const Architecture& arch, const Parameters& params)
        : layout(parent->layout), swap(swap), parent(parent),
          depth(parent->depth + 1), f(0) {
      layout.swap(swap.first, swap.second);
      f = g(params.alpha) + h(layers, arch, params); // NOLINT
    }

    /**
     * @returns true if the current sequence of SWAPs makes all gates
     * executable.
     */
    [[nodiscard]] bool isGoal(const LayerRef& front,
                              const Architecture& arch) const {
      return all_of(front.indices(), [&](const IndexGate& gate) {
        return arch.areAdjacent(layout.getHardwareIndex(gate.first),
                                layout.getHardwareIndex(gate.second));
      });
    }

  private:
    /**
     * @brief Calculate the path cost for the A* search algorithm.
     *
     * The path cost function is the weighted sum of the currently required
     * SWAPs.
     */
    [[nodiscard]] float g(float alpha) const {
      return alpha * static_cast<float>(depth);
    }

    /**
     * @brief Calculate the heuristic cost for the A* search algorithm.
     *
     * Computes the minimal number of SWAPs required to route each gate in
     * each layer. For each gate, this is determined by the shortest distance
     * between its hardware qubits. Intuitively, this is the number of SWAPs
     * that a naive router would insert to route the layers (with a constant
     * layout).
     */
    [[nodiscard]] float h(ArrayRef<LayerRef> layers, const Architecture& arch,
                          const Parameters& params) const {
      float costs{0};
      for (const auto& [decay, layer] : zip(params.decay, layers)) {
        for (const auto& [prog0, prog1] : layer.indices()) {
          const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
          const std::size_t nswaps = arch.distanceBetween(hw0, hw1) - 1;
          costs += decay * static_cast<float>(nswaps);
        }
      }
      return costs;
    }
  };

  struct [[nodiscard]] TrialResult {
    explicit TrialResult(Layout layout) : layout(std::move(layout)) {}

    /// @brief The computed initial layout.
    Layout layout;
    /// @brief A vector of SWAPs for each layer.
    SmallVector<SmallVector<IndexGate>> swaps;
    /// @brief The number of inserted SWAPs.
    std::size_t nswaps{};
  };

protected:
  using MappingPassBase::MappingPassBase;

  void runOnOperation() override {
    std::mt19937_64 rng{this->seed};
    IRRewriter rewriter(&getContext());

    Parameters params(this->alpha, this->lambda, this->nlookahead);
    // TODO: Hardcoded architecture.
    Architecture arch(
        "IBMNighthawk", 120,
        {{0, 12},    {12, 0},    {0, 1},     {1, 0},     {1, 13},    {13, 1},
         {1, 2},     {2, 1},     {2, 14},    {14, 2},    {2, 3},     {3, 2},
         {3, 15},    {15, 3},    {3, 4},     {4, 3},     {4, 16},    {16, 4},
         {4, 5},     {5, 4},     {5, 17},    {17, 5},    {5, 6},     {6, 5},
         {6, 18},    {18, 6},    {6, 7},     {7, 6},     {7, 19},    {19, 7},
         {7, 8},     {8, 7},     {8, 20},    {20, 8},    {8, 9},     {9, 8},
         {9, 21},    {21, 9},    {9, 10},    {10, 9},    {10, 22},   {22, 10},
         {10, 11},   {11, 10},   {11, 23},   {23, 11},   {12, 24},   {24, 12},
         {12, 13},   {13, 12},   {13, 25},   {25, 13},   {13, 14},   {14, 13},
         {14, 26},   {26, 14},   {14, 15},   {15, 14},   {15, 27},   {27, 15},
         {15, 16},   {16, 15},   {16, 28},   {28, 16},   {16, 17},   {17, 16},
         {17, 29},   {29, 17},   {17, 18},   {18, 17},   {18, 30},   {30, 18},
         {18, 19},   {19, 18},   {19, 31},   {31, 19},   {19, 20},   {20, 19},
         {20, 32},   {32, 20},   {20, 21},   {21, 20},   {21, 33},   {33, 21},
         {21, 22},   {22, 21},   {22, 34},   {34, 22},   {22, 23},   {23, 22},
         {23, 35},   {35, 23},   {24, 36},   {36, 24},   {24, 25},   {25, 24},
         {25, 37},   {37, 25},   {25, 26},   {26, 25},   {26, 38},   {38, 26},
         {26, 27},   {27, 26},   {27, 39},   {39, 27},   {27, 28},   {28, 27},
         {28, 40},   {40, 28},   {28, 29},   {29, 28},   {29, 41},   {41, 29},
         {29, 30},   {30, 29},   {30, 42},   {42, 30},   {30, 31},   {31, 30},
         {31, 43},   {43, 31},   {31, 32},   {32, 31},   {32, 44},   {44, 32},
         {32, 33},   {33, 32},   {33, 45},   {45, 33},   {33, 34},   {34, 33},
         {34, 46},   {46, 34},   {34, 35},   {35, 34},   {35, 47},   {47, 35},
         {36, 48},   {48, 36},   {36, 37},   {37, 36},   {37, 49},   {49, 37},
         {37, 38},   {38, 37},   {38, 50},   {50, 38},   {38, 39},   {39, 38},
         {39, 51},   {51, 39},   {39, 40},   {40, 39},   {40, 52},   {52, 40},
         {40, 41},   {41, 40},   {41, 53},   {53, 41},   {41, 42},   {42, 41},
         {42, 54},   {54, 42},   {42, 43},   {43, 42},   {43, 55},   {55, 43},
         {43, 44},   {44, 43},   {44, 56},   {56, 44},   {44, 45},   {45, 44},
         {45, 57},   {57, 45},   {45, 46},   {46, 45},   {46, 58},   {58, 46},
         {46, 47},   {47, 46},   {47, 59},   {59, 47},   {48, 60},   {60, 48},
         {48, 49},   {49, 48},   {49, 61},   {61, 49},   {49, 50},   {50, 49},
         {50, 62},   {62, 50},   {50, 51},   {51, 50},   {51, 63},   {63, 51},
         {51, 52},   {52, 51},   {52, 64},   {64, 52},   {52, 53},   {53, 52},
         {53, 65},   {65, 53},   {53, 54},   {54, 53},   {54, 66},   {66, 54},
         {54, 55},   {55, 54},   {55, 67},   {67, 55},   {55, 56},   {56, 55},
         {56, 68},   {68, 56},   {56, 57},   {57, 56},   {57, 69},   {69, 57},
         {57, 58},   {58, 57},   {58, 70},   {70, 58},   {58, 59},   {59, 58},
         {59, 71},   {71, 59},   {60, 72},   {72, 60},   {60, 61},   {61, 60},
         {61, 73},   {73, 61},   {61, 62},   {62, 61},   {62, 74},   {74, 62},
         {62, 63},   {63, 62},   {63, 75},   {75, 63},   {63, 64},   {64, 63},
         {64, 76},   {76, 64},   {64, 65},   {65, 64},   {65, 77},   {77, 65},
         {65, 66},   {66, 65},   {66, 78},   {78, 66},   {66, 67},   {67, 66},
         {67, 79},   {79, 67},   {67, 68},   {68, 67},   {68, 80},   {80, 68},
         {68, 69},   {69, 68},   {69, 81},   {81, 69},   {69, 70},   {70, 69},
         {70, 82},   {82, 70},   {70, 71},   {71, 70},   {71, 83},   {83, 71},
         {72, 84},   {84, 72},   {72, 73},   {73, 72},   {73, 85},   {85, 73},
         {73, 74},   {74, 73},   {74, 86},   {86, 74},   {74, 75},   {75, 74},
         {75, 87},   {87, 75},   {75, 76},   {76, 75},   {76, 88},   {88, 76},
         {76, 77},   {77, 76},   {77, 89},   {89, 77},   {77, 78},   {78, 77},
         {78, 90},   {90, 78},   {78, 79},   {79, 78},   {79, 91},   {91, 79},
         {79, 80},   {80, 79},   {80, 92},   {92, 80},   {80, 81},   {81, 80},
         {81, 93},   {93, 81},   {81, 82},   {82, 81},   {82, 94},   {94, 82},
         {82, 83},   {83, 82},   {83, 95},   {95, 83},   {84, 96},   {96, 84},
         {84, 85},   {85, 84},   {85, 97},   {97, 85},   {85, 86},   {86, 85},
         {86, 98},   {98, 86},   {86, 87},   {87, 86},   {87, 99},   {99, 87},
         {87, 88},   {88, 87},   {88, 100},  {100, 88},  {88, 89},   {89, 88},
         {89, 101},  {101, 89},  {89, 90},   {90, 89},   {90, 102},  {102, 90},
         {90, 91},   {91, 90},   {91, 103},  {103, 91},  {91, 92},   {92, 91},
         {92, 104},  {104, 92},  {92, 93},   {93, 92},   {93, 105},  {105, 93},
         {93, 94},   {94, 93},   {94, 106},  {106, 94},  {94, 95},   {95, 94},
         {95, 107},  {107, 95},  {96, 108},  {108, 96},  {96, 97},   {97, 96},
         {97, 109},  {109, 97},  {97, 98},   {98, 97},   {98, 110},  {110, 98},
         {98, 99},   {99, 98},   {99, 111},  {111, 99},  {99, 100},  {100, 99},
         {100, 112}, {112, 100}, {100, 101}, {101, 100}, {101, 113}, {113, 101},
         {101, 102}, {102, 101}, {102, 114}, {114, 102}, {102, 103}, {103, 102},
         {103, 115}, {115, 103}, {103, 104}, {104, 103}, {104, 116}, {116, 104},
         {104, 105}, {105, 104}, {105, 117}, {117, 105}, {105, 106}, {106, 105},
         {106, 118}, {118, 106}, {106, 107}, {107, 106}, {107, 119}, {119, 107},
         {108, 109}, {109, 108}, {109, 110}, {110, 109}, {110, 111}, {111, 110},
         {111, 112}, {112, 111}, {112, 113}, {113, 112}, {113, 114}, {114, 113},
         {114, 115}, {115, 114}, {115, 116}, {116, 115}, {116, 117}, {117, 116},
         {117, 118}, {118, 117}, {118, 119}, {119, 118}});

    for (auto func : getOperation().getOps<func::FuncOp>()) {
      const auto qubits = collectDynamicQubits(func.getFunctionBody());
      if (qubits.size() > arch.nqubits()) {
        func.emitError() << "the targeted architecture supports "
                         << arch.nqubits() << " qubits, got " << qubits.size();
        signalPassFailure();
        return;
      }

      const auto [ltr, rtl] = computeBidirectionalLayers(qubits);
      const auto ltrRef = splitLayers(ltr, 1);
      const auto rtlRef = splitLayers(rtl, 1);

      // Create trials. Currently this includes `ntrials` many random layouts.

      SmallVector<Layout> trials;
      trials.reserve(this->ntrials);
      for (std::size_t i = 0; i < this->ntrials; ++i) {
        trials.emplace_back(Layout::fit(ltr, arch, rng()));
      }

      // Execute each of the trials (possibly in parallel). Collect the results
      // and find the one with the fewest SWAPs.

      SmallVector<std::optional<TrialResult>> results(trials.size());
      parallelForEach(
          &getContext(), enumerate(trials), [&, this](auto indexedTrial) {
            auto [idx, layout] = indexedTrial;
            auto res = runMappingTrial(ltrRef, rtlRef, arch, params, layout);
            if (succeeded(res)) {
              results[idx] = std::move(*res);
            }
          });

      TrialResult* best = findBestTrial(results);
      if (best == nullptr) {
        signalPassFailure();
        return;
      }

      place(qubits, best->layout, func.getFunctionBody(), rewriter);
      commit(ltrRef, best->swaps, func.getFunctionBody(), arch, rewriter);

      assert(isExecutable(func.getFunctionBody(), arch).succeeded());
    }
  }

private:
  /**
   * @brief Find the best trial result in terms of the number of SWAPs.
   * @returns the best trial result or nullptr if no result is valid.
   */
  [[nodiscard]] static TrialResult*
  findBestTrial(MutableArrayRef<std::optional<TrialResult>> results) {
    TrialResult* best = nullptr;
    for (auto& opt : results) {
      if (opt.has_value()) {
        if (best == nullptr || best->nswaps > opt->nswaps) {
          best = &opt.value();
        }
      }
    }
    return best;
  }

  /**
   * @brief Run a mapping trial.
   * @details Use the SABRE Approach to improve the initial layout:
   * Traverse the layers from left-to-right-to-left and cold-route
   * along the way. Repeat this procedure "niterations" times.
   * @returns the trial result or failure() on failure.
   */
  FailureOr<TrialResult> runMappingTrial(ArrayRef<LayerRef> ltr,
                                         ArrayRef<LayerRef> rtl,
                                         const Architecture& arch,
                                         const Parameters& params,
                                         Layout& layout) {
    // Perform forwards and backwards traversals.
    for (std::size_t i = 0; i < this->niterations; ++i) {
      if (failed(route(ltr, arch, params, layout, [](const auto&) {}))) {
        return failure();
      }
      if (failed(route(rtl, arch, params, layout, [](const auto&) {}))) {
        return failure();
      }
    }

    TrialResult result(layout); // Copies the final initial layout.

    // Helper function that adds the SWAPs to the trial result.
    const auto collectSwaps = [&](ArrayRef<IndexGate> swaps) {
      result.nswaps += swaps.size();
      result.swaps.emplace_back(swaps);
    };

    // Perform final left-to-right traversal whilst collecting SWAPs.
    if (failed(route(ltr, arch, params, layout, collectSwaps))) {
      return failure();
    }

    return result;
  }

  /**
   * @brief Collect dynamic qubits contained in the given function body.
   * @returns a vector of SSA values produced by qco.alloc operations.
   */
  [[nodiscard]] static SmallVector<QubitValue>
  collectDynamicQubits(Region& funcBody) {
    return SmallVector<QubitValue>(map_range(
        funcBody.getOps<AllocOp>(), [](AllocOp op) { return op.getResult(); }));
  }

  /**
   * @brief Collect static qubits contained in the given function body.
   * @returns a vector of SSA values produced by qco.static operations.
   */
  [[nodiscard]] static SmallVector<QubitValue>
  collectStaticQubits(Region& funcBody, const Architecture& arch) {
    SmallVector<QubitValue> qubits(arch.nqubits());
    for (StaticOp op : funcBody.getOps<StaticOp>()) {
      qubits[op.getIndex()] = op;
    }
    return qubits;
  }

  /**
   * @brief Computes forwards and backwards layers.
   * @returns a pair of vectors of layers, where [0]=forward and [1]=backward.
   */
  [[nodiscard]] static std::pair<SmallVector<Layer>, SmallVector<Layer>>
  computeBidirectionalLayers(ArrayRef<QubitValue> qubits) {
    auto wires = toWires(qubits);
    const auto ltr = collectLayers<Direction::Forward>(wires);
    const auto rtl = collectLayers<Direction::Backward>(wires);
    return std::make_pair(ltr, rtl);
  }

  /**
   * @brief Perform A* search to find a sequence of SWAPs that makes the
   * two-qubit operations inside the first layer (the front) executable.
   * @details
   * The iteration budget is b^{3} node expansions, i.e. roughly a depth-3
   * search in a tree with branching factor b. A hard cap prevents impractical
   * runtimes on larger architectures.
   *
   * The branching factor b of the A* search is the product of the
   * architecture's maximum qubit degree and the maximum number of two-qubit
   * gates in any layer:
   *
   * b = maxDegree × ⌈N/2⌉
   *
   * @returns a vector of hardware-index pairs (each denoting a SWAP) or
   * failure() if A* fails.
   */
  [[nodiscard]] static FailureOr<SmallVector<IndexGate>>
  search(ArrayRef<LayerRef> layers, const Layout& layout,
         const Architecture& arch, const Parameters& params) {
    constexpr std::size_t cap = 25'000'000UL;
    const std::size_t b = arch.maxDegree() * ((arch.nqubits() + 1) / 2);
    const std::size_t budget = std::min(b * b * b, cap);

    llvm::SpecificBumpPtrAllocator<Node> arena;
    std::priority_queue<Node*, std::vector<Node*>, Node::ComparePointer>
        frontier;

    Node* root = std::construct_at(arena.Allocate(), layout);
    if (root->isGoal(layers.front(), arch)) {
      return SmallVector<IndexGate>{};
    }
    frontier.emplace(root);

    DenseMap<Layout, std::size_t, LayoutInfo> bestDepth;
    DenseSet<IndexGate> expansionSet;

    std::size_t i = 0;
    while (!frontier.empty() && i < budget) {
      Node* curr = frontier.top();
      frontier.pop();

      // Multiple sequences of SWAPs can lead to the same layout and the same
      // layout creates the same child-nodes. Thus, if we've seen a layout
      // already at a lower depth don't reexpand the current node (and hence
      // recreate the same child nodes).

      const auto [it, inserted] =
          bestDepth.try_emplace(curr->layout, curr->depth);
      if (!inserted) {
        const auto otherDepth = it->getSecond();
        if (curr->depth >= otherDepth) {
          ++i;
          continue;
        }

        it->second = curr->depth;
      }

      // If the currently visited node is a goal node, reconstruct the sequence
      // of SWAPs from this node to the root.

      if (curr->isGoal(layers.front(), arch)) {
        SmallVector<IndexGate> seq(curr->depth);
        std::size_t j = seq.size() - 1;
        for (Node* n = curr; n->parent != nullptr; n = n->parent) {
          seq[j] = n->swap;
          --j;
        }
        return seq;
      }

      // Given a layout, create child-nodes for each possible SWAP
      // between two neighbouring hardware qubits.

      expansionSet.clear();
      for (const IndexGate& gate : layers.front().indices()) {
        for (const auto prog : {gate.first, gate.second}) {
          const auto hw0 = curr->layout.getHardwareIndex(prog);
          for (const auto hw1 : arch.neighboursOf(hw0)) {
            // Ensure consistent hashing/comparison.
            const IndexGate swap = std::minmax(hw0, hw1);
            if (!expansionSet.insert(swap).second) {
              continue;
            }

            frontier.emplace(std::construct_at(arena.Allocate(), curr, swap,
                                               layers, arch, params));
          }
        }
      }

      ++i;
    }

    return failure();
  }

  /**
   * @brief Transform a range of qubit values to a vector of wire iterators.
   * @returns a vector of wire iterators.
   */
  template <typename QubitRange>
  static SmallVector<WireIterator> toWires(QubitRange qubits) {
    return SmallVector<WireIterator>(
        map_range(qubits, [](auto q) { return WireIterator(q); }));
  }

  /**
   * @returns true if the wire iterator has not reached the end (Forward) or the
   * start (Backward) of the wire.
   */
  template <Direction d> static bool proceedOnWire(const WireIterator& it) {
    if constexpr (d == Direction::Forward) {
      return it != std::default_sentinel;
    } else {
      return !isa<AllocOp>(it.operation());
    }
  }

  /**
   * @brief Split each layer into sub layers.
   * @param layers The layers to split.
   * @param sz The maximum size of the sub layers.
   */
  static SmallVector<LayerRef> splitLayers(ArrayRef<Layer> layers,
                                           const std::size_t sz) {
    SmallVector<LayerRef> refs;
    for (const auto& layer : layers) {
      for (std::size_t i = 0; i < layer.indices().size(); i += sz) {
        const auto effSz = std::min(sz, layer.size() - i);
        refs.emplace_back(layer.slice(i, effSz));
      }
    }
    return refs;
  }

  /**
   * @brief Skip the next two-qubit block of two wires.
   * @details Advances each of the two wire iterators until a two-qubit op is
   * found. If the ops match, repeat this process. Otherwise, stop.
   */
  template <Direction d>
  static void skipTwoQubitBlock(WireIterator& first, WireIterator& second) {
    constexpr auto step = d == Direction::Forward ? 1 : -1;

    const auto advanceUntilTwoQubitOp = [&](WireIterator& it) {
      while (proceedOnWire<d>(it)) {
        if (auto op = dyn_cast<UnitaryOpInterface>(it.operation())) {
          if (op.getNumQubits() > 1) {
            break;
          }
        }

        std::ranges::advance(it, step);
      }
    };

    while (true) {
      advanceUntilTwoQubitOp(first);
      advanceUntilTwoQubitOp(second);

      if (!proceedOnWire<d>(first) || !proceedOnWire<d>(second)) {
        break;
      }

      if (first.operation() != second.operation()) {
        break;
      }

      std::ranges::advance(first, step);
      std::ranges::advance(second, step);
    }
  }

  /**
   * @brief Collect the layers of independently executable two-qubit gates of a
   * circuit.
   * @details Depending on the template parameter, the function collects the
   * layers in forward or backward direction, respectively. Towards that end,
   * the function traverses the def-use chain of each qubit until a two-qubit
   * gate is found. If a two-qubit gate is visited twice, it is considered ready
   * and inserted into the layer. This process is repeated until no more
   * two-qubit are found anymore.
   * @returns a vector of layers.
   */
  template <Direction d>
  static SmallVector<Layer> collectLayers(MutableArrayRef<WireIterator> wires) {
    constexpr auto step = d == Direction::Forward ? 1 : -1;

    SmallVector<Layer> layers;
    DenseMap<UnitaryOpInterface, std::size_t> visited;

    while (true) {
      Layer layer{};
      for (const auto [index, it] : enumerate(wires)) {
        while (proceedOnWire<d>(it)) {
          const auto res =
              TypeSwitch<Operation*, WalkResult>(it.operation())
                  .Case<BarrierOp>([&](auto) {
                    std::ranges::advance(it, step);
                    return WalkResult::advance();
                  })
                  .template Case<UnitaryOpInterface>(
                      [&](UnitaryOpInterface op) {
                        assert(op.getNumQubits() > 0 && op.getNumQubits() <= 2);

                        if (op.getNumQubits() == 1) {
                          std::ranges::advance(it, step);
                          return WalkResult::advance();
                        }

                        if (visited.contains(op)) {
                          const auto otherIndex = visited[op];

                          layer.addGate(op, std::make_pair(index, otherIndex));

                          skipTwoQubitBlock<d>(wires[index], wires[otherIndex]);

                          visited.erase(op);
                        } else {
                          visited.try_emplace(op, index);
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

      if (layer.empty()) {
        break;
      }

      layers.emplace_back(layer);
      visited.clear();
    }

    return layers;
  }

  /**
   * @brief "Cold" routing of the given layers.
   * @details Iterates over a sliding window of layers and uses the A* search
   * engine to find a sequence of SWAPs that makes that layer executable.
   * Instead of inserting these SWAPs into the IR, this function only updates
   * (and hence modifies) the layout. The function calls the callback @p onSwaps
   * for each layer with the found sequence of SWAPs.
   * @returns failure() if A* search isn't able to find a solution.
   */
  template <typename OnSwaps>
  LogicalResult route(ArrayRef<LayerRef> layers, const Architecture& arch,
                      const Parameters& params, Layout& layout,
                      OnSwaps&& onSwaps) {
    auto&& callback = std::forward<OnSwaps>(onSwaps);

    for (std::size_t i = 0; i < layers.size(); ++i) {
      const std::size_t len = std::min(1 + nlookahead, layers.size() - i);
      const auto window = layers.slice(i, len);
      const auto swaps = search(window, layout, arch, params);
      if (failed(swaps)) {
        return failure();
      }

      for (const auto& [hw0, hw1] : *swaps) {
        layout.swap(hw0, hw1);
      }

      std::invoke(callback, *swaps);
    }

    return success();
  }

  /**
   * @brief Perform placement.
   * @details Replaces dynamic with static qubits. Extends the computation with
   * as many static qubits as the architecture supports.
   */
  static void place(ArrayRef<QubitValue> dynQubits, const Layout& layout,
                    Region& funcBody, IRRewriter& rewriter) {
    // 1. Replace existing dynamic allocations with mapped static ones.
    for (const auto [p, q] : enumerate(dynQubits)) {
      const auto hw = layout.getHardwareIndex(p);
      rewriter.setInsertionPoint(q.getDefiningOp());
      rewriter.replaceOpWithNewOp<StaticOp>(q.getDefiningOp(), hw);
    }

    // 2. Create static qubits for the remaining (unused) hardware indices.
    for (std::size_t p = dynQubits.size(); p < layout.nqubits(); ++p) {
      rewriter.setInsertionPointToStart(&funcBody.front());
      const auto hw = layout.getHardwareIndex(p);
      auto op = StaticOp::create(rewriter, rewriter.getUnknownLoc(), hw);
      rewriter.setInsertionPoint(funcBody.back().getTerminator());
      DeallocOp::create(rewriter, rewriter.getUnknownLoc(), op.getQubit());
    }
  }

  /**
   * @brief Inserts SWAPs into the IR.
   */
  void commit(ArrayRef<LayerRef> layers,
              ArrayRef<SmallVector<IndexGate>> swapsPerLayer, Region& funcBody,
              const Architecture& arch, IRRewriter& rewriter) {
    using DanglingMap =
        DenseMap<UnitaryOpInterface, SmallVector<WireIterator*>>;

    // Helper function that absorbs one-qubit unitaries.
    const auto advFront = [](WireIterator& it) {
      auto next = std::next(it);
      while (true) {
        if (isa<DeallocOp>(next.operation()) ||
            isa<MeasureOp>(next.operation()) ||
            isa<BarrierOp>(next.operation())) {
          break;
        }

        auto op = dyn_cast<UnitaryOpInterface>(next.operation());
        if (op && op.getNumQubits() > 1) {
          break;
        }

        std::ranges::advance(it, 1);
        std::ranges::advance(next, 1);
      }
    };

    // Helper function that advances past a two-qubit block.
    const auto advBlock = [&](WireIterator& first, WireIterator& second) {
      while (true) {
        std::ranges::advance(first, 1);
        std::ranges::advance(second, 1);

        advFront(first);
        advFront(second);

        // Is a >two-qubit unitary?
        auto firstOp =
            dyn_cast<UnitaryOpInterface>(std::next(first).operation());
        if (!firstOp || firstOp.getNumQubits() < 2) {
          break;
        }

        // Is a >two-qubit unitary?
        auto secondOp =
            dyn_cast<UnitaryOpInterface>(std::next(second).operation());
        if (!secondOp || secondOp.getNumQubits() < 2) {
          break;
        }

        // Both must be unitaries.
        if (isa<BarrierOp>(firstOp) || isa<BarrierOp>(secondOp)) {
          break;
        }

        // Not the same unitary, stop.
        if (firstOp.getOperation() != secondOp.getOperation()) {
          break;
        }
      }
    };

    const auto markReady = [](DanglingMap& map, UnitaryOpInterface op,
                              WireIterator& wireIt, auto&& onReady) {
      const auto [it, inserted] =
          map.try_emplace(op, SmallVector<WireIterator*>{&wireIt});
      if (!inserted) {
        it->second.emplace_back(&wireIt);
      }

      if (it->first.getNumQubits() == it->second.size()) {
        onReady(it->second);
        map.erase(it);
      }
    };

    DanglingMap map;
    auto wires = toWires(collectStaticQubits(funcBody, arch));
    for (const auto& [layer, swaps] : zip_equal(layers, swapsPerLayer)) {
      // Advance all wires to the next front of one-qubit outputs
      // (the SSA values).
      for_each(wires, advFront);

      DenseSet<Operation*> layerOps(layer.operations().begin(),
                                    layer.operations().end());

      // Apply the sequence of SWAPs and rewire the qubit SSA values.
      for (const auto& [hw0, hw1] : swaps) {
        const auto in0 = wires[hw0].qubit();
        const auto in1 = wires[hw1].qubit();

        auto op = SWAPOp::create(rewriter, rewriter.getUnknownLoc(), in0, in1);
        const auto out0 = op.getQubit0Out();
        const auto out1 = op.getQubit1Out();

        rewriter.replaceAllUsesExcept(in0, out1, op);
        rewriter.replaceAllUsesExcept(in1, out0, op);

        // Jump over the SWAPOp.
        std::ranges::advance(wires[hw0], 1);
        std::ranges::advance(wires[hw1], 1);
      }

      // Jump over "ready" gates.
      map.clear(); // Start with fresh map.
      for (auto& it : wires) {
        auto op = dyn_cast<UnitaryOpInterface>(std::next(it).operation());

        if (!op) {
          continue;
        }

        if (op.getNumQubits() < 2) {
          continue;
        }

        if (isa<BarrierOp>(op)) {
          markReady(map, op, it, [](MutableArrayRef<WireIterator*> ready) {
            for (WireIterator* it : ready) {
              std::ranges::advance(*it, 1);
            }
          });
          continue;
        }

        if (!layerOps.contains(op.getOperation())) {
          continue;
        }

        markReady(map, op, it, [&](MutableArrayRef<WireIterator*> it) {
          advBlock(*it[0], *it[1]);
        });
      }

      this->numSwaps += swaps.size();
    }

    for_each(funcBody.getBlocks(), [](Block& b) { sortTopologically(&b); });
  }
};

} // namespace

} // namespace mlir::qco

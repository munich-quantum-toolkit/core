/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/CBit/IR/CBitOps.h"

#include "mqt/Dialect/CBit/IR/CBitAttributes.h" // IWYU pragma: associated
#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Support/IntegerExpressions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

using namespace mlir;
using namespace mlir::cbit;

void AllocOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  const auto name =
      getOperation()->getAttrOfType<StringAttr>("mqt.register_name");
  setNameFn(getResult(), name ? name.getValue() : "c");
}

void LoadOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "b");
}

void ReadOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "val");
}

#include "mqt/Dialect/CBit/IR/CBitOpsDialect.cpp.inc"
#include "mqt/Dialect/CBit/IR/CBitOpsEnums.cpp.inc"

static void attachMemorySlotInterfaces(MLIRContext* context);

// shortcut: cap per-bit IR growth; raise with compile-time evidence.
static constexpr int64_t MAX_PROMOTED_REGISTER_WIDTH = 256;

void CBitDialect::initialize() {
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mqt/Dialect/CBit/IR/CBitOpsAttributes.cpp.inc"

      >();

  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mqt/Dialect/CBit/IR/CBitOpsTypes.cpp.inc"

      >();

  addOperations<
#define GET_OP_LIST
#include "mqt/Dialect/CBit/IR/CBitOps.cpp.inc"

      >();

  attachMemorySlotInterfaces(getContext());
}

#define GET_ATTRDEF_CLASSES
#include "mqt/Dialect/CBit/IR/CBitOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mqt/Dialect/CBit/IR/CBitOpsTypes.cpp.inc"

LogicalResult
RegisterType::verify(const function_ref<InFlightDiagnostic()> emitError,
                     const int64_t width) {
  if (width <= 0) {
    return emitError() << "register width must be positive";
  }
  return success();
}

void mlir::cbit::validateStaticRegisterIndex(
    Value reg, const std::variant<int64_t, Value>& index) {
  const auto type = dyn_cast<RegisterType>(reg.getType());
  if (!type) {
    llvm::reportFatalUsageError("Expected a CBit register");
  }

  const auto* constant = std::get_if<int64_t>(&index);
  if (constant == nullptr) {
    return;
  }
  if (*constant < 0) {
    llvm::reportFatalUsageError("Register index must be non-negative");
  }
  if (*constant >= type.getWidth()) {
    llvm::reportFatalUsageError("Register index is out of bounds");
  }
}

static LogicalResult verifyIndex(Operation* operation, Value registerValue,
                                 Value indexValue) {
  const auto index = getConstantIntValue(indexValue);
  if (!index) {
    return success();
  }

  if (*index < 0) {
    return operation->emitOpError("index must be non-negative");
  }

  const auto width = cast<RegisterType>(registerValue.getType()).getWidth();
  if (*index >= width) {
    return operation->emitOpError("index exceeds register width");
  }
  return success();
}

/// Find a known snapshot; a null index denotes a whole-register read.
static Value findKnownValue(Operation* read, Value reg, Value index) {
  const bool singleBit = cast<RegisterType>(reg.getType()).getWidth() == 1;
  const auto matchesIndex = [&](Value candidate) {
    return index ? isEqualConstantIntOrValue(index, candidate)
                 : singleBit && getConstantIntValue(candidate) == 0;
  };
  const bool readsWholeRegister =
      !index || (singleBit && getConstantIntValue(index) == 0);

  // Bound the work per read so long blocks cannot cause quadratic searches.
  unsigned scanned = 0;
  for (auto* previous = read->getPrevNode();
       previous != nullptr && scanned < 16;
       previous = previous->getPrevNode(), ++scanned) {
    if (auto load = dyn_cast<LoadOp>(previous)) {
      if (load.getReg() == reg && matchesIndex(load.getIndex())) {
        return load.getResult();
      }
      continue;
    }
    if (auto snapshot = dyn_cast<ReadOp>(previous)) {
      if (snapshot.getReg() == reg && readsWholeRegister) {
        return snapshot.getResult();
      }
      continue;
    }
    if (auto store = dyn_cast<StoreOp>(previous)) {
      if (store.getReg() == reg) {
        if (matchesIndex(store.getIndex())) {
          return store.getValue();
        }
        const auto storedIndex = getConstantIntValue(store.getIndex());
        const auto loadedIndex =
            index ? getConstantIntValue(index) : std::nullopt;
        if (storedIndex && loadedIndex && storedIndex != loadedIndex) {
          continue;
        }
      }
      // Distinct register arguments may refer to the same storage.
      return {};
    }
    if (auto write = dyn_cast<WriteOp>(previous)) {
      return write.getReg() == reg && readsWholeRegister ? write.getValue()
                                                         : Value{};
    }
    if (auto alloc = dyn_cast<AllocOp>(previous)) {
      if (alloc.getResult() == reg) {
        return {};
      }
      continue;
    }
    if (previous->getNumRegions() != 0 || !isMemoryEffectFree(previous)) {
      return {};
    }
  }
  return {};
}

static LogicalResult forwardKnownValue(Operation* read, Value reg, Value index,
                                       PatternRewriter& rewriter) {
  auto known = findKnownValue(read, reg, index);
  if (!known) {
    return failure();
  }
  rewriter.replaceOp(read, known);
  return success();
}

static Value readBits(OpBuilder& builder, ReadOp read,
                      function_ref<Value(int64_t)> load) {
  auto loc = read.getLoc();
  const auto type = read.getResult().getType();
  const auto width = type.getWidth();
  Value result;
  for (unsigned index = 0; index < width; ++index) {
    Value bit = load(index);
    if (width != 1) {
      bit = arith::ExtUIOp::create(builder, loc, type, bit);
    }
    if (index == 0) {
      result = bit;
    } else {
      auto shift = arith::ConstantIntOp::create(builder, loc, type, index);
      bit = arith::ShLIOp::create(builder, loc, bit, shift);
      result = arith::OrIOp::create(builder, loc, result, bit);
    }
  }
  return result;
}

static void writeBits(OpBuilder& builder, WriteOp write,
                      function_ref<void(int64_t, Value)> store) {
  auto loc = write.getLoc();
  const auto type = write.getValue().getType();
  const auto width = type.getWidth();
  for (unsigned index = 0; index < width; ++index) {
    Value bit = write.getValue();
    if (index != 0) {
      auto shift = arith::ConstantIntOp::create(builder, loc, type, index);
      bit = arith::ShRUIOp::create(builder, loc, bit, shift);
    }
    if (width != 1) {
      bit = arith::TruncIOp::create(builder, loc, builder.getI1Type(), bit);
    }
    store(index, bit);
  }
}

namespace {
struct FoldInitialReads final : OpRewritePattern<AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocOp alloc,
                                PatternRewriter& rewriter) const override {
    if (alloc.getInitialization() != Initialization::Zero) {
      return failure();
    }
    auto* block = alloc->getBlock();
    Operation* firstMutation = nullptr;
    for (Operation* user : alloc.getResult().getUsers()) {
      if (isa<LoadOp, ReadOp>(user)) {
        continue;
      }
      auto* ancestor = block->findAncestorOpInBlock(*user);
      if (ancestor == nullptr) {
        return failure();
      }
      if (firstMutation == nullptr ||
          ancestor->isBeforeInBlock(firstMutation)) {
        firstMutation = ancestor;
      }
    }
    bool changed = false;
    for (Operation* user :
         llvm::make_early_inc_range(alloc.getResult().getUsers())) {
      if (isa<LoadOp, ReadOp>(user) && user->getBlock() == block &&
          (firstMutation == nullptr || user->isBeforeInBlock(firstMutation))) {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            user, rewriter.getIntegerAttr(user->getResult(0).getType(), 0));
        changed = true;
      }
    }
    return success(changed);
  }
};

struct EraseWriteOnlyRegister final : OpRewritePattern<AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocOp alloc,
                                PatternRewriter& rewriter) const override {
    if (!llvm::all_of(alloc.getResult().getUsers(), [](Operation* user) {
          return isa<StoreOp, WriteOp>(user);
        })) {
      return failure();
    }
    for (auto* user :
         llvm::make_early_inc_range(alloc.getResult().getUsers())) {
      rewriter.eraseOp(user);
    }
    rewriter.eraseOp(alloc);
    return success();
  }
};

struct DecomposeRead final : OpRewritePattern<ReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReadOp read,
                                PatternRewriter& rewriter) const override {
    auto result = readBits(rewriter, read, [&](int64_t index) -> Value {
      auto indexValue =
          arith::ConstantIndexOp::create(rewriter, read.getLoc(), index);
      return LoadOp::create(rewriter, read.getLoc(), rewriter.getI1Type(),
                            read.getReg(), indexValue)
          .getResult();
    });
    rewriter.replaceOp(read, result);
    return success();
  }
};

struct DecomposeWrite final : OpRewritePattern<WriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WriteOp write,
                                PatternRewriter& rewriter) const override {
    writeBits(rewriter, write, [&](int64_t index, Value bit) {
      auto indexValue =
          arith::ConstantIndexOp::create(rewriter, write.getLoc(), index);
      StoreOp::create(rewriter, write.getLoc(), bit, write.getReg(),
                      indexValue);
    });
    rewriter.eraseOp(write);
    return success();
  }
};

struct RegisterSlots final
    : DestructurableTypeInterface::ExternalModel<RegisterSlots, RegisterType> {
  [[nodiscard]] std::optional<DenseMap<Attribute, Type>>
  getSubelementIndexMap(Type type) const {
    const auto width = cast<RegisterType>(type).getWidth();
    if (width <= 1 || width > MAX_PROMOTED_REGISTER_WIDTH) {
      return std::nullopt;
    }
    DenseMap<Attribute, Type> elements;
    elements.reserve(static_cast<unsigned>(width));
    Builder builder(type.getContext());
    for (int64_t index = 0; index < width; ++index) {
      elements[builder.getIndexAttr(index)] = builder.getI1Type();
    }
    return elements;
  }

  [[nodiscard]] Type getTypeAtIndex(Type type, Attribute index) const {
    auto integer = dyn_cast<IntegerAttr>(index);
    if (!integer || integer.getInt() < 0 ||
        integer.getInt() >= cast<RegisterType>(type).getWidth()) {
      return {};
    }
    return IntegerType::get(type.getContext(), 1);
  }
};

struct DestructureRegister final
    : DestructurableAllocationOpInterface::ExternalModel<DestructureRegister,
                                                         AllocOp> {
  SmallVector<DestructurableMemorySlot>
  getDestructurableSlots(Operation* op) const {
    auto alloc = cast<AllocOp>(op);
    const auto type = alloc.getType();
    auto elements =
        cast<DestructurableTypeInterface>(type).getSubelementIndexMap();
    if (!elements) {
      return {};
    }
    return {
        {{.ptr = alloc.getResult(), .elemType = type}, std::move(*elements)},
    };
  }

  DenseMap<Attribute, MemorySlot> destructure(
      Operation* op, const DestructurableMemorySlot& /*slot*/,
      const SmallPtrSetImpl<Attribute>& usedIndices, OpBuilder& builder,
      SmallVectorImpl<DestructurableAllocationOpInterface>& /*newAllocators*/)
      const {
    DenseMap<Attribute, MemorySlot> slots;
    slots.reserve(usedIndices.size());
    SmallVector<Attribute> indices(usedIndices.begin(), usedIndices.end());
    llvm::sort(indices, [](Attribute left, Attribute right) {
      return cast<IntegerAttr>(left).getInt() <
             cast<IntegerAttr>(right).getInt();
    });
    for (Attribute index : indices) {
      auto alloc = AllocOp::create(builder, op->getLoc(),
                                   RegisterType::get(op->getContext(), 1),
                                   cast<AllocOp>(op).getInitialization());
      slots[index] = {
          .ptr = alloc.getResult(),
          .elemType = builder.getI1Type(),
      };
    }
    return slots;
  }

  std::optional<DestructurableAllocationOpInterface>
  handleDestructuringComplete(Operation* op,
                              const DestructurableMemorySlot& /*slot*/,
                              OpBuilder& /*builder*/) const {
    op->erase();
    return std::nullopt;
  }
};

struct PromoteRegister final
    : PromotableAllocationOpInterface::ExternalModel<PromoteRegister, AllocOp> {
  SmallVector<MemorySlot> getPromotableSlots(Operation* op) const {
    auto alloc = cast<AllocOp>(op);
    if (alloc.getType().getWidth() != 1) {
      return {};
    }
    return {
        {
            .ptr = alloc.getResult(),
            .elemType = IntegerType::get(op->getContext(), 1),
        },
    };
  }

  Value getDefaultValue(Operation* op, const MemorySlot& /*slot*/,
                        OpBuilder& builder) const {
    // Reading an undefined bit before a store is UB; zero is a valid
    // refinement.
    return arith::ConstantIntOp::create(builder, op->getLoc(), 0, 1);
  }

  void handleBlockArgument(Operation* /*op*/, const MemorySlot& /*slot*/,
                           BlockArgument /*argument*/,
                           OpBuilder& /*builder*/) const {}

  std::optional<PromotableAllocationOpInterface>
  handlePromotionComplete(Operation* op, const MemorySlot& /*slot*/,
                          Value defaultValue, OpBuilder& /*builder*/) const {
    if (defaultValue && defaultValue.use_empty()) {
      defaultValue.getDefiningOp()->erase();
    }
    op->erase();
    return std::nullopt;
  }
};

template <typename Op>
struct PromoteAccess final
    : PromotableMemOpInterface::ExternalModel<PromoteAccess<Op>, Op> {
  static constexpr bool READS =
      std::is_same_v<Op, LoadOp> || std::is_same_v<Op, ReadOp>;

  bool loadsFrom(Operation* op, const MemorySlot& slot) const {
    return READS && cast<Op>(op).getReg() == slot.ptr;
  }
  bool storesTo(Operation* op, const MemorySlot& slot) const {
    return !READS && cast<Op>(op).getReg() == slot.ptr;
  }
  Value getStored(Operation* op, const MemorySlot& /*slot*/,
                  OpBuilder& /*builder*/, Value /*reachingDef*/,
                  const DataLayout& /*layout*/) const {
    if constexpr (!READS) {
      return cast<Op>(op).getValue();
    }
    return {};
  }
  bool canUsesBeRemoved(Operation* op, const MemorySlot& slot,
                        const SmallPtrSetImpl<OpOperand*>& blockingUses,
                        SmallVectorImpl<OpOperand*>& /*newBlockingUses*/,
                        const DataLayout& /*layout*/) const {
    auto access = cast<Op>(op);
    if constexpr (std::is_same_v<Op, LoadOp> || std::is_same_v<Op, StoreOp>) {
      if (getConstantIntValue(access.getIndex()) != 0) {
        return false;
      }
    }
    return blockingUses.size() == 1 &&
           (*blockingUses.begin())->get() == slot.ptr &&
           access.getReg() == slot.ptr &&
           access.getReg().getType().getWidth() == 1;
  }
  DeletionKind
  removeBlockingUses(Operation* op, const MemorySlot& /*slot*/,
                     const SmallPtrSetImpl<OpOperand*>& /*blockingUses*/,
                     OpBuilder& /*builder*/, Value reachingDef,
                     const DataLayout& /*layout*/) const {
    if constexpr (READS) {
      cast<Op>(op).getResult().replaceAllUsesWith(reachingDef);
    }
    return DeletionKind::Delete;
  }
};

template <typename Op>
struct DestructureAccess final
    : DestructurableAccessorOpInterface::ExternalModel<DestructureAccess<Op>,
                                                       Op> {
  bool canRewire(Operation* op, const DestructurableMemorySlot& slot,
                 SmallPtrSetImpl<Attribute>& usedIndices,
                 SmallVectorImpl<MemorySlot>& /*mustBeSafelyUsed*/,
                 const DataLayout& /*layout*/) const {
    auto access = cast<Op>(op);
    if (access.getReg() != slot.ptr) {
      return false;
    }
    if constexpr (std::is_same_v<Op, ReadOp>) {
      // Consumer-specific lowering handles wide comparisons before SROA.
      // General wide packing is not supported by every exporter.
      if (access.getType().getWidth() > 64) {
        return false;
      }
    }
    Builder builder(op->getContext());
    if constexpr (std::is_same_v<Op, LoadOp> || std::is_same_v<Op, StoreOp>) {
      auto index = getConstantIntValue(access.getIndex());
      if (!index) {
        return false;
      }
      usedIndices.insert(builder.getIndexAttr(*index));
    } else {
      for (auto [index, type] : slot.subelementTypes) {
        usedIndices.insert(index);
      }
    }
    return true;
  }

  DeletionKind rewire(Operation* op, const DestructurableMemorySlot& /*slot*/,
                      DenseMap<Attribute, MemorySlot>& subslots,
                      OpBuilder& builder, const DataLayout& /*layout*/) const {
    const OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op);
    auto access = cast<Op>(op);
    auto zero = arith::ConstantIndexOp::create(builder, op->getLoc(), 0);
    const auto reg = [&](int64_t index) {
      return subslots.at(builder.getIndexAttr(index)).ptr;
    };
    if constexpr (std::is_same_v<Op, LoadOp> || std::is_same_v<Op, StoreOp>) {
      access.getRegMutable().assign(
          reg(*getConstantIntValue(access.getIndex())));
      access.getIndexMutable().assign(zero);
      return DeletionKind::Keep;
    } else if constexpr (std::is_same_v<Op, ReadOp>) {
      auto value = readBits(builder, access, [&](int64_t index) -> Value {
        return LoadOp::create(builder, op->getLoc(), builder.getI1Type(),
                              reg(index), zero);
      });
      access.getResult().replaceAllUsesWith(value);
    } else {
      writeBits(builder, access, [&](int64_t index, Value bit) {
        StoreOp::create(builder, op->getLoc(), bit, reg(index), zero);
      });
    }
    return DeletionKind::Delete;
  }
};

} // namespace

static void attachMemorySlotInterfaces(MLIRContext* context) {
  RegisterType::attachInterface<RegisterSlots>(*context);
  AllocOp::attachInterface<PromoteRegister, DestructureRegister>(*context);
  LoadOp::attachInterface<PromoteAccess<LoadOp>, DestructureAccess<LoadOp>>(
      *context);
  StoreOp::attachInterface<PromoteAccess<StoreOp>, DestructureAccess<StoreOp>>(
      *context);
  ReadOp::attachInterface<PromoteAccess<ReadOp>, DestructureAccess<ReadOp>>(
      *context);
  WriteOp::attachInterface<PromoteAccess<WriteOp>, DestructureAccess<WriteOp>>(
      *context);
}

LogicalResult LoadOp::verify() {
  return verifyIndex(getOperation(), getReg(), getIndex());
}

LogicalResult ReadOp::verify() {
  if (std::cmp_not_equal(getResult().getType().getWidth(),
                         getReg().getType().getWidth())) {
    return emitOpError("result width must match register width");
  }
  return success();
}

LogicalResult WriteOp::verify() {
  if (std::cmp_not_equal(getValue().getType().getWidth(),
                         getReg().getType().getWidth())) {
    return emitOpError("value width must match register width");
  }
  return success();
}

static Value buildBitComparison(OpBuilder& builder, Location location,
                                const arith::CmpIPredicate predicate,
                                const llvm::APInt& rhs, ArrayRef<Value> bits,
                                bool reversed) {
  const auto unsignedPredicate = mqt::unsignedPredicate(predicate);
  const bool equality = unsignedPredicate == arith::CmpIPredicate::eq ||
                        unsignedPredicate == arith::CmpIPredicate::ne;
  const bool negate =
      equality ? unsignedPredicate == arith::CmpIPredicate::ne
               : (unsignedPredicate == arith::CmpIPredicate::ult ||
                  unsignedPredicate == arith::CmpIPredicate::ule) != reversed;
  const bool acceptEqual = arith::applyCmpPredicate(predicate, rhs, rhs);
  auto one = arith::ConstantIntOp::create(builder, location, 1, 1);
  Value result = arith::ConstantIntOp::create(builder, location,
                                              acceptEqual != negate ? 1 : 0, 1);
  // Each higher bit overrides the comparison of the lower bits. Build >= or
  // > with one AND/OR per bit, then complement for < or <=.
  for (auto [index, input] : llvm::enumerate(bits)) {
    Value bit = input;
    bool constantBit = rhs[static_cast<unsigned>(index)];
    if (unsignedPredicate != predicate && index + 1 == bits.size()) {
      bit = builder.createOrFold<arith::XOrIOp>(location, bit, one);
      constantBit = !constantBit;
    }
    if (equality && !constantBit) {
      bit = builder.createOrFold<arith::XOrIOp>(location, bit, one);
    }
    result = equality || constantBit
                 ? builder.createOrFold<arith::AndIOp>(location, bit, result)
                 : builder.createOrFold<arith::OrIOp>(location, bit, result);
  }
  return negate ? builder.createOrFold<arith::XOrIOp>(location, result, one)
                : result;
}

namespace {

/// Lower explicit register snapshots compared with constants at the read point.
struct LowerRegisterComparison final : OpRewritePattern<arith::CmpIOp> {
  explicit LowerRegisterComparison(MLIRContext* context, bool forPromotion)
      : OpRewritePattern(context), forPromotion_(forPromotion) {}

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter& rewriter) const override {
    const auto constantComparison = [](Operation* consumer) {
      auto cmp = dyn_cast<arith::CmpIOp>(consumer);
      APInt value;
      return cmp && (matchPattern(cmp.getLhs(), m_ConstantInt(&value)) ||
                     matchPattern(cmp.getRhs(), m_ConstantInt(&value)));
    };
    auto read = op.getLhs().getDefiningOp<cbit::ReadOp>();
    if (!read) {
      read = op.getRhs().getDefiningOp<cbit::ReadOp>();
    }
    if (!read || read.getType().getWidth() <= 64 || !constantComparison(op)) {
      return failure();
    }
    if (forPromotion_) {
      auto alloc = read.getReg().getDefiningOp<cbit::AllocOp>();
      if (!alloc || alloc.getType().getWidth() > MAX_PROMOTED_REGISTER_WIDTH ||
          !llvm::all_of(alloc.getResult().getUsers(), [&](Operation* user) {
            if (auto otherRead = dyn_cast<cbit::ReadOp>(user)) {
              return llvm::all_of(otherRead.getResult().getUsers(),
                                  constantComparison);
            }
            if (auto load = dyn_cast<cbit::LoadOp>(user)) {
              return getConstantIntValue(load.getIndex()).has_value();
            }
            if (auto store = dyn_cast<cbit::StoreOp>(user)) {
              return getConstantIntValue(store.getIndex()).has_value();
            }
            return isa<cbit::WriteOp>(user);
          })) {
        return failure();
      }
    }
    SmallVector<arith::CmpIOp> comparisons;
    for (auto* user : read.getResult().getUsers()) {
      if (constantComparison(user)) {
        comparisons.push_back(cast<arith::CmpIOp>(user));
      }
    }
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(read);
    SmallVector<Value> bits;
    bits.reserve(read.getType().getWidth());
    for (unsigned index = 0; index < read.getType().getWidth(); ++index) {
      auto position =
          arith::ConstantIndexOp::create(rewriter, read.getLoc(), index);
      bits.push_back(cbit::LoadOp::create(rewriter, read.getLoc(),
                                          rewriter.getI1Type(), read.getReg(),
                                          position));
    }
    for (auto cmp : comparisons) {
      const bool reversed = cmp.getRhs() == read.getResult();
      APInt constant;
      const bool matched = matchPattern(cmp.getOperand(reversed ? 0 : 1),
                                        m_ConstantInt(&constant));
      assert(matched && "expected a constant comparison operand");
      auto result = buildBitComparison(
          rewriter, cmp.getLoc(), cmp.getPredicate(), constant, bits, reversed);
      rewriter.replaceOp(cmp, result);
    }
    if (read->use_empty()) {
      rewriter.eraseOp(read);
    }
    return success();
  }

private:
  bool forPromotion_;
};

} // namespace

void mlir::cbit::populateRegisterComparisonPatterns(RewritePatternSet& patterns,
                                                    bool forPromotion) {
  patterns.add<LowerRegisterComparison>(patterns.getContext(), forPromotion);
  arith::CmpIOp::getCanonicalizationPatterns(patterns, patterns.getContext());
}

void mlir::cbit::populateCBitDecompositionPatterns(
    RewritePatternSet& patterns) {
  patterns.add<DecomposeRead, DecomposeWrite>(patterns.getContext());
}

void AllocOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<EraseWriteOnlyRegister, FoldInitialReads>(context);
}

LogicalResult LoadOp::canonicalize(LoadOp op, PatternRewriter& rewriter) {
  return forwardKnownValue(op, op.getReg(), op.getIndex(), rewriter);
}

LogicalResult ReadOp::canonicalize(ReadOp op, PatternRewriter& rewriter) {
  return forwardKnownValue(op, op.getReg(), {}, rewriter);
}

LogicalResult StoreOp::verify() {
  return verifyIndex(getOperation(), getReg(), getIndex());
}

#define GET_OP_CLASSES
#include "mqt/Dialect/CBit/IR/CBitOps.cpp.inc"

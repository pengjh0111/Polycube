#include "Transform/Passes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TypeName.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <utility>

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Transform/Passes.h.inc"

namespace {

using namespace mlir::cuda_tile;

static bool parseFlatPoints(llvm::StringRef encoded,
                            llvm::SmallVectorImpl<int64_t> &values) {
  while (!encoded.empty()) {
    auto [head, tail] = encoded.split(',');
    head = head.trim();
    if (!head.empty()) {
      int64_t parsedValue = 0;
      if (head.getAsInteger(10, parsedValue))
        return false;
      values.push_back(parsedValue);
    }
    encoded = tail;
  }
  return true;
}

static Value makeScalarI32Constant(OpBuilder &builder, Location loc,
                                   int64_t value) {
  auto scalarTy = TileType::get({}, builder.getI32Type());
  auto shapedTy = cast<ShapedType>(scalarTy);
  auto attr = DenseElementsAttr::get(
      shapedTy, builder.getIntegerAttr(builder.getI32Type(), value));
  auto constant = ConstantOp::create(
      builder, loc, scalarTy, cast<DenseIntOrFPElementsAttr>(attr));
  return constant.getResult();
}

static bool isDecodeChainOp(Operation *op) {
  return isa<AddIOp, SubIOp, MulIOp, DivIOp, RemIOp, AssumeOp>(op);
}

static void collectDecodeChain(Value root,
                               SmallPtrSetImpl<Operation *> &chain) {
  SmallVector<Value, 16> worklist{root};
  DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    for (Operation *user : current.getUsers()) {
      if (!isDecodeChainOp(user))
        continue;
      if (!chain.insert(user).second)
        continue;
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }
}

static SmallVector<Value, 4>
findTerminals(Block *block, const SmallPtrSetImpl<Operation *> &chain) {
  SmallVector<Value, 4> terminals;

  for (Operation &op : *block) {
    if (!chain.contains(&op))
      continue;

    for (Value result : op.getResults()) {
      auto tileTy = dyn_cast<TileType>(result.getType());
      if (!tileTy || !tileTy.getShape().empty())
        continue;
      auto elemTy = dyn_cast<IntegerType>(tileTy.getElementType());
      if (!elemTy || elemTy.getWidth() != 32)
        continue;

      // Terminal: this value is consumed by a MulIOp that is IN the chain,
      // AND that MulIOp's result has at least one user OUTSIDE the chain.
      // This matches pid_m/pid_n which feed into BLOCK_M/BLOCK_N multiply
      // ops (in chain), which in turn feed into reshape (outside chain).
      bool isTerminal = false;
      for (Operation *user : result.getUsers()) {
        if (!chain.contains(user))
          continue;
        if (!isa<MulIOp>(user))
          continue;
        // user is MulIOp in chain — check if its results feed outside
        for (Value mulResult : user->getResults()) {
          for (Operation *mulUser : mulResult.getUsers()) {
            if (!chain.contains(mulUser)) {
              isTerminal = true;
              break;
            }
          }
          if (isTerminal)
            break;
        }
        if (isTerminal)
          break;
      }
      if (isTerminal)
        terminals.push_back(result);
    }
  }

  if (!terminals.empty())
    return terminals;

  for (Operation &op : *block) {
    if (!chain.contains(&op))
      continue;

    for (Value result : op.getResults()) {
      auto tileTy = dyn_cast<TileType>(result.getType());
      if (!tileTy || !tileTy.getShape().empty())
        continue;
      auto elemTy = dyn_cast<IntegerType>(tileTy.getElementType());
      if (!elemTy || elemTy.getWidth() != 32)
        continue;

      bool hasExternalUser = false;
      bool allExternalAreMul = true;
      for (Operation *user : result.getUsers()) {
        if (!chain.contains(user)) {
          hasExternalUser = true;
          if (!isa<MulIOp>(user))
            allExternalAreMul = false;
        }
      }
      if (hasExternalUser && allExternalAreMul)
        terminals.push_back(result);
    }
  }

  return terminals;
}

class PolycubeBlockOrderPass final
    : public PassWrapper<PolycubeBlockOrderPass,
                         OperationPass<mlir::ModuleOp>> {
public:
  PolycubeBlockOrderPass() = default;

  PolycubeBlockOrderPass(std::string unitPointsFlat, int64_t fVal,
                         int64_t tmDim, int64_t tnDim, int64_t skDim)
      : unitPointsFlat(std::move(unitPointsFlat)), fVal(fVal), tmDim(tmDim),
        tnDim(tnDim), skDim(skDim) {}

  StringRef getArgument() const override { return "polycube-block-order"; }
  StringRef getDescription() const override {
    return "Rewrite get_tile_block_id decode chains into polycube scan-order arithmetic";
  }

  void runOnOperation() override {
    if (unitPointsFlat.empty() || fVal <= 0)
      return;

    SmallVector<int64_t, 24> flatValues;
    if (!parseFlatPoints(unitPointsFlat, flatValues))
      return;
    if (flatValues.empty() || flatValues.size() % 3 != 0)
      return;

    const unsigned totalBlocks = static_cast<unsigned>(fVal);
    if (flatValues.size() != totalBlocks * 3)
      return;

    getOperation().walk([&](GetTileBlockIdOp op) {
      rewriteBlockId(op, flatValues, totalBlocks);
    });
  }

private:
  void rewriteBlockId(GetTileBlockIdOp op, ArrayRef<int64_t> flatValues,
                      unsigned totalBlocks) {
    Value blockIdX = op.getBlockIdX();
    Value blockIdY = op.getBlockIdY();

    auto tileTy = dyn_cast<TileType>(blockIdX.getType());
    if (!tileTy)
      return;
    auto elemTy = dyn_cast<IntegerType>(tileTy.getElementType());
    if (!elemTy || elemTy.getWidth() != 32)
      return;

    Block *block = op->getBlock();
    if (!block)
      return;

    SmallPtrSet<Operation *, 32> chain;
    collectDecodeChain(blockIdX, chain);

    SmallVector<Value, 4> terminals = findTerminals(block, chain);
    if (terminals.size() < 2)
      return;

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    auto i32TileTy = TileType::get(tileTy.getShape(), builder.getI32Type());
    auto i1TileTy = TileType::get(tileTy.getShape(), builder.getI1Type());

    Value pidM = makeScalarI32Constant(builder, op.getLoc(), 0);
    Value pidN = makeScalarI32Constant(builder, op.getLoc(), 0);
    Value pidSK = makeScalarI32Constant(builder, op.getLoc(), 0);

    for (unsigned i = 0; i < totalBlocks; ++i) {
      const int64_t ptM = flatValues[3 * i + 0];
      const int64_t ptN = flatValues[3 * i + 1];
      const int64_t ptSK = flatValues[3 * i + 2];

      auto index = makeScalarI32Constant(builder, op.getLoc(), i);
      auto eq = CmpIOp::create(builder, op.getLoc(), i1TileTy,
                               ComparisonPredicate::EQUAL, blockIdX, index,
                               Signedness::Unsigned);

      auto cstM = makeScalarI32Constant(builder, op.getLoc(), ptM);
      auto sumM = AddIOp::create(builder, op.getLoc(), i32TileTy, pidM, cstM,
                                 IntegerOverflow::NONE);
      Value candM = sumM.getResult();

      auto cstN = makeScalarI32Constant(builder, op.getLoc(), ptN);
      auto sumN = AddIOp::create(builder, op.getLoc(), i32TileTy, pidN, cstN,
                                 IntegerOverflow::NONE);
      Value candN = sumN.getResult();

      auto cstSK = makeScalarI32Constant(builder, op.getLoc(), ptSK);
      auto sumSK = AddIOp::create(builder, op.getLoc(), i32TileTy, pidSK,
                                  cstSK, IntegerOverflow::NONE);
      Value candSK = sumSK.getResult();

      pidM = SelectOp::create(builder, op.getLoc(), eq.getResult(),
                              candM, pidM)
                 .getResult();
      pidN = SelectOp::create(builder, op.getLoc(), eq.getResult(),
                              candN, pidN)
                 .getResult();
      pidSK = SelectOp::create(builder, op.getLoc(), eq.getResult(),
                               candSK, pidSK)
                  .getResult();
    }

    terminals[0].replaceAllUsesWith(pidM);
    terminals[1].replaceAllUsesWith(pidN);
    if (!blockIdY.use_empty())
      blockIdY.replaceAllUsesWith(pidSK);
  }

  std::string unitPointsFlat;
  int64_t fVal = 0;
  int64_t tmDim = 1;
  int64_t tnDim = 1;
  int64_t skDim = 1;
};

} // namespace

std::unique_ptr<Pass> mlir::triton::createPolycubeBlockOrderPass() {
  return std::make_unique<PolycubeBlockOrderPass>();
}

std::unique_ptr<Pass> mlir::triton::createPolycubeBlockOrderPass(
    std::string unitPointsFlat, int64_t fVal, int64_t tmDim, int64_t tnDim,
    int64_t skDim) {
  return std::make_unique<PolycubeBlockOrderPass>(std::move(unitPointsFlat),
                                                  fVal, tmDim, tnDim, skDim);
}

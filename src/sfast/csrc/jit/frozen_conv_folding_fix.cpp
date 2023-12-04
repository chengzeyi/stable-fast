#include <torch/extension.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>

#include "frozen_conv_folding_fix.h"

namespace sfast {
namespace jit {

using namespace torch::jit;

/*
 *     torch._C._jit_pass_optimize_frozen_graph(mod.graph, optimize_numerics)
 * RuntimeError: stack_out && stack_out->size() == 1 INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/frozen_conv_folding.cpp":349, please report a bug to PyTorch.
*/
void FixFrozenConvFolding(std::shared_ptr<Graph> &graph) {
  FixFrozenConvFoldingOnBlock(graph->block());
  GRAPH_DUMP("After FixFrozenConvFolding: ", graph);
}

void FixFrozenConvFoldingOnBlock(Block *block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks()) {
      FixFrozenConvFoldingOnBlock(sub);
    }

    switch (it->kind()) {
    case prim::Constant: {
      // remove zero dim tensor constants, replacing with number equivalent
      if (it->output()->type()->isSubtypeOf(TensorType::get())) {
        auto t = toIValue(it->output())->toTensor();
        if (t.dim() == 0 && t.device().is_cpu()) {
          bool only_used_by_simple_arith = true;
          auto &uses = it->output()->uses();
          for (auto &u : uses) {
            if (u.user->kind() != aten::add && u.user->kind() != aten::sub &&
                u.user->kind() != aten::mul && u.user->kind() != aten::div) {
              only_used_by_simple_arith = false;
              break;
            }
          }
          if (only_used_by_simple_arith) {
            // c10::ScalarType dtype = c10::typeMetaToScalarType(t.dtype());
            at::Scalar s = t.item();
            WithInsertPoint guard(*it);
            Value *r = block->owningGraph()->insertConstant(s, c10::nullopt,
                                                            it->scope());
            // r->copyMetadata(it->output());
            it->output()->replaceAllUsesWith(r);
            it.destroyCurrent();
          }
        }
      }
    } break;
    default: {
      // do nothing
    } break;
    }
  }
}

} // namespace jit
} // namespace sfast

#include <torch/extension.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>

#include "scalar_tensor_erase.h"

namespace sfast {
namespace jit {

using namespace torch::jit;

void EraseScalarTensors(std::shared_ptr<Graph> &graph) {
  EraseScalarTensorsOnBlock(graph->block());
  GRAPH_DUMP("After EraseScalarTensors: ", graph);
}

void EraseScalarTensorsOnBlock(Block *block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks()) {
      EraseScalarTensorsOnBlock(sub);
    }

    switch (it->kind()) {
    case prim::Constant: {
      // remove zero dim tensor constants, replacing with number equivalent
      if (it->output()->type()->isSubtypeOf(TensorType::get())) {
        auto t = toIValue(it->output())->toTensor();
        if (t.dim() == 0) {
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
    } break;
    // case aten::Bool:
    // case aten::Float:
    // case aten::Int:
    // case aten::FloatImplicit:
    // case aten::IntImplicit:
    // case aten::ScalarImplicit:
    // case prim::NumToTensor: {
    //   it->output()->replaceAllUsesWith(it->inputs()[0]);
    //   it.destroyCurrent();
    // } break;
    default: {
      // do nothing
    } break;
    }
  }
}

} // namespace jit
} // namespace sfast

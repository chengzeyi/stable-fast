#include <torch/extension.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>

#include "device_constant_override.h"

namespace sfast {
namespace jit {

using namespace torch::jit;

void OverrideDeviceConstants(std::shared_ptr<Graph> &graph,
                             const c10::optional<at::Device> &device) {
  OverrideDeviceConstantsOnBlock(graph->block(), device);
  GRAPH_DUMP("After OverrideDeviceConstants: ", graph);
}

void OverrideDeviceConstantsOnBlock(Block *block,
                                    const c10::optional<at::Device> &device) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks()) {
      OverrideDeviceConstantsOnBlock(sub, device);
    }

    switch (it->kind()) {
    case prim::Constant: {
      if (it->output()->type()->isSubtypeOf(DeviceObjType::get())) {
        WithInsertPoint guard(*it);
        Value *r = block->owningGraph()->insertConstant(device, c10::nullopt,
                                                        it->scope());
        it->output()->replaceAllUsesWith(r);
        it.destroyCurrent();
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

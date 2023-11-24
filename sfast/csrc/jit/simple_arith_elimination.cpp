#include <torch/extension.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>

#include "simple_arith_elimination.h"

namespace sfast {
namespace jit {

using namespace torch::jit;

namespace {
bool is_constant_value_of(Value *value, double target) {
  if (value->node()->kind() != prim::Constant) {
    return false;
  }
  auto type = value->type();
  if (type == FloatType::get()) {
    return constant_as<double>(value) == target;
  } else if (type == IntType::get()) {
    return constant_as<int64_t>(value) == target;
  } else if (type == TensorType::get()) {
    auto tensor = toIValue(value)->toTensor();
    if (tensor.dim() == 0 && tensor.numel() == 1) {
      auto v = tensor.item();
      return (v.isFloatingPoint() && v.toDouble() == target) ||
             (v.isIntegral(false) && v.toInt() == target);
    }
    return false;
  } else {
    return false;
  }
}
} // namespace

void EliminateSimpleArith(std::shared_ptr<Graph> &graph) {
  EliminateSimpleArithOnBlock(graph->block());
  GRAPH_DUMP("After EliminateSimpleArith: ", graph);
}

void EliminateSimpleArithOnBlock(Block *block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks()) {
      EliminateSimpleArithOnBlock(sub);
    }
    int input_idx = -1;
    switch (it->kind()) {
    case aten::add: {
      if (it->inputs().size() == 2) {
        auto input0 = it->inputs()[0];
        auto input1 = it->inputs()[1];
        if (is_constant_value_of(input0, 0.0)) {
          input_idx = 1;
        } else if (is_constant_value_of(input1, 0.0)) {
          input_idx = 0;
        }
      }
      break;
    }
    case aten::sub: {
      if (it->inputs().size() == 2) {
        if (is_constant_value_of(it->inputs()[1], 0.0)) {
          input_idx = 0;
        }
      }
      break;
    }
    case aten::mul: {
      if (it->inputs().size() == 2) {
        auto input0 = it->inputs()[0];
        auto input1 = it->inputs()[1];
        if (is_constant_value_of(input0, 1.0)) {
          input_idx = 1;
        } else if (is_constant_value_of(input1, 1.0)) {
          input_idx = 0;
        }
      }
      break;
    }
    case aten::div: {
      if (it->inputs().size() == 2) {
        if (is_constant_value_of(it->inputs()[1], 1.0)) {
          input_idx = 0;
        }
      }
      break;
    }
    }
    if (input_idx != -1) {
      WithInsertPoint guard(*it);
      auto input = it->inputs()[input_idx];
      it->output()->replaceAllUsesWith(input);
      it.destroyCurrent();
    }
  }
}

} // namespace jit
} // namespace sfast

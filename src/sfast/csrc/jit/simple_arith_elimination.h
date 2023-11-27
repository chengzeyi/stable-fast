#pragma once
#include <torch/extension.h>

#include <torch/csrc/jit/ir/ir.h>

namespace sfast {
namespace jit {

using namespace torch::jit;

void EliminateSimpleArith(std::shared_ptr<Graph> &graph);
void EliminateSimpleArithOnBlock(Block *block);

} // namespace jit
} // namespace sfast

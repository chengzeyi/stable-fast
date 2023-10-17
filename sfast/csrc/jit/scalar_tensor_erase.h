#pragma once
#include <torch/extension.h>

#include <torch/csrc/jit/ir/ir.h>

namespace sfast {
namespace jit {

using namespace torch::jit;

void EraseScalarTensors(std::shared_ptr<Graph> &graph);
void EraseScalarTensorsOnBlock(Block *block);

} // namespace jit
} // namespace sfast

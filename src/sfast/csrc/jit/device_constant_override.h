#pragma once
#include <torch/extension.h>

#include <torch/csrc/jit/ir/ir.h>

namespace sfast {
namespace jit {

using namespace torch::jit;

void OverrideDeviceConstants(std::shared_ptr<Graph> &graph,
                             const c10::optional<at::Device> &device);
void OverrideDeviceConstantsOnBlock(Block *block,
                                    const c10::optional<at::Device> &device);

} // namespace jit
} // namespace sfast

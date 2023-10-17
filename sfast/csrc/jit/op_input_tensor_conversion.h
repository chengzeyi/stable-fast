#pragma once
#include <torch/extension.h>

#include <torch/csrc/jit/ir/ir.h>

namespace sfast {
namespace jit {

using namespace torch::jit;

void
ConvertOpInputTensors(std::shared_ptr<Graph> &graph, const std::string &op_name,
                      const c10::optional<at::Device> &device,
                      const c10::optional<at::ScalarType> &dtype,
                      const c10::optional<at::MemoryFormat> &memory_format,
                      const c10::optional<std::vector<int>> &indices);

void ConvertOpInputTensorsOnBlock(
    Block *block, const std::string &op_name,
    const c10::optional<at::Device> &device,
    const c10::optional<at::ScalarType> &dtype,
    const c10::optional<at::MemoryFormat> &memory_format,
    const c10::optional<std::vector<int>> &indices);

} // namespace jit
} // namespace sfast
